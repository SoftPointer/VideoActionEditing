#!/usr/bin/env python3
"""Externally admit a terminal formal32 job and conditionally seal Stage-A."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ELAPSED = re.compile(r"(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}\Z")
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
STEM = "saic-source-anchor-formal32-retfd-20260813-r1"
RELEASE = ROOT / "releases" / STEM
INPUTS = RELEASE / "inputs"
POSTFLIGHT_ROOT = RELEASE / "postflight"
OUTPUT_PARENT = ROOT / "runs" / STEM
LOG_DIR = ROOT / "slurm" / STEM
CHECKPOINT_PARENT = ROOT / "checkpoints"
WRAPPER = INPUTS / "auh_train_saic_source_anchor_v1.sbatch"
TRAINER = INPUTS / "train_saic_source_anchor_v1.py"
SUBMITTER = INPUTS / "submit_saic_source_anchor_formal32_v1.py"
POSTFLIGHT = POSTFLIGHT_ROOT / "postflight_saic_source_anchor_formal32_v1.py"
SOURCE_ARCHIVE = INPUTS / "videoedit-saic-20c2193-methods.tar"
SOURCE_MANIFEST = INPUTS / "source-anchor-v1-2a52753.json"
CHECKPOINT_MANIFEST = INPUTS / "bernini-r13-ff4c5d4-checkpoint.sha256"
FULL60_ADMISSION = INPUTS / "formal-full60-admission.json"
FULL60_ADMISSION_SOURCE = Path(
    "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SOURCE_PATH__"
)
RELEASE_MANIFEST = RELEASE / "release-manifest.json"
SUBMISSION_RECEIPT = OUTPUT_PARENT / "submission-receipt.json"
TRAINING_ADMISSION = OUTPUT_PARENT / "training-admission.json"
PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
PYTHON_SHA256 = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
SACCT = Path("/usr/bin/sacct")
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
)
VEOMNI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
)
CHECKPOINT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
WRAPPER_SHA256 = "__SOURCE_ANCHOR_WRAPPER_SHA256__"
TRAINER_SHA256 = "28059c7a1ea5d641b35a04e929fa454531ee6df8444a01cceefb3f3a63622a1c"
SUBMITTER_SHA256 = "__SOURCE_ANCHOR_SUBMITTER_SHA256__"
SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
SOURCE_MANIFEST_SHA256 = (
    "b56e9dca9085ba2f7b67acee04f29f7e45afd73a71b92172df5b19095fe33c4c"
)
SOURCE_MANIFEST_DIGEST = (
    "e3978e872c80c20b3bf8c974cd7324b62f54d58ca162e1760f548a02a2538d8d"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
FULL60_ADMISSION_SHA256 = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SHA256__"
FULL60_ADMISSION_DIGEST = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_DIGEST__"
FULL60_ADMISSION_SCHEMA = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SCHEMA__"
FULL60_ADMISSION_STATUS = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_STATUS__"
SACCT_FIELDS = [
    "JobIDRaw", "JobName", "Partition", "QOS", "State", "ExitCode",
    "ReqTRES%512", "AllocTRES%512", "NodeList", "Start", "End", "Elapsed",
    "SubmitLine%8192",
]
SACCT_KEYS = [
    "JobIDRaw", "JobName", "Partition", "QOS", "State", "ExitCode",
    "ReqTRES", "AllocTRES", "NodeList", "Start", "End", "Elapsed",
    "SubmitLine",
]
AUTHORITY_PRE = {
    "stage_a_training_authorized": True,
    "stage_a_checkpoint_release_authorized": False,
    "action_training_authorized": False,
    "semantic_action_authorized": False,
    "decoded_rgb_identity_authorized": False,
    "publication_beyond_stage_a_checkpoint": False,
}
FORMAL_POSTFLIGHT_AUTHORITY = {
    "formal_stage_a_training_completed": True,
    "stage_a_checkpoint_release": True,
    "stage_b": False,
    "semantic_action": False,
    "identity": False,
    "candidate_selection": False,
    "production": False,
}
FORMAL_NO_GO_AUTHORITY = {
    **FORMAL_POSTFLIGHT_AUTHORITY,
    "stage_a_checkpoint_release": False,
}
CHECKPOINT_RELEASE_AUTHORITY = {
    key: value
    for key, value in FORMAL_POSTFLIGHT_AUTHORITY.items()
    if key != "formal_stage_a_training_completed"
}
EXPORT_NAMES = [
    "SAIC_ANCHOR_ARCHIVE", "SAIC_ANCHOR_TRAINER",
    "SAIC_ANCHOR_SOURCE_MANIFEST", "SAIC_ANCHOR_CHECKPOINT_MANIFEST",
    "SAIC_ANCHOR_FULL60_ADMISSION", "SAIC_ANCHOR_RELEASE_MANIFEST",
    "SAIC_ANCHOR_SUBMISSION_RECEIPT", "SAIC_ANCHOR_OUTPUT_PARENT",
    "SAIC_ANCHOR_RELEASE_MANIFEST_SHA256",
    "SAIC_ANCHOR_RELEASE_MANIFEST_DIGEST", "SAIC_ANCHOR_WRAPPER_SHA256",
]


def die(message: str) -> None:
    raise SystemExit(f"postflight-saic-source-anchor-formal32-v1: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def validate_full60_deep_admission(value: Any) -> None:
    """Reject any terminal token that does not close the full60 evidence tree."""

    root_fields = {
        "schema_version", "status", "job_id", "job_success",
        "slurm_terminal_verified", "formal_admission",
        "formal_full60_result_claimed", "candidate_count", "attempt_count",
        "same_seed_proof_count", "master_receipt", "submission_receipt",
        "slurm_terminal_observation", "world8_closure",
        "failure_receipt_absent", "output_namespace_deep_closed", "authority",
        "receipt_digest",
    }
    master_fields = {
        "path", "sha256", "receipt_digest", "schema_version", "bank_id",
        "attempt_count", "row_count", "seed_cell_count", "branch_order",
        "master_deep_validated",
    }
    submission_fields = {
        "path", "sha256", "receipt_digest", "job_id",
        "request_candidate_count", "submission_deep_validated",
    }
    slurm_fields = {
        "executable_sha256", "stdout_sha256", "stderr_sha256",
        "exact_single_row", "state", "exit_code", "allocated_gpu_count",
        "alloc_tres", "node_list", "start", "end", "elapsed",
        "submit_line_sha256", "exact_submit_line",
    }
    world8_fields = {
        "retained_fd_world8_admission_path",
        "retained_fd_world8_admission_sha256",
        "retained_fd_world8_admission_digest",
        "compute_bash_probe_admission_path",
        "compute_bash_probe_admission_sha256",
        "compute_bash_probe_admission_digest",
        "archive_member_manifest_sha256", "archive_member_count",
        "archive_regular_file_count", "archive_directory_count",
        "runtime_origin_manifest_sha256", "runtime_origin_project_module_count",
        "underlying_world8_closure_deep_validated",
    }
    authority = {
        "scientific": False, "action": False, "identity": False,
        "training": False, "checkpoint": False, "publication": False,
    }
    expected_tres = {
        "billing": "32", "cpu": "32", "gres/gpu:mi210": "8",
        "gres/gpu": "8", "mem": "256G", "node": "1",
    }
    expected_sampling = {
        "model": "Bernini-R-1.3B-Diffusers", "native_arm": "t2v",
        "guidance_mode": "t2v_apg", "num_frames": 81,
        "latent_frames": 21, "fps": 25, "num_inference_steps": 40,
        "ulysses_size": 4,
        "target_initialization": "official_gen_wanx22_fresh_gaussian",
        "same_row_seed_gaussian_across_branches": True,
    }
    expected_semantic_closure = {
        "accepted_semantic_inputs": [
            "identity_scene_caption", "branch_start_state_caption",
            "branch_instruction",
        ],
        "real_source_video_path_present": False,
        "real_source_rgb_read": False,
        "real_source_latent_read_or_created": False,
        "real_source_noise_read_or_created": False,
        "target_video": False, "reference_image_or_video": False,
        "mask_flow_pose_track_trajectory": False, "motion_donor": False,
        "generated_proposal_as_condition_target_donor_or_noise": False,
    }
    expected_geometry_proxy = {
        "content": "constant_black_frames_created_without_source_media",
        "num_frames": 81, "fps": 25, "audio": False,
        "role": "legacy_native_runner_bucket_shape_only",
        "pixels_enter_transformer": False, "vae_latent_created": False,
        "source_path_or_bytes_used_to_create_proxy": False,
        "proxy_sha_bound_before_gpu_render": True,
    }
    expected_artifact_authority = {
        "proposal_media_requires_detached_full81_audit": True,
        "event_verified": False, "identity_preservation_verified": False,
        "seed_selection_authorized": False,
        "training_target_authorized": False,
        "optimizer_update_authorized": False,
    }
    sha = lambda item: type(item) is str and SHA256.fullmatch(item) is not None
    master = value.get("master_receipt") if isinstance(value, dict) else None
    submission = value.get("submission_receipt") if isinstance(value, dict) else None
    slurm = value.get("slurm_terminal_observation") if isinstance(value, dict) else None
    world8 = value.get("world8_closure") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != root_fields
        or value.get("schema_version")
        != "saic-t2v-topup-r6-formal-full60-terminal-admission-v1"
        or value.get("schema_version") != FULL60_ADMISSION_SCHEMA
        or value.get("status") != "terminal_completed_formal_full60_admitted"
        or value.get("status") != FULL60_ADMISSION_STATUS
        or type(value.get("job_id")) is not str
        or not value["job_id"].isdigit()
        or value.get("job_success") is not True
        or value.get("slurm_terminal_verified") is not True
        or value.get("formal_admission") is not True
        or value.get("formal_full60_result_claimed") is not True
        or value.get("candidate_count") != 60
        or value.get("attempt_count") != 60
        or value.get("same_seed_proof_count") != 20
        or value.get("failure_receipt_absent") is not True
        or value.get("output_namespace_deep_closed") is not True
        or value.get("authority") != authority
        or not isinstance(master, dict) or set(master) != master_fields
        or type(master.get("path")) is not str
        or not Path(master["path"]).is_absolute()
        or Path(master["path"]).name
        != "saic-pure-t2v-event-bank-topup-receipt.json"
        or not sha(master.get("sha256"))
        or not sha(master.get("receipt_digest"))
        or master.get("schema_version")
        != "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
        or master.get("bank_id")
        != "saic-text-only-hard-negative-topup-exact81-v2"
        or master.get("attempt_count") != 60
        or master.get("row_count") != 8
        or master.get("seed_cell_count") != 20
        or master.get("branch_order")
        != ["incomplete", "camera_only", "appearance_only"]
        or master.get("master_deep_validated") is not True
        or not isinstance(submission, dict)
        or set(submission) != submission_fields
        or type(submission.get("path")) is not str
        or not Path(submission["path"]).is_absolute()
        or submission["path"]
        != str(Path(master["path"]).parent) + ".submission.receipt.json"
        or not sha(submission.get("sha256"))
        or not sha(submission.get("receipt_digest"))
        or submission.get("job_id") != value.get("job_id")
        or submission.get("request_candidate_count") != 60
        or submission.get("submission_deep_validated") is not True
        or not isinstance(slurm, dict) or set(slurm) != slurm_fields
        or not all(sha(slurm.get(name)) for name in (
            "executable_sha256", "stdout_sha256", "stderr_sha256",
            "submit_line_sha256",
        ))
        or slurm.get("exact_single_row") is not True
        or slurm.get("state") != "COMPLETED"
        or slurm.get("exit_code") != "0:0"
        or slurm.get("allocated_gpu_count") != 8
        or slurm.get("alloc_tres") != expected_tres
        or slurm.get("exact_submit_line") is not True
        or any(type(slurm.get(name)) is not str or not slurm[name] for name in (
            "node_list", "start", "end", "elapsed",
        ))
        or not isinstance(world8, dict) or set(world8) != world8_fields
        or not all(sha(world8.get(name)) for name in (
            "retained_fd_world8_admission_sha256",
            "retained_fd_world8_admission_digest",
            "compute_bash_probe_admission_sha256",
            "compute_bash_probe_admission_digest",
        ))
        or world8.get("archive_member_manifest_sha256")
        != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
        or world8.get("archive_member_count") != 864
        or world8.get("archive_regular_file_count") != 853
        or world8.get("archive_directory_count") != 11
        or world8.get("runtime_origin_manifest_sha256")
        != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
        or world8.get("runtime_origin_project_module_count") != 14
        or world8.get("underlying_world8_closure_deep_validated") is not True
    ):
        die("formal full60 exact deep admission differs")

    def read_reference(
        path_value: Any, expected_sha: Any, expected_digest: Any,
        *, label: str, decode_json: bool,
    ) -> tuple[bytes, Any]:
        if type(path_value) is not str or type(expected_sha) is not str:
            die(f"{label} reference binding differs")
        path = Path(path_value)
        if not path.is_absolute() or path == Path("/"):
            die(f"{label} reference path differs")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor); public = path.lstat()
            if (
                path.resolve(strict=True) != path
                or not stat.S_ISREG(before.st_mode)
                or not stat.S_ISREG(public.st_mode)
                or stat.S_ISLNK(public.st_mode)
                or before.st_uid != os.getuid()
                or public.st_uid != os.getuid()
                or before.st_nlink != 1 or public.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o022
                or (before.st_dev, before.st_ino)
                != (public.st_dev, public.st_ino)
            ):
                die(f"{label} reference identity differs")
            digest = hashlib.sha256(); chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if decode_json:
                    chunks.append(chunk)
            after = os.fstat(descriptor); public_after = path.lstat()
            identity = lambda item: (
                item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                item.st_uid, item.st_nlink, stat.S_IMODE(item.st_mode),
            )
            if (
                identity(before) != identity(after)
                or identity(after) != identity(public_after)
                or digest.hexdigest() != expected_sha
            ):
                die(f"{label} reference changed during retained read")
            raw = b"".join(chunks) if decode_json else b""
        finally:
            os.close(descriptor)
        if not decode_json:
            return raw, None
        try:
            decoded = json.loads(raw.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            die(f"{label} reference encoding differs: {error}")
        unsigned_reference = dict(decoded) if isinstance(decoded, dict) else {}
        claimed_reference = unsigned_reference.pop("receipt_digest", None)
        if (
            not isinstance(decoded, dict)
            or raw != canonical(decoded) + b"\n"
            or claimed_reference != expected_digest
            or not sha(claimed_reference)
            or sha_bytes(canonical(unsigned_reference)) != claimed_reference
        ):
            die(f"{label} reference canonical seal differs")
        return raw, decoded

    _, master_value = read_reference(
        master["path"], master["sha256"], master["receipt_digest"],
        label="formal full60 master", decode_json=True,
    )
    master_exact_fields = {
        "schema_version", "bank_id", "top_up_only", "root_spec_raw_sha256",
        "base_v1_spec_raw_sha256", "base_v1_spec_content_sha256",
        "source_manifest_content_sha256", "topology", "sampling_contract",
        "semantic_input_closure", "geometry_proxy_contract",
        "artifact_authority", "attempt_count", "row_count", "seed_cell_count",
        "branch_order", "merged_branch_order",
        "six_branch_spec_merge_cell_count",
        "same_seed_official_gaussian_proofs", "attempts",
        "detached_full81_event_review_complete", "event_verified",
        "identity_preservation_verified", "seed_selection_authorized",
        "training_target_authorized",
        "optimizer_or_parameter_update_authorized", "receipt_digest",
    }
    attempt_fields = {
        "candidate_id", "row_id", "iid", "analysis_split", "branch", "seed",
        "receipt_path", "receipt_sha256", "receipt_digest", "mp4_path",
        "mp4_sha256", "event_audit_status",
    }
    proof_fields = {
        "iid", "seed", "branch_order",
        "official_gaussian_tensor_values_byte_equal",
        "official_gaussian_identity_digest",
    }
    attempt_receipt_fields = {
        "schema_version", "bank_id", "top_up_only", "root_spec_raw_sha256",
        "base_v1_spec_raw_sha256", "candidate_envelope_path",
        "candidate_envelope_sha256", "group_id", "actor_family",
        "visible_gpus", "candidate", "sampling_contract",
        "semantic_input_closure", "geometry_proxy_contract",
        "artifact_authority", "runtime_topology",
        "real_source_nonuse_certificate", "native_receipt_path",
        "native_receipt_sha256", "native_receipt_digest", "bucket_hw",
        "latent_shape", "artifacts", "event_audit_status", "event_verified",
        "identity_preservation_verified", "seed_selection_authorized",
        "training_target_authorized",
        "optimizer_or_parameter_update_authorized", "receipt_digest",
    }
    candidate_fields = {
        "candidate_id", "ordinal", "row_id", "iid", "analysis_split",
        "actor_family", "action_family_id", "initial_state_type",
        "terminal_state_type", "source_media_sha256_for_nonuse_audit",
        "source_geometry_hw", "source_caption_utf8_sha256",
        "identity_scene_caption", "identity_scene_caption_utf8_sha256",
        "branch", "hard_negative_relation", "branch_start_state_caption",
        "branch_start_state_caption_utf8_sha256", "branch_instruction",
        "branch_instruction_utf8_sha256", "full_t2v_caption",
        "full_t2v_caption_utf8_sha256", "seed", "paired_v1_candidate_ids",
        "paired_v1_cell_digest", "event_audit_status", "event_verified",
        "identity_preservation_verified", "seed_selection_authorized",
        "training_target_authorized", "optimizer_authorized",
    }
    nonuse_fields = {
        "source_media_sha256_for_nonuse_audit",
        "real_source_path_present_in_candidate_envelope",
        "real_source_path_passed_to_native_runner", "real_source_rgb_read",
        "real_source_latent_read_or_created",
        "real_source_noise_read_or_created", "target_video_read_or_created",
        "reference_image_or_video_read", "motion_donor_read",
        "geometry_proxy_path", "geometry_proxy_sha256",
        "proxy_bytes_differ_from_real_source",
        "proxy_used_for_bucket_shape_only", "proxy_vae_latent_created",
        "proxy_pixels_entered_transformer", "native_content_conditioning_count",
    }
    attempts = master_value.get("attempts")
    proofs = master_value.get("same_seed_official_gaussian_proofs")
    if (
        set(master_value) != master_exact_fields
        or master_value.get("schema_version") != master["schema_version"]
        or master_value.get("bank_id") != master["bank_id"]
        or master_value.get("top_up_only") is not True
        or master_value.get("root_spec_raw_sha256")
        != "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
        or master_value.get("base_v1_spec_raw_sha256")
        != "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
        or master_value.get("base_v1_spec_content_sha256")
        != "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
        or master_value.get("source_manifest_content_sha256")
        != "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
        or master_value.get("topology")
        != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master_value.get("sampling_contract") != expected_sampling
        or master_value.get("semantic_input_closure")
        != expected_semantic_closure
        or master_value.get("geometry_proxy_contract")
        != expected_geometry_proxy
        or master_value.get("artifact_authority")
        != expected_artifact_authority
        or master_value.get("attempt_count") != 60
        or master_value.get("row_count") != 8
        or master_value.get("seed_cell_count") != 20
        or master_value.get("branch_order") != master["branch_order"]
        or master_value.get("merged_branch_order") != [
            "forward", "reverse", "noop", "incomplete", "camera_only",
            "appearance_only",
        ]
        or master_value.get("six_branch_spec_merge_cell_count") != 20
        or any(not isinstance(master_value.get(name), dict) for name in (
            "sampling_contract", "semantic_input_closure",
            "geometry_proxy_contract", "artifact_authority",
        ))
        or not isinstance(attempts, list) or len(attempts) != 60
        or not isinstance(proofs, list) or len(proofs) != 20
        or any(master_value.get(name) is not False for name in (
            "detached_full81_event_review_complete", "event_verified",
            "identity_preservation_verified", "seed_selection_authorized",
            "training_target_authorized",
            "optimizer_or_parameter_update_authorized",
        ))
    ):
        die("formal full60 master deep closure differs")
    candidate_ids: set[str] = set()
    attempt_cells: set[tuple[str, int]] = set()
    cell_branches: dict[tuple[str, int], set[str]] = {}
    source_rows: set[str] = set()
    ordinals_by_group: dict[str, set[int]] = {"sp4-a": set(), "sp4-b": set()}
    attempt_root = Path(master["path"]).parent / "attempts"
    for attempt in attempts:
        candidate_id = attempt.get("candidate_id") if isinstance(attempt, dict) else None
        receipt_path = Path(str(attempt.get("receipt_path", ""))) if isinstance(
            attempt, dict
        ) else Path(".")
        mp4_path = Path(str(attempt.get("mp4_path", ""))) if isinstance(
            attempt, dict
        ) else Path(".")
        if (
            not isinstance(attempt, dict)
            or set(attempt) != attempt_fields
            or type(candidate_id) is not str
            or re.fullmatch(
                r"saic-topup-v2-[0-9a-f]{16}-"
                r"(?:incomplete|camera_only|appearance_only)-s[0-9]+",
                candidate_id,
            ) is None
            or candidate_id in candidate_ids
            or attempt.get("analysis_split") not in {"fit", "confirmation"}
            or re.fullmatch(
                r"(?:fit|confirmation)-(?:dog|human)-0[01]-[0-9a-f]{16}",
                str(attempt.get("row_id", "")),
            ) is None
            or not str(attempt.get("row_id", "")).endswith(
                "-" + str(attempt.get("iid", ""))
            )
            or attempt.get("branch")
            not in {"incomplete", "camera_only", "appearance_only"}
            or type(attempt.get("seed")) is not int
            or attempt["seed"] < 0
            or not candidate_id.endswith(
                f"-{attempt.get('branch')}-s{attempt.get('seed')}"
            )
            or attempt.get("event_audit_status")
            != "pending_detached_full81_review"
            or not sha(attempt.get("receipt_sha256"))
            or not sha(attempt.get("receipt_digest"))
            or not sha(attempt.get("mp4_sha256"))
            or receipt_path != attempt_root / candidate_id
                / "saic-event-topup-generation-receipt.json"
            or mp4_path.parent != receipt_path.parent
            or mp4_path.name != "t2v.mp4"
        ):
            die("formal full60 attempt row closure differs")
        _, attempt_value = read_reference(
            str(receipt_path), attempt["receipt_sha256"],
            attempt["receipt_digest"], label=f"formal attempt {candidate_id}",
            decode_json=True,
        )
        candidate = attempt_value.get("candidate")
        artifacts = attempt_value.get("artifacts")
        mp4_artifact = artifacts.get("mp4") if isinstance(artifacts, dict) else None
        nonuse = attempt_value.get("real_source_nonuse_certificate")
        formal_root = Path(master["path"]).parent
        actor = attempt.get("row_id", "").split("-")[1]
        expected_group = "sp4-a" if actor == "dog" else "sp4-b"
        expected_visible = [0, 1, 2, 3] if actor == "dog" else [4, 5, 6, 7]
        if (
            set(attempt_value) != attempt_receipt_fields
            or attempt_value.get("schema_version")
            != "bernini-saic-pure-t2v-event-topup-generation-receipt-v2"
            or attempt_value.get("bank_id") != master_value.get("bank_id")
            or attempt_value.get("top_up_only") is not True
            or attempt_value.get("root_spec_raw_sha256")
            != master_value.get("root_spec_raw_sha256")
            or attempt_value.get("base_v1_spec_raw_sha256")
            != master_value.get("base_v1_spec_raw_sha256")
            or attempt_value.get("sampling_contract") != expected_sampling
            or attempt_value.get("semantic_input_closure")
            != expected_semantic_closure
            or attempt_value.get("geometry_proxy_contract")
            != expected_geometry_proxy
            or attempt_value.get("artifact_authority")
            != expected_artifact_authority
            or not isinstance(candidate, dict) or set(candidate) != candidate_fields
            or candidate.get("candidate_id") != candidate_id
            or candidate.get("iid") != attempt.get("iid")
            or candidate.get("row_id") != attempt.get("row_id")
            or candidate.get("actor_family") != actor
            or candidate.get("branch") != attempt.get("branch")
            or candidate.get("seed") != attempt.get("seed")
            or candidate.get("analysis_split") != attempt.get("analysis_split")
            or type(candidate.get("ordinal")) is not int
            or not 0 <= candidate["ordinal"] < 30
            or candidate.get("event_audit_status")
            != "pending_detached_full81_review"
            or any(candidate.get(name) is not False for name in (
                "event_verified", "identity_preservation_verified",
                "seed_selection_authorized", "training_target_authorized",
                "optimizer_authorized",
            ))
            or any(not sha(candidate.get(name)) for name in (
                "source_media_sha256_for_nonuse_audit",
                "source_caption_utf8_sha256",
                "identity_scene_caption_utf8_sha256",
                "branch_start_state_caption_utf8_sha256",
                "branch_instruction_utf8_sha256",
                "full_t2v_caption_utf8_sha256", "paired_v1_cell_digest",
            ))
            or any(
                type(candidate.get(text_name)) is not str
                or sha_bytes(candidate[text_name].encode("utf-8"))
                != candidate.get(text_name + "_utf8_sha256")
                for text_name in (
                    "identity_scene_caption", "branch_start_state_caption",
                    "branch_instruction", "full_t2v_caption",
                )
            )
            or attempt_value.get("group_id") != expected_group
            or attempt_value.get("actor_family") != actor
            or attempt_value.get("visible_gpus") != expected_visible
            or attempt_value.get("runtime_topology") != {
                "world_size": 4, "ulysses_size": 4,
                "rocr_visible_devices": ",".join(map(str, expected_visible)),
            }
            or not isinstance(nonuse, dict) or set(nonuse) != nonuse_fields
            or nonuse.get("source_media_sha256_for_nonuse_audit")
            != candidate.get("source_media_sha256_for_nonuse_audit")
            or any(nonuse.get(name) is not False for name in (
                "real_source_path_present_in_candidate_envelope",
                "real_source_path_passed_to_native_runner",
                "real_source_rgb_read", "real_source_latent_read_or_created",
                "real_source_noise_read_or_created",
                "target_video_read_or_created", "reference_image_or_video_read",
                "motion_donor_read", "proxy_vae_latent_created",
                "proxy_pixels_entered_transformer",
            ))
            or nonuse.get("proxy_bytes_differ_from_real_source") is not True
            or nonuse.get("proxy_used_for_bucket_shape_only") is not True
            or nonuse.get("native_content_conditioning_count") != 0
            or not sha(nonuse.get("geometry_proxy_sha256"))
            or not isinstance(artifacts, dict)
            or set(artifacts) != {
                "mp4", "predecode_clean_latent", "official_initial_gaussian",
            }
            or not isinstance(mp4_artifact, dict)
            or mp4_artifact.get("path") != str(mp4_path)
            or mp4_artifact.get("sha256") != attempt["mp4_sha256"]
            or attempt_value.get("event_audit_status")
            != "pending_detached_full81_review"
            or any(attempt_value.get(name) is not False for name in (
                "event_verified", "identity_preservation_verified",
                "seed_selection_authorized", "training_target_authorized",
                "optimizer_or_parameter_update_authorized",
            ))
        ):
            die("formal full60 attempt receipt deep closure differs")
        envelope_path = Path(str(attempt_value["candidate_envelope_path"]))
        native_path = Path(str(attempt_value["native_receipt_path"]))
        proxy_path = Path(str(nonuse["geometry_proxy_path"]))
        if (
            not sha(attempt_value.get("candidate_envelope_sha256"))
            or not sha(attempt_value.get("native_receipt_sha256"))
            or not sha(attempt_value.get("native_receipt_digest"))
            or envelope_path != formal_root / "plan" / expected_group / (
                f"{candidate['ordinal']:04d}-{candidate_id}.json"
            )
            or native_path.parent != receipt_path.parent
            or native_path.name != "receipt.json"
            or not proxy_path.is_relative_to(formal_root)
        ):
            die("formal full60 attempt referenced-input closure differs")
        read_reference(
            str(envelope_path), attempt_value["candidate_envelope_sha256"], None,
            label=f"formal attempt envelope {candidate_id}", decode_json=False,
        )
        read_reference(
            str(native_path), attempt_value["native_receipt_sha256"], None,
            label=f"formal attempt native receipt {candidate_id}", decode_json=False,
        )
        read_reference(
            str(proxy_path), nonuse["geometry_proxy_sha256"], None,
            label=f"formal attempt geometry proxy {candidate_id}", decode_json=False,
        )
        for artifact_name, artifact in artifacts.items():
            expected_artifact_name = {
                "mp4": "t2v.mp4",
                "predecode_clean_latent": "t2v.normalized-clean-latent.safetensors",
                "official_initial_gaussian": "t2v.official-initial-gaussian.safetensors",
            }[artifact_name]
            if (
                not isinstance(artifact, dict)
                or type(artifact.get("path")) is not str
                or not sha(artifact.get("sha256"))
                or not Path(artifact["path"]).is_relative_to(receipt_path.parent)
                or Path(artifact["path"]).parent != receipt_path.parent
                or Path(artifact["path"]).name != expected_artifact_name
            ):
                die("formal full60 attempt artifact closure differs")
            read_reference(
                artifact["path"], artifact["sha256"], None,
                label=f"formal attempt {artifact_name} {candidate_id}",
                decode_json=False,
            )
        candidate_ids.add(candidate_id)
        ordinals_by_group[expected_group].add(candidate["ordinal"])
        cell = (str(attempt.get("iid")), int(attempt["seed"]))
        attempt_cells.add(cell)
        cell_branches.setdefault(cell, set()).add(str(attempt["branch"]))
        source_rows.add(str(attempt["row_id"]))
    proof_cells: set[tuple[str, int]] = set()
    for proof in proofs:
        if not isinstance(proof, dict):
            die("formal full60 Gaussian proof closure differs")
        cell = (
            str(proof.get("iid", "")),
            proof.get("seed"),
        )
        if (
            set(proof) != proof_fields
            or type(proof.get("seed")) is not int
            or cell in proof_cells
            or proof.get("branch_order")
            != ["incomplete", "camera_only", "appearance_only"]
            or proof.get("official_gaussian_tensor_values_byte_equal") is not True
            or not sha(proof.get("official_gaussian_identity_digest"))
        ):
            die("formal full60 Gaussian proof closure differs")
        proof_cells.add(cell)
    if (
        len(candidate_ids) != 60
        or len(source_rows) != 8
        or len(attempt_cells) != 20
        or attempt_cells != proof_cells
        or any(ordinals != set(range(30)) for ordinals in ordinals_by_group.values())
        or any(branches != {"incomplete", "camera_only", "appearance_only"}
               for branches in cell_branches.values())
    ):
        die("formal full60 candidate/cell closure differs")

    _, submission_value = read_reference(
        submission["path"], submission["sha256"], submission["receipt_digest"],
        label="formal full60 submission", decode_json=True,
    )
    submitted_job = submission_value.get("submitted_job")
    request = submission_value.get("request")
    inputs = submission_value.get("inputs")
    outputs = submission_value.get("outputs")
    submission_exact_fields = {
        "schema_version", "status", "submission_success", "job_success",
        "submitted_job", "request", "submission_boundary", "inputs",
        "canary_admission", "outputs", "authority", "threat_model",
        "receipt_digest",
    }
    if (
        set(submission_value) != submission_exact_fields
        or submission_value.get("schema_version")
        != "saic-t2v-topup-r6-formal-v2-r2-submission-v1"
        or submission_value.get("status") != "submitted"
        or submission_value.get("submission_success") is not True
        or submission_value.get("job_success") is not None
        or not isinstance(submitted_job, dict)
        or set(submitted_job) != {
            "job_id", "cluster", "stdout_sha256", "stderr_sha256",
        }
        or submitted_job.get("job_id") != value["job_id"]
        or not sha(submitted_job.get("stdout_sha256"))
        or not sha(submitted_job.get("stderr_sha256"))
        or not isinstance(request, dict)
        or request.get("candidate_count") != 60
        or request.get("job_name") != "saic-t2v-topup-r6-v2-r2"
        or request.get("partition") != "faculty"
        or request.get("qos") != "bgqos"
        or request.get("nodes") != 1 or request.get("ntasks") != 1
        or request.get("cpus_per_task") != 32
        or request.get("memory") != "256G"
        or request.get("walltime") != "24:00:00"
        or request.get("world_topology") != "two_concurrent_world4_sp4"
        or request.get("gpu_resource_requested") != "gpu:mi210:8"
        or request.get("hold") is not False
        or request.get("dependency") is not None
        or not isinstance(inputs, dict)
        or inputs.get("event_spec_sha256")
        != master_value.get("root_spec_raw_sha256")
        or inputs.get("source_archive_sha256")
        != "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
        or inputs.get("retained_fd_canary_admission_sha256")
        != world8["retained_fd_world8_admission_sha256"]
        or inputs.get("retained_fd_canary_admission_digest")
        != world8["retained_fd_world8_admission_digest"]
        or inputs.get("compute_bash_probe_admission_sha256")
        != world8["compute_bash_probe_admission_sha256"]
        or inputs.get("compute_bash_probe_admission_digest")
        != world8["compute_bash_probe_admission_digest"]
        or not isinstance(outputs, dict)
        or outputs.get("output_root") != str(Path(master["path"]).parent)
        or outputs.get("submission_receipt") != submission["path"]
        or outputs.get("fresh_before_submission") is not True
        or submission_value.get("authority") != {
            "diagnostic_event_bank_execution_authorized": True,
            "training": False, "checkpoint": False,
            "scientific_success_claimed": False,
            "action_edit_success_claimed": False,
            "job_success_claimed": False,
        }
    ):
        die("formal full60 submission deep closure differs")

    for prefix, schema, status in (
        (
            "retained_fd_world8_admission",
            "saic-formal-v2-retained-fd-world8-canary-admission-v1",
            "terminal_completed_retained_fd_world8_operational_admitted",
        ),
        (
            "compute_bash_probe_admission",
            "saic-compute-bash-retained-fd-probe-admission-v1",
            "terminal_completed_compute_bash_retained_fd_admitted",
        ),
    ):
        _, proof = read_reference(
            world8[prefix + "_path"], world8[prefix + "_sha256"],
            world8[prefix + "_digest"], label=prefix, decode_json=True,
        )
        if (
            proof.get("schema_version") != schema
            or proof.get("status") != status
            or proof.get("job_success") is not True
            or proof.get("slurm_terminal_verified") is not True
        ):
            die(f"{prefix} deep closure differs")
        if prefix == "retained_fd_world8_admission" and (
            proof.get("operational_canary_admitted") is not True
            or proof.get("formal_admission") is not False
            or proof.get("source_archive_sha256")
            != "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
            or proof.get("archive_member_manifest_sha256")
            != world8["archive_member_manifest_sha256"]
            or proof.get("archive_member_count") != 864
            or proof.get("archive_regular_file_count") != 853
            or proof.get("archive_directory_count") != 11
            or proof.get("runtime_origin_manifest_sha256")
            != world8["runtime_origin_manifest_sha256"]
            or proof.get("runtime_origin_project_module_count") != 14
            or proof.get("underlying_world8_closure_deep_validated") is not True
            or proof.get("science_generation_entered") is not False
            or not isinstance(proof.get("probe_admission_binding"), dict)
            or proof["probe_admission_binding"].get("sha256")
            != world8["compute_bash_probe_admission_sha256"]
            or proof["probe_admission_binding"].get("receipt_digest")
            != world8["compute_bash_probe_admission_digest"]
        ):
            die("retained-FD WORLD8 admission deep closure differs")

    sacct_fields = (
        "JobIDRaw,State,ExitCode,AllocTRES%512,NodeList,Start,End,Elapsed,"
        "SubmitLine%8192"
    )
    sacct_info = SACCT.lstat()
    sacct_descriptor = os.open(SACCT, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        sacct_before = os.fstat(sacct_descriptor)
        sacct_digest = hashlib.sha256()
        while True:
            chunk = os.read(sacct_descriptor, 1024 * 1024)
            if not chunk:
                break
            sacct_digest.update(chunk)
        sacct_after = os.fstat(sacct_descriptor)
    finally:
        os.close(sacct_descriptor)
    if (
        SACCT.resolve(strict=True) != SACCT
        or not stat.S_ISREG(sacct_info.st_mode)
        or stat.S_ISLNK(sacct_info.st_mode)
        or sacct_info.st_uid != 0 or sacct_info.st_nlink != 1
        or stat.S_IMODE(sacct_info.st_mode) & 0o022
        or not os.access(SACCT, os.X_OK)
        or (sacct_info.st_dev, sacct_info.st_ino)
        != (sacct_before.st_dev, sacct_before.st_ino)
        or (sacct_before.st_dev, sacct_before.st_ino, sacct_before.st_size,
            sacct_before.st_mtime_ns)
        != (sacct_after.st_dev, sacct_after.st_ino, sacct_after.st_size,
            sacct_after.st_mtime_ns)
        or sacct_digest.hexdigest() != SACCT_SHA256
    ):
        die("formal full60 sacct executable identity differs")
    completed = subprocess.run(
        [
            str(SACCT), "-j", value["job_id"], "-X", "--noheader", "-n", "-P",
            "-o", sacct_fields,
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        die(f"formal full60 sacct output encoding differs: {error}")
    columns = lines[0].split("|") if len(lines) == 1 else []
    alloc = {}
    if len(columns) == 9:
        alloc = dict(
            item.split("=", 1) for item in columns[3].split(",") if "=" in item
        )
    if (
        completed.returncode != 0
        or len(columns) != 9
        or columns[0] != value["job_id"]
        or columns[1] != "COMPLETED"
        or columns[2] != "0:0"
        or alloc != expected_tres
        or slurm.get("executable_sha256") != SACCT_SHA256
        or slurm.get("stdout_sha256") != sha_bytes(completed.stdout)
        or slurm.get("stderr_sha256") != sha_bytes(completed.stderr)
        or slurm.get("state") != columns[1]
        or slurm.get("exit_code") != columns[2]
        or slurm.get("alloc_tres") != alloc
        or slurm.get("node_list") != columns[4]
        or slurm.get("start") != columns[5]
        or slurm.get("end") != columns[6]
        or slurm.get("elapsed") != columns[7]
        or slurm.get("submit_line_sha256")
        != sha_bytes(columns[8].encode("ascii"))
        or re.fullmatch(r".* /proc/self/fd/([0-9]+)", columns[8]) is None
        or "--export=NONE," not in columns[8]
        or "--hold" in columns[8]
        or "--dependency" in columns[8]
    ):
        die("formal full60 live terminal Slurm closure differs")




def retained_bytes(
    path: Path,
    expected_sha: str | None,
    label: str,
    *,
    expected_mode: int | None = 0o444,
) -> tuple[int, bytes]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} path differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    before = os.fstat(descriptor); leaf = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(leaf.st_mode)
        or stat.S_ISLNK(leaf.st_mode) or before.st_nlink != 1 or leaf.st_nlink != 1
        or before.st_uid != os.getuid() or leaf.st_uid != os.getuid()
        or (
            expected_mode is not None
            and stat.S_IMODE(before.st_mode) != expected_mode
        )
        or (
            expected_mode is None
            and stat.S_IMODE(before.st_mode) & 0o022
        )
        or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
    ):
        os.close(descriptor); die(f"{label} identity differs")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk: break
        chunks.append(chunk)
    raw = b"".join(chunks)
    after = os.fstat(descriptor); leaf_after = path.lstat()
    key = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_nlink, stat.S_IMODE(value.st_mode),
    )
    if (
        key(before) != key(after)
        or key(after) != key(leaf_after)
        or (expected_sha is not None and sha_bytes(raw) != expected_sha)
    ):
        os.close(descriptor); die(f"{label} retained bytes differ")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return descriptor, raw


def reread_retained(
    descriptor: int,
    path: Path,
    raw: bytes,
    label: str,
    *,
    expected_mode: int | None = 0o444,
) -> None:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    after = os.fstat(descriptor); leaf = path.lstat()
    if (
        b"".join(chunks) != raw
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (after.st_dev, after.st_ino) != (leaf.st_dev, leaf.st_ino)
        or after.st_nlink != 1 or leaf.st_nlink != 1
        or after.st_uid != os.getuid() or leaf.st_uid != os.getuid()
        or not stat.S_ISREG(after.st_mode) or not stat.S_ISREG(leaf.st_mode)
        or stat.S_ISLNK(leaf.st_mode)
        or (
            expected_mode is not None
            and stat.S_IMODE(after.st_mode) != expected_mode
        )
        or (
            expected_mode is None and stat.S_IMODE(after.st_mode) & 0o022
        )
    ):
        die(f"{label} changed during postflight")


def decode_sealed(raw: bytes, expected_digest: str | None, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        die(f"{label} encoding differs: {error}")
    if not isinstance(value, dict):
        die(f"{label} is not one object")
    unsigned = dict(value); claimed = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None
        or (expected_digest is not None and claimed != expected_digest)
        or sha_bytes(canonical(unsigned)) != claimed
        or raw != canonical(value) + b"\n"
    ):
        die(f"{label} canonical seal differs")
    return value


def exact_directory(path: Path, label: str, mode: int) -> tuple[int, int]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} path differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode
    ):
        die(f"{label} identity differs")
    return info.st_dev, info.st_ino


def parse_tres(value: str, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in value.split(","):
        if token.count("=") != 1:
            die(f"{label} token differs")
        key, item = token.split("=", 1)
        if not key or key in result or not item:
            die(f"{label} closure differs")
        result[key] = item
    return result


def exact_executable(path: Path, expected_sha: str, label: str) -> None:
    info = path.lstat()
    if (
        path.resolve(strict=True) != path or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1
        or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(path, os.X_OK)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha
    ):
        die(f"root-owned {label} executable differs")


def observe_sacct(job_id: str, submission: dict[str, Any]) -> dict[str, Any]:
    exact_executable(SACCT, SACCT_SHA256, "sacct")
    argv = [
        str(SACCT), "-j", job_id, "-X", "--noheader", "-n", "-P",
        "-o", ",".join(SACCT_FIELDS),
    ]
    completed = subprocess.run(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("sacct stdout is not ASCII")
    lines = stdout.splitlines(); fields = lines[0].split("|") if len(lines) == 1 else []
    row = dict(zip(SACCT_KEYS, fields, strict=True)) if len(fields) == len(SACCT_KEYS) else {}
    expected_tres = {
        "billing": "32", "cpu": "32", "gres/gpu:mi210": "8",
        "gres/gpu": "8", "mem": "256G", "node": "1",
    }
    boundary = submission["submission_boundary"]
    exports = {
        "SAIC_ANCHOR_ARCHIVE": str(SOURCE_ARCHIVE),
        "SAIC_ANCHOR_TRAINER": str(TRAINER),
        "SAIC_ANCHOR_SOURCE_MANIFEST": str(SOURCE_MANIFEST),
        "SAIC_ANCHOR_CHECKPOINT_MANIFEST": str(CHECKPOINT_MANIFEST),
        "SAIC_ANCHOR_FULL60_ADMISSION": str(FULL60_ADMISSION),
        "SAIC_ANCHOR_RELEASE_MANIFEST": str(RELEASE_MANIFEST),
        "SAIC_ANCHOR_SUBMISSION_RECEIPT": str(SUBMISSION_RECEIPT),
        "SAIC_ANCHOR_OUTPUT_PARENT": str(OUTPUT_PARENT),
        "SAIC_ANCHOR_RELEASE_MANIFEST_SHA256": submission["inputs"][
            "release_manifest"
        ]["sha256"],
        "SAIC_ANCHOR_RELEASE_MANIFEST_DIGEST": submission["inputs"][
            "release_manifest"
        ]["receipt_digest"],
        "SAIC_ANCHOR_WRAPPER_SHA256": WRAPPER_SHA256,
    }
    expected_submit_line = " ".join([
        "/usr/bin/sbatch", "--parsable",
        f"--output={LOG_DIR}/saic-anchor-f32-r1-%j.out",
        f"--error={LOG_DIR}/saic-anchor-f32-r1-%j.err",
        "--export=NONE," + ",".join(
            f"{name}={exports[name]}" for name in EXPORT_NAMES
        ),
        f"/proc/self/fd/{boundary['wrapper_fd_number']}",
    ])
    if (
        completed.returncode != 0 or completed.stderr or len(lines) != 1
        or row.get("JobIDRaw") != job_id
        or row.get("JobName") != "saic-anchor-f32-r1"
        or row.get("Partition") != "faculty" or row.get("QOS") != "bgqos"
        or row.get("State") != "COMPLETED" or row.get("ExitCode") != "0:0"
        or parse_tres(str(row.get("ReqTRES", "")), "ReqTRES") != expected_tres
        or parse_tres(str(row.get("AllocTRES", "")), "AllocTRES") != expected_tres
        or not row.get("NodeList") or row.get("NodeList") in {"Unknown", "None assigned"}
        or not row.get("Start") or row.get("Start") == "Unknown"
        or not row.get("End") or row.get("End") == "Unknown"
        or ELAPSED.fullmatch(str(row.get("Elapsed", ""))) is None
        or list(exports) != EXPORT_NAMES
        or row.get("SubmitLine") != expected_submit_line
        or not isinstance(boundary.get("wrapper_fd_number"), int)
        or boundary["wrapper_fd_number"] < 3
    ):
        die("authoritative terminal Slurm accounting differs")
    return {
        "executable": str(SACCT), "executable_sha256": SACCT_SHA256,
        "argv": argv, "query_fields": SACCT_FIELDS,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "exact_single_row": True, "parsed_row": row,
        "request_tres_exact": True, "allocation_tres_exact": True,
        "submit_line_exact_retained_fd": True,
    }


def write_create_only(path: Path, payload: bytes, mode: int) -> None:
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0: die("create-only publication stalled")
            view = view[count:]
        os.fsync(descriptor); os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, len(payload) + 1) != payload:
            die("create-only same-FD reread differs")
        retained = os.fstat(descriptor); public = path.lstat()
        if (
            (retained.st_dev, retained.st_ino) != (public.st_dev, public.st_ino)
            or retained.st_nlink != 1 or public.st_nlink != 1
            or not stat.S_ISREG(retained.st_mode)
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or retained.st_uid != os.getuid() or public.st_uid != os.getuid()
            or stat.S_IMODE(retained.st_mode) != 0o600
        ):
            die("create-only publication identity differs")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        retained = os.fstat(descriptor); public = path.lstat()
        if (
            (retained.st_dev, retained.st_ino) != (public.st_dev, public.st_ino)
            or retained.st_nlink != 1 or public.st_nlink != 1
            or not stat.S_ISREG(retained.st_mode)
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or retained.st_uid != os.getuid() or public.st_uid != os.getuid()
            or stat.S_IMODE(retained.st_mode) != mode
            or retained.st_size != len(payload)
        ):
            die("create-only terminal publication identity differs")
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_sealed(path: Path, core: dict[str, Any], mode: int = 0o444) -> tuple[str, str]:
    value = {**core, "receipt_digest": sha_bytes(canonical(core))}
    raw = canonical(value) + b"\n"
    write_create_only(path, raw, mode)
    return sha_bytes(raw), value["receipt_digest"]


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        316, -100, os.fsencode(source), -100, os.fsencode(destination), 1
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def snapshot_row(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor); leaf = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(leaf.st_mode)
            or stat.S_ISLNK(leaf.st_mode)
            or before.st_uid != os.getuid() or leaf.st_uid != os.getuid()
            or before.st_nlink != 1 or leaf.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or stat.S_IMODE(leaf.st_mode) != 0o444
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        ):
            die(f"immutable snapshot identity differs: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor); leaf_after = path.lstat()
        key = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_nlink, value.st_uid, stat.S_IMODE(value.st_mode),
        )
        if key(before) != key(after) or key(after) != key(leaf_after):
            die(f"immutable snapshot changed while reading: {path}")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            die(f"immutable snapshot size differs: {path}")
        return {
            "path": str(path if published_path is None else published_path),
            "sha256": sha_bytes(raw),
            "byte_size": int(after.st_size),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
        }
    finally:
        os.close(descriptor)


def rows_digest(rows: dict[str, dict[str, Any]]) -> str:
    named = [{"name": name, **rows[name]} for name in sorted(rows)]
    return sha_bytes(canonical(named))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--submission-receipt-sha256", required=True)
    parser.add_argument("--submission-receipt-digest", required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--release-manifest-digest", required=True)
    parser.add_argument("--postflight-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if re.fullmatch(r"[1-9][0-9]*", args.job_id) is None:
        die("job ID differs")
    for value in (
        WRAPPER_SHA256, TRAINER_SHA256, SUBMITTER_SHA256,
        args.postflight_sha256,
        FULL60_ADMISSION_SHA256, FULL60_ADMISSION_DIGEST,
        args.submission_receipt_sha256, args.submission_receipt_digest,
        args.release_manifest_sha256, args.release_manifest_digest,
    ):
        if SHA256.fullmatch(value) is None:
            die("source/evidence SHA pin remains unresolved")
    if (
        "__SOURCE_ANCHOR_" in FULL60_ADMISSION_SCHEMA
        or "__SOURCE_ANCHOR_" in FULL60_ADMISSION_STATUS
        or "__SOURCE_ANCHOR_" in str(FULL60_ADMISSION_SOURCE)
    ):
        die("formal full60 semantic pin remains unresolved")
    invoked = Path(__file__).resolve(strict=True)
    if invoked != POSTFLIGHT or hashlib.sha256(invoked.read_bytes()).hexdigest() != args.postflight_sha256:
        die("postflight must execute from exact sealed release")
    exact_directory(RELEASE, "release root", 0o555)
    exact_directory(INPUTS, "release inputs", 0o555)
    exact_directory(POSTFLIGHT_ROOT, "release postflight", 0o555)
    output_identity = exact_directory(OUTPUT_PARENT, "output parent", 0o700)
    exact_directory(LOG_DIR, "Slurm log root", 0o700)
    exact_directory(CHECKPOINT_PARENT, "checkpoint parent", 0o700)
    if TRAINING_ADMISSION.exists() or TRAINING_ADMISSION.is_symlink():
        die("training admission is not fresh")
    expected_paths = {
        "wrapper": (WRAPPER, WRAPPER_SHA256),
        "trainer": (TRAINER, TRAINER_SHA256),
        "submitter": (SUBMITTER, SUBMITTER_SHA256),
        "postflight": (POSTFLIGHT, args.postflight_sha256),
        "source_archive": (SOURCE_ARCHIVE, SOURCE_ARCHIVE_SHA256),
        "source_manifest": (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        "checkpoint_manifest": (CHECKPOINT_MANIFEST, CHECKPOINT_MANIFEST_SHA256),
        "formal_full60_admission": (FULL60_ADMISSION, FULL60_ADMISSION_SHA256),
        "release_manifest": (RELEASE_MANIFEST, args.release_manifest_sha256),
        "submission_receipt": (SUBMISSION_RECEIPT, args.submission_receipt_sha256),
    }
    descriptors: list[int] = []
    raws: dict[str, bytes] = {}
    evidence_fds: dict[str, tuple[int, Path]] = {}
    try:
        for label, (path, expected_sha) in expected_paths.items():
            descriptor, raw = retained_bytes(path, expected_sha, label)
            descriptors.append(descriptor); raws[label] = raw
            evidence_fds[label] = (descriptor, path)
        submission = decode_sealed(
            raws["submission_receipt"], args.submission_receipt_digest,
            "submission receipt",
        )
        release = decode_sealed(
            raws["release_manifest"], args.release_manifest_digest,
            "release manifest",
        )
        full60 = decode_sealed(
            raws["formal_full60_admission"], FULL60_ADMISSION_DIGEST,
            "formal full60 admission",
        )
        validate_full60_deep_admission(full60)
        job_output = OUTPUT_PARENT / f"job-{args.job_id}"
        checkpoint_release = CHECKPOINT_PARENT / f"{STEM}-job-{args.job_id}"
        expected_release_inputs = {
            "checkpoint_manifest": {
                "path": str(CHECKPOINT_MANIFEST),
                "sha256": CHECKPOINT_MANIFEST_SHA256,
            },
            "formal_full60_admission": {
                "path": str(FULL60_ADMISSION),
                "sha256": FULL60_ADMISSION_SHA256,
            },
            "postflight": {
                "path": str(POSTFLIGHT), "sha256": args.postflight_sha256,
            },
            "source_archive": {
                "path": str(SOURCE_ARCHIVE), "sha256": SOURCE_ARCHIVE_SHA256,
            },
            "source_manifest": {
                "path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA256,
            },
            "submitter": {
                "path": str(SUBMITTER), "sha256": SUBMITTER_SHA256,
            },
            "trainer": {"path": str(TRAINER), "sha256": TRAINER_SHA256},
            "wrapper": {"path": str(WRAPPER), "sha256": WRAPPER_SHA256},
        }
        expected_submission_inputs = {
            **expected_release_inputs,
            "release_manifest": {
                "path": str(RELEASE_MANIFEST),
                "sha256": args.release_manifest_sha256,
                "receipt_digest": args.release_manifest_digest,
            },
        }
        if (
            set(submission) != {
                "schema_version", "status", "submission_success",
                "single_sbatch_call", "job_success", "submitted_job",
                "request", "submission_boundary", "inputs", "outputs",
                "formal_full60_admission", "authority", "receipt_digest",
            }
            or submission.get("schema_version")
            != "saic-source-anchor-formal32-submission-v1"
            or submission.get("status") != "submitted"
            or submission.get("submission_success") is not True
            or submission.get("single_sbatch_call") is not True
            or submission.get("job_success") is not None
            or submission.get("submitted_job", {}).get("job_id") != args.job_id
            or set(submission.get("submitted_job", {})) != {
                "job_id", "cluster", "stdout_sha256", "stderr_sha256",
            }
            or SHA256.fullmatch(str(
                submission.get("submitted_job", {}).get("stdout_sha256", "")
            )) is None
            or SHA256.fullmatch(str(
                submission.get("submitted_job", {}).get("stderr_sha256", "")
            )) is None
            or submission.get("outputs", {}).get("job_output") != str(job_output)
            or submission.get("outputs", {}).get("conditional_checkpoint_release")
            != str(checkpoint_release)
            or submission.get("outputs") != {
                "output_parent": str(OUTPUT_PARENT),
                "job_output": str(job_output),
                "submission_receipt": str(SUBMISSION_RECEIPT),
                "slurm_stdout": str(
                    LOG_DIR / f"saic-anchor-f32-r1-{args.job_id}.out"
                ),
                "slurm_stderr": str(
                    LOG_DIR / f"saic-anchor-f32-r1-{args.job_id}.err"
                ),
                "conditional_checkpoint_release": str(checkpoint_release),
            }
            or submission.get("request") != {
                "job_name": "saic-anchor-f32-r1", "partition": "faculty",
                "qos": "bgqos", "nodes": 1, "ntasks": 1,
                "cpus_per_task": 32, "memory": "256G",
                "walltime": "24:00:00", "gpu_resource": "gpu:mi210:8",
                "world_size": 8, "topology": "DP2xSP4", "hold": False,
                "dependency": None,
            }
            or submission.get("inputs") != expected_submission_inputs
            or submission.get("formal_full60_admission") != {
                "sha256": FULL60_ADMISSION_SHA256,
                "receipt_digest": FULL60_ADMISSION_DIGEST,
                "schema_version": FULL60_ADMISSION_SCHEMA,
                "status": FULL60_ADMISSION_STATUS,
            }
            or submission.get("authority") != AUTHORITY_PRE
            or release.get("schema_version")
            != "saic-source-anchor-formal32-release-manifest-v1"
            or release.get("status") != "sealed_before_single_sbatch_submission"
            or release.get("stem") != STEM
            or release.get("release_root") != str(RELEASE)
            or release.get("inputs") != expected_release_inputs
            or set(release) != {
                "schema_version", "status", "stem", "release_root", "inputs",
                "runtime", "formal_contract", "formal_full60_admission",
                "outputs", "authority", "receipt_digest",
            }
            or release.get("runtime") != {
                "python": str(PYTHON), "python_sha256": PYTHON_SHA256,
                "python_version_stdout_sha256": (
                    "55ae85cf4bdb38743edbcd53ea68ff36511997ec6c21b1e83d8bebc939bf056b"
                ),
                "sacct": str(SACCT), "sacct_sha256": SACCT_SHA256,
                "bernini_root": str(BERNINI_ROOT),
                "bernini_commit": "2d2b4591ac053ec25c6371b01a5a6746679e5793",
                "veomni_root": str(VEOMNI_ROOT),
                "veomni_commit": "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
                "checkpoint": str(CHECKPOINT),
                "checkpoint_tree_sha256": (
                    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
                ),
            }
            or release.get("formal_contract") != {
                "world_size": 8, "topology": "DP2xSP4", "frame_count": 81,
                "optimizer_updates_per_dp_arm": 32,
                "learning_rate": 1e-5, "seed": 20260809,
                "max_grad_norm": 1.0, "gradient_accumulation_steps": 1,
                "wrong_source_margin": 0.01, "ranking_weight": 1.0,
                "active_exact40_indices": [35, 36, 37, 38, 39],
                "adapter_blocks": list(range(23, 30)), "adapter_rank": 8,
                "adapter_projections": ["attn1.to_q", "attn1.to_out.0"],
            }
            or release.get("formal_full60_admission") != {
                "source_path": str(FULL60_ADMISSION_SOURCE),
                "release_copy": str(FULL60_ADMISSION),
                "sha256": FULL60_ADMISSION_SHA256,
                "receipt_digest": FULL60_ADMISSION_DIGEST,
                "schema_version": FULL60_ADMISSION_SCHEMA,
                "status": FULL60_ADMISSION_STATUS,
                "slurm_terminal_verified": True,
                "formal_admission": True,
                "exact_deep_validator": (
                    "saic-source-anchor-full60-deep-validator-v1"
                ),
                "job_id": full60["job_id"],
                "candidate_count": 60,
                "attempt_count": 60,
                "same_seed_proof_count": 20,
                "allocated_gpu_count": 8,
                "master_receipt_sha256": full60["master_receipt"]["sha256"],
                "submission_receipt_sha256": full60[
                    "submission_receipt"
                ]["sha256"],
                "retained_fd_world8_admission_sha256": full60[
                    "world8_closure"
                ]["retained_fd_world8_admission_sha256"],
                "compute_bash_probe_admission_sha256": full60[
                    "world8_closure"
                ]["compute_bash_probe_admission_sha256"],
                "archive_member_manifest_sha256": full60[
                    "world8_closure"
                ]["archive_member_manifest_sha256"],
                "runtime_origin_manifest_sha256": full60[
                    "world8_closure"
                ]["runtime_origin_manifest_sha256"],
                "underlying_world8_closure_deep_validated": True,
            }
            or release.get("outputs") != {
                "output_parent": str(OUTPUT_PARENT), "log_dir": str(LOG_DIR),
                "checkpoint_parent": str(CHECKPOINT_PARENT),
            }
            or release.get("authority") != AUTHORITY_PRE
            or full60.get("schema_version") != FULL60_ADMISSION_SCHEMA
            or full60.get("status") != FULL60_ADMISSION_STATUS
            or full60.get("slurm_terminal_verified") is not True
            or full60.get("formal_admission") is not True
        ):
            die("submission/release/full60 admission linkage differs")
        boundary = submission.get("submission_boundary", {})
        if (
            set(boundary) != {
                "environment_replaced", "exact_export_names",
                "reservation_created_before_sbatch", "same_inode_retained",
                "wrapper_submitted_from_retained_fd", "wrapper_fd_number",
                "reservation_device", "reservation_inode",
                "output_parent_device", "output_parent_inode",
                "log_dir_device", "log_dir_inode", "success_mode",
            }
            or boundary.get("environment_replaced") is not True
            or boundary.get("exact_export_names") != EXPORT_NAMES
            or boundary.get("reservation_inode") != SUBMISSION_RECEIPT.lstat().st_ino
            or boundary.get("output_parent_inode") != output_identity[1]
            or boundary.get("reservation_created_before_sbatch") is not True
            or boundary.get("same_inode_retained") is not True
            or boundary.get("wrapper_submitted_from_retained_fd") is not True
            or boundary.get("success_mode") != "0444"
        ):
            die("submission exact-once boundary differs")
        sacct = observe_sacct(args.job_id, submission)
        stdout_path = LOG_DIR / f"saic-anchor-f32-r1-{args.job_id}.out"
        stderr_path = LOG_DIR / f"saic-anchor-f32-r1-{args.job_id}.err"
        if set(LOG_DIR.iterdir()) != {stdout_path, stderr_path}:
            die("Slurm log namespace differs")
        logs: dict[str, Any] = {}
        log_retained: dict[str, tuple[int, Path, bytes]] = {}
        for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            descriptor, raw = retained_bytes(
                path, None, f"Slurm {label} log", expected_mode=None
            )
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid():
                die(f"Slurm {label} log owner differs")
            log_retained[label] = (descriptor, path, raw)
            logs[label] = {
                "path": str(path), "sha256": sha_bytes(raw),
                "size": len(raw), "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "device": info.st_dev, "inode": info.st_ino,
            }
        expected_output_entries = {SUBMISSION_RECEIPT, job_output}
        if set(OUTPUT_PARENT.iterdir()) != expected_output_entries:
            die("terminal training output namespace differs")
        exact_directory(job_output, "private training result", 0o700)
        receipt_path = job_output / "receipt.json"
        history_path = job_output / "history.json"
        run_receipt_fd, receipt_raw = retained_bytes(
            receipt_path, None, "private training receipt", expected_mode=0o600
        )
        history_fd, history_raw = retained_bytes(
            history_path, None, "private training history", expected_mode=0o600
        )
        descriptors.extend((run_receipt_fd, history_fd))
        run = decode_sealed(receipt_raw, None, "training run receipt")
        try:
            history = json.loads(history_raw.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            die(f"training history encoding differs: {error}")
        try:
            history_canonical = canonical(history) + b"\n"
        except (TypeError, ValueError, UnicodeError) as error:
            die(f"training history is not canonical finite ASCII JSON: {error}")
        if history_raw != history_canonical:
            die("training history canonical bytes differ")
        contract = run.get("run_contract", {})
        runtime = run.get("runtime", {})
        manifest = run.get("manifest", {})
        archive_closure = run.get("source_archive_closure", {})
        adapter = run.get("adapter", {})
        gate = run.get("heldout_gate", {})
        artifacts = run.get("artifacts", {})
        history_unsigned = dict(history)
        history_digest = history_unsigned.pop("history_digest", None)
        history_rows = history.get("rows")
        expected_updates = list(range(32))
        gate_unsigned = dict(gate) if isinstance(gate, dict) else {}
        gate_digest = gate_unsigned.pop("digest", None)
        if (
            run.get("schema_version") != "bernini-saic-source-anchor-run-receipt-v2"
            or run.get("complete") is not True
            or contract != {
                "world_size": 8, "data_parallel_size": 2,
                "sequence_parallel_size": 4, "frame_count": 81,
                "optimizer_updates": 32, "mode": "formal",
                "all_train_rows_used_once_as_clean_endpoint": True,
                "active_sigma_indices": [35, 36, 37, 38, 39],
                "train_rows_per_dp_arm": 32, "heldout_rows_per_dp_arm": 8,
                "learning_rate": 1e-5, "max_grad_norm": 1.0,
                "wrong_source_margin": 0.01, "ranking_weight": 1.0,
                "seed": 20260809, "gradient_accumulation_steps": 1,
                "active_adapter_blocks": list(range(23, 30)),
                "adapter_rank": 8,
                "adapter_projections": ["attn1.to_q", "attn1.to_out.0"],
                "source_video_count": 80,
                "source_fd_map_sha256": manifest.get("source_fd_map_sha256"),
                "source_supervisor_pid": manifest.get("source_supervisor_pid"),
                "source_video_pathname_fallback_allowed": False,
                "source_video_retained_inode_rehashed_before_and_after_decode": True,
                "archive_member_manifest_sha256": (
                    "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
                ),
                "extracted_tree_manifest_sha256": archive_closure.get(
                    "extracted_tree_manifest_sha256"
                ),
                "runtime_origin_manifest_sha256": (
                    "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
                ),
            }
            or manifest.get("file_sha256") != SOURCE_MANIFEST_SHA256
            or manifest.get("manifest_digest") != SOURCE_MANIFEST_DIGEST
            or manifest.get("train_count") != 64 or manifest.get("holdout_count") != 16
            or manifest.get("source_video_count") != 80
            or SHA256.fullmatch(str(manifest.get("source_fd_map_sha256", ""))) is None
            or type(manifest.get("source_supervisor_pid")) is not int
            or manifest["source_supervisor_pid"] <= 1
            or manifest.get("all_source_videos_read_only_via_parent_retained_proc_fds")
            is not True
            or manifest.get(
                "each_encoded_source_rehashed_before_and_after_complete_decode"
            ) is not True
            or set(archive_closure) != {
                "archive_member_manifest_sha256",
                "extracted_tree_manifest_sha256", "archive_member_count",
                "archive_regular_file_count", "archive_directory_count",
                "extracted_tree_manifest_source",
                "archive_binding_receipt_digest",
                "runtime_origin_project_module_count",
                "runtime_origin_manifest_sha256",
                "runtime_origin_receipt_digest",
                "runtime_import_origins_all_from_extracted_archive",
            }
            or archive_closure.get("archive_member_manifest_sha256")
            != "1f3c8af23f5b4d416cea04476900c5d479ad3000338746e11f0e655b995b0fcc"
            or SHA256.fullmatch(str(
                archive_closure.get("extracted_tree_manifest_sha256", ""))) is None
            or archive_closure.get("archive_member_count") != 864
            or archive_closure.get("archive_regular_file_count") != 853
            or archive_closure.get("archive_directory_count") != 11
            or archive_closure.get("extracted_tree_manifest_source")
            != "actual_lstat_after_extraction"
            or SHA256.fullmatch(str(
                archive_closure.get("archive_binding_receipt_digest", ""))) is None
            or archive_closure.get("runtime_origin_project_module_count") != 14
            or archive_closure.get("runtime_origin_manifest_sha256")
            != "2e9360581b21b56e6998e1e5db8df98e4cc66acf95fbb7819baffd1161eb98ba"
            or SHA256.fullmatch(str(
                archive_closure.get("runtime_origin_receipt_digest", ""))) is None
            or archive_closure.get(
                "runtime_import_origins_all_from_extracted_archive"
            ) is not True
            or runtime.get("slurm_job_id") != args.job_id
            or runtime.get("submission_receipt_sha256") != args.submission_receipt_sha256
            or runtime.get("submission_receipt_digest") != args.submission_receipt_digest
            or runtime.get("release_manifest_sha256") != args.release_manifest_sha256
            or runtime.get("release_manifest_digest") != args.release_manifest_digest
            or runtime.get("python_executable") != str(PYTHON)
            or runtime.get("python_executable_sha256") != PYTHON_SHA256
            or run.get("trainer_source_sha256") != TRAINER_SHA256
            or run.get("method_source_archive_sha256") != SOURCE_ARCHIVE_SHA256
            or run.get("formal_full60_admission_sha256") != FULL60_ADMISSION_SHA256
            or run.get("formal_full60_admission_digest") != FULL60_ADMISSION_DIGEST
            or run.get("action_training") is not False
            or run.get("action_stage_authorized") is not False
            or run.get("semantic_action_authorized") is not False
            or run.get("decoded_rgb_identity_authorized") is not False
            or run.get("source_anchor_checkpoint_publication_authorized") is not False
            or run.get("stage_a_checkpoint_release_requires_external_terminal_postflight") is not True
            or set(history) != {
                "schema_version", "complete", "optimizer_updates",
                "update_indices", "rows", "history_digest",
            }
            or history.get("schema_version") != "bernini-saic-source-anchor-history-v2"
            or history.get("complete") is not True
            or history.get("optimizer_updates") != 32
            or history.get("update_indices") != expected_updates
            or not isinstance(history_digest, str)
            or SHA256.fullmatch(history_digest) is None
            or sha_bytes(canonical(history_unsigned)) != history_digest
            or not isinstance(history_rows, list) or len(history_rows) != 32
            or [
                row.get("update_index")
                for row in history_rows if isinstance(row, dict)
            ] != expected_updates
            or any(
                set(row) != {"update_index", "dp_records"}
                or not isinstance(row.get("dp_records"), list)
                or len(row["dp_records"]) != 2
                or any(
                    not isinstance(record, dict)
                    for record in row["dp_records"]
                )
                or [record.get("dp_arm") for record in row["dp_records"]]
                != [0, 1]
                or any(
                    record.get("update_index") != row.get("update_index")
                    for record in row["dp_records"]
                )
                for row in history_rows if isinstance(row, dict)
            )
            or not isinstance(gate_digest, str)
            or SHA256.fullmatch(gate_digest) is None
            or sha_bytes(canonical(gate_unsigned)) != gate_digest
            or gate.get("checkpoint_publication_allowed")
            is not gate.get("noncompensating_all_pass")
        ):
            die("formal32 scientific/runtime receipt contract differs")
        scientific_status = run.get("status")
        if scientific_status not in {
            "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO",
            "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE",
        }:
            die("scientific classification differs")
        expected_audit_marker = (
            "[saic-anchor-f32-r1] STRONG_AUDIT_OK classification="
            f"{scientific_status} external_terminal_postflight_required=true\n"
        ).encode("ascii")
        if (
            log_retained["stdout"][2].count(expected_audit_marker) != 1
            or log_retained["stdout"][2].count(b"STRONG_AUDIT_OK") != 1
            or b"STRONG_AUDIT_OK" in log_retained["stderr"][2]
        ):
            die("strong-audit terminal log marker differs")
        adapter_fd: int | None = None
        adapter_raw: bytes | None = None
        adapter_path = job_output / "adapter.safetensors"
        if scientific_status == "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE":
            declared_adapter_sha = adapter.get("safetensors_roundtrip", {}).get(
                "file_sha256"
            )
            if not isinstance(declared_adapter_sha, str) or SHA256.fullmatch(
                declared_adapter_sha
            ) is None:
                die("adapter receipt SHA is absent")
            adapter_fd, adapter_raw = retained_bytes(
                adapter_path, declared_adapter_sha, "private adapter candidate",
                expected_mode=0o600,
            )
            descriptors.append(adapter_fd)
        reread_retained(
            run_receipt_fd, receipt_path, receipt_raw, "training receipt",
            expected_mode=0o600,
        )
        reread_retained(
            history_fd, history_path, history_raw, "training history",
            expected_mode=0o600,
        )
        for label, (descriptor, path, raw) in log_retained.items():
            reread_retained(
                descriptor, path, raw, f"Slurm {label} log", expected_mode=None
            )
        if adapter_fd is not None and adapter_raw is not None:
            reread_retained(
                adapter_fd, adapter_path, adapter_raw, "adapter candidate",
                expected_mode=0o600,
            )
        for label, (descriptor, path) in evidence_fds.items():
            reread_retained(
                descriptor, path, raws[label], label, expected_mode=0o444
            )
        if observe_sacct(args.job_id, submission) != sacct:
            die("terminal Slurm accounting changed before publication")
        terminal_slurm = {
            "job_id": args.job_id,
            "terminal_state": sacct["parsed_row"]["State"],
            "exit_code": sacct["parsed_row"]["ExitCode"],
            "world_size": 8,
            "data_parallel_size": 2,
            "sequence_parallel_size": 4,
            "request_tres_exact": sacct["request_tres_exact"],
            "allocation_tres_exact": sacct["allocation_tres_exact"],
            "exact_submit_line_verified": sacct[
                "submit_line_exact_retained_fd"
            ],
            "terminal_log_closure_verified": True,
            "authoritative_sacct_observation": sacct,
            "logs": logs,
        }
        formal_output: dict[str, Any]
        bundle: dict[str, Any]
        publication: dict[str, Any]
        if scientific_status == "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO":
            if (
                set(job_output.iterdir()) != {receipt_path, history_path}
                or gate.get("noncompensating_all_pass") is not False
                or gate.get("checkpoint_publication_allowed") is not False
                or adapter.get("checkpoint_candidate_materialized") is not False
                or adapter.get("checkpoint_published") is not False
                or run.get("source_anchor_checkpoint_candidate_eligible") is not False
                or artifacts != {"history.json": sha_bytes(history_raw)}
                or checkpoint_release.exists() or checkpoint_release.is_symlink()
            ):
                die("scientific NO_GO artifact closure differs")
            terminal_status = "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"
            formal_output = {
                "path": None,
                "same_formal_output_namespace": False,
                "fresh_create_only_release": False,
                "directory_mode": None,
            }
            bundle = {
                "adapter": None,
                "training_receipt": None,
                "training_history": None,
                "source_manifest": None,
                "checkpoint_manifest": None,
                "checkpoint_release": None,
                "file_count": 0,
                "files_digest": rows_digest({}),
            }
            publication = {
                "stage_a_checkpoint_release": False,
                "postflight_created_after_terminal_accounting": True,
                "adapter_mode": None,
                "directory_mode": None,
                "all_bundle_files_plain_single_link_0444": False,
            }
            authority = FORMAL_NO_GO_AUTHORITY
        elif scientific_status == "FORMAL_GATE_PASS_CHECKPOINT_CANDIDATE":
            if (
                set(job_output.iterdir()) != {receipt_path, history_path, adapter_path}
                or gate.get("noncompensating_all_pass") is not True
                or gate.get("checkpoint_publication_allowed") is not True
                or adapter.get("checkpoint_candidate_materialized") is not True
                or adapter.get("checkpoint_published") is not False
                or run.get("source_anchor_checkpoint_candidate_eligible") is not True
                or checkpoint_release.exists() or checkpoint_release.is_symlink()
            ):
                die("scientific PASS candidate closure differs")
            if adapter_fd is None or adapter_raw is None:
                die("retained adapter candidate is absent")
            adapter_sha = sha_bytes(adapter_raw)
            if adapter.get("safetensors_roundtrip", {}).get("file_sha256") != adapter_sha:
                die("adapter receipt SHA differs")
            if artifacts != {
                "adapter.safetensors": adapter_sha,
                "history.json": sha_bytes(history_raw),
            }:
                die("PASS training artifact manifest differs")
            stage = CHECKPOINT_PARENT / f".{checkpoint_release.name}.staging"
            if stage.exists() or stage.is_symlink():
                die("checkpoint release staging is not fresh")
            stage.mkdir(mode=0o700)
            write_create_only(stage / "adapter.safetensors", adapter_raw, 0o444)
            write_create_only(
                stage / "training-receipt.json", receipt_raw, 0o444
            )
            write_create_only(
                stage / "training-history.json", history_raw, 0o444
            )
            write_create_only(
                stage / "source-manifest.json", raws["source_manifest"], 0o444
            )
            write_create_only(
                stage / "checkpoint-content-manifest.sha256",
                raws["checkpoint_manifest"], 0o444,
            )
            payload_paths = {
                "adapter": stage / "adapter.safetensors",
                "training_receipt": stage / "training-receipt.json",
                "training_history": stage / "training-history.json",
                "source_manifest": stage / "source-manifest.json",
                "checkpoint_manifest": stage / "checkpoint-content-manifest.sha256",
            }
            published_payload_paths = {
                name: checkpoint_release / path.name
                for name, path in payload_paths.items()
            }
            payload_artifacts = {
                name: snapshot_row(
                    path, published_path=published_payload_paths[name]
                )
                for name, path in payload_paths.items()
            }
            release_core = {
                "schema_version": "saic-source-anchor-formal32-checkpoint-release-v1",
                "status": "FORMAL_GATE_PASS_CHECKPOINT_RELEASED",
                "complete": True,
                "release_path": str(checkpoint_release),
                "formal_output_namespace": str(checkpoint_release),
                "job_id": args.job_id,
                "source_release_manifest_sha256": args.release_manifest_sha256,
                "submission_receipt_sha256": args.submission_receipt_sha256,
                "postflight_source_sha256": args.postflight_sha256,
                "trainer_source_sha256": TRAINER_SHA256,
                "history_digest": history_digest,
                "heldout_gate": gate,
                "authority": CHECKPOINT_RELEASE_AUTHORITY,
                "artifacts": payload_artifacts,
                "payload_files_digest": rows_digest(payload_artifacts),
            }
            checkpoint_manifest_sha, checkpoint_manifest_digest = write_sealed(
                stage / "checkpoint-release.json", release_core
            )
            for item in stage.iterdir():
                info = item.lstat()
                if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
                    die("staged checkpoint artifact is not immutable")
            os.chmod(stage, 0o555)
            rename_noreplace(stage, checkpoint_release)
            checkpoint_parent_fd = os.open(
                CHECKPOINT_PARENT,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                os.fsync(checkpoint_parent_fd)
            finally:
                os.close(checkpoint_parent_fd)
            exact_directory(checkpoint_release, "checkpoint release", 0o555)
            expected_bundle_names = {
                "adapter.safetensors",
                "training-receipt.json",
                "training-history.json",
                "source-manifest.json",
                "checkpoint-content-manifest.sha256",
                "checkpoint-release.json",
            }
            if (
                set(item.name for item in checkpoint_release.iterdir())
                != expected_bundle_names
                or any(
                    item.lstat().st_nlink != 1 or stat.S_IMODE(item.lstat().st_mode) != 0o444
                    for item in checkpoint_release.iterdir()
                )
            ):
                die("published checkpoint release closure differs")
            bundle_paths = {
                **published_payload_paths,
                "checkpoint_release": checkpoint_release / "checkpoint-release.json",
            }
            bundle_rows = {
                name: snapshot_row(path) for name, path in bundle_paths.items()
            }
            if any(
                bundle_rows[name] != payload_artifacts[name]
                for name in payload_artifacts
            ):
                die("published payload snapshot differs from retained staging")
            if (
                bundle_rows["checkpoint_release"]["sha256"]
                != checkpoint_manifest_sha
                or checkpoint_manifest_digest
                != sha_bytes(canonical(release_core))
            ):
                die("checkpoint release manifest seal differs after publication")
            formal_output = {
                "path": str(checkpoint_release),
                "same_formal_output_namespace": True,
                "fresh_create_only_release": True,
                "directory_mode": "0555",
            }
            bundle = {
                **bundle_rows,
                "file_count": 6,
                "files_digest": rows_digest(bundle_rows),
            }
            publication = {
                "stage_a_checkpoint_release": True,
                "postflight_created_after_terminal_accounting": True,
                "adapter_mode": "0444",
                "directory_mode": "0555",
                "all_bundle_files_plain_single_link_0444": True,
            }
            terminal_status = "FORMAL_GATE_PASS_CHECKPOINT_RELEASED"
            authority = FORMAL_POSTFLIGHT_AUTHORITY
        else:
            die("scientific classification differs")
        released_history_path = (
            history_path if terminal_status == "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"
            else checkpoint_release / "training-history.json"
        )
        core = {
            "schema_version": "saic-source-anchor-formal32-terminal-admission-v1",
            "status": terminal_status,
            "complete": True,
            "slurm": terminal_slurm,
            "formal_output": formal_output,
            "bundle": bundle,
            "history": {
                "optimizer_updates": 32,
                "all_updates_present_exactly_once": True,
                "path": str(released_history_path),
                "sha256": sha_bytes(history_raw),
                "history_digest": history_digest,
            },
            "heldout_gate": gate,
            "publication": publication,
            "authority": authority,
        }
        admission_sha, admission_digest = write_sealed(TRAINING_ADMISSION, core)
        if set(OUTPUT_PARENT.iterdir()) != {
            SUBMISSION_RECEIPT, job_output, TRAINING_ADMISSION,
        }:
            die("terminal admitted output namespace differs")
        print(json.dumps({
            "status": terminal_status, "job_id": args.job_id,
            "training_admission": str(TRAINING_ADMISSION),
            "training_admission_sha256": admission_sha,
            "training_admission_digest": admission_digest,
            "checkpoint_release": (
                None if terminal_status == "OPERATIONAL_COMPLETED_SCIENTIFIC_NO_GO"
                else str(checkpoint_release)
            ),
            "action_training_authorized": False,
        }, sort_keys=True))
        return 0
    finally:
        for descriptor in descriptors:
            try: os.close(descriptor)
            except OSError: pass


if __name__ == "__main__":
    raise SystemExit(main())
