#!/usr/bin/env python3
"""Exactly-once submit the sealed SOURCE-ANCHOR formal32 WORLD8 job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
SOURCE_MANIFEST_SHA256 = (
    "b56e9dca9085ba2f7b67acee04f29f7e45afd73a71b92172df5b19095fe33c4c"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
FULL60_ADMISSION_SHA256 = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SHA256__"
FULL60_ADMISSION_DIGEST = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_DIGEST__"
FULL60_ADMISSION_SCHEMA = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_SCHEMA__"
FULL60_ADMISSION_STATUS = "__SOURCE_ANCHOR_FORMAL_FULL60_ADMISSION_STATUS__"
AUTHORITY = {
    "stage_a_training_authorized": True,
    "stage_a_checkpoint_release_authorized": False,
    "action_training_authorized": False,
    "semantic_action_authorized": False,
    "decoded_rgb_identity_authorized": False,
    "publication_beyond_stage_a_checkpoint": False,
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
    raise SystemExit(f"submit-saic-source-anchor-formal32-v1: {message}")


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




def sha_fd(descriptor: int) -> str:
    result = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        result.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return result.hexdigest()


def retained_exact(path: Path, expected_sha: str, label: str) -> tuple[int, bytes]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} path differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    before = os.fstat(descriptor)
    leaf = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or not stat.S_ISREG(leaf.st_mode)
        or stat.S_ISLNK(leaf.st_mode)
        or before.st_nlink != 1
        or leaf.st_nlink != 1
        or before.st_uid != os.getuid()
        or leaf.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o444
        or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        or SHA256.fullmatch(expected_sha) is None
        or sha_fd(descriptor) != expected_sha
    ):
        os.close(descriptor)
        die(f"{label} identity/bytes differ")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    after = os.fstat(descriptor)
    leaf_after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_nlink, value.st_uid, stat.S_IMODE(value.st_mode),
    )
    if identity(before) != identity(after) or identity(after) != identity(leaf_after):
        os.close(descriptor)
        die(f"{label} changed during retained read")
    return descriptor, b"".join(chunks)


def reread_retained_exact(
    descriptor: int, path: Path, expected_raw: bytes, label: str
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
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_nlink, value.st_uid, stat.S_IMODE(value.st_mode),
    )
    if (
        b"".join(chunks) != expected_raw
        or identity(before) != identity(after)
        or identity(after) != identity(leaf)
        or not stat.S_ISREG(after.st_mode) or not stat.S_ISREG(leaf.st_mode)
        or stat.S_ISLNK(leaf.st_mode)
        or after.st_uid != os.getuid() or leaf.st_uid != os.getuid()
        or after.st_nlink != 1 or leaf.st_nlink != 1
        or stat.S_IMODE(after.st_mode) != 0o444
    ):
        die(f"{label} changed before the single sbatch boundary")


def exact_directory(path: Path, label: str, mode: int) -> tuple[int, int]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} path differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != mode
    ):
        die(f"{label} identity differs")
    return info.st_dev, info.st_ino


def decode_sealed(raw: bytes, expected_sha: str, expected_digest: str, label: str) -> dict[str, Any]:
    if sha_bytes(raw) != expected_sha:
        die(f"{label} SHA differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        die(f"{label} encoding differs: {error}")
    if not isinstance(value, dict):
        die(f"{label} is not one object")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        claimed != expected_digest
        or sha_bytes(canonical(unsigned)) != claimed
        or raw != canonical(value) + b"\n"
    ):
        die(f"{label} canonical seal differs")
    return value


def write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(descriptor, view)
        if count <= 0:
            die("submission receipt write stalled")
        view = view[count:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--release-manifest-digest", required=True)
    parser.add_argument("--postflight-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for value in (
        WRAPPER_SHA256, TRAINER_SHA256, args.postflight_sha256,
        FULL60_ADMISSION_SHA256, FULL60_ADMISSION_DIGEST,
    ):
        if SHA256.fullmatch(value) is None:
            die("source/formal full60 SHA pins remain unresolved")
    if (
        "__SOURCE_ANCHOR_" in FULL60_ADMISSION_SCHEMA
        or "__SOURCE_ANCHOR_" in FULL60_ADMISSION_STATUS
        or "__SOURCE_ANCHOR_" in str(FULL60_ADMISSION_SOURCE)
        or SHA256.fullmatch(args.release_manifest_sha256) is None
        or SHA256.fullmatch(args.release_manifest_digest) is None
    ):
        die("formal full60/release manifest pins remain unresolved")
    exact_directory(RELEASE, "release root", 0o555)
    exact_directory(INPUTS, "release inputs", 0o555)
    exact_directory(POSTFLIGHT_ROOT, "release postflight", 0o555)
    output_identity = exact_directory(OUTPUT_PARENT, "fresh output parent", 0o700)
    log_identity = exact_directory(LOG_DIR, "fresh Slurm log root", 0o700)
    exact_directory(CHECKPOINT_PARENT, "checkpoint parent", 0o700)
    if any(OUTPUT_PARENT.iterdir()) or any(LOG_DIR.iterdir()):
        die("fresh output/log namespace is not empty")

    invoked = Path(__file__).resolve(strict=True)
    if invoked != SUBMITTER:
        die("submitter must execute from sealed release")
    self_sha = hashlib.sha256(invoked.read_bytes()).hexdigest()
    expected_hashes = {
        "wrapper": WRAPPER_SHA256,
        "trainer": TRAINER_SHA256,
        "submitter": self_sha,
        "postflight": args.postflight_sha256,
        "source_archive": SOURCE_ARCHIVE_SHA256,
        "source_manifest": SOURCE_MANIFEST_SHA256,
        "checkpoint_manifest": CHECKPOINT_MANIFEST_SHA256,
        "formal_full60_admission": FULL60_ADMISSION_SHA256,
    }
    paths = {
        "wrapper": WRAPPER, "trainer": TRAINER, "submitter": SUBMITTER,
        "postflight": POSTFLIGHT, "source_archive": SOURCE_ARCHIVE,
        "source_manifest": SOURCE_MANIFEST,
        "checkpoint_manifest": CHECKPOINT_MANIFEST,
        "formal_full60_admission": FULL60_ADMISSION,
    }
    descriptors: list[int] = []
    raw_files: dict[str, bytes] = {}
    try:
        retained: dict[str, int] = {}
        for label, path in paths.items():
            descriptor, raw = retained_exact(path, expected_hashes[label], label)
            descriptors.append(descriptor); retained[label] = descriptor
            raw_files[label] = raw
        if sha_bytes(raw_files["submitter"]) != self_sha:
            die("invoked submitter changed during retained read")
        manifest_fd, manifest_raw = retained_exact(
            RELEASE_MANIFEST, args.release_manifest_sha256, "release manifest"
        )
        descriptors.append(manifest_fd)
        manifest = decode_sealed(
            manifest_raw, args.release_manifest_sha256,
            args.release_manifest_digest, "release manifest",
        )
        expected_inputs = {
            label: {"path": str(paths[label]), "sha256": expected_hashes[label]}
            for label in sorted(paths)
        }
        if (
            set(manifest) != {
                "schema_version", "status", "stem", "release_root", "inputs",
                "runtime", "formal_contract", "formal_full60_admission",
                "outputs", "authority", "receipt_digest",
            }
            or manifest.get("schema_version")
            != "saic-source-anchor-formal32-release-manifest-v1"
            or manifest.get("status") != "sealed_before_single_sbatch_submission"
            or manifest.get("stem") != STEM
            or manifest.get("release_root") != str(RELEASE)
            or manifest.get("inputs") != expected_inputs
            or manifest.get("runtime") != {
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
            or manifest.get("formal_contract") != {
                "world_size": 8, "topology": "DP2xSP4", "frame_count": 81,
                "optimizer_updates_per_dp_arm": 32,
                "learning_rate": 1e-5, "seed": 20260809,
                "max_grad_norm": 1.0, "gradient_accumulation_steps": 1,
                "wrong_source_margin": 0.01, "ranking_weight": 1.0,
                "active_exact40_indices": [35, 36, 37, 38, 39],
                "adapter_blocks": list(range(23, 30)), "adapter_rank": 8,
                "adapter_projections": ["attn1.to_q", "attn1.to_out.0"],
            }
            or manifest.get("outputs") != {
                "output_parent": str(OUTPUT_PARENT), "log_dir": str(LOG_DIR),
                "checkpoint_parent": str(CHECKPOINT_PARENT),
            }
            or manifest.get("authority") != AUTHORITY
        ):
            die("release manifest content differs")
        full60 = decode_sealed(
            raw_files["formal_full60_admission"], FULL60_ADMISSION_SHA256,
            FULL60_ADMISSION_DIGEST, "formal full60 admission",
        )
        validate_full60_deep_admission(full60)
        if (
            full60.get("schema_version") != FULL60_ADMISSION_SCHEMA
            or full60.get("status") != FULL60_ADMISSION_STATUS
            or full60.get("slurm_terminal_verified") is not True
            or full60.get("formal_admission") is not True
            or manifest.get("formal_full60_admission") != {
                "source_path": str(
                    FULL60_ADMISSION_SOURCE
                ),
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
            or manifest.get("formal_full60_admission", {}).get("sha256")
            != FULL60_ADMISSION_SHA256
            or manifest.get("formal_full60_admission", {}).get("receipt_digest")
            != FULL60_ADMISSION_DIGEST
        ):
            die("formal full60 admission binding differs")
        if (
            set(RELEASE.iterdir())
            != {INPUTS, POSTFLIGHT_ROOT, RELEASE_MANIFEST}
            or set(INPUTS.iterdir())
            != {path for label, path in paths.items() if label != "postflight"}
            or set(POSTFLIGHT_ROOT.iterdir()) != {POSTFLIGHT}
        ):
            die("sealed release namespace differs")

        wrapper_fd = retained["wrapper"]
        os.set_inheritable(wrapper_fd, True)
        receipt_fd = os.open(
            SUBMISSION_RECEIPT,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        descriptors.append(receipt_fd)
        receipt_identity = os.fstat(receipt_fd)
        provisional = {
            "schema_version": "saic-source-anchor-formal32-submission-v1",
            "status": "reserved_before_sbatch_not_submission_success",
            "submission_success": False,
            "single_sbatch_call": False,
            "job_success": None,
            "authority": AUTHORITY,
        }
        write_all(receipt_fd, canonical(provisional) + b"\n")
        os.fsync(receipt_fd); fsync_directory(OUTPUT_PARENT)
        public = SUBMISSION_RECEIPT.lstat()
        if (
            set(OUTPUT_PARENT.iterdir()) != {SUBMISSION_RECEIPT}
            or (public.st_dev, public.st_ino)
            != (receipt_identity.st_dev, receipt_identity.st_ino)
            or public.st_nlink != 1
            or stat.S_IMODE(public.st_mode) != 0o600
            or exact_directory(OUTPUT_PARENT, "output parent boundary", 0o700)
            != output_identity
            or exact_directory(LOG_DIR, "log root boundary", 0o700) != log_identity
        ):
            die("pre-sbatch reservation boundary differs")
        for label, path in paths.items():
            reread_retained_exact(
                retained[label], path, raw_files[label], label
            )
        reread_retained_exact(
            manifest_fd, RELEASE_MANIFEST, manifest_raw, "release manifest"
        )
        exports = {
            "SAIC_ANCHOR_ARCHIVE": str(SOURCE_ARCHIVE),
            "SAIC_ANCHOR_TRAINER": str(TRAINER),
            "SAIC_ANCHOR_SOURCE_MANIFEST": str(SOURCE_MANIFEST),
            "SAIC_ANCHOR_CHECKPOINT_MANIFEST": str(CHECKPOINT_MANIFEST),
            "SAIC_ANCHOR_FULL60_ADMISSION": str(FULL60_ADMISSION),
            "SAIC_ANCHOR_RELEASE_MANIFEST": str(RELEASE_MANIFEST),
            "SAIC_ANCHOR_SUBMISSION_RECEIPT": str(SUBMISSION_RECEIPT),
            "SAIC_ANCHOR_OUTPUT_PARENT": str(OUTPUT_PARENT),
            "SAIC_ANCHOR_RELEASE_MANIFEST_SHA256": args.release_manifest_sha256,
            "SAIC_ANCHOR_RELEASE_MANIFEST_DIGEST": args.release_manifest_digest,
            "SAIC_ANCHOR_WRAPPER_SHA256": WRAPPER_SHA256,
        }
        if list(exports) != EXPORT_NAMES or any(
            "," in name or "," in value or "\n" in name or "\n" in value
            for name, value in exports.items()
        ):
            die("exact export closure differs")
        command = [
            "/usr/bin/sbatch", "--parsable",
            f"--output={LOG_DIR}/saic-anchor-f32-r1-%j.out",
            f"--error={LOG_DIR}/saic-anchor-f32-r1-%j.err",
            "--export=NONE," + ",".join(
                f"{name}={exports[name]}" for name in EXPORT_NAMES
            ),
            f"/proc/self/fd/{wrapper_fd}",
        ]
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=120,
            pass_fds=(wrapper_fd,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        try:
            stdout = completed.stdout.decode("ascii")
        except UnicodeDecodeError:
            die("sbatch stdout is not ASCII")
        match = re.fullmatch(r"([1-9][0-9]*)(?:;([^\n;]+))?\n?", stdout)
        if completed.returncode != 0 or completed.stderr or match is None:
            die(
                "single sbatch failed; 0600 reservation retained: "
                f"exit={completed.returncode} stderr_sha256={sha_bytes(completed.stderr)}"
            )
        job_id = match.group(1)
        core = {
            "schema_version": "saic-source-anchor-formal32-submission-v1",
            "status": "submitted", "submission_success": True,
            "single_sbatch_call": True, "job_success": None,
            "submitted_job": {
                "job_id": job_id, "cluster": match.group(2),
                "stdout_sha256": sha_bytes(completed.stdout),
                "stderr_sha256": sha_bytes(completed.stderr),
            },
            "request": {
                "job_name": "saic-anchor-f32-r1", "partition": "faculty",
                "qos": "bgqos", "nodes": 1, "ntasks": 1,
                "cpus_per_task": 32, "memory": "256G",
                "walltime": "24:00:00", "gpu_resource": "gpu:mi210:8",
                "world_size": 8, "topology": "DP2xSP4", "hold": False,
                "dependency": None,
            },
            "submission_boundary": {
                "environment_replaced": True,
                "exact_export_names": EXPORT_NAMES,
                "reservation_created_before_sbatch": True,
                "same_inode_retained": True,
                "wrapper_submitted_from_retained_fd": True,
                "wrapper_fd_number": wrapper_fd,
                "reservation_device": receipt_identity.st_dev,
                "reservation_inode": receipt_identity.st_ino,
                "output_parent_device": output_identity[0],
                "output_parent_inode": output_identity[1],
                "log_dir_device": log_identity[0],
                "log_dir_inode": log_identity[1],
                "success_mode": "0444",
            },
            "inputs": {
                **{
                    label: {"path": str(paths[label]), "sha256": expected_hashes[label]}
                    for label in sorted(paths)
                },
                "release_manifest": {
                    "path": str(RELEASE_MANIFEST),
                    "sha256": args.release_manifest_sha256,
                    "receipt_digest": args.release_manifest_digest,
                },
            },
            "outputs": {
                "output_parent": str(OUTPUT_PARENT),
                "job_output": str(OUTPUT_PARENT / f"job-{job_id}"),
                "submission_receipt": str(SUBMISSION_RECEIPT),
                "slurm_stdout": str(LOG_DIR / f"saic-anchor-f32-r1-{job_id}.out"),
                "slurm_stderr": str(LOG_DIR / f"saic-anchor-f32-r1-{job_id}.err"),
                "conditional_checkpoint_release": str(
                    CHECKPOINT_PARENT / f"{STEM}-job-{job_id}"
                ),
            },
            "formal_full60_admission": {
                "sha256": FULL60_ADMISSION_SHA256,
                "receipt_digest": FULL60_ADMISSION_DIGEST,
                "schema_version": FULL60_ADMISSION_SCHEMA,
                "status": FULL60_ADMISSION_STATUS,
            },
            "authority": AUTHORITY,
        }
        value = {**core, "receipt_digest": sha_bytes(canonical(core))}
        payload = canonical(value) + b"\n"
        os.lseek(receipt_fd, 0, os.SEEK_SET); os.ftruncate(receipt_fd, 0)
        write_all(receipt_fd, payload); os.fsync(receipt_fd)
        os.lseek(receipt_fd, 0, os.SEEK_SET)
        if os.read(receipt_fd, len(payload) + 1) != payload:
            die("terminal submission receipt same-FD reread differs")
        public = SUBMISSION_RECEIPT.lstat()
        if (
            (public.st_dev, public.st_ino)
            != (receipt_identity.st_dev, receipt_identity.st_ino)
            or public.st_nlink != 1
            or stat.S_IMODE(public.st_mode) != 0o600
            or set(OUTPUT_PARENT.iterdir()) - {SUBMISSION_RECEIPT}
        ):
            die("terminal submission reservation identity differs")
        fsync_directory(OUTPUT_PARENT); os.fchmod(receipt_fd, 0o444)
        os.fsync(receipt_fd)
        terminal = os.fstat(receipt_fd); public = SUBMISSION_RECEIPT.lstat()
        if (
            (terminal.st_dev, terminal.st_ino)
            != (public.st_dev, public.st_ino)
            or terminal.st_nlink != 1 or public.st_nlink != 1
            or stat.S_IMODE(terminal.st_mode) != 0o444
            or terminal.st_size != len(payload)
        ):
            die("sealed submission receipt identity differs")
        fsync_directory(OUTPUT_PARENT)
        print(json.dumps({
            "job_id": job_id, "submission_receipt": str(SUBMISSION_RECEIPT),
            "submission_receipt_sha256": sha_bytes(payload),
            "submission_receipt_digest": value["receipt_digest"],
            "postflight": str(POSTFLIGHT),
        }, sort_keys=True), flush=True)
        os._exit(0)
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
