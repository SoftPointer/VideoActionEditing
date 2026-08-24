"""Strict read-only validator for the admitted AUH compute-Bash retained-FD probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PROBE_JOB_ID = "134647"
PROBE_ADMISSION = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
PROBE_ADMISSION_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
PROBE_ADMISSION_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
PROBE_SUBMISSION = PROBE_ADMISSION.parent / "submission-receipt.json"
PROBE_SUBMISSION_SHA256 = (
    "7e7fe47ac314a82c164de97805eb87dd6c7cb706bae43bccabf28e88b21a9af8"
)
PROBE_SUBMISSION_DIGEST = (
    "f74bd6521875b627ab033b194ea5ee59d83f0516c409dc4504cdeaf15ff7fa2a"
)
PROBE_EVIDENCE = PROBE_ADMISSION.parent / "job-134647/operational-evidence.json"
PROBE_EVIDENCE_SHA256 = (
    "5212a3df97fa1ed57865c757a32797b17df8846fcecdcf341fb65a050e328edb"
)
PROBE_EVIDENCE_DIGEST = (
    "5105f296044eb44dd33c40dd527960c9f75d086ea9042ef6fe77f4d4cc3c29fc"
)
# These are authenticated producer-namespace provenance.  Cross-check them
# across the sealed records below, but never against a consumer mount's live
# st_dev/st_ino.  Live identity is enforced separately inside _read_exact.
PROBE_OUTPUT_IDENTITY = "48:16765211873908070669"
PROBE_SUBMISSION_IDENTITY = "48:16250935117908932902"
PROBE_WRAPPER_SHA256 = (
    "8283e73ddf240d1ed8946f5682910bcfafaf24a88ae6c175c80b6a4597a75016"
)
PROBE_POSTFLIGHT_SHA256 = (
    "40c58b60afc6f20b569d15dd780a16e97722afdfefb8186444192c3dbf0868b9"
)
PROBE_MANIFEST_SHA256 = (
    "b2bc9a1f3869c8303346cee3decc0978fe65c1efec7c6135a0ea2846f5795385"
)
PROBE_MANIFEST_DIGEST = (
    "a9e54fe997fed1be7c3a4960fefd5fd176981d72b1ca6e351a0a5d2d5206a26e"
)
PROBE_STEM = "compute-bash-retained-fd-probe-8283e73d-r1"
PROBE_RELEASE = PROBE_ADMISSION.parent.parent.parent / "releases" / PROBE_STEM
PROBE_MANIFEST = PROBE_RELEASE / "release-manifest.json"
PROBE_WRAPPER = PROBE_RELEASE / "auh_probe_compute_bash_retained_fd_v1.sbatch"
PROBE_POSTFLIGHT = PROBE_RELEASE / "postflight_compute_bash_retained_fd_probe_v1.py"
PROBE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
PROBE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
PROBE_SACCT = Path("/usr/bin/sacct")
PROBE_SACCT_SHA256 = (
    "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
)
EXPECTED_COMPUTE_BASH = {
    "path": "/usr/bin/bash",
    "sha256": "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4",
    "version_stdout_sha256": (
        "51bd40ffa4710175920033d329a0e9e1667e6b7f56178e302432ff4610d554a7"
    ),
    "version_first_line": "GNU bash, version 5.1.16(1)-release (x86_64-pc-linux-gnu)",
    "brace_fd_redirection_supported": True,
    "retained_fd_survives_bash_script_handoff": True,
    "varredir_close_option_required": False,
}
ZERO_AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}
ADMISSION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "submission_receipt_sha256",
    "submission_receipt_digest", "submission_receipt_identity",
    "operational_evidence_sha256", "operational_evidence_digest",
    "release_manifest_file_sha256", "release_manifest_digest",
    "wrapper_sha256", "postflight_sha256", "compute_bash",
    "compute_bash_observation", "retained_fd", "job_success",
    "slurm_terminal_verified", "slurm", "authority", "receipt_digest",
}
SUBMISSION_FIELDS = {
    "schema_version", "status", "submission_success", "job_success",
    "submitted_job", "request", "submission_boundary", "inputs", "outputs",
    "authority", "receipt_digest",
}
EVIDENCE_FIELDS = {
    "schema_version", "status", "slurm_job_id", "job_success",
    "slurm_terminal_verified", "compute_bash", "retained_fd",
    "submission_binding", "wrapper_sha256", "scientific_generation_entered",
    "authority", "receipt_digest",
}
MANIFEST_FIELDS = {
    "schema_version", "status", "stem", "release_root", "output_parent",
    "wrapper", "postflight", "executables", "authority", "receipt_digest",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_exact(path: Path, *, expected_sha: str) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute() or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("probe path is not absolute")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        leaf = path.lstat()
        if (
            path.resolve(strict=True) != path
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o444
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or leaf.st_uid != os.getuid() or leaf.st_nlink != 1
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        ):
            raise ValueError("probe retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        leaf_after = path.lstat()
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        )
        if (
            identity(before) != identity(after)
            or identity(after) != identity(leaf_after)
            or not stat.S_ISREG(after.st_mode) or after.st_nlink != 1
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != 0o444
            or not stat.S_ISREG(leaf_after.st_mode) or stat.S_ISLNK(leaf_after.st_mode)
            or leaf_after.st_uid != os.getuid() or leaf_after.st_nlink != 1
            or stat.S_IMODE(leaf_after.st_mode) != 0o444
            or hashlib.sha256(raw).hexdigest() != expected_sha
        ):
            raise ValueError("probe retained bytes changed")
        return raw, after
    finally:
        os.close(descriptor)


def _decode(raw: bytes, *, schema: str, fields: set[str]) -> dict[str, Any]:
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("probe schema closure differs")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        value.get("schema_version") != schema
        or raw != canonical(value) + b"\n"
        or digest != hashlib.sha256(canonical(unsigned)).hexdigest()
    ):
        raise ValueError("probe seal differs")
    return value


def validate_probe_admission(
    path: Path = PROBE_ADMISSION,
    *, expected_sha: str = PROBE_ADMISSION_SHA256,
    expected_digest: str = PROBE_ADMISSION_DIGEST,
) -> dict[str, Any]:
    if path != PROBE_ADMISSION or expected_sha != PROBE_ADMISSION_SHA256:
        raise ValueError("probe external pin differs")
    if (expected_digest != PROBE_ADMISSION_DIGEST
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None):
        raise ValueError("probe external digest differs")
    admission_raw, admission_info = _read_exact(path, expected_sha=expected_sha)
    admission = _decode(
        admission_raw,
        schema="saic-compute-bash-retained-fd-probe-admission-v1",
        fields=ADMISSION_FIELDS,
    )
    submission_raw, submission_info = _read_exact(
        PROBE_SUBMISSION, expected_sha=PROBE_SUBMISSION_SHA256,
    )
    submission = _decode(
        submission_raw,
        schema="saic-compute-bash-retained-fd-probe-submission-v1",
        fields=SUBMISSION_FIELDS,
    )
    evidence_raw, _ = _read_exact(PROBE_EVIDENCE, expected_sha=PROBE_EVIDENCE_SHA256)
    evidence = _decode(
        evidence_raw,
        schema="saic-compute-bash-retained-fd-probe-evidence-v1",
        fields=EVIDENCE_FIELDS,
    )
    manifest_raw, manifest_info = _read_exact(
        PROBE_MANIFEST, expected_sha=PROBE_MANIFEST_SHA256,
    )
    manifest = _decode(
        manifest_raw,
        schema="saic-compute-bash-retained-fd-probe-release-v1",
        fields=MANIFEST_FIELDS,
    )
    _, wrapper_info = _read_exact(PROBE_WRAPPER, expected_sha=PROBE_WRAPPER_SHA256)
    _, postflight_info = _read_exact(
        PROBE_POSTFLIGHT, expected_sha=PROBE_POSTFLIGHT_SHA256,
    )
    release_info = PROBE_RELEASE.lstat()
    if (
        PROBE_RELEASE.resolve(strict=True) != PROBE_RELEASE
        or not stat.S_ISDIR(release_info.st_mode)
        or stat.S_ISLNK(release_info.st_mode)
        or stat.S_IMODE(release_info.st_mode) != 0o555
        or release_info.st_nlink != 2
        or release_info.st_uid != os.getuid()
        or any(
            info.st_uid != release_info.st_uid
            for info in (manifest_info, wrapper_info, postflight_info)
        )
        or {entry.name for entry in PROBE_RELEASE.iterdir()} != {
            PROBE_MANIFEST.name, PROBE_WRAPPER.name, PROBE_POSTFLIGHT.name,
        }
    ):
        raise ValueError("probe release closure differs")
    binding = evidence.get("submission_binding")
    retained = admission.get("retained_fd")
    slurm = admission.get("slurm")
    slurm_fields = {
        "alloc_tres", "elapsed", "end", "exact_single_row", "exact_submit_line",
        "exit_code", "node_list", "retained_wrapper_fd", "sacct_sha256", "start",
        "state", "stdout_sha256", "submit_line_sha256",
    }
    expected_slurm = {
        "alloc_tres": {
            "billing": "4", "cpu": "4", "gres/gpu": "1",
            "gres/gpu:mi210": "1", "mem": "8G", "node": "1",
        },
        "elapsed": "00:00:05",
        "end": "2026-08-12T10:22:15",
        "exact_single_row": True,
        "exact_submit_line": True,
        "exit_code": "0:0",
        "node_list": "auh7-1b-gpu-186",
        "retained_wrapper_fd": 3,
        "sacct_sha256": "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e",
        "start": "2026-08-12T10:22:10",
        "state": "COMPLETED",
        "stdout_sha256": "15e6f2db7489babc845f7f5b9f7adf9aae9c9f05b42e1c6ce8b6b31b3ad07776",
        "submit_line_sha256": "684ac0026be2cf06387299d43ccb572bfc141eb00ef18e8ce806634cd799b52c",
    }
    expected_evidence_bash = {
        **EXPECTED_COMPUTE_BASH,
        "identity": "regular file:64768:21103803:0:1:755:1396520",
        "version_stdout_size": 308,
    }
    fixture = PROBE_ADMISSION.parent / "fd-fixture-job-134647"
    expected_retained = {
        "child_fd_identity": "48:16573158477051681205:1:444:26683",
        "child_proc_fd_path": "/proc/465355/fd/10",
        "logical_leaf_replaced_after_open": True,
        "original_retained_identity": "48:16573158477051681205:1:444:26683",
        "original_retained_path": str(fixture / "payload.retained"),
        "parent_and_child_inode_equal": True,
        "parent_fd_identity": "48:16573158477051681205:1:444:26683",
        "retained_payload_sha256": PROBE_WRAPPER_SHA256,
        "wrong_decoy_exit_code": 97,
        "wrong_decoy_identity": "48:7876493021143422963:1:444:28",
        "wrong_decoy_inode_differs": True,
        "wrong_decoy_path": str(fixture / "payload.logical"),
        "wrong_decoy_sha256": (
            "00ccec8295ecdab8718c6ddc835e6e574576f488b0870e25f98cfad489049c19"
        ),
        "wrong_decoy_sha_differs": True,
    }
    expected_request = {
        "job_name": "saic-bash-fd-probe1", "partition": "faculty",
        "qos": "bgqos", "nodes": 1, "ntasks": 1, "cpus_per_task": 4,
        "memory": "8G", "walltime": "00:05:00",
        "gpu_resource_requested": "gpu:mi210:1", "hold": False,
        "dependency": None, "scientific_generation": False,
    }
    expected_exports = [
        "SAIC_BASH_FD_PROBE_WRAPPER", "SAIC_BASH_FD_PROBE_WRAPPER_SHA256",
        "SAIC_BASH_FD_PROBE_PYTHON", "SAIC_BASH_FD_PROBE_PYTHON_SHA256",
        "SAIC_BASH_FD_PROBE_OUTPUT_PARENT", "SAIC_BASH_FD_PROBE_OUTPUT_DEVICE",
        "SAIC_BASH_FD_PROBE_OUTPUT_INODE", "SAIC_BASH_FD_PROBE_SUBMISSION",
        "SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE", "SAIC_BASH_FD_PROBE_SUBMISSION_INODE",
        "SAIC_BASH_FD_PROBE_POSTFLIGHT", "SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256",
        "SAIC_BASH_FD_PROBE_RELEASE_MANIFEST",
        "SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256",
        "SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST",
    ]
    expected_boundary = {
        "environment_replaced": True, "exact_job_export_names": expected_exports,
        "export_all": False, "reservation_created_before_sbatch": True,
        "same_inode_retained": True, "launcher_submitted_from_retained_fd": True,
        "retained_wrapper_fd": 3, "reservation_device": 48,
        "reservation_inode": 16250935117908932902,
        "output_parent_device": 48, "output_parent_inode": 16765211873908070669,
        "success_mode": "0444",
    }
    if (
        PROBE_SUBMISSION_IDENTITY
        != f"{expected_boundary['reservation_device']}:{expected_boundary['reservation_inode']}"
        or PROBE_OUTPUT_IDENTITY
        != f"{expected_boundary['output_parent_device']}:{expected_boundary['output_parent_inode']}"
    ):
        raise ValueError("probe historical boundary identity differs")
    expected_inputs = {
        "wrapper": str(PROBE_WRAPPER),
        "wrapper_sha256": PROBE_WRAPPER_SHA256,
        "python": str(PROBE_PYTHON),
        "python_sha256": PROBE_PYTHON_SHA256,
        "postflight": str(PROBE_POSTFLIGHT),
        "postflight_sha256": PROBE_POSTFLIGHT_SHA256,
        "release_manifest": str(PROBE_MANIFEST),
        "release_manifest_file_sha256": PROBE_MANIFEST_SHA256,
        "release_manifest_digest": PROBE_MANIFEST_DIGEST,
    }
    expected_outputs = {
        "output_parent": str(PROBE_ADMISSION.parent),
        "job_output_root": str(PROBE_ADMISSION.parent / "job-134647"),
        "submission_receipt": str(PROBE_SUBMISSION),
    }
    expected_manifest = {
        "schema_version": "saic-compute-bash-retained-fd-probe-release-v1",
        "status": "sealed_before_submission",
        "stem": PROBE_STEM,
        "release_root": str(PROBE_RELEASE),
        "output_parent": str(PROBE_ADMISSION.parent),
        "wrapper": {
            "path": str(PROBE_WRAPPER), "sha256": PROBE_WRAPPER_SHA256,
        },
        "postflight": {
            "path": str(PROBE_POSTFLIGHT), "sha256": PROBE_POSTFLIGHT_SHA256,
        },
        "executables": {
            "python": str(PROBE_PYTHON), "python_sha256": PROBE_PYTHON_SHA256,
            "sacct": str(PROBE_SACCT), "sacct_sha256": PROBE_SACCT_SHA256,
        },
        "authority": ZERO_AUTHORITY,
        "receipt_digest": PROBE_MANIFEST_DIGEST,
    }
    expected_submitted_job = {
        "cluster": None,
        "job_id": PROBE_JOB_ID,
        "stderr_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "stdout_sha256": (
            "a51c00607fa88b140868d381c7075b07a1c39eecefa9b6a09b9e10fc4c9a11cc"
        ),
    }
    output_info = PROBE_ADMISSION.parent.lstat()
    if (
        admission.get("receipt_digest") != expected_digest
        or admission.get("status")
        != "terminal_completed_compute_bash_retained_fd_admitted"
        or admission.get("slurm_job_id") != PROBE_JOB_ID
        or admission.get("job_success") is not True
        or admission.get("slurm_terminal_verified") is not True
        or admission.get("authority") != ZERO_AUTHORITY
        or admission.get("compute_bash") != EXPECTED_COMPUTE_BASH
        or admission.get("compute_bash_observation") != {
            "identity": "regular file:64768:21103803:0:1:755:1396520",
            "version_stdout_size": 308,
        }
        or admission.get("wrapper_sha256") != PROBE_WRAPPER_SHA256
        or admission.get("postflight_sha256") != PROBE_POSTFLIGHT_SHA256
        or admission.get("release_manifest_file_sha256") != PROBE_MANIFEST_SHA256
        or admission.get("release_manifest_digest") != PROBE_MANIFEST_DIGEST
        or admission.get("submission_receipt_sha256") != PROBE_SUBMISSION_SHA256
        or admission.get("submission_receipt_digest") != PROBE_SUBMISSION_DIGEST
        or admission.get("submission_receipt_identity") != PROBE_SUBMISSION_IDENTITY
        or admission.get("operational_evidence_sha256") != PROBE_EVIDENCE_SHA256
        or admission.get("operational_evidence_digest") != PROBE_EVIDENCE_DIGEST
        or retained != expected_retained
        or not isinstance(slurm, dict) or set(slurm) != slurm_fields
        or slurm != expected_slurm
        or manifest != expected_manifest
        # AUH may remap st_dev across login/compute mount namespaces while
        # preserving the underlying inode number.  Bind these live leaves to
        # their authenticated producer identities by inode only; _read_exact
        # already enforces full (st_dev, st_ino) FD/leaf identity locally.
        or admission_info.st_ino != 4033719430929478085
        or submission_info.st_ino != 16250935117908932902
        or output_info.st_ino != 16765211873908070669
        or PROBE_ADMISSION.parent.resolve(strict=True) != PROBE_ADMISSION.parent
        or not stat.S_ISDIR(output_info.st_mode) or stat.S_ISLNK(output_info.st_mode)
        or output_info.st_uid != os.getuid()
        or stat.S_IMODE(output_info.st_mode) != 0o700
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or submission.get("receipt_digest") != PROBE_SUBMISSION_DIGEST
        or submission.get("submitted_job") != expected_submitted_job
        or submission.get("request") != expected_request
        or submission.get("submission_boundary") != expected_boundary
        or submission.get("inputs") != expected_inputs
        or submission.get("outputs") != expected_outputs
        or submission.get("authority") != ZERO_AUTHORITY
        or evidence.get("receipt_digest") != PROBE_EVIDENCE_DIGEST
        or evidence.get("status") != "job_completed_awaiting_external_slurm_admission"
        or evidence.get("slurm_job_id") != PROBE_JOB_ID
        or evidence.get("job_success") is not None
        or evidence.get("slurm_terminal_verified") is not False
        or evidence.get("scientific_generation_entered") is not False
        or evidence.get("authority") != ZERO_AUTHORITY
        or evidence.get("wrapper_sha256") != PROBE_WRAPPER_SHA256
        or evidence.get("compute_bash") != expected_evidence_bash
        or retained != evidence.get("retained_fd")
        or binding != {
            "submission_receipt_path": str(PROBE_SUBMISSION),
            "submission_receipt_file_sha256": PROBE_SUBMISSION_SHA256,
            "submission_receipt_digest": PROBE_SUBMISSION_DIGEST,
            "submission_receipt_identity": PROBE_SUBMISSION_IDENTITY,
            "output_parent_identity": PROBE_OUTPUT_IDENTITY,
            "release_manifest_file_sha256": PROBE_MANIFEST_SHA256,
            "release_manifest_digest": PROBE_MANIFEST_DIGEST,
            "postflight_sha256": PROBE_POSTFLIGHT_SHA256,
        }
    ):
        raise ValueError("probe deep admission linkage differs")
    if binding["output_parent_identity"] != PROBE_OUTPUT_IDENTITY:
        raise ValueError("probe output identity differs")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "receipt_digest": expected_digest,
        "schema_version": admission["schema_version"],
        "status": admission["status"],
        "slurm_job_id": PROBE_JOB_ID,
        "compute_bash": EXPECTED_COMPUTE_BASH,
        "submission_receipt_sha256": PROBE_SUBMISSION_SHA256,
        "submission_receipt_digest": PROBE_SUBMISSION_DIGEST,
        "submission_receipt_path": str(PROBE_SUBMISSION),
        "operational_evidence_sha256": PROBE_EVIDENCE_SHA256,
        "operational_evidence_digest": PROBE_EVIDENCE_DIGEST,
        "operational_evidence_path": str(PROBE_EVIDENCE),
        "release_manifest_path": str(PROBE_MANIFEST),
        "release_manifest_file_sha256": PROBE_MANIFEST_SHA256,
        "release_manifest_digest": PROBE_MANIFEST_DIGEST,
        "wrapper_sha256": PROBE_WRAPPER_SHA256,
        "postflight_sha256": PROBE_POSTFLIGHT_SHA256,
        "authority": ZERO_AUTHORITY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--digest", required=True)
    args = parser.parse_args()
    binding = validate_probe_admission(
        Path(args.path), expected_sha=args.sha256, expected_digest=args.digest,
    )
    os.write(1, canonical(binding) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
