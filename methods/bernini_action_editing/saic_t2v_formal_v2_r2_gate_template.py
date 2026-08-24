#!/usr/bin/env python3
"""R2 template: three-proof admission gate for the SAIC formal full60 job.

This source is deliberately not releasable while any ``__R2_*__`` pin remains.
The first gate retains Job 134393 as the exact-60 dynamic-rendezvous proof.  The
second proof is the separately admitted compute-Bash retained-FD probe.  The
third admits a fresh retained-FD WORLD8 transport canary only when its external
postflight receipt binds that exact probe and a new authoritative ``sacct``
observation is byte-pinned.  None of these proofs is scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Mapping, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CANARY_JOB_ID = "134393"
CANARY_RECEIPT_SHA256 = (
    "6927a2945fac87622beb167b96bc6e04b2d26d1bc0d957bfd130f379380cbc8d"
)
CANARY_RECEIPT_DIGEST = (
    "773bb9df35add9319d9dd8ff0d39c7402bdfbbd2a3f2c52a3551bb00c2a5e52b"
)
CANARY_SUBMISSION_SHA256 = (
    "1840f3be3d96573e341c153d498e8447eac9c7250657bdbcae3aa8b1da1246af"
)
CANARY_SUBMISSION_DIGEST = (
    "be0ac76378d7bd199c01f8e02fb080ab8db3013be66bcd978b62d18ffed38796"
)
CANARY_RECEIPT_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "saic-r6-dynamic-rendezvous-v2-be3f82d-6314a4d7-r1/job-134393/"
    "canary-receipt.json"
)
CANARY_SUBMISSION_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "saic-r6-dynamic-rendezvous-v2-be3f82d-6314a4d7-r1/"
    "submission-receipt.json"
)
RETAINED_FD_CANARY_JOB_ID = "__R2_RETAINED_FD_CANARY_JOB_ID__"
RETAINED_FD_CANARY_ADMISSION_PATH = "__R2_RETAINED_FD_CANARY_ADMISSION_PATH__"
RETAINED_FD_CANARY_ADMISSION_SHA256 = "__R2_RETAINED_FD_CANARY_ADMISSION_SHA256__"
RETAINED_FD_CANARY_ADMISSION_DIGEST = "__R2_RETAINED_FD_CANARY_ADMISSION_DIGEST__"
RETAINED_FD_CANARY_SACCT_STDOUT_SHA256 = (
    "__R2_RETAINED_FD_CANARY_SACCT_STDOUT_SHA256__"
)
RETAINED_FD_CANARY_NODELIST = "__R2_RETAINED_FD_CANARY_NODELIST__"
RETAINED_FD_CANARY_START = "__R2_RETAINED_FD_CANARY_START__"
RETAINED_FD_CANARY_END = "__R2_RETAINED_FD_CANARY_END__"
RETAINED_FD_CANARY_ELAPSED = "__R2_RETAINED_FD_CANARY_ELAPSED__"
RETAINED_FD_CANARY_SUBMIT_LINE = "__R2_RETAINED_FD_CANARY_SUBMIT_LINE__"
COMPUTE_BASH_PROBE_JOB_ID = "134647"
COMPUTE_BASH_PROBE_ADMISSION_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
COMPUTE_BASH_PROBE_ADMISSION_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
COMPUTE_BASH_PROBE_ADMISSION_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
PROBE_VALIDATOR_SHA256 = (
    "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
)
COMPUTE_BASH_PATH = "/usr/bin/bash"
COMPUTE_BASH_SHA256 = "__R2_COMPUTE_BASH_SHA256__"
COMPUTE_BASH_VERSION_STDOUT_SHA256 = "__R2_COMPUTE_BASH_VERSION_STDOUT_SHA256__"
COMPUTE_BASH_VERSION_FIRST_LINE = "__R2_COMPUTE_BASH_VERSION_FIRST_LINE__"
GUARD_V2_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
CANARY_LAUNCHER_SHA256 = (
    "6314a4d7f9fd99ab9713f6d956f6fd0f6571511aff43e2d0dd9b80f7cc439cea"
)
CANARY_ARCHIVE_SHA256 = (
    "d7dad8b3af1bd06a6bb0bb5ddfa66607302bc98c4acf95d8cf065efba86ae7c6"
)
RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
ARCHIVED_GUARD_V1_SHA256 = (
    "6666d3bd822baeb5d6f3ecb8033c71510d80e68c4059dfbbd09ef4dc4d100a9f"
)
BASE_V1_SPEC_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
CONTRACT_RUNTIME_SHA256 = (
    "508dde8d995dcc8deeccb47b35be71b9915a86964626383660d8eed952ef5278"
)
STATIC_FFMPEG_PATH = (
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
STATIC_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
STATIC_FFMPEG_VERSION_STDOUT_SHA256 = (
    "389368da4bcd4e22d7bf9134f3a8c24dd36027de7d963015230969a87c9e3339"
)
STATIC_FFMPEG_VERSION_FIRST_LINE = (
    "ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  "
    "Copyright (c) 2000-2024 the FFmpeg developers"
)
FORMAL_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
FORMAL_RELEASE_ROOT = (
    FORMAL_ROOT
    + "/releases/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
FORMAL_OUTPUT_ROOT = (
    FORMAL_ROOT
    + "/runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
FORMAL_SUBMISSION_RECEIPT = FORMAL_OUTPUT_ROOT + ".submission.receipt.json"
FORMAL_SLURM_LOG_DIR = (
    FORMAL_ROOT + "/slurm/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
FORMAL_BASE_LAUNCHER = (
    FORMAL_RELEASE_ROOT
    + "/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.base.sbatch"
)
FORMAL_MATERIALIZER = (
    FORMAL_RELEASE_ROOT
    + "/inputs/materialize_saic_t2v_topup_formal_v2_launcher.py"
)
FORMAL_EFFECTIVE_LAUNCHER = (
    FORMAL_RELEASE_ROOT
    + "/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_formal_v2_effective.sbatch"
)
FORMAL_WRAPPER = (
    FORMAL_RELEASE_ROOT
    + "/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v5.sbatch"
)
FORMAL_GATE = FORMAL_RELEASE_ROOT + "/inputs/saic_t2v_formal_v2_r2_gate.py"
FORMAL_GUARD_V2 = FORMAL_RELEASE_ROOT + "/inputs/saic_t2v_rendezvous_guard_v2.py"
FORMAL_PROBE_VALIDATOR = (
    FORMAL_RELEASE_ROOT + "/inputs/probe_admission_binding_v1.py"
)
FORMAL_SOURCE_ARCHIVE = (
    FORMAL_RELEASE_ROOT + "/inputs/videoedit-saic-20c2193-methods.tar"
)
FORMAL_SOURCE_MANIFEST = (
    FORMAL_RELEASE_ROOT + "/inputs/saic_reversible_source_set_v1.json"
)
FORMAL_EVENT_SPEC = (
    FORMAL_RELEASE_ROOT + "/inputs/saic_pure_t2v_event_bank_topup_v2.json"
)
FORMAL_CHECKPOINT_MANIFEST = (
    FORMAL_RELEASE_ROOT + "/inputs/bernini-r13-ff4c5d4-checkpoint.sha256"
)
FORMAL_BASE_LAUNCHER_SHA256 = (
    "12c1b2baaecfd479f65f9b5dbf0dbae17cd87196767e93c254fe2cffc895f29d"
)
FORMAL_MATERIALIZER_SHA256 = (
    "5585abb927206d0813caca4cec8dc10b846fe0ace704538bb12e0ad5cffe8b97"
)
FORMAL_EFFECTIVE_LAUNCHER_SHA256 = (
    "4227b4a00b7b2dea786457baad56b4fdcb4b476929e9619cb533a353b369f9f0"
)
FORMAL_SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
FORMAL_SOURCE_REVISION = "20c2193954e780e9654347754b1485f3492fbea5"
FORMAL_SOURCE_MANIFEST_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
FORMAL_EVENT_SPEC_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
FORMAL_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
FORMAL_PYTHON = "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
FORMAL_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
FORMAL_BERNINI_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
)
FORMAL_VEOMNI_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6"
)
FORMAL_CHECKPOINT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
THREAT_MODEL = {
    "pathname_replacement_rename_symlink_leaf_swap_resistance": True,
    "retained_fd_admission_roots": [
        "formal_gate", "effective_launcher", "rendezvous_guard_v2",
        "compute_bash_probe_validator",
    ],
    "shared_science_paths_assumed_not_concurrently_replaced_by_same_uid": [
        "source_archive", "source_manifest", "event_spec",
        "checkpoint_manifest", "python", "static_ffmpeg",
        "bernini_root", "veomni_root", "checkpoint",
    ],
    "same_inode_in_place_mutation_resistance_claimed": False,
    "sealed_release_permissions_are_not_claimed_as_same_uid_immutability": True,
    "three_independent_operational_proof_objects_required": True,
    "exact60_lifecycle_probe_world8_transport_non_substitutability": True,
    "compute_bash_exact_identity_pinned": True,
}
SACCT_SHA256 = (
    "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
)
SACCT_STDOUT_SHA256 = (
    "bef5b0af2ba6fdc4ebffeceb0980f6a6a823d474c82d18add3f97ae4d23922ce"
)
EMPTY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
SACCT_FIELDS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "ElapsedRaw", "Start", "End",
]
SACCT_ARGV = [
    "/usr/bin/sacct", "-j", CANARY_JOB_ID, "-X", "--noheader", "-n",
    "-P", "-o", ",".join(SACCT_FIELDS),
]
SACCT_PARSED_ROW = {
    "JobIDRaw": CANARY_JOB_ID,
    "State": "COMPLETED",
    "ExitCode": "0:0",
    "AllocTRES": (
        "billing=32,cpu=32,gres/gpu:mi210=8,gres/gpu=8,mem=64G,node=1"
    ),
    "NodeList": "auh7-1b-gpu-270",
    "ElapsedRaw": "278",
    "Start": "2026-08-11T23:34:52",
    "End": "2026-08-11T23:39:30",
}
RETAINED_FD_CANARY_SACCT_FIELDS = [
    "JobIDRaw", "State", "ExitCode", "AllocTRES%512", "NodeList",
    "Start", "End", "Elapsed", "SubmitLine%8192",
]
RETAINED_FD_CANARY_SACCT_ARGV = [
    "/usr/bin/sacct", "-j", RETAINED_FD_CANARY_JOB_ID, "-X",
    "--noheader", "-n", "-P", "-o",
    ",".join(RETAINED_FD_CANARY_SACCT_FIELDS),
]
RETAINED_FD_CANARY_SACCT_PARSED_ROW = {
    "JobIDRaw": RETAINED_FD_CANARY_JOB_ID,
    "State": "COMPLETED",
    "ExitCode": "0:0",
    "AllocTRES": (
        "billing=16,cpu=16,gres/gpu:mi210=8,gres/gpu=8,mem=32G,node=1"
    ),
    "NodeList": RETAINED_FD_CANARY_NODELIST,
    "Start": RETAINED_FD_CANARY_START,
    "End": RETAINED_FD_CANARY_END,
    "Elapsed": RETAINED_FD_CANARY_ELAPSED,
    "SubmitLine": RETAINED_FD_CANARY_SUBMIT_LINE,
}

EXPORT_NAMES = [
    "SAIC_T2V_FV2_BASE_LAUNCHER",
    "SAIC_T2V_FV2_BASE_LAUNCHER_SHA256",
    "SAIC_T2V_FV2_MATERIALIZER",
    "SAIC_T2V_FV2_MATERIALIZER_SHA256",
    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER",
    "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256",
    "SAIC_T2V_FV2_WRAPPER",
    "SAIC_T2V_FV2_WRAPPER_SHA256",
    "SAIC_T2V_FV2_GATE",
    "SAIC_T2V_FV2_GATE_SHA256",
    "SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD",
    "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256",
    "SAIC_T2V_FV2_CANARY_RECEIPT",
    "SAIC_T2V_FV2_CANARY_SUBMISSION_RECEIPT",
    "SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION",
    "SAIC_T2V_FV2_PROBE_VALIDATOR",
    "SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST",
    "SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT",
    "SAIC_T2V_V3_SOURCE_ARCHIVE",
    "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256",
    "SAIC_T2V_V3_SOURCE_REVISION",
    "SAIC_T2V_V3_SOURCE_MANIFEST",
    "SAIC_T2V_V3_SOURCE_MANIFEST_SHA256",
    "SAIC_T2V_V3_EVENT_SPEC",
    "SAIC_T2V_V3_EVENT_SPEC_SHA256",
    "BERNINI_OFFICIAL_ROOT",
    "BERNINI_VEOMNI_ROOT",
    "BERNINI_ACTION_CHECKPOINT",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
    "SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256",
    "SAIC_T2V_V3_OUTPUT_ROOT",
    "SAIC_T2V_V3_PYTHON_BIN",
    "SAIC_T2V_FV2_PYTHON_SHA256",
    "SAIC_T2V_V3_STATIC_FFMPEG",
    "SAIC_T2V_FV2_STATIC_FFMPEG_SHA256",
    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256",
    "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE",
    "SAIC_T2V_FV2_COMPUTE_BASH",
    "SAIC_T2V_FV2_COMPUTE_BASH_SHA256",
    "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256",
    "SAIC_T2V_FV2_SLURM_LOG_DIR",
]


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_plain(path: Path, expected_sha: str, label: str) -> Path:
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or SHA256.fullmatch(expected_sha) is None
    ):
        raise RuntimeError(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or sha_file(path) != expected_sha
    ):
        raise RuntimeError(f"{label} bytes differ")
    return path


def load_guard(path: Path, expected_sha: str):
    retained = re.fullmatch(r"/proc/[0-9]+/fd/[0-9]+", str(path)) is not None
    if retained:
        info = path.stat()
        payload = path.read_bytes()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or sha_bytes(payload) != expected_sha
        ):
            raise RuntimeError("formal retained guard bytes differ")
        module = types.ModuleType("formal_guard_v2")
        module.__file__ = str(path)
        exec(compile(payload, FORMAL_GUARD_V2, "exec"), module.__dict__)
    else:
        exact_plain(path, expected_sha, "formal external guard v2")
        spec = importlib.util.spec_from_file_location("formal_guard_v2", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("formal guard import differs")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if (
        module.EXPECTED_RUNTIME_SHA256 != RUNTIME_SHA256
        or module.CLAIM_SCHEMA_VERSION
        != "saic-t2v-topup-rendezvous-port-claim-v2"
    ):
        raise RuntimeError("formal guard contract differs")
    return module


def load_probe_validator(path: Path, expected_sha: str):
    if expected_sha != PROBE_VALIDATOR_SHA256:
        raise RuntimeError("formal probe-validator SHA pin differs")
    retained = re.fullmatch(r"/proc/[0-9]+/fd/[0-9]+", str(path)) is not None
    if retained:
        info = path.stat()
        payload = path.read_bytes()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or sha_bytes(payload) != expected_sha
        ):
            raise RuntimeError("formal retained probe-validator bytes differ")
    else:
        exact_plain(path, expected_sha, "formal probe-admission validator")
        payload = path.read_bytes()
    module = types.ModuleType("formal_probe_admission_binding_v1")
    module.__file__ = str(path)
    exec(compile(payload, FORMAL_PROBE_VALIDATOR, "exec"), module.__dict__)
    if (
        module.PROBE_JOB_ID != COMPUTE_BASH_PROBE_JOB_ID
        or module.PROBE_ADMISSION != Path(COMPUTE_BASH_PROBE_ADMISSION_PATH)
        or module.PROBE_ADMISSION_SHA256
        != COMPUTE_BASH_PROBE_ADMISSION_SHA256
        or module.PROBE_ADMISSION_DIGEST
        != COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        or module.EXPECTED_COMPUTE_BASH != expected_compute_bash_identity()
    ):
        raise RuntimeError("formal probe-validator contract differs")
    return module


def validate_compute_bash_probe_admission(
    validator, admission_path: Path,
) -> dict[str, Any]:
    ensure_r2_release_pins_resolved()
    if admission_path != Path(COMPUTE_BASH_PROBE_ADMISSION_PATH):
        raise RuntimeError("compute-Bash probe admission path differs")
    binding = validator.validate_probe_admission(
        admission_path,
        expected_sha=COMPUTE_BASH_PROBE_ADMISSION_SHA256,
        expected_digest=COMPUTE_BASH_PROBE_ADMISSION_DIGEST,
    )
    if (
        not isinstance(binding, dict)
        or set(binding) != PROBE_ADMISSION_BINDING_FIELDS
        or binding.get("path") != COMPUTE_BASH_PROBE_ADMISSION_PATH
        or binding.get("sha256") != COMPUTE_BASH_PROBE_ADMISSION_SHA256
        or binding.get("receipt_digest")
        != COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        or binding.get("schema_version")
        != "saic-compute-bash-retained-fd-probe-admission-v1"
        or binding.get("status")
        != "terminal_completed_compute_bash_retained_fd_admitted"
        or binding.get("slurm_job_id") != COMPUTE_BASH_PROBE_JOB_ID
        or binding.get("compute_bash") != expected_compute_bash_identity()
        or binding.get("authority") != {
            "scientific": False,
            "generation": False,
            "training": False,
            "publication": False,
            "formal_job_authorized": False,
        }
        or any(
            SHA256.fullmatch(str(binding.get(field))) is None
            for field in (
                "submission_receipt_sha256",
                "operational_evidence_sha256",
                "release_manifest_file_sha256",
                "wrapper_sha256",
                "postflight_sha256",
            )
        )
        or any(
            not isinstance(binding.get(field), str)
            or not Path(binding[field]).is_absolute()
            for field in (
                "submission_receipt_path", "operational_evidence_path",
                "release_manifest_path",
            )
        )
    ):
        raise RuntimeError("compute-Bash probe admission binding differs")
    return binding


def decode_fixed(
    module, path: Path, *, raw_sha: str, schema: str, label: str
) -> tuple[bytes, dict[str, Any]]:
    raw = module.wait_ready_bytes(path, label=label)
    if sha_bytes(raw) != raw_sha:
        raise RuntimeError(f"{label} raw SHA differs")
    value = module._decode_sealed(raw, schema_version=schema)
    return raw, value


def validate_canary(module, terminal_path: Path, submission_path: Path) -> dict:
    terminal_raw, terminal = decode_fixed(
        module,
        terminal_path,
        raw_sha=CANARY_RECEIPT_SHA256,
        schema="saic-r6-dynamic-rendezvous-all8-canary-v2",
        label="pinned successful rendezvous canary terminal receipt",
    )
    submission_raw, submission = decode_fixed(
        module,
        submission_path,
        raw_sha=CANARY_SUBMISSION_SHA256,
        schema="saic-r6-rendezvous-canary-submission-v2",
        label="pinned rendezvous canary submission receipt",
    )
    rows = terminal.get("candidate_rows")
    if (
        terminal.get("receipt_digest") != CANARY_RECEIPT_DIGEST
        or terminal.get("status") != "operational_dynamic_rendezvous_exact60_pass"
        or terminal.get("slurm_job_id") != CANARY_JOB_ID
        or terminal.get("topology") != "two_concurrent_world4_groups"
        or terminal.get("requested_gpu_count") != 8
        or terminal.get("candidate_count") != 60
        or terminal.get("completion_receipt_count") != 60
        or terminal.get("rank_packet_count") != 240
        or terminal.get("group_candidate_counts") != {"sp4-a": 30, "sp4-b": 30}
        or terminal.get("unique_actual_master_port_count") != 60
        or terminal.get("permanent_claim_count") != 60
        or terminal.get("collision_receipt_count") != 2
        or terminal.get("all_launch_rdzv_id_count") != 62
        or terminal.get("all_launch_rdzv_ids_unique") is not True
        or terminal.get("torch_disable_share_rdzv_tcp_store") != "0"
        or terminal.get("shared_tcp_store_bootstrap") is not True
        or terminal.get("legacy_listener_count") != 60
        or terminal.get("legacy_port_range_inclusive") != [48730, 48789]
        or terminal.get("legacy_listener_held_through_all_torchruns") is not True
        or terminal.get("legacy_listener_clean_shutdown_before_receipt") is not True
        or terminal.get("actual_ports_disjoint_from_legacy_ports") is not True
        or terminal.get("eaddrinuse_observed") is not False
        or terminal.get("guard_sha256") != GUARD_V2_SHA256
        or terminal.get("source_archive_sha256") != CANARY_ARCHIVE_SHA256
        or terminal.get("runtime_sha256") != RUNTIME_SHA256
        or terminal.get("launcher_sha256") != CANARY_LAUNCHER_SHA256
        or terminal.get("executed_launcher_sha256") != CANARY_LAUNCHER_SHA256
        or terminal.get("submission_receipt_path") != str(submission_path)
        or terminal.get("submission_receipt_sha256") != CANARY_SUBMISSION_SHA256
        or terminal.get("submission_receipt_digest") != CANARY_SUBMISSION_DIGEST
        or terminal.get("submission_job_id") != CANARY_JOB_ID
        or terminal.get("scientific_generation_entered") is not False
        or terminal.get("scientific_output_created") is not False
        or terminal.get("authority") != module.AUTHORITY
        or not isinstance(rows, list)
        or len(rows) != 60
    ):
        raise RuntimeError("pinned canary terminal content differs")
    row_fields = {
        "group_id", "candidate_index", "candidate_id",
        "successful_launch_ordinal", "rdzv_id", "actual_master_port",
        "claim_receipt_digest", "admission_receipt_digest",
        "completion_receipt_digest",
    }
    if (
        any(not isinstance(row, dict) or set(row) != row_fields for row in rows)
        or {(row["group_id"], row["candidate_index"]) for row in rows}
        != {(group, index) for group in ("sp4-a", "sp4-b") for index in range(30)}
        or len({row["candidate_id"] for row in rows}) != 60
        or len({row["rdzv_id"] for row in rows}) != 60
        or len({row["actual_master_port"] for row in rows}) != 60
        or any(
            not isinstance(row["successful_launch_ordinal"], int)
            or not 1 <= row["successful_launch_ordinal"] <= 16
            or not isinstance(row["actual_master_port"], int)
            or not 1024 <= row["actual_master_port"] <= 65535
            or any(
                SHA256.fullmatch(str(row[field])) is None
                for field in (
                    "claim_receipt_digest", "admission_receipt_digest",
                    "completion_receipt_digest",
                )
            )
            for row in rows
        )
    ):
        raise RuntimeError("pinned canary candidate rows differ")
    submitted = submission.get("submitted_job")
    request = submission.get("request")
    if (
        submission.get("receipt_digest") != CANARY_SUBMISSION_DIGEST
        or submission.get("status") != "submitted"
        or submission.get("submission_success") is not True
        or submission.get("job_success") is not None
        or not isinstance(submitted, dict)
        or submitted.get("job_id") != CANARY_JOB_ID
        or not isinstance(request, dict)
        or request.get("gpu_resource_requested") != "gpu:mi210:8"
        or request.get("world_topology") != "two_concurrent_world4"
        or request.get("candidate_count") != 60
        or request.get("hold") is not False
        or request.get("dependency") is not None
        or request.get("scientific_generation") is not False
        or submission.get("inputs", {}).get("guard_sha256") != GUARD_V2_SHA256
        or submission.get("inputs", {}).get("launcher_sha256")
        != CANARY_LAUNCHER_SHA256
        or submission.get("inputs", {}).get("source_archive_sha256")
        != CANARY_ARCHIVE_SHA256
        or submission.get("inputs", {}).get("runtime_sha256") != RUNTIME_SHA256
        or submission.get("outputs", {}).get("job_output_root")
        != str(terminal_path.parent)
        or submission.get("outputs", {}).get("submission_receipt")
        != str(submission_path)
    ):
        raise RuntimeError("pinned canary submission content differs")
    failure = terminal_path.parent / "canary-failure-receipt.json"
    forbidden = terminal_path.parent / "forbidden-attempts"
    forbidden_info = forbidden.lstat()
    if (
        failure.exists()
        or failure.is_symlink()
        or forbidden.resolve(strict=True) != forbidden
        or not stat.S_ISDIR(forbidden_info.st_mode)
        or stat.S_ISLNK(forbidden_info.st_mode)
        or forbidden_info.st_uid != os.getuid()
        or stat.S_IMODE(forbidden_info.st_mode) & 0o022
        or any(forbidden.iterdir())
    ):
        raise RuntimeError("pinned canary failure/scientific closure differs")
    return {
        "job_id": CANARY_JOB_ID,
        "terminal_receipt_path": str(terminal_path),
        "terminal_receipt_sha256": sha_bytes(terminal_raw),
        "terminal_receipt_digest": terminal["receipt_digest"],
        "submission_receipt_path": str(submission_path),
        "submission_receipt_sha256": sha_bytes(submission_raw),
        "submission_receipt_digest": submission["receipt_digest"],
        "slurm_state_required": "COMPLETED",
        "slurm_exit_code_required": "0:0",
        "allocated_gpu_resource_required": "gres/gpu:mi210=8",
    }


RETAINED_FD_CANARY_ADMISSION_FIELDS = {
    "schema_version", "status", "job_id", "job_success",
    "slurm_terminal_verified", "operational_canary_admitted",
    "formal_admission", "external_formal_submitter_exact_pin_required",
    "submission_receipt_path", "submission_receipt_sha256",
    "submission_receipt_digest", "output_parent_identity",
    "submission_receipt_identity", "operational_evidence_path",
    "operational_evidence_sha256", "operational_evidence_digest",
    "failure_receipt_absent", "wrapper_sha256", "payload_sha256",
    "guard_sha256", "runtime_sha256", "python_sha256",
    "release_manifest_path", "release_manifest_sha256",
    "release_manifest_digest", "postflight_path", "postflight_sha256",
    "postflight_sha256_pinned_by_release_manifest",
    "retained_verification_identities", "sacct_terminal_observation",
    "slurm_logs", "output_namespace_deep_closed",
    "underlying_world8_closure_deep_validated",
    "science_generation_entered", "probe_validator_sha256",
    "probe_admission_binding", "authority",
    "receipt_digest",
}
PROBE_ADMISSION_BINDING_FIELDS = {
    "path", "sha256", "receipt_digest", "schema_version", "status",
    "slurm_job_id", "compute_bash", "submission_receipt_sha256",
    "submission_receipt_digest", "submission_receipt_path",
    "operational_evidence_sha256", "operational_evidence_digest",
    "operational_evidence_path", "release_manifest_path",
    "release_manifest_file_sha256", "release_manifest_digest",
    "wrapper_sha256", "postflight_sha256", "authority",
}


def ensure_r2_release_pins_resolved() -> None:
    values = {
        "retained canary job ID": RETAINED_FD_CANARY_JOB_ID,
        "retained canary admission path": RETAINED_FD_CANARY_ADMISSION_PATH,
        "retained canary admission SHA": RETAINED_FD_CANARY_ADMISSION_SHA256,
        "retained canary admission digest": RETAINED_FD_CANARY_ADMISSION_DIGEST,
        "retained canary sacct stdout SHA":
            RETAINED_FD_CANARY_SACCT_STDOUT_SHA256,
        "retained canary node": RETAINED_FD_CANARY_NODELIST,
        "retained canary start": RETAINED_FD_CANARY_START,
        "retained canary end": RETAINED_FD_CANARY_END,
        "retained canary elapsed": RETAINED_FD_CANARY_ELAPSED,
        "retained canary submit line": RETAINED_FD_CANARY_SUBMIT_LINE,
        "compute bash SHA": COMPUTE_BASH_SHA256,
        "compute bash version SHA": COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "compute bash first line": COMPUTE_BASH_VERSION_FIRST_LINE,
        "compute bash probe job ID": COMPUTE_BASH_PROBE_JOB_ID,
        "compute bash probe admission path": COMPUTE_BASH_PROBE_ADMISSION_PATH,
        "compute bash probe admission SHA": COMPUTE_BASH_PROBE_ADMISSION_SHA256,
        "compute bash probe admission digest":
            COMPUTE_BASH_PROBE_ADMISSION_DIGEST,
        "probe validator SHA": PROBE_VALIDATOR_SHA256,
    }
    if any("__R2_" in value for value in values.values()):
        raise RuntimeError("formal r2 template pins remain unresolved")
    for label in (
        "retained canary admission SHA", "retained canary admission digest",
        "retained canary sacct stdout SHA", "compute bash SHA",
        "compute bash version SHA", "compute bash probe admission SHA",
        "compute bash probe admission digest", "probe validator SHA",
    ):
        if SHA256.fullmatch(values[label]) is None:
            raise RuntimeError(f"{label} differs")
    if (
        not RETAINED_FD_CANARY_JOB_ID.isdigit()
        or not COMPUTE_BASH_PROBE_JOB_ID.isdigit()
        or not Path(RETAINED_FD_CANARY_ADMISSION_PATH).is_absolute()
        or not Path(COMPUTE_BASH_PROBE_ADMISSION_PATH).is_absolute()
        or "/proc/self/fd/" not in RETAINED_FD_CANARY_SUBMIT_LINE
        or "--export=NONE," not in RETAINED_FD_CANARY_SUBMIT_LINE
        or "--hold" in RETAINED_FD_CANARY_SUBMIT_LINE
        or "--dependency" in RETAINED_FD_CANARY_SUBMIT_LINE
    ):
        raise RuntimeError("formal r2 retained-canary release pins differ")


def expected_compute_bash_identity() -> dict[str, Any]:
    return {
        "path": COMPUTE_BASH_PATH,
        "sha256": COMPUTE_BASH_SHA256,
        "version_stdout_sha256": COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "version_first_line": COMPUTE_BASH_VERSION_FIRST_LINE,
        "brace_fd_redirection_supported": True,
        "retained_fd_survives_bash_script_handoff": True,
        "varredir_close_option_required": False,
    }


def validate_compute_bash() -> dict[str, Any]:
    ensure_r2_release_pins_resolved()
    path = Path(COMPUTE_BASH_PATH)
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(path, os.X_OK)
        or sha_file(path) != COMPUTE_BASH_SHA256
    ):
        raise RuntimeError("root-owned compute bash executable differs")
    completed = subprocess.run(
        [str(path), "--version"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=60, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("compute bash version output is not ASCII") from error
    if (
        completed.returncode != 0
        or completed.stderr != b""
        or sha_bytes(completed.stdout) != COMPUTE_BASH_VERSION_STDOUT_SHA256
        or not lines
        or lines[0] != COMPUTE_BASH_VERSION_FIRST_LINE
    ):
        raise RuntimeError("compute bash version contract differs")
    return expected_compute_bash_identity()


def expected_retained_fd_canary_sacct_observation(phase: str) -> dict[str, Any]:
    submit_line = RETAINED_FD_CANARY_SUBMIT_LINE
    retained_match = re.fullmatch(
        r".* /proc/self/fd/([0-9]+)", submit_line,
    )
    if (
        retained_match is None
        or str(int(retained_match.group(1))) != retained_match.group(1)
        or int(retained_match.group(1)) < 3
    ):
        raise RuntimeError("pinned retained-canary submit line differs")
    return {
        "executable": "/usr/bin/sacct",
        "executable_sha256": SACCT_SHA256,
        "argv": RETAINED_FD_CANARY_SACCT_ARGV,
        "query_fields": RETAINED_FD_CANARY_SACCT_FIELDS,
        "returncode": 0,
        "stdout_sha256": RETAINED_FD_CANARY_SACCT_STDOUT_SHA256,
        "stderr_sha256": EMPTY_SHA256,
        "parsed_row": RETAINED_FD_CANARY_SACCT_PARSED_ROW,
        "exact_single_row": True,
        "submit_line_sha256": sha_bytes(submit_line.encode("ascii")),
        "retained_wrapper_fd": int(retained_match.group(1)),
        "exact_submit_line": True,
        "observation_phase": phase,
    }


def observe_retained_fd_canary_sacct(phase: str) -> dict[str, Any]:
    ensure_r2_release_pins_resolved()
    sacct = Path("/usr/bin/sacct")
    info = sacct.lstat()
    if (
        sacct.resolve(strict=True) != sacct
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(sacct, os.X_OK)
        or sha_file(sacct) != SACCT_SHA256
    ):
        raise RuntimeError("root-owned sacct executable differs for retained canary")
    completed = subprocess.run(
        RETAINED_FD_CANARY_SACCT_ARGV, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=60, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("retained-canary sacct stdout is not ASCII") from error
    lines = stdout.splitlines()
    fields = lines[0].split("|") if len(lines) == 1 else []
    keys = [item.split("%", 1)[0] for item in RETAINED_FD_CANARY_SACCT_FIELDS]
    parsed = dict(zip(keys, fields, strict=True)) if len(fields) == len(keys) else {}
    submit_line = str(parsed.get("SubmitLine", ""))
    retained_match = re.fullmatch(
        r".* /proc/self/fd/([0-9]+)", submit_line,
    )
    observed = {
        "executable": str(sacct),
        "executable_sha256": SACCT_SHA256,
        "argv": RETAINED_FD_CANARY_SACCT_ARGV,
        "query_fields": RETAINED_FD_CANARY_SACCT_FIELDS,
        "returncode": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "parsed_row": parsed,
        "exact_single_row": len(lines) == 1,
        "submit_line_sha256": sha_bytes(submit_line.encode("ascii")),
        "retained_wrapper_fd": (
            int(retained_match.group(1)) if retained_match is not None else -1
        ),
        "exact_submit_line": (
            retained_match is not None
            and str(int(retained_match.group(1))) == retained_match.group(1)
            and int(retained_match.group(1)) >= 3
            and submit_line == RETAINED_FD_CANARY_SUBMIT_LINE
        ),
        "observation_phase": phase,
    }
    if observed != expected_retained_fd_canary_sacct_observation(phase):
        raise RuntimeError("authoritative retained-canary sacct observation differs")
    return observed


def validate_retained_fd_canary(
    module,
    admission_path: Path,
    probe_binding: Mapping[str, Any],
) -> dict[str, Any]:
    ensure_r2_release_pins_resolved()
    if str(admission_path) != RETAINED_FD_CANARY_ADMISSION_PATH:
        raise RuntimeError("retained-canary admission path differs")
    info = admission_path.lstat()
    if (
        admission_path.resolve(strict=True) != admission_path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
    ):
        raise RuntimeError("retained-canary admission identity differs")
    raw = module.wait_ready_bytes(
        admission_path, label="pinned retained-FD WORLD8 canary admission"
    )
    if sha_bytes(raw) != RETAINED_FD_CANARY_ADMISSION_SHA256:
        raise RuntimeError("retained-canary admission raw SHA differs")
    value = module._decode_sealed(
        raw,
        schema_version="saic-formal-v2-retained-fd-world8-canary-admission-v1",
        exact_fields=RETAINED_FD_CANARY_ADMISSION_FIELDS,
    )
    identities = value.get("retained_verification_identities")
    output_parent_identity = value.get("output_parent_identity")
    submission_receipt_identity = value.get("submission_receipt_identity")
    logs = value.get("slurm_logs")
    authority = {
        "scientific": False,
        "generation": False,
        "training": False,
        "publication": False,
        "formal_job_authorized": False,
        "operational_gate": "exact_saic_t2v_topup_r6_formal_v2_release_only",
        "reusable_for_other_release": False,
        "authorizes_formal_submission_by_itself": False,
    }
    if (
        value.get("receipt_digest") != RETAINED_FD_CANARY_ADMISSION_DIGEST
        or value.get("status")
        != "terminal_completed_retained_fd_world8_operational_admitted"
        or value.get("job_id") != RETAINED_FD_CANARY_JOB_ID
        or value.get("job_success") is not True
        or value.get("slurm_terminal_verified") is not True
        or value.get("operational_canary_admitted") is not True
        or value.get("formal_admission") is not False
        or value.get("external_formal_submitter_exact_pin_required") is not True
        or value.get("failure_receipt_absent") is not True
        or value.get("guard_sha256") != GUARD_V2_SHA256
        or value.get("runtime_sha256") != RUNTIME_SHA256
        or value.get("python_sha256") != FORMAL_PYTHON_SHA256
        or value.get("postflight_sha256_pinned_by_release_manifest") is not True
        or value.get("output_namespace_deep_closed") is not True
        or value.get("underlying_world8_closure_deep_validated") is not True
        or value.get("science_generation_entered") is not False
        or value.get("probe_validator_sha256") != PROBE_VALIDATOR_SHA256
        or value.get("probe_admission_binding") != probe_binding
        or value.get("authority") != authority
        or value.get("sacct_terminal_observation")
        != expected_retained_fd_canary_sacct_observation(
            "external_postflight_after_canary_terminal"
        )
        or any(
            not isinstance(item, dict)
            or set(item) != {"device", "inode"}
            or not isinstance(item["device"], int)
            or item["device"] < 0
            or not isinstance(item["inode"], int)
            or item["inode"] <= 0
            for item in (output_parent_identity, submission_receipt_identity)
        )
        or not isinstance(identities, dict)
        or set(identities) != {
            "postflight", "release_manifest", "guard", "probe_validator",
        }
        or any(
            not isinstance(item, dict)
            or set(item) != {"device", "inode"}
            or not isinstance(item["device"], int)
            or item["device"] < 0
            or not isinstance(item["inode"], int)
            or item["inode"] <= 0
            for item in identities.values()
        )
        or not isinstance(logs, dict)
        or set(logs) != {"stdout", "stderr"}
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size"}
            or not isinstance(item["path"], str)
            or not item["path"].startswith(FORMAL_ROOT + "/slurm/")
            or SHA256.fullmatch(str(item["sha256"])) is None
            or not isinstance(item["size"], int)
            or item["size"] < 0
            for item in logs.values()
        )
        or logs["stderr"]["size"] != 0
    ):
        raise RuntimeError("pinned retained-FD WORLD8 canary admission differs")
    return {
        "job_id": RETAINED_FD_CANARY_JOB_ID,
        "admission_path": str(admission_path),
        "admission_sha256": sha_bytes(raw),
        "admission_digest": value["receipt_digest"],
        "operational_evidence_path": value["operational_evidence_path"],
        "operational_evidence_sha256": value["operational_evidence_sha256"],
        "operational_evidence_digest": value["operational_evidence_digest"],
        "wrapper_sha256": value["wrapper_sha256"],
        "payload_sha256": value["payload_sha256"],
        "guard_sha256": value["guard_sha256"],
        "runtime_sha256": value["runtime_sha256"],
        "probe_validator_sha256": value["probe_validator_sha256"],
        "probe_admission_binding": value["probe_admission_binding"],
        "compute_bash": value["probe_admission_binding"]["compute_bash"],
        "external_postflight_sacct_observation":
            value["sacct_terminal_observation"],
        "slurm_state_required": "COMPLETED",
        "slurm_exit_code_required": "0:0",
        "allocated_gpu_resource_required": "gres/gpu:mi210=8",
        "science_generation_entered": False,
        "formal_submission_authorized_by_canary_alone": False,
    }


def observe_canary_sacct() -> dict[str, Any]:
    sacct = Path("/usr/bin/sacct")
    info = sacct.lstat()
    if (
        sacct.resolve(strict=True) != sacct
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(sacct, os.X_OK)
        or sha_file(sacct) != SACCT_SHA256
    ):
        raise RuntimeError("root-owned sacct executable differs")
    completed = subprocess.run(
        SACCT_ARGV,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("sacct stdout is not ASCII") from error
    lines = stdout.splitlines()
    fields = lines[0].split("|") if len(lines) == 1 else []
    keys = [field.split("%", 1)[0] for field in SACCT_FIELDS]
    parsed = dict(zip(keys, fields, strict=True)) if len(fields) == len(keys) else {}
    observation = {
        "executable": str(sacct),
        "executable_sha256": SACCT_SHA256,
        "argv": SACCT_ARGV,
        "query_fields": SACCT_FIELDS,
        "returncode": completed.returncode,
        "stdout_sha256": sha_bytes(completed.stdout),
        "stderr_sha256": sha_bytes(completed.stderr),
        "parsed_row": parsed,
        "exact_single_row": len(lines) == 1,
        "observation_phase": "runtime_gate_after_formal_sbatch",
        "runtime_recheck_passed_before_output": True,
    }
    if observation != {
        "executable": "/usr/bin/sacct",
        "executable_sha256": SACCT_SHA256,
        "argv": SACCT_ARGV,
        "query_fields": SACCT_FIELDS,
        "returncode": 0,
        "stdout_sha256": SACCT_STDOUT_SHA256,
        "stderr_sha256": EMPTY_SHA256,
        "parsed_row": SACCT_PARSED_ROW,
        "exact_single_row": True,
        "observation_phase": "runtime_gate_after_formal_sbatch",
        "runtime_recheck_passed_before_output": True,
    }:
        raise RuntimeError("authoritative runtime sacct observation differs")
    return observation


def expected_submitter_sacct_precheck() -> dict[str, Any]:
    return {
        "executable": "/usr/bin/sacct",
        "executable_sha256": SACCT_SHA256,
        "argv": SACCT_ARGV,
        "query_fields": SACCT_FIELDS,
        "returncode": 0,
        "stdout_sha256": SACCT_STDOUT_SHA256,
        "stderr_sha256": EMPTY_SHA256,
        "parsed_row": SACCT_PARSED_ROW,
        "exact_single_row": True,
        "observation_phase": "submitter_before_formal_sbatch",
        "precheck_passed_before_only_sbatch": True,
    }


def expected_inputs_from_environment() -> dict[str, Any]:
    get = os.environ.__getitem__
    value = {
        "wrapper": get("SAIC_T2V_FV2_WRAPPER"),
        "wrapper_sha256": get("SAIC_T2V_FV2_WRAPPER_SHA256"),
        "base_launcher": get("SAIC_T2V_FV2_BASE_LAUNCHER"),
        "base_launcher_sha256": get("SAIC_T2V_FV2_BASE_LAUNCHER_SHA256"),
        "materializer": get("SAIC_T2V_FV2_MATERIALIZER"),
        "materializer_sha256": get("SAIC_T2V_FV2_MATERIALIZER_SHA256"),
        "effective_launcher": get("SAIC_T2V_FV2_EFFECTIVE_LAUNCHER"),
        "effective_launcher_sha256": get(
            "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256"
        ),
        "gate": get("SAIC_T2V_FV2_GATE"),
        "gate_sha256": get("SAIC_T2V_FV2_GATE_SHA256"),
        "rendezvous_guard": get("SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD"),
        "rendezvous_guard_sha256": get(
            "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256"
        ),
        "retained_fd_canary_admission": get(
            "SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION"
        ),
        "retained_fd_canary_admission_sha256":
            RETAINED_FD_CANARY_ADMISSION_SHA256,
        "retained_fd_canary_admission_digest":
            RETAINED_FD_CANARY_ADMISSION_DIGEST,
        "retained_fd_canary_job_id": RETAINED_FD_CANARY_JOB_ID,
        "probe_validator": get("SAIC_T2V_FV2_PROBE_VALIDATOR"),
        "probe_validator_sha256": get(
            "SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256"
        ),
        "compute_bash_probe_admission": get(
            "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION"
        ),
        "compute_bash_probe_admission_sha256": get(
            "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256"
        ),
        "compute_bash_probe_admission_digest": get(
            "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST"
        ),
        "source_archive": get("SAIC_T2V_V3_SOURCE_ARCHIVE"),
        "source_archive_sha256": get("SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256"),
        "generation_runtime_sha256": RUNTIME_SHA256,
        "archived_rendezvous_guard_v1_sha256": ARCHIVED_GUARD_V1_SHA256,
        "base_v1_spec_sha256": BASE_V1_SPEC_SHA256,
        "contract_runtime_sha256": CONTRACT_RUNTIME_SHA256,
        "effective_dynamic_plan_schema_version":
            "saic-t2v-topup-rendezvous-dynamic-plan-v2",
        "scientific_spec_changed_for_rendezvous_guard_v2": False,
        "source_revision": get("SAIC_T2V_V3_SOURCE_REVISION"),
        "source_manifest": get("SAIC_T2V_V3_SOURCE_MANIFEST"),
        "source_manifest_sha256": get("SAIC_T2V_V3_SOURCE_MANIFEST_SHA256"),
        "event_spec": get("SAIC_T2V_V3_EVENT_SPEC"),
        "event_spec_sha256": get("SAIC_T2V_V3_EVENT_SPEC_SHA256"),
        "checkpoint_manifest": get("BERNINI_CHECKPOINT_CONTENT_MANIFEST"),
        "checkpoint_manifest_sha256": get(
            "SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256"
        ),
        "python": get("SAIC_T2V_V3_PYTHON_BIN"),
        "python_sha256": get("SAIC_T2V_FV2_PYTHON_SHA256"),
        "static_ffmpeg": get("SAIC_T2V_V3_STATIC_FFMPEG"),
        "static_ffmpeg_sha256": get("SAIC_T2V_FV2_STATIC_FFMPEG_SHA256"),
        "static_ffmpeg_version_stdout_sha256": get(
            "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256"
        ),
        "static_ffmpeg_version_first_line": get(
            "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE"
        ),
        "compute_bash": get("SAIC_T2V_FV2_COMPUTE_BASH"),
        "compute_bash_sha256": get("SAIC_T2V_FV2_COMPUTE_BASH_SHA256"),
        "compute_bash_version_stdout_sha256": get(
            "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256"
        ),
        # The exact first line is embedded in these gate bytes.  AUH Bash emits
        # a comma in it, so it is deliberately not passed via Slurm --export.
        "compute_bash_version_first_line": COMPUTE_BASH_VERSION_FIRST_LINE,
        "bernini_root": get("BERNINI_OFFICIAL_ROOT"),
        "veomni_root": get("BERNINI_VEOMNI_ROOT"),
        "checkpoint": get("BERNINI_ACTION_CHECKPOINT"),
    }
    fixed = {
        "wrapper": FORMAL_WRAPPER,
        "base_launcher": FORMAL_BASE_LAUNCHER,
        "base_launcher_sha256": FORMAL_BASE_LAUNCHER_SHA256,
        "materializer": FORMAL_MATERIALIZER,
        "materializer_sha256": FORMAL_MATERIALIZER_SHA256,
        "effective_launcher": FORMAL_EFFECTIVE_LAUNCHER,
        "effective_launcher_sha256": FORMAL_EFFECTIVE_LAUNCHER_SHA256,
        "gate": FORMAL_GATE,
        "rendezvous_guard": FORMAL_GUARD_V2,
        "rendezvous_guard_sha256": GUARD_V2_SHA256,
        "retained_fd_canary_admission": RETAINED_FD_CANARY_ADMISSION_PATH,
        "retained_fd_canary_admission_sha256":
            RETAINED_FD_CANARY_ADMISSION_SHA256,
        "retained_fd_canary_admission_digest":
            RETAINED_FD_CANARY_ADMISSION_DIGEST,
        "retained_fd_canary_job_id": RETAINED_FD_CANARY_JOB_ID,
        "probe_validator": FORMAL_PROBE_VALIDATOR,
        "probe_validator_sha256": PROBE_VALIDATOR_SHA256,
        "compute_bash_probe_admission": COMPUTE_BASH_PROBE_ADMISSION_PATH,
        "compute_bash_probe_admission_sha256":
            COMPUTE_BASH_PROBE_ADMISSION_SHA256,
        "compute_bash_probe_admission_digest":
            COMPUTE_BASH_PROBE_ADMISSION_DIGEST,
        "source_archive": FORMAL_SOURCE_ARCHIVE,
        "source_archive_sha256": FORMAL_SOURCE_ARCHIVE_SHA256,
        "generation_runtime_sha256": RUNTIME_SHA256,
        "archived_rendezvous_guard_v1_sha256": ARCHIVED_GUARD_V1_SHA256,
        "base_v1_spec_sha256": BASE_V1_SPEC_SHA256,
        "contract_runtime_sha256": CONTRACT_RUNTIME_SHA256,
        "effective_dynamic_plan_schema_version":
            "saic-t2v-topup-rendezvous-dynamic-plan-v2",
        "scientific_spec_changed_for_rendezvous_guard_v2": False,
        "source_revision": FORMAL_SOURCE_REVISION,
        "source_manifest": FORMAL_SOURCE_MANIFEST,
        "source_manifest_sha256": FORMAL_SOURCE_MANIFEST_SHA256,
        "event_spec": FORMAL_EVENT_SPEC,
        "event_spec_sha256": FORMAL_EVENT_SPEC_SHA256,
        "checkpoint_manifest": FORMAL_CHECKPOINT_MANIFEST,
        "checkpoint_manifest_sha256": FORMAL_CHECKPOINT_MANIFEST_SHA256,
        "python": FORMAL_PYTHON,
        "python_sha256": FORMAL_PYTHON_SHA256,
        "static_ffmpeg": STATIC_FFMPEG_PATH,
        "static_ffmpeg_sha256": STATIC_FFMPEG_SHA256,
        "static_ffmpeg_version_stdout_sha256":
            STATIC_FFMPEG_VERSION_STDOUT_SHA256,
        "static_ffmpeg_version_first_line": STATIC_FFMPEG_VERSION_FIRST_LINE,
        "compute_bash": COMPUTE_BASH_PATH,
        "compute_bash_sha256": COMPUTE_BASH_SHA256,
        "compute_bash_version_stdout_sha256":
            COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "compute_bash_version_first_line": COMPUTE_BASH_VERSION_FIRST_LINE,
        "bernini_root": FORMAL_BERNINI_ROOT,
        "veomni_root": FORMAL_VEOMNI_ROOT,
        "checkpoint": FORMAL_CHECKPOINT,
    }
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise RuntimeError("formal hard-pinned input environment differs")
    if (
        SHA256.fullmatch(str(value.get("wrapper_sha256"))) is None
        or SHA256.fullmatch(str(value.get("gate_sha256"))) is None
    ):
        raise RuntimeError("formal trust-root SHA environment differs")
    return value


def validate_own_submission(
    module,
    receipt_path: Path,
    *,
    job_id: str,
    canary: Mapping[str, Any],
    probe_binding: Mapping[str, Any],
    retained_canary: Mapping[str, Any],
) -> dict[str, Any]:
    raw = module.wait_ready_bytes(receipt_path, label="own formal submission receipt")
    value = module._decode_sealed(
        raw,
        schema_version="saic-t2v-topup-r6-formal-v2-r2-submission-v1",
        exact_fields={
            "schema_version", "status", "submission_success", "job_success",
            "submitted_job", "request", "submission_boundary", "inputs",
            "canary_admission", "outputs", "authority", "threat_model",
            "receipt_digest",
        },
    )
    submitted = value.get("submitted_job")
    boundary = value.get("submission_boundary")
    output_root = os.environ["SAIC_T2V_V3_OUTPUT_ROOT"]
    expected_canary_admission = {
        **canary,
        "sacct_observation": expected_submitter_sacct_precheck(),
        "compute_bash_probe_admission": probe_binding,
        "retained_fd_world8": {
            **retained_canary,
            "submitter_sacct_observation":
                expected_retained_fd_canary_sacct_observation(
                    "submitter_before_formal_sbatch"
                ),
        },
    }
    if (
        not job_id.isdigit()
        or value.get("status") != "submitted"
        or value.get("submission_success") is not True
        or value.get("job_success") is not None
        or not isinstance(submitted, dict)
        or set(submitted) != {
            "job_id", "cluster", "stdout_sha256", "stderr_sha256"
        }
        or submitted.get("job_id") != job_id
        or (
            submitted.get("cluster") is not None
            and (
                not isinstance(submitted.get("cluster"), str)
                or not submitted["cluster"]
                or "\n" in submitted["cluster"]
                or ";" in submitted["cluster"]
            )
        )
        or SHA256.fullmatch(str(submitted.get("stdout_sha256"))) is None
        or SHA256.fullmatch(str(submitted.get("stderr_sha256"))) is None
        or value.get("request") != {
            "job_name": "saic-t2v-topup-r6-v2-r2", "partition": "faculty",
            "qos": "bgqos", "nodes": 1, "ntasks": 1,
            "cpus_per_task": 32, "memory": "256G", "walltime": "24:00:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4_sp4",
            "candidate_count": 60,
            "dynamic_plan_schema_version":
                "saic-t2v-topup-rendezvous-dynamic-plan-v2",
            "fixed_candidate_set_and_order": True,
            "scientific_spec_changed_for_rendezvous_guard_v2": False,
            "hold": False, "dependency": None,
        }
        or not isinstance(boundary, dict)
        or set(boundary) != {
            "environment_replaced", "exact_job_export_names", "export_all",
            "reservation_created_before_sbatch", "same_inode_retained",
            "launcher_submitted_from_retained_fd", "reservation_device",
            "reservation_inode", "success_mode",
            "runtime_retained_fd_admission_roots",
            "pathname_replacement_resistant_admission_handoff",
            "compute_bash_exact_identity_pinned",
            "varredir_close_option_required",
        }
        or boundary.get("environment_replaced") is not True
        or boundary.get("exact_job_export_names") != EXPORT_NAMES
        or boundary.get("export_all") is not False
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("same_inode_retained") is not True
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or boundary.get("runtime_retained_fd_admission_roots")
        != [
            "formal_gate", "effective_launcher", "rendezvous_guard_v2",
            "compute_bash_probe_validator",
        ]
        or boundary.get("pathname_replacement_resistant_admission_handoff")
        is not True
        or boundary.get("compute_bash_exact_identity_pinned") is not True
        or boundary.get("varredir_close_option_required") is not False
        or not isinstance(boundary.get("reservation_device"), int)
        or boundary["reservation_device"] < 0
        or not isinstance(boundary.get("reservation_inode"), int)
        or boundary["reservation_inode"] <= 0
        or boundary.get("success_mode") != "0444"
        or value.get("inputs") != expected_inputs_from_environment()
        or value.get("canary_admission") != expected_canary_admission
        or value.get("outputs") != {
            "output_root": output_root,
            "submission_receipt": str(receipt_path),
            "slurm_log_dir": os.environ["SAIC_T2V_FV2_SLURM_LOG_DIR"],
            "fresh_before_submission": True,
        }
        or value.get("authority") != {
            "diagnostic_event_bank_execution_authorized": True,
            "training": False, "checkpoint": False,
            "scientific_success_claimed": False,
            "action_edit_success_claimed": False,
            "job_success_claimed": False,
        }
        or value.get("threat_model") != THREAT_MODEL
    ):
        raise RuntimeError("own formal submission receipt differs")
    if Path(output_root).exists() or Path(output_root).is_symlink():
        raise RuntimeError("formal output appeared before admission completed")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--guard-sha256", required=True)
    parser.add_argument("--canary-receipt", required=True)
    parser.add_argument("--canary-submission-receipt", required=True)
    parser.add_argument("--retained-fd-canary-admission", required=True)
    parser.add_argument("--probe-validator", required=True)
    parser.add_argument("--probe-validator-sha256", required=True)
    parser.add_argument("--compute-bash-probe-admission", required=True)
    parser.add_argument("--compute-bash-probe-admission-sha256", required=True)
    parser.add_argument("--compute-bash-probe-admission-digest", required=True)
    parser.add_argument("--own-submission-receipt", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_r2_release_pins_resolved()
    if args.guard_sha256 != GUARD_V2_SHA256:
        raise SystemExit("formal guard-v2 SHA pin differs")
    if (
        os.environ.get("SAIC_T2V_V3_STATIC_FFMPEG") != STATIC_FFMPEG_PATH
        or os.environ.get("SAIC_T2V_FV2_STATIC_FFMPEG_SHA256")
        != STATIC_FFMPEG_SHA256
        or os.environ.get("SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256")
        != STATIC_FFMPEG_VERSION_STDOUT_SHA256
        or os.environ.get("SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE")
        != STATIC_FFMPEG_VERSION_FIRST_LINE
    ):
        raise SystemExit("formal static ffmpeg pins differ")
    gate_fd_path = os.environ.get("SAIC_T2V_FV2_GATE_FD_PATH", "")
    gate_path = Path(__file__)
    if (
        re.fullmatch(r"/proc/[0-9]+/fd/[0-9]+", gate_fd_path) is None
        or str(gate_path) != gate_fd_path
        or os.environ.get("SAIC_T2V_FV2_GATE") != FORMAL_GATE
        or os.environ.get("SAIC_T2V_FV2_GATE_SHA256") != sha_file(gate_path)
        or os.environ.get("SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT")
        != FORMAL_SUBMISSION_RECEIPT
        or os.environ.get("SAIC_T2V_V3_OUTPUT_ROOT") != FORMAL_OUTPUT_ROOT
        or os.environ.get("SAIC_T2V_FV2_SLURM_LOG_DIR") != FORMAL_SLURM_LOG_DIR
        or args.canary_receipt != CANARY_RECEIPT_PATH
        or args.canary_submission_receipt != CANARY_SUBMISSION_PATH
        or args.retained_fd_canary_admission
        != RETAINED_FD_CANARY_ADMISSION_PATH
        or args.probe_validator_sha256 != PROBE_VALIDATOR_SHA256
        or args.compute_bash_probe_admission
        != COMPUTE_BASH_PROBE_ADMISSION_PATH
        or args.compute_bash_probe_admission_sha256
        != COMPUTE_BASH_PROBE_ADMISSION_SHA256
        or args.compute_bash_probe_admission_digest
        != COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        or args.own_submission_receipt != FORMAL_SUBMISSION_RECEIPT
        or os.environ.get("SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION")
        != RETAINED_FD_CANARY_ADMISSION_PATH
        or os.environ.get("SAIC_T2V_FV2_PROBE_VALIDATOR")
        != FORMAL_PROBE_VALIDATOR
        or os.environ.get("SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256")
        != PROBE_VALIDATOR_SHA256
        or os.environ.get("SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION")
        != COMPUTE_BASH_PROBE_ADMISSION_PATH
        or os.environ.get(
            "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256"
        ) != COMPUTE_BASH_PROBE_ADMISSION_SHA256
        or os.environ.get(
            "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST"
        ) != COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        or os.environ.get("SAIC_T2V_FV2_COMPUTE_BASH") != COMPUTE_BASH_PATH
        or os.environ.get("SAIC_T2V_FV2_COMPUTE_BASH_SHA256")
        != COMPUTE_BASH_SHA256
        or os.environ.get("SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256")
        != COMPUTE_BASH_VERSION_STDOUT_SHA256
    ):
        raise SystemExit("running formal gate bytes differ")
    expected_inputs_from_environment()
    module = load_guard(Path(args.guard), args.guard_sha256)
    if os.environ.get("SAIC_T2V_FV2_RENDEZVOUS_GUARD_FD_PATH") != args.guard:
        raise SystemExit("retained formal guard fd path differs")
    if os.environ.get("SAIC_T2V_FV2_PROBE_VALIDATOR_FD_PATH") != args.probe_validator:
        raise SystemExit("retained formal probe-validator fd path differs")
    validator = load_probe_validator(
        Path(args.probe_validator), args.probe_validator_sha256,
    )
    exact_plain(
        Path(os.environ["SAIC_T2V_FV2_MATERIALIZER"]),
        os.environ["SAIC_T2V_FV2_MATERIALIZER_SHA256"],
        "formal-v2 deterministic launcher materializer",
    )
    canary = validate_canary(
        module, Path(args.canary_receipt), Path(args.canary_submission_receipt)
    )
    probe_binding = validate_compute_bash_probe_admission(
        validator, Path(args.compute_bash_probe_admission),
    )
    retained_canary = validate_retained_fd_canary(
        module, Path(args.retained_fd_canary_admission), probe_binding,
    )
    validate_compute_bash()
    observe_canary_sacct()
    observe_retained_fd_canary_sacct("runtime_gate_after_formal_sbatch")
    validate_own_submission(
        module,
        Path(args.own_submission_receipt),
        job_id=args.slurm_job_id,
        canary=canary,
        probe_binding=probe_binding,
        retained_canary=retained_canary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
