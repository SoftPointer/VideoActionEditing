#!/usr/bin/env python3
"""Fail-closed admission gate for the guard-v2 SAIC formal full60 job."""

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
    + "/releases/saic-t2v-topup-r6-formal-v2-guard1a38-20260812-r1"
)
FORMAL_OUTPUT_ROOT = (
    FORMAL_ROOT
    + "/runs/t2v-events-topup-r6-formal-v2-guard1a38-20260812-r1"
)
FORMAL_SUBMISSION_RECEIPT = FORMAL_OUTPUT_ROOT + ".submission.receipt.json"
FORMAL_SLURM_LOG_DIR = (
    FORMAL_ROOT + "/slurm/saic-t2v-topup-r6-formal-v2-guard1a38-20260812-r1"
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
    + "/inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v4.sbatch"
)
FORMAL_GATE = FORMAL_RELEASE_ROOT + "/inputs/saic_t2v_formal_v2_gate.py"
FORMAL_GUARD_V2 = FORMAL_RELEASE_ROOT + "/inputs/saic_t2v_rendezvous_guard_v2.py"
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
    ],
    "shared_science_paths_assumed_not_concurrently_replaced_by_same_uid": [
        "source_archive", "source_manifest", "event_spec",
        "checkpoint_manifest", "python", "static_ffmpeg",
        "bernini_root", "veomni_root", "checkpoint",
    ],
    "same_inode_in_place_mutation_resistance_claimed": False,
    "sealed_release_permissions_are_not_claimed_as_same_uid_immutability": True,
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
) -> dict[str, Any]:
    raw = module.wait_ready_bytes(receipt_path, label="own formal submission receipt")
    value = module._decode_sealed(
        raw,
        schema_version="saic-t2v-topup-r6-formal-v2-submission-v1",
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
            "job_name": "saic-t2v-topup-r6-v2", "partition": "faculty",
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
        }
        or boundary.get("environment_replaced") is not True
        or boundary.get("exact_job_export_names") != EXPORT_NAMES
        or boundary.get("export_all") is not False
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("same_inode_retained") is not True
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or boundary.get("runtime_retained_fd_admission_roots")
        != ["formal_gate", "effective_launcher", "rendezvous_guard_v2"]
        or boundary.get("pathname_replacement_resistant_admission_handoff")
        is not True
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
    parser.add_argument("--own-submission-receipt", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        or args.own_submission_receipt != FORMAL_SUBMISSION_RECEIPT
    ):
        raise SystemExit("running formal gate bytes differ")
    expected_inputs_from_environment()
    module = load_guard(Path(args.guard), args.guard_sha256)
    if os.environ.get("SAIC_T2V_FV2_RENDEZVOUS_GUARD_FD_PATH") != args.guard:
        raise SystemExit("retained formal guard fd path differs")
    exact_plain(
        Path(os.environ["SAIC_T2V_FV2_MATERIALIZER"]),
        os.environ["SAIC_T2V_FV2_MATERIALIZER_SHA256"],
        "formal-v2 deterministic launcher materializer",
    )
    canary = validate_canary(
        module, Path(args.canary_receipt), Path(args.canary_submission_receipt)
    )
    observe_canary_sacct()
    validate_own_submission(
        module,
        Path(args.own_submission_receipt),
        job_id=args.slurm_job_id,
        canary=canary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
