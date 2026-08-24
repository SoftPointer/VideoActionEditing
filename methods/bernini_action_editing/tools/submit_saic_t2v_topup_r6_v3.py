#!/usr/bin/env python3
"""R2 template: exactly-once submit three-proof SAIC pure-T2V full60.

This source fails closed until every ``__R2_*__`` value is replaced from one
successful retained-FD WORLD8 postflight admission binding the standalone
compute-Bash probe, and the final gate/wrapper byte hashes.  It never treats
any operational proof as scientific proof.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
from typing import Any, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

EXPECTED_BASE_LAUNCHER_SHA256 = (
    "12c1b2baaecfd479f65f9b5dbf0dbae17cd87196767e93c254fe2cffc895f29d"
)
EXPECTED_MATERIALIZER_SHA256 = (
    "5585abb927206d0813caca4cec8dc10b846fe0ace704538bb12e0ad5cffe8b97"
)
EXPECTED_EFFECTIVE_LAUNCHER_SHA256 = (
    "4227b4a00b7b2dea786457baad56b4fdcb4b476929e9619cb533a353b369f9f0"
)
EXPECTED_WRAPPER_SHA256 = (
    "4d5572f0c2da3efe84b87f5bf20db53facea8a853de04404388e8dd65b373f5d"
)
EXPECTED_GATE_SHA256 = (
    "877186f668f3ba89b9d887e81fbfa32a2d15b40f0e8b5f9c47b159bf88ad4151"
)
EXPECTED_GUARD_V2_SHA256 = (
    "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
)
EXPECTED_SOURCE_REVISION = "20c2193954e780e9654347754b1485f3492fbea5"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
EXPECTED_EVENT_SPEC_SHA256 = (
    "d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
EXPECTED_RUNTIME_SHA256 = (
    "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
)
EXPECTED_ARCHIVED_GUARD_V1_SHA256 = (
    "6666d3bd822baeb5d6f3ecb8033c71510d80e68c4059dfbbd09ef4dc4d100a9f"
)
EXPECTED_BASE_V1_SPEC_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
EXPECTED_CONTRACT_RUNTIME_SHA256 = (
    "508dde8d995dcc8deeccb47b35be71b9915a86964626383660d8eed952ef5278"
)
EXPECTED_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
EXPECTED_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
EXPECTED_STATIC_FFMPEG = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
    "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
EXPECTED_STATIC_FFMPEG_SHA256 = (
    "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
)
EXPECTED_STATIC_FFMPEG_VERSION_STDOUT_SHA256 = (
    "389368da4bcd4e22d7bf9134f3a8c24dd36027de7d963015230969a87c9e3339"
)
EXPECTED_STATIC_FFMPEG_VERSION_FIRST_LINE = (
    "ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  "
    "Copyright (c) 2000-2024 the FFmpeg developers"
)
EXPECTED_BERNINI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591"
)
EXPECTED_VEOMNI_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_graft_v1_20260810/"
    "phase_a_native_gpu_canary_dual4_all8_v1/releases/"
    "source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6"
)
EXPECTED_CHECKPOINT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
)
EXPECTED_CANARY_RECEIPT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "saic-r6-dynamic-rendezvous-v2-be3f82d-6314a4d7-r1/job-134393/"
    "canary-receipt.json"
)
EXPECTED_CANARY_SUBMISSION_RECEIPT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "saic-r6-dynamic-rendezvous-v2-be3f82d-6314a4d7-r1/"
    "submission-receipt.json"
)
EXPECTED_RETAINED_FD_CANARY_ADMISSION = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/canaries/saic-formal-v2-retained-fd-world8-canary-3571e921-6a83aaec-r7/canary-admission.json"
)
EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256 = (
    "4da203469b022fcdcf7a4d6ba377769d6b220e5b6ce0066e60587104f8405e26"
)
EXPECTED_RETAINED_FD_CANARY_ADMISSION_DIGEST = (
    "616c1a7587679975329e1211653720a7a2726c7cd676bcb0eedeea4bdda7b50d"
)
EXPECTED_RETAINED_FD_CANARY_JOB_ID = "134908"
EXPECTED_PROBE_VALIDATOR_SHA256 = (
    "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
)
EXPECTED_COMPUTE_BASH_PROBE_JOB_ID = "134647"
EXPECTED_COMPUTE_BASH_PROBE_ADMISSION = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"
)
EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
EXPECTED_COMPUTE_BASH = Path("/usr/bin/bash")
EXPECTED_COMPUTE_BASH_SHA256 = "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4"
EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256 = (
    "51bd40ffa4710175920033d329a0e9e1667e6b7f56178e302432ff4610d554a7"
)
EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE = "GNU bash, version 5.1.16(1)-release (x86_64-pc-linux-gnu)"
FORMAL_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
FORMAL_RELEASE_ROOT = (
    FORMAL_ROOT
    / "releases/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
EXPECTED_WRAPPER = (
    FORMAL_RELEASE_ROOT
    / "inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v5.sbatch"
)
EXPECTED_BASE_LAUNCHER = (
    FORMAL_RELEASE_ROOT
    / "inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.base.sbatch"
)
EXPECTED_MATERIALIZER = (
    FORMAL_RELEASE_ROOT
    / "inputs/materialize_saic_t2v_topup_formal_v2_launcher.py"
)
EXPECTED_EFFECTIVE_LAUNCHER = (
    FORMAL_RELEASE_ROOT
    / "inputs/auh_generate_saic_pure_t2v_event_bank_topup_all8_formal_v2_effective.sbatch"
)
EXPECTED_GATE = FORMAL_RELEASE_ROOT / "inputs/saic_t2v_formal_v2_r2_gate.py"
EXPECTED_GUARD_V2 = FORMAL_RELEASE_ROOT / "inputs/saic_t2v_rendezvous_guard_v2.py"
EXPECTED_PROBE_VALIDATOR = (
    FORMAL_RELEASE_ROOT / "inputs/probe_admission_binding_v1.py"
)
EXPECTED_SOURCE_ARCHIVE = (
    FORMAL_RELEASE_ROOT / "inputs/videoedit-saic-20c2193-methods.tar"
)
EXPECTED_SOURCE_MANIFEST = (
    FORMAL_RELEASE_ROOT / "inputs/saic_reversible_source_set_v1.json"
)
EXPECTED_EVENT_SPEC = (
    FORMAL_RELEASE_ROOT / "inputs/saic_pure_t2v_event_bank_topup_v2.json"
)
EXPECTED_CHECKPOINT_MANIFEST = (
    FORMAL_RELEASE_ROOT / "inputs/bernini-r13-ff4c5d4-checkpoint.sha256"
)
EXPECTED_OUTPUT_ROOT = (
    FORMAL_ROOT / "runs/t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1"
)
EXPECTED_SUBMISSION_RECEIPT = Path(
    str(EXPECTED_OUTPUT_ROOT) + ".submission.receipt.json"
)
EXPECTED_SLURM_LOG_DIR = (
    FORMAL_ROOT / "slurm/saic-t2v-topup-r6-formal-v2-r2-retfd-20260812-r1"
)

ARCHIVE_RUNTIME = (
    "methods/bernini_action_editing/generate_saic_pure_t2v_event_bank_topup_v2.py"
)
ARCHIVE_GUARD_V1 = "methods/bernini_action_editing/saic_t2v_rendezvous_guard_v1.py"
ARCHIVE_BASE_LAUNCHER = (
    "methods/bernini_action_editing/scripts/"
    "auh_generate_saic_pure_t2v_event_bank_topup_all8_v3.sbatch"
)
ARCHIVE_SOURCE_MANIFEST = (
    "methods/bernini_action_editing/assets/saic_reversible_source_set_v1.json"
)
ARCHIVE_EVENT_SPEC = (
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_topup_v2.json"
)


def die(message: str) -> None:
    raise SystemExit(f"submit-saic-t2v-topup-r6-v2: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def exact_file(value: str, expected_sha: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or SHA256.fullmatch(expected_sha) is None:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o444
        or digest_file(path) != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def exact_executable(value: str, expected_path: Path, expected_sha: str, label: str) -> Path:
    path = Path(value)
    if (
        path != expected_path
        or not path.is_absolute()
        or path.resolve(strict=True) != path
        or SHA256.fullmatch(expected_sha) is None
    ):
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
        or not os.access(path, os.X_OK)
        or digest_file(path) != expected_sha
    ):
        die(f"{label} bytes differ")
    return path


def exact_directory(value: str, expected_path: Path | None, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.resolve(strict=True) != path
        or (expected_path is not None and path != expected_path)
    ):
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        die(f"{label} is not an exact private directory")
    return path


def directory_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        wrote = os.write(descriptor, payload[offset:])
        if wrote <= 0:
            die("submission receipt write stalled")
        offset += wrote


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        die(f"{name} import differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_payloads(archive: Path, revision: str) -> dict[str, bytes]:
    if revision != EXPECTED_SOURCE_REVISION or REVISION.fullmatch(revision) is None:
        die("source revision differs")
    required = {
        ARCHIVE_RUNTIME,
        ARCHIVE_GUARD_V1,
        ARCHIVE_BASE_LAUNCHER,
        ARCHIVE_SOURCE_MANIFEST,
        ARCHIVE_EVENT_SPEC,
    }
    values: dict[str, bytes] = {}
    with tarfile.open(archive, "r:*") as handle:
        if handle.pax_headers.get("comment") != revision:
            die("source archive revision differs")
        for member in handle.getmembers():
            original = PurePosixPath(member.name)
            if original.is_absolute() or ".." in original.parts:
                die("source archive member escaped")
            normalized = original.as_posix()
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                die("source archive contains non-plain entry")
            if normalized in required:
                if not member.isfile() or normalized in values:
                    die("source archive required member differs")
                extracted = handle.extractfile(member)
                if extracted is None:
                    die("source archive member is unreadable")
                values[normalized] = extracted.read()
    if set(values) != required:
        die("source archive scientific closure differs")
    expected = {
        ARCHIVE_RUNTIME: EXPECTED_RUNTIME_SHA256,
        ARCHIVE_GUARD_V1: EXPECTED_ARCHIVED_GUARD_V1_SHA256,
        ARCHIVE_BASE_LAUNCHER: EXPECTED_BASE_LAUNCHER_SHA256,
        ARCHIVE_SOURCE_MANIFEST: EXPECTED_SOURCE_MANIFEST_SHA256,
        ARCHIVE_EVENT_SPEC: EXPECTED_EVENT_SPEC_SHA256,
    }
    for name, expected_sha in expected.items():
        if digest_bytes(values[name]) != expected_sha:
            die(f"source archive member SHA differs: {name}")
    return values


def validate_materialization(
    materializer: Path, base_launcher: Path, effective_launcher: Path
) -> None:
    module = load_module(materializer, "formal_v2_materializer")
    if (
        module.EXPECTED_BASE_SHA256 != EXPECTED_BASE_LAUNCHER_SHA256
        or module.EXPECTED_GUARD_V2_SHA256 != EXPECTED_GUARD_V2_SHA256
    ):
        die("materializer embedded pins differ")
    expected = module.transform(base_launcher.read_bytes())
    actual = effective_launcher.read_bytes()
    if expected != actual or digest_bytes(expected) != EXPECTED_EFFECTIVE_LAUNCHER_SHA256:
        die("effective launcher deterministic derivation differs")


def validate_canary(gate: Path, guard: Path) -> tuple[Any, dict[str, Any]]:
    module = load_module(gate, "formal_v2_gate_for_submitter")
    module.ensure_r2_release_pins_resolved()
    if (
        module.GUARD_V2_SHA256 != EXPECTED_GUARD_V2_SHA256
        or module.RUNTIME_SHA256 != EXPECTED_RUNTIME_SHA256
        or module.ARCHIVED_GUARD_V1_SHA256 != EXPECTED_ARCHIVED_GUARD_V1_SHA256
        or module.BASE_V1_SPEC_SHA256 != EXPECTED_BASE_V1_SPEC_SHA256
        or module.CONTRACT_RUNTIME_SHA256 != EXPECTED_CONTRACT_RUNTIME_SHA256
        or module.STATIC_FFMPEG_PATH != str(EXPECTED_STATIC_FFMPEG)
        or module.STATIC_FFMPEG_SHA256 != EXPECTED_STATIC_FFMPEG_SHA256
        or module.STATIC_FFMPEG_VERSION_STDOUT_SHA256
        != EXPECTED_STATIC_FFMPEG_VERSION_STDOUT_SHA256
        or module.STATIC_FFMPEG_VERSION_FIRST_LINE
        != EXPECTED_STATIC_FFMPEG_VERSION_FIRST_LINE
        or module.FORMAL_RELEASE_ROOT != str(FORMAL_RELEASE_ROOT)
        or module.FORMAL_OUTPUT_ROOT != str(EXPECTED_OUTPUT_ROOT)
        or module.FORMAL_SUBMISSION_RECEIPT != str(EXPECTED_SUBMISSION_RECEIPT)
        or module.FORMAL_SLURM_LOG_DIR != str(EXPECTED_SLURM_LOG_DIR)
        or module.RETAINED_FD_CANARY_JOB_ID
        != EXPECTED_RETAINED_FD_CANARY_JOB_ID
        or module.RETAINED_FD_CANARY_ADMISSION_PATH
        != str(EXPECTED_RETAINED_FD_CANARY_ADMISSION)
        or module.RETAINED_FD_CANARY_ADMISSION_SHA256
        != EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256
        or module.RETAINED_FD_CANARY_ADMISSION_DIGEST
        != EXPECTED_RETAINED_FD_CANARY_ADMISSION_DIGEST
        or module.PROBE_VALIDATOR_SHA256 != EXPECTED_PROBE_VALIDATOR_SHA256
        or module.COMPUTE_BASH_PROBE_JOB_ID
        != EXPECTED_COMPUTE_BASH_PROBE_JOB_ID
        or module.COMPUTE_BASH_PROBE_ADMISSION_PATH
        != str(EXPECTED_COMPUTE_BASH_PROBE_ADMISSION)
        or module.COMPUTE_BASH_PROBE_ADMISSION_SHA256
        != EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256
        or module.COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        != EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_DIGEST
        or module.FORMAL_PROBE_VALIDATOR != str(EXPECTED_PROBE_VALIDATOR)
        or module.COMPUTE_BASH_PATH != str(EXPECTED_COMPUTE_BASH)
        or module.COMPUTE_BASH_SHA256 != EXPECTED_COMPUTE_BASH_SHA256
        or module.COMPUTE_BASH_VERSION_STDOUT_SHA256
        != EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256
        or module.COMPUTE_BASH_VERSION_FIRST_LINE
        != EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE
    ):
        die("formal gate embedded pins differ")
    guard_module = module.load_guard(guard, EXPECTED_GUARD_V2_SHA256)
    canary = module.validate_canary(
        guard_module, EXPECTED_CANARY_RECEIPT, EXPECTED_CANARY_SUBMISSION_RECEIPT
    )
    return module, canary


def observe_canary_sacct(gate_module: Any) -> dict[str, Any]:
    sacct = exact_executable(
        "/usr/bin/sacct", Path("/usr/bin/sacct"), gate_module.SACCT_SHA256, "sacct"
    )
    argv = list(gate_module.SACCT_ARGV)
    if not argv or argv[0] != str(sacct):
        die("sacct argv differs")
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        die("sacct stdout is not ASCII")
    lines = stdout.splitlines()
    if (
        completed.returncode != 0
        or digest_bytes(completed.stdout) != gate_module.SACCT_STDOUT_SHA256
        or digest_bytes(completed.stderr) != gate_module.EMPTY_SHA256
        or len(lines) != 1
    ):
        die("pinned canary sacct raw observation differs")
    fields = lines[0].split("|")
    keys = [field.split("%", 1)[0] for field in gate_module.SACCT_FIELDS]
    if len(fields) != len(keys):
        die("pinned canary sacct field count differs")
    parsed = dict(zip(keys, fields, strict=True))
    if parsed != gate_module.SACCT_PARSED_ROW:
        die("pinned canary sacct parsed row differs")
    return {
        "executable": str(sacct),
        "executable_sha256": gate_module.SACCT_SHA256,
        "argv": argv,
        "query_fields": list(gate_module.SACCT_FIELDS),
        "returncode": completed.returncode,
        "stdout_sha256": digest_bytes(completed.stdout),
        "stderr_sha256": digest_bytes(completed.stderr),
        "parsed_row": parsed,
        "exact_single_row": True,
        "observation_phase": "submitter_before_formal_sbatch",
        "precheck_passed_before_only_sbatch": True,
    }


def validate_static_ffmpeg(path: Path) -> None:
    completed = subprocess.run(
        [str(path), "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        lines = completed.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError:
        die("static ffmpeg version output is not ASCII")
    if (
        completed.returncode != 0
        or completed.stderr != b""
        or digest_bytes(completed.stdout)
        != EXPECTED_STATIC_FFMPEG_VERSION_STDOUT_SHA256
        or not lines
        or lines[0] != EXPECTED_STATIC_FFMPEG_VERSION_FIRST_LINE
        or "static" not in lines[0]
    ):
        die("static ffmpeg version contract differs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--base-launcher", required=True)
    parser.add_argument("--base-launcher-sha256", required=True)
    parser.add_argument("--materializer", required=True)
    parser.add_argument("--materializer-sha256", required=True)
    parser.add_argument("--effective-launcher", required=True)
    parser.add_argument("--effective-launcher-sha256", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--rendezvous-guard", required=True)
    parser.add_argument("--rendezvous-guard-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--event-spec", required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--static-ffmpeg", required=True)
    parser.add_argument("--static-ffmpeg-sha256", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canary-receipt", required=True)
    parser.add_argument("--canary-submission-receipt", required=True)
    parser.add_argument("--retained-fd-canary-admission", required=True)
    parser.add_argument("--probe-validator", required=True)
    parser.add_argument("--probe-validator-sha256", required=True)
    parser.add_argument("--compute-bash-probe-admission", required=True)
    parser.add_argument("--compute-bash-probe-admission-sha256", required=True)
    parser.add_argument("--compute-bash-probe-admission-digest", required=True)
    parser.add_argument("--compute-bash", required=True)
    parser.add_argument("--compute-bash-sha256", required=True)
    parser.add_argument("--compute-bash-version-stdout-sha256", required=True)
    parser.add_argument("--compute-bash-version-first-line", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--slurm-log-dir", required=True)
    return parser


def require_pin(actual: str, expected: str, label: str) -> None:
    if actual != expected or SHA256.fullmatch(expected) is None:
        die(f"{label} SHA pin differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_pin(args.wrapper_sha256, EXPECTED_WRAPPER_SHA256, "formal wrapper")
    require_pin(
        args.base_launcher_sha256, EXPECTED_BASE_LAUNCHER_SHA256, "base launcher"
    )
    require_pin(
        args.materializer_sha256, EXPECTED_MATERIALIZER_SHA256, "materializer"
    )
    require_pin(
        args.effective_launcher_sha256,
        EXPECTED_EFFECTIVE_LAUNCHER_SHA256,
        "effective launcher",
    )
    require_pin(args.gate_sha256, EXPECTED_GATE_SHA256, "formal gate")
    require_pin(
        args.rendezvous_guard_sha256, EXPECTED_GUARD_V2_SHA256, "guard v2"
    )
    require_pin(
        args.probe_validator_sha256,
        EXPECTED_PROBE_VALIDATOR_SHA256,
        "probe admission validator",
    )
    require_pin(
        args.compute_bash_probe_admission_sha256,
        EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256,
        "compute-Bash probe admission",
    )
    require_pin(
        args.compute_bash_probe_admission_digest,
        EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_DIGEST,
        "compute-Bash probe admission digest",
    )
    require_pin(
        args.source_archive_sha256,
        EXPECTED_SOURCE_ARCHIVE_SHA256,
        "source archive",
    )
    require_pin(
        args.source_manifest_sha256,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "source manifest",
    )
    require_pin(args.event_spec_sha256, EXPECTED_EVENT_SPEC_SHA256, "event spec")
    require_pin(
        args.checkpoint_manifest_sha256,
        EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint manifest",
    )
    require_pin(args.python_sha256, EXPECTED_PYTHON_SHA256, "Python")
    require_pin(
        args.static_ffmpeg_sha256,
        EXPECTED_STATIC_FFMPEG_SHA256,
        "static ffmpeg",
    )
    require_pin(
        args.compute_bash_sha256,
        EXPECTED_COMPUTE_BASH_SHA256,
        "compute bash",
    )
    require_pin(
        args.compute_bash_version_stdout_sha256,
        EXPECTED_COMPUTE_BASH_VERSION_STDOUT_SHA256,
        "compute bash version stdout",
    )
    if args.source_revision != EXPECTED_SOURCE_REVISION:
        die("source revision pin differs")
    if Path(args.canary_receipt) != EXPECTED_CANARY_RECEIPT:
        die("canary terminal receipt path differs")
    if Path(args.canary_submission_receipt) != EXPECTED_CANARY_SUBMISSION_RECEIPT:
        die("canary submission receipt path differs")
    if (
        Path(args.retained_fd_canary_admission)
        != EXPECTED_RETAINED_FD_CANARY_ADMISSION
    ):
        die("retained-FD canary admission path differs")
    if (
        Path(args.compute_bash_probe_admission)
        != EXPECTED_COMPUTE_BASH_PROBE_ADMISSION
    ):
        die("compute-Bash probe admission path differs")
    if args.compute_bash_version_first_line != EXPECTED_COMPUTE_BASH_VERSION_FIRST_LINE:
        die("compute bash version first line differs")
    fixed_paths = {
        "wrapper": (Path(args.wrapper), EXPECTED_WRAPPER),
        "base launcher": (Path(args.base_launcher), EXPECTED_BASE_LAUNCHER),
        "materializer": (Path(args.materializer), EXPECTED_MATERIALIZER),
        "effective launcher": (
            Path(args.effective_launcher), EXPECTED_EFFECTIVE_LAUNCHER
        ),
        "formal gate": (Path(args.gate), EXPECTED_GATE),
        "guard v2": (Path(args.rendezvous_guard), EXPECTED_GUARD_V2),
        "probe admission validator": (
            Path(args.probe_validator), EXPECTED_PROBE_VALIDATOR
        ),
        "compute-Bash probe admission": (
            Path(args.compute_bash_probe_admission),
            EXPECTED_COMPUTE_BASH_PROBE_ADMISSION,
        ),
        "source archive": (Path(args.source_archive), EXPECTED_SOURCE_ARCHIVE),
        "source manifest": (
            Path(args.source_manifest), EXPECTED_SOURCE_MANIFEST
        ),
        "event spec": (Path(args.event_spec), EXPECTED_EVENT_SPEC),
        "checkpoint manifest": (
            Path(args.checkpoint_manifest), EXPECTED_CHECKPOINT_MANIFEST
        ),
        "retained-FD canary admission": (
            Path(args.retained_fd_canary_admission),
            EXPECTED_RETAINED_FD_CANARY_ADMISSION,
        ),
        "compute bash": (Path(args.compute_bash), EXPECTED_COMPUTE_BASH),
        "output root": (Path(args.output_root), EXPECTED_OUTPUT_ROOT),
        "submission receipt": (Path(args.receipt), EXPECTED_SUBMISSION_RECEIPT),
        "Slurm log directory": (Path(args.slurm_log_dir), EXPECTED_SLURM_LOG_DIR),
    }
    for label, (actual, expected) in fixed_paths.items():
        if actual != expected:
            die(f"{label} hard-pinned path differs")

    wrapper = exact_file(args.wrapper, args.wrapper_sha256, "formal wrapper")
    base_launcher = exact_file(
        args.base_launcher, args.base_launcher_sha256, "frozen base launcher"
    )
    materializer = exact_file(
        args.materializer, args.materializer_sha256, "launcher materializer"
    )
    effective_launcher = exact_file(
        args.effective_launcher,
        args.effective_launcher_sha256,
        "effective launcher",
    )
    gate = exact_file(args.gate, args.gate_sha256, "formal gate")
    guard = exact_file(
        args.rendezvous_guard, args.rendezvous_guard_sha256, "guard v2"
    )
    probe_validator = exact_file(
        args.probe_validator,
        args.probe_validator_sha256,
        "compute-Bash probe admission validator",
    )
    compute_bash_probe_admission = exact_file(
        args.compute_bash_probe_admission,
        args.compute_bash_probe_admission_sha256,
        "compute-Bash probe admission",
    )
    archive = exact_file(
        args.source_archive, args.source_archive_sha256, "source archive"
    )
    source_manifest = exact_file(
        args.source_manifest, args.source_manifest_sha256, "source manifest"
    )
    event_spec = exact_file(args.event_spec, args.event_spec_sha256, "event spec")
    checkpoint_manifest = exact_file(
        args.checkpoint_manifest,
        args.checkpoint_manifest_sha256,
        "checkpoint manifest",
    )
    retained_fd_canary_admission = exact_file(
        args.retained_fd_canary_admission,
        EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256,
        "retained-FD canary admission",
    )
    python_bin = exact_executable(
        args.python, EXPECTED_PYTHON, args.python_sha256, "Python"
    )
    ffmpeg_bin = exact_executable(
        args.static_ffmpeg,
        EXPECTED_STATIC_FFMPEG,
        args.static_ffmpeg_sha256,
        "static ffmpeg",
    )
    compute_bash = exact_executable(
        args.compute_bash,
        EXPECTED_COMPUTE_BASH,
        args.compute_bash_sha256,
        "compute bash",
    )
    validate_static_ffmpeg(ffmpeg_bin)
    bernini_root = exact_directory(
        args.bernini_root, EXPECTED_BERNINI_ROOT, "Bernini root"
    )
    veomni_root = exact_directory(
        args.veomni_root, EXPECTED_VEOMNI_ROOT, "VeOmni root"
    )
    checkpoint = exact_directory(
        args.checkpoint, EXPECTED_CHECKPOINT, "checkpoint"
    )

    archived = archive_payloads(archive, args.source_revision)
    if source_manifest.read_bytes() != archived[ARCHIVE_SOURCE_MANIFEST]:
        die("external source manifest differs from archive")
    if event_spec.read_bytes() != archived[ARCHIVE_EVENT_SPEC]:
        die("external event spec differs from archive")
    validate_materialization(materializer, base_launcher, effective_launcher)
    gate_module, canary = validate_canary(gate, guard)
    sacct_observation = observe_canary_sacct(gate_module)
    probe_binding = gate_module.validate_compute_bash_probe_admission(
        gate_module.load_probe_validator(
            probe_validator, EXPECTED_PROBE_VALIDATOR_SHA256,
        ),
        compute_bash_probe_admission,
    )
    retained_fd_canary = gate_module.validate_retained_fd_canary(
        gate_module.load_guard(guard, EXPECTED_GUARD_V2_SHA256),
        retained_fd_canary_admission,
        probe_binding,
    )
    gate_module.validate_compute_bash()
    retained_fd_canary_sacct = gate_module.observe_retained_fd_canary_sacct(
        "submitter_before_formal_sbatch"
    )
    canary_admission = {
        **canary,
        "sacct_observation": sacct_observation,
        "compute_bash_probe_admission": probe_binding,
        "retained_fd_world8": {
            **retained_fd_canary,
            "submitter_sacct_observation": retained_fd_canary_sacct,
        },
    }

    output_root = Path(args.output_root)
    if not output_root.is_absolute() or output_root == Path("/"):
        die("output root identity differs")
    output_parent = exact_directory(str(output_root.parent), None, "output parent")
    if output_root != output_parent / output_root.name or SAFE_NAME.fullmatch(
        output_root.name
    ) is None:
        die("output root is not canonical")
    if output_root.exists() or output_root.is_symlink():
        die("output root is not fresh")
    receipt = Path(args.receipt)
    if receipt != Path(str(output_root) + ".submission.receipt.json"):
        die("submission receipt path differs")
    if receipt.exists() or receipt.is_symlink():
        die("submission receipt is not fresh")
    slurm_log_dir = exact_directory(args.slurm_log_dir, None, "Slurm log directory")
    output_parent_identity = directory_identity(output_parent)
    log_dir_identity = directory_identity(slurm_log_dir)
    pre_reservation_siblings = {entry.name for entry in os.scandir(output_parent)}
    if receipt.name in pre_reservation_siblings or output_root.name in pre_reservation_siblings:
        die("fresh formal namespace differs")

    expected_inputs = {
        "wrapper": str(wrapper),
        "wrapper_sha256": args.wrapper_sha256,
        "base_launcher": str(base_launcher),
        "base_launcher_sha256": args.base_launcher_sha256,
        "materializer": str(materializer),
        "materializer_sha256": args.materializer_sha256,
        "effective_launcher": str(effective_launcher),
        "effective_launcher_sha256": args.effective_launcher_sha256,
        "gate": str(gate),
        "gate_sha256": args.gate_sha256,
        "rendezvous_guard": str(guard),
        "rendezvous_guard_sha256": args.rendezvous_guard_sha256,
        "retained_fd_canary_admission": str(retained_fd_canary_admission),
        "retained_fd_canary_admission_sha256":
            EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256,
        "retained_fd_canary_admission_digest":
            EXPECTED_RETAINED_FD_CANARY_ADMISSION_DIGEST,
        "retained_fd_canary_job_id": EXPECTED_RETAINED_FD_CANARY_JOB_ID,
        "probe_validator": str(probe_validator),
        "probe_validator_sha256": args.probe_validator_sha256,
        "compute_bash_probe_admission": str(compute_bash_probe_admission),
        "compute_bash_probe_admission_sha256":
            args.compute_bash_probe_admission_sha256,
        "compute_bash_probe_admission_digest":
            args.compute_bash_probe_admission_digest,
        "source_archive": str(archive),
        "source_archive_sha256": args.source_archive_sha256,
        "generation_runtime_sha256": EXPECTED_RUNTIME_SHA256,
        "archived_rendezvous_guard_v1_sha256":
            EXPECTED_ARCHIVED_GUARD_V1_SHA256,
        "base_v1_spec_sha256": EXPECTED_BASE_V1_SPEC_SHA256,
        "contract_runtime_sha256": EXPECTED_CONTRACT_RUNTIME_SHA256,
        "effective_dynamic_plan_schema_version":
            "saic-t2v-topup-rendezvous-dynamic-plan-v2",
        "scientific_spec_changed_for_rendezvous_guard_v2": False,
        "source_revision": args.source_revision,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": args.source_manifest_sha256,
        "event_spec": str(event_spec),
        "event_spec_sha256": args.event_spec_sha256,
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "python": str(python_bin),
        "python_sha256": args.python_sha256,
        "static_ffmpeg": str(ffmpeg_bin),
        "static_ffmpeg_sha256": args.static_ffmpeg_sha256,
        "static_ffmpeg_version_stdout_sha256":
            EXPECTED_STATIC_FFMPEG_VERSION_STDOUT_SHA256,
        "static_ffmpeg_version_first_line":
            EXPECTED_STATIC_FFMPEG_VERSION_FIRST_LINE,
        "compute_bash": str(compute_bash),
        "compute_bash_sha256": args.compute_bash_sha256,
        "compute_bash_version_stdout_sha256":
            args.compute_bash_version_stdout_sha256,
        "compute_bash_version_first_line":
            args.compute_bash_version_first_line,
        "bernini_root": str(bernini_root),
        "veomni_root": str(veomni_root),
        "checkpoint": str(checkpoint),
    }
    exports = {
        "SAIC_T2V_FV2_BASE_LAUNCHER": str(base_launcher),
        "SAIC_T2V_FV2_BASE_LAUNCHER_SHA256": args.base_launcher_sha256,
        "SAIC_T2V_FV2_MATERIALIZER": str(materializer),
        "SAIC_T2V_FV2_MATERIALIZER_SHA256": args.materializer_sha256,
        "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER": str(effective_launcher),
        "SAIC_T2V_FV2_EFFECTIVE_LAUNCHER_SHA256":
            args.effective_launcher_sha256,
        "SAIC_T2V_FV2_WRAPPER": str(wrapper),
        "SAIC_T2V_FV2_WRAPPER_SHA256": args.wrapper_sha256,
        "SAIC_T2V_FV2_GATE": str(gate),
        "SAIC_T2V_FV2_GATE_SHA256": args.gate_sha256,
        "SAIC_T2V_V4_EXTERNAL_RENDEZVOUS_GUARD": str(guard),
        "SAIC_T2V_V3_RENDEZVOUS_GUARD_SHA256": args.rendezvous_guard_sha256,
        "SAIC_T2V_FV2_CANARY_RECEIPT": str(EXPECTED_CANARY_RECEIPT),
        "SAIC_T2V_FV2_CANARY_SUBMISSION_RECEIPT":
            str(EXPECTED_CANARY_SUBMISSION_RECEIPT),
        "SAIC_T2V_FV2_RETAINED_FD_CANARY_ADMISSION":
            str(retained_fd_canary_admission),
        "SAIC_T2V_FV2_PROBE_VALIDATOR": str(probe_validator),
        "SAIC_T2V_FV2_PROBE_VALIDATOR_SHA256":
            args.probe_validator_sha256,
        "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION":
            str(compute_bash_probe_admission),
        "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_SHA256":
            args.compute_bash_probe_admission_sha256,
        "SAIC_T2V_FV2_COMPUTE_BASH_PROBE_ADMISSION_DIGEST":
            args.compute_bash_probe_admission_digest,
        "SAIC_T2V_FV2_OWN_SUBMISSION_RECEIPT": str(receipt),
        "SAIC_T2V_V3_SOURCE_ARCHIVE": str(archive),
        "SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256": args.source_archive_sha256,
        "SAIC_T2V_V3_SOURCE_REVISION": args.source_revision,
        "SAIC_T2V_V3_SOURCE_MANIFEST": str(source_manifest),
        "SAIC_T2V_V3_SOURCE_MANIFEST_SHA256": args.source_manifest_sha256,
        "SAIC_T2V_V3_EVENT_SPEC": str(event_spec),
        "SAIC_T2V_V3_EVENT_SPEC_SHA256": args.event_spec_sha256,
        "BERNINI_OFFICIAL_ROOT": str(bernini_root),
        "BERNINI_VEOMNI_ROOT": str(veomni_root),
        "BERNINI_ACTION_CHECKPOINT": str(checkpoint),
        "BERNINI_CHECKPOINT_CONTENT_MANIFEST": str(checkpoint_manifest),
        "SAIC_T2V_FV2_CHECKPOINT_MANIFEST_SHA256":
            args.checkpoint_manifest_sha256,
        "SAIC_T2V_V3_OUTPUT_ROOT": str(output_root),
        "SAIC_T2V_V3_PYTHON_BIN": str(python_bin),
        "SAIC_T2V_FV2_PYTHON_SHA256": args.python_sha256,
        "SAIC_T2V_V3_STATIC_FFMPEG": str(ffmpeg_bin),
        "SAIC_T2V_FV2_STATIC_FFMPEG_SHA256": args.static_ffmpeg_sha256,
        "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_STDOUT_SHA256":
            EXPECTED_STATIC_FFMPEG_VERSION_STDOUT_SHA256,
        "SAIC_T2V_FV2_STATIC_FFMPEG_VERSION_FIRST_LINE":
            EXPECTED_STATIC_FFMPEG_VERSION_FIRST_LINE,
        "SAIC_T2V_FV2_COMPUTE_BASH": str(compute_bash),
        "SAIC_T2V_FV2_COMPUTE_BASH_SHA256": args.compute_bash_sha256,
        "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_STDOUT_SHA256":
            args.compute_bash_version_stdout_sha256,
        "SAIC_T2V_FV2_SLURM_LOG_DIR": str(slurm_log_dir),
    }
    if list(exports) != gate_module.EXPORT_NAMES:
        die("formal exact export name/order differs from gate")
    if any(
        "," in name or "," in value or "\n" in name or "\n" in value
        for name, value in exports.items()
    ):
        die("sbatch export value differs")

    if not Path("/proc/self/fd").is_dir():
        die("Linux retained-fd wrapper transport is unavailable")
    wrapper_descriptor = os.open(wrapper, os.O_RDONLY | os.O_NOFOLLOW)
    wrapper_info = os.fstat(wrapper_descriptor)
    wrapper_identity = (wrapper_info.st_dev, wrapper_info.st_ino)
    if (
        not stat.S_ISREG(wrapper_info.st_mode)
        or wrapper_info.st_nlink != 1
        or stat.S_IMODE(wrapper_info.st_mode) != 0o444
        or digest_descriptor(wrapper_descriptor) != EXPECTED_WRAPPER_SHA256
    ):
        os.close(wrapper_descriptor)
        die("retained formal wrapper bytes differ")

    descriptor = os.open(
        receipt, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    reservation = os.fstat(descriptor)
    reservation_identity = (reservation.st_dev, reservation.st_ino)
    if (
        not stat.S_ISREG(reservation.st_mode)
        or reservation.st_nlink != 1
        or stat.S_IMODE(reservation.st_mode) != 0o600
    ):
        os.close(wrapper_descriptor)
        os.close(descriptor)
        die("submission reservation identity differs")
    provisional = {
        "schema_version": "saic-t2v-topup-r6-formal-v2-r2-submission-v1",
        "status": "reserved_before_sbatch_not_submission_success",
        "submission_success": False,
        "job_success": None,
        "wrapper_sha256": args.wrapper_sha256,
        "effective_launcher_sha256": args.effective_launcher_sha256,
        "guard_v2_sha256": args.rendezvous_guard_sha256,
        "source_archive_sha256": args.source_archive_sha256,
        "canary_job_id": gate_module.CANARY_JOB_ID,
        "retained_fd_canary_job_id": EXPECTED_RETAINED_FD_CANARY_JOB_ID,
        "retained_fd_canary_admission_sha256":
            EXPECTED_RETAINED_FD_CANARY_ADMISSION_SHA256,
        "compute_bash_probe_job_id": EXPECTED_COMPUTE_BASH_PROBE_JOB_ID,
        "compute_bash_probe_admission_sha256":
            EXPECTED_COMPUTE_BASH_PROBE_ADMISSION_SHA256,
    }
    write_all(descriptor, canonical(provisional) + b"\n")
    os.fsync(descriptor)
    fsync_directory(output_parent)
    public_before_sbatch = receipt.lstat()
    current_siblings = {entry.name for entry in os.scandir(output_parent)}
    retained_before_sbatch = os.fstat(wrapper_descriptor)
    if (
        current_siblings != pre_reservation_siblings | {receipt.name}
        or output_root.exists()
        or output_root.is_symlink()
        or directory_identity(output_parent) != output_parent_identity
        or directory_identity(slurm_log_dir) != log_dir_identity
        or receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public_before_sbatch.st_mode)
        or stat.S_ISLNK(public_before_sbatch.st_mode)
        or public_before_sbatch.st_nlink != 1
        or stat.S_IMODE(public_before_sbatch.st_mode) != 0o600
        or (public_before_sbatch.st_dev, public_before_sbatch.st_ino)
        != reservation_identity
        or not stat.S_ISREG(retained_before_sbatch.st_mode)
        or retained_before_sbatch.st_nlink != 1
        or stat.S_IMODE(retained_before_sbatch.st_mode) != 0o444
        or (retained_before_sbatch.st_dev, retained_before_sbatch.st_ino)
        != wrapper_identity
        or digest_descriptor(wrapper_descriptor) != EXPECTED_WRAPPER_SHA256
    ):
        os.close(wrapper_descriptor)
        os.close(descriptor)
        die("pre-sbatch formal boundary differs")

    command = [
        "/usr/bin/sbatch",
        "--parsable",
        f"--output={slurm_log_dir}/saic-t2v-topup-r6-v2-r2-%j.out",
        f"--error={slurm_log_dir}/saic-t2v-topup-r6-v2-r2-%j.err",
        "--export=NONE," + ",".join(
            f"{name}={value}" for name, value in exports.items()
        ),
        f"/proc/self/fd/{wrapper_descriptor}",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            pass_fds=(wrapper_descriptor,),
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    finally:
        try:
            os.close(wrapper_descriptor)
        except OSError:
            pass
    try:
        stdout_text = completed.stdout.decode("ascii")
    except UnicodeDecodeError:
        os.close(descriptor)
        die("sbatch stdout is not ASCII")
    match = re.fullmatch(r"([0-9]+)(?:;([^\n;]+))?\n?", stdout_text)
    if completed.returncode != 0 or match is None:
        os.close(descriptor)
        die(
            "sbatch failed; 0600 reservation retained: "
            f"exit={completed.returncode} "
            f"stderr_sha256={digest_bytes(completed.stderr)}"
        )
    job_id = match.group(1)

    submitted_provisional = {
        **provisional,
        "status": "sbatch_returned_job_id_receipt_not_terminal",
        "submitted_job_id": job_id,
        "sbatch_stdout_sha256": digest_bytes(completed.stdout),
        "sbatch_stderr_sha256": digest_bytes(completed.stderr),
    }
    staged = canonical(submitted_provisional) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, staged)
    os.fsync(descriptor)

    public = receipt.lstat()
    if (
        directory_identity(output_parent) != output_parent_identity
        or directory_identity(slurm_log_dir) != log_dir_identity
        or output_parent.resolve(strict=True) != output_parent
        or receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(descriptor)
        die("post-sbatch submission reservation pathname differs")

    core = {
        "schema_version": "saic-t2v-topup-r6-formal-v2-r2-submission-v1",
        "status": "submitted",
        "submission_success": True,
        "job_success": None,
        "submitted_job": {
            "job_id": job_id,
            "cluster": match.group(2),
            "stdout_sha256": digest_bytes(completed.stdout),
            "stderr_sha256": digest_bytes(completed.stderr),
        },
        "request": {
            "job_name": "saic-t2v-topup-r6-v2-r2",
            "partition": "faculty",
            "qos": "bgqos",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "256G",
            "walltime": "24:00:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4_sp4",
            "candidate_count": 60,
            "dynamic_plan_schema_version":
                "saic-t2v-topup-rendezvous-dynamic-plan-v2",
            "fixed_candidate_set_and_order": True,
            "scientific_spec_changed_for_rendezvous_guard_v2": False,
            "hold": False,
            "dependency": None,
        },
        "submission_boundary": {
            "environment_replaced": True,
            "exact_job_export_names": list(exports),
            "export_all": False,
            "reservation_created_before_sbatch": True,
            "same_inode_retained": True,
            "launcher_submitted_from_retained_fd": True,
            "runtime_retained_fd_admission_roots": [
                "formal_gate", "effective_launcher", "rendezvous_guard_v2",
                "compute_bash_probe_validator",
            ],
            "pathname_replacement_resistant_admission_handoff": True,
            "compute_bash_exact_identity_pinned": True,
            "varredir_close_option_required": False,
            "reservation_device": reservation.st_dev,
            "reservation_inode": reservation.st_ino,
            "success_mode": "0444",
        },
        "inputs": expected_inputs,
        "canary_admission": canary_admission,
        "outputs": {
            "output_root": str(output_root),
            "submission_receipt": str(receipt),
            "slurm_log_dir": str(slurm_log_dir),
            "fresh_before_submission": True,
        },
        "authority": {
            "diagnostic_event_bank_execution_authorized": True,
            "training": False,
            "checkpoint": False,
            "scientific_success_claimed": False,
            "action_edit_success_claimed": False,
            "job_success_claimed": False,
        },
        "threat_model": gate_module.THREAT_MODEL,
    }
    value = {**core, "receipt_digest": digest_bytes(canonical(core))}
    payload = canonical(value) + b"\n"
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    write_all(descriptor, payload)
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.read(descriptor, len(payload) + 1) != payload:
        os.close(descriptor)
        die("submission receipt reread differs before terminal publication")
    public = receipt.lstat()
    if (
        receipt.resolve(strict=True) != receipt
        or not stat.S_ISREG(public.st_mode)
        or stat.S_ISLNK(public.st_mode)
        or public.st_nlink != 1
        or stat.S_IMODE(public.st_mode) != 0o600
        or public.st_size != len(payload)
        or (public.st_dev, public.st_ino) != reservation_identity
    ):
        os.close(descriptor)
        die("submission receipt pathname differs before terminal publication")
    fsync_directory(output_parent)
    os.close(descriptor)

    publisher = os.open(receipt, os.O_RDWR | os.O_NOFOLLOW)
    try:
        observed = os.fstat(publisher)
        public = receipt.lstat()
        if (
            directory_identity(output_parent) != output_parent_identity
            or directory_identity(slurm_log_dir) != log_dir_identity
            or output_parent.resolve(strict=True) != output_parent
            or receipt.resolve(strict=True) != receipt
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_size != len(payload)
            or (observed.st_dev, observed.st_ino) != reservation_identity
            or not stat.S_ISREG(public.st_mode)
            or stat.S_ISLNK(public.st_mode)
            or public.st_nlink != 1
            or stat.S_IMODE(public.st_mode) != 0o600
            or (public.st_dev, public.st_ino) != reservation_identity
        ):
            die("public submission reservation differs")
        os.lseek(publisher, 0, os.SEEK_SET)
        if os.read(publisher, len(payload) + 1) != payload:
            die("public submission payload differs")
    except BaseException:
        os.close(publisher)
        raise
    os.fchmod(publisher, 0o444)
    os.fsync(publisher)
    sealed = os.fstat(publisher)
    public = receipt.lstat()
    if (
        not stat.S_ISREG(sealed.st_mode)
        or sealed.st_nlink != 1
        or stat.S_IMODE(sealed.st_mode) != 0o444
        or sealed.st_size != len(payload)
        or (sealed.st_dev, sealed.st_ino) != reservation_identity
        or (public.st_dev, public.st_ino) != reservation_identity
        or stat.S_IMODE(public.st_mode) != 0o444
    ):
        os.close(publisher)
        die("sealed formal submission receipt differs")
    fsync_directory(output_parent)
    try:
        os.close(publisher)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
