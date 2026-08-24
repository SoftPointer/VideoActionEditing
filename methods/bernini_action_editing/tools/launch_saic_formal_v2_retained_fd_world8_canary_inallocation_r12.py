#!/usr/bin/env python3
"""Launch one zero-science WORLD8 canary step inside allocation 134936.

The login-side process reserves one 0600 receipt with O_EXCL before invoking a
single full-8-GPU srun.  The compute-side bootstrap opens that same receipt and
the immutable wrapper with O_NOFOLLOW, seals the step identity, and executes
the wrapper through its retained descriptor.  It never creates another job.
"""

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
PARENT_ALLOCATION_JOB_ID = "134936"
EXPECTED_NODE = "auh7-1b-gpu-185"
STEP_RECEIPT_SCHEMA = "saic-formal-v2-retained-fd-world8-inallocation-r12-step-launch-v1"
PROVISIONAL_SCHEMA = STEP_RECEIPT_SCHEMA + "-reservation"
STEM = "saic-formal-v2-retained-fd-world8-canary-inallocation-r12"
ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809"
)
RELEASE = ROOT / "releases" / STEM
INPUTS = RELEASE / "inputs"
POSTFLIGHT_ROOT = RELEASE / "postflight"
OUTPUT_PARENT = ROOT / "canaries" / STEM
LOG_DIR = ROOT / "slurm" / STEM
RECEIPT = OUTPUT_PARENT / "inallocation-r12-step-launch-receipt.json"
CLIENT_RECEIPT = OUTPUT_PARENT / "inallocation-r12-srun-client-receipt.json"
LAUNCHER = INPUTS / "launch_saic_formal_v2_retained_fd_world8_canary_inallocation_r12.py"
WRAPPER = INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_inallocation_r12.sh"
PAYLOAD = INPUTS / "auh_canary_saic_formal_v2_retained_fd_world8_payload_inallocation_r12.sh"
GUARD = INPUTS / "saic_t2v_rendezvous_guard_v2.py"
RUNTIME = INPUTS / "generate_saic_pure_t2v_event_bank_topup_v2.py"
SOURCE_ARCHIVE = INPUTS / "videoedit-saic-20c2193-methods.tar"
PROBE_VALIDATOR = INPUTS / "probe_admission_binding_v1.py"
POSTFLIGHT = POSTFLIGHT_ROOT / "postflight_saic_formal_v2_retained_fd_world8_canary_inallocation_r12.py"
RELEASE_MANIFEST = RELEASE / "release-manifest.json"
PYTHON = Path("/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12")
COMPUTE_BOOTSTRAP_PYTHON = Path("/usr/bin/python3")
SRUN = Path("/usr/bin/srun")

# Filled only after the payload, wrapper, and postflight are frozen.
EXPECTED_WRAPPER_SHA256 = "cd179e3e1a141c971765e9cdfb5f25ffbc22f784bde0c2fc23ccfb617a44771d"
EXPECTED_PAYLOAD_SHA256 = "4b30a3fd8cdce12f24b86e499274c64734c969d0c425e2232859449b5800d3df"
EXPECTED_POSTFLIGHT_SHA256 = "ac462c800d2954bf4ba841b7b90a530c1d9dd0c93ef7f85bb01c38d3246c0cb8"
EXPECTED_GUARD_SHA256 = "1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965"
EXPECTED_RUNTIME_SHA256 = "3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36"
EXPECTED_SOURCE_ARCHIVE_SHA256 = "3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b"
EXPECTED_PROBE_VALIDATOR_SHA256 = "3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b"
EXPECTED_PYTHON_SHA256 = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
AUTHORITY = {
    "scientific": False,
    "generation": False,
    "training": False,
    "publication": False,
    "formal_job_authorized": False,
}
RELEASE_MANIFEST_FIELDS = {
    "schema_version", "status", "stem", "release_root", "output_parent",
    "log_directory", "parent_allocation_job_id", "expected_node", "inputs",
    "postflight", "immutable_ancestor", "executables", "probe_admission",
    "authority", "receipt_digest",
}


def die(message: str) -> None:
    raise SystemExit(f"launch-saic-fv2-fd-world8-inallocation-r12: {message}")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def retained_file(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor); leaf = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not stat.S_ISREG(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
            or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino)
        ):
            die(f"{label} retained identity differs")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks); after = os.fstat(descriptor)
        leaf_after = path.lstat()
        identity = lambda info: (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            stat.S_IMODE(info.st_mode), info.st_nlink, info.st_uid,
        )
        if (
            identity(before) != identity(after) or after.st_size != len(raw)
            or (after.st_dev, after.st_ino)
            != (leaf_after.st_dev, leaf_after.st_ino)
        ):
            die(f"{label} changed during retained read")
        return raw, after
    finally:
        os.close(descriptor)


def sha_file(path: Path) -> str:
    raw, _ = retained_file(path, str(path))
    return hashlib.sha256(raw).hexdigest()


def exact_file(
    path: Path, expected_sha: str, label: str, *, executable: bool = False,
) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    raw, info = retained_file(path, label)
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (not executable and stat.S_IMODE(info.st_mode) != 0o444)
        or (executable and (stat.S_IMODE(info.st_mode) & 0o022) != 0)
        or (executable and not os.access(path, os.X_OK))
        or SHA256.fullmatch(expected_sha) is None
        or hashlib.sha256(raw).hexdigest() != expected_sha
    ):
        die(f"{label} bytes/mode differ")
    return path


def exact_directory(path: Path, expected: Path, label: str) -> Path:
    if path != expected or not path.is_absolute() or path.resolve(strict=True) != path:
        die(f"{label} identity differs")
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700
    ):
        die(f"{label} owner/mode differ")
    return path


def retain_directory(path: Path, label: str) -> tuple[int, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    leaf = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or not stat.S_ISDIR(leaf.st_mode) or stat.S_ISLNK(leaf.st_mode)
        or (info.st_dev, info.st_ino) != (leaf.st_dev, leaf.st_ino)
    ):
        os.close(descriptor)
        die(f"{label} retained identity differs")
    return descriptor, info


def write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        wrote = os.write(descriptor, view)
        if wrote <= 0:
            die("receipt write stalled")
        view = view[wrote:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


BOOTSTRAP = r'''import hashlib,json,os,re,stat,sys
from pathlib import Path

def fail(message):
    raise SystemExit("inallocation bootstrap: " + message)
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def read_fd(fd):
    os.lseek(fd,0,os.SEEK_SET); chunks=[]
    while True:
        chunk=os.read(fd,1024*1024)
        if not chunk: break
        chunks.append(chunk)
    os.lseek(fd,0,os.SEEK_SET); return b"".join(chunks)
def write_all(fd,raw):
    view=memoryview(raw)
    while view:
        wrote=os.write(fd,view)
        if wrote<=0: fail("receipt write stalled")
        view=view[wrote:]

(wrapper_path,wrapper_sha,receipt_path,output_parent,log_directory,release_manifest,
 release_manifest_sha,release_manifest_digest,parent_job,expected_node,
 expected_schema,provisional_schema,expected_bootstrap_sha,expected_bootstrap_size,payload,payload_sha,guard,guard_sha,
 runtime,runtime_sha,source_archive,source_archive_sha,python,python_sha,
 postflight,postflight_sha,probe_validator,probe_validator_sha,
 probe_admission,probe_admission_sha,probe_admission_digest)=sys.argv[1:]
if os.environ.get("SLURM_JOB_ID") != parent_job: fail("parent allocation differs")
step_id=os.environ.get("SLURM_STEP_ID","")
if re.fullmatch(r"[0-9]+",step_id) is None: fail("step id differs")
job_step_id=parent_job+"."+step_id
node=os.environ.get("SLURMD_NODENAME","")
if node != expected_node: fail("compute node differs")
if os.environ.get("SLURM_JOB_NODELIST") != expected_node: fail("node list differs")

wrapper_fd=os.open(wrapper_path,os.O_RDONLY|os.O_NOFOLLOW)
receipt_fd=os.open(receipt_path,os.O_RDWR|os.O_NOFOLLOW)
python_fd=os.open(python,os.O_RDONLY|os.O_NOFOLLOW)
try:
    wi=os.fstat(wrapper_fd); ri=os.fstat(receipt_fd); pi=os.fstat(python_fd)
    wrapper_leaf=Path(wrapper_path).lstat(); receipt_leaf=Path(receipt_path).lstat()
    python_leaf=Path(python).lstat()
    if (not stat.S_ISREG(wi.st_mode) or wi.st_nlink!=1 or stat.S_IMODE(wi.st_mode)!=0o444
            or (wi.st_dev,wi.st_ino)!=(wrapper_leaf.st_dev,wrapper_leaf.st_ino)
            or hashlib.sha256(read_fd(wrapper_fd)).hexdigest()!=wrapper_sha):
        fail("retained wrapper differs")
    if (not stat.S_ISREG(ri.st_mode) or ri.st_nlink!=1 or stat.S_IMODE(ri.st_mode)!=0o600
            or (ri.st_dev,ri.st_ino)!=(receipt_leaf.st_dev,receipt_leaf.st_ino)):
        fail("reserved receipt differs")
    if (not stat.S_ISREG(pi.st_mode) or pi.st_nlink!=1
            or stat.S_IMODE(pi.st_mode)&0o022
            or (pi.st_dev,pi.st_ino)!=(python_leaf.st_dev,python_leaf.st_ino)
            or hashlib.sha256(read_fd(python_fd)).hexdigest()!=python_sha):
        fail("retained science Python differs")
    try: reservation=json.loads(read_fd(receipt_fd).decode("ascii"))
    except (UnicodeDecodeError,json.JSONDecodeError): fail("reservation encoding differs")
    claimed=reservation.pop("receipt_digest",None)
    if (set(reservation)!={"schema_version","status","parent_allocation_job_id",
            "parent_job_success","exact_srun_argv","exact_srun_argv_digest",
            "compute_bootstrap_sha256","compute_bootstrap_size_bytes",
            "release_manifest_digest","output_parent_identity",
            "log_directory_identity","authority"}
            or reservation.get("schema_version")!=provisional_schema
            or reservation.get("status")!="reserved_before_srun"
            or reservation.get("parent_allocation_job_id")!=parent_job
            or reservation.get("parent_job_success") is not None
            or reservation.get("compute_bootstrap_sha256")!=expected_bootstrap_sha
            or reservation.get("compute_bootstrap_size_bytes")!=int(expected_bootstrap_size)
            or reservation.get("release_manifest_digest")!=release_manifest_digest
            or re.fullmatch(r"[0-9]+:[0-9]+",reservation.get("output_parent_identity", "")) is None
            or re.fullmatch(r"[0-9]+:[0-9]+",reservation.get("log_directory_identity", "")) is None
            or reservation.get("authority")!={"scientific":False,"generation":False,"training":False,"publication":False,"formal_job_authorized":False}
            or claimed!=hashlib.sha256(canonical(reservation)).hexdigest()):
        fail("reservation seal differs")
    for directory,label,identity in (
            (output_parent,"output parent",reservation["output_parent_identity"]),
            (log_directory,"log directory",reservation["log_directory_identity"])):
        directory_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            directory_info=os.fstat(directory_fd); directory_leaf=Path(directory).lstat()
            if (not stat.S_ISDIR(directory_info.st_mode)
                    or stat.S_IMODE(directory_info.st_mode)!=0o700
                    or (directory_info.st_dev,directory_info.st_ino)
                    !=(directory_leaf.st_dev,directory_leaf.st_ino)
                    or f"{directory_info.st_dev}:{directory_info.st_ino}"!=identity):
                fail(label+" retained identity differs")
        finally:
            os.close(directory_fd)
    launch_argv=reservation.get("exact_srun_argv")
    argv_digest=reservation.get("exact_srun_argv_digest")
    if (not isinstance(launch_argv,list) or not all(isinstance(x,str) for x in launch_argv)
            or argv_digest!=hashlib.sha256(canonical(launch_argv)).hexdigest()):
        fail("srun argv binding differs")
    core={
      "schema_version":expected_schema,"status":"compute_step_bootstrap_admitted",
      "parent_allocation_job_id":parent_job,"step_id":step_id,
      "job_step_id":job_step_id,"node":node,
      "exact_srun_argv":launch_argv,"exact_srun_argv_digest":argv_digest,
      "compute_bootstrap_sha256":expected_bootstrap_sha,
      "compute_bootstrap_size_bytes":int(expected_bootstrap_size),
      "release_manifest":release_manifest,
      "release_manifest_file_sha256":release_manifest_sha,
      "release_manifest_digest":release_manifest_digest,
      "output_parent_identity":reservation["output_parent_identity"],
      "log_directory_identity":reservation["log_directory_identity"],
      "step_success":None,"parent_job_success":None,
      "bootstrap_boundary":{
        "receipt_reserved_before_srun":True,"receipt_same_inode":True,
        "receipt_opened_o_nofollow_inside_step":True,
        "wrapper_opened_o_nofollow_inside_step":True,
        "wrapper_executed_from_retained_fd":True,
        "compute_bootstrap_transported_over_srun_stdin":True,
        "compute_bootstrap_stdin_sha256_verified_inside_step":True,
        "compute_bootstrap_pathname_execution":False,
        "compute_bootstrap_interpreter":"/usr/bin/python3",
        "compute_bootstrap_interpreter_trust":"host_os_absolute_path",
        "science_python_opened_o_nofollow_inside_step":True,
        "science_python_retained_fd_prepared_for_wrapper":True,
        "receipt_success_mode":"0444"
      },
      "authority":{"scientific":False,"generation":False,"training":False,"publication":False,"formal_job_authorized":False},
    }
    value=dict(core); value["receipt_digest"]=hashlib.sha256(canonical(core)).hexdigest()
    raw=canonical(value)+b"\n"
    os.lseek(receipt_fd,0,os.SEEK_SET); os.ftruncate(receipt_fd,0); write_all(receipt_fd,raw)
    os.fsync(receipt_fd); os.lseek(receipt_fd,0,os.SEEK_SET)
    if os.read(receipt_fd,len(raw)+1)!=raw: fail("sealed receipt reread differs")
    os.fchmod(receipt_fd,0o444); os.set_inheritable(receipt_fd,True)
    os.set_inheritable(wrapper_fd,True)
    os.set_inheritable(python_fd,True)
    env={
      "PATH":"/usr/bin:/bin","LANG":"C","LC_ALL":"C",
      "SLURM_JOB_ID":parent_job,"SLURM_STEP_ID":step_id,
      "SLURM_JOB_NODELIST":os.environ["SLURM_JOB_NODELIST"],"SLURMD_NODENAME":node,
      "SLURM_LOCALID":os.environ.get("SLURM_LOCALID","0"),
      "SLURM_PROCID":os.environ.get("SLURM_PROCID","0"),
      "SLURM_NTASKS":os.environ.get("SLURM_NTASKS","1"),
      "SLURM_STEP_GPUS":os.environ.get("SLURM_STEP_GPUS",""),
      "SLURM_JOB_GPUS":os.environ.get("SLURM_JOB_GPUS",""),
      "ROCR_VISIBLE_DEVICES":os.environ.get("ROCR_VISIBLE_DEVICES",""),
      "HIP_VISIBLE_DEVICES":os.environ.get("HIP_VISIBLE_DEVICES",""),
      "SAIC_FV2_FD_CANARY_PAYLOAD":payload,"SAIC_FV2_FD_CANARY_PAYLOAD_SHA256":payload_sha,
      "SAIC_FV2_FD_CANARY_GUARD":guard,"SAIC_FV2_FD_CANARY_GUARD_SHA256":guard_sha,
      "SAIC_FV2_FD_CANARY_RUNTIME":runtime,"SAIC_FV2_FD_CANARY_RUNTIME_SHA256":runtime_sha,
      "SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE":source_archive,"SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256":source_archive_sha,
      "SAIC_FV2_FD_CANARY_PYTHON":python,"SAIC_FV2_FD_CANARY_PYTHON_SHA256":python_sha,
      "SAIC_FV2_FD_CANARY_OUTPUT_PARENT":output_parent,
      "SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT":receipt_path,
      "SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_DEVICE":str(ri.st_dev),
      "SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_INODE":str(ri.st_ino),
      "SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_FD_NUMBER":str(receipt_fd),
      "SAIC_FV2_FD_CANARY_BOOTSTRAP_WRAPPER_FD_NUMBER":str(wrapper_fd),
      "SAIC_FV2_FD_CANARY_BOOTSTRAP_PYTHON_FD_NUMBER":str(python_fd),
      "SAIC_FV2_FD_CANARY_WRAPPER":wrapper_path,"SAIC_FV2_FD_CANARY_WRAPPER_SHA256":wrapper_sha,
      "SAIC_FV2_FD_CANARY_POSTFLIGHT":postflight,"SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256":postflight_sha,
      "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST":release_manifest,
      "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256":release_manifest_sha,
      "SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST":release_manifest_digest,
      "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR":probe_validator,
      "SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256":probe_validator_sha,
      "SAIC_FV2_FD_CANARY_PROBE_ADMISSION":probe_admission,
      "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256":probe_admission_sha,
      "SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST":probe_admission_digest,
    }
    os.execve("/usr/bin/bash",["/usr/bin/bash",f"/proc/self/fd/{wrapper_fd}"],env)
finally:
    try: os.close(wrapper_fd)
    except OSError: pass
    try: os.close(receipt_fd)
    except OSError: pass
    try: os.close(python_fd)
    except OSError: pass
'''

STDIN_LOADER = (
    "import hashlib,sys;"
    "raw=sys.stdin.buffer.read();expected=sys.argv[1];expected_size=int(sys.argv[2]);"
    "(_ for _ in ()).throw(SystemExit('compute bootstrap stdin sha differs')) "
    "if hashlib.sha256(raw).hexdigest()!=expected or len(raw)!=expected_size else None;"
    "sys.argv=[sys.argv[0],*sys.argv[3:]];"
    "exec(compile(raw,'inallocation-r12-step-bootstrap','exec'),{'__name__':'__main__'})"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--release-manifest-sha256", required=True)
    value.add_argument("--release-manifest-digest", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = parser().parse_args(argv)
    sources = (
        (WRAPPER, EXPECTED_WRAPPER_SHA256, "wrapper"),
        (PAYLOAD, EXPECTED_PAYLOAD_SHA256, "payload"),
        (GUARD, EXPECTED_GUARD_SHA256, "guard"),
        (RUNTIME, EXPECTED_RUNTIME_SHA256, "runtime"),
        (SOURCE_ARCHIVE, EXPECTED_SOURCE_ARCHIVE_SHA256, "source archive"),
        (PROBE_VALIDATOR, EXPECTED_PROBE_VALIDATOR_SHA256, "probe validator"),
        (POSTFLIGHT, EXPECTED_POSTFLIGHT_SHA256, "postflight"),
    )
    for path, expected_sha, label in sources:
        exact_file(path, expected_sha, label)
    exact_file(PYTHON, EXPECTED_PYTHON_SHA256, "Python", executable=True)
    exact_directory(OUTPUT_PARENT, OUTPUT_PARENT, "output parent")
    exact_directory(LOG_DIR, LOG_DIR, "log directory")
    output_dir_fd, output_dir_before = retain_directory(
        OUTPUT_PARENT, "output parent"
    )
    log_dir_fd, log_dir_before = retain_directory(LOG_DIR, "log directory")
    manifest = exact_file(
        RELEASE_MANIFEST, args.release_manifest_sha256, "release manifest"
    )
    raw_manifest, _ = retained_file(manifest, "release manifest parse")
    try:
        manifest_value = json.loads(raw_manifest.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        die(f"release manifest encoding differs: {error}")
    unsigned = dict(manifest_value)
    claimed = unsigned.pop("receipt_digest", None)
    if (
        set(manifest_value) != RELEASE_MANIFEST_FIELDS
        or manifest_value.get("schema_version")
        != "saic-formal-v2-retained-fd-world8-inallocation-r12-release-manifest-v1"
        or manifest_value.get("status") != "sealed_before_inallocation_step"
        or claimed != args.release_manifest_digest
        or claimed != digest(unsigned)
        or raw_manifest != canonical(manifest_value) + b"\n"
        or manifest_value.get("stem") != STEM
        or manifest_value.get("output_parent") != str(OUTPUT_PARENT)
        or manifest_value.get("log_directory") != str(LOG_DIR)
        or manifest_value.get("parent_allocation_job_id")
        != PARENT_ALLOCATION_JOB_ID
        or manifest_value.get("expected_node") != EXPECTED_NODE
        or manifest_value.get("inputs") != {
            "guard": {"path": str(GUARD), "sha256": EXPECTED_GUARD_SHA256},
            "launcher": {"path": str(LAUNCHER), "sha256": sha_file(LAUNCHER)},
            "payload": {"path": str(PAYLOAD), "sha256": EXPECTED_PAYLOAD_SHA256},
            "probe_validator": {
                "path": str(PROBE_VALIDATOR),
                "sha256": EXPECTED_PROBE_VALIDATOR_SHA256,
            },
            "runtime": {"path": str(RUNTIME), "sha256": EXPECTED_RUNTIME_SHA256},
            "source_archive": {
                "path": str(SOURCE_ARCHIVE),
                "sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            },
            "wrapper": {"path": str(WRAPPER), "sha256": EXPECTED_WRAPPER_SHA256},
        }
        or manifest_value.get("postflight") != {
            "path": str(POSTFLIGHT), "sha256": EXPECTED_POSTFLIGHT_SHA256,
        }
        or manifest_value.get("immutable_ancestor") != {
            "release_root": str(ROOT / "releases/saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10"),
            "release_manifest_file_sha256": "a358a18e0b5ea497f1c6d99cbcfdf1b4b51229c5f596309c02a0b39bb51055ba",
            "guard_source_path": str(ROOT / "releases/saic-formal-v2-retained-fd-world8-canary-96335bf5-fb3f1ac4-r10/inputs/saic_t2v_rendezvous_guard_v2.py"),
            "guard_sha256": EXPECTED_GUARD_SHA256,
            "guard_copied_from_external_immutable_release": True,
            "local_guard_source_forbidden": True,
        }
        or manifest_value.get("executables") != {
            "python": str(PYTHON), "python_sha256": EXPECTED_PYTHON_SHA256,
            "compute_bootstrap_python": str(COMPUTE_BOOTSTRAP_PYTHON),
            "compute_bootstrap_python_trust": "host_os_absolute_path",
            "sacct": "/usr/bin/sacct",
            "sacct_sha256": "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e",
        }
        or manifest_value.get("probe_admission") != {
            "path": str(ROOT / "canaries/compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"),
            "file_sha256": "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8",
            "receipt_digest": "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7",
        }
        or manifest_value.get("authority") != AUTHORITY
    ):
        die("release manifest binding differs")
    if os.listdir(output_dir_fd):
        die("fresh output parent is not empty")
    if os.listdir(log_dir_fd):
        die("fresh log directory is not empty")
    # The output directory is still empty here; no step-specific namespace can
    # exist until compute reports the previously unknown SLURM_STEP_ID.
    try:
        os.stat(RECEIPT.name, dir_fd=output_dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        die("step receipt reservation is not fresh")

    bootstrap_args = [
        str(WRAPPER), EXPECTED_WRAPPER_SHA256, str(RECEIPT), str(OUTPUT_PARENT),
        str(LOG_DIR),
        str(RELEASE_MANIFEST), args.release_manifest_sha256,
        args.release_manifest_digest, PARENT_ALLOCATION_JOB_ID, EXPECTED_NODE,
        STEP_RECEIPT_SCHEMA, PROVISIONAL_SCHEMA,
        hashlib.sha256(BOOTSTRAP.encode("ascii")).hexdigest(),
        str(len(BOOTSTRAP.encode("ascii"))), str(PAYLOAD),
        EXPECTED_PAYLOAD_SHA256, str(GUARD), EXPECTED_GUARD_SHA256,
        str(RUNTIME), EXPECTED_RUNTIME_SHA256, str(SOURCE_ARCHIVE),
        EXPECTED_SOURCE_ARCHIVE_SHA256, str(PYTHON), EXPECTED_PYTHON_SHA256,
        str(POSTFLIGHT), EXPECTED_POSTFLIGHT_SHA256, str(PROBE_VALIDATOR),
        EXPECTED_PROBE_VALIDATOR_SHA256,
        str(ROOT / "canaries/compute-bash-retained-fd-probe-8283e73d-r1/probe-admission.json"),
        "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8",
        "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7",
    ]
    command = [
        str(SRUN), f"--jobid={PARENT_ALLOCATION_JOB_ID}", "--overlap",
        "--job-name=saic-fv2-fd-w8-inalloc-r12",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--mem=32G",
        "--gres=gpu:mi210:8", f"--nodelist={EXPECTED_NODE}", "--exact",
        "--kill-on-bad-exit=1", "--export=NONE", "--input=0",
        "--open-mode=truncate",
        f"--output={LOG_DIR}/saic-fv2-fd-w8-inalloc-r12-%J.out",
        f"--error={LOG_DIR}/saic-fv2-fd-w8-inalloc-r12-%J.err",
        str(COMPUTE_BOOTSTRAP_PYTHON), "-I", "-B", "-c", STDIN_LOADER,
        hashlib.sha256(BOOTSTRAP.encode("ascii")).hexdigest(),
        str(len(BOOTSTRAP.encode("ascii"))),
    ]
    command.extend(bootstrap_args)
    if len(" ".join(command).encode("ascii")) >= 8192:
        die("exact srun command exceeds admitted SubmitLine width")
    command_digest = digest(command)
    provisional_core = {
        "schema_version": PROVISIONAL_SCHEMA,
        "status": "reserved_before_srun",
        "parent_allocation_job_id": PARENT_ALLOCATION_JOB_ID,
        "parent_job_success": None,
        "exact_srun_argv": command,
        "exact_srun_argv_digest": command_digest,
        "compute_bootstrap_sha256": hashlib.sha256(
            BOOTSTRAP.encode("ascii")
        ).hexdigest(),
        "compute_bootstrap_size_bytes": len(BOOTSTRAP.encode("ascii")),
        "release_manifest_digest": args.release_manifest_digest,
        "output_parent_identity": (
            f"{output_dir_before.st_dev}:{output_dir_before.st_ino}"
        ),
        "log_directory_identity": f"{log_dir_before.st_dev}:{log_dir_before.st_ino}",
        "authority": AUTHORITY,
    }
    provisional = dict(provisional_core)
    provisional["receipt_digest"] = digest(provisional_core)
    receipt_fd = os.open(
        RECEIPT.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600, dir_fd=output_dir_fd,
    )
    try:
        before = os.fstat(receipt_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            die("receipt reservation identity differs")
        write_all(receipt_fd, canonical(provisional) + b"\n")
        os.fsync(receipt_fd)
        os.fsync(output_dir_fd)
        completed = subprocess.run(
            command, input=BOOTSTRAP.encode("ascii"), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=1200,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        after = os.fstat(receipt_fd)
        public = RECEIPT.lstat()
        output_dir_after = os.fstat(output_dir_fd)
        output_public = OUTPUT_PARENT.lstat()
        log_dir_after = os.fstat(log_dir_fd)
        log_public = LOG_DIR.lstat()
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (public.st_dev, public.st_ino) != (before.st_dev, before.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o444 or after.st_nlink != 1
            or (output_dir_after.st_dev, output_dir_after.st_ino)
            != (output_dir_before.st_dev, output_dir_before.st_ino)
            or (output_public.st_dev, output_public.st_ino)
            != (output_dir_before.st_dev, output_dir_before.st_ino)
            or (log_dir_after.st_dev, log_dir_after.st_ino)
            != (log_dir_before.st_dev, log_dir_before.st_ino)
            or (log_public.st_dev, log_public.st_ino)
            != (log_dir_before.st_dev, log_dir_before.st_ino)
        ):
            die("compute bootstrap did not seal same receipt inode")
        os.lseek(receipt_fd, 0, os.SEEK_SET)
        final_raw = os.read(receipt_fd, after.st_size + 1)
        final_value = json.loads(final_raw.decode("ascii"))
        final_unsigned = dict(final_value)
        final_claimed = final_unsigned.pop("receipt_digest", None)
        if (
            final_value.get("schema_version") != STEP_RECEIPT_SCHEMA
            or final_value.get("parent_allocation_job_id") != PARENT_ALLOCATION_JOB_ID
            or final_value.get("node") != EXPECTED_NODE
            or final_value.get("exact_srun_argv_digest") != command_digest
            or final_value.get("compute_bootstrap_sha256")
            != hashlib.sha256(BOOTSTRAP.encode("ascii")).hexdigest()
            or final_value.get("compute_bootstrap_size_bytes")
            != len(BOOTSTRAP.encode("ascii"))
            or final_value.get("output_parent_identity")
            != f"{output_dir_before.st_dev}:{output_dir_before.st_ino}"
            or final_value.get("log_directory_identity")
            != f"{log_dir_before.st_dev}:{log_dir_before.st_ino}"
            or final_value.get("step_success") is not None
            or final_value.get("parent_job_success") is not None
            or final_claimed != digest(final_unsigned)
            or final_raw != canonical(final_value) + b"\n"
        ):
            die("sealed step launch receipt differs")
        if completed.returncode != 0:
            die(f"srun step failed with exit {completed.returncode}")
        if completed.stdout or completed.stderr:
            die(
                "srun client streams differ: "
                f"stdout_sha256={hashlib.sha256(completed.stdout).hexdigest()} "
                f"stderr_sha256={hashlib.sha256(completed.stderr).hexdigest()}"
            )
        client_core = {
            "schema_version": (
                "saic-formal-v2-retained-fd-world8-inallocation-r12-srun-client-v1"
            ),
            "status": "srun_client_terminal_success",
            "parent_allocation_job_id": PARENT_ALLOCATION_JOB_ID,
            "job_step_id": final_value.get("job_step_id"),
            "exact_srun_argv_digest": command_digest,
            "step_launch_receipt_digest": final_value.get("receipt_digest"),
            "srun_client_returncode": completed.returncode,
            "srun_client_stdout_sha256": hashlib.sha256(
                completed.stdout
            ).hexdigest(),
            "srun_client_stdout_size": len(completed.stdout),
            "srun_client_stderr_sha256": hashlib.sha256(
                completed.stderr
            ).hexdigest(),
            "srun_client_stderr_size": len(completed.stderr),
            "parent_job_success": None,
            "authority": AUTHORITY,
        }
        client_value = dict(client_core)
        client_value["receipt_digest"] = digest(client_core)
        client_raw = canonical(client_value) + b"\n"
        client_fd = os.open(
            CLIENT_RECEIPT.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=output_dir_fd,
        )
        try:
            write_all(client_fd, client_raw)
            os.fsync(client_fd)
            os.fchmod(client_fd, 0o444)
            client_info = os.fstat(client_fd)
            client_public = os.stat(
                CLIENT_RECEIPT.name, dir_fd=output_dir_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(client_info.st_mode)
                or client_info.st_nlink != 1
                or stat.S_IMODE(client_info.st_mode) != 0o444
                or client_info.st_size != len(client_raw)
                or (client_info.st_dev, client_info.st_ino)
                != (client_public.st_dev, client_public.st_ino)
            ):
                die("srun client receipt retained seal differs")
            os.fsync(output_dir_fd)
        finally:
            os.close(client_fd)
    finally:
        os.close(receipt_fd)
        os.close(output_dir_fd)
        os.close(log_dir_fd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
