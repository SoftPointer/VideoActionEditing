#!/usr/bin/env python3
"""Materialize the GPU4-7 AUH-env root entry for the matched evaluation.

This file is a release builder, not the trusted entry itself.  It creates one
immutable Bash payload whose literal bytes must either be copied into the
Slurm controller spool or streamed unchanged to ``/bin/bash -p -s`` by a
trusted controller.  Executing the generated payload by its ordinary named
path is forbidden by the payload itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping, Sequence


SCHEMA = "full644-exploratory-matched-root-launch-release-gpu47-v3"
INPUT_SCHEMA = "full644-exploratory-matched-root-launch-input-gpu47-v3"
RECEIPT_SCHEMA = "full644-exploratory-matched-root-launch-receipt-gpu47-v3"
GPU_VISIBILITY_SCHEMA = "full644-exploratory-matched-gpu47-visibility-v1"
GPU_ADMISSION_SCHEMA = "full644-exploratory-matched-gpu47-admission-v1"
GPU_ADMISSION_MAX_AGE_NS = 10 * 60 * 1_000_000_000
FULL16_CAMPAIGN = "full16-production"
CASE00_CANARY_CAMPAIGN = "case00-pair-canary"
TASK_IDS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)
CANARY_TASK_IDS = TASK_IDS[:2]
ENTRY_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-captured-runner-entry-authority-v1"
)
EXPECTED_STATIC_SHA256 = {
    "runner": "d622ecb240ec87e787d98afdd46c310b940e77dd9511f2a7095cb539a72cdd86",
    "bridge": "494c8970e9f29af90bd359988cb80ab79106ec0a6228d53090945daa58fed7fb",
    "adapter": "e6f495f9a212ed5a61997bebbe229591d0d21cdf74e323af339e1f940ddd781b",
    "eval_v1": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "eval_v2": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "model_authority": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "torchrun_source": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    "torchrun_handler_source": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    "model_manifest": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
}
VARREDIR_CLOSE_COMPAT = (
    "if shopt -q varredir_close 2>/dev/null; then "
    "shopt -u varredir_close; fi"
)

_IDENTITY_FIELDS = {
    "device",
    "inode",
    "uid",
    "gid",
    "mode",
    "nlink",
    "rdev",
    "size",
    "blocks",
    "mtime_ns",
    "ctime_ns",
}


class RootLaunchReleaseError(RuntimeError):
    """The captured-source root release differs."""


def gpu47_visibility_contract() -> dict[str, Any]:
    return {
        "schema_version": GPU_VISIBILITY_SCHEMA,
        "visibility_variable": "ROCR_VISIBLE_DEVICES",
        "rocr_visible_devices": "4,5,6,7",
        "physical_gpu_indices": [4, 5, 6, 7],
        "logical_gpu_indices": [0, 1, 2, 3],
        "logical_to_physical_order_not_inferred_from_mask": True,
        "empirical_pci_uuid_admission_required": True,
        "secondary_visibility_variables_absent": True,
        "world_size": 4,
        "slurm_step_reserved_gpu_indices": list(range(8)),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise RootLaunchReleaseError("duplicate JSON key")
        value[key] = item
    return value


def _identity(info: os.stat_result) -> dict[str, int]:
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


def _stable_file(
    path_value: str | Path, *, executable: bool = False, return_raw: bool = False
) -> dict[str, Any]:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RootLaunchReleaseError(f"release file path differs: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or _identity(after) != identity
        or _identity(named) != identity
        or len(raw) != before.st_size
        or (executable and not before.st_mode & 0o111)
    ):
        raise RootLaunchReleaseError(f"release file identity differs: {path}")
    result = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "identity": identity,
    }
    if return_raw:
        result["_raw"] = raw
    return result


def _load_input(path_value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _stable_file(path_value, return_raw=True)
    path = Path(identity["path"])
    raw = identity.pop("_raw")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RootLaunchReleaseError("launch input is not strict JSON") from error
    fields = {
        "schema_version",
        "entry_mode",
        "runner",
        "bridge",
        "adapter",
        "eval_v1",
        "eval_v2",
        "model_authority",
        "python",
        "ffmpeg",
        "torchrun_source",
        "torchrun_handler_source",
        "plan",
        "output_report",
        "runner_attestation",
        "model_root",
        "model_manifest",
        "bernini_root",
        "veomni_root",
        "authority_root",
        "rank_cache_root",
        "holder_job_id",
        "expected_node",
        "campaign_mode",
        "gpu_admission_receipt",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema_version") != INPUT_SCHEMA
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("entry_mode") not in {"trusted_stdin", "slurm_spool"}
        or not isinstance(value.get("holder_job_id"), str)
        or not value["holder_job_id"]
        or not isinstance(value.get("expected_node"), str)
        or not value["expected_node"]
        or value.get("campaign_mode")
        not in {FULL16_CAMPAIGN, CASE00_CANARY_CAMPAIGN}
    ):
        raise RootLaunchReleaseError("launch input closure differs")
    return value, identity


def _canonical_path(raw: Any, *, exists: bool, directory: bool) -> Path:
    if not isinstance(raw, (str, Path)):
        raise RootLaunchReleaseError("launch path is not a string")
    path = Path(raw)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
    ):
        raise RootLaunchReleaseError(f"launch path differs: {path}")
    if exists:
        resolved = path.resolve(strict=True)
        if resolved != path or path.is_dir() is not directory:
            raise RootLaunchReleaseError(f"launch path type differs: {path}")
    elif path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise RootLaunchReleaseError(f"fresh launch path differs: {path}")
    return path


def _runner_arguments(value: Mapping[str, Any], identities: Mapping[str, Any]) -> list[str]:
    arguments = [
        "--campaign-mode", value["campaign_mode"],
        "--gpu-admission-receipt", value["gpu_admission_receipt"],
        "--gpu-admission-receipt-sha256", identities["gpu_admission_receipt"]["sha256"],
        "--plan", value["plan"],
        "--plan-sha256", identities["plan"]["sha256"],
        "--output-report", value["output_report"],
        "--runner-attestation", value["runner_attestation"],
        "--runner-sha256", identities["runner"]["sha256"],
        "--bridge-script", value["bridge"],
        "--bridge-script-sha256", identities["bridge"]["sha256"],
        "--adapter-script", value["adapter"],
        "--adapter-script-sha256", identities["adapter"]["sha256"],
        "--eval-v1-source", value["eval_v1"],
        "--eval-v1-source-sha256", identities["eval_v1"]["sha256"],
        "--eval-v2-source", value["eval_v2"],
        "--eval-v2-source-sha256", identities["eval_v2"]["sha256"],
        "--model-authority-source", value["model_authority"],
        "--model-authority-source-sha256", identities["model_authority"]["sha256"],
        "--python", value["python"],
        "--python-sha256", identities["python"]["sha256"],
        "--ffmpeg-executable", value["ffmpeg"],
        "--ffmpeg-executable-sha256", identities["ffmpeg"]["sha256"],
        "--torchrun-source", value["torchrun_source"],
        "--torchrun-source-sha256", identities["torchrun_source"]["sha256"],
        "--torchrun-handler-source", value["torchrun_handler_source"],
        "--torchrun-handler-source-sha256", identities["torchrun_handler_source"]["sha256"],
        "--model-root", value["model_root"],
        "--model-manifest", value["model_manifest"],
        "--model-manifest-sha256", identities["model_manifest"]["sha256"],
        "--bernini-root", value["bernini_root"],
        "--veomni-root", value["veomni_root"],
        "--authority-root", value["authority_root"],
        "--rank-cache-root", value["rank_cache_root"],
        "--holder-job-id", value["holder_job_id"],
        "--expected-node", value["expected_node"],
        "--expected-allocation-gpu-count", "8",
    ]
    if any(type(item) is not str or not item or "\x00" in item for item in arguments):
        raise RootLaunchReleaseError("runner argument differs")
    return arguments


def _validate_gpu_admission_header(
    raw: bytes,
    *,
    value: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        admission = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise RootLaunchReleaseError("GPU47 admission is not strict JSON") from error
    unsigned = dict(admission) if isinstance(admission, dict) else {}
    receipt_digest = unsigned.pop("receipt_digest", None)
    observed_at = admission.get("observed_at_unix_ns") if isinstance(admission, dict) else None
    now_ns = time.time_ns()
    if (
        not isinstance(admission, dict)
        or raw != canonical_json_bytes(admission) + b"\n"
        or admission.get("schema_version") != GPU_ADMISSION_SCHEMA
        or admission.get("status") != "PASS"
        or admission.get("holder_job_id") != value["holder_job_id"]
        or admission.get("node") != value["expected_node"]
        or admission.get("intended_campaign_mode") != value["campaign_mode"]
        or admission.get("intended_plan_sha256") != identities["plan"]["sha256"]
        or admission.get("single_use") is not True
        or type(observed_at) is not int
        or observed_at <= 0
        or now_ns < observed_at
        or now_ns - observed_at > GPU_ADMISSION_MAX_AGE_NS
        or admission.get("slurm_job_gpu_indices") != list(range(8))
        or admission.get("slurm_step_gpu_indices") != list(range(8))
        or admission.get("slurm_gpus_on_node") != 8
        or admission.get("rocr_visible_devices") != "4,5,6,7"
        or admission.get("secondary_visibility_variables_absent") is not True
        or admission.get("visible_physical_indices") != [4, 5, 6, 7]
        or admission.get("excluded_physical_indices") != [0, 1, 2, 3]
        or admission.get("device_cgroup_hard_isolation_claimed") is not False
        or admission.get("canary_or_model_execution_performed") is not False
        or admission.get("isolation_scope")
        != "ROCr-only logical isolation; not device-cgroup hard isolation"
        or receipt_digest != object_sha256(unsigned)
    ):
        raise RootLaunchReleaseError("GPU47 admission header closure differs")
    return admission


ROOT_BOOTSTRAP = r'''import os,sys
if len(sys.argv)!=15: raise RuntimeError("root bootstrap argv differs")
raw_fd,spec_raw,release_digest,bootstrap_sha,entry_mode,job_id,step_id,gpus_on_node,gpus_per_node,step_gpus,node_count,step_node_count,job_nodelist,step_nodelist=sys.argv[1:]
if not step_id.isascii() or not step_id.isdecimal() or int(step_id)<=0 or str(int(step_id))!=step_id: raise RuntimeError("numbered Slurm step differs")
try: python_fd=int(raw_fd)
except ValueError as error: raise RuntimeError("held Python FD differs") from error
if python_fd<3 or str(python_fd)!=raw_fd or os.get_inheritable(python_fd) is not True: raise RuntimeError("held Python FD entry differs")
python_before=os.fstat(python_fd); os.set_inheritable(python_fd,False)
if os.get_inheritable(python_fd): raise RuntimeError("held Python FD remained inheritable")
import hashlib,json,stat,types
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate root JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
def ident(value): return {"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"nlink":value.st_nlink,"rdev":value.st_rdev,"size":value.st_size,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
def read_fd(fd,size):
 chunks=[]; offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  chunks.append(block); offset+=len(block)
 raw=b"".join(chunks)
 if len(raw)!=size: raise RuntimeError("captured root read differs")
 return raw
spec=json.loads(spec_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
gpu_visibility={"empirical_pci_uuid_admission_required":True,"logical_gpu_indices":[0,1,2,3],"logical_to_physical_order_not_inferred_from_mask":True,"physical_gpu_indices":[4,5,6,7],"rocr_visible_devices":"4,5,6,7","schema_version":"full644-exploratory-matched-gpu47-visibility-v1","secondary_visibility_variables_absent":True,"slurm_step_reserved_gpu_indices":[0,1,2,3,4,5,6,7],"visibility_variable":"ROCR_VISIBLE_DEVICES","world_size":4}
campaign=spec.get("campaign_mode"); selected=spec.get("selected_task_ids"); expected=[f"shared8-{index:02d}-{arm}" for index in range(8) for arm in ("base","full644")] if campaign=="full16-production" else ["shared8-00-base","shared8-00-full644"] if campaign=="case00-pair-canary" else None
slurm_contract={"caller_synthesized_slurm_facts_forbidden":True,"required_absent_names":["SLURM_JOB_GPUS","SLURM_JOB_NUM_NODES"],"required_source_names":["SLURM_JOB_ID","SLURM_STEP_ID","SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_NODE","SLURM_STEP_GPUS","SLURM_NNODES","SLURM_STEP_NUM_NODES","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST"]}
if type(spec) is not dict or canonical(spec)!=spec_raw or digest(spec)!=release_digest or spec.get("schema_version")!="full644-exploratory-matched-root-launch-release-gpu47-v3" or spec.get("entry_mode")!=entry_mode or spec.get("gpu_visibility_contract")!=gpu_visibility or spec.get("isolation_scope")!="ROCr-only logical isolation; not device-cgroup hard isolation" or selected!=expected or spec.get("formal_full16_report")!=(campaign=="full16-production") or spec.get("canary_stops_after_pair_for_manual_visual_review")!=(campaign=="case00-pair-canary") or spec.get("slurm_environment_contract")!=slurm_contract or job_id!=spec.get("holder_job_id") or gpus_on_node!="8" or gpus_per_node!="8" or step_gpus!="0,1,2,3,4,5,6,7" or node_count!="1" or step_node_count!="1" or job_nodelist!=spec.get("expected_node") or step_nodelist!=spec.get("expected_node"): raise RuntimeError("root release digest differs")
python_row=spec.get("identities",{}).get("python"); runner_row=spec.get("identities",{}).get("runner")
admission_row=spec.get("identities",{}).get("gpu_admission_receipt"); admission=spec.get("empirical_gpu_admission")
if type(python_row) is not dict or type(runner_row) is not dict or type(admission_row) is not dict or type(admission) is not dict or set(admission)!={"identity","path","receipt_digest","runner_replays_before_model_open","sha256"} or admission.get("identity")!=admission_row.get("identity") or admission.get("path")!=admission_row.get("path") or admission.get("sha256")!=admission_row.get("sha256") or admission.get("runner_replays_before_model_open") is not True or type(admission.get("receipt_digest")) is not str or len(admission["receipt_digest"])!=64 or type(spec.get("runner_arguments")) is not list or any(type(value) is not str or not value for value in spec["runner_arguments"]) or spec["runner_arguments"][:4] != ["--campaign-mode",campaign,"--gpu-admission-receipt",admission_row.get("path")]: raise RuntimeError("root release rows differ")
python_path=python_row.get("path"); runner_path=runner_row.get("path")
if not os.path.isabs(python_path) or os.path.normpath(python_path)!=python_path or not os.path.isabs(runner_path) or os.path.normpath(runner_path)!=runner_path: raise RuntimeError("root release paths differ")
python_raw=read_fd(python_fd,python_before.st_size); python_after=os.fstat(python_fd); python_named=os.lstat(python_path); python_process=os.stat("/proc/self/exe")
if not stat.S_ISREG(python_before.st_mode) or python_before.st_nlink!=1 or not python_before.st_mode&0o111 or ident(python_before)!=ident(python_after) or ident(python_before)!=ident(python_named) or ident(python_before)!=ident(python_process) or ident(python_before)!=python_row.get("identity") or hashlib.sha256(python_raw).hexdigest()!=python_row.get("sha256"): raise RuntimeError("held Python identity differs")
flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0); runner_fd=os.open(runner_path,flags); os.set_inheritable(runner_fd,False)
runner_before=os.fstat(runner_fd); runner_raw=read_fd(runner_fd,runner_before.st_size); runner_after=os.fstat(runner_fd); runner_named=os.lstat(runner_path)
if not stat.S_ISREG(runner_before.st_mode) or runner_before.st_nlink!=1 or stat.S_IMODE(runner_before.st_mode)!=0o444 or ident(runner_before)!=ident(runner_after) or ident(runner_before)!=ident(runner_named) or ident(runner_before)!=runner_row.get("identity") or hashlib.sha256(runner_raw).hexdigest()!=runner_row.get("sha256"): raise RuntimeError("captured runner identity differs")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): raise RuntimeError("root Python environment differs")
entry={"schema_version":"full644-exploratory-matched-captured-runner-entry-authority-v1","runner_fd":runner_fd,"runner_path":runner_path,"runner_sha256":runner_row["sha256"],"runner_identity":ident(runner_before),"python_fd":python_fd,"python_path":python_path,"python_sha256":python_row["sha256"],"python_identity":ident(python_before),"release_digest":release_digest,"bootstrap_sha256":bootstrap_sha,"entry_method":"slurm-spooled-or-trusted-stdin-held-python-fd-v1","slurm_export_none_required":True,"bash_privileged_startup_required":True,"captured_source_entry":True}
entry["authority_digest"]=digest(entry)
os.environ.clear(); os.environ.update({"FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY":canonical(entry),"SLURM_JOB_ID":job_id,"SLURM_STEP_ID":step_id,"SLURM_GPUS_ON_NODE":gpus_on_node,"SLURM_GPUS_PER_NODE":gpus_per_node,"SLURM_STEP_GPUS":step_gpus,"SLURM_NNODES":node_count,"SLURM_STEP_NUM_NODES":step_node_count,"SLURM_JOB_NODELIST":job_nodelist,"SLURM_STEP_NODELIST":step_nodelist})
try: runner_source=runner_raw.decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("captured runner is not UTF-8") from error
sys.argv=[runner_path,*spec["runner_arguments"]]
module=types.ModuleType("__main__"); module.__file__=runner_path; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module
exec(compile(runner_source,runner_path,"exec",dont_inherit=True),module.__dict__)'''


def _shell_quote(value: str) -> str:
    if "\x00" in value:
        raise RootLaunchReleaseError("shell literal contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_release(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    file_roles = tuple(EXPECTED_STATIC_SHA256) + (
        "python",
        "ffmpeg",
        "plan",
    )
    identities = {
        role: _stable_file(value[role], executable=role in {"python", "ffmpeg"})
        for role in file_roles
    }
    admission_identity = _stable_file(
        value["gpu_admission_receipt"], return_raw=True
    )
    admission_raw = admission_identity.pop("_raw")
    identities["gpu_admission_receipt"] = admission_identity
    for role, expected in EXPECTED_STATIC_SHA256.items():
        if identities[role]["sha256"] != expected:
            raise RootLaunchReleaseError(f"hard-pinned {role} SHA differs")
    for role in ("runner", "bridge", "adapter", "eval_v1", "eval_v2", "model_authority"):
        if stat.S_IMODE(identities[role]["identity"]["mode"]) != 0o444:
            raise RootLaunchReleaseError(f"release source is not 0444: {role}")
    if stat.S_IMODE(
        identities["gpu_admission_receipt"]["identity"]["mode"]
    ) != 0o400:
        raise RootLaunchReleaseError("GPU47 admission receipt is not 0400")
    admission = _validate_gpu_admission_header(
        admission_raw, value=value, identities=identities
    )
    for role in ("model_root", "bernini_root", "veomni_root"):
        _canonical_path(value[role], exists=True, directory=True)
    for role in ("output_report", "runner_attestation", "authority_root", "rank_cache_root"):
        _canonical_path(value[role], exists=False, directory=False)
    if value["output_report"] == value["runner_attestation"]:
        raise RootLaunchReleaseError("final artifacts overlap")
    runner_arguments = _runner_arguments(value, identities)
    release: dict[str, Any] = {
        "schema_version": SCHEMA,
        "entry_mode": value["entry_mode"],
        "external_root_of_trust": (
            "slurm-controller-spooled-script-bytes"
            if value["entry_mode"] == "slurm_spool"
            else "trusted-controller-streamed-stdin-bytes"
        ),
        "bash_path": "/bin/bash",
        "bash_privileged_mode": True,
        "slurm_export_none": True,
        "python_is_executed_from_held_fd": True,
        "runner_is_compiled_from_captured_fd_bytes": True,
        "named_payload_execution_forbidden": True,
        "expected_allocation_gpu_count": 8,
        "campaign_mode": value["campaign_mode"],
        "selected_task_ids": list(
            TASK_IDS
            if value["campaign_mode"] == FULL16_CAMPAIGN
            else CANARY_TASK_IDS
        ),
        "formal_full16_report": value["campaign_mode"] == FULL16_CAMPAIGN,
        "canary_stops_after_pair_for_manual_visual_review": (
            value["campaign_mode"] == CASE00_CANARY_CAMPAIGN
        ),
        "gpu_visibility_contract": gpu47_visibility_contract(),
        "isolation_scope": (
            "ROCr-only logical isolation; not device-cgroup hard isolation"
        ),
        "slurm_environment_contract": {
            "required_source_names": [
                "SLURM_JOB_ID",
                "SLURM_STEP_ID",
                "SLURM_GPUS_ON_NODE",
                "SLURM_GPUS_PER_NODE",
                "SLURM_STEP_GPUS",
                "SLURM_NNODES",
                "SLURM_STEP_NUM_NODES",
                "SLURM_JOB_NODELIST",
                "SLURM_STEP_NODELIST",
            ],
            "required_absent_names": [
                "SLURM_JOB_GPUS",
                "SLURM_JOB_NUM_NODES",
            ],
            "caller_synthesized_slurm_facts_forbidden": True,
        },
        "empirical_gpu_admission": {
            "path": identities["gpu_admission_receipt"]["path"],
            "sha256": identities["gpu_admission_receipt"]["sha256"],
            "identity": identities["gpu_admission_receipt"]["identity"],
            "runner_replays_before_model_open": True,
            "receipt_digest": admission["receipt_digest"],
        },
        "holder_job_id": value["holder_job_id"],
        "expected_node": value["expected_node"],
        "identities": identities,
        "runner_arguments": runner_arguments,
    }
    release_digest = object_sha256(release)
    bootstrap_sha = hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest()
    mode_check = (
        '[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]]'
        if value["entry_mode"] == "trusted_stdin"
        else '[[ "$0" == /var/spool/slurmd/job*/slurm_script ]]'
    )
    lines = [
        "#!/bin/bash -p",
        "set -euo pipefail",
        "umask 077",
        '[[ "$-" == *p* ]] || { echo "privileged Bash entry required" >&2; exit 91; }',
        f"{mode_check} || {{ echo \"named payload execution forbidden\" >&2; exit 92; }}",
        '[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || { echo "startup environment forbidden" >&2; exit 93; }',
        f'[[ "${{SLURM_JOB_ID-}}" == {_shell_quote(value["holder_job_id"])} ]] || exit 94',
        '[[ "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ && "${SLURM_GPUS_ON_NODE-}" == "8" && "${SLURM_GPUS_PER_NODE-}" == "8" ]] || exit 95',
        '[[ "${SLURM_STEP_GPUS-}" == "0,1,2,3,4,5,6,7" && "${SLURM_NNODES-}" == "1" && "${SLURM_STEP_NUM_NODES-}" == "1" ]] || exit 96',
        f'[[ "${{SLURM_JOB_NODELIST-}}" == {_shell_quote(value["expected_node"])} && "${{SLURM_STEP_NODELIST-}}" == {_shell_quote(value["expected_node"])} ]] || exit 97',
        '[[ -z "${SLURM_JOB_GPUS+x}" && -z "${SLURM_JOB_NUM_NODES+x}" ]] || exit 99',
        VARREDIR_CLOSE_COMPAT,
        f"readonly FULL644_PINNED_PYTHON={_shell_quote(value['python'])}",
        'exec {FULL644_PYTHON_FD}<"$FULL644_PINNED_PYTHON"',
        '[[ "$FULL644_PYTHON_FD" =~ ^[0-9]+$ && -r "/proc/self/fd/$FULL644_PYTHON_FD" ]] || exit 98',
        "exec -c \"/proc/self/fd/$FULL644_PYTHON_FD\" -I -S -B -c "
        + _shell_quote(ROOT_BOOTSTRAP)
        + " \"$FULL644_PYTHON_FD\" "
        + _shell_quote(canonical_json_bytes(release).decode("utf-8"))
        + " "
        + _shell_quote(release_digest)
        + " "
        + _shell_quote(bootstrap_sha)
        + " "
        + _shell_quote(value["entry_mode"])
        + ' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_GPUS_ON_NODE" "$SLURM_GPUS_PER_NODE" "$SLURM_STEP_GPUS" "$SLURM_NNODES" "$SLURM_STEP_NUM_NODES" "$SLURM_JOB_NODELIST" "$SLURM_STEP_NODELIST"',
    ]
    return release, ("\n".join(lines) + "\n").encode("utf-8")


def _write_create_only(path_value: str | Path, payload: bytes, mode: int) -> None:
    path = _canonical_path(path_value, exists=False, directory=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RootLaunchReleaseError("release write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize(input_path: str, payload_path: str, receipt_path: str) -> dict[str, Any]:
    value, input_identity = _load_input(input_path)
    release, payload = build_release(value)
    payload_target = _canonical_path(payload_path, exists=False, directory=False)
    receipt_target = _canonical_path(receipt_path, exists=False, directory=False)
    if payload_target == receipt_target:
        raise RootLaunchReleaseError("payload/receipt overlap")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "MATERIALIZED_NOT_SUBMITTED",
        "launch_input": input_identity,
        "release": release,
        "release_digest": object_sha256(release),
        "root_bootstrap_sha256": hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest(),
        "payload_path": str(payload_target),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size": len(payload),
        "payload_mode": 0o444,
        "receipt_path": str(receipt_target),
        "required_entry": (
            "sbatch --export=NONE <payload>"
            if value["entry_mode"] == "slurm_spool"
            else "trusted controller: srun --export=NONE /bin/bash -p -s < <payload>"
        ),
        "named_payload_execution_forbidden": True,
        "submission_or_execution_performed": False,
        "remote_execution_authorized_by_this_receipt": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    _write_create_only(payload_target, payload, 0o444)
    _write_create_only(
        receipt_target, canonical_json_bytes(receipt) + b"\n", 0o400
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-input", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize(args.launch_input, args.payload, args.receipt)
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
