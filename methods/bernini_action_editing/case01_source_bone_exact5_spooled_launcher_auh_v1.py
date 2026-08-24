#!/usr/bin/env python3
"""Build the independent captured-entry launcher for the case01 exact-five canary.

This builder does not import or execute any historical root-launch builder.  It
materializes a fresh immutable payload whose release has exactly eighteen
identities: the sixteen r5f roles (with ``runner`` bound to the exact5 wrapper)
plus the adjacent frozen runner and exact5 evaluator.  The payload may only be
streamed to privileged Bash by a trusted controller inside the bound Slurm
allocation; ordinary named execution is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


SCHEMA = "case01-source-bone-exact5-root-launch-release-auh-v1"
INPUT_SCHEMA = "case01-source-bone-exact5-root-launch-input-auh-v1"
RECEIPT_SCHEMA = "case01-source-bone-exact5-root-launch-receipt-auh-v1"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
TASK_IDS = (
    "case01-exact_original-full644",
    "case01-codec_only_present-full644",
    "case01-bone_removed-full644",
    "case01-bone_translated_up150-full644",
    "case01-sham_control_up150-full644",
)
ENTRY_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-captured-runner-entry-authority-v1"
)
EXPECTED_STATIC_SHA256 = {
    "runner": "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea",
    "frozen_runner": "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223",
    "exact5_eval": "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58",
    "bridge": "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "adapter": "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "base_adapter": "53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca",
    "eval_v1": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "eval_v2": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "model_authority": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "torchrun_source": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    "torchrun_handler_source": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    "torch_local_agent_source": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
    "torch_dynamic_rendezvous_source": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
    "torch_multiprocessing_api_source": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
    "model_manifest": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
}
IDENTITY_ROLES = tuple(EXPECTED_STATIC_SHA256) + ("python", "ffmpeg", "plan")
if len(IDENTITY_ROLES) != 18 or len(set(IDENTITY_ROLES)) != 18:
    raise RuntimeError("exact18 role definition differs")

_SOURCE_ROLES = {
    "runner",
    "frozen_runner",
    "exact5_eval",
    "bridge",
    "adapter",
    "base_adapter",
    "eval_v1",
    "eval_v2",
    "model_authority",
}
_EXECUTABLE_ROLES = {"python", "ffmpeg"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Exact5RootLaunchError(RuntimeError):
    """The exact-five launch release or one of its authorities differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Exact5RootLaunchError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise Exact5RootLaunchError("duplicate JSON key")
        result[key] = value
    return result


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


def _read_fd(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise Exact5RootLaunchError("stable file short read")
    return raw


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
        raise Exact5RootLaunchError(f"release path differs: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        first = _read_fd(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or first != second
        or _identity(middle) != identity
        or _identity(after) != identity
        or _identity(named) != identity
        or (executable and not before.st_mode & 0o111)
    ):
        raise Exact5RootLaunchError(f"release identity differs: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": hashlib.sha256(first).hexdigest(),
        "identity": identity,
    }
    if return_raw:
        result["_raw"] = first
    return result


def _canonical_target(path_value: str | Path) -> Path:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise Exact5RootLaunchError(f"fresh target differs: {path}")
    return path


def _canonical_directory(path_value: str | Path) -> Path:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
    ):
        raise Exact5RootLaunchError(f"runtime directory differs: {path}")
    return path


def _load_input(path_value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _stable_file(path_value, return_raw=True)
    raw = identity.pop("_raw")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise Exact5RootLaunchError("launch input is not strict JSON") from error
    fields = {
        "schema_version", "entry_mode", *IDENTITY_ROLES[:-3], "python",
        "ffmpeg", "plan", "output_report", "runner_attestation",
        "model_root", "bernini_root", "veomni_root", "authority_root",
        "rank_cache_root", "holder_job_id", "expected_node", "campaign_mode",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("schema_version") != INPUT_SCHEMA
        or value.get("entry_mode") != "trusted_stdin"
        or value.get("campaign_mode") != CAMPAIGN
        or type(value.get("holder_job_id")) is not str
        or not value["holder_job_id"]
        or type(value.get("expected_node")) is not str
        or not value["expected_node"]
    ):
        raise Exact5RootLaunchError("launch input closure differs")
    return value, identity


def _validate_adjacency(value: Mapping[str, Any]) -> None:
    method_roles = (
        "runner", "frozen_runner", "exact5_eval", "bridge", "adapter",
        "base_adapter", "eval_v1", "eval_v2", "model_authority",
    )
    paths = [Path(value[role]) for role in method_roles]
    expected_names = {
        "runner": "case01_source_bone_exact5_runner_v1.py",
        "frozen_runner": "full644_exploratory_matched_runner_auh_r5.py",
        "exact5_eval": "case01_source_bone_exact5_eval_v1.py",
        "bridge": "full644_exploratory_matched_torchrun_fd_bridge_v2.py",
        "adapter": "full644_exploratory_matched_infer_adapter_auh_r5f.py",
        "base_adapter": "full644_exploratory_matched_infer_adapter_v2.py",
        "eval_v1": "full644_exploratory_matched_eval_v1.py",
        "eval_v2": "full644_exploratory_matched_eval_v2.py",
        "model_authority": "action_preservation_decoded_eval_model_authority_v2.py",
    }
    if (
        len({str(path) for path in paths}) != len(paths)
        or len({path.parent for path in paths}) != 1
        or any(path.name != expected_names[role] for role, path in zip(method_roles, paths))
    ):
        raise Exact5RootLaunchError("exact5 source adjacency differs")


def _runner_arguments(value: Mapping[str, Any], identities: Mapping[str, Any]) -> list[str]:
    arguments = [
        "--campaign-mode", CAMPAIGN,
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
        "--torch-local-agent-source", value["torch_local_agent_source"],
        "--torch-local-agent-source-sha256", identities["torch_local_agent_source"]["sha256"],
        "--torch-dynamic-rendezvous-source", value["torch_dynamic_rendezvous_source"],
        "--torch-dynamic-rendezvous-source-sha256", identities["torch_dynamic_rendezvous_source"]["sha256"],
        "--torch-multiprocessing-api-source", value["torch_multiprocessing_api_source"],
        "--torch-multiprocessing-api-source-sha256", identities["torch_multiprocessing_api_source"]["sha256"],
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
        raise Exact5RootLaunchError("runner argument differs")
    return arguments


ROOT_BOOTSTRAP = r'''import os,sys
if len(sys.argv)!=15: raise RuntimeError("exact5 root bootstrap argv differs")
raw_fd,spec_raw,release_digest,bootstrap_sha,entry_mode,job_id,step_id,gpus_on_node,gpus_per_node,step_gpus,node_count,step_node_count,job_nodelist,step_nodelist=sys.argv[1:]
if entry_mode!="trusted_stdin" or not step_id.isascii() or not step_id.isdecimal() or int(step_id)<=0 or str(int(step_id))!=step_id: raise RuntimeError("exact5 root entry differs")
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
def replay_named(role,row,executable=False):
 path=row.get("path")
 if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.islink(path): raise RuntimeError("named role path differs: "+role)
 flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0); fd=os.open(path,flags)
 try:
  before=os.fstat(fd); first=read_fd(fd,before.st_size); middle=os.fstat(fd); second=read_fd(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 finally: os.close(fd)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or (executable and not before.st_mode&0o111) or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or ident(before)!=row.get("identity") or first!=second or hashlib.sha256(first).hexdigest()!=row.get("sha256"): raise RuntimeError("named role identity differs: "+role)
 return first,before
spec=json.loads(spec_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
expected_fields={"schema_version","entry_mode","external_root_of_trust","bash_path","bash_privileged_mode","slurm_export_none","python_is_executed_from_held_fd","runner_is_compiled_from_captured_fd_bytes","all_exact18_named_identities_replayed_before_runner","named_payload_execution_forbidden","expected_allocation_gpu_count","campaign_mode","task_count","selected_task_ids","exploratory_only","retry_allowed","partial_outputs_are_not_results","slurm_environment_contract","holder_job_id","expected_node","identities","runner_arguments"}
expected_roles=("runner","frozen_runner","exact5_eval","bridge","adapter","base_adapter","eval_v1","eval_v2","model_authority","torchrun_source","torchrun_handler_source","torch_local_agent_source","torch_dynamic_rendezvous_source","torch_multiprocessing_api_source","model_manifest","python","ffmpeg","plan")
expected_tasks=["case01-exact_original-full644","case01-codec_only_present-full644","case01-bone_removed-full644","case01-bone_translated_up150-full644","case01-sham_control_up150-full644"]
slurm_contract={"caller_synthesized_slurm_facts_forbidden":True,"required_absent_names":["SLURM_JOB_GPUS","SLURM_JOB_NUM_NODES"],"required_source_names":["SLURM_JOB_ID","SLURM_STEP_ID","SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_NODE","SLURM_STEP_GPUS","SLURM_NNODES","SLURM_STEP_NUM_NODES","SLURM_JOB_NODELIST","SLURM_STEP_NODELIST"]}
if type(spec) is not dict or set(spec)!=expected_fields or canonical(spec)!=spec_raw or digest(spec)!=release_digest or spec.get("schema_version")!="case01-source-bone-exact5-root-launch-release-auh-v1" or spec.get("entry_mode")!=entry_mode or spec.get("external_root_of_trust")!="trusted-controller-streamed-stdin-bytes" or spec.get("bash_path")!="/bin/bash" or spec.get("bash_privileged_mode") is not True or spec.get("slurm_export_none") is not True or spec.get("python_is_executed_from_held_fd") is not True or spec.get("runner_is_compiled_from_captured_fd_bytes") is not True or spec.get("all_exact18_named_identities_replayed_before_runner") is not True or spec.get("named_payload_execution_forbidden") is not True or spec.get("expected_allocation_gpu_count")!=8 or spec.get("campaign_mode")!="case01-source-bone-exact5-r64-canary" or spec.get("task_count")!=5 or spec.get("selected_task_ids")!=expected_tasks or spec.get("exploratory_only") is not True or spec.get("retry_allowed") is not False or spec.get("partial_outputs_are_not_results") is not True or spec.get("slurm_environment_contract")!=slurm_contract or job_id!=spec.get("holder_job_id") or gpus_on_node!="8" or gpus_per_node!="8" or step_gpus!="0,1,2,3,4,5,6,7" or node_count!="1" or step_node_count!="1" or job_nodelist!=spec.get("expected_node") or step_nodelist!=spec.get("expected_node"): raise RuntimeError("exact5 root release differs")
identities=spec.get("identities")
if type(identities) is not dict or set(identities)!=set(expected_roles) or len(identities)!=18 or len({row.get("path") for row in identities.values() if type(row) is dict})!=18: raise RuntimeError("exact18 identity closure differs")
arguments=spec.get("runner_arguments")
argument_flags=("--campaign-mode","--plan","--plan-sha256","--output-report","--runner-attestation","--runner-sha256","--bridge-script","--bridge-script-sha256","--adapter-script","--adapter-script-sha256","--eval-v1-source","--eval-v1-source-sha256","--eval-v2-source","--eval-v2-source-sha256","--model-authority-source","--model-authority-source-sha256","--python","--python-sha256","--ffmpeg-executable","--ffmpeg-executable-sha256","--torchrun-source","--torchrun-source-sha256","--torchrun-handler-source","--torchrun-handler-source-sha256","--torch-local-agent-source","--torch-local-agent-source-sha256","--torch-dynamic-rendezvous-source","--torch-dynamic-rendezvous-source-sha256","--torch-multiprocessing-api-source","--torch-multiprocessing-api-source-sha256","--model-root","--model-manifest","--model-manifest-sha256","--bernini-root","--veomni-root","--authority-root","--rank-cache-root","--holder-job-id","--expected-node","--expected-allocation-gpu-count")
if type(arguments) is not list or len(arguments)!=2*len(argument_flags) or tuple(arguments[::2])!=argument_flags or any(type(value) is not str or not value for value in arguments): raise RuntimeError("exact5 runner argument vector differs")
argument_values=dict(zip(arguments[::2],arguments[1::2]))
bound_roles={"--plan":"plan","--plan-sha256":"plan","--runner-sha256":"runner","--bridge-script":"bridge","--bridge-script-sha256":"bridge","--adapter-script":"adapter","--adapter-script-sha256":"adapter","--eval-v1-source":"eval_v1","--eval-v1-source-sha256":"eval_v1","--eval-v2-source":"eval_v2","--eval-v2-source-sha256":"eval_v2","--model-authority-source":"model_authority","--model-authority-source-sha256":"model_authority","--python":"python","--python-sha256":"python","--ffmpeg-executable":"ffmpeg","--ffmpeg-executable-sha256":"ffmpeg","--torchrun-source":"torchrun_source","--torchrun-source-sha256":"torchrun_source","--torchrun-handler-source":"torchrun_handler_source","--torchrun-handler-source-sha256":"torchrun_handler_source","--torch-local-agent-source":"torch_local_agent_source","--torch-local-agent-source-sha256":"torch_local_agent_source","--torch-dynamic-rendezvous-source":"torch_dynamic_rendezvous_source","--torch-dynamic-rendezvous-source-sha256":"torch_dynamic_rendezvous_source","--torch-multiprocessing-api-source":"torch_multiprocessing_api_source","--torch-multiprocessing-api-source-sha256":"torch_multiprocessing_api_source","--model-manifest":"model_manifest","--model-manifest-sha256":"model_manifest"}
for flag,role in bound_roles.items():
 expected=identities[role]["sha256"] if flag.endswith("-sha256") else identities[role]["path"]
 if argument_values.get(flag)!=expected: raise RuntimeError("exact5 runner identity argument differs: "+flag)
if argument_values.get("--campaign-mode")!="case01-source-bone-exact5-r64-canary" or argument_values.get("--holder-job-id")!=job_id or argument_values.get("--expected-node")!=spec.get("expected_node") or argument_values.get("--expected-allocation-gpu-count")!="8" or argument_values.get("--output-report")==argument_values.get("--runner-attestation"): raise RuntimeError("exact5 runner authority argument differs")
python_row=identities["python"]; runner_row=identities["runner"]; python_path=python_row.get("path"); runner_path=runner_row.get("path")
python_raw=read_fd(python_fd,python_before.st_size); python_after=os.fstat(python_fd); python_named=os.lstat(python_path); python_process=os.stat("/proc/self/exe")
if not stat.S_ISREG(python_before.st_mode) or python_before.st_nlink!=1 or not python_before.st_mode&0o111 or ident(python_before)!=ident(python_after) or ident(python_before)!=ident(python_named) or ident(python_before)!=ident(python_process) or ident(python_before)!=python_row.get("identity") or hashlib.sha256(python_raw).hexdigest()!=python_row.get("sha256"): raise RuntimeError("held Python identity differs")
runner_raw,runner_before=replay_named("runner",runner_row)
runner_fd=os.open(runner_path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)); os.set_inheritable(runner_fd,False)
if ident(os.fstat(runner_fd))!=ident(runner_before) or hashlib.sha256(read_fd(runner_fd,runner_before.st_size)).hexdigest()!=runner_row["sha256"]: raise RuntimeError("captured wrapper changed")
for role in expected_roles:
 if role not in {"python","runner"}: replay_named(role,identities[role],role=="ffmpeg")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): raise RuntimeError("root Python environment differs")
entry={"schema_version":"full644-exploratory-matched-captured-runner-entry-authority-v1","runner_fd":runner_fd,"runner_path":runner_path,"runner_sha256":runner_row["sha256"],"runner_identity":ident(runner_before),"python_fd":python_fd,"python_path":python_path,"python_sha256":python_row["sha256"],"python_identity":ident(python_before),"release_digest":release_digest,"bootstrap_sha256":bootstrap_sha,"entry_method":"slurm-spooled-or-trusted-stdin-held-python-fd-v1","slurm_export_none_required":True,"bash_privileged_startup_required":True,"captured_source_entry":True}
entry["authority_digest"]=digest(entry)
os.environ.clear(); os.environ.update({"FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY":canonical(entry),"SLURM_JOB_ID":job_id,"SLURM_STEP_ID":step_id,"SLURM_GPUS_ON_NODE":gpus_on_node,"SLURM_GPUS_PER_NODE":gpus_per_node,"SLURM_STEP_GPUS":step_gpus,"SLURM_NNODES":node_count,"SLURM_STEP_NUM_NODES":step_node_count,"SLURM_JOB_NODELIST":job_nodelist,"SLURM_STEP_NODELIST":step_nodelist})
try: runner_source=runner_raw.decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("captured wrapper is not UTF-8") from error
sys.argv=[runner_path,*arguments]
module=types.ModuleType("__main__"); module.__file__=runner_path; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module
exec(compile(runner_source,runner_path,"exec",dont_inherit=True),module.__dict__)'''


def _shell_quote(value: str) -> str:
    if "\x00" in value:
        raise Exact5RootLaunchError("shell literal contains NUL")
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_release(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    _validate_adjacency(value)
    identities = {
        role: _stable_file(value[role], executable=role in _EXECUTABLE_ROLES)
        for role in IDENTITY_ROLES
    }
    if tuple(identities) != IDENTITY_ROLES or len(identities) != 18:
        raise Exact5RootLaunchError("exact18 role order differs")
    paths = [row["path"] for row in identities.values()]
    if len(set(paths)) != 18:
        raise Exact5RootLaunchError("exact18 identity paths overlap")
    for role, expected in EXPECTED_STATIC_SHA256.items():
        if identities[role]["sha256"] != expected:
            raise Exact5RootLaunchError(f"hard-pinned {role} SHA differs")
    for role in _SOURCE_ROLES:
        if stat.S_IMODE(identities[role]["identity"]["mode"]) != 0o444:
            raise Exact5RootLaunchError(f"release source is not 0444: {role}")
    if stat.S_IMODE(identities["plan"]["identity"]["mode"]) != 0o444:
        raise Exact5RootLaunchError("exact5 plan is not 0444")
    plan_identity = _stable_file(value["plan"], return_raw=True)
    plan_raw = plan_identity.pop("_raw")
    if plan_identity != identities["plan"]:
        raise Exact5RootLaunchError("exact5 plan identity changed")
    try:
        plan = json.loads(
            plan_raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise Exact5RootLaunchError("exact5 plan is not strict JSON") from error
    tasks = plan.get("tasks") if type(plan) is dict else None
    if (
        type(plan) is not dict
        or plan_raw != canonical_json_bytes(plan) + b"\n"
        or plan.get("schema_version") != "case01-source-bone-exact5-r64-plan-v1"
        or plan.get("experiment_id")
        != "case01-288545b9c031491a-source-bone-exact5-r64-v1"
        or plan.get("production_ready") is not True
        or plan.get("launch_allowed") is not True
        or plan.get("task_count") != 5
        or type(tasks) is not list
        or [row.get("task_id") if type(row) is dict else None for row in tasks]
        != list(TASK_IDS)
    ):
        raise Exact5RootLaunchError("exact5 plan campaign/task closure differs")
    for role in ("model_root", "bernini_root", "veomni_root"):
        _canonical_directory(value[role])
    outputs = [
        _canonical_target(value[role])
        for role in ("output_report", "runner_attestation", "authority_root", "rank_cache_root")
    ]
    if len(set(outputs)) != len(outputs) or any(str(path) in paths for path in outputs):
        raise Exact5RootLaunchError("fresh output paths overlap")
    arguments = _runner_arguments(value, identities)
    release: dict[str, Any] = {
        "schema_version": SCHEMA,
        "entry_mode": "trusted_stdin",
        "external_root_of_trust": "trusted-controller-streamed-stdin-bytes",
        "bash_path": "/bin/bash",
        "bash_privileged_mode": True,
        "slurm_export_none": True,
        "python_is_executed_from_held_fd": True,
        "runner_is_compiled_from_captured_fd_bytes": True,
        "all_exact18_named_identities_replayed_before_runner": True,
        "named_payload_execution_forbidden": True,
        "expected_allocation_gpu_count": 8,
        "campaign_mode": CAMPAIGN,
        "task_count": 5,
        "selected_task_ids": list(TASK_IDS),
        "exploratory_only": True,
        "retry_allowed": False,
        "partial_outputs_are_not_results": True,
        "slurm_environment_contract": {
            "required_source_names": [
                "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_GPUS_ON_NODE",
                "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS", "SLURM_NNODES",
                "SLURM_STEP_NUM_NODES", "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
            ],
            "required_absent_names": ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"],
            "caller_synthesized_slurm_facts_forbidden": True,
        },
        "holder_job_id": value["holder_job_id"],
        "expected_node": value["expected_node"],
        "identities": identities,
        "runner_arguments": arguments,
    }
    release_digest = object_sha256(release)
    bootstrap_sha = hashlib.sha256(ROOT_BOOTSTRAP.encode("utf-8")).hexdigest()
    lines = [
        "#!/bin/bash -p", "set -euo pipefail", "umask 077",
        '[[ "$-" == *p* ]] || { echo "privileged Bash entry required" >&2; exit 91; }',
        '[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || { echo "named payload execution forbidden" >&2; exit 92; }',
        '[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || { echo "startup environment forbidden" >&2; exit 93; }',
        f'[[ "${{SLURM_JOB_ID-}}" == {_shell_quote(value["holder_job_id"])} ]] || exit 94',
        '[[ "${SLURM_STEP_ID-}" =~ ^[1-9][0-9]*$ && "${SLURM_GPUS_ON_NODE-}" == "8" && "${SLURM_GPUS_PER_NODE-}" == "8" ]] || exit 95',
        '[[ "${SLURM_STEP_GPUS-}" == "0,1,2,3,4,5,6,7" && "${SLURM_NNODES-}" == "1" && "${SLURM_STEP_NUM_NODES-}" == "1" ]] || exit 96',
        f'[[ "${{SLURM_JOB_NODELIST-}}" == {_shell_quote(value["expected_node"])} && "${{SLURM_STEP_NODELIST-}}" == {_shell_quote(value["expected_node"])} ]] || exit 97',
        '[[ -z "${SLURM_JOB_GPUS+x}" && -z "${SLURM_JOB_NUM_NODES+x}" ]] || exit 98',
        "if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi",
        f"readonly EXACT5_PINNED_PYTHON={_shell_quote(value['python'])}",
        'exec {EXACT5_PYTHON_FD}<"$EXACT5_PINNED_PYTHON"',
        '[[ "$EXACT5_PYTHON_FD" =~ ^[0-9]+$ && -r "/proc/self/fd/$EXACT5_PYTHON_FD" ]] || exit 99',
        "exec -c \"/proc/self/fd/$EXACT5_PYTHON_FD\" -I -S -B -c "
        + _shell_quote(ROOT_BOOTSTRAP)
        + ' "$EXACT5_PYTHON_FD" '
        + _shell_quote(canonical_json_bytes(release).decode("utf-8")) + " "
        + _shell_quote(release_digest) + " " + _shell_quote(bootstrap_sha) + " "
        + _shell_quote("trusted_stdin")
        + ' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_GPUS_ON_NODE" "$SLURM_GPUS_PER_NODE" "$SLURM_STEP_GPUS" "$SLURM_NNODES" "$SLURM_STEP_NUM_NODES" "$SLURM_JOB_NODELIST" "$SLURM_STEP_NODELIST"',
    ]
    return release, ("\n".join(lines) + "\n").encode("utf-8")


def _write_create_only(path: Path, payload: bytes, mode: int) -> None:
    target = _canonical_target(path)
    descriptor = os.open(
        target,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0,
    )
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise Exact5RootLaunchError("create-only write made no progress")
            offset += count
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        named = target.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0
            or before.st_nlink != 1
            or _identity(before) != _identity(named)
            or _read_fd(descriptor, len(payload)) != payload
        ):
            raise Exact5RootLaunchError("create-only staging replay differs")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize(input_path: str, payload_path: str, receipt_path: str) -> dict[str, Any]:
    value, input_identity = _load_input(input_path)
    payload_target = _canonical_target(payload_path)
    receipt_target = _canonical_target(receipt_path)
    if payload_target == receipt_target or str(payload_target) == input_identity["path"] or str(receipt_target) == input_identity["path"]:
        raise Exact5RootLaunchError("launch artifact paths overlap")
    release, payload = build_release(value)
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
        "required_entry": "trusted controller: one srun --export=NONE /bin/bash -p -s from held payload bytes",
        "named_payload_execution_forbidden": True,
        "submission_or_execution_performed": False,
        "remote_execution_authorized_by_this_receipt": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    _write_create_only(payload_target, payload, 0o444)
    _write_create_only(receipt_target, canonical_json_bytes(receipt) + b"\n", 0o400)
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
