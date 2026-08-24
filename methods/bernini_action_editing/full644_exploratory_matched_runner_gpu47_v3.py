#!/usr/bin/env python3
"""GPU4-7 AUH-env-bound case00 canary runner for the full644 matched eval.

The runner consumes a frozen v1 16-task plan and executes either the exact
case00 Base/R64 canary pair or the full plan, in plan order, under one
holder-lifetime exact-23 model capture.  It
never retries or resumes.  The base and full644 arms therefore share the same
model-capture digest; each adapted task additionally receives a fresh retained
adapter capture and the externally pinned terminal checkpoint manifest.

This file is eval-only.  It does not train, submit Slurm work, SSH, build an
HTML page, or authorize a scientific claim.  It is intended to run inside an
already allocated AUH step after a separate frozen launcher has pinned every
CLI file identity.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Mapping, Sequence


EXPECTED_EVAL_V1_SHA256 = (
    "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d"
)
EXPECTED_EVAL_V2_SHA256 = (
    "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982"
)
EXPECTED_MODEL_AUTHORITY_SHA256 = (
    "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
)
TORCHRUN_SOURCE_SHA256 = (
    "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
)
TORCHRUN_HANDLER_SHA256 = (
    "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87"
)
SITE_PACKAGES_ENV = "FULL644_MATCHED_SITE_PACKAGES_ROOT"
PYTHON_EXECUTABLE_BINDING_ENV = "FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING"
ENTRY_AUTHORITY_ENV = "FULL644_MATCHED_CAPTURED_RUNNER_ENTRY_AUTHORITY"
ENTRY_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-captured-runner-entry-authority-v1"
)
GPU_VISIBILITY_SCHEMA = "full644-exploratory-matched-gpu47-visibility-v1"
GPU_ADMISSION_SCHEMA = "full644-exploratory-matched-gpu47-admission-v1"
GPU_ADMISSION_MAX_AGE_NS = 10 * 60 * 1_000_000_000
ROCR_PHYSICAL_GPU_MASK = "4,5,6,7"
PHYSICAL_GPU_INDICES = (4, 5, 6, 7)
LOGICAL_GPU_INDICES = (0, 1, 2, 3)
_SECONDARY_GPU_MASKS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def _bootstrap_read_source(
    path_value: str | Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    """Capture exact UTF-8 source without consulting import bytecode."""

    path = Path(path_value).expanduser().resolve(strict=True)
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_sha256)
    ):
        raise RuntimeError(f"{label} source SHA differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or digest.hexdigest() != expected_sha256
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or identity
        != (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_uid,
            named.st_gid,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
    ):
        raise RuntimeError(f"{label} source identity differs")
    try:
        source = b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} source is not UTF-8") from error
    return path, source


def _bootstrap_load_source_module(
    name: str,
    path_value: str | Path,
    expected_sha256: str,
    *,
    require_absent: bool,
) -> Any:
    path, source = _bootstrap_read_source(
        path_value, expected_sha256, label=name
    )
    existing = sys.modules.get(name)
    if existing is not None:
        if require_absent:
            raise RuntimeError(f"{name} was imported before source-only bootstrap")
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve(strict=True) != path:
            raise RuntimeError(f"{name} existing origin differs")
        return existing
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, origin=str(path)
    )
    sys.modules[name] = module
    try:
        exec(
            compile(source, str(path), "exec", dont_inherit=True),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if module.__cached__ is not None or module.__file__ != str(path):
        raise RuntimeError(f"{name} executed-source origin differs")
    return module


_METHOD_ROOT = Path(__file__).resolve(strict=True).parent
model_authority = _bootstrap_load_source_module(
    "action_preservation_decoded_eval_model_authority_v2",
    _METHOD_ROOT / "action_preservation_decoded_eval_model_authority_v2.py",
    EXPECTED_MODEL_AUTHORITY_SHA256,
    require_absent=(__name__ == "__main__"),
)
v1 = _bootstrap_load_source_module(
    "full644_exploratory_matched_eval_v1",
    _METHOD_ROOT / "full644_exploratory_matched_eval_v1.py",
    EXPECTED_EVAL_V1_SHA256,
    require_absent=(__name__ == "__main__"),
)
v2 = _bootstrap_load_source_module(
    "full644_exploratory_matched_eval_v2",
    _METHOD_ROOT / "full644_exploratory_matched_eval_v2.py",
    EXPECTED_EVAL_V2_SHA256,
    require_absent=(__name__ == "__main__"),
)


SCHEMA = "full644-exploratory-matched-runner-attestation-gpu47-v3"
TASK_SCHEMA = "full644-exploratory-matched-runner-task-gpu47-v3"
FAILURE_SCHEMA = "full644-exploratory-matched-runner-failure-gpu47-v3"
CONSUMPTION_CHAIN_SCHEMA = "full644-exploratory-matched-consumption-chain-v2"
EXEC_AUTHORITY_SCHEMA = "full644-exploratory-matched-exec-authority-v2"
PUBLICATION_HANDOFF_ENV = "FULL644_MATCHED_PUBLICATION_HANDOFF_AUTHORITY"
PUBLICATION_HANDOFF_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-publication-handoff-authority-v1"
)
PUBLICATION_HANDOFF_PAYLOAD_SCHEMA = (
    "full644-exploratory-matched-publication-handoff-payload-v1"
)
EXPECTED_MODEL_MANIFEST_SHA256 = model_authority.MODEL_MANIFEST_SHA256
EXPECTED_BERNINI_COMMIT = v1.EXPECTED_BERNINI_COMMIT
EXPECTED_VEOMNI_COMMIT = v1.EXPECTED_VEOMNI_COMMIT
EXPECTED_CHECKPOINT_TREE_SHA256 = v1.EXPECTED_CHECKPOINT_TREE_SHA256
TERMINAL_CP644_ADAPTER_README_SHA256 = (
    "f9e934433c6ae81516474d416c6b1dcf5193fe08bf41205363cf0b5b33760c1e"
)
TASK_IDS = tuple(
    f"shared8-{index:02d}-{arm}"
    for index in range(8)
    for arm in ("base", "full644")
)
FULL16_CAMPAIGN = "full16-production"
CASE00_CANARY_CAMPAIGN = "case00-pair-canary"
CANARY_TASK_IDS = TASK_IDS[:2]
_BASE_ARTIFACT_SUFFIXES = (
    "-model-capture.json",
    "-model-pre-use.json",
    "-consumption-input.json",
    "-model-post-use.json",
    "-eval-consumption-chain.json",
)
_ADAPTER_ARTIFACT_SUFFIXES = (
    "-adapter-capture.json",
    "-adapter-pre-use.json",
    "-adapter-post-use.json",
    "-adapter-final.json",
)


class MatchedRunnerV2Error(RuntimeError):
    """The external execution envelope or one task differs."""


def gpu47_visibility_contract(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    if (
        environment.get("ROCR_VISIBLE_DEVICES") != ROCR_PHYSICAL_GPU_MASK
        or any(key in environment for key in _SECONDARY_GPU_MASKS)
    ):
        raise MatchedRunnerV2Error("GPU47 ROCr-only visibility differs")
    return {
        "schema_version": GPU_VISIBILITY_SCHEMA,
        "visibility_variable": "ROCR_VISIBLE_DEVICES",
        "rocr_visible_devices": ROCR_PHYSICAL_GPU_MASK,
        "physical_gpu_indices": list(PHYSICAL_GPU_INDICES),
        "logical_gpu_indices": list(LOGICAL_GPU_INDICES),
        "logical_to_physical_order_not_inferred_from_mask": True,
        "empirical_pci_uuid_admission_required": True,
        "secondary_visibility_variables_absent": True,
        "world_size": 4,
        "slurm_step_reserved_gpu_indices": list(range(8)),
    }


ISOLATED_BRIDGE_BOOTSTRAP = r'''import fcntl,hashlib,json,os,stat,sys,types
ENV="FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING"
HENV="FULL644_MATCHED_PUBLICATION_HANDOFF_AUTHORITY"
SCHEMA="full644-exploratory-matched-exec-authority-v2"
HSCHEMA="full644-exploratory-matched-publication-handoff-authority-v1"
ROLES=["python_executable","bridge_source","adapter_source","ffmpeg_executable"]
RF={"role","fd","source_path","sha256","identity"}
IF={"device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"}
MF={"schema_version","task_id","model_capture_digest","adapter_capture_digest","fd_count","fd_rows","fd_rows_digest","namespace_root_count","publication_root_count","exact_allowlist_only","proc_self_fd_consumption_required","cross_process_proc_fd_access_forbidden","ptrace_authorization_used","fd_binding_digest"}
MRF={"fd","scope","role","relative_path","source_path","identity"}
MSCHEMA="bernini-action-preservation-inherited-fd-binding-v3"
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate exec-authority JSON key")
  out[key]=value
 return out
def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def ident(value): return {"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"nlink":value.st_nlink,"rdev":value.st_rdev,"size":value.st_size,"blocks":getattr(value,"st_blocks",0),"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
def read(row):
 fd=row["fd"]; before=os.fstat(fd); chunks=[]; offset=0
 while offset<before.st_size:
  block=os.pread(fd,min(1048576,before.st_size-offset),offset)
  if not block: break
  chunks.append(block); offset+=len(block)
 after=os.fstat(fd); raw=b"".join(chunks)
 if ident(before)!=row["identity"] or ident(after)!=row["identity"] or len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=row["sha256"]: raise RuntimeError("exec-authority FD replay differs")
 return raw
def seal_model(raw,code_fds):
 model=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 if type(model) is not dict or set(model)!=MF or canonical(model)!=raw: raise RuntimeError("model authority binding differs")
 unsigned=dict(model); claimed=unsigned.pop("fd_binding_digest",None); rows=model.get("fd_rows")
 if model.get("schema_version")!=MSCHEMA or model.get("exact_allowlist_only") is not True or model.get("proc_self_fd_consumption_required") is not True or model.get("cross_process_proc_fd_access_forbidden") is not True or model.get("ptrace_authorization_used") is not False or claimed!=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest() or type(rows) is not list or type(model.get("fd_count")) is not int or model["fd_count"]!=len(rows) or model.get("fd_rows_digest")!=hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest(): raise RuntimeError("model authority digest differs")
 mfds=[]; roles=[]
 for item in rows:
  identity=item.get("identity") if type(item) is dict else None; fd=item.get("fd") if type(item) is dict else None; role=item.get("role") if type(item) is dict else None; scope=item.get("scope") if type(item) is dict else None
  if type(item) is not dict or set(item)!=MRF or type(fd) is not int or fd<3 or scope not in {"model","adapter","task"} or role not in {"file","namespace_root","publication_root"} or type(item.get("relative_path")) is not str or type(item.get("source_path")) is not str or not os.path.isabs(item["source_path"]) or os.path.normpath(item["source_path"])!=item["source_path"] or type(identity) is not dict or set(identity)!=IF or any(type(value) is not int for value in identity.values()) or (role=="file" and not stat.S_ISREG(identity["mode"])) or (role!="file" and not stat.S_ISDIR(identity["mode"])) or not os.get_inheritable(fd): raise RuntimeError("model authority row differs")
  observed=ident(os.fstat(fd)); mutable=scope=="task" and role=="publication_root"; immutable={key for key in ("device","inode","uid","gid","mode","rdev")}
  if (observed if not mutable else {key:observed[key] for key in immutable})!=(identity if not mutable else {key:identity[key] for key in immutable}): raise RuntimeError("model authority FD identity differs")
  mfds.append(fd); roles.append((scope,role,item["relative_path"]))
 if mfds!=sorted(mfds) or len(mfds)!=len(set(mfds)) or set(mfds)&set(code_fds) or len(mfds) not in {25,30} or roles.count(("task","publication_root","."))!=1 or sum(scope=="model" and role=="file" for scope,role,_ in roles)!=23 or sum(scope=="model" and role=="namespace_root" for scope,role,_ in roles)!=1 or sum(scope=="adapter" and role=="file" for scope,role,_ in roles)!=(0 if model.get("adapter_capture_digest") is None else 4) or sum(scope=="adapter" and role=="namespace_root" for scope,role,_ in roles)!=(0 if model.get("adapter_capture_digest") is None else 1): raise RuntimeError("model authority FD allowlist differs")
 for fd in mfds: os.set_inheritable(fd,False)
 if any(os.get_inheritable(fd) for fd in mfds): raise RuntimeError("model authority FDs remain inheritable")
 return mfds,model["task_id"]
raw=os.environ.get(ENV)
if raw is None: raise RuntimeError("exec-authority binding is absent")
binding=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if type(binding) is not dict or set(binding)!={"schema_version","rows","rows_digest","binding_digest"} or binding.get("schema_version")!=SCHEMA or canonical(binding)!=raw: raise RuntimeError("exec-authority binding differs")
unsigned=dict(binding); claimed=unsigned.pop("binding_digest",None); rows=binding.get("rows")
if type(rows) is not list or [row.get("role") if type(row) is dict else None for row in rows]!=ROLES or binding.get("rows_digest")!=hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest() or claimed!=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest(): raise RuntimeError("exec-authority digest differs")
fds=[]; payload={}
for row in rows:
 if type(row) is not dict or set(row)!=RF or type(row.get("fd")) is not int or row["fd"]<3 or type(row.get("source_path")) is not str or not os.path.isabs(row["source_path"]) or os.path.normpath(row["source_path"])!=row["source_path"] or type(row.get("sha256")) is not str or len(row["sha256"])!=64 or any(ch not in "0123456789abcdef" for ch in row["sha256"]) or type(row.get("identity")) is not dict or set(row["identity"])!=IF or any(type(value) is not int for value in row["identity"].values()) or not stat.S_ISREG(row["identity"]["mode"]) or row["identity"]["nlink"]!=1 or not os.get_inheritable(row["fd"]): raise RuntimeError("exec-authority row differs")
 fds.append(row["fd"]); payload[row["role"]]=read(row)
if len(fds)!=len(set(fds)) or fds!=sorted(fds): raise RuntimeError("exec-authority FD allowlist differs")
model_raw=os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
if model_raw is None: raise RuntimeError("model authority binding is absent")
model_fds,model_task_id=seal_model(model_raw,fds)
handoff_raw=os.environ.get(HENV)
if handoff_raw is None: raise RuntimeError("publication handoff authority is absent")
handoff=json.loads(handoff_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if type(handoff) is not dict or set(handoff)!={"schema_version","task_id","fd","initial_identity","capacity","authority_digest"} or canonical(handoff)!=handoff_raw: raise RuntimeError("publication handoff authority differs")
handoff_unsigned=dict(handoff); handoff_claimed=handoff_unsigned.pop("authority_digest",None); handoff_fd=handoff.get("fd"); handoff_identity=handoff.get("initial_identity")
if handoff.get("schema_version")!=HSCHEMA or handoff.get("task_id")!=model_task_id or handoff.get("capacity")!=65536 or handoff_claimed!=hashlib.sha256(canonical(handoff_unsigned).encode("utf-8")).hexdigest() or type(handoff_fd) is not int or handoff_fd<3 or type(handoff_identity) is not dict or set(handoff_identity)!=IF or any(type(value) is not int for value in handoff_identity.values()) or handoff_fd in set(fds)|set(model_fds) or not os.get_inheritable(handoff_fd): raise RuntimeError("publication handoff digest differs")
handoff_observed=os.fstat(handoff_fd)
if ident(handoff_observed)!=handoff_identity or not stat.S_ISREG(handoff_observed.st_mode) or handoff_observed.st_nlink!=0 or stat.S_IMODE(handoff_observed.st_mode)!=0o600 or handoff_observed.st_size!=0 or fcntl.fcntl(handoff_fd,fcntl.F_GET_SEALS)!=0: raise RuntimeError("empty publication handoff replay differs")
python_row,bridge_row,adapter_row,ffmpeg_row=rows
if not python_row["identity"]["mode"]&0o111 or not ffmpeg_row["identity"]["mode"]&0o111: raise RuntimeError("retained executable mode differs")
try: bridge_source=payload["bridge_source"].decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("captured bridge source is not UTF-8") from error
for fd in [*fds,handoff_fd]: os.set_inheritable(fd,False)
if any(os.get_inheritable(fd) for fd in [*fds,handoff_fd]): raise RuntimeError("exec/handoff authority FDs remain inheritable")
os.close(bridge_row["fd"])
rank_rows=[python_row,adapter_row,ffmpeg_row]
rank_binding={"schema_version":SCHEMA,"rows":rank_rows,"rows_digest":hashlib.sha256(canonical(rank_rows).encode("utf-8")).hexdigest()}
rank_binding["binding_digest"]=hashlib.sha256(canonical(rank_binding).encode("utf-8")).hexdigest()
os.environ[ENV]=canonical(rank_binding)
sys.argv=[bridge_row["source_path"],*sys.argv[1:]]
module=types.ModuleType("__main__"); module.__file__=bridge_row["source_path"]; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module
exec(compile(bridge_source,bridge_row["source_path"],"exec",dont_inherit=True),module.__dict__)'''


def canonical_json_bytes(value: Any) -> bytes:
    return v1.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return v1.object_sha256(value)


_EXEC_IDENTITY_FIELDS = {
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
_EXEC_ROW_FIELDS = {"role", "fd", "source_path", "sha256", "identity"}
_OUTER_EXEC_ROLES = (
    "python_executable",
    "bridge_source",
    "adapter_source",
    "ffmpeg_executable",
)


def _stat_identity(info: os.stat_result) -> dict[str, int]:
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


def _pread_all(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size <= 0 or not hasattr(os, "pread"):
        raise MatchedRunnerV2Error("retained executable pread is unavailable")
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
        raise MatchedRunnerV2Error("retained executable read is incomplete")
    return raw


def validate_captured_runner_entry(
    value: Mapping[str, Any] | None = None,
    *,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    if value is None:
        raw = os.environ.get(ENTRY_AUTHORITY_ENV)
        if raw is None:
            raise MatchedRunnerV2Error(
                "captured runner entry authority is absent"
            )
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise MatchedRunnerV2Error(
                "captured runner entry authority is not JSON"
            ) from error
        if canonical_json_bytes(value).decode("utf-8") != raw:
            raise MatchedRunnerV2Error(
                "captured runner entry authority is not canonical JSON"
            )
    fields = {
        "schema_version",
        "runner_fd",
        "runner_path",
        "runner_sha256",
        "runner_identity",
        "python_fd",
        "python_path",
        "python_sha256",
        "python_identity",
        "release_digest",
        "bootstrap_sha256",
        "entry_method",
        "slurm_export_none_required",
        "bash_privileged_startup_required",
        "captured_source_entry",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MatchedRunnerV2Error("captured runner entry closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    runner_identity = row.get("runner_identity")
    python_identity = row.get("python_identity")
    runner_path = Path(row.get("runner_path", ""))
    python_path = Path(row.get("python_path", ""))
    if (
        row.get("schema_version") != ENTRY_AUTHORITY_SCHEMA
        or claimed != object_sha256(unsigned)
        or row.get("entry_method")
        != "slurm-spooled-or-trusted-stdin-held-python-fd-v1"
        or row.get("slurm_export_none_required") is not True
        or row.get("bash_privileged_startup_required") is not True
        or row.get("captured_source_entry") is not True
        or type(row.get("runner_fd")) is not int
        or row["runner_fd"] < 3
        or type(row.get("python_fd")) is not int
        or row["python_fd"] < 3
        or row["runner_fd"] == row["python_fd"]
        or not runner_path.is_absolute()
        or not python_path.is_absolute()
        or os.path.normpath(str(runner_path)) != str(runner_path)
        or os.path.normpath(str(python_path)) != str(python_path)
        or runner_path.resolve(strict=True) != runner_path
        or python_path.resolve(strict=True) != python_path
        or runner_path.is_symlink()
        or python_path.is_symlink()
        or not isinstance(runner_identity, Mapping)
        or set(runner_identity) != _EXEC_IDENTITY_FIELDS
        or not isinstance(python_identity, Mapping)
        or set(python_identity) != _EXEC_IDENTITY_FIELDS
        or any(
            type(item) is not int
            for identity in (runner_identity, python_identity)
            for item in identity.values()
        )
        or any(
            type(row.get(field)) is not str
            or v1.SHA256_RE.fullmatch(row[field]) is None
            for field in (
                "runner_sha256",
                "python_sha256",
                "release_digest",
                "bootstrap_sha256",
            )
        )
    ):
        raise MatchedRunnerV2Error("captured runner entry value differs")
    try:
        runner_before = os.fstat(row["runner_fd"])
        python_before = os.fstat(row["python_fd"])
        runner_raw = _pread_all(row["runner_fd"], runner_before.st_size)
        python_raw = _pread_all(row["python_fd"], python_before.st_size)
        runner_after = os.fstat(row["runner_fd"])
        python_after = os.fstat(row["python_fd"])
        runner_named = runner_path.lstat()
        python_named = python_path.lstat()
        process_python = os.stat("/proc/self/exe")
        runner_inheritable = os.get_inheritable(row["runner_fd"])
        python_inheritable = os.get_inheritable(row["python_fd"])
    except OSError as error:
        raise MatchedRunnerV2Error(
            "captured runner entry descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(runner_before.st_mode)
        or runner_before.st_nlink != 1
        or stat.S_IMODE(runner_before.st_mode) != 0o444
        or not stat.S_ISREG(python_before.st_mode)
        or python_before.st_nlink != 1
        or not python_before.st_mode & 0o111
        or _stat_identity(runner_before) != dict(runner_identity)
        or _stat_identity(runner_after) != dict(runner_identity)
        or _stat_identity(runner_named) != dict(runner_identity)
        or _stat_identity(python_before) != dict(python_identity)
        or _stat_identity(python_after) != dict(python_identity)
        or _stat_identity(python_named) != dict(python_identity)
        or _stat_identity(process_python) != dict(python_identity)
        or hashlib.sha256(runner_raw).hexdigest() != row["runner_sha256"]
        or hashlib.sha256(python_raw).hexdigest() != row["python_sha256"]
        or runner_inheritable
        or python_inheritable
        or Path(__file__).resolve(strict=True) != runner_path
        or (args is not None and args.runner_sha256 != row["runner_sha256"])
        or (args is not None and args.python_sha256 != row["python_sha256"])
        or (args is not None and Path(args.python) != python_path)
    ):
        raise MatchedRunnerV2Error("captured runner entry replay differs")
    row["runner_identity"] = dict(runner_identity)
    row["python_identity"] = dict(python_identity)
    return row


def close_captured_runner_entry(value: Mapping[str, Any] | None) -> None:
    closed: set[int] = set()
    if not isinstance(value, Mapping):
        return
    for field in ("runner_fd", "python_fd"):
        descriptor = value.get(field)
        if type(descriptor) is int and descriptor not in closed:
            closed.add(descriptor)
            try:
                os.close(descriptor)
            except OSError:
                pass


def _handoff_seal_mask() -> int:
    names = ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE")
    if any(not hasattr(fcntl, name) for name in names):
        raise MatchedRunnerV2Error("Linux memfd seals are unavailable")
    return sum(int(getattr(fcntl, name)) for name in names)


def _handoff_immutable_identity(info: os.stat_result) -> dict[str, int]:
    identity = _stat_identity(info)
    return {
        key: identity[key]
        for key in ("device", "inode", "uid", "gid", "mode", "nlink", "rdev")
    }


def validate_empty_publication_handoff(
    value: Mapping[str, Any], *, expected_inheritable: bool
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "fd",
        "initial_identity",
        "capacity",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MatchedRunnerV2Error("publication handoff authority closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    identity = row.get("initial_identity")
    if (
        row.get("schema_version") != PUBLICATION_HANDOFF_AUTHORITY_SCHEMA
        or claimed != object_sha256(unsigned)
        or type(row.get("task_id")) is not str
        or row["task_id"] not in TASK_IDS
        or type(row.get("fd")) is not int
        or row["fd"] < 3
        or type(row.get("capacity")) is not int
        or row["capacity"] != 65536
        or not isinstance(identity, Mapping)
        or set(identity) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in identity.values())
    ):
        raise MatchedRunnerV2Error("publication handoff authority differs")
    try:
        observed = os.fstat(row["fd"])
        seals = fcntl.fcntl(row["fd"], fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(row["fd"])
    except (OSError, AttributeError) as error:
        raise MatchedRunnerV2Error(
            "publication handoff descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 0
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != 0
        or _stat_identity(observed) != dict(identity)
        or seals != 0
        or inheritable is not expected_inheritable
    ):
        raise MatchedRunnerV2Error("empty publication handoff replay differs")
    row["initial_identity"] = dict(identity)
    return row


def create_publication_handoff(task_id: str) -> dict[str, Any]:
    if task_id not in TASK_IDS or not hasattr(os, "memfd_create"):
        raise MatchedRunnerV2Error("Linux publication handoff is unavailable")
    flags = int(getattr(os, "MFD_CLOEXEC", 0)) | int(
        getattr(os, "MFD_ALLOW_SEALING", 0)
    )
    if not flags & int(getattr(os, "MFD_ALLOW_SEALING", 0)):
        raise MatchedRunnerV2Error("sealable memfd creation is unavailable")
    descriptor = os.memfd_create(
        f"full644-matched-{task_id}-publication", flags
    )
    try:
        os.fchmod(descriptor, 0o600)
        os.set_inheritable(descriptor, False)
        row: dict[str, Any] = {
            "schema_version": PUBLICATION_HANDOFF_AUTHORITY_SCHEMA,
            "task_id": task_id,
            "fd": descriptor,
            "initial_identity": _stat_identity(os.fstat(descriptor)),
            "capacity": 65536,
        }
        row["authority_digest"] = object_sha256(row)
        return validate_empty_publication_handoff(
            row, expected_inheritable=False
        )
    except BaseException:
        os.close(descriptor)
        raise


def read_sealed_publication_handoff(
    value: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "task_id",
        "fd",
        "initial_identity",
        "capacity",
        "authority_digest",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MatchedRunnerV2Error("sealed handoff authority closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    initial = row.get("initial_identity")
    if (
        row.get("schema_version") != PUBLICATION_HANDOFF_AUTHORITY_SCHEMA
        or row.get("task_id") != task.get("task_id")
        or claimed != object_sha256(unsigned)
        or type(row.get("fd")) is not int
        or row["fd"] < 3
        or row.get("capacity") != 65536
        or not isinstance(initial, Mapping)
        or set(initial) != _EXEC_IDENTITY_FIELDS
    ):
        raise MatchedRunnerV2Error("sealed handoff authority differs")
    try:
        before = os.fstat(row["fd"])
        raw = _pread_all(row["fd"], before.st_size)
        after = os.fstat(row["fd"])
        seals = fcntl.fcntl(row["fd"], fcntl.F_GET_SEALS)
        inheritable = os.get_inheritable(row["fd"])
    except (OSError, AttributeError) as error:
        raise MatchedRunnerV2Error("sealed handoff is unavailable") from error
    if (
        _handoff_immutable_identity(before)
        != {
            key: initial[key]
            for key in ("device", "inode", "uid", "gid", "mode", "nlink", "rdev")
        }
        or _stat_identity(before) != _stat_identity(after)
        or before.st_size <= 0
        or before.st_size > row["capacity"]
        or seals != _handoff_seal_mask()
        or inheritable
    ):
        raise MatchedRunnerV2Error("sealed handoff replay differs")
    try:
        payload = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MatchedRunnerV2Error("sealed handoff is not JSON") from error
    if (
        not isinstance(payload, dict)
        or raw != canonical_json_bytes(payload) + b"\n"
    ):
        raise MatchedRunnerV2Error("sealed handoff is not canonical JSON")
    _validate_embedded_digest(payload, "payload_digest", label="publication handoff")
    output = task.get("output")
    identity_fields = _EXEC_IDENTITY_FIELDS
    if (
        set(payload)
        != {
            "schema_version",
            "task_id",
            "output_path",
            "output_identity",
            "output_sha256",
            "output_size",
            "receipt_path",
            "receipt_identity",
            "receipt_sha256",
            "receipt_size",
            "receipt_digest",
            "payload_digest",
        }
        or payload.get("schema_version") != PUBLICATION_HANDOFF_PAYLOAD_SCHEMA
        or payload.get("task_id") != row["task_id"]
        or not isinstance(output, Mapping)
        or payload.get("output_path") != output.get("video_path")
        or payload.get("receipt_path") != output.get("receipt_path")
        or not isinstance(payload.get("output_identity"), Mapping)
        or set(payload["output_identity"]) != identity_fields
        or not isinstance(payload.get("receipt_identity"), Mapping)
        or set(payload["receipt_identity"]) != identity_fields
        or any(
            type(item) is not int
            for field in ("output_identity", "receipt_identity")
            for item in payload[field].values()
        )
        or type(payload.get("output_sha256")) is not str
        or v1.SHA256_RE.fullmatch(payload["output_sha256"]) is None
        or type(payload.get("receipt_sha256")) is not str
        or v1.SHA256_RE.fullmatch(payload["receipt_sha256"]) is None
        or type(payload.get("receipt_digest")) is not str
        or v1.SHA256_RE.fullmatch(payload["receipt_digest"]) is None
        or type(payload.get("output_size")) is not int
        or payload["output_size"] <= 0
        or type(payload.get("receipt_size")) is not int
        or payload["receipt_size"] <= 0
    ):
        raise MatchedRunnerV2Error("sealed handoff payload differs")
    return payload


def close_publication_handoff(value: Mapping[str, Any] | None) -> None:
    if not isinstance(value, Mapping):
        return
    descriptor = value.get("fd")
    if type(descriptor) is int:
        try:
            os.close(descriptor)
        except OSError:
            pass


def capture_exec_authority(
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    role_to_identity = {
        "python_executable": identities.get("python"),
        "bridge_source": identities.get("bridge"),
        "adapter_source": identities.get("adapter"),
        "ffmpeg_executable": identities.get("ffmpeg"),
    }
    rows: list[dict[str, Any]] = []
    opened: list[int] = []
    try:
        for role in _OUTER_EXEC_ROLES:
            identity = role_to_identity[role]
            if not isinstance(identity, Mapping):
                raise MatchedRunnerV2Error("exec-authority identity is absent")
            path = Path(identity["path"])
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            opened.append(descriptor)
            os.set_inheritable(descriptor, False)
            before = os.fstat(descriptor)
            raw = _pread_all(descriptor, before.st_size)
            after = os.fstat(descriptor)
            named = path.lstat()
            projected = _stat_identity(before)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or projected != _stat_identity(after)
                or projected != _stat_identity(named)
                or hashlib.sha256(raw).hexdigest() != identity.get("sha256")
                or before.st_dev != identity.get("device")
                or before.st_ino != identity.get("inode")
                or before.st_uid != identity.get("uid")
                or before.st_gid != identity.get("gid")
                or stat.S_IMODE(before.st_mode) != identity.get("mode")
                or before.st_size != identity.get("size")
                or (
                    role in {"python_executable", "ffmpeg_executable"}
                    and not before.st_mode & 0o111
                )
            ):
                raise MatchedRunnerV2Error("exec-authority capture differs")
            if role in {"bridge_source", "adapter_source"}:
                try:
                    raw.decode("utf-8", "strict")
                except UnicodeDecodeError as error:
                    raise MatchedRunnerV2Error(
                        "captured executable source is not UTF-8"
                    ) from error
            rows.append(
                {
                    "role": role,
                    "fd": descriptor,
                    "source_path": str(path),
                    "sha256": identity["sha256"],
                    "identity": projected,
                }
            )
        if [row["fd"] for row in rows] != sorted(row["fd"] for row in rows):
            raise MatchedRunnerV2Error("exec-authority FD order differs")
        binding: dict[str, Any] = {
            "schema_version": EXEC_AUTHORITY_SCHEMA,
            "rows": rows,
            "rows_digest": object_sha256(rows),
        }
        binding["binding_digest"] = object_sha256(binding)
        return binding
    except BaseException:
        for descriptor in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def validate_exec_authority(
    value: Mapping[str, Any],
    *,
    expected_inheritable: bool,
    rehash: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedRunnerV2Error("exec-authority binding differs")
    binding = dict(value)
    unsigned = dict(binding)
    claimed = unsigned.pop("binding_digest", None)
    rows = binding.get("rows")
    if (
        set(binding)
        != {"schema_version", "rows", "rows_digest", "binding_digest"}
        or binding.get("schema_version") != EXEC_AUTHORITY_SCHEMA
        or type(rows) is not list
        or [row.get("role") if isinstance(row, Mapping) else None for row in rows]
        != list(_OUTER_EXEC_ROLES)
        or binding.get("rows_digest") != object_sha256(rows)
        or claimed != object_sha256(unsigned)
    ):
        raise MatchedRunnerV2Error("exec-authority digest differs")
    numbers: list[int] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != _EXEC_ROW_FIELDS
            or type(row.get("fd")) is not int
            or row["fd"] < 3
            or type(row.get("source_path")) is not str
            or not Path(row["source_path"]).is_absolute()
            or os.path.normpath(row["source_path"]) != row["source_path"]
            or type(row.get("sha256")) is not str
            or v1.SHA256_RE.fullmatch(row["sha256"]) is None
            or not isinstance(row.get("identity"), Mapping)
            or set(row["identity"]) != _EXEC_IDENTITY_FIELDS
            or any(type(item) is not int for item in row["identity"].values())
        ):
            raise MatchedRunnerV2Error("exec-authority row differs")
        try:
            before = os.fstat(row["fd"])
            inheritable = os.get_inheritable(row["fd"])
            raw = _pread_all(row["fd"], before.st_size) if rehash else None
            after = os.fstat(row["fd"])
        except OSError as error:
            raise MatchedRunnerV2Error("exec-authority FD is unavailable") from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _stat_identity(before) != dict(row["identity"])
            or _stat_identity(after) != dict(row["identity"])
            or inheritable is not expected_inheritable
            or (
                raw is not None
                and hashlib.sha256(raw).hexdigest() != row["sha256"]
            )
        ):
            raise MatchedRunnerV2Error("exec-authority FD replay differs")
        numbers.append(row["fd"])
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise MatchedRunnerV2Error("exec-authority FD allowlist differs")
    return binding


def close_exec_authority(value: Mapping[str, Any]) -> None:
    rows = value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        return
    for row in rows:
        descriptor = row.get("fd") if isinstance(row, Mapping) else None
        if type(descriptor) is int:
            try:
                os.close(descriptor)
            except OSError:
                pass


def capture_ffprobe_authority(
    identity: Mapping[str, Any], producer: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(identity["path"])
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        raw = _pread_all(descriptor, before.st_size)
        middle = os.fstat(descriptor)
        named = path.lstat()
        after = os.fstat(descriptor)
        projected = _stat_identity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not before.st_mode & 0o111
            or projected != _stat_identity(middle)
            or projected != _stat_identity(named)
            or projected != _stat_identity(after)
            or hashlib.sha256(raw).hexdigest() != identity.get("sha256")
        ):
            raise MatchedRunnerV2Error("ffprobe retained capture differs")
        value: dict[str, Any] = {
            "schema_version": v2.FFPROBE_AUTHORITY_SCHEMA,
            "fd": descriptor,
            "source_path": str(path),
            "sha256": identity["sha256"],
            "identity": projected,
        }
        value["authority_digest"] = object_sha256(value)
        return v2.validate_retained_ffprobe_authority(value, producer)
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def close_ffprobe_authority(value: Mapping[str, Any] | None) -> None:
    descriptor = value.get("fd") if isinstance(value, Mapping) else None
    if type(descriptor) is int:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _identity(path_value: str | Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    _, observed, size = v1._stable_file(
        path, expected_sha256=expected_sha256, return_bytes=False
    )
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or size <= 0
    ):
        raise MatchedRunnerV2Error(f"pinned file identity differs: {path}")
    return {
        "path": str(path),
        "sha256": observed,
        "size": size,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "nlink": info.st_nlink,
    }


def _canonical_source_authority(task: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(task["source_video"])
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise MatchedRunnerV2Error("task source path is not canonical")
    before = path.lstat()
    _, sha256, size = v1._stable_file(
        path,
        expected_sha256=task["source_video_sha256"],
        return_bytes=False,
    )
    after = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or v1._identity(before) != v1._identity(after)
        or size != before.st_size
    ):
        raise MatchedRunnerV2Error("task source stable identity differs")
    return v2._source_stat_projection(path, before, sha256)


def validate_task_order(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    v2.validate_plan(plan)
    tasks = plan.get("tasks")
    if (
        plan.get("production_ready") is not True
        or not isinstance(tasks, list)
        or len(tasks) != 16
        or tuple(task.get("task_id") for task in tasks) != TASK_IDS
    ):
        raise MatchedRunnerV2Error("matched task order/production closure differs")
    output_parents = {
        str(Path(task["output"]["video_path"]).parent) for task in tasks
    } | {
        str(Path(task["output"]["receipt_path"]).parent) for task in tasks
    }
    if len(output_parents) != 1:
        raise MatchedRunnerV2Error("all task outputs must share one publication root")
    root = Path(next(iter(output_parents)))
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise MatchedRunnerV2Error("task publication root differs")
    internal_paths: set[str] = set()
    for index, task in enumerate(tasks):
        output = task["output"]
        if (
            output.get("create_only") is not True
            or Path(output["receipt_path"])
            != Path(output["video_path"]).with_name(
                Path(output["video_path"]).name + ".receipt.json"
            )
            or Path(output["video_path"]).exists()
            or Path(output["video_path"]).is_symlink()
            or Path(output["receipt_path"]).exists()
            or Path(output["receipt_path"]).is_symlink()
        ):
            raise MatchedRunnerV2Error("planned output is not exact and fresh")
        prefix = f".matched-v2-{index:02d}-{task['task_id']}"
        suffixes = list(_BASE_ARTIFACT_SUFFIXES) + [
            ".log",
            "-runner-task.json",
        ]
        if task["arm"] == "full644":
            suffixes.extend(_ADAPTER_ARTIFACT_SUFFIXES)
        for suffix in suffixes:
            path = root / (prefix + suffix)
            if path.exists() or path.is_symlink() or str(path) in internal_paths:
                raise MatchedRunnerV2Error(
                    "planned internal task artifact is not exact and fresh"
                )
            internal_paths.add(str(path))
    return [dict(task) for task in tasks]


def select_campaign_tasks(
    plan: Mapping[str, Any], campaign_mode: str
) -> tuple[dict[str, Any], ...]:
    """Select only an exact frozen campaign; arbitrary subsets are forbidden."""

    full_tasks = validate_task_order(plan)
    if campaign_mode == FULL16_CAMPAIGN:
        selected = tuple(full_tasks)
        expected = TASK_IDS
    elif campaign_mode == CASE00_CANARY_CAMPAIGN:
        selected = tuple(full_tasks[:2])
        expected = CANARY_TASK_IDS
    else:
        raise MatchedRunnerV2Error("campaign mode differs")
    if tuple(task["task_id"] for task in selected) != expected:
        raise MatchedRunnerV2Error("campaign task selection differs")
    return selected


def task_input_digest(plan: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    return object_sha256(
        {
            "schema_version": "full644-exploratory-matched-task-input-v2",
            "plan_digest": plan["plan_digest"],
            "task": task,
        }
    )


def build_eval_consumption_chain(
    *,
    task_id: str,
    consumption_input_digest: str,
    model_capture_digest: str,
    model_pre_use_digest: str,
    model_post_use_digest: str,
    adapter_capture_digest: str | None,
    adapter_pre_use_digest: str | None,
    adapter_post_use_digest: str | None,
    adapter_final_digest: str | None,
    native_inference_receipt_digest: str,
    native_receipt_file_sha256: str,
    native_output_sha256: str,
) -> dict[str, Any]:
    required = (
        consumption_input_digest,
        model_capture_digest,
        model_pre_use_digest,
        model_post_use_digest,
        native_inference_receipt_digest,
        native_receipt_file_sha256,
        native_output_sha256,
    )
    optional = (
        adapter_capture_digest,
        adapter_pre_use_digest,
        adapter_post_use_digest,
        adapter_final_digest,
    )
    if (
        not isinstance(task_id, str)
        or not task_id
        or any(type(value) is not str or v1.SHA256_RE.fullmatch(value) is None for value in required)
        or (any(value is None for value in optional) and not all(value is None for value in optional))
        or any(
            value is not None
            and (type(value) is not str or v1.SHA256_RE.fullmatch(value) is None)
            for value in optional
        )
    ):
        raise MatchedRunnerV2Error("eval consumption-chain digest closure differs")
    row: dict[str, Any] = {
        "schema_version": CONSUMPTION_CHAIN_SCHEMA,
        "task_id": task_id,
        "consumption_input_digest": consumption_input_digest,
        "model_capture_digest": model_capture_digest,
        "model_pre_use_digest": model_pre_use_digest,
        "model_post_use_digest": model_post_use_digest,
        "adapter_capture_digest": adapter_capture_digest,
        "adapter_pre_use_digest": adapter_pre_use_digest,
        "adapter_post_use_digest": adapter_post_use_digest,
        "adapter_final_digest": adapter_final_digest,
        "native_inference_receipt_digest": native_inference_receipt_digest,
        "native_receipt_file_sha256": native_receipt_file_sha256,
        "native_output_sha256": native_output_sha256,
        "event_order": [
            "native_output_and_receipt_published",
            "adapter_post_use_replayed_or_base_control",
            "adapter_final_closed_or_base_control",
            "model_post_use_replayed",
            "eval_consumption_chain_sealed",
        ],
        "native_publication_completed_before_parent_post_use_replay": True,
        "parent_post_use_closed_before_native_publication": False,
        "all_post_use_replays_completed_before_runner_result": True,
        "training_loss_read_or_used": False,
    }
    row["consumption_digest"] = object_sha256(row)
    return row


def validate_eval_consumption_chain(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedRunnerV2Error("eval consumption chain differs")
    row = dict(value)
    expected = build_eval_consumption_chain(
        task_id=row.get("task_id"),
        consumption_input_digest=row.get("consumption_input_digest"),
        model_capture_digest=row.get("model_capture_digest"),
        model_pre_use_digest=row.get("model_pre_use_digest"),
        model_post_use_digest=row.get("model_post_use_digest"),
        adapter_capture_digest=row.get("adapter_capture_digest"),
        adapter_pre_use_digest=row.get("adapter_pre_use_digest"),
        adapter_post_use_digest=row.get("adapter_post_use_digest"),
        adapter_final_digest=row.get("adapter_final_digest"),
        native_inference_receipt_digest=row.get("native_inference_receipt_digest"),
        native_receipt_file_sha256=row.get("native_receipt_file_sha256"),
        native_output_sha256=row.get("native_output_sha256"),
    )
    if row != expected:
        raise MatchedRunnerV2Error("eval consumption chain digest differs")
    return row


def build_inference_arguments(
    *,
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
    bernini_root: str,
    veomni_root: str,
    model_view_root: str,
    consumption_input_path: str,
    consumption_input_sha256: str,
    consumption_input_digest: str,
    source_authority: Mapping[str, Any],
    adapter_view_root: str | None,
) -> list[str]:
    if task.get("task_id") not in TASK_IDS:
        raise MatchedRunnerV2Error("task identity differs")
    arguments = [
        "--bernini-root",
        bernini_root,
        "--veomni-root",
        veomni_root,
        "--checkpoint",
        model_view_root,
        "--source-video",
        task["source_video"],
        "--source-video-sha256",
        task["source_video_sha256"],
        "--source-video-authority",
        canonical_json_bytes(source_authority).decode("utf-8"),
        "--instruction",
        task["instruction"],
        "--output",
        task["output"]["video_path"],
        "--num-inference-steps",
        "40",
        "--seed",
        str(task["seed"]),
        "--source-onset-policy",
        "none",
        "--expected-bernini-commit",
        EXPECTED_BERNINI_COMMIT,
        "--expected-veomni-commit",
        EXPECTED_VEOMNI_COMMIT,
        "--expected-checkpoint-tree-sha256",
        EXPECTED_CHECKPOINT_TREE_SHA256,
        "--method-source-revision",
        plan["producer"]["method_source_revision"],
        "--method-source-archive-sha256",
        plan["producer"]["method_source_archive_sha256"],
        "--model-consumption-input",
        consumption_input_path,
        "--model-consumption-input-sha256",
        consumption_input_sha256,
        "--model-consumption-input-digest",
        consumption_input_digest,
        "--task-input-digest",
        task_input_digest(plan, task),
    ]
    if task["arm"] == "base":
        if adapter_view_root is not None or task.get("adapter") is not None:
            raise MatchedRunnerV2Error("base task unexpectedly binds an adapter")
        arguments.append("--base-only")
    elif task["arm"] == "full644":
        if not isinstance(adapter_view_root, str) or not adapter_view_root:
            raise MatchedRunnerV2Error("full644 task lacks retained adapter view")
        checkpoint = plan["checkpoint_manifest"]
        arguments.extend(
            [
                "--adapter-checkpoint",
                adapter_view_root,
                "--adapter-checkpoint-manifest",
                checkpoint["path"],
                "--adapter-checkpoint-manifest-sha256",
                checkpoint["sha256"],
            ]
        )
    else:
        raise MatchedRunnerV2Error("task arm differs")
    return arguments


def build_torchrun_argv(
    *,
    python_path: str,
    python_sha256: str,
    bridge_script: str,
    bridge_sha256: str,
    adapter_script: str,
    adapter_script_sha256: str,
    ffmpeg_executable: str,
    ffmpeg_executable_sha256: str,
    torchrun_source: str,
    torchrun_source_sha256: str,
    torchrun_handler_source: str,
    torchrun_handler_source_sha256: str,
    inference_arguments: Sequence[str],
) -> list[str]:
    return [
        python_path,
        "-I",
        "-S",
        "-B",
        "-c",
        ISOLATED_BRIDGE_BOOTSTRAP,
        "--bridge-sha256",
        bridge_sha256,
        "--adapter-script",
        adapter_script,
        "--adapter-script-sha256",
        adapter_script_sha256,
        "--rank-python",
        python_path,
        "--rank-python-sha256",
        python_sha256,
        "--ffmpeg-executable",
        ffmpeg_executable,
        "--ffmpeg-executable-sha256",
        ffmpeg_executable_sha256,
        "--torchrun-source",
        torchrun_source,
        "--torchrun-source-sha256",
        torchrun_source_sha256,
        "--torchrun-handler-source",
        torchrun_handler_source,
        "--torchrun-handler-source-sha256",
        torchrun_handler_source_sha256,
        "--",
        *inference_arguments,
    ]


def execute_task_sequence(
    tasks: Sequence[Mapping[str, Any]],
    executor: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    *,
    expected_task_ids: Sequence[str] = TASK_IDS,
) -> list[dict[str, Any]]:
    expected = tuple(expected_task_ids)
    if (
        expected not in {TASK_IDS, CANARY_TASK_IDS}
        or len(tasks) != len(expected)
        or tuple(task.get("task_id") for task in tasks) != expected
    ):
        raise MatchedRunnerV2Error("refusing a partial or reordered task sequence")
    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        result = executor(task, index)
        if not isinstance(result, Mapping) or result.get("task_id") != task["task_id"]:
            raise MatchedRunnerV2Error("task executor result differs")
        results.append(dict(result))
    if len(results) != len(expected):
        raise MatchedRunnerV2Error("not all matched tasks executed")
    return results


def _write_json_at(
    directory_fd: int,
    basename: str,
    value: Mapping[str, Any],
    *,
    mode: int = 0o400,
) -> tuple[Path, str]:
    if (
        not basename
        or basename in {".", ".."}
        or "/" in basename
        or "\x00" in basename
        or type(mode) is not int
        or mode not in {0o400, 0o444}
    ):
        raise MatchedRunnerV2Error("authority artifact basename differs")
    payload = canonical_json_bytes(value) + b"\n"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise MatchedRunnerV2Error(
            f"authority artifact is not fresh: {basename}"
        )
    try:
        # The final basename itself is the create-only reservation.  Mode 000
        # is deliberately non-authoritative: a failed writer leaves a
        # tombstone which every consumer rejects, and the run may not retry.
        # This works on NFS/Lustre without relying on renameat2 support.
        descriptor = os.open(basename, flags, 0, dir_fd=directory_fd)
    except FileExistsError as error:
        raise MatchedRunnerV2Error(
            "authority artifact is not fresh"
        ) from error
    committed = False
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise MatchedRunnerV2Error("authority artifact write made no progress")
            offset += written
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        replay = b""
        while len(replay) < len(payload):
            block = os.read(descriptor, len(payload) - len(replay))
            if not block:
                break
            replay += block
        after = os.fstat(descriptor)
        named = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        if (
            replay != payload
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_mode != after.st_mode
            or before.st_nlink != 1
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0
            or before.st_dev != named.st_dev
            or before.st_ino != named.st_ino
            or before.st_mode != named.st_mode
            or before.st_nlink != named.st_nlink
            or before.st_size != len(payload)
        ):
            raise MatchedRunnerV2Error("authority artifact replay differs")
        # Acceptance commit: this is intentionally the final fallible file
        # operation.  No fsync/stat/replay may run after the permission change;
        # downstream readers accept only the requested final mode plus digest.
        os.fchmod(descriptor, mode)
        committed = True
    finally:
        try:
            os.close(descriptor)
        except OSError:
            if not committed:
                raise
    return Path(f"/proc/self/fd/{directory_fd}/{basename}"), payload_sha256


def _directory_immutable_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": info.st_mode,
        "rdev": info.st_rdev,
    }


def _open_held_directory(path_value: str | Path) -> tuple[int, dict[str, int]]:
    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or not path.is_dir()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise MatchedRunnerV2Error("held directory path differs")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        named = path.lstat()
        identity = _directory_immutable_identity(before)
        if (
            not stat.S_ISDIR(before.st_mode)
            or identity != _directory_immutable_identity(named)
        ):
            raise MatchedRunnerV2Error("held directory identity differs")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _validate_held_directory(
    descriptor: int,
    path_value: str | Path,
    identity: Mapping[str, Any],
) -> None:
    path = Path(path_value)
    try:
        held = os.fstat(descriptor)
        named = path.lstat()
    except OSError as error:
        raise MatchedRunnerV2Error("held directory disappeared") from error
    if (
        path.is_symlink()
        or not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or _directory_immutable_identity(held) != dict(identity)
        or _directory_immutable_identity(named) != dict(identity)
        or os.get_inheritable(descriptor)
    ):
        raise MatchedRunnerV2Error("held directory replay differs")


def _fd_child_path(directory_fd: int, basename: str) -> Path:
    if (
        type(directory_fd) is not int
        or directory_fd < 0
        or not basename
        or basename in {".", ".."}
        or "/" in basename
        or "\x00" in basename
    ):
        raise MatchedRunnerV2Error("held child basename differs")
    return Path(f"/proc/self/fd/{directory_fd}/{basename}")


def _read_native_receipt(path: Path) -> tuple[dict[str, Any], str]:
    return v1._load_receipt(path)


def _read_native_receipt_at(
    directory_fd: int, logical_path: Path
) -> tuple[dict[str, Any], str, dict[str, int]]:
    basename = logical_path.name
    proc_path = _fd_child_path(directory_fd, basename)
    try:
        before = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        receipt, sha256 = _read_native_receipt(proc_path)
        after = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise MatchedRunnerV2Error("native receipt leaf is unavailable") from error
    identity = _stat_identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o400
        or identity != _stat_identity(after)
    ):
        raise MatchedRunnerV2Error("native receipt leaf authority differs")
    return receipt, sha256, identity


def _read_native_output_at(
    directory_fd: int, logical_path: Path
) -> tuple[str, int, dict[str, int]]:
    basename = logical_path.name
    proc_path = _fd_child_path(directory_fd, basename)
    try:
        before = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
        _, sha256, size = v1._stable_file(proc_path, return_bytes=False)
        after = os.stat(basename, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise MatchedRunnerV2Error("native output leaf is unavailable") from error
    identity = _stat_identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o444
        or size <= 0
        or identity != _stat_identity(after)
    ):
        raise MatchedRunnerV2Error("native output leaf authority differs")
    return sha256, size, identity


def _capture_native_publication_at(
    directory_fd: int, task: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open and retain the exact receipt/output leaf inodes through final audit."""

    output_value = task.get("output")
    if not isinstance(output_value, Mapping):
        raise MatchedRunnerV2Error("task publication closure differs")
    output_path = Path(output_value.get("video_path", ""))
    receipt_path = Path(output_value.get("receipt_path", ""))
    if (
        output_path.parent != receipt_path.parent
        or output_path.name in {"", ".", ".."}
        or receipt_path.name in {"", ".", ".."}
        or "/" in output_path.name
        or "/" in receipt_path.name
    ):
        raise MatchedRunnerV2Error("task publication paths differ")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    try:
        receipt_fd = os.open(receipt_path.name, flags, dir_fd=directory_fd)
        descriptors.append(receipt_fd)
        os.set_inheritable(receipt_fd, False)
        output_fd = os.open(output_path.name, flags, dir_fd=directory_fd)
        descriptors.append(output_fd)
        os.set_inheritable(output_fd, False)
        receipt_info = os.fstat(receipt_fd)
        output_info = os.fstat(output_fd)
        receipt_raw = _pread_all(receipt_fd, receipt_info.st_size)
        output_raw = _pread_all(output_fd, output_info.st_size)
        try:
            receipt = json.loads(receipt_raw.decode("utf-8", "strict"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise MatchedRunnerV2Error(
                "retained native receipt is not JSON"
            ) from error
        if (
            not isinstance(receipt, dict)
            or receipt_raw != canonical_json_bytes(receipt) + b"\n"
        ):
            raise MatchedRunnerV2Error(
                "retained native receipt is not canonical JSON"
            )
        authority: dict[str, Any] = {
            "schema_version": v2.PUBLICATION_AUTHORITY_SCHEMA,
            "task_id": task.get("task_id"),
            "output_path": str(output_path),
            "output_fd": output_fd,
            "output_identity": _stat_identity(output_info),
            "output_sha256": hashlib.sha256(output_raw).hexdigest(),
            "output_size": len(output_raw),
            "receipt_path": str(receipt_path),
            "receipt_fd": receipt_fd,
            "receipt_identity": _stat_identity(receipt_info),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_size": len(receipt_raw),
        }
        authority["authority_digest"] = object_sha256(authority)
        try:
            validated = v2.validate_retained_publication_authority(
                authority, task
            )
        except v2.MatchedEvalV2Error as error:
            raise MatchedRunnerV2Error(str(error)) from error
        descriptors.clear()
        return receipt, validated
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _close_publication_authorities(
    values: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    closed: set[int] = set()
    if not isinstance(values, Mapping):
        return
    for row in values.values():
        if not isinstance(row, Mapping):
            continue
        for field in ("receipt_fd", "output_fd"):
            descriptor = row.get(field)
            if type(descriptor) is int and descriptor not in closed:
                closed.add(descriptor)
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _sanitized_environment(
    *,
    inherited: Mapping[str, Any],
    exec_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
    rank_cache_root: Path,
) -> dict[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    if (
        not isinstance(job_id, str)
        or not job_id
        or job_id.strip() != job_id
        or not isinstance(step_id, str)
        or not step_id
        or step_id.strip() != step_id
    ):
        raise MatchedRunnerV2Error("exact child Slurm identity is absent")
    executable = validate_exec_authority(
        exec_authority,
        expected_inheritable=False,
        rehash=True,
    )
    handoff = validate_empty_publication_handoff(
        publication_handoff, expected_inheritable=False
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SLURM_JOB_ID": job_id,
        "SLURM_STEP_ID": step_id,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MODELING_BACKEND": "hf",
        "ROCR_VISIBLE_DEVICES": ROCR_PHYSICAL_GPU_MASK,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_TRIM_THRESHOLD_": "131072",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "FULL644_MATCHED_RANK_CACHE_ROOT": str(rank_cache_root),
        PYTHON_EXECUTABLE_BINDING_ENV: canonical_json_bytes(executable).decode(
            "utf-8"
        ),
        model_authority.INHERITED_FD_BINDING_ENV: (
            model_authority.inherited_fd_environment_value(inherited)
        ),
        PUBLICATION_HANDOFF_ENV: canonical_json_bytes(handoff).decode("utf-8"),
    }
    gpu47_visibility_contract(environment)
    return environment


def _open_log(output_root_fd: int, basename: str) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        return os.open(basename, flags, 0o600, dir_fd=output_root_fd)
    except FileExistsError as error:
        raise MatchedRunnerV2Error(f"task log is not fresh: {basename}") from error


def _run_subprocess(
    argv: Sequence[str],
    environment: Mapping[str, str],
    inherited: Mapping[str, Any],
    exec_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
    log_descriptor: int,
) -> int:
    binding = model_authority.validate_inherited_fd_binding(
        inherited,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    executable = validate_exec_authority(
        exec_authority,
        expected_inheritable=False,
        rehash=True,
    )
    handoff = validate_empty_publication_handoff(
        publication_handoff, expected_inheritable=False
    )
    authority_fds = model_authority.inherited_fd_numbers(binding)
    code_fds = tuple(row["fd"] for row in executable["rows"])
    if (
        set(authority_fds) & set(code_fds)
        or handoff["fd"] in set(authority_fds) | set(code_fds)
    ):
        raise MatchedRunnerV2Error("model/code/handoff FD allowlists overlap")
    pass_fds = tuple(sorted((*authority_fds, *code_fds, handoff["fd"])))
    python_row = executable["rows"][0]
    if python_row["role"] != "python_executable":
        raise MatchedRunnerV2Error("retained Python executable role differs")
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=log_descriptor,
        stderr=subprocess.STDOUT,
        shell=False,
        close_fds=True,
        pass_fds=pass_fds,
        executable=f"/proc/self/fd/{python_row['fd']}",
        env=dict(environment),
    )
    model_authority.validate_inherited_fd_binding(
        binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    validate_exec_authority(
        executable,
        expected_inheritable=False,
        rehash=True,
    )
    return_code = int(process.wait())
    model_authority.validate_inherited_fd_binding(
        binding,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    validate_exec_authority(
        executable,
        expected_inheritable=False,
        rehash=True,
    )
    if return_code != 0:
        validate_empty_publication_handoff(
            handoff, expected_inheritable=False
        )
    return return_code


class RunnerExecution:
    def __init__(
        self,
        args: argparse.Namespace,
        plan: Mapping[str, Any],
        tasks: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.args = args
        self.plan = dict(plan)
        campaign_mode = getattr(args, "campaign_mode", FULL16_CAMPAIGN)
        self.tasks = tuple(
            select_campaign_tasks(plan, campaign_mode)
            if tasks is None
            else tasks
        )
        if campaign_mode == FULL16_CAMPAIGN:
            expected_ids = TASK_IDS
        elif campaign_mode == CASE00_CANARY_CAMPAIGN:
            expected_ids = CANARY_TASK_IDS
        else:
            raise MatchedRunnerV2Error("campaign mode differs")
        if tuple(task.get("task_id") for task in self.tasks) != expected_ids:
            raise MatchedRunnerV2Error("campaign task selection differs")
        self.output_root = Path(self.tasks[0]["output"]["video_path"]).parent
        self.authority_root = Path(args.authority_root)
        self.rank_cache_root = Path(args.rank_cache_root)
        self.output_root_fd: int | None = None
        self.output_root_identity: dict[str, int] | None = None
        self.authority_root_fd: int | None = None
        self.model: model_authority.ModelAuthority | None = None
        self.task_results: list[dict[str, Any]] = []
        self.publication_authorities: dict[str, dict[str, Any]] = {}
        self.publication_handoffs: dict[str, dict[str, Any]] = {}
        self.exec_authority = validate_exec_authority(
            args.exec_authority,
            expected_inheritable=False,
            rehash=True,
        )
        self.ffprobe_authority = v2.validate_retained_ffprobe_authority(
            args.ffprobe_authority,
            self.plan["producer"],
        )
        ffmpeg_rows = [
            row
            for row in self.exec_authority["rows"]
            if row["role"] == "ffmpeg_executable"
        ]
        if len(ffmpeg_rows) != 1:
            raise MatchedRunnerV2Error("ffmpeg exec-authority row differs")
        self.ffmpeg_exec_authority_digest = object_sha256(ffmpeg_rows[0])
        for root, label in (
            (self.authority_root, "authority root"),
            (self.rank_cache_root, "rank-cache root"),
        ):
            if (
                not root.is_absolute()
                or os.path.normpath(str(root)) != str(root)
                or root.exists()
                or root.is_symlink()
                or not root.parent.is_dir()
                or root.parent.is_symlink()
                or root.parent.resolve(strict=True) != root.parent
            ):
                raise MatchedRunnerV2Error(f"{label} is not fresh")
            root.mkdir(mode=0o700)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            self.output_root_fd = os.open(self.output_root, directory_flags)
            os.set_inheritable(self.output_root_fd, False)
            self.output_root_identity = _directory_immutable_identity(
                os.fstat(self.output_root_fd)
            )
            _validate_held_directory(
                self.output_root_fd,
                self.output_root,
                self.output_root_identity,
            )
            self.authority_root_fd = os.open(self.authority_root, directory_flags)
            os.set_inheritable(self.authority_root_fd, False)
        except BaseException:
            self.close_descriptors()
            raise

    def close_descriptors(self) -> None:
        _close_publication_authorities(self.publication_authorities)
        self.publication_authorities = {}
        for row in self.publication_handoffs.values():
            close_publication_handoff(row)
        self.publication_handoffs = {}
        for field in ("authority_root_fd", "output_root_fd"):
            descriptor = getattr(self, field, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                setattr(self, field, None)
        close_exec_authority(self.exec_authority)
        close_ffprobe_authority(self.ffprobe_authority)
        self.ffprobe_authority = None

    def capture_model(self) -> None:
        model_root = Path(self.args.model_root).resolve(strict=True)
        sample = (model_root / "config.json").lstat()
        self.model = model_authority.ModelAuthority.capture(
            model_root=model_root,
            manifest_path=self.args.model_manifest,
            private_parent=self.authority_root,
            private_parent_fd=self.authority_root_fd,
            view_name="model_fd_view",
            expected_uid=sample.st_uid,
            expected_gid=sample.st_gid,
            expected_device=sample.st_dev,
            expected_manifest_sha256=self.args.model_manifest_sha256,
            expected_file_mode=0o644,
        )

    def execute_one(self, task: Mapping[str, Any], index: int) -> dict[str, Any]:
        if (
            self.model is None
            or self.output_root_fd is None
            or self.output_root_identity is None
        ):
            raise MatchedRunnerV2Error("model authority is absent")
        _validate_held_directory(
            self.output_root_fd,
            self.output_root,
            self.output_root_identity,
        )
        task_id = task["task_id"]
        prefix = f".matched-v2-{index:02d}-{task_id}"
        model_pre = self.model.begin_task(task_id)
        adapter: model_authority.AdapterAuthority | None = None
        adapter_pre: dict[str, Any] | None = None
        adapter_post: dict[str, Any] | None = None
        adapter_final: dict[str, Any] | None = None
        artifact_refs: dict[str, dict[str, str]] = {}

        def write_artifact(
            role: str, suffix: str, value: Mapping[str, Any]
        ) -> tuple[Path, str]:
            basename = prefix + suffix
            path, sha256 = _write_json_at(
                self.output_root_fd,
                basename,
                value,
            )
            artifact_refs[role] = {"basename": basename, "sha256": sha256}
            return path, sha256

        try:
            if task["arm"] == "full644":
                checkpoint_root = Path(self.plan["checkpoint_manifest"]["path"]).parent
                receipt_info = (checkpoint_root / "receipt.json").lstat()
                adapter = model_authority.AdapterAuthority.capture(
                    task_id=task_id,
                    checkpoint_root=checkpoint_root,
                    expected_sha256={
                        "receipt.json": self.plan["checkpoint_manifest"][
                            "training_receipt_sha256"
                        ],
                        "adapter/README.md": (
                            TERMINAL_CP644_ADAPTER_README_SHA256
                        ),
                        "adapter/adapter_config.json": self.plan[
                            "checkpoint_manifest"
                        ]["adapter_config_sha256"],
                        "adapter/adapter_model.safetensors": self.plan[
                            "checkpoint_manifest"
                        ]["adapter_model_sha256"],
                    },
                    private_parent=self.authority_root,
                    private_parent_fd=self.authority_root_fd,
                    view_name=f"adapter_fd_view_{index:02d}",
                    expected_uid=receipt_info.st_uid,
                    expected_gid=receipt_info.st_gid,
                    # The v5 worker runs under umask 077; PEFT payloads and
                    # the NamedTemporaryFile training receipt are therefore
                    # sealed as owner-only 0600 files in the final hardlink
                    # publication.
                    expected_file_mode=0o600,
                )
                adapter_pre = adapter.begin_use()
            model_capture_path, model_capture_sha = write_artifact(
                "model_capture",
                "-model-capture.json",
                self.model.capture_receipt,
            )
            write_artifact("model_pre_use", "-model-pre-use.json", model_pre)
            adapter_capture_path: Path | None = None
            adapter_capture_sha: str | None = None
            if adapter is not None:
                adapter_capture_path, adapter_capture_sha = write_artifact(
                    "adapter_capture",
                    "-adapter-capture.json",
                    adapter.capture_receipt,
                )
                if adapter_pre is None:
                    raise MatchedRunnerV2Error("adapter pre-use replay is absent")
                write_artifact(
                    "adapter_pre_use", "-adapter-pre-use.json", adapter_pre
                )
            task_root_binding = model_authority.task_publication_root_binding(
                descriptor=self.output_root_fd, path=self.output_root
            )
            inherited = model_authority.build_inherited_fd_binding(
                task_id=task_id,
                model_capture=self.model.capture_receipt,
                adapter_capture=(None if adapter is None else adapter.capture_receipt),
                task_publication_root=task_root_binding,
            )
            consumption = model_authority.build_consumption_input(
                task_id=task_id,
                physical_bindings_digest=self.args.physical_bindings_digest,
                model_capture=self.model.capture_receipt,
                model_pre_use=model_pre,
                model_capture_receipt_path=model_capture_path,
                model_capture_receipt_sha256=model_capture_sha,
                adapter_capture=(None if adapter is None else adapter.capture_receipt),
                adapter_pre_use=adapter_pre,
                adapter_capture_receipt_path=adapter_capture_path,
                adapter_capture_receipt_sha256=adapter_capture_sha,
                inherited_fd_binding=inherited,
                task_publication_root=task_root_binding,
                production_mode=True,
            )
            consumption_path, consumption_sha = write_artifact(
                "consumption_input",
                "-consumption-input.json",
                consumption,
            )
            source_authority = _canonical_source_authority(task)
            inference_arguments = build_inference_arguments(
                plan=self.plan,
                task=task,
                bernini_root=self.args.bernini_root,
                veomni_root=self.args.veomni_root,
                model_view_root=consumption["model"]["view_root"],
                consumption_input_path=str(consumption_path),
                consumption_input_sha256=consumption_sha,
                consumption_input_digest=consumption["consumption_input_digest"],
                source_authority=source_authority,
                adapter_view_root=(
                    None if adapter is None else consumption["adapter"]["view_root"]
                ),
            )
            argv = build_torchrun_argv(
                python_path=self.args.python,
                python_sha256=self.args.python_sha256,
                bridge_script=self.args.bridge_script,
                bridge_sha256=self.args.bridge_script_sha256,
                adapter_script=self.args.adapter_script,
                adapter_script_sha256=self.args.adapter_script_sha256,
                ffmpeg_executable=self.args.ffmpeg_executable,
                ffmpeg_executable_sha256=self.args.ffmpeg_executable_sha256,
                torchrun_source=self.args.torchrun_source,
                torchrun_source_sha256=self.args.torchrun_source_sha256,
                torchrun_handler_source=self.args.torchrun_handler_source,
                torchrun_handler_source_sha256=(
                    self.args.torchrun_handler_source_sha256
                ),
                inference_arguments=inference_arguments,
            )
            task_cache_root = self.rank_cache_root / f"task-{index:02d}-{task_id}"
            task_cache_root.mkdir(mode=0o700)
            if task_id in self.publication_handoffs:
                raise MatchedRunnerV2Error(
                    "task publication handoff already exists"
                )
            publication_handoff = create_publication_handoff(task_id)
            self.publication_handoffs[task_id] = publication_handoff
            environment = _sanitized_environment(
                inherited=inherited,
                exec_authority=self.exec_authority,
                publication_handoff=publication_handoff,
                rank_cache_root=task_cache_root,
            )
            log_name = prefix + ".log"
            log_descriptor = _open_log(self.output_root_fd, log_name)
            try:
                return_code = _run_subprocess(
                    argv,
                    environment,
                    inherited,
                    self.exec_authority,
                    publication_handoff,
                    log_descriptor,
                )
                os.fsync(log_descriptor)
                os.fchmod(log_descriptor, 0o400)
            finally:
                os.close(log_descriptor)
            if return_code != 0:
                raise MatchedRunnerV2Error(
                    f"task {task_id} returned {return_code}; retries are forbidden"
                )
            handoff_payload = read_sealed_publication_handoff(
                publication_handoff, task
            )
            native, publication_authority = _capture_native_publication_at(
                self.output_root_fd,
                task,
            )
            if task_id in self.publication_authorities:
                _close_publication_authorities(
                    {task_id: publication_authority}
                )
                raise MatchedRunnerV2Error(
                    "task retained publication authority already exists"
                )
            self.publication_authorities[task_id] = publication_authority
            native_file_sha = publication_authority["receipt_sha256"]
            native_receipt_identity = publication_authority["receipt_identity"]
            native_output_sha = publication_authority["output_sha256"]
            native_output_size = publication_authority["output_size"]
            native_output_identity = publication_authority["output_identity"]
            if any(
                handoff_payload[field] != publication_authority[target]
                for field, target in (
                    ("output_path", "output_path"),
                    ("output_identity", "output_identity"),
                    ("output_sha256", "output_sha256"),
                    ("output_size", "output_size"),
                    ("receipt_path", "receipt_path"),
                    ("receipt_identity", "receipt_identity"),
                    ("receipt_sha256", "receipt_sha256"),
                    ("receipt_size", "receipt_size"),
                )
            ):
                raise MatchedRunnerV2Error(
                    "child publication handoff/parent capture differs"
                )
            if (
                native.get("receipt_digest") is None
                or handoff_payload.get("receipt_digest")
                != native.get("receipt_digest")
                or native.get("output", {}).get("sha256") != native_output_sha
                or native.get("output", {}).get("size") != native_output_size
                or native.get("model_consumption", {}).get(
                    "consumption_input_digest"
                )
                != consumption["consumption_input_digest"]
            ):
                raise MatchedRunnerV2Error("native publication/consumption binding differs")
            if adapter is not None:
                adapter_post = adapter.end_use()
                adapter_final = adapter.finalize_and_close()
            model_post = self.model.end_task(task_id)
            if adapter_post is not None:
                write_artifact(
                    "adapter_post_use", "-adapter-post-use.json", adapter_post
                )
            if adapter_final is not None:
                write_artifact(
                    "adapter_final", "-adapter-final.json", adapter_final
                )
            write_artifact("model_post_use", "-model-post-use.json", model_post)
            chain = build_eval_consumption_chain(
                task_id=task_id,
                consumption_input_digest=consumption["consumption_input_digest"],
                model_capture_digest=self.model.capture_digest,
                model_pre_use_digest=model_pre["use_digest"],
                model_post_use_digest=model_post["use_digest"],
                adapter_capture_digest=(
                    None if adapter is None else adapter.capture_digest
                ),
                adapter_pre_use_digest=(
                    None if adapter_pre is None else adapter_pre["use_digest"]
                ),
                adapter_post_use_digest=(
                    None if adapter_post is None else adapter_post["use_digest"]
                ),
                adapter_final_digest=(
                    None
                    if adapter_final is None
                    else adapter_final["adapter_final_digest"]
                ),
                native_inference_receipt_digest=native["receipt_digest"],
                native_receipt_file_sha256=native_file_sha,
                native_output_sha256=native_output_sha,
            )
            validate_eval_consumption_chain(chain)
            write_artifact(
                "eval_consumption_chain", "-eval-consumption-chain.json", chain
            )
            self.model.record_task_consumption(chain["consumption_digest"])
            result: dict[str, Any] = {
                "schema_version": TASK_SCHEMA,
                "task_index": index,
                "task_id": task_id,
                "arm": task["arm"],
                "plan_digest": self.plan["plan_digest"],
                "task_input_digest": task_input_digest(self.plan, task),
                "argv_digest": object_sha256(argv),
                "environment_digest": object_sha256(environment),
                "ffmpeg_exec_authority_digest": (
                    self.ffmpeg_exec_authority_digest
                ),
                "publication_handoff_authority_digest": (
                    publication_handoff["authority_digest"]
                ),
                "publication_handoff_payload_digest": handoff_payload[
                    "payload_digest"
                ],
                "return_code": 0,
                "attempt_count": 1,
                "retry_allowed": False,
                "model_capture_digest": self.model.capture_digest,
                "adapter_capture_digest": (
                    None if adapter is None else adapter.capture_digest
                ),
                "consumption_input_digest": consumption[
                    "consumption_input_digest"
                ],
                "consumption_digest": chain["consumption_digest"],
                "native_receipt_digest": native["receipt_digest"],
                "native_receipt_file_sha256": native_file_sha,
                "native_output_sha256": native_output_sha,
                "native_output_size": native_output_size,
                "native_receipt_identity": native_receipt_identity,
                "native_output_identity": native_output_identity,
                "output_path": task["output"]["video_path"],
                "receipt_path": task["output"]["receipt_path"],
                "log_basename": log_name,
                "authority_artifacts": artifact_refs,
                "native_publication_completed_before_parent_post_use_replay": True,
                "parent_post_use_closed_before_native_publication": False,
                "post_use_replay_complete": True,
            }
            result["task_result_digest"] = object_sha256(result)
            _write_json_at(
                self.output_root_fd,
                prefix + "-runner-task.json",
                result,
            )
            _validate_held_directory(
                self.output_root_fd,
                self.output_root,
                self.output_root_identity,
            )
            return result
        except BaseException:
            if adapter is not None and not getattr(adapter, "_closed", False):
                adapter.abort(reason="matched task failed before adapter final replay")
            raise

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            self.capture_model()
            if self.model is None:
                raise MatchedRunnerV2Error("model capture failed")
            expected_ids = tuple(task["task_id"] for task in self.tasks)
            self.task_results = execute_task_sequence(
                self.tasks,
                self.execute_one,
                expected_task_ids=expected_ids,
            )
            model_final = self.model.finalize(
                expected_task_count=len(expected_ids)
            )
            self.model.close()
            return self.task_results, model_final
        except BaseException:
            if self.model is not None and not getattr(self.model, "_closed", False):
                self.model.abort(reason="matched evaluation failed before final rehash")
            raise


def _exact_gpu_indices(raw: str | None, count: int, *, label: str) -> list[int]:
    expected = list(range(count))
    if raw == f"0-{count - 1}":
        return expected
    if not isinstance(raw, str) or not raw:
        raise MatchedRunnerV2Error(f"{label} GPU authority is absent")
    parts = raw.split(",")
    try:
        values = [int(value) for value in parts]
    except ValueError as error:
        raise MatchedRunnerV2Error(f"{label} GPU authority differs") from error
    if values != expected or parts != [str(value) for value in expected]:
        raise MatchedRunnerV2Error(f"{label} GPU authority differs")
    return values


def _allocation_authority(
    holder_job_id: str, expected_node: str, expected_gpu_count: int
) -> dict[str, Any]:
    observed_job = os.environ.get("SLURM_JOB_ID")
    observed_node = socket.gethostname().split(".", 1)[0]
    if (
        observed_job != holder_job_id
        or observed_node != expected_node
        or type(expected_gpu_count) is not int
        or expected_gpu_count != 8
    ):
        raise MatchedRunnerV2Error("runner is outside the pinned allocation/node")
    step_id = os.environ.get("SLURM_STEP_ID")
    gpu_count_raw = os.environ.get("SLURM_GPUS_ON_NODE")
    gpus_per_node_raw = os.environ.get("SLURM_GPUS_PER_NODE")
    step_gpu_raw = os.environ.get("SLURM_STEP_GPUS")
    node_count_raw = os.environ.get("SLURM_NNODES")
    step_node_count_raw = os.environ.get("SLURM_STEP_NUM_NODES")
    job_nodelist_raw = os.environ.get("SLURM_JOB_NODELIST")
    step_nodelist_raw = os.environ.get("SLURM_STEP_NODELIST")
    step_gpus = _exact_gpu_indices(
        step_gpu_raw, expected_gpu_count, label="Slurm step"
    )
    if (
        not isinstance(step_id, str)
        or not step_id
        or step_id.strip() != step_id
        or not step_id.isascii()
        or not step_id.isdecimal()
        or int(step_id) <= 0
        or str(int(step_id)) != step_id
        or gpu_count_raw != str(expected_gpu_count)
        or gpus_per_node_raw != str(expected_gpu_count)
        or step_gpu_raw != "0,1,2,3,4,5,6,7"
        or node_count_raw != "1"
        or step_node_count_raw != "1"
        or job_nodelist_raw != expected_node
        or step_nodelist_raw != expected_node
        or "SLURM_JOB_GPUS" in os.environ
        or "SLURM_JOB_NUM_NODES" in os.environ
    ):
        raise MatchedRunnerV2Error("Slurm step authority is absent")
    gpu_visibility = gpu47_visibility_contract(
        {"ROCR_VISIBLE_DEVICES": ROCR_PHYSICAL_GPU_MASK}
    )
    return {
        "holder_job_id": holder_job_id,
        "node": expected_node,
        "slurm_step_id": step_id,
        "slurm_environment_source_names": {
            "job_id": "SLURM_JOB_ID",
            "step_id": "SLURM_STEP_ID",
            "gpu_count": "SLURM_GPUS_ON_NODE",
            "gpus_per_node": "SLURM_GPUS_PER_NODE",
            "step_gpu_indices": "SLURM_STEP_GPUS",
            "job_node_count": "SLURM_NNODES",
            "step_node_count": "SLURM_STEP_NUM_NODES",
            "job_nodelist": "SLURM_JOB_NODELIST",
            "step_nodelist": "SLURM_STEP_NODELIST",
        },
        "slurm_environment_raw_values": {
            "SLURM_JOB_ID": observed_job,
            "SLURM_STEP_ID": step_id,
            "SLURM_GPUS_ON_NODE": gpu_count_raw,
            "SLURM_GPUS_PER_NODE": gpus_per_node_raw,
            "SLURM_STEP_GPUS": step_gpu_raw,
            "SLURM_NNODES": node_count_raw,
            "SLURM_STEP_NUM_NODES": step_node_count_raw,
            "SLURM_JOB_NODELIST": job_nodelist_raw,
            "SLURM_STEP_NODELIST": step_nodelist_raw,
        },
        "slurm_observed_absent_fields": [
            "SLURM_JOB_GPUS",
            "SLURM_JOB_NUM_NODES",
        ],
        "normalized_slurm_authority": {
            "job_node_count": 1,
            "step_node_count": 1,
            "gpu_count_on_node": expected_gpu_count,
            "gpus_per_node": expected_gpu_count,
            "step_gpu_indices": step_gpus,
            "job_node": expected_node,
            "step_node": expected_node,
        },
        "world_size": 4,
        "ulysses_size": 4,
        "reserved_gpu_count": expected_gpu_count,
        "visible_gpu_indices": list(PHYSICAL_GPU_INDICES),
        "logical_gpu_indices": list(LOGICAL_GPU_INDICES),
        "gpu_visibility_contract": gpu_visibility,
        "step_reserves_all_eight_while_world4_uses_physical_4_7": True,
        "gpu_isolation_scope": (
            "ROCr-only logical isolation; not device-cgroup hard isolation"
        ),
    }


def _preflight_final_artifacts(
    args: argparse.Namespace, tasks: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    values = {
        "output_report": args.output_report,
        "runner_attestation": args.runner_attestation,
    }
    normalized: dict[str, str] = {}
    occupied = {
        str(Path(task["output"][field]))
        for task in tasks
        for field in ("video_path", "receipt_path")
    }
    for index, task in enumerate(tasks):
        prefix = f".matched-v2-{index:02d}-{task['task_id']}"
        suffixes = list(_BASE_ARTIFACT_SUFFIXES) + [
            ".log",
            "-runner-task.json",
        ]
        if task.get("arm") == "full644":
            suffixes.extend(_ADAPTER_ARTIFACT_SUFFIXES)
        occupied.update(
            str(Path(task["output"]["video_path"]).parent / (prefix + suffix))
            for suffix in suffixes
        )
    for label, raw in values.items():
        path = Path(raw).expanduser()
        if (
            not path.is_absolute()
            or os.path.normpath(str(path)) != str(path)
            or path.name in {"", ".", ".."}
            or path.exists()
            or path.is_symlink()
            or not path.parent.is_dir()
            or path.parent.is_symlink()
            or path.parent.resolve(strict=True) != path.parent
            or str(path) in occupied
        ):
            raise MatchedRunnerV2Error(f"{label} is not one fresh canonical path")
        normalized[label] = str(path)
    if len(set(normalized.values())) != len(normalized):
        raise MatchedRunnerV2Error("final report/attestation paths overlap")
    return normalized


def _hold_final_artifact_parents(
    final_artifacts: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    held: dict[str, dict[str, Any]] = {}
    try:
        for label in ("output_report", "runner_attestation"):
            path = Path(final_artifacts[label])
            descriptor, identity = _open_held_directory(path.parent)
            try:
                os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                os.close(descriptor)
                raise MatchedRunnerV2Error(
                    f"{label} is not fresh through retained parent"
                )
            held[label] = {
                "path": path,
                "parent_fd": descriptor,
                "parent_identity": identity,
            }
        return held
    except BaseException:
        for row in held.values():
            try:
                os.close(row["parent_fd"])
            except OSError:
                pass
        raise


def _validate_final_parent(row: Mapping[str, Any]) -> None:
    path = row.get("path")
    descriptor = row.get("parent_fd")
    identity = row.get("parent_identity")
    if not isinstance(path, Path) or type(descriptor) is not int or not isinstance(identity, Mapping):
        raise MatchedRunnerV2Error("final parent binding differs")
    _validate_held_directory(descriptor, path.parent, identity)


def _close_final_parents(held: Mapping[str, Mapping[str, Any]]) -> None:
    closed: set[int] = set()
    for row in held.values():
        descriptor = row.get("parent_fd")
        if type(descriptor) is int and descriptor not in closed:
            closed.add(descriptor)
            try:
                os.close(descriptor)
            except OSError:
                pass


def _validate_embedded_digest(
    value: Mapping[str, Any], field: str, *, label: str
) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if (
        type(claimed) is not str
        or v1.SHA256_RE.fullmatch(claimed) is None
        or claimed != object_sha256(unsigned)
    ):
        raise MatchedRunnerV2Error(f"{label} embedded digest differs")
    return claimed


def _load_canonical_0400_json(path: Path) -> tuple[dict[str, Any], str]:
    raw, sha256, _ = v1._stable_file(path, return_bytes=True)
    info = path.lstat()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MatchedRunnerV2Error("runner artifact is not JSON") from error
    if (
        not isinstance(value, dict)
        or raw != canonical_json_bytes(value) + b"\n"
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise MatchedRunnerV2Error("runner artifact stable replay differs")
    return value, sha256


def validate_gpu47_admission_receipt(
    path_value: str | Path,
    expected_sha256: str,
    *,
    holder_job_id: str,
    expected_node: str,
    expected_slurm_step_id: str,
    campaign_mode: str,
    plan_sha256: str,
    output_root: Path,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Replay the fresh external GPU inventory/mapping admission receipt."""

    path = Path(path_value)
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
    ):
        raise MatchedRunnerV2Error("GPU47 admission path differs")
    value, observed_sha256 = _load_canonical_0400_json(path)
    fields = {
        "schema_version",
        "status",
        "holder_job_id",
        "node",
        "slurm_step_id",
        "observed_at_unix_ns",
        "intended_campaign_mode",
        "intended_plan_sha256",
        "intended_output_root",
        "single_use",
        "slurm_job_gpu_indices",
        "slurm_step_gpu_indices",
        "slurm_gpus_on_node",
        "rocr_visible_devices",
        "secondary_visibility_variables_absent",
        "physical_inventory",
        "all8_runtime_devices",
        "visible_runtime_devices",
        "visible_physical_indices",
        "excluded_physical_indices",
        "torch_visible_device_count",
        "hip_visible_device_count",
        "pci_bus_is_authoritative_join_key",
        "physical_unique_id_replayed",
        "hip_uuid_replayed",
        "logical_order_is_observation_only",
        "external_workload_baseline",
        "isolation_scope",
        "device_cgroup_hard_isolation_claimed",
        "canary_or_model_execution_performed",
        "receipt_digest",
    }
    unsigned = dict(value)
    receipt_digest = unsigned.pop("receipt_digest", None)
    observed_at = value.get("observed_at_unix_ns")
    current_ns = time.time_ns() if now_ns is None else now_ns
    if (
        observed_sha256 != expected_sha256
        or set(value) != fields
        or value.get("schema_version") != GPU_ADMISSION_SCHEMA
        or value.get("status") != "PASS"
        or value.get("holder_job_id") != holder_job_id
        or value.get("node") != expected_node
        or value.get("slurm_step_id") != expected_slurm_step_id
        or type(observed_at) is not int
        or observed_at <= 0
        or current_ns < observed_at
        or current_ns - observed_at > GPU_ADMISSION_MAX_AGE_NS
        or value.get("intended_campaign_mode") != campaign_mode
        or value.get("intended_plan_sha256") != plan_sha256
        or value.get("intended_output_root") != str(output_root)
        or value.get("single_use") is not True
        or value.get("slurm_job_gpu_indices") != list(range(8))
        or value.get("slurm_step_gpu_indices") != list(range(8))
        or value.get("slurm_gpus_on_node") != 8
        or value.get("rocr_visible_devices") != ROCR_PHYSICAL_GPU_MASK
        or value.get("secondary_visibility_variables_absent") is not True
        or value.get("visible_physical_indices") != list(PHYSICAL_GPU_INDICES)
        or value.get("excluded_physical_indices") != [0, 1, 2, 3]
        or value.get("torch_visible_device_count") != 4
        or value.get("hip_visible_device_count") != 4
        or value.get("pci_bus_is_authoritative_join_key") is not True
        or value.get("physical_unique_id_replayed") is not True
        or value.get("hip_uuid_replayed") is not True
        or value.get("logical_order_is_observation_only") is not True
        or value.get("isolation_scope")
        != "ROCr-only logical isolation; not device-cgroup hard isolation"
        or value.get("device_cgroup_hard_isolation_claimed") is not False
        or value.get("canary_or_model_execution_performed") is not False
        or receipt_digest != object_sha256(unsigned)
    ):
        raise MatchedRunnerV2Error("GPU47 admission receipt closure differs")

    physical = value.get("physical_inventory")
    all8 = value.get("all8_runtime_devices")
    visible = value.get("visible_runtime_devices")
    physical_fields = {
        "physical_index",
        "pci_bus_id",
        "rocm_unique_id",
        "hip_uuid_hex",
    }
    runtime_fields = physical_fields | {"logical_index"}

    def valid_identity(row: Any, expected_fields: set[str]) -> bool:
        return (
            isinstance(row, Mapping)
            and set(row) == expected_fields
            and type(row.get("physical_index")) is int
            and row["physical_index"] in range(8)
            and re.fullmatch(
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                str(row.get("pci_bus_id")),
            )
            is not None
            and re.fullmatch(r"[0-9a-f]{8,64}", str(row.get("rocm_unique_id")))
            is not None
            and set(str(row["rocm_unique_id"])) != {"0"}
            and re.fullmatch(r"[0-9a-f]{32}", str(row.get("hip_uuid_hex")))
            is not None
            and str(row["hip_uuid_hex"]) != "0" * 32
            and (
                "logical_index" not in expected_fields
                or type(row.get("logical_index")) is int
            )
        )

    if (
        not isinstance(physical, list)
        or len(physical) != 8
        or not isinstance(all8, list)
        or len(all8) != 8
        or not isinstance(visible, list)
        or len(visible) != 4
        or any(not valid_identity(row, physical_fields) for row in physical)
        or any(not valid_identity(row, runtime_fields) for row in all8)
        or any(not valid_identity(row, runtime_fields) for row in visible)
        or [row["physical_index"] for row in physical] != list(range(8))
        or [row["logical_index"] for row in all8] != list(range(8))
        or [row["logical_index"] for row in visible] != list(range(4))
        or {row["physical_index"] for row in all8} != set(range(8))
        or {row["physical_index"] for row in visible}
        != set(PHYSICAL_GPU_INDICES)
    ):
        raise MatchedRunnerV2Error("GPU47 admission device rows differ")
    physical_by_index = {row["physical_index"]: row for row in physical}
    identity_fields = ("pci_bus_id", "rocm_unique_id", "hip_uuid_hex")
    if (
        len({row["pci_bus_id"] for row in physical}) != 8
        or len({row["rocm_unique_id"] for row in physical}) != 8
        or len({row["hip_uuid_hex"] for row in physical}) != 8
        or any(
            any(
                row[field] != physical_by_index[row["physical_index"]][field]
                for field in identity_fields
            )
            for row in [*all8, *visible]
        )
    ):
        raise MatchedRunnerV2Error("GPU47 PCI/unique-ID/UUID join differs")
    baseline = value.get("external_workload_baseline")
    if (
        not isinstance(baseline, Mapping)
        or set(baseline)
        != {
            "controller_process_count",
            "inference_process_count",
            "gpu_process_count",
            "all8_gpu_indices_observed",
            "all8_baseline_observed",
            "snapshot_sha256",
        }
        or baseline.get("controller_process_count") != 0
        or baseline.get("inference_process_count") != 0
        or baseline.get("gpu_process_count") != 0
        or baseline.get("all8_gpu_indices_observed") != list(range(8))
        or baseline.get("all8_baseline_observed") is not True
        or v1.SHA256_RE.fullmatch(str(baseline.get("snapshot_sha256"))) is None
    ):
        raise MatchedRunnerV2Error("GPU47 external-workload baseline differs")
    return value


def replay_task_authority_artifacts(
    output_root: Path,
    output_root_fd: int,
    task_result: Mapping[str, Any],
    verified_result: Mapping[str, Any] | None = None,
    publication_authority: Mapping[str, Any] | None = None,
    publication_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    references = task_result.get("authority_artifacts")
    arm = task_result.get("arm")
    expected_roles = {
        "model_capture",
        "model_pre_use",
        "consumption_input",
        "model_post_use",
        "eval_consumption_chain",
    }
    if arm == "full644":
        expected_roles |= {
            "adapter_capture",
            "adapter_pre_use",
            "adapter_post_use",
            "adapter_final",
        }
    elif arm != "base":
        raise MatchedRunnerV2Error("task artifact arm differs")
    if not isinstance(references, Mapping) or set(references) != expected_roles:
        raise MatchedRunnerV2Error("task artifact role closure differs")
    values: dict[str, dict[str, Any]] = {}
    replay_rows: list[dict[str, str]] = []
    for role in sorted(expected_roles):
        reference = references[role]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"basename", "sha256"}
            or not isinstance(reference["basename"], str)
            or reference["basename"] in {"", ".", ".."}
            or "/" in reference["basename"]
            or v1.SHA256_RE.fullmatch(reference.get("sha256", "")) is None
        ):
            raise MatchedRunnerV2Error("task artifact reference differs")
        path = _fd_child_path(output_root_fd, reference["basename"])
        value, observed_sha = _load_canonical_0400_json(path)
        if observed_sha != reference["sha256"]:
            raise MatchedRunnerV2Error("task artifact SHA replay differs")
        values[role] = value
        replay_rows.append(
            {
                "role": role,
                "basename": reference["basename"],
                "sha256": observed_sha,
            }
        )
    chain = validate_eval_consumption_chain(values["eval_consumption_chain"])
    digest_fields = {
        "model_capture": "capture_digest",
        "model_pre_use": "use_digest",
        "consumption_input": "consumption_input_digest",
        "model_post_use": "use_digest",
    }
    if arm == "full644":
        digest_fields.update(
            {
                "adapter_capture": "capture_digest",
                "adapter_pre_use": "use_digest",
                "adapter_post_use": "use_digest",
                "adapter_final": "adapter_final_digest",
            }
        )
    for role, field in digest_fields.items():
        _validate_embedded_digest(values[role], field, label=role)
    try:
        validated_consumption = model_authority.validate_consumption_input(
            values["consumption_input"]
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise MatchedRunnerV2Error(str(error)) from error
    if validated_consumption != values["consumption_input"]:
        raise MatchedRunnerV2Error("consumption input replay differs")
    if (
        chain["task_id"] != task_result.get("task_id")
        or chain["consumption_digest"] != task_result.get("consumption_digest")
        or values["model_capture"].get("capture_digest")
        != chain["model_capture_digest"]
        or values["model_pre_use"].get("use_digest")
        != chain["model_pre_use_digest"]
        or values["model_post_use"].get("use_digest")
        != chain["model_post_use_digest"]
        or values["consumption_input"].get("consumption_input_digest")
        != chain["consumption_input_digest"]
        or values["consumption_input"].get("task_id") != chain["task_id"]
        or values["consumption_input"].get("model", {}).get("capture_digest")
        != chain["model_capture_digest"]
        or values["consumption_input"].get("model", {}).get("pre_use_digest")
        != chain["model_pre_use_digest"]
    ):
        raise MatchedRunnerV2Error("task post-use chain replay differs")
    if arm == "full644" and (
        values["adapter_capture"].get("capture_digest")
        != chain["adapter_capture_digest"]
        or values["adapter_pre_use"].get("use_digest")
        != chain["adapter_pre_use_digest"]
        or values["adapter_post_use"].get("use_digest")
        != chain["adapter_post_use_digest"]
        or values["adapter_final"].get("adapter_final_digest")
        != chain["adapter_final_digest"]
        or values["consumption_input"].get("adapter", {}).get("capture_digest")
        != chain["adapter_capture_digest"]
        or values["consumption_input"].get("adapter", {}).get("pre_use_digest")
        != chain["adapter_pre_use_digest"]
    ):
        raise MatchedRunnerV2Error("task adapter post-use chain replay differs")
    if arm == "base" and values["consumption_input"].get("adapter") is not None:
        raise MatchedRunnerV2Error("base consumption unexpectedly binds adapter")

    result = dict(task_result)
    task_result_digest = _validate_embedded_digest(
        result, "task_result_digest", label="task result"
    )
    index = result.get("task_index")
    task_id = result.get("task_id")
    if (
        type(index) is not int
        or index not in range(16)
        or task_id != TASK_IDS[index]
        or result.get("schema_version") != TASK_SCHEMA
        or result.get("return_code") != 0
        or result.get("attempt_count") != 1
        or result.get("retry_allowed") is not False
        or result.get("model_capture_digest") != chain["model_capture_digest"]
        or result.get("adapter_capture_digest")
        != chain["adapter_capture_digest"]
        or result.get("consumption_input_digest")
        != chain["consumption_input_digest"]
        or result.get("consumption_digest") != chain["consumption_digest"]
        or result.get("native_receipt_digest")
        != chain["native_inference_receipt_digest"]
        or result.get("native_receipt_file_sha256")
        != chain["native_receipt_file_sha256"]
        or result.get("native_output_sha256") != chain["native_output_sha256"]
        or type(result.get("environment_digest")) is not str
        or v1.SHA256_RE.fullmatch(result["environment_digest"]) is None
    ):
        raise MatchedRunnerV2Error("task result/chain cross-link differs")
    runner_task_path = _fd_child_path(
        output_root_fd,
        f".matched-v2-{index:02d}-{task_id}-runner-task.json",
    )
    runner_task, runner_task_sha = _load_canonical_0400_json(runner_task_path)
    if runner_task != result:
        raise MatchedRunnerV2Error("persisted runner-task result differs")

    receipt_path = Path(result.get("receipt_path", ""))
    output_path = Path(result.get("output_path", ""))
    if receipt_path.parent != output_root or output_path.parent != output_root:
        raise MatchedRunnerV2Error("task native publication root differs")
    publication_task = {
        "task_id": task_id,
        "output": {
            "video_path": str(output_path),
            "receipt_path": str(receipt_path),
        },
    }
    if publication_authority is None:
        raise MatchedRunnerV2Error("retained task publication authority is absent")
    if publication_handoff is None:
        raise MatchedRunnerV2Error("retained task publication handoff is absent")
    handoff_payload = read_sealed_publication_handoff(
        publication_handoff, publication_task
    )
    try:
        retained_publication = v2.validate_retained_publication_authority(
            publication_authority, publication_task
        )
    except v2.MatchedEvalV2Error as error:
        raise MatchedRunnerV2Error(str(error)) from error
    receipt_raw = _pread_all(
        retained_publication["receipt_fd"],
        retained_publication["receipt_size"],
    )
    try:
        native_receipt = json.loads(receipt_raw.decode("utf-8", "strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MatchedRunnerV2Error("retained native receipt is not JSON") from error
    if (
        not isinstance(native_receipt, dict)
        or receipt_raw != canonical_json_bytes(native_receipt) + b"\n"
    ):
        raise MatchedRunnerV2Error("retained native receipt is not canonical JSON")
    _validate_embedded_digest(
        native_receipt, "receipt_digest", label="native receipt"
    )
    receipt_file_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt_identity = retained_publication["receipt_identity"]
    output_sha = retained_publication["output_sha256"]
    output_size = retained_publication["output_size"]
    output_identity = retained_publication["output_identity"]
    if receipt_file_sha != retained_publication["receipt_sha256"]:
        raise MatchedRunnerV2Error("retained receipt bytes differ")
    if (
        result.get("publication_handoff_authority_digest")
        != publication_handoff.get("authority_digest")
        or result.get("publication_handoff_payload_digest")
        != handoff_payload.get("payload_digest")
        or handoff_payload.get("receipt_digest")
        != native_receipt.get("receipt_digest")
        or any(
            handoff_payload[field] != retained_publication[target]
            for field, target in (
                ("output_path", "output_path"),
                ("output_identity", "output_identity"),
                ("output_sha256", "output_sha256"),
                ("output_size", "output_size"),
                ("receipt_path", "receipt_path"),
                ("receipt_identity", "receipt_identity"),
                ("receipt_sha256", "receipt_sha256"),
                ("receipt_size", "receipt_size"),
            )
        )
    ):
        raise MatchedRunnerV2Error("retained child publication handoff differs")
    if (
        native_receipt.get("receipt_digest")
        != chain["native_inference_receipt_digest"]
        or receipt_file_sha != chain["native_receipt_file_sha256"]
        or output_sha != chain["native_output_sha256"]
        or result.get("native_output_size") != output_size
        or result.get("native_receipt_identity") != receipt_identity
        or result.get("native_output_identity") != output_identity
        or native_receipt.get("output", {}).get("path") != str(output_path)
        or native_receipt.get("output", {}).get("sha256") != output_sha
        or native_receipt.get("output", {}).get("size") != output_size
        or native_receipt.get("model_consumption", {}).get(
            "consumption_input_digest"
        )
        != chain["consumption_input_digest"]
        or native_receipt.get("model_consumption", {}).get(
            "model_capture_digest"
        )
        != chain["model_capture_digest"]
        or native_receipt.get("model_consumption", {}).get(
            "adapter_capture_digest"
        )
        != chain["adapter_capture_digest"]
    ):
        raise MatchedRunnerV2Error("current native publication/chain differs")
    if verified_result is not None:
        expected_verified = {
            "task_id": task_id,
            "arm": arm,
            "receipt_path": str(receipt_path),
            "receipt_file_sha256": receipt_file_sha,
            "receipt_digest": native_receipt["receipt_digest"],
            "output_path": str(output_path),
            "output_sha256": output_sha,
            "output_size": output_size,
        }
        if (
            not isinstance(verified_result, Mapping)
            or any(verified_result.get(key) != value for key, value in expected_verified.items())
        ):
            raise MatchedRunnerV2Error("v2 report/native publication cross-link differs")
    return {
        "task_id": task_result["task_id"],
        "artifact_count": len(replay_rows),
        "artifact_rows_digest": object_sha256(replay_rows),
        "consumption_digest": chain["consumption_digest"],
        "task_result_digest": task_result_digest,
        "runner_task_file_sha256": runner_task_sha,
        "native_receipt_file_sha256": receipt_file_sha,
        "native_receipt_mode": stat.S_IMODE(receipt_identity["mode"]),
        "native_receipt_nlink": receipt_identity["nlink"],
        "native_output_sha256": output_sha,
        "publication_authority_digest": retained_publication[
            "authority_digest"
        ],
        "publication_handoff_authority_digest": publication_handoff[
            "authority_digest"
        ],
        "publication_handoff_payload_digest": handoff_payload[
            "payload_digest"
        ],
        "retained_receipt_and_output_fds_replayed": True,
        "v2_verified_result_cross_linked": verified_result is not None,
        "all_post_use_artifacts_replayed": True,
    }


def verify_case00_canary_pair(
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    execution: "RunnerExecution",
) -> dict[str, Any]:
    """Verify exactly case00 without creating a formal full16 report."""

    if (
        tuple(task.get("task_id") for task in tasks) != CANARY_TASK_IDS
        or execution.output_root_fd is None
        or execution.output_root_identity is None
        or set(execution.publication_authorities) != set(CANARY_TASK_IDS)
        or plan.get("production_ready") is not True
    ):
        raise MatchedRunnerV2Error("case00 canary task closure differs")
    if v2.validate_terminal_checkpoint_manifest(
        plan["checkpoint_manifest"]["path"],
        plan["checkpoint_manifest"]["sha256"],
    ) != plan["checkpoint_manifest"]:
        raise MatchedRunnerV2Error("terminal checkpoint changed before canary verify")
    verified_with_receipts = [
        v2.verify_arm(
            task,
            plan["producer"],
            publication_root=execution.output_root,
            publication_root_fd=execution.output_root_fd,
            ffprobe_authority=execution.ffprobe_authority,
            publication_authority=execution.publication_authorities[
                task["task_id"]
            ],
        )
        for task in tasks
    ]
    base, adapted = verified_with_receipts
    if base.get("arm") != "base" or adapted.get("arm") != "full644":
        raise MatchedRunnerV2Error("case00 canary arm order differs")
    if not v1._same_exact_json_value(
        base["receipt"]["model_consumption"]["model_capture_digest"],
        adapted["receipt"]["model_consumption"]["model_capture_digest"],
    ):
        raise MatchedRunnerV2Error("case00 canary model capture differs")
    for key in ("input", "preprocessing", "prompt_contract", "sampling"):
        if not v1._same_exact_json_value(
            base["receipt"].get(key), adapted["receipt"].get(key)
        ):
            raise MatchedRunnerV2Error(f"case00 canary pair differs on {key}")
    for key in (
        "method_source_revision",
        "method_source_archive_sha256",
        "bernini_commit",
        "infer_lora_source_sha256",
        "veomni_commit",
        "bernini_inference_files",
        "checkpoint_tree_sha256",
        "runtime_versions",
    ):
        if not v1._same_exact_json_value(
            base["receipt"].get(key), adapted["receipt"].get(key)
        ):
            raise MatchedRunnerV2Error(f"case00 canary runtime differs on {key}")
    verified: list[dict[str, Any]] = []
    for row in verified_with_receipts:
        clean = dict(row)
        clean.pop("receipt")
        verified.append(clean)
    report: dict[str, Any] = {
        "schema_version": (
            "full644-exploratory-matched-gpu47-case00-canary-report-v1"
        ),
        "status": "CANARY_COMPLETE_AWAITING_VISUAL_REVIEW",
        "campaign_mode": CASE00_CANARY_CAMPAIGN,
        "full_source_authorized_plan_schema_version": plan["schema_version"],
        "full_source_authorized_plan_digest": plan["plan_digest"],
        "selected_task_ids": list(CANARY_TASK_IDS),
        "unselected_task_ids": list(TASK_IDS[2:]),
        "unselected_task_count": 14,
        "unselected_tasks_executed": False,
        "unselected_outputs_published": False,
        "pair_count": 1,
        "verified_task_count": 2,
        "formal_full16_report": False,
        "html_generated": False,
        "manual_visual_review_required_before_full16": True,
        "retained_publication_root_fd_replayed": True,
        "retained_ffprobe_executable_fd_replayed": True,
        "retained_publication_leaf_fds_replayed": True,
        "results": verified,
        "claim_limits": dict(v1.CLAIM_LIMITS),
    }
    report["report_digest"] = object_sha256(report)
    return report


def verify_unselected_canary_artifacts_absent(
    plan: Mapping[str, Any], execution: "RunnerExecution"
) -> list[str]:
    """Prove every public and internal leaf for tasks 02..15 is absent."""

    if (
        execution.output_root_fd is None
        or tuple(task["task_id"] for task in plan["tasks"]) != TASK_IDS
    ):
        raise MatchedRunnerV2Error("canary unselected-task closure differs")
    checked: list[str] = []
    for index, task in enumerate(plan["tasks"][2:], start=2):
        basenames = [
            Path(task["output"]["video_path"]).name,
            Path(task["output"]["receipt_path"]).name,
        ]
        prefix = f".matched-v2-{index:02d}-{task['task_id']}"
        suffixes = list(_BASE_ARTIFACT_SUFFIXES) + [
            ".log",
            "-runner-task.json",
        ]
        if task["arm"] == "full644":
            suffixes.extend(_ADAPTER_ARTIFACT_SUFFIXES)
        basenames.extend(prefix + suffix for suffix in suffixes)
        for basename in basenames:
            try:
                os.stat(
                    basename,
                    dir_fd=execution.output_root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                checked.append(basename)
            else:
                raise MatchedRunnerV2Error(
                    "unselected canary task artifact was published"
                )
    return checked


def _complete_execution(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    execution: RunnerExecution,
    final_parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entry_authority = validate_captured_runner_entry(
        args.entry_authority, args=args
    )
    task_results, model_final = execution.run()
    selected_task_ids = tuple(task["task_id"] for task in tasks)
    selected_task_count = len(selected_task_ids)
    is_canary = args.campaign_mode == CASE00_CANARY_CAMPAIGN
    if (
        (is_canary and selected_task_ids != CANARY_TASK_IDS)
        or (not is_canary and selected_task_ids != TASK_IDS)
    ):
        raise MatchedRunnerV2Error("campaign completion task closure differs")
    if execution.output_root_fd is None or execution.output_root_identity is None:
        raise MatchedRunnerV2Error("retained output root disappeared")
    _validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    _validate_embedded_digest(model_final, "model_final_digest", label="model final")
    if (
        model_final.get("task_count") != selected_task_count
        or model_final.get("model_capture_digest")
        != task_results[0]["model_capture_digest"]
        or model_final.get("task_consumption_digests")
        != [row["consumption_digest"] for row in task_results]
    ):
        raise MatchedRunnerV2Error("model final/task consumption closure differs")
    if any(
        row.get("ffmpeg_exec_authority_digest")
        != execution.ffmpeg_exec_authority_digest
        for row in task_results
    ):
        raise MatchedRunnerV2Error("task ffmpeg execution authority differs")
    unselected_absence_rows = (
        verify_unselected_canary_artifacts_absent(plan, execution)
        if is_canary
        else []
    )
    report = (
        verify_case00_canary_pair(plan, tasks, execution)
        if is_canary
        else v2.verify_results(
            plan,
            publication_root_fd=execution.output_root_fd,
            ffprobe_authority=execution.ffprobe_authority,
            publication_authorities=execution.publication_authorities,
        )
    )
    _validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    if is_canary and verify_unselected_canary_artifacts_absent(
        plan, execution
    ) != unselected_absence_rows:
        raise MatchedRunnerV2Error("canary unselected absence replay differs")
    verified_rows = report.get("results")
    if (
        not isinstance(verified_rows, list)
        or len(verified_rows) != selected_task_count
        or {row.get("task_id") for row in verified_rows}
        != set(selected_task_ids)
        or report.get("retained_publication_root_fd_replayed") is not True
        or report.get("retained_ffprobe_executable_fd_replayed") is not True
        or report.get("retained_publication_leaf_fds_replayed") is not True
    ):
        raise MatchedRunnerV2Error("v2 verified-result task closure differs")
    verified_by_task = {row["task_id"]: row for row in verified_rows}
    artifact_replays = [
        replay_task_authority_artifacts(
            execution.output_root,
            execution.output_root_fd,
            row,
            verified_by_task[row["task_id"]],
            execution.publication_authorities[row["task_id"]],
            execution.publication_handoffs[row["task_id"]],
        )
        for row in task_results
    ]
    _validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    report_parent = final_parents["output_report"]
    attestation_parent = final_parents["runner_attestation"]
    _validate_final_parent(report_parent)
    _validate_final_parent(attestation_parent)
    report_path = report_parent["path"]
    _, report_sha = _write_json_at(
        report_parent["parent_fd"],
        report_path.name,
        report,
        mode=0o444,
    )
    _validate_final_parent(report_parent)
    attestation: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": (
            "CANARY_COMPLETE_AWAITING_VISUAL_REVIEW"
            if is_canary
            else "COMPLETE"
        ),
        "campaign_mode": args.campaign_mode,
        "formal_full16_report": not is_canary,
        "manual_visual_review_required_before_full16": is_canary,
        "plan": {
            "path": str(Path(args.plan).resolve(strict=True)),
            "sha256": args.plan_sha256,
            "plan_digest": plan["plan_digest"],
        },
        "physical_bindings": dict(bindings),
        "gpu_visibility_contract": dict(
            bindings["allocation"]["gpu_visibility_contract"]
        ),
        "gpu_admission": {
            "path": str(Path(args.gpu_admission_receipt).resolve(strict=True)),
            "sha256": args.gpu_admission_receipt_sha256,
            "receipt_digest": bindings["gpu_admission"]["receipt_digest"],
            "observed_at_unix_ns": bindings["gpu_admission"][
                "observed_at_unix_ns"
            ],
            "slurm_step_id": bindings["gpu_admission"]["slurm_step_id"],
            "visible_runtime_devices": bindings["gpu_admission"][
                "visible_runtime_devices"
            ],
            "pci_bus_unique_id_hip_uuid_join_replayed": True,
            "external_workload_baseline_replayed": True,
        },
        "captured_runner_entry": {
            "authority_digest": entry_authority["authority_digest"],
            "release_digest": entry_authority["release_digest"],
            "bootstrap_sha256": entry_authority["bootstrap_sha256"],
            "captured_source_entry": True,
            "held_through_attestation_publication": True,
        },
        "retained_publication_root": {
            "path": str(execution.output_root),
            "fd": execution.output_root_fd,
            "immutable_identity": execution.output_root_identity,
            "held_through_attestation_publication": True,
        },
        "retained_ffprobe_executable": {
            "authority_digest": execution.ffprobe_authority[
                "authority_digest"
            ],
            "fd": execution.ffprobe_authority["fd"],
            "source_path": execution.ffprobe_authority["source_path"],
            "sha256": execution.ffprobe_authority["sha256"],
            "held_through_result_verification": True,
        },
        "retained_task_publications": {
            task_id: {
                "authority_digest": row["authority_digest"],
                "receipt_fd": row["receipt_fd"],
                "output_fd": row["output_fd"],
                "held_through_result_verification": True,
            }
            for task_id, row in sorted(
                execution.publication_authorities.items()
            )
        },
        "retained_child_publication_handoffs": {
            task_id: {
                "authority_digest": row["authority_digest"],
                "fd": row["fd"],
                "payload_digest": read_sealed_publication_handoff(
                    row,
                    next(
                        task for task in tasks if task["task_id"] == task_id
                    ),
                )["payload_digest"],
                "held_sealed_through_attestation": True,
            }
            for task_id, row in sorted(execution.publication_handoffs.items())
        },
        "retained_final_parents": {
            label: {
                "path": str(row["path"].parent),
                "fd": row["parent_fd"],
                "immutable_identity": row["parent_identity"],
            }
            for label, row in final_parents.items()
        },
        "task_count": selected_task_count,
        "task_ids": list(selected_task_ids),
        "unselected_task_ids": list(TASK_IDS[2:]) if is_canary else [],
        "unselected_task_count": 14 if is_canary else 0,
        "unselected_tasks_executed": False,
        "unselected_outputs_and_internal_artifacts_absent": is_canary,
        "unselected_absence_row_count": len(unselected_absence_rows),
        "unselected_absence_rows_digest": object_sha256(
            unselected_absence_rows
        ),
        "task_result_digests": [row["task_result_digest"] for row in task_results],
        "task_environment_digests": [row["environment_digest"] for row in task_results],
        "ffmpeg_exec_authority_digest": (
            execution.ffmpeg_exec_authority_digest
        ),
        "all_rank0_encoders_used_retained_ffmpeg_executable": True,
        "task_results": task_results,
        "task_artifact_replays": artifact_replays,
        "campaign_report_exactly_cross_linked_to_all_task_chains": True,
        "runner_task_json_replayed_for_all_tasks": True,
        "all_selected_tasks_attempted_exactly_once": True,
        "all_selected_tasks_succeeded": True,
        "retry_count": 0,
        "native_publication_before_parent_post_use_replay": True,
        "no_false_post_use_before_publication_claim": True,
        "native_receipts_replayed_0400_single_link": all(
            row["native_receipt_mode"] == 0o400
            and row["native_receipt_nlink"] == 1
            for row in artifact_replays
        ),
        "model_capture_digest": task_results[0]["model_capture_digest"],
        "same_model_capture_all_selected_tasks": len(
            {row["model_capture_digest"] for row in task_results}
        )
        == 1,
        "model_final": model_final,
        "verified_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "report_digest": report["report_digest"],
            "verified_task_count": report["verified_task_count"],
        },
        "external_runner_attestation_present": True,
        "receipt_contract_alone_did_not_prove_execution": True,
        "exploratory_only": True,
        "scientific_claim_authorized": False,
        "formal_claim_authorized": False,
    }
    if (
        attestation["same_model_capture_all_selected_tasks"] is not True
        or attestation["native_receipts_replayed_0400_single_link"] is not True
    ):
        raise MatchedRunnerV2Error("final matched attestation closure differs")
    attestation["attestation_digest"] = object_sha256(attestation)
    _validate_held_directory(
        execution.output_root_fd,
        execution.output_root,
        execution.output_root_identity,
    )
    _validate_final_parent(report_parent)
    _validate_final_parent(attestation_parent)
    if is_canary and verify_unselected_canary_artifacts_absent(
        plan, execution
    ) != unselected_absence_rows:
        raise MatchedRunnerV2Error("canary unselected absence final replay differs")
    validate_captured_runner_entry(entry_authority, args=args)
    attestation_path = attestation_parent["path"]
    _write_json_at(
        attestation_parent["parent_fd"],
        attestation_path.name,
        attestation,
        mode=0o444,
    )
    return attestation


def execute(args: argparse.Namespace) -> dict[str, Any]:
    entry_authority = validate_captured_runner_entry(
        args.entry_authority, args=args
    )
    allocation = _allocation_authority(
        args.holder_job_id,
        args.expected_node,
        args.expected_allocation_gpu_count,
    )
    plan = v2.load_plan(args.plan, args.plan_sha256)
    tasks = select_campaign_tasks(plan, args.campaign_mode)
    selected_task_ids = tuple(task["task_id"] for task in tasks)
    expected_task_ids = (
        TASK_IDS
        if args.campaign_mode == FULL16_CAMPAIGN
        else CANARY_TASK_IDS
    )
    if selected_task_ids != expected_task_ids:
        raise MatchedRunnerV2Error("campaign task selection differs")
    output_root = Path(tasks[0]["output"]["video_path"]).parent
    gpu_admission = validate_gpu47_admission_receipt(
        args.gpu_admission_receipt,
        args.gpu_admission_receipt_sha256,
        holder_job_id=args.holder_job_id,
        expected_node=args.expected_node,
        expected_slurm_step_id=allocation["slurm_step_id"],
        campaign_mode=args.campaign_mode,
        plan_sha256=args.plan_sha256,
        output_root=output_root,
    )
    # A canary must start with all sixteen public and internal task names fresh,
    # even though only the exact case00 pair is permitted to execute.
    final_artifacts = _preflight_final_artifacts(
        args, validate_task_order(plan)
    )
    identities = {
        "runner": _identity(__file__, args.runner_sha256),
        "bridge": _identity(args.bridge_script, args.bridge_script_sha256),
        "adapter": _identity(args.adapter_script, args.adapter_script_sha256),
        "eval_v1": _identity(args.eval_v1_source, args.eval_v1_source_sha256),
        "eval_v2": _identity(args.eval_v2_source, args.eval_v2_source_sha256),
        "model_authority": _identity(
            args.model_authority_source,
            args.model_authority_source_sha256,
        ),
        "python": _identity(args.python, args.python_sha256),
        "torchrun_source": _identity(
            args.torchrun_source, args.torchrun_source_sha256
        ),
        "torchrun_handler_source": _identity(
            args.torchrun_handler_source,
            args.torchrun_handler_source_sha256,
        ),
        "model_manifest": _identity(
            args.model_manifest, args.model_manifest_sha256
        ),
        "ffmpeg": _identity(
            args.ffmpeg_executable,
            args.ffmpeg_executable_sha256,
        ),
        "ffprobe": _identity(
            plan["producer"]["ffprobe_path"],
            plan["producer"]["ffprobe_sha256"],
        ),
    }
    if (
        args.model_manifest_sha256 != EXPECTED_MODEL_MANIFEST_SHA256
        or args.eval_v1_source_sha256 != EXPECTED_EVAL_V1_SHA256
        or args.eval_v2_source_sha256 != EXPECTED_EVAL_V2_SHA256
        or args.model_authority_source_sha256
        != EXPECTED_MODEL_AUTHORITY_SHA256
        or args.torchrun_source_sha256 != TORCHRUN_SOURCE_SHA256
        or args.torchrun_handler_source_sha256
        != TORCHRUN_HANDLER_SHA256
    ):
        raise MatchedRunnerV2Error("model/Torch Elastic exact source pin differs")
    infer_identity = _identity(
        plan["producer"]["infer_lora_path"],
        plan["producer"]["infer_lora_sha256"],
    )
    if (
        infer_identity["sha256"]
        != "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"
        or Path(args.adapter_script).resolve(strict=True).parent
        != Path(plan["producer"]["infer_lora_path"]).resolve(strict=True).parent
        or Path(args.bridge_script).resolve(strict=True).parent
        != Path(args.adapter_script).resolve(strict=True).parent
        or Path(__file__).resolve(strict=True).parent
        != Path(args.adapter_script).resolve(strict=True).parent
        or Path(v1.__file__).resolve(strict=True)
        != Path(args.eval_v1_source).resolve(strict=True)
        or Path(v2.__file__).resolve(strict=True)
        != Path(args.eval_v2_source).resolve(strict=True)
        or Path(model_authority.__file__).resolve(strict=True)
        != Path(args.model_authority_source).resolve(strict=True)
        or getattr(model_authority, "__cached__", None) is not None
        or getattr(v1, "__cached__", None) is not None
        or getattr(v2, "__cached__", None) is not None
    ):
        raise MatchedRunnerV2Error("runner/bridge/adapter/frozen infer release differs")
    ffprobe_authority = capture_ffprobe_authority(
        identities["ffprobe"],
        plan["producer"],
    )
    try:
        exec_authority = capture_exec_authority(identities)
    except BaseException:
        close_ffprobe_authority(ffprobe_authority)
        raise
    try:
        args.exec_authority = exec_authority
        args.ffprobe_authority = ffprobe_authority
        bindings: dict[str, Any] = {
            "schema_version": "full644-exploratory-matched-physical-bindings-v2",
            "plan_path": str(Path(args.plan).resolve(strict=True)),
            "plan_sha256": args.plan_sha256,
            "plan_digest": plan["plan_digest"],
            "allocation": allocation,
            "gpu_admission": gpu_admission,
            "identities": {**identities, "infer_lora": infer_identity},
            "captured_runner_entry": entry_authority,
            "captured_runner_entry_required": True,
            "exec_authority": exec_authority,
            "exec_authority_retained_source_and_python_fds": True,
            "ffprobe_authority": ffprobe_authority,
            "ffprobe_retained_executable_fd": True,
            "isolated_child_interpreters": "-I -S -B",
            "child_environment_exact_allowlist": True,
            "model_root": str(Path(args.model_root).resolve(strict=True)),
            "bernini_root": str(Path(args.bernini_root).resolve(strict=True)),
            "veomni_root": str(Path(args.veomni_root).resolve(strict=True)),
            "campaign_mode": args.campaign_mode,
            "formal_full16_report": args.campaign_mode == FULL16_CAMPAIGN,
            "task_count": len(tasks),
            "task_ids": list(selected_task_ids),
            "retry_allowed": False,
            "final_artifacts": final_artifacts,
        }
        bindings["physical_bindings_digest"] = object_sha256(bindings)
        args.physical_bindings_digest = bindings["physical_bindings_digest"]
    except BaseException:
        close_exec_authority(exec_authority)
        close_ffprobe_authority(ffprobe_authority)
        raise
    final_parents: dict[str, dict[str, Any]] = {}
    execution: RunnerExecution | None = None
    try:
        final_parents = _hold_final_artifact_parents(final_artifacts)
        execution = RunnerExecution(args, plan, tasks)
        return _complete_execution(
            args,
            plan,
            tasks,
            bindings,
            execution,
            final_parents,
        )
    finally:
        if execution is not None:
            execution.close_descriptors()
        else:
            close_exec_authority(exec_authority)
            close_ffprobe_authority(ffprobe_authority)
        _close_final_parents(final_parents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-mode",
        required=True,
        choices=(FULL16_CAMPAIGN, CASE00_CANARY_CAMPAIGN),
    )
    parser.add_argument("--gpu-admission-receipt", required=True)
    parser.add_argument("--gpu-admission-receipt-sha256", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--runner-attestation", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--bridge-script", required=True)
    parser.add_argument("--bridge-script-sha256", required=True)
    parser.add_argument("--adapter-script", required=True)
    parser.add_argument("--adapter-script-sha256", required=True)
    parser.add_argument("--eval-v1-source", required=True)
    parser.add_argument(
        "--eval-v1-source-sha256", default=EXPECTED_EVAL_V1_SHA256
    )
    parser.add_argument("--eval-v2-source", required=True)
    parser.add_argument("--eval-v2-source-sha256", required=True)
    parser.add_argument("--model-authority-source", required=True)
    parser.add_argument(
        "--model-authority-source-sha256",
        default=EXPECTED_MODEL_AUTHORITY_SHA256,
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--ffmpeg-executable", required=True)
    parser.add_argument("--ffmpeg-executable-sha256", required=True)
    parser.add_argument("--torchrun-source", required=True)
    parser.add_argument(
        "--torchrun-source-sha256", default=TORCHRUN_SOURCE_SHA256
    )
    parser.add_argument("--torchrun-handler-source", required=True)
    parser.add_argument(
        "--torchrun-handler-source-sha256",
        default=TORCHRUN_HANDLER_SHA256,
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument(
        "--model-manifest-sha256", default=EXPECTED_MODEL_MANIFEST_SHA256
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--rank-cache-root", required=True)
    parser.add_argument("--holder-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-allocation-gpu-count", type=int, default=8)
    return parser


def _require_isolated_runner_startup() -> None:
    if (
        sys.platform != "linux"
        or not Path("/proc/self/fd").is_dir()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise MatchedRunnerV2Error(
            "runner requires captured-source -I -S -B bootstrap"
        )


def main(argv: Sequence[str] | None = None) -> int:
    _require_isolated_runner_startup()
    entry_authority = validate_captured_runner_entry()
    args: argparse.Namespace | None = None
    try:
        try:
            args = build_parser().parse_args(argv)
            args.entry_authority = validate_captured_runner_entry(
                entry_authority, args=args
            )
            result = execute(args)
        except Exception as error:
            if args is None:
                raise
            failure = {
                "schema_version": FAILURE_SCHEMA,
                "status": "FAILED_NO_RETRY",
                "error_type": type(error).__name__,
                "error": str(error),
                "plan_path": str(Path(args.plan)),
                "plan_sha256": args.plan_sha256,
                "runner_path": str(Path(__file__).resolve(strict=True)),
                "retry_allowed": False,
                "partial_outputs_are_not_results": True,
                "scientific_claim_authorized": False,
            }
            failure["failure_digest"] = object_sha256(failure)
            try:
                v1.write_create_only(args.runner_attestation, failure)
            except Exception:
                pass
            raise
        try:
            print(canonical_json_bytes(result).decode("utf-8"))
        except (BrokenPipeError, OSError):
            pass
        return 0
    finally:
        close_captured_runner_entry(entry_authority)


if __name__ == "__main__":
    raise SystemExit(main())
