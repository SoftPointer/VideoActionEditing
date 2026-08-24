#!/usr/bin/env python3
"""GPU4-7 retained-FD relay from one torchrun coordinator to four ranks.

The outer matched runner grants its authority descriptors to this process with
``pass_fds``.  This bridge validates every recorded descriptor identity while
the descriptors are necessarily inheritable at the exec boundary, restores
CLOEXEC, and then installs one task-local hook on the pinned Torch Elastic
``SubprocessHandler``.  The hook permits exactly four rank spawns, exact argv,
and the exact Torch-produced rank environment overlay.  Each spawn receives no
extra ``pass_fds`` beyond the authority binding.  The rank adapter performs the
same validate-then-seal transition before importing frozen inference code.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Iterator, Mapping, Sequence

SCHEMA = "full644-exploratory-matched-torchrun-fd-bridge-gpu47-v3"
GPU_VISIBILITY_SCHEMA = "full644-exploratory-matched-gpu47-visibility-v1"
ROCR_PHYSICAL_GPU_MASK = "4,5,6,7"
PHYSICAL_GPU_INDICES = (4, 5, 6, 7)
LOGICAL_GPU_INDICES = (0, 1, 2, 3)
_SECONDARY_GPU_MASKS = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
MODEL_AUTHORITY_SHA256 = (
    "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"
)
TORCHRUN_SOURCE_SHA256 = (
    "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c"
)
TORCHRUN_HANDLER_SHA256 = (
    "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87"
)
SITE_PACKAGES_ENV = "FULL644_MATCHED_SITE_PACKAGES_ROOT"
RANK_CACHE_ENV = "FULL644_MATCHED_RANK_CACHE_ROOT"
EXEC_AUTHORITY_ENV = "FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING"
FFMPEG_AUTHORITY_ENV = "FULL644_MATCHED_FFMPEG_EXEC_AUTHORITY"
PUBLICATION_HANDOFF_ENV = "FULL644_MATCHED_PUBLICATION_HANDOFF_AUTHORITY"
PUBLICATION_HANDOFF_AUTHORITY_SCHEMA = (
    "full644-exploratory-matched-publication-handoff-authority-v1"
)
EXEC_AUTHORITY_SCHEMA = "full644-exploratory-matched-exec-authority-v2"
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_WORKER_ENV_KEYS = {
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "OMP_NUM_THREADS",
}
_COORDINATOR_FIXED_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
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
}
_COORDINATOR_DYNAMIC_ENV_KEYS = {
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    RANK_CACHE_ENV,
    EXEC_AUTHORITY_ENV,
    PUBLICATION_HANDOFF_ENV,
    "APV2_EVAL_INHERITED_AUTHORITY_FDS",
}


class TorchrunFDBridgeV2Error(RuntimeError):
    """The coordinator/rank FD or process envelope differs."""


def gpu47_visibility_contract(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    if (
        environment.get("ROCR_VISIBLE_DEVICES") != ROCR_PHYSICAL_GPU_MASK
        or any(key in environment for key in _SECONDARY_GPU_MASKS)
    ):
        raise TorchrunFDBridgeV2Error("GPU47 ROCr-only visibility differs")
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


ISOLATED_RANK_BOOTSTRAP = r'''import fcntl,hashlib,json,os,stat,sys,types
ENV="FULL644_MATCHED_PYTHON_EXECUTABLE_BINDING"
FENV="FULL644_MATCHED_FFMPEG_EXEC_AUTHORITY"
HENV="FULL644_MATCHED_PUBLICATION_HANDOFF_AUTHORITY"
SCHEMA="full644-exploratory-matched-exec-authority-v2"
FSCHEMA="full644-exploratory-matched-ffmpeg-exec-authority-v1"
HSCHEMA="full644-exploratory-matched-publication-handoff-authority-v1"
ROLES=["python_executable","adapter_source","ffmpeg_executable"]
RF={"role","fd","source_path","sha256","identity"}
IF={"device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"}
MF={"schema_version","task_id","model_capture_digest","adapter_capture_digest","fd_count","fd_rows","fd_rows_digest","namespace_root_count","publication_root_count","exact_allowlist_only","proc_self_fd_consumption_required","cross_process_proc_fd_access_forbidden","ptrace_authorization_used","fd_binding_digest"}
MRF={"fd","scope","role","relative_path","source_path","identity"}
MSCHEMA="bernini-action-preservation-inherited-fd-binding-v3"
def pairs(items):
 out={}
 for key,value in items:
  if key in out: raise RuntimeError("duplicate rank exec-authority JSON key")
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
 if ident(before)!=row["identity"] or ident(after)!=row["identity"] or len(raw)!=before.st_size or hashlib.sha256(raw).hexdigest()!=row["sha256"]: raise RuntimeError("rank exec-authority FD replay differs")
 return raw
def seal_model(raw,code_fds):
 model=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
 if type(model) is not dict or set(model)!=MF or canonical(model)!=raw: raise RuntimeError("rank model authority binding differs")
 unsigned=dict(model); claimed=unsigned.pop("fd_binding_digest",None); rows=model.get("fd_rows")
 if model.get("schema_version")!=MSCHEMA or model.get("exact_allowlist_only") is not True or model.get("proc_self_fd_consumption_required") is not True or model.get("cross_process_proc_fd_access_forbidden") is not True or model.get("ptrace_authorization_used") is not False or claimed!=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest() or type(rows) is not list or type(model.get("fd_count")) is not int or model["fd_count"]!=len(rows) or model.get("fd_rows_digest")!=hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest(): raise RuntimeError("rank model authority digest differs")
 mfds=[]; roles=[]
 for item in rows:
  identity=item.get("identity") if type(item) is dict else None; fd=item.get("fd") if type(item) is dict else None; role=item.get("role") if type(item) is dict else None; scope=item.get("scope") if type(item) is dict else None
  if type(item) is not dict or set(item)!=MRF or type(fd) is not int or fd<3 or scope not in {"model","adapter","task"} or role not in {"file","namespace_root","publication_root"} or type(item.get("relative_path")) is not str or type(item.get("source_path")) is not str or not os.path.isabs(item["source_path"]) or os.path.normpath(item["source_path"])!=item["source_path"] or type(identity) is not dict or set(identity)!=IF or any(type(value) is not int for value in identity.values()) or (role=="file" and not stat.S_ISREG(identity["mode"])) or (role!="file" and not stat.S_ISDIR(identity["mode"])) or not os.get_inheritable(fd): raise RuntimeError("rank model authority row differs")
  observed=ident(os.fstat(fd)); mutable=scope=="task" and role=="publication_root"; immutable={key for key in ("device","inode","uid","gid","mode","rdev")}
  if (observed if not mutable else {key:observed[key] for key in immutable})!=(identity if not mutable else {key:identity[key] for key in immutable}): raise RuntimeError("rank model authority FD identity differs")
  mfds.append(fd); roles.append((scope,role,item["relative_path"]))
 if mfds!=sorted(mfds) or len(mfds)!=len(set(mfds)) or set(mfds)&set(code_fds) or len(mfds) not in {25,30} or roles.count(("task","publication_root","."))!=1 or sum(scope=="model" and role=="file" for scope,role,_ in roles)!=23 or sum(scope=="model" and role=="namespace_root" for scope,role,_ in roles)!=1 or sum(scope=="adapter" and role=="file" for scope,role,_ in roles)!=(0 if model.get("adapter_capture_digest") is None else 4) or sum(scope=="adapter" and role=="namespace_root" for scope,role,_ in roles)!=(0 if model.get("adapter_capture_digest") is None else 1): raise RuntimeError("rank model authority FD allowlist differs")
 for fd in mfds: os.set_inheritable(fd,False)
 if any(os.get_inheritable(fd) for fd in mfds): raise RuntimeError("rank model authority FDs remain inheritable")
 return mfds,model["task_id"]
raw=os.environ.get(ENV)
if raw is None: raise RuntimeError("rank exec-authority binding is absent")
binding=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if type(binding) is not dict or set(binding)!={"schema_version","rows","rows_digest","binding_digest"} or binding.get("schema_version")!=SCHEMA or canonical(binding)!=raw: raise RuntimeError("rank exec-authority binding differs")
unsigned=dict(binding); claimed=unsigned.pop("binding_digest",None); rows=binding.get("rows")
if type(rows) is not list or [row.get("role") if type(row) is dict else None for row in rows]!=ROLES or binding.get("rows_digest")!=hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest() or claimed!=hashlib.sha256(canonical(unsigned).encode("utf-8")).hexdigest(): raise RuntimeError("rank exec-authority digest differs")
fds=[]; payload={}
for row in rows:
 if type(row) is not dict or set(row)!=RF or type(row.get("fd")) is not int or row["fd"]<3 or type(row.get("source_path")) is not str or not os.path.isabs(row["source_path"]) or os.path.normpath(row["source_path"])!=row["source_path"] or type(row.get("sha256")) is not str or len(row["sha256"])!=64 or any(ch not in "0123456789abcdef" for ch in row["sha256"]) or type(row.get("identity")) is not dict or set(row["identity"])!=IF or any(type(value) is not int for value in row["identity"].values()) or not stat.S_ISREG(row["identity"]["mode"]) or row["identity"]["nlink"]!=1 or not os.get_inheritable(row["fd"]): raise RuntimeError("rank exec-authority row differs")
 fds.append(row["fd"]); payload[row["role"]]=read(row)
if len(fds)!=len(set(fds)) or fds!=sorted(fds): raise RuntimeError("rank exec-authority FD allowlist differs")
model_raw=os.environ.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
if model_raw is None: raise RuntimeError("rank model authority binding is absent")
model_fds,model_task_id=seal_model(model_raw,fds)
handoff_raw=os.environ.get(HENV)
if handoff_raw is None: raise RuntimeError("rank publication handoff authority is absent")
handoff=json.loads(handoff_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
if type(handoff) is not dict or set(handoff)!={"schema_version","task_id","fd","initial_identity","capacity","authority_digest"} or canonical(handoff)!=handoff_raw: raise RuntimeError("rank publication handoff authority differs")
handoff_unsigned=dict(handoff); handoff_claimed=handoff_unsigned.pop("authority_digest",None); handoff_fd=handoff.get("fd"); handoff_identity=handoff.get("initial_identity")
if handoff.get("schema_version")!=HSCHEMA or handoff.get("task_id")!=model_task_id or handoff.get("capacity")!=65536 or handoff_claimed!=hashlib.sha256(canonical(handoff_unsigned).encode("utf-8")).hexdigest() or type(handoff_fd) is not int or handoff_fd<3 or type(handoff_identity) is not dict or set(handoff_identity)!=IF or any(type(value) is not int for value in handoff_identity.values()) or handoff_fd in set(fds)|set(model_fds) or not os.get_inheritable(handoff_fd): raise RuntimeError("rank publication handoff digest differs")
handoff_observed=os.fstat(handoff_fd)
if ident(handoff_observed)!=handoff_identity or not stat.S_ISREG(handoff_observed.st_mode) or handoff_observed.st_nlink!=0 or stat.S_IMODE(handoff_observed.st_mode)!=0o600 or handoff_observed.st_size!=0 or fcntl.fcntl(handoff_fd,fcntl.F_GET_SEALS)!=0: raise RuntimeError("rank empty publication handoff replay differs")
python_row,adapter_row,ffmpeg_row=rows
if not python_row["identity"]["mode"]&0o111 or not ffmpeg_row["identity"]["mode"]&0o111: raise RuntimeError("retained rank executable mode differs")
try: adapter_source=payload["adapter_source"].decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("captured adapter source is not UTF-8") from error
for fd in [*fds,handoff_fd]: os.set_inheritable(fd,False)
if any(os.get_inheritable(fd) for fd in [*fds,handoff_fd]): raise RuntimeError("rank exec/handoff authority FDs remain inheritable")
os.close(python_row["fd"]); os.close(adapter_row["fd"])
ffmpeg_binding={"schema_version":FSCHEMA,"row":ffmpeg_row}
ffmpeg_binding["authority_digest"]=hashlib.sha256(canonical(ffmpeg_binding).encode("utf-8")).hexdigest()
os.environ[FENV]=canonical(ffmpeg_binding)
os.environ.pop(ENV,None)
sys.argv=[adapter_row["source_path"],*sys.argv[1:]]
module=types.ModuleType("__main__"); module.__file__=adapter_row["source_path"]; module.__package__=None; module.__loader__=None; module.__spec__=None; module.__cached__=None; module.__builtins__=__builtins__; sys.modules["__main__"]=module
exec(compile(adapter_source,adapter_row["source_path"],"exec",dont_inherit=True),module.__dict__)'''


def _read_exact_source(
    path_value: str | Path, expected_sha256: str, *, label: str
) -> tuple[Path, str]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise TorchrunFDBridgeV2Error(f"{label} source SHA differs")
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
        raise TorchrunFDBridgeV2Error(f"{label} source identity differs")
    try:
        source = b"".join(chunks).decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise TorchrunFDBridgeV2Error(f"{label} source is not UTF-8") from error
    return path, source


def _load_exact_source_module(
    name: str,
    path_value: str | Path,
    expected_sha256: str,
    *,
    require_absent: bool,
) -> Any:
    """Execute pinned UTF-8 source directly; never consult a ``.pyc``."""

    path, source = _read_exact_source(path_value, expected_sha256, label=name)
    existing = sys.modules.get(name)
    if existing is not None:
        if require_absent:
            raise TorchrunFDBridgeV2Error(f"{name} was imported before source capture")
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve(strict=True) != path:
            raise TorchrunFDBridgeV2Error(f"{name} existing origin differs")
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
        raise TorchrunFDBridgeV2Error(f"{name} executed-source origin differs")
    return module


_MODEL_AUTHORITY_PATH = (
    Path(__file__).resolve(strict=True).parent
    / "action_preservation_decoded_eval_model_authority_v2.py"
)
model_authority = _load_exact_source_module(
    "action_preservation_decoded_eval_model_authority_v2",
    _MODEL_AUTHORITY_PATH,
    MODEL_AUTHORITY_SHA256,
    require_absent=(__name__ == "__main__"),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
_RANK_EXEC_ROLES = (
    "python_executable",
    "adapter_source",
    "ffmpeg_executable",
)


def _exec_stat_identity(info: os.stat_result) -> dict[str, int]:
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


def _pread_exact(descriptor: int, size: int) -> bytes:
    if type(size) is not int or size <= 0 or not hasattr(os, "pread"):
        raise TorchrunFDBridgeV2Error("retained executable pread is unavailable")
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
        raise TorchrunFDBridgeV2Error("retained executable read is incomplete")
    return raw


def load_rank_exec_authority(
    *, expected_inheritable: bool, rehash: bool
) -> dict[str, Any]:
    raw = os.environ.get(EXEC_AUTHORITY_ENV)
    if raw is None:
        raise TorchrunFDBridgeV2Error("rank exec-authority binding is absent")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TorchrunFDBridgeV2Error(
            "rank exec-authority binding is not JSON"
        ) from error
    if not isinstance(value, dict) or _canonical_json_bytes(value).decode("utf-8") != raw:
        raise TorchrunFDBridgeV2Error("rank exec-authority binding differs")
    unsigned = dict(value)
    claimed = unsigned.pop("binding_digest", None)
    rows = value.get("rows")
    if (
        set(value)
        != {"schema_version", "rows", "rows_digest", "binding_digest"}
        or value.get("schema_version") != EXEC_AUTHORITY_SCHEMA
        or type(rows) is not list
        or [row.get("role") if isinstance(row, Mapping) else None for row in rows]
        != list(_RANK_EXEC_ROLES)
        or value.get("rows_digest")
        != hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()
        or claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise TorchrunFDBridgeV2Error("rank exec-authority digest differs")
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
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            or not isinstance(row.get("identity"), Mapping)
            or set(row["identity"]) != _EXEC_IDENTITY_FIELDS
            or any(type(item) is not int for item in row["identity"].values())
            or (
                row.get("role")
                in {"python_executable", "ffmpeg_executable"}
                and not row["identity"]["mode"] & 0o111
            )
        ):
            raise TorchrunFDBridgeV2Error("rank exec-authority row differs")
        try:
            before = os.fstat(row["fd"])
            inheritable = os.get_inheritable(row["fd"])
            raw_file = _pread_exact(row["fd"], before.st_size) if rehash else None
            after = os.fstat(row["fd"])
        except OSError as error:
            raise TorchrunFDBridgeV2Error(
                "rank exec-authority FD is unavailable"
            ) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or _exec_stat_identity(before) != dict(row["identity"])
            or _exec_stat_identity(after) != dict(row["identity"])
            or inheritable is not expected_inheritable
            or (
                raw_file is not None
                and hashlib.sha256(raw_file).hexdigest() != row["sha256"]
            )
        ):
            raise TorchrunFDBridgeV2Error("rank exec-authority FD replay differs")
        numbers.append(row["fd"])
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise TorchrunFDBridgeV2Error("rank exec-authority FD allowlist differs")
    return value


def load_empty_publication_handoff(
    *, expected_inheritable: bool
) -> dict[str, Any]:
    raw = os.environ.get(PUBLICATION_HANDOFF_ENV)
    if raw is None:
        raise TorchrunFDBridgeV2Error("publication handoff authority is absent")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TorchrunFDBridgeV2Error(
            "publication handoff authority is not JSON"
        ) from error
    unsigned = dict(value) if isinstance(value, dict) else {}
    claimed = unsigned.pop("authority_digest", None)
    identity = value.get("initial_identity") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or _canonical_json_bytes(value).decode("utf-8") != raw
        or set(value)
        != {
            "schema_version",
            "task_id",
            "fd",
            "initial_identity",
            "capacity",
            "authority_digest",
        }
        or value.get("schema_version") != PUBLICATION_HANDOFF_AUTHORITY_SCHEMA
        or claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        or type(value.get("task_id")) is not str
        or not value["task_id"]
        or type(value.get("fd")) is not int
        or value["fd"] < 3
        or value.get("capacity") != 65536
        or not isinstance(identity, Mapping)
        or set(identity) != _EXEC_IDENTITY_FIELDS
        or any(type(item) is not int for item in identity.values())
    ):
        raise TorchrunFDBridgeV2Error("publication handoff authority differs")
    try:
        observed = os.fstat(value["fd"])
        inheritable = os.get_inheritable(value["fd"])
        seals = fcntl.fcntl(value["fd"], fcntl.F_GET_SEALS)
    except (OSError, AttributeError) as error:
        raise TorchrunFDBridgeV2Error(
            "publication handoff descriptor is unavailable"
        ) from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 0
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != 0
        or _exec_stat_identity(observed) != dict(identity)
        or inheritable is not expected_inheritable
        or seals != 0
    ):
        raise TorchrunFDBridgeV2Error("empty publication handoff replay differs")
    return {**value, "initial_identity": dict(identity)}


def _stable_file_identity(
    path_value: str | Path, expected_sha256: str
) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise TorchrunFDBridgeV2Error("pinned source SHA differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
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
        or size <= 0
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
        raise TorchrunFDBridgeV2Error(f"pinned source identity differs: {path}")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "size": size,
        "mode": stat.S_IMODE(before.st_mode),
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "nlink": before.st_nlink,
    }


def _stable_directory_identity(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.resolve(strict=True) != path
        or path.is_symlink()
    ):
        raise TorchrunFDBridgeV2Error("site-packages path differs")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        middle = os.fstat(descriptor)
        named = path.lstat()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_rdev,
    )
    if (
        not stat.S_ISDIR(before.st_mode)
        or identity
        != (
            middle.st_dev,
            middle.st_ino,
            middle.st_mode,
            middle.st_uid,
            middle.st_gid,
            middle.st_rdev,
        )
        or identity
        != (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_uid,
            named.st_gid,
            named.st_rdev,
        )
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_rdev,
        )
    ):
        raise TorchrunFDBridgeV2Error("site-packages directory identity differs")
    return {
        "path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "uid": before.st_uid,
        "gid": before.st_gid,
        "rdev": before.st_rdev,
    }


def _require_isolated_startup() -> None:
    if (
        sys.flags.no_site != 1
        or sys.flags.isolated != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
    ):
        raise TorchrunFDBridgeV2Error(
            "bridge requires -I -S -B before authority FD entry"
        )


def validate_coordinator_environment(
    environment: Mapping[str, str],
    *,
    inherited: Mapping[str, Any],
    exec_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
) -> dict[str, str]:
    """Bind Torch imports to the exact empty-base environment from the runner."""

    expected_keys = set(_COORDINATOR_FIXED_ENV) | _COORDINATOR_DYNAMIC_ENV_KEYS
    inherited_literal = model_authority.inherited_fd_environment_value(inherited)
    executable_literal = _canonical_json_bytes(exec_authority).decode("utf-8")
    handoff_literal = _canonical_json_bytes(publication_handoff).decode("utf-8")
    cache_literal = environment.get(RANK_CACHE_ENV)
    cache_path = Path(cache_literal) if isinstance(cache_literal, str) else Path()
    if (
        set(environment) != expected_keys
        or any(
            type(key) is not str or type(value) is not str
            for key, value in environment.items()
        )
        or any(
            environment.get(key) != value
            for key, value in _COORDINATOR_FIXED_ENV.items()
        )
        or environment.get("APV2_EVAL_INHERITED_AUTHORITY_FDS")
        != inherited_literal
        or environment.get(EXEC_AUTHORITY_ENV) != executable_literal
        or environment.get(PUBLICATION_HANDOFF_ENV) != handoff_literal
        or any(
            not environment.get(key)
            or environment[key].strip() != environment[key]
            for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")
        )
        or not cache_path.is_absolute()
        or os.path.normpath(str(cache_path)) != str(cache_path)
    ):
        raise TorchrunFDBridgeV2Error(
            "coordinator environment allowlist differs"
        )
    return {key: environment[key] for key in sorted(expected_keys)}


def _configure_coordinator_pycache() -> Path:
    raw = os.environ.get(RANK_CACHE_ENV)
    if raw is None:
        raise TorchrunFDBridgeV2Error("coordinator cache root is absent")
    root = Path(raw)
    if (
        not root.is_absolute()
        or os.path.normpath(str(root)) != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise TorchrunFDBridgeV2Error("coordinator cache root differs")
    coordinator = root / "coordinator"
    try:
        coordinator.mkdir(mode=0o700)
        for name in ("pycache", "home", "hf", "torch", "xdg", "tmp"):
            (coordinator / name).mkdir(mode=0o700)
    except FileExistsError as error:
        raise TorchrunFDBridgeV2Error(
            "coordinator cache is not fresh"
        ) from error
    cache = coordinator / "pycache"
    sys.pycache_prefix = str(cache)
    if sys.pycache_prefix != str(cache) or any(cache.iterdir()):
        raise TorchrunFDBridgeV2Error("coordinator bytecode cache differs")
    os.environ.update(
        {
            "HOME": str(coordinator / "home"),
            "HF_HOME": str(coordinator / "hf"),
            "TORCH_HOME": str(coordinator / "torch"),
            "XDG_CACHE_HOME": str(coordinator / "xdg"),
            "TMPDIR": str(coordinator / "tmp"),
            "TMP": str(coordinator / "tmp"),
            "TEMP": str(coordinator / "tmp"),
        }
    )
    return cache


def _site_packages_for_torch_sources(
    torchrun_source: str | Path, handler_source: str | Path
) -> Path:
    run = Path(torchrun_source).resolve(strict=True)
    handler = Path(handler_source).resolve(strict=True)
    try:
        run_root = run.parents[2]
        handler_root = handler.parents[5]
    except IndexError as error:
        raise TorchrunFDBridgeV2Error("Torch source layout differs") from error
    if (
        run != run_root / "torch/distributed/run.py"
        or handler
        != handler_root
        / "torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py"
        or run_root != handler_root
    ):
        raise TorchrunFDBridgeV2Error("Torch source/site-packages layout differs")
    return run_root


def _activate_pinned_site_packages(
    torchrun_source: str | Path, handler_source: str | Path
) -> dict[str, Any]:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise TorchrunFDBridgeV2Error("Torch was imported before FD sealing")
    root = _site_packages_for_torch_sources(torchrun_source, handler_source)
    identity = _stable_directory_identity(root)
    if any(
        value
        and Path(value).resolve(strict=False) == root
        for value in sys.path
    ):
        raise TorchrunFDBridgeV2Error(
            "site-packages was active before pinned activation"
        )
    sys.path.append(str(root))
    if sys.path[-1] != str(root):
        raise TorchrunFDBridgeV2Error("site-packages activation differs")
    return identity


def load_bootstrap_sealed_authority_fds() -> dict[str, Any]:
    """Replay the descriptor table already sealed by the ``-c`` bootstrap."""

    try:
        inherited = model_authority.load_inherited_fd_environment(
            verify_open_fds=True,
            expected_inheritable=False,
        )
        model_authority.validate_inherited_fd_binding(
            inherited,
            verify_open_fds=True,
            expected_inheritable=False,
        )
    except model_authority.ModelConsumptionAuthorityError as error:
        raise TorchrunFDBridgeV2Error(str(error)) from error
    return inherited


def _parse_exact_int(environment: Mapping[str, str], key: str) -> int:
    raw = environment.get(key)
    try:
        value = int(raw) if raw is not None else -1
    except ValueError as error:
        raise TorchrunFDBridgeV2Error(f"rank environment {key} differs") from error
    if raw is None or str(value) != raw:
        raise TorchrunFDBridgeV2Error(f"rank environment {key} differs")
    return value


def validate_rank_environment(
    environment: Mapping[str, str],
    *,
    base_environment: Mapping[str, str],
    inherited_literal: str,
) -> int:
    if (
        type(environment) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in environment.items())
        or set(environment) != set(base_environment) | _WORKER_ENV_KEYS
        or any(
            environment.get(key) != value
            for key, value in base_environment.items()
            if key not in _WORKER_ENV_KEYS
        )
        or environment.get(model_authority.INHERITED_FD_BINDING_ENV)
        != inherited_literal
    ):
        raise TorchrunFDBridgeV2Error("rank environment allowlist differs")
    rank = _parse_exact_int(environment, "RANK")
    local_rank = _parse_exact_int(environment, "LOCAL_RANK")
    port = _parse_exact_int(environment, "MASTER_PORT")
    error_file = environment.get("TORCHELASTIC_ERROR_FILE", "")
    error_path = Path(error_file)
    if (
        rank not in range(4)
        or local_rank != rank
        or _parse_exact_int(environment, "ROLE_RANK") != rank
        or _parse_exact_int(environment, "GROUP_RANK") != 0
        or _parse_exact_int(environment, "WORLD_SIZE") != 4
        or _parse_exact_int(environment, "LOCAL_WORLD_SIZE") != 4
        or _parse_exact_int(environment, "ROLE_WORLD_SIZE") != 4
        or _parse_exact_int(environment, "GROUP_WORLD_SIZE") != 1
        or environment.get("ROLE_NAME") != "default"
        or environment.get("MASTER_ADDR") != "localhost"
        or port not in range(1, 65536)
        or environment.get("TORCHELASTIC_RESTART_COUNT") != "0"
        or environment.get("TORCHELASTIC_MAX_RESTARTS") != "0"
        or _UUID_RE.fullmatch(environment.get("TORCHELASTIC_RUN_ID", "")) is None
        # PyTorch 2.7.1 --standalone selects the c10d rendezvous handler;
        # RendezvousHandler.use_agent_store is False (only static TCP is True).
        or environment.get("TORCHELASTIC_USE_AGENT_STORE") != "False"
        or environment.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") != "1"
        or environment.get("OMP_NUM_THREADS") != "4"
        or not error_path.is_absolute()
        or os.path.normpath(error_file) != error_file
        or error_path.name != "error.json"
        or error_path.parent.name != str(rank)
    ):
        raise TorchrunFDBridgeV2Error("rank environment values differ")
    return rank


@contextmanager
def patched_rank_spawner(
    handler_class: type,
    *,
    inherited: Mapping[str, Any],
    exec_authority: Mapping[str, Any],
    publication_handoff: Mapping[str, Any],
    expected_rank_argv: Sequence[str],
    base_environment: Mapping[str, str],
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> Iterator[set[int]]:
    original = handler_class._popen
    expected_argv = tuple(expected_rank_argv)
    if not expected_argv or any(type(value) is not str for value in expected_argv):
        raise TorchrunFDBridgeV2Error("rank argv differs")
    binding = model_authority.validate_inherited_fd_binding(
        inherited,
        verify_open_fds=True,
        expected_inheritable=False,
    )
    executable = load_rank_exec_authority(
        expected_inheritable=False,
        rehash=True,
    )
    if executable != dict(exec_authority):
        raise TorchrunFDBridgeV2Error("rank exec-authority changed")
    handoff = load_empty_publication_handoff(expected_inheritable=False)
    if handoff != dict(publication_handoff):
        raise TorchrunFDBridgeV2Error("publication handoff authority changed")
    authority_fds = model_authority.inherited_fd_numbers(binding)
    code_fds = tuple(row["fd"] for row in executable["rows"])
    if (
        set(authority_fds) & set(code_fds)
        or handoff["fd"] in set(authority_fds) | set(code_fds)
    ):
        raise TorchrunFDBridgeV2Error(
            "rank model/code/handoff FD allowlists overlap"
        )
    pass_fds = tuple(sorted((*authority_fds, *code_fds, handoff["fd"])))
    python_row = executable["rows"][0]
    if python_row["role"] != "python_executable":
        raise TorchrunFDBridgeV2Error("rank Python executable role differs")
    inherited_literal = model_authority.inherited_fd_environment_value(binding)
    ranks: set[int] = set()

    def authority_popen(self: Any, args: Sequence[str], env: Mapping[str, str]) -> Any:
        if tuple(args) != expected_argv:
            raise TorchrunFDBridgeV2Error("rank process argv differs")
        rank = validate_rank_environment(
            env,
            base_environment=base_environment,
            inherited_literal=inherited_literal,
        )
        if rank in ranks or len(ranks) >= 4:
            raise TorchrunFDBridgeV2Error("rank spawn duplicate/retry differs")
        model_authority.validate_inherited_fd_binding(
            binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if load_rank_exec_authority(
            expected_inheritable=False,
            rehash=True,
        ) != executable:
            raise TorchrunFDBridgeV2Error("pre-rank exec-authority changed")
        process = popen_factory(
            args=tuple(args),
            env=dict(env),
            stdout=self._stdout,
            stderr=self._stderr,
            start_new_session=True,
            close_fds=True,
            pass_fds=pass_fds,
            executable=f"/proc/self/fd/{python_row['fd']}",
        )
        model_authority.validate_inherited_fd_binding(
            binding,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if load_rank_exec_authority(
            expected_inheritable=False,
            rehash=True,
        ) != executable:
            raise TorchrunFDBridgeV2Error("post-rank exec-authority changed")
        ranks.add(rank)
        return process

    handler_class._popen = authority_popen
    try:
        yield ranks
    finally:
        hook_changed = handler_class._popen is not authority_popen
        handler_class._popen = original
        if hook_changed or handler_class._popen is not original:
            raise TorchrunFDBridgeV2Error("Torch Elastic rank hook was not restored")


def _load_pinned_torchrun(
    *, torchrun_source: str, handler_source: str
) -> tuple[Any, type, dict[str, Any]]:
    run_identity = _stable_file_identity(torchrun_source, TORCHRUN_SOURCE_SHA256)
    handler_identity = _stable_file_identity(
        handler_source, TORCHRUN_HANDLER_SHA256
    )
    site_identity = _activate_pinned_site_packages(
        torchrun_source, handler_source
    )
    handler_name = (
        "torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler"
    )
    handler_module = _load_exact_source_module(
        handler_name,
        handler_source,
        TORCHRUN_HANDLER_SHA256,
        require_absent=True,
    )
    handler_package = importlib.import_module(
        "torch.distributed.elastic.multiprocessing.subprocess_handler"
    )
    handlers_module = importlib.import_module(
        "torch.distributed.elastic.multiprocessing.subprocess_handler.handlers"
    )
    api_module = importlib.import_module("torch.distributed.elastic.multiprocessing.api")
    run_module = _load_exact_source_module(
        "torch.distributed.run",
        torchrun_source,
        TORCHRUN_SOURCE_SHA256,
        require_absent=True,
    )
    handler_class = handler_module.SubprocessHandler
    if (
        Path(run_module.__file__).resolve(strict=True) != Path(run_identity["path"])
        or Path(handler_module.__file__).resolve(strict=True)
        != Path(handler_identity["path"])
        or handler_package.SubprocessHandler is not handler_class
        or handlers_module.SubprocessHandler is not handler_class
        or api_module.SubprocessHandler is not handler_class
        or run_module.__cached__ is not None
        or handler_module.__cached__ is not None
    ):
        raise TorchrunFDBridgeV2Error("pinned Torch Elastic origins differ")
    return run_module, handler_class, {
        "torchrun": run_identity,
        "subprocess_handler": handler_identity,
        "site_packages": site_identity,
    }


def _split_bridge_arguments(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    if values.count("--") != 1:
        raise TorchrunFDBridgeV2Error("bridge/inference argv delimiter differs")
    index = values.index("--")
    bridge_values = values[:index]
    inference_values = values[index + 1 :]
    if not inference_values:
        raise TorchrunFDBridgeV2Error("inference argv is absent")
    return bridge_values, inference_values


def build_torchrun_arguments(expected_rank_argv: Sequence[str]) -> list[str]:
    """Build the exact pinned Torch 2.7.1 local-elastic argv.

    ``--standalone`` selects a localhost rendezvous endpoint but does not by
    itself force the worker bootstrap address: Torch derives that address from
    the local-node address, whose default is the host FQDN.  Pinning
    ``--local-addr=localhost`` makes the emitted ``MASTER_ADDR`` match the rank
    environment contract instead of relying on node-specific DNS spelling.
    """

    values = list(expected_rank_argv)
    if not values or any(type(value) is not str or not value for value in values):
        raise TorchrunFDBridgeV2Error("rank argv differs")
    return [
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=4",
        "--max_restarts=0",
        "--local-addr=localhost",
        "--no-python",
        *values,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-sha256", required=True)
    parser.add_argument("--adapter-script", required=True)
    parser.add_argument("--adapter-script-sha256", required=True)
    parser.add_argument("--rank-python", required=True)
    parser.add_argument("--rank-python-sha256", required=True)
    parser.add_argument("--ffmpeg-executable", required=True)
    parser.add_argument("--ffmpeg-executable-sha256", required=True)
    parser.add_argument("--torchrun-source", required=True)
    parser.add_argument(
        "--torchrun-source-sha256", default=TORCHRUN_SOURCE_SHA256
    )
    parser.add_argument("--torchrun-handler-source", required=True)
    parser.add_argument(
        "--torchrun-handler-source-sha256", default=TORCHRUN_HANDLER_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_isolated_startup()
    inherited = load_bootstrap_sealed_authority_fds()
    exec_authority = load_rank_exec_authority(
        expected_inheritable=False,
        rehash=True,
    )
    publication_handoff = load_empty_publication_handoff(
        expected_inheritable=False
    )
    if publication_handoff["task_id"] != inherited.get("task_id"):
        raise TorchrunFDBridgeV2Error(
            "publication handoff task identity differs"
        )
    base_environment = validate_coordinator_environment(
        os.environ,
        inherited=inherited,
        exec_authority=exec_authority,
        publication_handoff=publication_handoff,
    )
    gpu_visibility = gpu47_visibility_contract(base_environment)
    coordinator_cache = _configure_coordinator_pycache()
    coordinator_root = coordinator_cache.parent
    base_environment.update(
        {
            "HOME": str(coordinator_root / "home"),
            "HF_HOME": str(coordinator_root / "hf"),
            "TORCH_HOME": str(coordinator_root / "torch"),
            "XDG_CACHE_HOME": str(coordinator_root / "xdg"),
            "TMPDIR": str(coordinator_root / "tmp"),
            "TMP": str(coordinator_root / "tmp"),
            "TEMP": str(coordinator_root / "tmp"),
        }
    )
    if dict(os.environ) != base_environment:
        raise TorchrunFDBridgeV2Error(
            "coordinator cache environment differs"
        )
    bridge_values, inference_arguments = _split_bridge_arguments(
        sys.argv[1:] if argv is None else argv
    )
    args = build_parser().parse_args(bridge_values)
    if (
        args.torchrun_source_sha256 != TORCHRUN_SOURCE_SHA256
        or args.torchrun_handler_source_sha256 != TORCHRUN_HANDLER_SHA256
    ):
        raise TorchrunFDBridgeV2Error("Torch Elastic source pins differ")
    identities_before = {
        "bridge": _stable_file_identity(__file__, args.bridge_sha256),
        "adapter": _stable_file_identity(
            args.adapter_script, args.adapter_script_sha256
        ),
        "rank_python": _stable_file_identity(
            args.rank_python, args.rank_python_sha256
        ),
        "ffmpeg": _stable_file_identity(
            args.ffmpeg_executable, args.ffmpeg_executable_sha256
        ),
        "model_authority": _stable_file_identity(
            _MODEL_AUTHORITY_PATH, MODEL_AUTHORITY_SHA256
        ),
    }
    run_module, handler_class, torch_identities = _load_pinned_torchrun(
        torchrun_source=args.torchrun_source,
        handler_source=args.torchrun_handler_source,
    )
    identities_before.update(torch_identities)
    python_row, adapter_row, ffmpeg_row = exec_authority["rows"]
    if (
        python_row["source_path"] != identities_before["rank_python"]["path"]
        or python_row["sha256"] != identities_before["rank_python"]["sha256"]
        or adapter_row["source_path"] != identities_before["adapter"]["path"]
        or adapter_row["sha256"] != identities_before["adapter"]["sha256"]
        or ffmpeg_row["source_path"] != identities_before["ffmpeg"]["path"]
        or ffmpeg_row["sha256"] != identities_before["ffmpeg"]["sha256"]
    ):
        raise TorchrunFDBridgeV2Error(
            "rank retained executable/source identity differs"
        )
    if SITE_PACKAGES_ENV in os.environ:
        raise TorchrunFDBridgeV2Error(
            "site-packages authority was present before bridge capture"
        )
    os.environ[SITE_PACKAGES_ENV] = identities_before["site_packages"]["path"]
    base_environment[SITE_PACKAGES_ENV] = identities_before["site_packages"][
        "path"
    ]
    if dict(os.environ) != base_environment:
        raise TorchrunFDBridgeV2Error(
            "coordinator Torch environment differs"
        )
    expected_rank_argv = [
        identities_before["rank_python"]["path"],
        "-I",
        "-S",
        "-B",
        "-c",
        ISOLATED_RANK_BOOTSTRAP,
        *inference_arguments,
    ]
    torchrun_arguments = build_torchrun_arguments(expected_rank_argv)
    identities_after: dict[str, Any] = {}
    try:
        with patched_rank_spawner(
            handler_class,
            inherited=inherited,
            exec_authority=exec_authority,
            publication_handoff=publication_handoff,
            expected_rank_argv=expected_rank_argv,
            base_environment=base_environment,
        ) as spawned_ranks:
            return_value = run_module.main(torchrun_arguments)
    finally:
        model_authority.validate_inherited_fd_binding(
            inherited,
            verify_open_fds=True,
            expected_inheritable=False,
        )
        if load_rank_exec_authority(
            expected_inheritable=False,
            rehash=True,
        ) != exec_authority:
            raise TorchrunFDBridgeV2Error("bridge exec-authority changed during task")
        identities_after = {
            "bridge": _stable_file_identity(__file__, args.bridge_sha256),
            "adapter": _stable_file_identity(
                args.adapter_script, args.adapter_script_sha256
            ),
            "rank_python": _stable_file_identity(
                args.rank_python, args.rank_python_sha256
            ),
            "ffmpeg": _stable_file_identity(
                args.ffmpeg_executable,
                args.ffmpeg_executable_sha256,
            ),
            "model_authority": _stable_file_identity(
                _MODEL_AUTHORITY_PATH, MODEL_AUTHORITY_SHA256
            ),
            "torchrun": _stable_file_identity(
                args.torchrun_source, TORCHRUN_SOURCE_SHA256
            ),
            "subprocess_handler": _stable_file_identity(
                args.torchrun_handler_source, TORCHRUN_HANDLER_SHA256
            ),
            "site_packages": _stable_directory_identity(
                identities_before["site_packages"]["path"]
            ),
        }
        if (
            identities_after != identities_before
            or gpu47_visibility_contract(os.environ) != gpu_visibility
            or run_module.__cached__ is not None
            or handler_class.__module__
            != "torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler"
            or sys.pycache_prefix != str(coordinator_cache)
            or any(coordinator_cache.iterdir())
        ):
            raise TorchrunFDBridgeV2Error("bridge origins changed during task")
    if return_value not in (None, 0) or spawned_ranks != {0, 1, 2, 3}:
        raise TorchrunFDBridgeV2Error("exact four-rank torchrun completion differs")
    receipt = {
        "schema_version": SCHEMA,
        "rank_count": 4,
        "ranks_spawned_once": [0, 1, 2, 3],
        "max_restarts": 0,
        "rank_argv_digest": hashlib.sha256(
            _canonical_json_bytes(expected_rank_argv)
        ).hexdigest(),
        "authority_fd_count": len(model_authority.inherited_fd_numbers(inherited)),
        "authority_fds_resealed_in_coordinator": True,
        "exec_authority_digest": exec_authority["binding_digest"],
        "retained_python_and_adapter_source_fds": True,
        "rank_entry_validate_then_seal_required": True,
        "gpu_visibility_contract": gpu_visibility,
        "gpu_isolation_scope": (
            "ROCr-only logical isolation; not device-cgroup hard isolation"
        ),
        "isolated_no_site_exec": "-I -S -B",
        "pinned_source_only_torchrun_and_handler": True,
        "fresh_empty_coordinator_pycache": str(coordinator_cache),
        "hook_restored": True,
        "origins": identities_after,
    }
    receipt["receipt_digest"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    print(_canonical_json_bytes(receipt).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
