#!/usr/bin/env python3
"""Create-only exact8 uploader and captured Phase-A author for r7/exact15-r3.

The production plan is deliberately supplied as canonical JSON together with
an independently reviewed literal SHA-256.  Upload captures every local input
from one descriptor (two complete reads plus full identity replay), then sends
a self-contained Python program to the remote host.  That program creates a
fresh bundle, writes every member with ``O_EXCL|O_NOFOLLOW``, and seals both
directories read-only.  It never repairs or overwrites a partial deployment.

The separate Phase-A author captures the prepare helper from one retained file
descriptor and compiles those captured bytes.  It never starts the controller
argv printed by the helper.  A create-only authorization receipt is published
only after helper, bundle, request, and retained work-root identities replay.
No command in this module launches Slurm, torchrun, inference, training, or a
GPU process.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "bernini-action-preservation-decoded-eval-exact8-deployment-plan-v1"
UPLOAD_SCHEMA = "bernini-action-preservation-decoded-eval-exact8-upload-v1"
AUTHOR_SCHEMA = "bernini-action-preservation-decoded-eval-phase-a-authority-v1"
GENERATION = "preservation-v2-decoded-eval-exact15-r3"
EXPECTED_REMOTE_PATHS: Mapping[str, str] = {
    "archive": "exact15-r3-release/source.tar",
    "manifest": "exact15-r3-release/source.manifest.json",
    "envelope": "exact15-r3-release/deployment-envelope.json",
    "controller": "action_preservation_decoded_eval_deployment_controller_v1.py",
    "runtime": "action_preservation_decoded_eval_verified_release_v1.py",
    "source_preprocessing": (
        "action_preservation_decoded_eval_r7_source_preprocessing_authority_v1.json"
    ),
    "input_authority": (
        "action_preservation_decoded_eval_r7_exact15_r3_input_authority.json"
    ),
    "prepare_helper": (
        "prepare_action_preservation_decoded_eval_r7_exact15_r3_v1.py"
    ),
}
EXPECTED_TOP_LEVEL = tuple(
    sorted({path.split("/", 1)[0] for path in EXPECTED_REMOTE_PATHS.values()})
)
EXPECTED_RELEASE_ENTRIES = tuple(
    sorted(path.split("/", 1)[1] for path in EXPECTED_REMOTE_PATHS.values()
           if "/" in path)
)
EXPECTED_PHASE_A_PATH_NAMES: Mapping[str, str] = {
    "deployment_request": "deployment-request.json",
    "materialized_release": "materialized-release",
    "controller_authority_receipt": "controller-authority.json",
    "deployment_receipt": "deployment-receipt.json",
    "source_runtime_spec": "source-runtime-spec.json",
    "source_spec_authority_receipt": "source-spec-authority.json",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class Exact8DeploymentError(RuntimeError):
    """An exact8 deployment or Phase-A authority invariant differs."""


def fail(message: str) -> None:
    raise Exact8DeploymentError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Exact8DeploymentError("value is not finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _absolute(value: Any, *, label: str) -> Path:
    if type(value) is not str:
        fail(f"{label} is not a path")
    path = Path(value)
    if (
        not path.is_absolute() or value == os.path.sep
        or os.path.normpath(value) != value
    ):
        fail(f"{label} must be a normalized absolute non-root path")
    return path


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_uid, value.st_gid,
        value.st_mode, value.st_nlink, value.st_rdev, value.st_size,
        getattr(value, "st_blocks", 0), value.st_mtime_ns, value.st_ctime_ns,
    )


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                fail(f"{label} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Exact8DeploymentError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root differs")
    return value


def validate_plan(value: Any, *, expected_authority_digest: str) -> dict[str, Any]:
    fields = {
        "schema_version", "generation", "bundle_root", "work_root",
        "phase_a_authorization_receipt_path", "expected_uid", "expected_gid",
        "bundle_root_final_nlink", "release_directory_final_nlink",
        "phase_a_work_final_nlink",
        "phase_a_status", "files", "phase_a_paths", "automatic_retry",
        "network_allowed", "scientific_promotion_authorized",
        "authority_digest",
    }
    if type(value) is not dict or set(value) != fields:
        fail("deployment plan field closure differs")
    row = dict(value)
    unsigned = dict(row)
    claimed = unsigned.pop("authority_digest", None)
    if (
        type(expected_authority_digest) is not str
        or SHA256_RE.fullmatch(expected_authority_digest) is None
        or claimed != expected_authority_digest
        or claimed != object_sha256(unsigned)
    ):
        fail("deployment plan external authority digest differs")
    if (
        row["schema_version"] != PLAN_SCHEMA
        or row["generation"] != GENERATION
        or row["automatic_retry"] is not False
        or row["network_allowed"] is not False
        or row["scientific_promotion_authorized"] is not False
        or type(row["expected_uid"]) is not int or row["expected_uid"] < 0
        or type(row["expected_gid"]) is not int or row["expected_gid"] < 0
        or any(type(row[field]) is not int or row[field] < 2 for field in (
            "bundle_root_final_nlink", "release_directory_final_nlink",
            "phase_a_work_final_nlink",
        ))
        or type(row["phase_a_status"]) is not str or not row["phase_a_status"]
    ):
        fail("deployment plan authority differs")
    bundle = _absolute(row["bundle_root"], label="bundle root")
    work = _absolute(row["work_root"], label="work root")
    authorization = _absolute(
        row["phase_a_authorization_receipt_path"],
        label="Phase-A authorization receipt",
    )
    if (
        len({bundle, work, authorization}) != 3
        or any(left in right.parents or right in left.parents
               for index, left in enumerate((bundle, work, authorization))
               for right in (bundle, work, authorization)[index + 1:])
    ):
        fail("deployment plan root topology differs")
    paths = row["phase_a_paths"]
    if type(paths) is not dict or set(paths) != set(EXPECTED_PHASE_A_PATH_NAMES):
        fail("Phase-A path closure differs")
    for role, basename in EXPECTED_PHASE_A_PATH_NAMES.items():
        if paths[role] != str(work / basename):
            fail(f"Phase-A {role} path differs")
    files = row["files"]
    if type(files) is not dict or set(files) != set(EXPECTED_REMOTE_PATHS):
        fail("exact8 role closure differs")
    local_paths: list[Path] = []
    for role, relative in EXPECTED_REMOTE_PATHS.items():
        item = files[role]
        if type(item) is not dict or set(item) != {
            "source_path", "source_mode", "remote_path", "remote_mode",
            "sha256", "size",
        }:
            fail(f"exact8 {role} field closure differs")
        local_paths.append(_absolute(item["source_path"], label=f"{role} source"))
        if (
            item["remote_path"] != relative
            or type(item["sha256"]) is not str
            or SHA256_RE.fullmatch(item["sha256"]) is None
            or type(item["size"]) is not int or item["size"] <= 0
            or type(item["source_mode"]) is not int
            or type(item["remote_mode"]) is not int
            or item["source_mode"] not in {0o444, 0o555, 0o644, 0o755}
            or item["remote_mode"] != 0o444
        ):
            fail(f"exact8 {role} pin differs")
    if len(set(local_paths)) != 8:
        fail("exact8 source paths are not distinct")
    return row


def build_plan(
    *, bundle_root: Path, work_root: Path,
    phase_a_authorization_receipt_path: Path,
    expected_uid: int, expected_gid: int, phase_a_status: str,
    files: Mapping[str, Mapping[str, Any]],
    bundle_root_final_nlink: int = 3,
    release_directory_final_nlink: int = 2,
    phase_a_work_final_nlink: int = 2,
) -> dict[str, Any]:
    """Build a canonical plan; production still requires its external digest."""
    value: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA, "generation": GENERATION,
        "bundle_root": str(bundle_root), "work_root": str(work_root),
        "phase_a_authorization_receipt_path": str(
            phase_a_authorization_receipt_path
        ),
        "expected_uid": expected_uid, "expected_gid": expected_gid,
        "bundle_root_final_nlink": bundle_root_final_nlink,
        "release_directory_final_nlink": release_directory_final_nlink,
        "phase_a_work_final_nlink": phase_a_work_final_nlink,
        "phase_a_status": phase_a_status,
        "files": {key: dict(item) for key, item in files.items()},
        "phase_a_paths": {
            role: str(work_root / name)
            for role, name in EXPECTED_PHASE_A_PATH_NAMES.items()
        },
        "automatic_retry": False, "network_allowed": False,
        "scientific_promotion_authorized": False,
    }
    value["authority_digest"] = object_sha256(value)
    validate_plan(value, expected_authority_digest=value["authority_digest"])
    return value


def capture_local_payloads(
    plan: Mapping[str, Any], *, expected_authority_digest: str,
) -> dict[str, str]:
    """Capture all exact8 local files; returns base64, never remote paths."""
    row = validate_plan(plan, expected_authority_digest=expected_authority_digest)
    if not hasattr(os, "O_NOFOLLOW"):
        fail("safe local capture is unavailable")
    payloads: dict[str, str] = {}
    observed_inodes: set[tuple[int, int]] = set()
    for role in sorted(EXPECTED_REMOTE_PATHS):
        pin = row["files"][role]
        path = Path(pin["source_path"])
        if path.resolve(strict=True) != path:
            fail(f"{role} source path is not canonical")
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            before = os.fstat(descriptor)
            first = _read_fd(descriptor)
            middle = os.fstat(descriptor)
            second = _read_fd(descriptor)
            after = os.fstat(descriptor)
            named = path.lstat()
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or _identity(before) != _identity(middle)
            or _identity(before) != _identity(after)
            or _identity(before) != _identity(named)
            or first != second or len(first) != pin["size"]
            or hashlib.sha256(first).hexdigest() != pin["sha256"]
            or stat.S_IMODE(before.st_mode) != pin["source_mode"]
        ):
            fail(f"{role} local physical identity or bytes differ")
        key = (before.st_dev, before.st_ino)
        if key in observed_inodes:
            fail("exact8 local files alias one inode")
        observed_inodes.add(key)
        payloads[role] = base64.b64encode(first).decode("ascii")
    return payloads


# The remote engine is standard-library-only.  Tests execute the rendered
# programs in a clean subprocess, so the deployed code path itself is covered.
REMOTE_ENGINE_SOURCE = r'''
import base64,contextlib,hashlib,io,json,os,pathlib,re,stat,sys

SHA=re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_REMOTE_PATHS={
 "archive":"exact15-r3-release/source.tar",
 "manifest":"exact15-r3-release/source.manifest.json",
 "envelope":"exact15-r3-release/deployment-envelope.json",
 "controller":"action_preservation_decoded_eval_deployment_controller_v1.py",
 "runtime":"action_preservation_decoded_eval_verified_release_v1.py",
 "source_preprocessing":"action_preservation_decoded_eval_r7_source_preprocessing_authority_v1.json",
 "input_authority":"action_preservation_decoded_eval_r7_exact15_r3_input_authority.json",
 "prepare_helper":"prepare_action_preservation_decoded_eval_r7_exact15_r3_v1.py",
}
PHASE_NAMES={"deployment_request":"deployment-request.json","materialized_release":"materialized-release","controller_authority_receipt":"controller-authority.json","deployment_receipt":"deployment-receipt.json","source_runtime_spec":"source-runtime-spec.json","source_spec_authority_receipt":"source-spec-authority.json"}

def die(message): raise RuntimeError(message)
def canon(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(canon(value)).hexdigest()
def ident(value): return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)
def readfd(fd):
 os.lseek(fd,0,os.SEEK_SET); out=[]
 while True:
  block=os.read(fd,1024*1024)
  if not block:return b"".join(out)
  out.append(block)
def absolute(value,label):
 if type(value) is not str or not os.path.isabs(value) or value==os.path.sep or os.path.normpath(value)!=value:die(label+" path differs")
 return pathlib.Path(value)
def validate(plan,expected):
 fields={"schema_version","generation","bundle_root","work_root","phase_a_authorization_receipt_path","expected_uid","expected_gid","bundle_root_final_nlink","release_directory_final_nlink","phase_a_work_final_nlink","phase_a_status","files","phase_a_paths","automatic_retry","network_allowed","scientific_promotion_authorized","authority_digest"}
 if type(plan) is not dict or set(plan)!=fields:die("plan closure differs")
 unsigned=dict(plan); claimed=unsigned.pop("authority_digest",None)
 if type(expected) is not str or not SHA.fullmatch(expected) or claimed!=expected or claimed!=digest(unsigned):die("external plan digest differs")
 if plan["schema_version"]!="bernini-action-preservation-decoded-eval-exact8-deployment-plan-v1" or plan["generation"]!="preservation-v2-decoded-eval-exact15-r3" or plan["automatic_retry"] is not False or plan["network_allowed"] is not False or plan["scientific_promotion_authorized"] is not False:die("plan authority differs")
 bundle=absolute(plan["bundle_root"],"bundle"); work=absolute(plan["work_root"],"work"); receipt=absolute(plan["phase_a_authorization_receipt_path"],"receipt")
 if len({bundle,work,receipt})!=3:die("root topology differs")
 if type(plan["expected_uid"]) is not int or type(plan["expected_gid"]) is not int or plan["expected_uid"]<0 or plan["expected_gid"]<0:die("owner pin differs")
 if any(type(plan[field]) is not int or plan[field]<2 for field in ("bundle_root_final_nlink","release_directory_final_nlink","phase_a_work_final_nlink")):die("directory link-count pin differs")
 if type(plan["phase_a_status"]) is not str or not plan["phase_a_status"] or type(plan["phase_a_paths"]) is not dict or set(plan["phase_a_paths"])!=set(PHASE_NAMES):die("phase path closure differs")
 for role,name in PHASE_NAMES.items():
  if plan["phase_a_paths"][role]!=str(work/name):die("phase path differs: "+role)
 if type(plan["files"]) is not dict or set(plan["files"])!=set(EXPECTED_REMOTE_PATHS):die("exact8 closure differs")
 for role,relative in EXPECTED_REMOTE_PATHS.items():
  item=plan["files"][role]
  if type(item) is not dict or set(item)!={"source_path","source_mode","remote_path","remote_mode","sha256","size"}:die("file pin closure differs: "+role)
  absolute(item["source_path"],role+" source")
  if item["remote_path"]!=relative or type(item["sha256"]) is not str or not SHA.fullmatch(item["sha256"]) or type(item["size"]) is not int or item["size"]<=0 or item["remote_mode"]!=0o444:die("file pin differs: "+role)
 return plan
def open_dir(path):
 path=absolute(str(path),"directory")
 fd=os.open(os.path.sep,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
 try:
  for part in path.parts[1:]:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=fd); os.close(fd); fd=nxt
  return fd
 except BaseException:
  os.close(fd); raise
def open_parent(path): return open_dir(path.parent),path.name
def entries(fd): return sorted(os.listdir(fd))
def check_dir(fd,parent_fd,name,mode,uid,gid,expected,nlink):
 observed=os.fstat(fd); named=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
 if ident(observed)!=ident(named) or not stat.S_ISDIR(observed.st_mode) or (observed.st_uid,observed.st_gid,stat.S_IMODE(observed.st_mode),observed.st_nlink)!=(uid,gid,mode,nlink) or entries(fd)!=sorted(expected):die("directory identity/closure differs: "+name)
 return observed
def capture_at(dirfd,name,pin,uid,gid,hold=False):
 fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=dirfd)
 before=os.fstat(fd); first=readfd(fd); middle=os.fstat(fd); second=readfd(fd); after=os.fstat(fd); named=os.stat(name,dir_fd=dirfd,follow_symlinks=False)
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=second or len(first)!=pin["size"] or hashlib.sha256(first).hexdigest()!=pin["sha256"] or (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode))!=(uid,gid,pin["remote_mode"]):
  os.close(fd); die("captured file differs: "+name)
 binding={"path":None,"sha256":pin["sha256"],"size":pin["size"],"mode":pin["remote_mode"],"device":before.st_dev,"inode":before.st_ino,"uid":before.st_uid,"gid":before.st_gid,"nlink":before.st_nlink}
 if hold:return fd,first,before,binding
 os.close(fd); return binding
def create_file(dirfd,name,raw,pin,uid,gid):
 fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),pin["remote_mode"],dir_fd=dirfd)
 try:
  offset=0
  while offset<len(raw):
   count=os.write(fd,raw[offset:])
   if count<=0:die("write made no progress: "+name)
   offset+=count
  os.fchmod(fd,pin["remote_mode"]);os.fsync(fd)
  before=os.fstat(fd);first=readfd(fd);middle=os.fstat(fd);second=readfd(fd);after=os.fstat(fd);named=os.stat(name,dir_fd=dirfd,follow_symlinks=False)
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=raw or second!=raw or len(raw)!=pin["size"] or hashlib.sha256(raw).hexdigest()!=pin["sha256"] or (before.st_uid,before.st_gid,stat.S_IMODE(before.st_mode))!=(uid,gid,pin["remote_mode"]):die("published file differs: "+name)
  return before
 finally:os.close(fd)
def create_json_at(parent_fd,name,value,uid,gid):
 raw=canon(value)+b"\n";pin={"sha256":hashlib.sha256(raw).hexdigest(),"size":len(raw),"remote_mode":0o444}
 observed=create_file(parent_fd,name,raw,pin,uid,gid)
 return {"path":None,"sha256":pin["sha256"],"size":pin["size"],"mode":0o444,"device":observed.st_dev,"inode":observed.st_ino,"uid":observed.st_uid,"gid":observed.st_gid,"nlink":observed.st_nlink}
def row_matches(value,row):
 return type(row) is dict and set(row)=={"device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"} and all(row[key]==item for key,item in zip(("device","inode","uid","gid","mode","nlink","rdev","size","blocks","mtime_ns","ctime_ns"),ident(value)))
def immutable_matches(value,row):
 return type(row) is dict and set(row)=={"device","inode","uid","gid","mode","rdev"} and row=={"device":value.st_dev,"inode":value.st_ino,"uid":value.st_uid,"gid":value.st_gid,"mode":value.st_mode,"rdev":value.st_rdev}
def validate_bundle(plan,hold_helper=False):
 root=pathlib.Path(plan["bundle_root"]);parent_fd,name=open_parent(root);root_fd=None;release_fd=None;held=None
 try:
  root_fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=parent_fd)
  check_dir(root_fd,parent_fd,name,0o555,plan["expected_uid"],plan["expected_gid"],set(p.split("/",1)[0] for p in EXPECTED_REMOTE_PATHS.values()),plan["bundle_root_final_nlink"])
  release_fd=os.open("exact15-r3-release",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=root_fd)
  check_dir(release_fd,root_fd,"exact15-r3-release",0o555,plan["expected_uid"],plan["expected_gid"],[p.split("/",1)[1] for p in EXPECTED_REMOTE_PATHS.values() if "/" in p],plan["release_directory_final_nlink"])
  rows={}
  for role in sorted(EXPECTED_REMOTE_PATHS):
   pin=plan["files"][role];parts=pin["remote_path"].split("/");fd=release_fd if len(parts)==2 else root_fd;leaf=parts[-1]
   if role=="prepare_helper" and hold_helper:
    held=capture_at(fd,leaf,pin,plan["expected_uid"],plan["expected_gid"],True);binding=held[3]
   else:binding=capture_at(fd,leaf,pin,plan["expected_uid"],plan["expected_gid"])
   binding["path"]=str(root/pin["remote_path"]);rows[role]=binding
  return parent_fd,root_fd,release_fd,held,rows
 except BaseException:
  if held is not None:os.close(held[0])
  if release_fd is not None:os.close(release_fd)
  if root_fd is not None:os.close(root_fd)
  os.close(parent_fd);raise

def do_upload(plan,expected,payload):
 validate(plan,expected)
 if type(payload) is not dict or set(payload)!=set(EXPECTED_REMOTE_PATHS):die("payload closure differs")
 raw={}
 for role in sorted(payload):
  try:raw[role]=base64.b64decode(payload[role].encode("ascii"),validate=True)
  except Exception as error:raise RuntimeError("payload base64 differs: "+role) from error
  pin=plan["files"][role]
  if len(raw[role])!=pin["size"] or hashlib.sha256(raw[role]).hexdigest()!=pin["sha256"]:die("payload pin differs: "+role)
 root=pathlib.Path(plan["bundle_root"]);parent_fd,name=open_parent(root);root_fd=None;release_fd=None
 try:
  try:os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
  except FileNotFoundError:pass
  else:die("bundle root collision")
  os.mkdir(name,0o700,dir_fd=parent_fd);root_fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=parent_fd)
  check_dir(root_fd,parent_fd,name,0o700,plan["expected_uid"],plan["expected_gid"],[],2)
  os.mkdir("exact15-r3-release",0o700,dir_fd=root_fd);release_fd=os.open("exact15-r3-release",os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=root_fd)
  check_dir(release_fd,root_fd,"exact15-r3-release",0o700,plan["expected_uid"],plan["expected_gid"],[],2)
  rows=[]
  for role in sorted(EXPECTED_REMOTE_PATHS,key=lambda key:EXPECTED_REMOTE_PATHS[key]):
   pin=plan["files"][role];parts=pin["remote_path"].split("/");fd=release_fd if len(parts)==2 else root_fd
   observed=create_file(fd,parts[-1],raw[role],pin,plan["expected_uid"],plan["expected_gid"])
   rows.append({"role":role,"path":str(root/pin["remote_path"]),"sha256":pin["sha256"],"size":pin["size"],"mode":pin["remote_mode"],"device":observed.st_dev,"inode":observed.st_ino,"uid":observed.st_uid,"gid":observed.st_gid,"nlink":observed.st_nlink})
  os.fsync(release_fd);os.fchmod(release_fd,0o555);os.fsync(release_fd)
  check_dir(release_fd,root_fd,"exact15-r3-release",0o555,plan["expected_uid"],plan["expected_gid"],[p.split("/",1)[1] for p in EXPECTED_REMOTE_PATHS.values() if "/" in p],plan["release_directory_final_nlink"])
  os.fsync(root_fd);os.fchmod(root_fd,0o555);os.fsync(root_fd)
  check_dir(root_fd,parent_fd,name,0o555,plan["expected_uid"],plan["expected_gid"],set(p.split("/",1)[0] for p in EXPECTED_REMOTE_PATHS.values()),plan["bundle_root_final_nlink"]);os.fsync(parent_fd)
  receipt={"schema_version":"bernini-action-preservation-decoded-eval-exact8-upload-v1","generation":plan["generation"],"authority_digest":expected,"bundle_root":str(root),"bundle_root_mode":0o555,"release_directory_mode":0o555,"exact8_file_count":8,"files":rows,"fresh_root_created":True,"initial_root_exact_empty":True,"o_excl_no_follow":True,"same_fd_double_read_named_replay":True,"remote_upload_performed":True,"gpu_used":False,"automatic_retry":False,"scientific_promotion_authorized":False}
  receipt["receipt_digest"]=digest(receipt);return receipt
 finally:
  if release_fd is not None:os.close(release_fd)
  if root_fd is not None:os.close(root_fd)
  os.close(parent_fd)

def do_author(plan,expected,helper_literal):
 validate(plan,expected)
 helper_pin=plan["files"]["prepare_helper"]
 if type(helper_literal) is not dict or set(helper_literal)!={"sha256","size","mode"} or helper_literal!={"sha256":helper_pin["sha256"],"size":helper_pin["size"],"mode":helper_pin["remote_mode"]}:die("external helper literal differs")
 work=pathlib.Path(plan["work_root"]);work_parent_fd,work_name=open_parent(work);auth=pathlib.Path(plan["phase_a_authorization_receipt_path"]);auth_parent_fd,auth_name=open_parent(auth)
 parent_fd=root_fd=release_fd=helper_fd=None
 try:
  for fd,name,label in ((work_parent_fd,work_name,"work root"),(auth_parent_fd,auth_name,"authorization receipt")):
   try:os.stat(name,dir_fd=fd,follow_symlinks=False)
   except FileNotFoundError:pass
   else:die(label+" is not fresh")
  parent_fd,root_fd,release_fd,held,rows=validate_bundle(plan,True);helper_fd,helper_raw,helper_before,helper_binding=held
  helper_path=pathlib.Path(rows["prepare_helper"]["path"])
  namespace={"__name__":"_apv2_exact15_r3_captured_prepare","__file__":str(helper_path),"__package__":None,"__spec__":None,"__builtins__":__builtins__}
  exec(compile(helper_raw,str(helper_path),"exec",dont_inherit=True),namespace)
  required={"main","BUNDLE_ROOT","WORK_ROOT","DEPLOYMENT_REQUEST_PATH","MATERIALIZED_RELEASE_ROOT","CONTROLLER_AUTHORITY_PATH","DEPLOYMENT_RECEIPT_PATH","SOURCE_SPEC_PATH","SOURCE_SPEC_AUTHORITY_PATH"}
  if not required.issubset(namespace) or not callable(namespace["main"]):die("captured helper API closure differs")
  projections={"BUNDLE_ROOT":plan["bundle_root"],"WORK_ROOT":plan["work_root"],"DEPLOYMENT_REQUEST_PATH":plan["phase_a_paths"]["deployment_request"],"MATERIALIZED_RELEASE_ROOT":plan["phase_a_paths"]["materialized_release"],"CONTROLLER_AUTHORITY_PATH":plan["phase_a_paths"]["controller_authority_receipt"],"DEPLOYMENT_RECEIPT_PATH":plan["phase_a_paths"]["deployment_receipt"],"SOURCE_SPEC_PATH":plan["phase_a_paths"]["source_runtime_spec"],"SOURCE_SPEC_AUTHORITY_PATH":plan["phase_a_paths"]["source_spec_authority_receipt"]}
  for key,value in projections.items():
   if str(namespace[key])!=value:die("captured helper path projection differs: "+key)
  stream=io.StringIO()
  with contextlib.redirect_stdout(stream):code=namespace["main"](["phase-a"])
  if code!=0:die("captured helper Phase-A exit differs")
  output=stream.getvalue()
  if not output.endswith("\n") or output.count("\n")!=1:die("captured helper stdout closure differs")
  result=json.loads(output)
  result_fields={"status","deployment_request","request_digest","work_root_authority","work_root_initial","work_root_after_request","phase_a_expected_final_entries","controller_argv","controller_bootstrap_source_sha256","remote_process_executed","gpu_used"}
  if canon(result)+b"\n"!=output.encode("utf-8") or type(result) is not dict or set(result)!=result_fields or result.get("status")!=plan["phase_a_status"] or type(result.get("request_digest")) is not str or not SHA.fullmatch(result["request_digest"]) or type(result.get("controller_bootstrap_source_sha256")) is not str or not SHA.fullmatch(result["controller_bootstrap_source_sha256"]) or result.get("phase_a_expected_final_entries")!=["controller-authority.json","deployment-receipt.json","deployment-request.json","materialized-release"] or result.get("remote_process_executed") is not False or result.get("gpu_used") is not False:die("captured helper Phase-A result differs")
  request=result.get("deployment_request")
  if type(request) is not dict or set(request)!={"path","sha256","size","mode"} or request["path"]!=plan["phase_a_paths"]["deployment_request"] or request["mode"]!=0o444:die("Phase-A request binding differs")
  authority=result.get("work_root_authority");initial=result.get("work_root_initial");after_result=result.get("work_root_after_request")
  if type(authority) is not dict or type(initial) is not dict or type(after_result) is not dict:die("Phase-A work authority is absent")
  authority_unsigned=dict(authority);authority_claimed=authority_unsigned.pop("authority_digest",None)
  if authority.get("schema_version")!="bernini-action-preservation-decoded-eval-work-root-authority-v1" or authority.get("path")!=str(work) or authority.get("parent_path")!=str(work.parent) or authority.get("initial_entries")!=[] or authority.get("retained_parent_fd_through_request_publication") is not True or authority.get("retained_root_fd_through_request_publication") is not True or authority_claimed!=digest(authority_unsigned):die("Phase-A work authority differs")
  work_fd=os.open(work_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0),dir_fd=work_parent_fd)
  try:
   work_identity=check_dir(work_fd,work_parent_fd,work_name,0o700,plan["expected_uid"],plan["expected_gid"],["deployment-request.json"],plan["phase_a_work_final_nlink"])
   if not immutable_matches(work_identity,authority.get("immutable_identity")) or not row_matches(work_identity,after_result.get("identity")) or initial.get("path")!=str(work) or initial.get("entries")!=[] or after_result.get("path")!=str(work) or after_result.get("entries")!=["deployment-request.json"]:die("Phase-A retained work-root projection differs")
   pin={"sha256":request["sha256"],"size":request["size"],"remote_mode":0o444};request_fd,request_raw,_request_stat,request_binding=capture_at(work_fd,"deployment-request.json",pin,plan["expected_uid"],plan["expected_gid"],True)
   try:request_value=json.loads(request_raw)
   finally:os.close(request_fd)
   if canon(request_value)+b"\n"!=request_raw or request_value.get("work_root_authority")!=authority:die("request work-root authority continuity differs")
   request_binding["path"]=request["path"]
   work_after=os.fstat(work_fd);work_named=os.stat(work_name,dir_fd=work_parent_fd,follow_symlinks=False)
   if ident(work_identity)!=ident(work_after) or ident(work_identity)!=ident(work_named):die("work root replaced during Phase-A")
  finally:os.close(work_fd)
  helper_after=os.fstat(helper_fd);helper_again=readfd(helper_fd);helper_named=os.stat(helper_path.name,dir_fd=root_fd,follow_symlinks=False)
  if ident(helper_before)!=ident(helper_after) or ident(helper_before)!=ident(helper_named) or helper_again!=helper_raw:die("helper replaced during captured execution")
  check_dir(root_fd,parent_fd,pathlib.Path(plan["bundle_root"]).name,0o555,plan["expected_uid"],plan["expected_gid"],set(p.split("/",1)[0] for p in EXPECTED_REMOTE_PATHS.values()),plan["bundle_root_final_nlink"])
  controller_argv=result.get("controller_argv")
  if type(controller_argv) is not list or not controller_argv or not all(type(item) is str for item in controller_argv):die("Phase-A controller argv differs")
  receipt={"schema_version":"bernini-action-preservation-decoded-eval-phase-a-authority-v1","generation":plan["generation"],"authority_digest":expected,"bundle_root":plan["bundle_root"],"helper":rows["prepare_helper"],"external_helper_literal":{"sha256":helper_literal["sha256"],"size":helper_literal["size"],"mode":helper_literal["mode"]},"helper_executed_from_same_fd_captured_bytes":True,"work_root":{"path":str(work),"device":work_identity.st_dev,"inode":work_identity.st_ino,"uid":work_identity.st_uid,"gid":work_identity.st_gid,"mode":0o700,"entries":["deployment-request.json"]},"deployment_request":request_binding,"controller_invocation":{"argv":controller_argv,"executed":False},"preauthorized_receipt_paths":dict(plan["phase_a_paths"]),"remote_process_executed":False,"gpu_used":False,"automatic_retry":False,"network_used":False,"scientific_promotion_authorized":False}
  receipt["receipt_digest"]=digest(receipt);binding=create_json_at(auth_parent_fd,auth_name,receipt,plan["expected_uid"],plan["expected_gid"]);binding["path"]=str(auth);os.fsync(auth_parent_fd)
  return {"status":"PHASE_A_AUTHORIZED_NOT_EXECUTED","authorization_receipt":binding,"receipt_digest":receipt["receipt_digest"],"controller_argv_executed":False,"gpu_used":False}
 finally:
  if helper_fd is not None:os.close(helper_fd)
  if release_fd is not None:os.close(release_fd)
  if root_fd is not None:os.close(root_fd)
  if parent_fd is not None:os.close(parent_fd)
  os.close(auth_parent_fd);os.close(work_parent_fd)
'''


def _render(
    *, mode: str, plan: Mapping[str, Any], expected_authority_digest: str,
    payloads: Mapping[str, str] | None = None,
    helper_literal: Mapping[str, Any] | None = None,
) -> bytes:
    validate_plan(plan, expected_authority_digest=expected_authority_digest)
    envelope: dict[str, Any] = {
        "mode": mode, "plan": dict(plan),
        "expected_authority_digest": expected_authority_digest,
    }
    if mode == "upload":
        if type(payloads) is not dict or set(payloads) != set(EXPECTED_REMOTE_PATHS):
            fail("rendered upload payload closure differs")
        envelope["payloads"] = dict(payloads)
    elif mode == "authorize-phase-a":
        pin = plan["files"]["prepare_helper"]
        expected_helper = {
            "sha256": pin["sha256"], "size": pin["size"],
            "mode": pin["remote_mode"],
        }
        if helper_literal != expected_helper:
            fail("external literal helper SHA/size/mode differs")
        envelope["helper_literal"] = dict(helper_literal)
    else:
        fail("remote program mode differs")
    encoded = base64.b64encode(canonical_json_bytes(envelope)).decode("ascii")
    trailer = r'''
envelope=json.loads(base64.b64decode(__ENVELOPE_B64__.encode("ascii"),validate=True))
if canon(envelope)!=base64.b64decode(__ENVELOPE_B64__.encode("ascii"),validate=True):die("remote envelope serialization differs")
if envelope["mode"]=="upload":result=do_upload(envelope["plan"],envelope["expected_authority_digest"],envelope["payloads"])
elif envelope["mode"]=="authorize-phase-a":result=do_author(envelope["plan"],envelope["expected_authority_digest"],envelope["helper_literal"])
else:die("remote envelope mode differs")
sys.stdout.buffer.write(canon(result)+b"\n")
'''.replace("__ENVELOPE_B64__", repr(encoded))
    return (REMOTE_ENGINE_SOURCE + trailer).encode("utf-8")


def render_upload_program(
    plan: Mapping[str, Any], *, expected_authority_digest: str,
    payloads: Mapping[str, str],
) -> bytes:
    return _render(
        mode="upload", plan=plan,
        expected_authority_digest=expected_authority_digest,
        payloads=payloads,
    )


def render_phase_a_author_program(
    plan: Mapping[str, Any], *, expected_authority_digest: str,
    helper_sha256: str, helper_size: int, helper_mode: int,
) -> bytes:
    return _render(
        mode="authorize-phase-a", plan=plan,
        expected_authority_digest=expected_authority_digest,
        helper_literal={
            "sha256": helper_sha256, "size": helper_size,
            "mode": helper_mode,
        },
    )


def execute_remote_program(
    program: bytes, *, host: str = "auh",
) -> dict[str, Any]:
    """Execute one reviewed program remotely; callers choose upload or author."""
    if type(program) is not bytes or not program:
        fail("remote program bytes differ")
    process = subprocess.run(
        [
            "/usr/bin/ssh", "-o", "BatchMode=yes", host,
            "/usr/bin/python3.10", "-I", "-S", "-B", "-",
        ],
        input=program, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise Exact8DeploymentError(
            "remote exact8 program failed: "
            + process.stderr.decode("utf-8", "replace")
        )
    raw = process.stdout
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        fail("remote receipt stdout closure differs")
    value = _strict_json(raw, label="remote receipt")
    if raw != canonical_json_bytes(value) + b"\n":
        fail("remote receipt serialization differs")
    return value


# Filled only after the exact15-r3 helper/controller pair passes independent
# audit.  Keeping this fail-closed prevents an obsolete pin from being used.
PRODUCTION_PLAN: Mapping[str, Any] | None = None
PRODUCTION_AUTHORITY_DIGEST: str | None = None


def _production() -> tuple[dict[str, Any], str]:
    if PRODUCTION_PLAN is None or PRODUCTION_AUTHORITY_DIGEST is None:
        fail("production exact8 plan is not frozen")
    return (
        validate_plan(
            PRODUCTION_PLAN,
            expected_authority_digest=PRODUCTION_AUTHORITY_DIGEST,
        ),
        PRODUCTION_AUTHORITY_DIGEST,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--host", default="auh")
    author = commands.add_parser("authorize-phase-a")
    author.add_argument("--host", default="auh")
    commands.add_parser("print-plan")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan, authority = _production()
    if args.command == "print-plan":
        result: Mapping[str, Any] = plan
    elif args.command == "upload":
        payloads = capture_local_payloads(
            plan, expected_authority_digest=authority
        )
        result = execute_remote_program(
            render_upload_program(
                plan, expected_authority_digest=authority, payloads=payloads
            ),
            host=args.host,
        )
    else:
        helper = plan["files"]["prepare_helper"]
        result = execute_remote_program(
            render_phase_a_author_program(
                plan, expected_authority_digest=authority,
                helper_sha256=helper["sha256"], helper_size=helper["size"],
                helper_mode=helper["remote_mode"],
            ),
            host=args.host,
        )
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHOR_SCHEMA", "EXPECTED_REMOTE_PATHS", "Exact8DeploymentError",
    "GENERATION", "PLAN_SCHEMA", "PRODUCTION_AUTHORITY_DIGEST",
    "PRODUCTION_PLAN", "UPLOAD_SCHEMA", "build_plan",
    "canonical_json_bytes", "capture_local_payloads", "execute_remote_program",
    "object_sha256", "render_phase_a_author_program",
    "render_upload_program", "validate_plan",
]
