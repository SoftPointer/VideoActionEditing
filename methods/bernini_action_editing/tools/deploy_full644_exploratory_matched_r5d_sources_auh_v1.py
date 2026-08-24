#!/usr/bin/env python3
"""Create-only deployment of the six frozen r5d sources to AUH.

The controller is inert unless supplied the exact authorization token printed
by ``--audit-local``.  It captures every local source through a held descriptor,
ships canonical bytes over stdin, stages remotely with O_EXCL, and publishes
with Linux renameat2(RENAME_NOREPLACE).  It never removes or overwrites a path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Any, Sequence


SCHEMA = "full644-exploratory-matched-r5d-source-deploy-auh-v1"
PAYLOAD_SCHEMA = f"{SCHEMA}-payload"
REMOTE_RECEIPT_SCHEMA = f"{SCHEMA}-receipt"
LOCAL_ROOT = Path(__file__).resolve().parents[3]
REMOTE_SOURCE_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
)
REMOTE_STAGE_PARENT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments"
)
REMOTE_PYTHON = Path(
    "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
)
REMOTE_PYTHON_SHA256 = (
    "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
)
REMOTE_PYTHON_SIZE = 31_490_256
REMOTE_UID = 2012
REMOTE_GID = 2000

# Exact final six-file closure.  The consumed v1 materializer is deliberately
# absent; the fresh r2-bound implementation has a v2 filename and new digest.
SOURCE_SPECS: tuple[tuple[str, str, int], ...] = (
    (
        "methods/bernini_action_editing/"
        "full644_exploratory_matched_infer_adapter_auh_r5d.py",
        "5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a",
        25_854,
    ),
    (
        "methods/bernini_action_editing/"
        "full644_exploratory_matched_spooled_launcher_auh_r5d.py",
        "85ccc17b30d97a7bf048702cd8a8ed10c3421e01721902fea7db6242eac45753",
        15_851,
    ),
    (
        "methods/bernini_action_editing/"
        "full644_exploratory_matched_r5d_root_bootstrap_probe_runner_v1.py",
        "e4890e5d45c6a3982bab03f311711effc87efd29718ff8d5726ad4580b8a3845",
        17_798,
    ),
    (
        "methods/bernini_action_editing/"
        "full644_exploratory_matched_r5d_static_nomodel_probe_v1.py",
        "4b17a1919a6ef928d572f769d6713a4764b0d31dc6da77eb7498cc5152c6de6c",
        44_864,
    ),
    (
        "methods/bernini_action_editing/"
        "full644_exploratory_matched_r5d_cpu_consumption_probe_v1.py",
        "5c7f5caf5ad73aecacedda618e941308e4fc1b94218b71cdc44e88afc3d3f0ea",
        56_009,
    ),
    (
        "methods/bernini_action_editing/tools/"
        "materialize_full644_exploratory_matched_r5d_case00_package_v2.py",
        "79a3fc988bbdcd74abf13823a7046ffba9c984f660e57b0c0678b4412cfa96e6",
        36_169,
    ),
)


class SourceDeployError(RuntimeError):
    """The frozen source deployment contract differs."""


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


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_rdev,
        info.st_size,
        getattr(info, "st_blocks", 0),
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def pread_exact(descriptor: int, size: int) -> bytes:
    blocks: list[bytes] = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        blocks.append(block)
        offset += len(block)
    raw = b"".join(blocks)
    if len(raw) != size:
        raise SourceDeployError("held local read is incomplete")
    return raw


def manifest_value() -> dict[str, Any]:
    if len(SOURCE_SPECS) != 6 or len({row[0] for row in SOURCE_SPECS}) != 6:
        raise SourceDeployError("six-file source closure differs")
    return {
        "schema_version": f"{SCHEMA}-manifest",
        "source_root": str(REMOTE_SOURCE_ROOT),
        "stage_parent": str(REMOTE_STAGE_PARENT),
        "python": {
            "mode": 0o755,
            "path": str(REMOTE_PYTHON),
            "sha256": REMOTE_PYTHON_SHA256,
            "size": REMOTE_PYTHON_SIZE,
        },
        "files": [
            {"mode": 0o644, "relative": relative, "sha256": digest, "size": size}
            for relative, digest, size in SOURCE_SPECS
        ],
    }


def stage_root(manifest_sha256: str) -> Path:
    return REMOTE_STAGE_PARENT / (
        "bernini_full644_exploratory_matched_r5d_source_deploy_"
        f"{manifest_sha256[:20]}_r1"
    )


REMOTE_BODY = r'''
import base64
import ctypes
import errno
import hashlib
import json
import os
import stat
import sys

class Refusal(RuntimeError):
    pass

def fail(message):
    raise Refusal(message)

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def objsha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def pairs(items):
    result={}
    for key,value in items:
        if key in result:
            fail("duplicate JSON key")
        result[key]=value
    return result

def ident(info):
    return (info.st_dev,info.st_ino,info.st_uid,info.st_gid,info.st_mode,info.st_nlink,info.st_rdev,info.st_size,getattr(info,"st_blocks",0),info.st_mtime_ns,info.st_ctime_ns)

def pread(descriptor,size):
    blocks=[]
    offset=0
    while offset<size:
        block=os.pread(descriptor,min(1048576,size-offset),offset)
        if not block:
            break
        blocks.append(block)
        offset+=len(block)
    raw=b"".join(blocks)
    if len(raw)!=size:
        fail("held read is incomplete")
    return raw

def open_plain_directory(path,expected_mode):
    if not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path:
        fail("authority directory path differs: "+path)
    named=os.lstat(path)
    descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0))
    opened=os.fstat(descriptor)
    if ident(named)!=ident(opened) or not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(opened.st_mode) or opened.st_uid!=EXPECTED_UID or opened.st_gid!=EXPECTED_GID or stat.S_IMODE(opened.st_mode)!=expected_mode:
        fail("authority directory identity differs: "+path)
    return descriptor

def open_child_directory(parent_fd,name,expected_mode):
    named=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    descriptor=os.open(name,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0),dir_fd=parent_fd)
    opened=os.fstat(descriptor)
    if ident(named)!=ident(opened) or not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(opened.st_mode) or opened.st_uid!=EXPECTED_UID or opened.st_gid!=EXPECTED_GID or stat.S_IMODE(opened.st_mode)!=expected_mode:
        fail("child authority directory differs: "+name)
    return descriptor

def stable_file_at(parent_fd,name,digest,size,mode):
    descriptor=os.open(name,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0),dir_fd=parent_fd)
    try:
        before=os.fstat(descriptor)
        first=pread(descriptor,before.st_size)
        middle=os.fstat(descriptor)
        second=pread(descriptor,before.st_size)
        after=os.fstat(descriptor)
        named=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    finally:
        os.close(descriptor)
    if ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_uid!=EXPECTED_UID or before.st_gid!=EXPECTED_GID or stat.S_IMODE(before.st_mode)!=mode or before.st_nlink!=1 or before.st_size!=size or first!=second or hashlib.sha256(first).hexdigest()!=digest:
        fail("source file identity or bytes differ: "+name)
    return first

def target_state(parent_fd,name,digest,size):
    try:
        os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError:
        return "absent"
    stable_file_at(parent_fd,name,digest,size,0o644)
    return "exact-existing"

def write_stage(stage_fd,name,raw,digest):
    flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0)
    descriptor=os.open(name,flags,0,dir_fd=stage_fd)
    try:
        offset=0
        while offset<len(raw):
            count=os.write(descriptor,raw[offset:])
            if count<=0:
                fail("stage write made no progress")
            offset+=count
        os.fsync(descriptor)
        if pread(descriptor,len(raw))!=raw:
            fail("stage replay differs before commit")
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0 or info.st_uid!=EXPECTED_UID or info.st_gid!=EXPECTED_GID or info.st_nlink!=1 or info.st_size!=len(raw) or hashlib.sha256(pread(descriptor,len(raw))).hexdigest()!=digest:
            fail("stage precommit identity differs")
        os.fchmod(descriptor,0o644)
        os.fsync(descriptor)
        info=os.fstat(descriptor)
        named=os.stat(name,dir_fd=stage_fd,follow_symlinks=False)
        if ident(info)!=ident(named) or stat.S_IMODE(info.st_mode)!=0o644 or info.st_nlink!=1:
            fail("stage commit identity differs")
    finally:
        os.close(descriptor)

def rename_noreplace(source_fd,source_name,target_fd,target_name):
    library=ctypes.CDLL(None,use_errno=True)
    function=getattr(library,"renameat2",None)
    if function is None:
        fail("renameat2(RENAME_NOREPLACE) is unavailable")
    function.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint)
    function.restype=ctypes.c_int
    result=function(source_fd,os.fsencode(source_name),target_fd,os.fsencode(target_name),1)
    if result!=0:
        number=ctypes.get_errno()
        if number in (errno.EEXIST,errno.ENOTEMPTY):
            fail("atomic publication target appeared")
        fail("atomic no-replace publication failed: errno="+str(number))

def write_receipt(stage_fd,value):
    raw=canonical(value)+b"\n"
    digest=hashlib.sha256(raw).hexdigest()
    name="deployment-receipt.json"
    descriptor=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0),0,dir_fd=stage_fd)
    try:
        offset=0
        while offset<len(raw):
            count=os.write(descriptor,raw[offset:])
            if count<=0:
                fail("receipt write made no progress")
            offset+=count
        os.fsync(descriptor)
        if pread(descriptor,len(raw))!=raw:
            fail("receipt replay differs")
        info=os.fstat(descriptor)
        if stat.S_IMODE(info.st_mode)!=0 or info.st_nlink!=1:
            fail("receipt precommit identity differs")
        os.fchmod(descriptor,0o400)
        os.fsync(descriptor)
        info=os.fstat(descriptor)
        named=os.stat(name,dir_fd=stage_fd,follow_symlinks=False)
        if ident(info)!=ident(named) or stat.S_IMODE(info.st_mode)!=0o400 or info.st_uid!=EXPECTED_UID or info.st_gid!=EXPECTED_GID or info.st_nlink!=1:
            fail("receipt commit identity differs")
    finally:
        os.close(descriptor)
    os.fsync(stage_fd)
    return digest

if sys.platform!="linux" or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode or sys.argv!=["-c"]:
    fail("isolated remote entry differs")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"):
    fail("remote environment differs")
if os.geteuid()!=EXPECTED_UID or os.getegid()!=EXPECTED_GID or "torch" in sys.modules or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
    fail("remote process authority differs")

python_fd=os.open("/proc/self/exe",os.O_RDONLY|getattr(os,"O_CLOEXEC",0))
try:
    python_info=os.fstat(python_fd)
    python_raw=pread(python_fd,python_info.st_size)
    python_again=pread(python_fd,python_info.st_size)
    python_named=os.lstat(EXPECTED_PYTHON)
    if ident(python_info)!=ident(os.fstat(python_fd)) or ident(python_info)!=ident(python_named) or not stat.S_ISREG(python_info.st_mode) or python_info.st_uid!=EXPECTED_UID or python_info.st_gid!=EXPECTED_GID or stat.S_IMODE(python_info.st_mode)!=0o755 or python_info.st_nlink!=1 or python_info.st_size!=EXPECTED_PYTHON_SIZE or python_raw!=python_again or hashlib.sha256(python_raw).hexdigest()!=EXPECTED_PYTHON_SHA256:
        fail("remote Python identity differs")
finally:
    os.close(python_fd)

payload_raw=sys.stdin.buffer.read()
if not payload_raw.endswith(b"\n") or payload_raw.count(b"\n")!=1:
    fail("payload framing differs")
try:
    payload=json.loads(payload_raw[:-1].decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
except (UnicodeError,ValueError,TypeError) as error:
    raise Refusal("payload is not strict JSON") from error
if canonical(payload)+b"\n"!=payload_raw or not isinstance(payload,dict) or set(payload)!={"schema_version","manifest","manifest_sha256","remote_bootstrap_sha256","files","authority_sha256"}:
    fail("payload field closure differs")
unsigned=dict(payload)
claimed=unsigned.pop("authority_sha256")
if claimed!=objsha(unsigned) or payload["schema_version"]!=EXPECTED_PAYLOAD_SCHEMA or payload["manifest"]!=EXPECTED_MANIFEST or payload["manifest_sha256"]!=EXPECTED_MANIFEST_SHA256 or payload["remote_bootstrap_sha256"]!=EXPECTED_REMOTE_BOOTSTRAP_SHA256:
    fail("payload authority differs")
files=payload["files"]
if not isinstance(files,list) or len(files)!=len(EXPECTED_SPECS):
    fail("payload file count differs")
captured={}
for row,spec in zip(files,EXPECTED_SPECS):
    relative,digest,size=spec
    if not isinstance(row,dict) or set(row)!={"relative","sha256","size","mode","content_b64"} or row["relative"]!=relative or row["sha256"]!=digest or row["size"]!=size or row["mode"]!=0o644:
        fail("payload source spec differs")
    try:
        raw=base64.b64decode(row["content_b64"],validate=True)
    except Exception as error:
        raise Refusal("payload base64 differs") from error
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:
        fail("payload source bytes differ")
    captured[relative]=raw

root_fd=open_plain_directory(EXPECTED_SOURCE_ROOT,0o755)
methods_fd=open_child_directory(root_fd,"methods",0o755)
bernini_fd=open_child_directory(methods_fd,"bernini_action_editing",0o755)
tools_fd=open_child_directory(bernini_fd,"tools",0o755)
stage_parent_fd=open_plain_directory(EXPECTED_STAGE_PARENT,0o755)
descriptors=(root_fd,methods_fd,bernini_fd,tools_fd,stage_parent_fd)
try:
    if len({os.fstat(fd).st_dev for fd in (bernini_fd,tools_fd,stage_parent_fd)})!=1:
        fail("stage and target filesystems differ")
    targets=[]
    initial=[]
    for relative,digest,size in EXPECTED_SPECS:
        prefix="methods/bernini_action_editing/tools/"
        if relative.startswith(prefix):
            parent_fd=tools_fd
            name=relative[len(prefix):]
        else:
            prefix="methods/bernini_action_editing/"
            if not relative.startswith(prefix):
                fail("target relative path differs")
            parent_fd=bernini_fd
            name=relative[len(prefix):]
        if not name or "/" in name or name in (".",".."):
            fail("target basename differs")
        state=target_state(parent_fd,name,digest,size)
        targets.append((relative,digest,size,parent_fd,name,state))
        initial.append({"relative":relative,"state":state})
    if all(row[-1]=="exact-existing" for row in targets):
        result={"schema_version":EXPECTED_RECEIPT_SCHEMA,"manifest_sha256":EXPECTED_MANIFEST_SHA256,"payload_authority_sha256":claimed,"stage_root":None,"initial":initial,"published":[],"verified_existing":[row[0] for row in targets]}
        sys.stdout.buffer.write(canonical(result)+b"\n")
        sys.stdout.buffer.flush()
        os._exit(0)
    stage_name=os.path.basename(EXPECTED_STAGE_ROOT)
    if os.path.dirname(EXPECTED_STAGE_ROOT)!=EXPECTED_STAGE_PARENT or not stage_name or "/" in stage_name:
        fail("stage root binding differs")
    try:
        os.stat(stage_name,dir_fd=stage_parent_fd,follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        fail("fresh stage root already exists")
    old_umask=os.umask(0o077)
    try:
        os.mkdir(stage_name,0o700,dir_fd=stage_parent_fd)
    finally:
        os.umask(old_umask)
    os.fsync(stage_parent_fd)
    stage_fd=open_child_directory(stage_parent_fd,stage_name,0o700)
    try:
        stage_names={}
        for index,(relative,digest,size,parent_fd,name,state) in enumerate(targets):
            if state=="exact-existing":
                continue
            stage_file=f"{index:02d}-{digest[:16]}.stage"
            write_stage(stage_fd,stage_file,captured[relative],digest)
            stage_names[relative]=stage_file
        os.fsync(stage_fd)
        for relative,digest,size,parent_fd,name,state in targets:
            replay=target_state(parent_fd,name,digest,size)
            if replay!=state:
                fail("target state changed before publication")
            if state=="absent":
                stable_file_at(stage_fd,stage_names[relative],digest,size,0o644)
        published=[]
        for relative,digest,size,parent_fd,name,state in targets:
            if state=="exact-existing":
                continue
            rename_noreplace(stage_fd,stage_names[relative],parent_fd,name)
            os.fsync(parent_fd)
            stable_file_at(parent_fd,name,digest,size,0o644)
            try:
                os.stat(stage_names[relative],dir_fd=stage_fd,follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                fail("stage source remains after publication")
            published.append(relative)
        for relative,digest,size,parent_fd,name,state in targets:
            stable_file_at(parent_fd,name,digest,size,0o644)
        receipt={"schema_version":EXPECTED_RECEIPT_SCHEMA,"manifest_sha256":EXPECTED_MANIFEST_SHA256,"payload_authority_sha256":claimed,"stage_root":EXPECTED_STAGE_ROOT,"initial":initial,"published":published,"verified_existing":[row[0] for row in targets if row[-1]=="exact-existing"]}
        receipt_sha256=write_receipt(stage_fd,receipt)
        output=dict(receipt)
        output["receipt_sha256"]=receipt_sha256
        sys.stdout.buffer.write(canonical(output)+b"\n")
        sys.stdout.buffer.flush()
        os._exit(0)
    finally:
        os.close(stage_fd)
finally:
    for descriptor in reversed(descriptors):
        os.close(descriptor)
'''


def remote_source(manifest: dict[str, Any], manifest_sha256: str) -> str:
    stage = str(stage_root(manifest_sha256))
    header = "\n".join(
        (
            f"EXPECTED_SOURCE_ROOT={str(REMOTE_SOURCE_ROOT)!r}",
            f"EXPECTED_STAGE_PARENT={str(REMOTE_STAGE_PARENT)!r}",
            f"EXPECTED_STAGE_ROOT={stage!r}",
            f"EXPECTED_PYTHON={str(REMOTE_PYTHON)!r}",
            f"EXPECTED_PYTHON_SHA256={REMOTE_PYTHON_SHA256!r}",
            f"EXPECTED_PYTHON_SIZE={REMOTE_PYTHON_SIZE!r}",
            f"EXPECTED_UID={REMOTE_UID!r}",
            f"EXPECTED_GID={REMOTE_GID!r}",
            f"EXPECTED_SPECS={SOURCE_SPECS!r}",
            f"EXPECTED_MANIFEST={manifest!r}",
            f"EXPECTED_MANIFEST_SHA256={manifest_sha256!r}",
            f"EXPECTED_PAYLOAD_SCHEMA={PAYLOAD_SCHEMA!r}",
            f"EXPECTED_RECEIPT_SCHEMA={REMOTE_RECEIPT_SCHEMA!r}",
            "EXPECTED_REMOTE_BOOTSTRAP_SHA256="
            f"{hashlib.sha256(REMOTE_BODY.encode('utf-8')).hexdigest()!r}",
        )
    )
    return header + "\n" + REMOTE_BODY


def remote_bootstrap_digest(source: str) -> str:
    del source
    # This non-self-referential digest authenticates the complete generic
    # remote program body.  The local audit separately binds the final header
    # plus body bytes actually passed to ``python -c``.
    return hashlib.sha256(REMOTE_BODY.encode("utf-8")).hexdigest()


def capture_sources() -> tuple[list[dict[str, Any]], list[tuple[int, Path, tuple[int, ...]]]]:
    files: list[dict[str, Any]] = []
    held: list[tuple[int, Path, tuple[int, ...]]] = []
    try:
        for relative, digest, size in SOURCE_SPECS:
            path = LOCAL_ROOT / relative
            if (
                not path.is_absolute()
                or os.path.normpath(str(path)) != str(path)
                or path.is_symlink()
            ):
                raise SourceDeployError(f"local source path differs: {relative}")
            named = os.lstat(path)
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            opened = os.fstat(descriptor)
            raw = pread_exact(descriptor, opened.st_size)
            middle = os.fstat(descriptor)
            again = pread_exact(descriptor, opened.st_size)
            after = os.fstat(descriptor)
            if (
                identity(named) != identity(opened)
                or identity(opened) != identity(middle)
                or identity(opened) != identity(after)
                or not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o644
                or opened.st_nlink != 1
                or opened.st_size != size
                or raw != again
                or hashlib.sha256(raw).hexdigest() != digest
            ):
                os.close(descriptor)
                raise SourceDeployError(f"local source identity or bytes differ: {relative}")
            files.append(
                {
                    "content_b64": base64.b64encode(raw).decode("ascii"),
                    "mode": 0o644,
                    "relative": relative,
                    "sha256": digest,
                    "size": size,
                }
            )
            held.append((descriptor, path, identity(opened)))
    except BaseException:
        for descriptor, _, _ in held:
            os.close(descriptor)
        raise
    return files, held


def replay_held(held: Sequence[tuple[int, Path, tuple[int, ...]]]) -> None:
    for descriptor, path, expected in held:
        if identity(os.fstat(descriptor)) != expected or identity(os.lstat(path)) != expected:
            raise SourceDeployError(f"held local source changed: {path}")


def build_payload(files: list[dict[str, Any]]) -> tuple[bytes, dict[str, Any], str, str]:
    manifest = manifest_value()
    manifest_sha256 = object_sha256(manifest)
    source = remote_source(manifest, manifest_sha256)
    bootstrap_sha256 = remote_bootstrap_digest(source)
    unsigned = {
        "schema_version": PAYLOAD_SCHEMA,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "remote_bootstrap_sha256": bootstrap_sha256,
        "files": files,
    }
    payload = dict(unsigned)
    payload["authority_sha256"] = object_sha256(unsigned)
    return canonical_json_bytes(payload) + b"\n", payload, source, bootstrap_sha256


def authorization_token(manifest_sha256: str, bootstrap_sha256: str) -> str:
    return hashlib.sha256(
        b"r5d-source-deploy-explicit-authorization-v1\0"
        + manifest_sha256.encode("ascii")
        + b"\0"
        + bootstrap_sha256.encode("ascii")
    ).hexdigest()


def audit_value(
    payload: dict[str, Any], bootstrap_sha256: str, source: str
) -> dict[str, Any]:
    manifest_sha256 = payload["manifest_sha256"]
    return {
        "schema_version": f"{SCHEMA}-local-audit",
        "authorization_token": authorization_token(manifest_sha256, bootstrap_sha256),
        "file_count": len(payload["files"]),
        "files": [
            {key: row[key] for key in ("relative", "sha256", "size", "mode")}
            for row in payload["files"]
        ],
        "manifest_sha256": manifest_sha256,
        "payload_authority_sha256": payload["authority_sha256"],
        "remote_bootstrap_sha256": bootstrap_sha256,
        "remote_executed_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "remote_source_root": str(REMOTE_SOURCE_ROOT),
        "remote_stage_root": str(stage_root(manifest_sha256)),
    }


def execute_remote(payload_raw: bytes, source: str) -> dict[str, Any]:
    inner = (
        "exec -c "
        + shlex.quote(str(REMOTE_PYTHON))
        + " -I -S -B -c "
        + shlex.quote(source)
    )
    remote_command = (
        "/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C "
        "HOME=/vast/users/guangyi.chen BASH_ENV=/dev/null "
        "/bin/bash -p -c "
        + shlex.quote(inner)
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "auh", remote_command],
        input=payload_raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr != b"":
        raise SourceDeployError(
            "remote deployment failed closed: "
            f"returncode={completed.returncode}, stderr={completed.stderr!r}"
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8", "strict"))
    except (UnicodeError, ValueError, TypeError) as error:
        raise SourceDeployError("remote receipt is not JSON") from error
    if canonical_json_bytes(value) + b"\n" != completed.stdout:
        raise SourceDeployError("remote receipt is not canonical JSON")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-local", action="store_true")
    group.add_argument("--execute", metavar="AUTHORIZATION_TOKEN")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    files, held = capture_sources()
    try:
        payload_raw, payload, source, bootstrap_sha256 = build_payload(files)
        audit = audit_value(payload, bootstrap_sha256, source)
        replay_held(held)
        if args.audit_local:
            sys.stdout.buffer.write(canonical_json_bytes(audit) + b"\n")
            return 0
        if args.execute != audit["authorization_token"]:
            raise SourceDeployError("explicit authorization token differs")
        result = execute_remote(payload_raw, source)
        replay_held(held)
        if (
            result.get("schema_version") != REMOTE_RECEIPT_SCHEMA
            or result.get("manifest_sha256") != audit["manifest_sha256"]
            or result.get("payload_authority_sha256")
            != audit["payload_authority_sha256"]
        ):
            raise SourceDeployError("remote receipt binding differs")
        expected_relatives = [row[0] for row in SOURCE_SPECS]
        published = result.get("published")
        existing = result.get("verified_existing")
        initial = result.get("initial")
        if (
            not isinstance(published, list)
            or not isinstance(existing, list)
            or not isinstance(initial, list)
            or len(initial) != 6
            or [row.get("relative") for row in initial] != expected_relatives
            or any(
                not isinstance(row, dict)
                or set(row) != {"relative", "state"}
                or row.get("state") not in {"absent", "exact-existing"}
                for row in initial
            )
            or published
            != [row["relative"] for row in initial if row["state"] == "absent"]
            or existing
            != [
                row["relative"]
                for row in initial
                if row["state"] == "exact-existing"
            ]
            or set(published) | set(existing) != set(expected_relatives)
            or set(published) & set(existing) != set()
            or len(published) + len(existing) != len(expected_relatives)
            or result.get("stage_root")
            != (str(stage_root(audit["manifest_sha256"])) if published else None)
            or set(result)
            != (
                {
                    "schema_version",
                    "manifest_sha256",
                    "payload_authority_sha256",
                    "stage_root",
                    "initial",
                    "published",
                    "verified_existing",
                    "receipt_sha256",
                }
                if published
                else {
                    "schema_version",
                    "manifest_sha256",
                    "payload_authority_sha256",
                    "stage_root",
                    "initial",
                    "published",
                    "verified_existing",
                }
            )
            or (
                published
                and (
                    not isinstance(result.get("receipt_sha256"), str)
                    or len(result["receipt_sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in result["receipt_sha256"])
                )
            )
        ):
            raise SourceDeployError("remote receipt field closure differs")
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    finally:
        for descriptor, _, _ in held:
            os.close(descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceDeployError as error:
        print(f"r5d source deployment refused: {error}", file=sys.stderr)
        raise SystemExit(96) from error
