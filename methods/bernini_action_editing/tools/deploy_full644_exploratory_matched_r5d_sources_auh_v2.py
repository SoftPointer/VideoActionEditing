#!/usr/bin/env python3
"""Captured-entry, create-only deployment of the six frozen r5d sources.

This file is not a valid named entry point.  An isolated root-owned local
Python must capture its exact bytes through a held descriptor, then execute
those captured bytes while injecting the five ``__R5D_CAPTURED_*`` globals.
The controller is otherwise inert.  Its token binds that captured controller,
the full remote ``python -c`` source, the root-owned remote bootstrap, the
six-file manifest, and the exact effective SSH configuration.
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


SCHEMA = "full644-exploratory-matched-r5d-source-deploy-auh-v2"
PAYLOAD_SCHEMA = f"{SCHEMA}-payload"
REMOTE_RECEIPT_SCHEMA = f"{SCHEMA}-receipt"
REMOTE_RECEIPT_ENVELOPE_SCHEMA = f"{SCHEMA}-receipt-envelope"
LOCAL_ROOT = Path("/Users/siriuschu/ML/VideoEditing/VideoEdit")
CONTROLLER_PATH = LOCAL_ROOT / (
    "methods/bernini_action_editing/tools/"
    "deploy_full644_exploratory_matched_r5d_sources_auh_v2.py"
)
LOCAL_PYTHON = Path(
    "/Library/Developer/CommandLineTools/Library/Frameworks/"
    "Python3.framework/Versions/3.9/bin/python3.9"
)
LOCAL_PYTHON_SHA256 = "d23458804881b5c23d3aacae44311b9c43f961c4eba3a23163572aaebf58f44f"
LOCAL_PYTHON_SIZE = 102_352
LOCAL_PYTHON_NLINK = 1
LOCAL_UID = 501
LOCAL_GID = 20
LOCAL_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "HOME": "/Users/siriuschu",
    "TMPDIR": "/private/tmp",
    "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
}
SSH_PATH = Path("/usr/bin/ssh")
SSH_SHA256 = "75ae4b414b57e0c52ad1cb24a9d7dae2496071fdf153c7fc8e94db3c9c4b0faa"
SSH_SIZE = 1_474_128
SSH_IDENTITY = Path("/Users/siriuschu/.ssh/ciai2/id_ed25519")
SSH_IDENTITY_SHA256 = "b41c88847bd284896de55df0231c5c6fced1d1b32a4a3aca6e8682b5eaaf8651"
SSH_IDENTITY_SIZE = 419
SSH_KNOWN_HOSTS = Path("/Users/siriuschu/.ssh/known_hosts")
SSH_KNOWN_HOSTS_SHA256 = "3337d55aea085faada7937b20aa4cd12a908c13f1c4142704832bba46145bbaa"
SSH_KNOWN_HOSTS_SIZE = 18_620
SSH_CONFIG_SHA256 = "b6bb87712da224d42f449f86a81ee129d4cabea4043b1fed3fb4d8e829d4e4ee"
SSH_CONFIG_SIZE = 3_931
SSH_ARGUMENTS = (
    str(SSH_PATH),
    "-F", "/dev/null", "-T",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", f"IdentityFile={SSH_IDENTITY}",
    "-o", f"UserKnownHostsFile={SSH_KNOWN_HOSTS}",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UpdateHostKeys=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "RequestTTY=no",
    "-o", "KbdInteractiveAuthentication=no",
    "-o", "PasswordAuthentication=no",
    "-o", "PreferredAuthentications=publickey",
    "-o", "PubkeyAuthentication=yes",
    "-o", "GSSAPIAuthentication=no",
    "-o", "HostbasedAuthentication=no",
    "-o", "CanonicalizeHostname=no",
    "-o", "CheckHostIP=no",
    "-o", "VerifyHostKeyDNS=no",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ControlPersist=no",
    "-o", "ForwardAgent=no",
    "-o", "ForwardX11=no",
    "-o", "ForwardX11Trusted=no",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ConnectTimeout=30",
    "-o", "ConnectionAttempts=1",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "-o", "LogLevel=ERROR",
    "-p", "22",
    "guangyi.chen@172.27.112.248",
)
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
REMOTE_SYSTEM_PYTHON = Path("/usr/bin/python3.10")
REMOTE_SYSTEM_PYTHON_SHA256 = (
    "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
)
REMOTE_SYSTEM_PYTHON_SIZE = 5_937_800
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


def _stable_local_file(
    path: Path,
    *,
    sha256: str,
    size: int,
    uid: int,
    gid: int,
    mode: int,
    nlink: int,
) -> tuple[int, tuple[int, ...]]:
    if (
        not path.is_absolute()
        or os.path.normpath(str(path)) != str(path)
        or path.is_symlink()
    ):
        raise SourceDeployError(f"local authority path differs: {path}")
    named = os.lstat(path)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    opened = os.fstat(descriptor)
    first = pread_exact(descriptor, opened.st_size)
    middle = os.fstat(descriptor)
    second = pread_exact(descriptor, opened.st_size)
    after = os.fstat(descriptor)
    if (
        identity(named) != identity(opened)
        or identity(opened) != identity(middle)
        or identity(opened) != identity(after)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != uid
        or opened.st_gid != gid
        or stat.S_IMODE(opened.st_mode) != mode
        or opened.st_nlink != nlink
        or opened.st_size != size
        or first != second
        or hashlib.sha256(first).hexdigest() != sha256
    ):
        os.close(descriptor)
        raise SourceDeployError(f"local authority identity or bytes differ: {path}")
    return descriptor, identity(opened)


def validate_captured_execution() -> tuple[int, tuple[int, ...], str, str]:
    raw = globals().get("__R5D_CAPTURED_SOURCE_RAW")
    claimed = globals().get("__R5D_CAPTURED_SOURCE_SHA256")
    path = globals().get("__R5D_CAPTURED_SOURCE_PATH")
    descriptor = globals().get("__R5D_CAPTURED_SOURCE_FD")
    entry_sha256 = globals().get("__R5D_CAPTURED_ENTRY_SHA256")
    if (
        not isinstance(raw, bytes)
        or not isinstance(claimed, str)
        or len(claimed) != 64
        or hashlib.sha256(raw).hexdigest() != claimed
        or path != str(CONTROLLER_PATH)
        or type(descriptor) is not int
        or descriptor < 3
        or not isinstance(entry_sha256, str)
        or len(entry_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claimed + entry_sha256)
    ):
        raise SourceDeployError("captured controller entry authority differs")
    if (
        sys.platform != "darwin"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or not sys.dont_write_bytecode
        or os.environ != LOCAL_ENV
        or os.geteuid() != LOCAL_UID
        or os.getegid() != LOCAL_GID
        or sys.executable != str(LOCAL_PYTHON)
    ):
        raise SourceDeployError("isolated local entry differs")
    python_fd, _ = _stable_local_file(
        LOCAL_PYTHON,
        sha256=LOCAL_PYTHON_SHA256,
        size=LOCAL_PYTHON_SIZE,
        uid=0,
        gid=0,
        mode=0o755,
        nlink=LOCAL_PYTHON_NLINK,
    )
    os.close(python_fd)
    opened = os.fstat(descriptor)
    first = pread_exact(descriptor, opened.st_size)
    middle = os.fstat(descriptor)
    second = pread_exact(descriptor, opened.st_size)
    named = os.lstat(CONTROLLER_PATH)
    if (
        identity(opened) != identity(middle)
        or identity(opened) != identity(named)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != LOCAL_UID
        or opened.st_gid != LOCAL_GID
        or stat.S_IMODE(opened.st_mode) != 0o644
        or opened.st_nlink != 1
        or opened.st_size != len(raw)
        or first != raw
        or second != raw
    ):
        raise SourceDeployError("held controller replay differs")
    compile(raw.decode("utf-8", "strict"), str(CONTROLLER_PATH), "exec", dont_inherit=True)
    return descriptor, identity(opened), claimed, entry_sha256


def replay_captured_execution(
    descriptor: int, expected: tuple[int, ...], claimed: str
) -> None:
    opened = os.fstat(descriptor)
    if (
        identity(opened) != expected
        or identity(os.lstat(CONTROLLER_PATH)) != expected
        or hashlib.sha256(pread_exact(descriptor, opened.st_size)).hexdigest() != claimed
    ):
        raise SourceDeployError("captured controller changed")


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
        "system_bootstrap_python": {
            "mode": 0o755,
            "path": str(REMOTE_SYSTEM_PYTHON),
            "sha256": REMOTE_SYSTEM_PYTHON_SHA256,
            "size": REMOTE_SYSTEM_PYTHON_SIZE,
        },
        "files": [
            {"mode": 0o644, "relative": relative, "sha256": digest, "size": size}
            for relative, digest, size in SOURCE_SPECS
        ],
    }


def stage_root(manifest_sha256: str) -> Path:
    return REMOTE_STAGE_PARENT / (
        "bernini_full644_exploratory_matched_r5d_source_deploy_"
        f"{manifest_sha256[:20]}_r2"
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

def replay_directory(path,descriptor,expected_mode):
    if not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path:
        fail("directory replay path differs: "+path)
    named=os.lstat(path)
    opened=os.fstat(descriptor)
    if ident(named)!=ident(opened) or not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(opened.st_mode) or opened.st_uid!=EXPECTED_UID or opened.st_gid!=EXPECTED_GID or stat.S_IMODE(opened.st_mode)!=expected_mode:
        fail("directory replay identity differs: "+path)

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

def require_absent_at(parent_fd,name):
    try:
        os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError:
        return
    fail("target must be absent before deployment: "+name)

def stable_target_absolute(path,parent_fd,name,digest,size):
    if not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path:
        fail("target absolute path differs: "+path)
    raw=stable_file_at(parent_fd,name,digest,size,0o644)
    absolute=os.lstat(path)
    relative=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    if ident(absolute)!=ident(relative):
        fail("target absolute identity differs: "+path)
    return raw,absolute

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
    committed=False
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
        committed=True
        os.fsync(stage_fd)
        return descriptor,raw,digest,ident(info)
    finally:
        if not committed:
            os.close(descriptor)

if sys.platform!="linux" or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode or len(sys.argv)!=3 or sys.argv[0]!="-c" or sys.argv[1]!="--source-sha256" or not isinstance(sys.argv[2],str) or len(sys.argv[2])!=64 or any(character not in "0123456789abcdef" for character in sys.argv[2]):
    fail("isolated remote entry differs")
executed_source_sha256=sys.argv[2]
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
if claimed!=objsha(unsigned) or payload["schema_version"]!=EXPECTED_PAYLOAD_SCHEMA or payload["manifest"]!=EXPECTED_MANIFEST or payload["manifest_sha256"]!=EXPECTED_MANIFEST_SHA256 or payload["remote_bootstrap_sha256"]!=executed_source_sha256:
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
        absolute=EXPECTED_SOURCE_ROOT+"/"+relative
        require_absent_at(parent_fd,name)
        targets.append((relative,digest,size,parent_fd,name,absolute))
    replay_directory(EXPECTED_SOURCE_ROOT,root_fd,0o755)
    replay_directory(EXPECTED_SOURCE_ROOT+"/methods",methods_fd,0o755)
    replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing",bernini_fd,0o755)
    replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing/tools",tools_fd,0o755)
    replay_directory(EXPECTED_STAGE_PARENT,stage_parent_fd,0o755)
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
        replay_directory(EXPECTED_STAGE_PARENT,stage_parent_fd,0o755)
        replay_directory(EXPECTED_STAGE_ROOT,stage_fd,0o700)
        stage_names={}
        for index,(relative,digest,size,parent_fd,name,absolute) in enumerate(targets):
            stage_file=f"{index:02d}-{digest[:16]}.stage"
            write_stage(stage_fd,stage_file,captured[relative],digest)
            stage_names[relative]=stage_file
        os.fsync(stage_fd)
        for relative,digest,size,parent_fd,name,absolute in targets:
            require_absent_at(parent_fd,name)
            stable_file_at(stage_fd,stage_names[relative],digest,size,0o644)
        replay_directory(EXPECTED_SOURCE_ROOT,root_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods",methods_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing",bernini_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing/tools",tools_fd,0o755)
        replay_directory(EXPECTED_STAGE_PARENT,stage_parent_fd,0o755)
        replay_directory(EXPECTED_STAGE_ROOT,stage_fd,0o700)
        published=[]
        for relative,digest,size,parent_fd,name,absolute in targets:
            rename_noreplace(stage_fd,stage_names[relative],parent_fd,name)
            os.fsync(parent_fd)
            stable_target_absolute(absolute,parent_fd,name,digest,size)
            try:
                os.stat(stage_names[relative],dir_fd=stage_fd,follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                fail("stage source remains after publication")
            published.append(relative)
        if published!=[row[0] for row in EXPECTED_SPECS]:
            fail("published source closure differs")
        replay_directory(EXPECTED_SOURCE_ROOT,root_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods",methods_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing",bernini_fd,0o755)
        replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing/tools",tools_fd,0o755)
        replay_directory(EXPECTED_STAGE_PARENT,stage_parent_fd,0o755)
        replay_directory(EXPECTED_STAGE_ROOT,stage_fd,0o700)
        final=[]
        for relative,digest,size,parent_fd,name,absolute in targets:
            raw,info=stable_target_absolute(absolute,parent_fd,name,digest,size)
            final.append({"absolute_path":absolute,"gid":info.st_gid,"mode":stat.S_IMODE(info.st_mode),"nlink":info.st_nlink,"relative":relative,"sha256":hashlib.sha256(raw).hexdigest(),"size":info.st_size,"uid":info.st_uid})
        receipt={"schema_version":EXPECTED_RECEIPT_SCHEMA,"manifest_sha256":EXPECTED_MANIFEST_SHA256,"payload_authority_sha256":claimed,"remote_executed_source_sha256":executed_source_sha256,"stage_root":EXPECTED_STAGE_ROOT,"published":published,"final_files":final}
        receipt_fd,receipt_raw,receipt_sha256,receipt_identity=write_receipt(stage_fd,receipt)
        try:
            replay_directory(EXPECTED_SOURCE_ROOT,root_fd,0o755)
            replay_directory(EXPECTED_SOURCE_ROOT+"/methods",methods_fd,0o755)
            replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing",bernini_fd,0o755)
            replay_directory(EXPECTED_SOURCE_ROOT+"/methods/bernini_action_editing/tools",tools_fd,0o755)
            replay_directory(EXPECTED_STAGE_PARENT,stage_parent_fd,0o755)
            replay_directory(EXPECTED_STAGE_ROOT,stage_fd,0o700)
            for relative,digest,size,parent_fd,name,absolute in targets:
                stable_target_absolute(absolute,parent_fd,name,digest,size)
            held_info=os.fstat(receipt_fd)
            named_info=os.stat("deployment-receipt.json",dir_fd=stage_fd,follow_symlinks=False)
            held_raw=pread(receipt_fd,held_info.st_size)
            if ident(held_info)!=receipt_identity or ident(held_info)!=ident(named_info) or held_raw!=receipt_raw or hashlib.sha256(held_raw).hexdigest()!=receipt_sha256 or held_info.st_size!=len(receipt_raw) or stat.S_IMODE(held_info.st_mode)!=0o400 or held_info.st_nlink!=1:
                fail("final held receipt replay differs")
        finally:
            os.close(receipt_fd)
        output={"schema_version":EXPECTED_RECEIPT_ENVELOPE_SCHEMA,"receipt":receipt,"receipt_identity":{"gid":named_info.st_gid,"mode":stat.S_IMODE(named_info.st_mode),"nlink":named_info.st_nlink,"size":named_info.st_size,"uid":named_info.st_uid},"receipt_raw_b64":base64.b64encode(receipt_raw).decode("ascii"),"receipt_sha256":receipt_sha256,"receipt_size":len(receipt_raw)}
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
            f"EXPECTED_RECEIPT_ENVELOPE_SCHEMA={REMOTE_RECEIPT_ENVELOPE_SCHEMA!r}",
        )
    )
    return header + "\n" + REMOTE_BODY


def remote_bootstrap_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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


def authorization_token(
    manifest_sha256: str,
    remote_source_sha256: str,
    system_bootstrap_sha256: str,
    controller_sha256: str,
    entry_sha256: str,
) -> str:
    return hashlib.sha256(
        b"r5d-source-deploy-explicit-authorization-v2\0"
        + manifest_sha256.encode("ascii")
        + b"\0"
        + remote_source_sha256.encode("ascii")
        + b"\0"
        + system_bootstrap_sha256.encode("ascii")
        + b"\0"
        + controller_sha256.encode("ascii")
        + b"\0"
        + entry_sha256.encode("ascii")
        + b"\0"
        + SSH_CONFIG_SHA256.encode("ascii")
    ).hexdigest()


def audit_value(
    payload: dict[str, Any],
    remote_source_sha256: str,
    source: str,
    controller_sha256: str,
    entry_sha256: str,
) -> dict[str, Any]:
    manifest_sha256 = payload["manifest_sha256"]
    system_bootstrap_sha256 = hashlib.sha256(
        remote_system_bootstrap_source().encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": f"{SCHEMA}-local-audit",
        "authorization_token": authorization_token(
            manifest_sha256,
            remote_source_sha256,
            system_bootstrap_sha256,
            controller_sha256,
            entry_sha256,
        ),
        "captured_controller_sha256": controller_sha256,
        "captured_entry_sha256": entry_sha256,
        "file_count": len(payload["files"]),
        "files": [
            {key: row[key] for key in ("relative", "sha256", "size", "mode")}
            for row in payload["files"]
        ],
        "manifest_sha256": manifest_sha256,
        "payload_authority_sha256": payload["authority_sha256"],
        "remote_bootstrap_sha256": remote_source_sha256,
        "remote_executed_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "remote_system_bootstrap_sha256": system_bootstrap_sha256,
        "ssh_config_sha256": SSH_CONFIG_SHA256,
        "remote_source_root": str(REMOTE_SOURCE_ROOT),
        "remote_stage_root": str(stage_root(manifest_sha256)),
    }


REMOTE_SYSTEM_BOOTSTRAP = r'''
import hashlib
import os
import stat
import sys

def fail(message):
    raise RuntimeError(message)

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
        fail("bootstrap held read is incomplete")
    return raw

def stream_exact(descriptor,size):
    blocks=[]
    remaining=size
    while remaining:
        block=os.read(descriptor,remaining)
        if not block:
            fail("bootstrap framed input is incomplete")
        blocks.append(block)
        remaining-=len(block)
    return b"".join(blocks)

if sys.platform!="linux" or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode or len(sys.argv)!=5 or sys.argv[0]!="-c":
    fail("system bootstrap entry differs")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"):
    fail("system bootstrap environment differs")
vace_path,vace_sha256,vace_size_raw,source_sha256=sys.argv[1:]
if vace_path!=EXPECTED_VACE_PATH or vace_sha256!=EXPECTED_VACE_SHA256 or vace_size_raw!=str(EXPECTED_VACE_SIZE) or len(source_sha256)!=64 or any(character not in "0123456789abcdef" for character in source_sha256):
    fail("system bootstrap argv differs")

system_fd=os.open("/proc/self/exe",os.O_RDONLY|getattr(os,"O_CLOEXEC",0))
try:
    info=os.fstat(system_fd)
    first=pread(system_fd,info.st_size)
    second=pread(system_fd,info.st_size)
    named=os.lstat(EXPECTED_SYSTEM_PYTHON)
    if ident(info)!=ident(os.fstat(system_fd)) or ident(info)!=ident(named) or not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or stat.S_IMODE(info.st_mode)!=0o755 or info.st_nlink!=1 or info.st_size!=EXPECTED_SYSTEM_PYTHON_SIZE or first!=second or hashlib.sha256(first).hexdigest()!=EXPECTED_SYSTEM_PYTHON_SHA256:
        fail("system bootstrap Python identity differs")
finally:
    os.close(system_fd)

length=int.from_bytes(stream_exact(0,8),"big")
if length<=0 or length>131072:
    fail("remote source frame length differs")
source_raw=stream_exact(0,length)
if hashlib.sha256(source_raw).hexdigest()!=source_sha256:
    fail("remote source frame digest differs")
source=source_raw.decode("utf-8","strict")
compile(source,"<r5d-source-deploy-v2>","exec",dont_inherit=True)

vace_fd=os.open(vace_path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
info=os.fstat(vace_fd)
first=pread(vace_fd,info.st_size)
middle=os.fstat(vace_fd)
second=pread(vace_fd,info.st_size)
named=os.lstat(vace_path)
if ident(info)!=ident(middle) or ident(info)!=ident(named) or not stat.S_ISREG(info.st_mode) or info.st_uid!=EXPECTED_UID or info.st_gid!=EXPECTED_GID or stat.S_IMODE(info.st_mode)!=0o755 or info.st_nlink!=1 or info.st_size!=EXPECTED_VACE_SIZE or first!=second or hashlib.sha256(first).hexdigest()!=vace_sha256:
    fail("held VACE Python identity differs")
os.set_inheritable(vace_fd,True)
if ident(os.fstat(vace_fd))!=ident(os.lstat(vace_path)) or hashlib.sha256(pread(vace_fd,info.st_size)).hexdigest()!=vace_sha256:
    fail("held VACE Python changed before exec")
os.execve("/proc/self/fd/"+str(vace_fd),[vace_path,"-I","-S","-B","-c",source,"--source-sha256",source_sha256],{})
'''


def remote_system_bootstrap_source() -> str:
    header = "\n".join(
        (
            f"EXPECTED_SYSTEM_PYTHON={str(REMOTE_SYSTEM_PYTHON)!r}",
            f"EXPECTED_SYSTEM_PYTHON_SHA256={REMOTE_SYSTEM_PYTHON_SHA256!r}",
            f"EXPECTED_SYSTEM_PYTHON_SIZE={REMOTE_SYSTEM_PYTHON_SIZE!r}",
            f"EXPECTED_VACE_PATH={str(REMOTE_PYTHON)!r}",
            f"EXPECTED_VACE_SHA256={REMOTE_PYTHON_SHA256!r}",
            f"EXPECTED_VACE_SIZE={REMOTE_PYTHON_SIZE!r}",
            f"EXPECTED_UID={REMOTE_UID!r}",
            f"EXPECTED_GID={REMOTE_GID!r}",
        )
    )
    return header + "\n" + REMOTE_SYSTEM_BOOTSTRAP


def _capture_ssh_authority() -> list[tuple[int, Path, tuple[int, ...]]]:
    held: list[tuple[int, Path, tuple[int, ...]]] = []
    specs = (
        (SSH_PATH, SSH_SHA256, SSH_SIZE, 0, 0, 0o755, 1),
        (SSH_IDENTITY, SSH_IDENTITY_SHA256, SSH_IDENTITY_SIZE, LOCAL_UID, LOCAL_GID, 0o600, 1),
        (SSH_KNOWN_HOSTS, SSH_KNOWN_HOSTS_SHA256, SSH_KNOWN_HOSTS_SIZE, LOCAL_UID, LOCAL_GID, 0o600, 1),
    )
    try:
        for path, digest, size, uid, gid, mode, nlink in specs:
            descriptor, expected = _stable_local_file(
                path,
                sha256=digest,
                size=size,
                uid=uid,
                gid=gid,
                mode=mode,
                nlink=nlink,
            )
            held.append((descriptor, path, expected))
        configured = subprocess.run(
            [SSH_ARGUMENTS[0], "-G", *SSH_ARGUMENTS[1:]],
            env=LOCAL_ENV,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if (
            configured.returncode != 0
            or configured.stderr != b""
            or len(configured.stdout) != SSH_CONFIG_SIZE
            or hashlib.sha256(configured.stdout).hexdigest() != SSH_CONFIG_SHA256
        ):
            raise SourceDeployError("effective SSH configuration differs")
        replay_held(held)
        return held
    except BaseException:
        for descriptor, _, _ in held:
            os.close(descriptor)
        raise


def execute_remote(payload_raw: bytes, source: str) -> dict[str, Any]:
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    system_bootstrap = remote_system_bootstrap_source()
    remote_command = (
        "/usr/bin/env -i "
        + shlex.quote(str(REMOTE_SYSTEM_PYTHON))
        + " -I -S -B -c "
        + shlex.quote(system_bootstrap)
        + " " + shlex.quote(str(REMOTE_PYTHON))
        + " " + shlex.quote(REMOTE_PYTHON_SHA256)
        + " " + shlex.quote(str(REMOTE_PYTHON_SIZE))
        + " " + shlex.quote(source_sha256)
    )
    ssh_held = _capture_ssh_authority()
    try:
        completed = subprocess.run(
            [*SSH_ARGUMENTS, remote_command],
            env=LOCAL_ENV,
            input=len(source.encode("utf-8")).to_bytes(8, "big")
            + source.encode("utf-8")
            + payload_raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        replay_held(ssh_held)
    finally:
        for descriptor, _, _ in ssh_held:
            os.close(descriptor)
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
    controller_fd, controller_identity, controller_sha256, entry_sha256 = (
        validate_captured_execution()
    )
    args = parse_args(argv)
    files, held = capture_sources()
    try:
        payload_raw, payload, source, remote_source_sha256 = build_payload(files)
        audit = audit_value(
            payload,
            remote_source_sha256,
            source,
            controller_sha256,
            entry_sha256,
        )
        replay_held(held)
        replay_captured_execution(
            controller_fd, controller_identity, controller_sha256
        )
        ssh_held = _capture_ssh_authority()
        try:
            replay_held(ssh_held)
        finally:
            for descriptor, _, _ in ssh_held:
                os.close(descriptor)
        if args.audit_local:
            sys.stdout.buffer.write(canonical_json_bytes(audit) + b"\n")
            return 0
        if args.execute != audit["authorization_token"]:
            raise SourceDeployError("explicit authorization token differs")
        result = execute_remote(payload_raw, source)
        replay_held(held)
        replay_captured_execution(
            controller_fd, controller_identity, controller_sha256
        )
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "schema_version",
                "receipt",
                "receipt_identity",
                "receipt_raw_b64",
                "receipt_sha256",
                "receipt_size",
            }
            or result.get("schema_version") != REMOTE_RECEIPT_ENVELOPE_SCHEMA
        ):
            raise SourceDeployError("remote receipt binding differs")
        expected_relatives = [row[0] for row in SOURCE_SPECS]
        receipt = result.get("receipt")
        identity_value = result.get("receipt_identity")
        try:
            receipt_raw = base64.b64decode(result.get("receipt_raw_b64"), validate=True)
        except (TypeError, ValueError) as error:
            raise SourceDeployError("remote held receipt base64 differs") from error
        if (
            not isinstance(receipt, dict)
            or set(receipt)
            != {
                "schema_version",
                "manifest_sha256",
                "payload_authority_sha256",
                "remote_executed_source_sha256",
                "stage_root",
                "published",
                "final_files",
            }
            or receipt.get("schema_version") != REMOTE_RECEIPT_SCHEMA
            or receipt.get("manifest_sha256") != audit["manifest_sha256"]
            or receipt.get("payload_authority_sha256")
            != audit["payload_authority_sha256"]
            or receipt.get("remote_executed_source_sha256")
            != audit["remote_executed_source_sha256"]
            or receipt.get("stage_root")
            != str(stage_root(audit["manifest_sha256"]))
            or receipt.get("published") != expected_relatives
            or canonical_json_bytes(receipt) + b"\n" != receipt_raw
            or result.get("receipt_size") != len(receipt_raw)
            or not isinstance(result.get("receipt_sha256"), str)
            or hashlib.sha256(receipt_raw).hexdigest()
            != result.get("receipt_sha256")
            or identity_value
            != {
                "gid": REMOTE_GID,
                "mode": 0o400,
                "nlink": 1,
                "size": len(receipt_raw),
                "uid": REMOTE_UID,
            }
        ):
            raise SourceDeployError("remote receipt field closure differs")
        expected_final = [
            {
                "absolute_path": str(REMOTE_SOURCE_ROOT / relative),
                "gid": REMOTE_GID,
                "mode": 0o644,
                "nlink": 1,
                "relative": relative,
                "sha256": digest,
                "size": size,
                "uid": REMOTE_UID,
            }
            for relative, digest, size in SOURCE_SPECS
        ]
        if receipt.get("final_files") != expected_final:
            raise SourceDeployError("remote final source closure differs")
        sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
        return 0
    finally:
        for descriptor, _, _ in held:
            os.close(descriptor)
        os.close(controller_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceDeployError as error:
        print(f"r5d source deployment refused: {error}", file=sys.stderr)
        raise SystemExit(96) from error
