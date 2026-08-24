#!/bin/bash -p
set -euo pipefail
umask 077

[[ "$-" == *p* ]] || { printf '%s\n' 'controller must be streamed to /bin/bash -p -s' >&2; exit 90; }
[[ "$0" == "bash" || "$0" == "/bin/bash" || "$0" == "-bash" ]] || { printf '%s\n' 'named controller execution forbidden' >&2; exit 91; }
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || { printf '%s\n' 'ambient shell startup variable present' >&2; exit 92; }
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly ROOT_PYTHON=/usr/bin/python3.10
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ ]] || exit 93
exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - "$ROOT_PYTHON_FD" consumption <<'PY'
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time

ROOT = "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1"
JOB_ID = "143812"
NODE = "auh7-1b-gpu-293"
ROOT_PYTHON = "/usr/bin/python3.10"
ROOT_PYTHON_SHA256 = "11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE = 5937800
SACCT = "/usr/bin/sacct"
SACCT_SHA256 = "fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e"
SACCT_SIZE = 85952
PROBE = ROOT + "/diagnostics/full644_exploratory_matched_r5f_cpu_consumption_probe_v1.py"
PROBE_SHA256 = "fd64fafc9580c8f25c88d79ca603a0dbf192ea98f77403e82f14d4e17c6905f6"
PROBE_SIZE = 56009
VACE_PYTHON = "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
VACE_PYTHON_SHA256 = "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
VACE_PYTHON_SIZE = 31490256
METHODS_ROOT = ROOT + "/release/methods/bernini_action_editing"
SITE_PACKAGES_ROOT = "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
WORK_ROOT = ROOT + "/diagnostics/cpu_consumption_probe_work_r1"
RECEIPT_PATH = WORK_ROOT + "/r5d-cpu-consumption-probe.json"
CONFIGS = {
    "consumption": {
        "step_name": "f644-r5f-consume-v1",
        "parent": ROOT + "/diagnostics",
        "result_parent": WORK_ROOT,
        "result": "r5d-cpu-consumption-probe.json",
        "stdout": "r5f_cpu_consumption_probe.stdout.log",
        "stderr": "r5f_cpu_consumption_probe.stderr.log",
        "evidence": "r5f_cpu_consumption_probe.sacct-and-replay.json",
    },
}

COMPUTE_PAYLOAD = r'''#!/bin/bash -p
set -euo pipefail
umask 077
[[ "$-" == *p* ]] || exit 81
[[ "$0" == "/bin/bash" || "$0" == "bash" || "$0" == "-bash" ]] || exit 82
[[ -z "${BASH_ENV-}" && -z "${ENV-}" ]] || exit 83
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi
readonly ROOT_PYTHON=/usr/bin/python3.10
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ ]] || exit 84
exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B - "$ROOT_PYTHON_FD" <<'CAPTURED_CONSUMPTION_BOOTSTRAP'
import hashlib,os,stat,sys
ROOT_PYTHON="/usr/bin/python3.10"
ROOT_PYTHON_SHA256="11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96"
ROOT_PYTHON_SIZE=5937800
PROBE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1/diagnostics/full644_exploratory_matched_r5f_cpu_consumption_probe_v1.py"
PROBE_SHA256="fd64fafc9580c8f25c88d79ca603a0dbf192ea98f77403e82f14d4e17c6905f6"
PROBE_SIZE=56009
VACE_PYTHON="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
VACE_PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"
VACE_PYTHON_SIZE=31490256
METHODS_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1/release/methods/bernini_action_editing"
SITE_ROOT="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages"
WORK="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5f_job143812_node293_case00_847b91a2_c91de7eb_d70eac5c_r1/diagnostics/cpu_consumption_probe_work_r1"
RECEIPT=WORK+"/r5d-cpu-consumption-probe.json"
def ident(v):
 return (v.st_dev,v.st_ino,v.st_uid,v.st_gid,v.st_mode,v.st_nlink,v.st_rdev,v.st_size,getattr(v,"st_blocks",0),v.st_mtime_ns,v.st_ctime_ns)
def pread(fd,size):
 out=[]; off=0
 while off<size:
  block=os.pread(fd,min(1048576,size-off),off)
  if not block: break
  out.append(block); off+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: raise RuntimeError("captured file short read")
 return raw
if len(sys.argv)!=2 or not sys.argv[1].isascii() or not sys.argv[1].isdecimal(): raise RuntimeError("bootstrap argv differs")
rootfd=int(sys.argv[1])
if rootfd<3 or not os.get_inheritable(rootfd) or os.lseek(rootfd,0,os.SEEK_CUR)!=0: raise RuntimeError("root Python FD differs")
before=os.fstat(rootfd); raw=pread(rootfd,before.st_size); after=os.fstat(rootfd)
if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o755 or before.st_uid!=0 or before.st_gid!=0 or before.st_nlink!=1 or before.st_size!=ROOT_PYTHON_SIZE or ident(before)!=ident(after) or ident(before)!=ident(os.lstat(ROOT_PYTHON)) or ident(before)!=ident(os.stat("/proc/self/exe")) or hashlib.sha256(raw).hexdigest()!=ROOT_PYTHON_SHA256 or os.lseek(rootfd,0,os.SEEK_CUR)!=0: raise RuntimeError("root Python authority differs")
os.set_inheritable(rootfd,False)
def capture(path,pin,size,mode,uid,gid):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_NOFOLLOW",0)); before=os.fstat(fd); raw=pread(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=uid or before.st_gid!=gid or before.st_nlink!=1 or before.st_size!=size or ident(before)!=ident(after) or ident(before)!=ident(named) or hashlib.sha256(raw).hexdigest()!=pin or os.lseek(fd,0,os.SEEK_CUR)!=0: raise RuntimeError("captured authority differs: "+path)
 return fd
pyfd=capture(VACE_PYTHON,VACE_PYTHON_SHA256,VACE_PYTHON_SIZE,0o755,2012,2000)
srcfd=capture(PROBE,PROBE_SHA256,PROBE_SIZE,0o444,2012,2000)
parent=os.path.dirname(WORK); leaf=os.path.basename(WORK); pfd=os.open(parent,os.O_RDONLY|os.O_CLOEXEC|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
try:
 try: os.stat(leaf,dir_fd=pfd,follow_symlinks=False)
 except FileNotFoundError: pass
 else: raise RuntimeError("consumption work root is not fresh")
 os.mkdir(leaf,0o700,dir_fd=pfd); os.fsync(pfd)
 current=os.stat(leaf,dir_fd=pfd,follow_symlinks=False)
 if not stat.S_ISDIR(current.st_mode) or stat.S_IMODE(current.st_mode)!=0o700 or current.st_uid!=2012 or current.st_gid!=2000: raise RuntimeError("consumption work root identity differs")
finally: os.close(pfd)
os.set_inheritable(pyfd,True); os.set_inheritable(srcfd,True)
argv=[VACE_PYTHON,"-I","-S","-B","/proc/self/fd/"+str(srcfd),"--methods-root",METHODS_ROOT,"--site-packages-root",SITE_ROOT,"--work-root",WORK,"--receipt",RECEIPT,"--probe-sha256",PROBE_SHA256]
env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","LC_ALL":"C.UTF-8","CUDA_VISIBLE_DEVICES":"","HIP_VISIBLE_DEVICES":"","ROCR_VISIBLE_DEVICES":""}
os.execve("/proc/self/fd/"+str(pyfd),argv,env)
CAPTURED_CONSUMPTION_BOOTSTRAP
'''.encode("ascii")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(value):
    return {
        "device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
        "gid": value.st_gid, "mode": value.st_mode, "nlink": value.st_nlink,
        "rdev": value.st_rdev, "size": value.st_size,
        "blocks": getattr(value, "st_blocks", 0), "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def object_identity(value):
    return (value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode, value.st_nlink, value.st_rdev)


def pread_exact(descriptor, size):
    chunks = []
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise RuntimeError("short pread")
    return raw


def verify_self(descriptor):
    if descriptor < 3 or not os.get_inheritable(descriptor) or os.lseek(descriptor, 0, os.SEEK_CUR) != 0:
        raise RuntimeError("root Python inherited FD or offset differs")
    before = os.fstat(descriptor)
    named = os.lstat(ROOT_PYTHON)
    process = os.stat("/proc/self/exe")
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o755
        or before.st_uid != 0 or before.st_gid != 0 or before.st_nlink != 1
        or before.st_size != ROOT_PYTHON_SIZE or identity(before) != identity(named)
        or identity(before) != identity(process) or identity(before) != identity(after)
        or hashlib.sha256(raw).hexdigest() != ROOT_PYTHON_SHA256
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        raise RuntimeError("root Python authority differs")
    os.set_inheritable(descriptor, False)
    if os.get_inheritable(descriptor):
        raise RuntimeError("root Python FD remained inheritable")
    return before


def open_pinned(path, expected_sha256, expected_size, expected_uid, expected_gid, expected_mode):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    before = os.fstat(descriptor)
    named = os.lstat(path)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_uid != expected_uid or before.st_gid != expected_gid or before.st_nlink != 1
        or before.st_size != expected_size or identity(before) != identity(named)
        or identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected_sha256
        or os.lseek(descriptor, 0, os.SEEK_CUR) != 0
    ):
        os.close(descriptor)
        raise RuntimeError("pinned file authority differs: " + path)
    return descriptor, before, raw


def replay_pinned(descriptor, path, expected_before, expected_sha256):
    before = os.fstat(descriptor)
    named = os.lstat(path)
    raw = pread_exact(descriptor, before.st_size)
    after = os.fstat(descriptor)
    if identity(before) != identity(expected_before) or identity(before) != identity(named) or identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("pinned file post replay differs: " + path)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeError, ValueError, TypeError) as error:
        raise RuntimeError("strict JSON differs") from error
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise RuntimeError("canonical JSON bytes differ")
    return value


def fresh_at(parent_fd, name):
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError("fresh path gate failed: " + name)


def stable_read_at(parent_fd, name, expected_mode, allow_empty=False):
    descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        raw = pread_exact(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_uid != 2012 or before.st_gid != 2000 or before.st_nlink != 1
            or (not raw and not allow_empty) or identity(before) != identity(after)
            or identity(before) != identity(named)
        ):
            raise RuntimeError("stable artifact replay differs: " + name)
        return raw, identity(before)
    finally:
        os.close(descriptor)


def write_create_only_at(parent_fd, name, raw):
    descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o000, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0 or before.st_nlink != 1 or before.st_size != 0:
            raise RuntimeError("create-only staging identity differs")
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise RuntimeError("create-only write differs")
            offset += written
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if pread_exact(descriptor, len(raw)) != raw or staged.st_size != len(raw) or staged.st_dev != before.st_dev or staged.st_ino != before.st_ino or identity(staged) != identity(named):
            raise RuntimeError("create-only staging replay differs")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_memory(raw):
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT])", raw)
    if match is None:
        raise RuntimeError("memory TRES syntax differs")
    return int(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]


def parse_tres(raw):
    result = {}
    for token in raw.split(","):
        if token.count("=") != 1:
            raise RuntimeError("TRES token differs")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise RuntimeError("TRES closure or duplicate differs")
        result[key] = value
    allowed = {"cpu", "mem", "node", "billing", "gres/gpu", "gres/gpu:mi210"}
    if not set(result).issubset(allowed):
        raise RuntimeError("unexpected TRES key")
    if (
        result.get("cpu") != "4" or result.get("node") != "1"
        or parse_memory(result.get("mem", "")) != 8 * 1024**3
        or result.get("gres/gpu") != "8" or result.get("gres/gpu:mi210") != "8"
        or ("billing" in result and result["billing"] != "4")
    ):
        raise RuntimeError("allocated TRES differs")
    return result


def parse_sacct_row(line, expected_step):
    columns = line.split("|")
    if not columns or columns[0] != expected_step:
        return None
    if len(columns) != 9:
        raise RuntimeError("sacct row field closure differs")
    return columns


def validate_sacct_raw_fixture():
    fixture = (
        "143812.160|f644-r5f-consume-v1|COMPLETED|0:0|1|4|"
        "cpu=4,gres/gpu:mi210=8,gres/gpu=8,mem=8192M,node=1|"
        "auh7-1b-gpu-293|85286912"
    )
    parsed = parse_sacct_row(fixture, "143812.160")
    if parsed is None or len(parsed) != 9 or parsed[-1] != "85286912":
        raise RuntimeError("AUH sacct exact-nine raw fixture differs")
    for hostile in (fixture + "|", fixture + "|extra", fixture.rsplit("|", 1)[0]):
        try:
            parse_sacct_row(hostile, "143812.160")
        except RuntimeError:
            continue
        raise RuntimeError("AUH sacct hostile raw fixture accepted")


def validate_receipt(variant, value, expected_step):
    body = dict(value)
    claimed = body.pop("receipt_digest", None)
    if type(claimed) is not str or not re.fullmatch(r"[0-9a-f]{64}", claimed) or digest(body) != claimed:
        raise RuntimeError("receipt digest differs")
    expected_keys = {"schema_version", "status", "probe_source", "python", "source_authority", "execution_contract", "arms", "hostile_gates", "summary"}
    execution = body.get("execution_contract")
    summary = body.get("summary")
    probe_source = body.get("probe_source")
    if (
        variant != "consumption" or set(body) != expected_keys
        or body.get("schema_version") != "full644-exploratory-matched-r5f-cpu-consumption-probe-v1"
        or body.get("status") != "PASS"
        or summary != {"successful_arms": ["base", "full"], "rejected_hostiles": ["digest", "task", "adapter_namespace", "adapter_leaf"], "success_count": 2, "hostile_rejection_count": 4}
        or type(execution) is not dict or execution.get("linux_cpu_only") is not True
        or execution.get("isolated_no_site_no_bytecode") is not True or execution.get("torch_imported") is not False
        or execution.get("torch_cuda_initialized") is not False or execution.get("torch_distributed_initialized") is not False
        or execution.get("gpu_device_descriptors") != [] or execution.get("real_exec_pass_fds") is not True
        or execution.get("children_executed_from_parent_verified_captured_source") is not True
        or set(body.get("arms", {})) != {"base", "full"}
        or set(body.get("hostile_gates", {})) != {"digest", "task", "adapter_namespace", "adapter_leaf"}
        or type(probe_source) is not dict or probe_source.get("named_path") != PROBE
        or probe_source.get("sha256") != PROBE_SHA256 or probe_source.get("size") != PROBE_SIZE
    ):
        raise RuntimeError("consumption receipt semantic contract differs")
    return claimed


if len(sys.argv) != 3:
    raise RuntimeError("controller argv differs")
root_fd_raw, variant = sys.argv[1:]
if not root_fd_raw.isascii() or not root_fd_raw.isdecimal() or str(int(root_fd_raw)) != root_fd_raw:
    raise RuntimeError("root Python FD syntax differs")
root_fd = int(root_fd_raw)
if variant not in CONFIGS:
    raise RuntimeError("controller variant differs")
if set(os.environ) not in (set(), {"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"] != "C.UTF-8"):
    raise RuntimeError("controller bootstrap environment differs")
os.environ.clear()
validate_sacct_raw_fixture()
root_python_before = verify_self(root_fd)
config = CONFIGS[variant]

parent_fd = os.open(config["parent"], os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
parent_before = os.fstat(parent_fd)
parent_named = os.lstat(config["parent"])
if (
    not stat.S_ISDIR(parent_before.st_mode) or stat.S_IMODE(parent_before.st_mode) != 0o755
    or parent_before.st_uid != 2012 or parent_before.st_gid != 2000
    or object_identity(parent_before) != object_identity(parent_named)
):
    raise RuntimeError("held result parent differs")
probe_fd, probe_before, _ = open_pinned(PROBE, PROBE_SHA256, PROBE_SIZE, 2012, 2000, 0o444)
sacct_fd, sacct_before, _ = open_pinned(SACCT, SACCT_SHA256, SACCT_SIZE, 0, 0, 0o755)
# This exact mode-000 empty inode was created by the one already-completed
# controller attempt before its postflight field-name bug.  Its creating FD
# is gone, so first authenticate it by dirfd lstat, then owner-chmod that same
# inode to 0600 and reopen it.  This replay-only validator never starts srun.
attempt_origin = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if (
    not stat.S_ISREG(attempt_origin.st_mode) or stat.S_IMODE(attempt_origin.st_mode) != 0
    or attempt_origin.st_uid != 2012 or attempt_origin.st_gid != 2000
    or attempt_origin.st_nlink != 1 or attempt_origin.st_size != 0
):
    raise RuntimeError("prior single-attempt tombstone differs")
os.chmod(config["evidence"], 0o600, dir_fd=parent_fd, follow_symlinks=False)
attempt_fd = os.open(config["evidence"], os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
attempt_before = os.fstat(attempt_fd)
attempt_named = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if (
    not stat.S_ISREG(attempt_before.st_mode) or stat.S_IMODE(attempt_before.st_mode) != 0o600
    or attempt_before.st_uid != 2012 or attempt_before.st_gid != 2000
    or attempt_before.st_nlink != 1 or attempt_before.st_size != 0
    or (attempt_before.st_dev, attempt_before.st_ino, attempt_before.st_uid, attempt_before.st_gid,
        attempt_before.st_nlink, attempt_before.st_rdev, attempt_before.st_size)
       != (attempt_origin.st_dev, attempt_origin.st_ino, attempt_origin.st_uid, attempt_origin.st_gid,
           attempt_origin.st_nlink, attempt_origin.st_rdev, attempt_origin.st_size)
    or identity(attempt_before) != identity(attempt_named)
):
    raise RuntimeError("prior tombstone reopen identity differs")
exact_subprocess_environment = {"LANG": "C", "LC_ALL": "C"}
stdout_raw, stdout_identity = stable_read_at(parent_fd, config["stdout"], 0o400)
stderr_raw, stderr_identity = stable_read_at(parent_fd, config["stderr"], 0o400, allow_empty=True)
if stderr_raw != b"":
    raise RuntimeError("prior srun/probe stderr is not empty")
lines = stdout_raw.splitlines(keepends=True)
if len(lines) != 2:
    raise RuntimeError("probe stdout line closure differs")
step_match = re.fullmatch(rb'CONTROLLER_STEP=\{"job_id":"143812","step_id":"([1-9][0-9]*)"\}\n', lines[0])
if step_match is None:
    raise RuntimeError("controller step handoff differs")
step_id = step_match.group(1).decode("ascii")
full_step = JOB_ID + "." + step_id
result_parent_fd = os.open(config["result_parent"], os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
result_parent_info = os.fstat(result_parent_fd)
if (
    not stat.S_ISDIR(result_parent_info.st_mode) or stat.S_IMODE(result_parent_info.st_mode) != 0o700
    or result_parent_info.st_uid != 2012 or result_parent_info.st_gid != 2000
):
    raise RuntimeError("consumption result parent differs")
receipt_raw, receipt_identity = stable_read_at(result_parent_fd, config["result"], 0o400)
receipt = strict_json(receipt_raw)
receipt_digest = validate_receipt(variant, receipt, step_id)
reference = strict_json(lines[1])
expected_reference = {
    "path": RECEIPT_PATH, "sha256": hashlib.sha256(receipt_raw).hexdigest(), "size": len(receipt_raw),
    "mode": 0o400, "receipt_digest": receipt_digest,
}
if reference != expected_reference:
    raise RuntimeError("probe receipt reference differs")

sacct_argv = [
    SACCT, "--jobs=" + full_step, "--noheader", "--parsable2", "--noconvert",
    "--format=JobIDRaw,JobName%64,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%256,NodeList%64,MaxRSS",
]
sacct_attempts = 0
sacct_stdout = b""
sacct_stderr = b""
sacct_columns = None
for sacct_attempts in range(1, 31):
    observed = subprocess.run(
        sacct_argv, executable="/proc/self/fd/" + str(sacct_fd), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, pass_fds=(sacct_fd,),
        cwd="/", env=exact_subprocess_environment, check=False,
    )
    if observed.returncode != 0 or observed.stderr != b"":
        raise RuntimeError("held sacct query failed")
    sacct_stdout = observed.stdout
    sacct_stderr = observed.stderr
    exact_rows = []
    for line in sacct_stdout.decode("utf-8", "strict").splitlines():
        columns = parse_sacct_row(line, full_step)
        if columns is not None:
            exact_rows.append(columns)
    if len(exact_rows) > 1:
        raise RuntimeError("duplicate exact sacct row")
    if len(exact_rows) == 1:
        candidate = exact_rows[0]
        if candidate[2] not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING"}:
            sacct_columns = candidate
            break
    time.sleep(1)
if sacct_columns is None:
    raise RuntimeError("terminal sacct row unavailable")
row_step, row_name, row_state, row_exit, row_elapsed, row_cpus, row_tres, row_node, row_maxrss = sacct_columns
tres = parse_tres(row_tres)
if (
    row_step != full_step or row_name != config["step_name"] or row_state != "COMPLETED" or row_exit != "0:0"
    or not row_elapsed.isascii() or not row_elapsed.isdecimal() or row_cpus != "4" or row_node != NODE
):
    raise RuntimeError("terminal sacct semantics differ")

replay_pinned(probe_fd, PROBE, probe_before, PROBE_SHA256)
replay_pinned(sacct_fd, SACCT, sacct_before, SACCT_SHA256)
root_python_after = os.fstat(root_fd)
root_python_raw_after = pread_exact(root_fd, root_python_after.st_size)
if (
    identity(root_python_after) != identity(root_python_before)
    or identity(root_python_after) != identity(os.lstat(ROOT_PYTHON))
    or identity(root_python_after) != identity(os.stat("/proc/self/exe"))
    or hashlib.sha256(root_python_raw_after).hexdigest() != ROOT_PYTHON_SHA256
    or os.lseek(root_fd, 0, os.SEEK_CUR) != 0
):
    raise RuntimeError("root Python post replay differs")
parent_after = os.fstat(parent_fd)
if object_identity(parent_after) != object_identity(parent_before) or object_identity(parent_after) != object_identity(os.lstat(config["parent"])):
    raise RuntimeError("held result parent post replay differs")

stdout_written, stdout_identity_after = stable_read_at(parent_fd, config["stdout"], 0o400)
stderr_written, stderr_identity_after = stable_read_at(parent_fd, config["stderr"], 0o400, allow_empty=True)
if stdout_written != stdout_raw or stderr_written != stderr_raw or stdout_identity_after != stdout_identity or stderr_identity_after != stderr_identity:
    raise RuntimeError("controller log publication replay differs")

evidence = {
    "schema_version": "full644-r5f-consumption-replay-only-controller-evidence-v2", "status": "PASS",
    "campaign_mode": "case00-pair-canary", "holder_job_id": JOB_ID, "node": NODE, "numeric_step": full_step,
    "replay_only": True, "srun_called_by_replay": False, "subprocess_calls_by_replay": ["sacct"],
    "original_single_srun_attempt": True,
    "original_controller_sha256": "141fe609dbe196e3ad950c0ac48edf3c17c3e3a3c2987dcb93d263ffd5004807",
    "original_postflight_rejection": "probe_source.path field-name mismatch; actual frozen schema uses named_path",
    "existing_attempt_tombstone_recovery": {"authenticated_mode": 0, "temporary_write_mode": 0o600, "same_inode_reopened": True, "final_commit_mode": 0o400},
    "sacct_executable": {"path": SACCT, "sha256": SACCT_SHA256, "executed_via_retained_fd": True},
    "root_python": {"path": ROOT_PYTHON, "sha256": ROOT_PYTHON_SHA256, "executed_via_retained_fd": True},
    "payload": {
        "path": "trusted-controller-embedded-compute-payload",
        "sha256": hashlib.sha256(COMPUTE_PAYLOAD).hexdigest(), "size": len(COMPUTE_PAYLOAD),
        "original_execution_from_sealed_memfd_stdin": True,
    },
    "probe_source": {"path": PROBE, "sha256": PROBE_SHA256, "size": PROBE_SIZE, "login_node_retained_fd_replayed": True},
    "step_handoff": {"job_id": JOB_ID, "step_id": step_id, "canonical_stdout_line": lines[0].decode("ascii").rstrip("\n")},
    "requested_resources": {"cpus_per_task": 4, "memory": "8G", "gpus_per_node": 8, "exclusive": True, "exact": True, "overlap": False},
    "sacct": {
        "query_argv": sacct_argv, "query_attempts": sacct_attempts,
        "raw_stdout_sha256": hashlib.sha256(sacct_stdout).hexdigest(), "raw_stderr_sha256": hashlib.sha256(sacct_stderr).hexdigest(),
        "row": {"JobIDRaw": row_step, "JobName": row_name, "State": row_state, "ExitCode": row_exit,
                "ElapsedRaw": row_elapsed, "AllocCPUS": row_cpus, "AllocTRES": tres, "NodeList": row_node, "MaxRSS": row_maxrss},
    },
    "receipt_replay": {
        "path": RECEIPT_PATH, "file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "receipt_digest": receipt_digest, "identity": receipt_identity, "canonical_json_plus_lf": True,
        "semantic_contract_replayed": True,
    },
    "stdout": {"path": config["parent"] + "/" + config["stdout"], "sha256": hashlib.sha256(stdout_raw).hexdigest(), "identity": stdout_identity},
    "stderr": {"path": config["parent"] + "/" + config["stderr"], "sha256": hashlib.sha256(stderr_raw).hexdigest(), "identity": stderr_identity, "empty": True},
    "formal_report_generated": False, "html_generated": False,
    "external_trust_boundary": "root-owned default Slurm config, dynamic loader, shared libraries, plugins, kernel, and /bin/bash",
}
evidence["evidence_digest"] = digest(evidence)
evidence_raw = canonical(evidence) + b"\n"
attempt_current = os.fstat(attempt_fd)
attempt_named = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if (
    attempt_current.st_dev != attempt_before.st_dev or attempt_current.st_ino != attempt_before.st_ino
    or identity(attempt_current) != identity(attempt_named) or stat.S_IMODE(attempt_current.st_mode) != 0o600 or attempt_current.st_size != 0
):
    raise RuntimeError("single-attempt tombstone replay differs")
offset = 0
while offset < len(evidence_raw):
    count = os.write(attempt_fd, evidence_raw[offset:])
    if count <= 0:
        raise RuntimeError("evidence write differs")
    offset += count
os.fsync(attempt_fd)
staged = os.fstat(attempt_fd)
staged_named = os.stat(config["evidence"], dir_fd=parent_fd, follow_symlinks=False)
if pread_exact(attempt_fd, len(evidence_raw)) != evidence_raw or staged.st_size != len(evidence_raw) or identity(staged) != identity(staged_named):
    raise RuntimeError("evidence staged replay differs")
# Acceptance commit: no fallible provenance checks follow this chmod.
os.fchmod(attempt_fd, 0o400)
pass_line = (
    "R5F_CPU_CONSUMPTION_REPLAY_ONLY_PASS step=" + full_step + " receipt_digest=" + receipt_digest
    + " evidence_digest=" + evidence["evidence_digest"] + "\n"
).encode("ascii")
try:
    os.write(1, pass_line)
except OSError:
    pass
os._exit(0)
PY
