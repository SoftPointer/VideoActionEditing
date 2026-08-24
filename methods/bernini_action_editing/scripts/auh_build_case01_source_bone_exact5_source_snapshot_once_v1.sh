#!/bin/bash -p
# Trusted-stdin, create-only construction of the sealed exact5 source snapshot.
# No Slurm command appears in this controller.

set -Eeuo pipefail
umask 077

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || {
  /usr/bin/printf 'exact5 snapshot refused: shell entry differs\n' >&2
  exit 96
}
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C \
  && "${LANG:-}" == C && "${HOME:-}" == /vast/users/guangyi.chen \
  && "${BASH_ENV:-}" == /dev/null ]] || {
  /usr/bin/printf 'exact5 snapshot refused: entry environment differs\n' >&2
  exit 96
}
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || {
  /usr/bin/printf 'exact5 snapshot refused: loader/GPU environment differs\n' >&2
  exit 96
}
if builtin declare -F | /usr/bin/grep . >/dev/null; then
  /usr/bin/printf 'exact5 snapshot refused: preloaded Bash function exists\n' >&2
  exit 96
fi
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi

readonly ROOT_PYTHON=/usr/bin/python3.10
readonly STAGING_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_staging_v1
readonly OLD_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_r5f_source_snapshot_21_20260820_r1
readonly TARGET_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1
readonly BUILDER="$STAGING_ROOT/methods/bernini_action_editing/tools/build_case01_source_bone_exact5_source_snapshot_v1.py"

[[ -f "$ROOT_PYTHON" && -x "$ROOT_PYTHON" && ! -L "$ROOT_PYTHON" \
  && -f "$BUILDER" && ! -L "$BUILDER" \
  && ! -e "$TARGET_ROOT" && ! -L "$TARGET_ROOT" ]] || {
  /usr/bin/printf 'exact5 snapshot refused: held entry or fresh target differs\n' >&2
  exit 96
}
exec {ROOT_PYTHON_FD}<"$ROOT_PYTHON"
exec {BUILDER_FD}<"$BUILDER"
[[ "$ROOT_PYTHON_FD" =~ ^[0-9]+$ && "$BUILDER_FD" =~ ^[0-9]+$ \
  && "$ROOT_PYTHON_FD" -ge 3 && "$BUILDER_FD" -ge 3 \
  && "$ROOT_PYTHON_FD" != "$BUILDER_FD" ]] || exit 97

SNAPSHOT_BOOTSTRAP='import hashlib,os,stat,sys
def fail(message): raise RuntimeError(message)
def ident(value): return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,value.st_mode,value.st_nlink,value.st_rdev,value.st_size,getattr(value,"st_blocks",0),value.st_mtime_ns,value.st_ctime_ns)
def read(fd,size):
 out=[]; offset=0
 while offset<size:
  block=os.pread(fd,min(1048576,size-offset),offset)
  if not block: break
  out.append(block); offset+=len(block)
 raw=b"".join(out)
 if len(raw)!=size: fail("held read size differs")
 return raw
def held(fd,path,digest,size,mode,uid,gid,process=False):
 before=os.fstat(fd); first=read(fd,before.st_size); middle=os.fstat(fd); second=read(fd,before.st_size); after=os.fstat(fd); named=os.lstat(path)
 if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=mode or before.st_uid!=uid or before.st_gid!=gid or before.st_nlink!=1 or before.st_size!=size or ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=digest or (process and ident(before)!=ident(os.stat("/proc/self/exe"))): fail("held authority differs: "+path)
 return first
if len(sys.argv)!=7: fail("snapshot bootstrap argv differs")
pyfd,srcfd=int(sys.argv[1]),int(sys.argv[2]); python_path,builder_path,staging,old,target=sys.argv[3:]
if pyfd<3 or srcfd<3 or pyfd==srcfd or not os.get_inheritable(pyfd) or not os.get_inheritable(srcfd): fail("snapshot bootstrap descriptor differs")
if sys.stdin.buffer.read(1)!=b"": fail("snapshot bootstrap stdin differs")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): fail("snapshot bootstrap environment differs")
if sys.platform!="linux" or not os.path.isdir("/proc/self/fd") or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode: fail("snapshot isolated startup differs")
if os.geteuid()!=2012 or os.getegid()!=2000: fail("snapshot process owner differs")
expected_staging="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_staging_v1"
expected_old="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_r5f_source_snapshot_21_20260820_r1"
expected_target="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
if staging!=expected_staging or old!=expected_old or target!=expected_target or builder_path!=staging+"/methods/bernini_action_editing/tools/build_case01_source_bone_exact5_source_snapshot_v1.py": fail("snapshot path binding differs")
if os.path.lexists(target): fail("fresh snapshot target exists before preflight")
for directory in (staging,old,os.path.dirname(target)):
 info=os.lstat(directory)
 if os.path.realpath(directory)!=directory or not stat.S_ISDIR(info.st_mode) or info.st_uid!=2012 or info.st_gid!=2000 or stat.S_IMODE(info.st_mode)&0o002: fail("snapshot authority directory differs: "+directory)
held(pyfd,python_path,"11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96",5937800,0o755,0,0,True)
builder_raw=held(srcfd,builder_path,"906db79519f8689f4ec3a2ceee626f788cd8d9f032178ba2b61346a5108d9a69",19690,stat.S_IMODE(os.fstat(srcfd).st_mode),2012,2000)
if stat.S_IMODE(os.fstat(srcfd).st_mode)&0o022: fail("staged builder is writable by group/world")
try: source=builder_raw.decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("snapshot builder is not UTF-8") from error
compile(source,builder_path,"exec",dont_inherit=True)
if os.path.lexists(target): fail("fresh snapshot target exists after complete preflight")
if ident(os.fstat(pyfd))!=ident(os.lstat(python_path)) or ident(os.fstat(pyfd))!=ident(os.stat("/proc/self/exe")): fail("held root Python changed")
if ident(os.fstat(srcfd))!=ident(os.lstat(builder_path)) or hashlib.sha256(read(srcfd,os.fstat(srcfd).st_size)).hexdigest()!="906db79519f8689f4ec3a2ceee626f788cd8d9f032178ba2b61346a5108d9a69": fail("held snapshot builder changed")
os.set_inheritable(srcfd,False); os.set_inheritable(pyfd,True)
os.execve("/proc/self/fd/"+str(pyfd),[python_path,"-I","-S","-B","-c",source],{})'
readonly SNAPSHOT_BOOTSTRAP

exec -c "/proc/self/fd/$ROOT_PYTHON_FD" -I -S -B -c "$SNAPSHOT_BOOTSTRAP" \
  "$ROOT_PYTHON_FD" "$BUILDER_FD" "$ROOT_PYTHON" "$BUILDER" \
  "$STAGING_ROOT" "$OLD_ROOT" "$TARGET_ROOT"
