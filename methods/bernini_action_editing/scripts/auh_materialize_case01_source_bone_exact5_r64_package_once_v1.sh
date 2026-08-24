#!/bin/bash -p
# Trusted-stdin, no-Slurm materialization of the fresh exact5 R64 package.

set -Eeuo pipefail
umask 077

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || {
  /usr/bin/printf 'exact5 package refused: shell entry differs\n' >&2
  exit 96
}
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C \
  && "${LANG:-}" == C && "${HOME:-}" == /vast/users/guangyi.chen \
  && "${BASH_ENV:-}" == /dev/null ]] || {
  /usr/bin/printf 'exact5 package refused: entry environment differs\n' >&2
  exit 96
}
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || {
  /usr/bin/printf 'exact5 package refused: loader/GPU environment differs\n' >&2
  exit 96
}
if builtin declare -F | /usr/bin/grep . >/dev/null; then
  /usr/bin/printf 'exact5 package refused: preloaded Bash function exists\n' >&2
  exit 96
fi
if shopt -q varredir_close 2>/dev/null; then shopt -u varredir_close; fi

readonly SOURCE_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1
readonly TARGET_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1
readonly RANK_CACHE=/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache
readonly VACE_PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly MATERIALIZER="$SOURCE_ROOT/methods/bernini_action_editing/tools/materialize_case01_source_bone_exact5_r64_package_v1.py"

[[ -f "$VACE_PYTHON" && -x "$VACE_PYTHON" && ! -L "$VACE_PYTHON" \
  && -f "$MATERIALIZER" && ! -L "$MATERIALIZER" \
  && ! -e "$TARGET_ROOT" && ! -L "$TARGET_ROOT" \
  && ! -e "$RANK_CACHE" && ! -L "$RANK_CACHE" ]] || {
  /usr/bin/printf 'exact5 package refused: held entry/fresh target differs\n' >&2
  exit 96
}
exec {VACE_PYTHON_FD}<"$VACE_PYTHON"
exec {MATERIALIZER_FD}<"$MATERIALIZER"
[[ "$VACE_PYTHON_FD" =~ ^[0-9]+$ && "$MATERIALIZER_FD" =~ ^[0-9]+$ \
  && "$VACE_PYTHON_FD" -ge 3 && "$MATERIALIZER_FD" -ge 3 \
  && "$VACE_PYTHON_FD" != "$MATERIALIZER_FD" ]] || exit 97

PACKAGE_BOOTSTRAP='import hashlib,json,os,stat,sys
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
if len(sys.argv)!=7: fail("package bootstrap argv differs")
pyfd,srcfd=int(sys.argv[1]),int(sys.argv[2]); python_path,materializer_path,source_root,target_root=sys.argv[3:]
if pyfd<3 or srcfd<3 or pyfd==srcfd or not os.get_inheritable(pyfd) or not os.get_inheritable(srcfd): fail("package bootstrap descriptor differs")
if sys.stdin.buffer.read(1)!=b"": fail("package bootstrap stdin differs")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"): fail("package bootstrap environment differs")
if sys.platform!="linux" or not os.path.isdir("/proc/self/fd") or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode: fail("package isolated startup differs")
if os.geteuid()!=2012 or os.getegid()!=2000: fail("package process owner differs")
expected_source="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_source_snapshot_24_20260821_r1"
expected_target="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_object_grounded_case01_0821_exact5_r64_canary_v1"
if source_root!=expected_source or target_root!=expected_target or materializer_path!=source_root+"/methods/bernini_action_editing/tools/materialize_case01_source_bone_exact5_r64_package_v1.py": fail("package path binding differs")
if os.path.lexists(target_root) or os.path.lexists("/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"): fail("package target/cache is not fresh")
for directory,mode in ((source_root,0o555),(os.path.dirname(target_root),None)):
 info=os.lstat(directory)
 if os.path.realpath(directory)!=directory or not stat.S_ISDIR(info.st_mode) or info.st_uid!=2012 or info.st_gid!=2000 or (mode is not None and stat.S_IMODE(info.st_mode)!=mode) or stat.S_IMODE(info.st_mode)&0o002: fail("package authority directory differs: "+directory)
held(pyfd,python_path,"8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",31490256,0o755,2012,2000,True)
materializer_raw=held(srcfd,materializer_path,"937023f00ab5aa86ce0a5c6274c0901be35e4bd852152b55928abbf26c6102b8",49511,0o444,2012,2000)
try: source=materializer_raw.decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("package materializer is not UTF-8") from error
compile(source,materializer_path,"exec",dont_inherit=True)
if os.path.lexists(target_root) or os.path.lexists("/tmp/bernini-case01-exact5-r64-job143808-node292-r1-rank-cache"): fail("package target/cache changed during preflight")
if ident(os.fstat(pyfd))!=ident(os.lstat(python_path)) or ident(os.fstat(pyfd))!=ident(os.stat("/proc/self/exe")): fail("held VACE Python changed")
if ident(os.fstat(srcfd))!=ident(os.lstat(materializer_path)) or hashlib.sha256(read(srcfd,os.fstat(srcfd).st_size)).hexdigest()!="937023f00ab5aa86ce0a5c6274c0901be35e4bd852152b55928abbf26c6102b8": fail("held package materializer changed")
os.set_inheritable(srcfd,False); os.set_inheritable(pyfd,True)
argv=[python_path,"-I","-S","-B","-c",source,"--job-id","143808","--node","auh7-1b-gpu-292","--source-root",source_root,"--materializer-sha256","937023f00ab5aa86ce0a5c6274c0901be35e4bd852152b55928abbf26c6102b8"]
os.execve("/proc/self/fd/"+str(pyfd),argv,{})'
readonly PACKAGE_BOOTSTRAP

exec -c "/proc/self/fd/$VACE_PYTHON_FD" -I -S -B -c "$PACKAGE_BOOTSTRAP" \
  "$VACE_PYTHON_FD" "$MATERIALIZER_FD" "$VACE_PYTHON" "$MATERIALIZER" \
  "$SOURCE_ROOT" "$TARGET_ROOT"
