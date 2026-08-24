#!/bin/bash -p
# One SP4 cache or 20-update preservation-v2 canary inside a retained holder.
# This script is release material; the detached four-holder controller invokes
# it through a privileged/no-startup Bash with a small explicit environment.

set -Eeuo pipefail
umask 077

fail() { echo "[action-preservation-v2-node] ERROR: $*" >&2; exit 2; }
sha256_file() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }

readonly confirmation="${ACTION_PRESERVATION_NODE_CONFIRM:?set node confirmation}"
readonly expected_confirmation=run-approved-action-preservation-v2-seed20260818-r1
readonly mode="${ACTION_PRESERVATION_MODE:?set cache or train mode}"
readonly archive="${ACTION_PRESERVATION_SOURCE_ARCHIVE:?set source archive}"
readonly archive_sha="${ACTION_PRESERVATION_SOURCE_ARCHIVE_SHA256:?pin source archive SHA-256}"
readonly release_manifest="${ACTION_PRESERVATION_RELEASE_MANIFEST:?set release manifest}"
readonly release_manifest_sha="${ACTION_PRESERVATION_RELEASE_MANIFEST_SHA256:?pin release manifest SHA-256}"
readonly revision="${ACTION_PRESERVATION_SOURCE_REVISION:?pin source revision}"
readonly manifest="${ACTION_PRESERVATION_SOURCE_MANIFEST:?set source data manifest}"
readonly manifest_sha="${ACTION_PRESERVATION_SOURCE_MANIFEST_SHA256:?pin source manifest SHA-256}"
readonly cache="${ACTION_PRESERVATION_CACHE:?set teacher cache path}"
readonly output="${ACTION_PRESERVATION_OUTPUT:?set fresh output path}"
readonly seed="${ACTION_PRESERVATION_SEED:?set initialization/cache seed}"
readonly arm="${ACTION_PRESERVATION_ARM:-v2_onset_all}"
readonly frozen_site_packages="${ACTION_PRESERVATION_FROZEN_SITE_PACKAGES:?pin frozen site-packages}"
readonly torchrun_path="${ACTION_PRESERVATION_TORCHRUN_PATH:?pin torchrun source path}"
readonly torchrun_sha="${ACTION_PRESERVATION_TORCHRUN_SHA256:?pin torchrun source SHA-256}"
readonly torchrun_size="${ACTION_PRESERVATION_TORCHRUN_SIZE:?pin torchrun source size}"

[[ "${confirmation}" == "${expected_confirmation}" ]] || fail "confirmation differs"
[[ "${mode}" == cache || "${mode}" == train ]] || fail "mode differs"
[[ "${seed}" == 20260818 ]] || fail "v2 seed differs"
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${archive_sha}${release_manifest_sha}${manifest_sha}" =~ ^[0-9a-f]{192}$ ]] || fail "source SHA pin differs"
[[ "${frozen_site_packages}" == /vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages ]] || \
  fail "frozen site-packages authority differs"
[[ "${torchrun_path}" == /vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/run.py && \
  "${torchrun_sha}" == 1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c && \
  "${torchrun_size}" == 31587 ]] || fail "torchrun source authority differs"
case "${arm}" in
  v2_onset_all|v2_noop020_all|v2_func010_all|v2_func025_all|v2_func050_all|\
  v2_onset_cross_qo|v2_func010_cross_qo|v2_func025_cross_qo) ;;
  *) fail "v2 arm differs: ${arm}" ;;
esac
[[ -f "${archive}" && ! -L "${archive}" && "$(/usr/bin/stat -c '%h|%a|%u' "${archive}")" == "1|444|2012" ]] || \
  fail "source archive topology differs"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "source manifest topology differs"
[[ -f "${release_manifest}" && ! -L "${release_manifest}" && \
  "$(/usr/bin/stat -c '%h|%a|%u' "${release_manifest}")" == "1|444|2012" ]] || \
  fail "release manifest topology differs"
[[ "$(sha256_file "${archive}")" == "${archive_sha}" ]] || fail "source archive SHA differs"
[[ "$(sha256_file "${release_manifest}")" == "${release_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ "$(sha256_file "${manifest}")" == "${manifest_sha}" ]] || fail "source manifest SHA differs"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output is not fresh"

cache_contract=()
if [[ "${mode}" == train ]]; then
  readonly expected_cache_sha="${ACTION_PRESERVATION_EXPECTED_CACHE_SHA256:?pin teacher cache SHA-256}"
  [[ "${expected_cache_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "teacher cache SHA pin differs"
  [[ -f "${cache}" && ! -L "${cache}" && "$(/usr/bin/stat -c '%h|%a|%u' "${cache}")" == "1|444|2012" ]] || \
    fail "teacher cache topology differs"
  [[ "$(sha256_file "${cache}")" == "${expected_cache_sha}" ]] || fail "teacher cache SHA differs"
  cache_contract=(--expected-cache-sha256 "${expected_cache_sha}")
fi

readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly root_python=/usr/bin/python3.10
readonly root_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
[[ -f "${python_bin}" && ! -L "${python_bin}" && \
  "$(/usr/bin/stat -c '%h|%a|%u|%g|%s' "${python_bin}")" == \
  "1|755|2012|2000|31490256" ]] || fail "frozen Python topology differs"
[[ "$(sha256_file "${python_bin}")" == "${python_sha}" ]] || fail "frozen Python SHA differs"
[[ -f "${root_python}" && ! -L "${root_python}" && \
  "$(/usr/bin/stat -c '%h|%a|%u|%g|%s' "${root_python}")" == \
  "1|755|0|0|5937800" ]] || fail "root bootstrap Python topology differs"
[[ "$(sha256_file "${root_python}")" == "${root_python_sha}" ]] || \
  fail "root bootstrap Python SHA differs"

readonly source_revision="${revision}"
readonly frozen_python="${python_bin}"
readonly frozen_python_sha="${python_sha}"
runtime_bootstrap_source=""
if ! IFS= read -r -d '' runtime_bootstrap_source <<'PY'
import hashlib,io,json,os,stat,sys,tarfile
archive,archive_sha,manifest_path,manifest_sha,revision,interpreter,frozen,frozen_sha,frozen_size=sys.argv[1:10]
runtime_args=sys.argv[10:]
def ident(x):
 return (x.st_dev,x.st_ino,x.st_uid,x.st_gid,stat.S_IMODE(x.st_mode),x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
def stable(path,expected,mode):
 if not os.path.isabs(path) or os.path.realpath(path)!=path: raise RuntimeError("bootstrap path differs")
 fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
 try:
  a=os.fstat(fd); first=b""
  while True:
   block=os.read(fd,1024*1024)
   if not block: break
   first+=block
  b=os.fstat(fd); os.lseek(fd,0,os.SEEK_SET); second=b""
  while True:
   block=os.read(fd,1024*1024)
   if not block: break
   second+=block
  c=os.fstat(fd); named=os.lstat(path)
  if not stat.S_ISREG(a.st_mode) or a.st_uid!=os.getuid() or a.st_nlink!=1 or stat.S_IMODE(a.st_mode)!=mode: raise RuntimeError("bootstrap topology differs")
  if ident(a)!=ident(b) or ident(a)!=ident(c) or ident(a)!=ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=expected: raise RuntimeError("bootstrap stable capture differs")
  return first
 finally: os.close(fd)
def pairs(items):
 value={}
 for key,item in items:
  if key in value: raise RuntimeError("duplicate manifest key")
  value[key]=item
 return value
manifest_raw=stable(manifest_path,manifest_sha,0o444)
manifest=json.loads(manifest_raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
canonical=json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
if canonical!=manifest_raw or manifest.get("content_revision")!=revision: raise RuntimeError("bootstrap manifest differs")
runtime_relative="action_preservation_verified_release_v1.py"
rows=[row for row in manifest.get("files",[]) if row.get("path")==runtime_relative]
if len(rows)!=1: raise RuntimeError("verified runtime row differs")
row=rows[0]
if set(row)!={"path","mode","size","sha256"} or row["mode"]!=0o444: raise RuntimeError("verified runtime row shape differs")
archive_raw=stable(archive,archive_sha,0o444)
with tarfile.open(fileobj=io.BytesIO(archive_raw),mode="r:") as handle:
 member_name=manifest["member_root"]+"/"+runtime_relative
 members=[member for member in handle.getmembers() if member.name==member_name]
 if len(members)!=1 or not members[0].isfile() or members[0].linkname: raise RuntimeError("verified runtime member differs")
 source=handle.extractfile(members[0]).read()
if len(source)!=row["size"] or hashlib.sha256(source).hexdigest()!=row["sha256"]: raise RuntimeError("verified runtime bytes differ")
source_text=source.decode("utf-8")
display=manifest_path+"!"+manifest["member_root"]+"/"+runtime_relative
if interpreter=="root":
 sys.argv=[display,*runtime_args]
 scope={"__name__":"__main__","__file__":display,"__package__":None,"__spec__":None,"__cached__":None,"__builtins__":__builtins__}
 exec(compile(source,display,"exec",dont_inherit=True),scope)
elif interpreter=="frozen":
 if not hasattr(os,"supports_fd") or os.execve not in os.supports_fd: raise RuntimeError("fd exec is unavailable")
 fd=os.open(frozen,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
 a=os.fstat(fd); first=b""
 while True:
  block=os.read(fd,1024*1024)
  if not block: break
  first+=block
 b=os.fstat(fd); os.lseek(fd,0,os.SEEK_SET); second=b""
 while True:
  block=os.read(fd,1024*1024)
  if not block: break
  second+=block
 c=os.fstat(fd); named=os.lstat(frozen)
 if not stat.S_ISREG(a.st_mode) or (a.st_uid,a.st_gid,stat.S_IMODE(a.st_mode),a.st_nlink,a.st_size)!=(2012,2000,0o755,1,int(frozen_size)): raise RuntimeError("frozen Python topology differs")
 if ident(a)!=ident(b) or ident(a)!=ident(c) or ident(a)!=ident(named) or first!=second or hashlib.sha256(first).hexdigest()!=frozen_sha: raise RuntimeError("frozen Python bytes differ")
 os.execve(fd,[frozen,"-I","-S","-B","-c",source_text,*runtime_args],os.environ)
else: raise RuntimeError("bootstrap interpreter differs")
PY
then
  [[ -n "${runtime_bootstrap_source}" ]] || fail "verified runtime bootstrap source is empty"
fi
readonly runtime_bootstrap_source

torchrun_bootstrap_source=""
if ! IFS= read -r -d '' torchrun_bootstrap_source <<'PY'
import hashlib,sys
from types import ModuleType
source,origin,expected_sha,site_packages=sys.argv[1:5]
runtime_args=sys.argv[5:]
raw=source.encode("utf-8")
if hashlib.sha256(raw).hexdigest()!=expected_sha: raise RuntimeError("captured torchrun SHA differs")
if site_packages!="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages": raise RuntimeError("frozen site-packages path differs")
if "sitecustomize" in sys.modules or "usercustomize" in sys.modules: raise RuntimeError("automatic site customization ran before captured torchrun")
sys.path.append(site_packages)
sys.argv=[origin,*runtime_args]
module=ModuleType("__main__")
module.__file__=origin
module.__package__="torch.distributed"
module.__loader__=None
module.__spec__=None
module.__cached__=None
module.__builtins__=__builtins__
sys.modules["__main__"]=module
exec(compile(raw,origin,"exec",dont_inherit=True),module.__dict__)
PY
then
  [[ -n "${torchrun_bootstrap_source}" ]] || fail "isolated torchrun bootstrap source is empty"
fi
readonly torchrun_bootstrap_source

run_release_runtime() {
  local interpreter="$1"
  shift
  "${root_python}" -I -S -B -c "${runtime_bootstrap_source}" \
    "${archive}" "${archive_sha}" "${release_manifest}" "${release_manifest_sha}" \
    "${source_revision}" "${interpreter}" "${frozen_python}" "${frozen_python_sha}" 31490256 "$@"
}


unset BASH_ENV ENV PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset LD_PRELOAD GCONV_PATH LOCPATH CDPATH GLOBIGNORE
while IFS='=' read -r name _; do
  case "${name}" in BASH_FUNC_*|SHELLOPTS|BASHOPTS) unset "${name}" 2>/dev/null || true ;; esac
done < <(/usr/bin/env)

scratch="$(/usr/bin/mktemp -d "/tmp/action-preservation-v2-${arm}.XXXXXXXX")" || \
  fail "scratch creation failed"
readonly scratch
/usr/bin/mkdir -m 700 "${scratch}/miopen-user" \
  "${scratch}/miopen-custom" "${scratch}/torch-extensions" "${scratch}/triton"
run_release_runtime root extract \
  --archive "${archive}" --expected-archive-sha256 "${archive_sha}" \
  --manifest "${release_manifest}" --expected-manifest-sha256 "${release_manifest_sha}" \
  --expected-content-revision "${source_revision}" --output-root "${scratch}/source" \
  >"${scratch}/materialization.json" || fail "verified node release extraction failed"
/usr/bin/chmod 0444 "${scratch}/materialization.json"
readonly runner="${scratch}/source/methods/bernini_action_editing/train_self_generated_action_quotient_v1.py"
[[ -f "${runner}" && ! -L "${runner}" && "$(/usr/bin/stat -c '%a|%u' "${runner}")" == "444|2012" ]] || \
  fail "materialized trainer differs"

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MIOPEN_USER_DB_PATH="${scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${scratch}/torch-extensions"
export TRITON_CACHE_DIR="${scratch}/triton"

echo "ACTION_PRESERVATION_V2_SCRATCH_RETAINED path=${scratch}" >&2
exec "${root_python}" -I -S -B -c '
import hashlib,os,stat,sys
python_path,python_sha,python_size,torchrun_path,torchrun_sha,torchrun_size,site_packages,torchrun_bootstrap=sys.argv[1:9]
launcher_args=sys.argv[9:]
python_size=int(python_size); torchrun_size=int(torchrun_size)
if python_path!="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12" or python_sha!="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a" or python_size!=31490256: raise RuntimeError("frozen Python literal authority differs")
if site_packages!="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages": raise RuntimeError("frozen site-packages literal authority differs")
if torchrun_path!="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/run.py" or torchrun_sha!="1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c" or torchrun_size!=31587: raise RuntimeError("torchrun literal authority differs")
flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_CLOEXEC",0)
def ident(x):
 return (x.st_dev,x.st_ino,x.st_uid,x.st_gid,x.st_mode,x.st_nlink,x.st_rdev,x.st_size,getattr(x,"st_blocks",0),x.st_mtime_ns,x.st_ctime_ns)
def stable(path,expected,expected_size,expected_mode,expected_uid,expected_gid,retain):
 if not os.path.isabs(path) or os.path.realpath(path)!=path: raise RuntimeError("stable executable/source path differs")
 fd=os.open(path,flags)
 try:
  a=os.fstat(fd); first=b""
  while True:
   block=os.read(fd,1024*1024)
   if not block: break
   first+=block
  b=os.fstat(fd); os.lseek(fd,0,os.SEEK_SET); second=b""
  while True:
   block=os.read(fd,1024*1024)
   if not block: break
   second+=block
  c=os.fstat(fd); named=os.lstat(path)
  if not stat.S_ISREG(a.st_mode) or stat.S_ISLNK(named.st_mode): raise RuntimeError("stable executable/source kind differs")
  if (a.st_uid,a.st_gid,stat.S_IMODE(a.st_mode),a.st_nlink,a.st_size)!=(expected_uid,expected_gid,expected_mode,1,expected_size): raise RuntimeError("stable executable/source topology differs")
  if ident(a)!=ident(b) or ident(a)!=ident(c) or ident(a)!=ident(named): raise RuntimeError("stable executable/source identity changed")
  if first!=second or len(first)!=expected_size or hashlib.sha256(first).hexdigest()!=expected: raise RuntimeError("stable executable/source bytes differ")
  if retain: return fd,first
  os.close(fd); return None,first
 except BaseException:
  os.close(fd); raise
torchrun_fd,torchrun_raw=stable(torchrun_path,torchrun_sha,torchrun_size,0o644,2012,2000,False)
if torchrun_fd is not None: raise RuntimeError("torchrun descriptor retention differs")
try: torchrun_source=torchrun_raw.decode("utf-8","strict")
except UnicodeDecodeError as error: raise RuntimeError("captured torchrun is not UTF-8") from error
fd,_=stable(python_path,python_sha,python_size,0o755,2012,2000,True)
try:
 os.execve(fd,[python_path,"-I","-S","-B","-c",torchrun_bootstrap,torchrun_source,torchrun_path,torchrun_sha,site_packages,*launcher_args],os.environ)
finally:
 os.close(fd)
' "${python_bin}" "${python_sha}" 31490256 \
  "${torchrun_path}" "${torchrun_sha}" "${torchrun_size}" \
  "${frozen_site_packages}" "${torchrun_bootstrap_source}" \
  --standalone --nproc_per_node=4 \
  --no-python "${root_python}" -I -S -B -c "${runtime_bootstrap_source}" \
  "${archive}" "${archive_sha}" "${release_manifest}" "${release_manifest_sha}" \
  "${source_revision}" frozen "${python_bin}" "${python_sha}" 31490256 \
  verified-run --release-root "${scratch}/source" --manifest "${release_manifest}" \
  --expected-manifest-sha256 "${release_manifest_sha}" \
  --expected-content-revision "${source_revision}" \
  --target train_self_generated_action_quotient_v1.py -- \
  --mode "${mode}" --objective-family preservation_v2 \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" --source-manifest "${manifest}" \
  --source-manifest-sha256 "${manifest_sha}" --cache "${cache}" \
  "${cache_contract[@]}" --output "${output}" --arm "${arm}" \
  --slots 5 --limit-cells 0 --max-steps 20 --seed "${seed}" \
  --method-source-revision "${revision}" \
  --method-source-archive-sha256 "${archive_sha}"
