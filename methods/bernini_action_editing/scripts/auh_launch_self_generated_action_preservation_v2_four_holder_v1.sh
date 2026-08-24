#!/bin/bash -p
# Fresh 8-arm x 20-update preservation-v2 mechanism canary on four retained
# AUH holders.  This controller never cancels/releases/requeues a parent job,
# never retries, and never treats training loss as scientific promotion.

set -Eeuo pipefail
umask 077
export PATH=/opt/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset BASH_ENV ENV PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
unset LD_PRELOAD LD_LIBRARY_PATH GLIBC_TUNABLES GCONV_PATH LOCPATH CDPATH GLOBIGNORE

fail() { echo "[action-preservation-v2] ERROR: $*" >&2; exit 2; }
sha256_file() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }

readonly confirmation="${APV2_CONFIRM:?set explicit preservation-v2 confirmation}"
readonly release_root="${APV2_RELEASE_ROOT:?set immutable release root}"
readonly experiment_root="${APV2_EXPERIMENT_ROOT:?set fresh experiment root}"
readonly archive_sha="${APV2_ARCHIVE_SHA256:?pin source archive SHA-256}"
readonly release_manifest_sha="${APV2_RELEASE_MANIFEST_SHA256:?pin release manifest SHA-256}"
readonly controller_sha="${APV2_CONTROLLER_SHA256:?pin detached controller SHA-256}"
readonly envelope_sha="${APV2_DEPLOYMENT_ENVELOPE_SHA256:?pin deployment envelope SHA-256}"
readonly source_revision="${APV2_SOURCE_REVISION:?pin source revision}"

readonly expected_confirmation=launch-approved-action-preservation-v2-four-holder-r1
readonly canary_seed=20260818
readonly root_python=/usr/bin/python3.10
readonly root_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
readonly frozen_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly frozen_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly frozen_site_packages=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages
readonly torchrun_path=/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/run.py
readonly torchrun_sha=1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c
readonly torchrun_size=31587
readonly squeue_sha=ce95d4147756cebe87597f06c8563e2e8392d62af4a8b906f9c5d77fddc8e17c
readonly srun_sha=2b8f60b30edf7efed35bb00864651da1b0bec68e75f942ce58b5ff82bc43cd9e
readonly ssh_sha=3a9c5d143150f0b2816ab1a5a7c58a9f970280b061f617abee54d2834a498b53
readonly bash_sha=59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4
readonly bash_size=1396520
readonly source_data_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_quotient_job140846_v4/source_only/manifest.json
readonly source_data_manifest_sha=62fee73b3d84015f2e72edcd4da14b51f7695980a4ba892420ca137aa50e9ad8
readonly source_data_manifest_digest=2fb367ed6f06275705e0b71020dd87fd68e13a010e80ef0bd2a122c94070f503
readonly archive="${release_root}/source.tar"
readonly release_manifest="${release_root}/source.manifest.json"
readonly controller="${release_root}/auh_launch_self_generated_action_preservation_v2_four_holder_v1.sh"
readonly envelope="${release_root}/deployment-envelope.json"
readonly materialized="${experiment_root}/materialized"
readonly node_runner="${materialized}/methods/bernini_action_editing/scripts/auh_run_self_generated_action_preservation_v2.sh"
readonly auditor="${materialized}/methods/bernini_action_editing/audit_self_generated_action_preservation_v2.py"
readonly verified_runtime="${materialized}/methods/bernini_action_editing/action_preservation_verified_release_v1.py"
readonly completion_publisher="${materialized}/methods/bernini_action_editing/action_preservation_completion_publisher_v1.py"
readonly cache="${experiment_root}/teacher-cache-preservation-v2-seed20260818-row4-sigma5.pt"

[[ "${confirmation}" == "${expected_confirmation}" ]] || fail "confirmation differs"
[[ "${release_root}" == /vast/users/guangyi.chen/* && "${experiment_root}" == /vast/users/guangyi.chen/* ]] || fail "root path differs"
[[ "${release_root}" != / && "${experiment_root}" != / && "${release_root}" != "${experiment_root}" ]] || fail "root topology differs"
[[ "${experiment_root}" != "${release_root}"/* && "${release_root}" != "${experiment_root}"/* ]] || fail "release/experiment ancestry overlaps"
[[ "${archive_sha}${release_manifest_sha}${controller_sha}${envelope_sha}" =~ ^[0-9a-f]{256}$ ]] || fail "release SHA pin differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ ! -e "${experiment_root}" && ! -L "${experiment_root}" ]] || fail "experiment root is not fresh"
[[ "$(/usr/bin/realpath -m -- "${experiment_root}")" == "${experiment_root}" ]] || fail "experiment root is not canonical"
[[ -d "${release_root}" && ! -L "${release_root}" && "$(/usr/bin/readlink -f -- "${release_root}")" == "${release_root}" ]] || fail "release root differs"
[[ "$(/usr/bin/readlink -f -- "$0")" == "${controller}" ]] || fail "executing controller is not sealed release copy"
[[ -f "${root_python}" && ! -L "${root_python}" && "$(/usr/bin/stat -c '%h|%a|%u|%g|%s' "${root_python}")" == "1|755|0|0|5937800" ]] || fail "root Python topology differs"
[[ "$(sha256_file "${root_python}")" == "${root_python_sha}" ]] || fail "root Python SHA differs"
[[ -f "${frozen_python}" && ! -L "${frozen_python}" && \
  "$(/usr/bin/stat -c '%h|%a|%u|%g|%s' "${frozen_python}")" == \
  "1|755|2012|2000|31490256" ]] || fail "frozen Python topology differs"
[[ "$(sha256_file "${frozen_python}")" == "${frozen_python_sha}" ]] || fail "frozen Python SHA differs"
for binary_pin in \
  "/usr/bin/squeue:${squeue_sha}:130864" \
  "/usr/bin/srun:${srun_sha}:164720" \
  "/usr/bin/ssh:${ssh_sha}:846888"; do
  binary="${binary_pin%%:*}"; rest="${binary_pin#*:}"; digest="${rest%%:*}"; size="${rest##*:}"
  [[ -f "${binary}" && ! -L "${binary}" && \
    "$(/usr/bin/stat -c '%h|%a|%u|%g|%s' "${binary}")" == \
    "1|755|0|0|${size}" ]] || fail "cluster binary topology differs: ${binary}"
  [[ "$(sha256_file "${binary}")" == "${digest}" ]] || fail "cluster binary SHA differs: ${binary}"
done

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

run_release_runtime() {
  local interpreter="$1"
  shift
  "${root_python}" -I -S -B -c "${runtime_bootstrap_source}" \
    "${archive}" "${archive_sha}" "${release_manifest}" "${release_manifest_sha}" \
    "${source_revision}" "${interpreter}" "${frozen_python}" "${frozen_python_sha}" 31490256 "$@"
}

seal_shared_evidence_file() {
  "${root_python}" -I -S -B -c '
import hashlib,os,stat,sys
path=sys.argv[1]; expected=sys.argv[2]
fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
def ident(value):
 return (value.st_dev,value.st_ino,value.st_uid,value.st_gid,stat.S_IMODE(value.st_mode),value.st_nlink,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
try:
 os.fchmod(fd,0o444); os.fsync(fd); before=os.fstat(fd)
 assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1 and stat.S_IMODE(before.st_mode)==0o444 and before.st_size>0
 first=b""
 while True:
  block=os.read(fd,1024*1024)
  if not block: break
  first+=block
 middle=os.fstat(fd); os.lseek(fd,0,os.SEEK_SET); second=b""
 while True:
  block=os.read(fd,1024*1024)
  if not block: break
  second+=block
 after=os.fstat(fd); named=os.lstat(path)
 assert first==second and ident(before)==ident(middle)==ident(after)==ident(named)
 digest=hashlib.sha256(first).hexdigest()
 assert not expected or digest==expected
finally: os.close(fd)
parent=os.open(os.path.dirname(path),os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
try: os.fsync(parent)
finally: os.close(parent)
print(digest)
' "$1" "${2:-}"
}

readonly expected_release_entries=$'auh_launch_self_generated_action_preservation_v2_four_holder_v1.sh\ndeployment-envelope.json\nsource.manifest.json\nsource.tar'
observed_release_entries="$(/usr/bin/find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"
[[ "${observed_release_entries}" == "${expected_release_entries}" ]] || fail "release entry closure differs"
for pair in \
  "${archive}:${archive_sha}:444" \
  "${release_manifest}:${release_manifest_sha}:444" \
  "${controller}:${controller_sha}:555" \
  "${envelope}:${envelope_sha}:444"; do
  path="${pair%%:*}"; rest="${pair#*:}"; digest="${rest%%:*}"; mode="${rest##*:}"
  [[ -f "${path}" && ! -L "${path}" && "$(/usr/bin/stat -c '%h|%a|%u' "${path}")" == "1|${mode}|2012" ]] || fail "release file topology differs: ${path}"
  [[ "$(sha256_file "${path}")" == "${digest}" ]] || fail "release file SHA differs: ${path}"
done
[[ -f "${source_data_manifest}" && ! -L "${source_data_manifest}" ]] || fail "source data manifest differs"
[[ "$(sha256_file "${source_data_manifest}")" == "${source_data_manifest_sha}" ]] || fail "source data manifest SHA differs"

jobs=(136719 136141 136309 136140)
nodes=(auh7-1b-gpu-306 auh7-1b-gpu-299 auh7-1b-gpu-280 auh7-1b-gpu-215)
holder_preflight() {
  local job="$1" node="$2" observed_steps snapshot
  [[ "$(/usr/bin/squeue -j "${job}" -h -o '%T|%N|%u|%C|%m|%b')" == \
    "RUNNING|${node}|guangyi.chen|64|64G|gres/gpu:mi210:8" ]] || fail "holder state/resources differ: ${job}"
  observed_steps="$(/usr/bin/squeue -s -j "${job}" -h -o '%i' | LC_ALL=C /usr/bin/sort)"
  [[ "${observed_steps}" == "${job}.batch"$'\n'"${job}.extern" ]] || fail "holder has a numbered step: ${job}"
  snapshot="$(/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=10 -- "${node}" \
    '/usr/bin/rocm-smi --showuse --showmemuse --showpids --json')" || fail "GPU probe failed: ${node}"
  /usr/bin/printf '%s' "${snapshot}" | "${root_python}" -I -S -B -c '
import json,sys
x=json.load(sys.stdin)
assert set(x)=={f"card{i}" for i in range(8)}|{"system"}
for i in range(8):
 c=x[f"card{i}"]
 assert c["GPU use (%)"]=="0" and c["GPU Memory Allocated (VRAM%)"]=="0"
for key,value in x["system"].items():
 assert key.startswith("PID") and key[3:].isdigit()
 fields=[item.strip() for item in value.split(",")]
 assert fields[0]=="gpuagent" and fields[1:]==["0","0","0","0"]
' || fail "GPU/KFD is not idle: ${node}"
}
holder_preflight_after_owned_step() {
  local job="$1" node="$2" attempt observed_steps
  for attempt in $(/usr/bin/seq 1 30); do
    observed_steps="$(/usr/bin/squeue -s -j "${job}" -h -o '%i' | LC_ALL=C /usr/bin/sort)"
    if [[ "${observed_steps}" == "${job}.batch"$'\n'"${job}.extern" ]]; then
      holder_preflight "${job}" "${node}"
      return 0
    fi
    /usr/bin/sleep 1
  done
  fail "owned cache step did not leave holder cleanly: ${job}"
}
for index in 0 1 2 3; do
  holder_preflight "${jobs[$index]}" "${nodes[$index]}"
done

/usr/bin/mkdir -m 700 "${experiment_root}"
/usr/bin/mkdir -m 700 "${experiment_root}/logs" "${experiment_root}/runs"
run_release_runtime root extract \
  --archive "${archive}" --expected-archive-sha256 "${archive_sha}" \
  --manifest "${release_manifest}" --expected-manifest-sha256 "${release_manifest_sha}" \
  --expected-content-revision "${source_revision}" --output-root "${materialized}" \
  >"${experiment_root}/logs/materialization.json" || fail "verified release extraction failed"
materialization_sha="$(seal_shared_evidence_file "${experiment_root}/logs/materialization.json")" || \
  fail "materialization receipt sealing failed"
readonly materialization_sha
[[ "${materialization_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "materialization receipt SHA differs"
[[ -x "${node_runner}" && -f "${auditor}" && -f "${verified_runtime}" && \
  -f "${completion_publisher}" ]] || fail "materialized entrypoint differs"

common_env=(
  ACTION_PRESERVATION_NODE_CONFIRM=run-approved-action-preservation-v2-seed20260818-r1
  ACTION_PRESERVATION_SOURCE_ARCHIVE="${archive}"
  ACTION_PRESERVATION_SOURCE_ARCHIVE_SHA256="${archive_sha}"
  ACTION_PRESERVATION_RELEASE_MANIFEST="${release_manifest}"
  ACTION_PRESERVATION_RELEASE_MANIFEST_SHA256="${release_manifest_sha}"
  ACTION_PRESERVATION_SOURCE_REVISION="${source_revision}"
  ACTION_PRESERVATION_SOURCE_MANIFEST="${source_data_manifest}"
  ACTION_PRESERVATION_SOURCE_MANIFEST_SHA256="${source_data_manifest_sha}"
  ACTION_PRESERVATION_SEED="${canary_seed}"
  ACTION_PRESERVATION_FROZEN_SITE_PACKAGES="${frozen_site_packages}"
  ACTION_PRESERVATION_TORCHRUN_PATH="${torchrun_path}"
  ACTION_PRESERVATION_TORCHRUN_SHA256="${torchrun_sha}"
  ACTION_PRESERVATION_TORCHRUN_SIZE="${torchrun_size}"
)

readonly clean_home=/vast/users/guangyi.chen
node_shell_command=(
  "${root_python}" -I -S -B -c "${runtime_bootstrap_source}"
  "${archive}" "${archive_sha}" "${release_manifest}" "${release_manifest_sha}"
  "${source_revision}" root "${frozen_python}" "${frozen_python_sha}" 31490256
  verified-shell-run --release-root "${materialized}"
  --manifest "${release_manifest}"
  --expected-manifest-sha256 "${release_manifest_sha}"
  --expected-content-revision "${source_revision}"
  --target scripts/auh_run_self_generated_action_preservation_v2.sh
  --expected-bash-sha256 "${bash_sha}" --expected-bash-size "${bash_size}" --
)
run_release_auditor() {
  run_release_runtime frozen verified-run \
    --release-root "${materialized}" --manifest "${release_manifest}" \
    --expected-manifest-sha256 "${release_manifest_sha}" \
    --expected-content-revision "${source_revision}" \
    --target audit_self_generated_action_preservation_v2.py -- "$@"
}
# The cache is a full 4-IID x 5-sigma authority; partial cache canaries are
# intentionally forbidden because they provide no low-sigma coverage proof.
/usr/bin/srun --jobid=136719 --overlap --nodes=1 --ntasks=1 --nodelist=auh7-1b-gpu-306 \
  --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --immediate=5 --kill-on-bad-exit=1 \
  --job-name=apv2-cache-full \
  /usr/bin/env -i HOME="${clean_home}" USER=guangyi.chen LOGNAME=guangyi.chen \
  PATH=/opt/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  SLURM_JOB_ID=136719 ROCR_VISIBLE_DEVICES=0,1,2,3 "${common_env[@]}" \
  ACTION_PRESERVATION_MODE=cache ACTION_PRESERVATION_ARM=v2_onset_all \
  ACTION_PRESERVATION_CACHE="${cache}" ACTION_PRESERVATION_OUTPUT="${cache}" \
  "${node_shell_command[@]}" \
  >"${experiment_root}/logs/cache-full.log" 2>&1 || fail "full v2 cache step failed"
[[ -f "${cache}" && -f "${cache}.receipt.json" && ! -L "${cache}" && ! -L "${cache}.receipt.json" ]] || fail "full v2 cache is absent"
cache_sha="$(seal_shared_evidence_file "${cache}")" || fail "cache sealing failed"
readonly cache_sha
[[ "${cache_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "cache SHA differs"
cache_receipt_sha="$(seal_shared_evidence_file "${cache}.receipt.json")" || \
  fail "cache receipt sealing failed"
readonly cache_receipt_sha
[[ "${cache_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "cache receipt SHA differs"

run_release_auditor \
  validate-cache --cache "${cache}" --expected-cache-sha256 "${cache_sha}" \
  --source-manifest "${source_data_manifest}" --source-manifest-sha256 "${source_data_manifest_sha}" \
  --method-source-revision "${source_revision}" --method-source-archive-sha256 "${archive_sha}" \
  >"${experiment_root}/logs/cache-audit.json" || fail "v2 cache audit failed"
cache_audit_sha="$(seal_shared_evidence_file "${experiment_root}/logs/cache-audit.json")" || \
  fail "v2 cache audit sealing failed"
readonly cache_audit_sha
[[ "${cache_audit_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "v2 cache audit SHA differs"

# Re-establish exclusivity after the long cache build and before all arms.
for index in 0 1 2 3; do
  holder_preflight_after_owned_step "${jobs[$index]}" "${nodes[$index]}"
done

arms=(
  v2_onset_all v2_noop020_all
  v2_func010_all v2_func025_all
  v2_func050_all v2_onset_cross_qo
  v2_func010_cross_qo v2_func025_cross_qo
)
arm_jobs=(136719 136719 136141 136141 136309 136309 136140 136140)
arm_nodes=(
  auh7-1b-gpu-306 auh7-1b-gpu-306
  auh7-1b-gpu-299 auh7-1b-gpu-299
  auh7-1b-gpu-280 auh7-1b-gpu-280
  auh7-1b-gpu-215 auh7-1b-gpu-215
)
arm_groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)
pids=()
for index in 0 1 2 3 4 5 6 7; do
  arm="${arms[$index]}"; job="${arm_jobs[$index]}"; node="${arm_nodes[$index]}"; group="${arm_groups[$index]}"
  output="${experiment_root}/runs/${arm}"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "arm output is not fresh: ${arm}"
  /usr/bin/srun --jobid="${job}" --overlap --nodes=1 --ntasks=1 --nodelist="${node}" \
    --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 --immediate=5 --kill-on-bad-exit=1 \
    --job-name="apv2-${arm}" \
    /usr/bin/env -i HOME="${clean_home}" USER=guangyi.chen LOGNAME=guangyi.chen \
    PATH=/opt/slurm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    SLURM_JOB_ID="${job}" ROCR_VISIBLE_DEVICES="${group}" "${common_env[@]}" \
    ACTION_PRESERVATION_MODE=train ACTION_PRESERVATION_ARM="${arm}" \
    ACTION_PRESERVATION_CACHE="${cache}" ACTION_PRESERVATION_EXPECTED_CACHE_SHA256="${cache_sha}" \
    ACTION_PRESERVATION_OUTPUT="${output}" \
    "${node_shell_command[@]}" \
    >"${experiment_root}/logs/train-${arm}.log" 2>&1 &
  pids+=("$!")
  /usr/bin/printf 'LAUNCHED arm=%s job=%s node=%s gpus=%s pid=%s\n' \
    "${arm}" "${job}" "${node}" "${group}" "${pids[-1]}"
done

status=0
for index in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$index]}"; then
    /usr/bin/printf 'COMPLETE arm=%s\n' "${arms[$index]}"
  else
    /usr/bin/printf 'FAILED arm=%s\n' "${arms[$index]}" >&2
    status=1
  fi
done
(( status == 0 )) || fail "one or more v2 arms failed; no retry was attempted"

# A successful child exit is not sufficient: prove every owned step and GPU
# client has disappeared before auditing or sealing completion.
for index in 0 1 2 3; do
  holder_preflight_after_owned_step "${jobs[$index]}" "${nodes[$index]}"
done

# Every successful arm seals its own checkpoint namespace.  Seal the common
# runs namespace before physical auditing so no checkpoint name can be added.
/usr/bin/chmod 0555 "${experiment_root}/runs"
[[ "$(/usr/bin/stat -c '%a|%u' "${experiment_root}/runs")" == "555|2012" ]] || \
  fail "completed runs root topology differs"

run_release_auditor \
  validate-training --experiment-root "${experiment_root}" --cache-sha256 "${cache_sha}" \
  --source-manifest-sha256 "${source_data_manifest_sha}" --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${archive_sha}" \
  >"${experiment_root}/logs/training-audit.json" || fail "v2 training audit failed"
training_audit_sha="$(seal_shared_evidence_file "${experiment_root}/logs/training-audit.json")" || \
  fail "v2 training audit sealing failed"
readonly training_audit_sha
[[ "${training_audit_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "v2 training audit SHA differs"

# Seal every ordinary log only after its writer is reaped.  The verified
# materialized tree and per-arm checkpoint trees are already sealed by their
# respective create-only producers.
for evidence_log in \
  "${experiment_root}/logs/cache-full.log" \
  "${experiment_root}/logs/train-v2_onset_all.log" \
  "${experiment_root}/logs/train-v2_noop020_all.log" \
  "${experiment_root}/logs/train-v2_func010_all.log" \
  "${experiment_root}/logs/train-v2_func025_all.log" \
  "${experiment_root}/logs/train-v2_func050_all.log" \
  "${experiment_root}/logs/train-v2_onset_cross_qo.log" \
  "${experiment_root}/logs/train-v2_func010_cross_qo.log" \
  "${experiment_root}/logs/train-v2_func025_cross_qo.log"; do
  evidence_log_sha="$(seal_shared_evidence_file "${evidence_log}")" || fail "log sealing failed: ${evidence_log}"
  [[ "${evidence_log_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "log SHA differs: ${evidence_log}"
done
/usr/bin/chmod 0555 "${experiment_root}/logs"

readonly expected_root_entries=$'logs\nmaterialized\nruns\nteacher-cache-preservation-v2-seed20260818-row4-sigma5.pt\nteacher-cache-preservation-v2-seed20260818-row4-sigma5.pt.receipt.json'
observed_root_entries="$(/usr/bin/find "${experiment_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"
[[ "${observed_root_entries}" == "${expected_root_entries}" ]] || fail "precommit experiment root entry closure differs"
[[ -d "${experiment_root}" && ! -L "${experiment_root}" && "$(/usr/bin/stat -c '%a|%u' "${experiment_root}")" == "700|2012" ]] || \
  fail "precommit experiment root topology differs"
for sealed_directory in "${experiment_root}/logs" "${experiment_root}/materialized" "${experiment_root}/runs"; do
  [[ -d "${sealed_directory}" && ! -L "${sealed_directory}" && "$(/usr/bin/stat -c '%a|%u' "${sealed_directory}")" == "555|2012" ]] || \
    fail "sealed experiment directory differs: ${sealed_directory}"
done

# Replay the trusted auditor from captured release bytes after every writer
# has exited and every non-root namespace is sealed.
run_release_auditor \
  validate-cache --cache "${cache}" --expected-cache-sha256 "${cache_sha}" \
  --source-manifest "${source_data_manifest}" --source-manifest-sha256 "${source_data_manifest_sha}" \
  --method-source-revision "${source_revision}" --method-source-archive-sha256 "${archive_sha}" \
  >/dev/null || fail "sealed cache final replay failed"
run_release_auditor \
  validate-training --experiment-root "${experiment_root}" --cache-sha256 "${cache_sha}" \
  --source-manifest-sha256 "${source_data_manifest_sha}" --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${archive_sha}" \
  >/dev/null || fail "sealed training final replay failed"

# The held-FD publisher captures every authority and the complete retained
# experiment tree before its bounded final commit.  Replacing this shell
# guarantees there is no fallible post-marker work.
exec "${root_python}" -I -S -B -c "${runtime_bootstrap_source}" \
  "${archive}" "${archive_sha}" "${release_manifest}" "${release_manifest_sha}" \
  "${source_revision}" frozen "${frozen_python}" "${frozen_python_sha}" 31490256 \
  verified-run --release-root "${materialized}" --manifest "${release_manifest}" \
  --expected-manifest-sha256 "${release_manifest_sha}" \
  --expected-content-revision "${source_revision}" \
  --target action_preservation_completion_publisher_v1.py -- \
  --experiment-root "${experiment_root}" \
  --cache-sha256 "${cache_sha}" --cache-receipt-sha256 "${cache_receipt_sha}" \
  --cache-audit-sha256 "${cache_audit_sha}" --training-audit-sha256 "${training_audit_sha}" \
  --source-archive "${archive}" --source-archive-sha256 "${archive_sha}" \
  --release-manifest "${release_manifest}" --release-manifest-sha256 "${release_manifest_sha}" \
  --controller "${controller}" --controller-sha256 "${controller_sha}" \
  --deployment-envelope "${envelope}" --deployment-envelope-sha256 "${envelope_sha}" \
  --source-data-manifest "${source_data_manifest}" \
  --source-data-manifest-sha256 "${source_data_manifest_sha}" \
  --source-data-manifest-digest "${source_data_manifest_digest}" \
  --source-revision "${source_revision}"
