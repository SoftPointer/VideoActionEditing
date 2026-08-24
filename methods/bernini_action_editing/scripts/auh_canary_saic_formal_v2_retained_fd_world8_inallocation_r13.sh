#!/usr/bin/bash
# Compute-side retained-FD wrapper for one full-8-GPU step in allocation 134936.

set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin LANG=C LC_ALL=C

readonly expected_payload_sha=36438ff41f3f56786cbcaad992759aa05d2a1db338a6cfc83e2d2b5c9c580951
readonly expected_guard_sha=1a38b2ac18f46e818b1596db884b883dff8e0612fcd5b0cf1ed78aca377ac965
readonly expected_runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly expected_archive_sha=3f6a713c762751b06723448b22e627ec6571eae502d7311811005db91812ee7b
readonly expected_python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_validator_sha=3b5e78a0815fbfdee0404293ad800c640133deacc6b1bfafec12a02ca40ef85b
readonly expected_parent=134936
readonly expected_node=auh7-1b-gpu-185
readonly expected_receipt_schema=saic-formal-v2-retained-fd-world8-inallocation-r13-step-launch-v1
readonly stage1_nonce=e38a552392652a2a422615179eab139b97300f3b82d62f52a1989307680ee68c

fail() { echo "[saic-fv2-fd-world8-inallocation-r13] ERROR: $*" >&2; exit 2; }

payload="${SAIC_FV2_FD_CANARY_PAYLOAD:?}"
payload_sha="${SAIC_FV2_FD_CANARY_PAYLOAD_SHA256:?}"
guard="${SAIC_FV2_FD_CANARY_GUARD:?}"
guard_sha="${SAIC_FV2_FD_CANARY_GUARD_SHA256:?}"
runtime="${SAIC_FV2_FD_CANARY_RUNTIME:?}"
runtime_sha="${SAIC_FV2_FD_CANARY_RUNTIME_SHA256:?}"
source_archive="${SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE:?}"
source_archive_sha="${SAIC_FV2_FD_CANARY_SOURCE_ARCHIVE_SHA256:?}"
python_bin="${SAIC_FV2_FD_CANARY_PYTHON:?}"
python_sha="${SAIC_FV2_FD_CANARY_PYTHON_SHA256:?}"
output_parent="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT:?}"
step_receipt="${SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT:?}"
step_receipt_device="${SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_DEVICE:?}"
step_receipt_inode="${SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_INODE:?}"
bootstrap_receipt_fd="${SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_FD_NUMBER:?}"
bootstrap_wrapper_fd="${SAIC_FV2_FD_CANARY_BOOTSTRAP_WRAPPER_FD_NUMBER:?}"
bootstrap_python_fd="${SAIC_FV2_FD_CANARY_BOOTSTRAP_PYTHON_FD_NUMBER:?}"
wrapper_path="${SAIC_FV2_FD_CANARY_WRAPPER:?}"
wrapper_sha="${SAIC_FV2_FD_CANARY_WRAPPER_SHA256:?}"
postflight="${SAIC_FV2_FD_CANARY_POSTFLIGHT:?}"
postflight_sha="${SAIC_FV2_FD_CANARY_POSTFLIGHT_SHA256:?}"
release_manifest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST:?}"
release_manifest_sha="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_SHA256:?}"
release_manifest_digest="${SAIC_FV2_FD_CANARY_RELEASE_MANIFEST_DIGEST:?}"
probe_validator="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR:?}"
probe_validator_sha="${SAIC_FV2_FD_CANARY_PROBE_VALIDATOR_SHA256:?}"
probe_admission="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION:?}"
probe_admission_sha="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_SHA256:?}"
probe_admission_digest="${SAIC_FV2_FD_CANARY_PROBE_ADMISSION_DIGEST:?}"

[[ "${SLURM_JOB_ID:?}" == "${expected_parent}" ]] || fail "parent allocation differs"
[[ "${SLURM_STEP_ID:?}" =~ ^[0-9]+$ ]] || fail "step id differs"
[[ "${SLURM_JOB_NODELIST:?}" == "${expected_node}" && \
   "${SLURMD_NODENAME:?}" == "${expected_node}" ]] || fail "node differs"
[[ "${SLURM_NTASKS:?}" == 1 && "${SLURM_PROCID:?}" == 0 && \
   "${SLURM_LOCALID:?}" == 0 ]] || fail "single-task step topology differs"
job_step_id="${SLURM_JOB_ID}.${SLURM_STEP_ID}"
[[ "${payload_sha}" == "${expected_payload_sha}" && \
   "${guard_sha}" == "${expected_guard_sha}" && \
   "${runtime_sha}" == "${expected_runtime_sha}" && \
   "${source_archive_sha}" == "${expected_archive_sha}" && \
   "${python_sha}" == "${expected_python_sha}" && \
   "${probe_validator_sha}" == "${expected_validator_sha}" ]] || fail "pin differs"
for path_name in payload guard runtime source_archive python_bin output_parent \
  step_receipt wrapper_path postflight release_manifest probe_validator \
  probe_admission; do
  [[ "${!path_name}" == /* ]] || fail "${path_name} is not absolute"
done
for fd_name in bootstrap_receipt_fd bootstrap_wrapper_fd bootstrap_python_fd; do
  [[ "${!fd_name}" =~ ^[0-9]+$ && "${!fd_name}" -ge 3 ]] || fail "${fd_name} differs"
done

# Stage 0 retains all release sources and the output directory.  The wrapper
# and receipt descriptors already came from the compute-side O_NOFOLLOW
# bootstrap and are duplicated, never reopened for execution.
if [[ "${SAIC_FV2_FD_CANARY_INALLOCATION_STAGE:-}" != "${stage1_nonce}" ]]; then
  exec /usr/bin/python3 -I -B - \
    "${bootstrap_wrapper_fd}" "${wrapper_sha}" "${bootstrap_receipt_fd}" \
    "${bootstrap_python_fd}" "${python_sha}" \
    "${step_receipt_device}" "${step_receipt_inode}" "${guard}" "${guard_sha}" \
    "${payload}" "${payload_sha}" "${probe_validator}" "${probe_validator_sha}" \
    "${source_archive}" "${source_archive_sha}" "${output_parent}" \
    "${expected_parent}" "${SLURM_STEP_ID}" "${expected_node}" "${stage1_nonce}" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path

(wrapper_number, wrapper_sha, receipt_number, python_number, python_sha,
 receipt_device, receipt_inode,
 guard_path, guard_sha, payload_path, payload_sha, validator_path, validator_sha,
 archive_path, archive_sha, output_parent, parent, step, node, nonce) = sys.argv[1:]
fds=[]
def read(fd):
    os.lseek(fd,0,os.SEEK_SET); chunks=[]
    while True:
        chunk=os.read(fd,1024*1024)
        if not chunk: break
        chunks.append(chunk)
    os.lseek(fd,0,os.SEEK_SET); return b"".join(chunks)
def duplicate(number, expected_sha, label):
    fd=os.dup(int(number)); fds.append(fd); info=os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
            or stat.S_IMODE(info.st_mode)!=0o444
            or hashlib.sha256(read(fd)).hexdigest()!=expected_sha):
        raise SystemExit(label+" retained fd differs")
    os.set_inheritable(fd,True); return fd
def duplicate_executable(number, expected_sha, label):
    fd=os.dup(int(number)); fds.append(fd); info=os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
            or stat.S_IMODE(info.st_mode)&0o022
            or hashlib.sha256(read(fd)).hexdigest()!=expected_sha):
        raise SystemExit(label+" retained fd differs")
    os.set_inheritable(fd,True); return fd
def open_source(value, expected_sha, label):
    path=Path(value); fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW); fds.append(fd)
    info=os.fstat(fd); leaf=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
            or stat.S_IMODE(info.st_mode)!=0o444 or stat.S_ISLNK(leaf.st_mode)
            or (info.st_dev,info.st_ino)!=(leaf.st_dev,leaf.st_ino)
            or hashlib.sha256(read(fd)).hexdigest()!=expected_sha):
        raise SystemExit(label+" source differs")
    os.set_inheritable(fd,True); return fd
wrapper_fd=duplicate(wrapper_number,wrapper_sha,"wrapper")
python_fd=duplicate_executable(python_number,python_sha,"science Python")
receipt_fd=os.dup(int(receipt_number)); fds.append(receipt_fd); receipt=os.fstat(receipt_fd)
receipt_leaf=Path(os.environ["SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT"]).lstat()
if (not stat.S_ISREG(receipt.st_mode) or receipt.st_nlink!=1
        or stat.S_IMODE(receipt.st_mode)!=0o444
        or receipt.st_dev!=int(receipt_device) or receipt.st_ino!=int(receipt_inode)
        or (receipt.st_dev,receipt.st_ino)!=(receipt_leaf.st_dev,receipt_leaf.st_ino)):
    raise SystemExit("step receipt retained fd differs")
os.set_inheritable(receipt_fd,True)
guard_fd=open_source(guard_path,guard_sha,"guard")
payload_fd=open_source(payload_path,payload_sha,"payload")
validator_fd=open_source(validator_path,validator_sha,"probe validator")
archive_fd=open_source(archive_path,archive_sha,"source archive")
output_fd=os.open(output_parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); fds.append(output_fd)
output=os.fstat(output_fd); output_leaf=Path(output_parent).lstat()
if (not stat.S_ISDIR(output.st_mode) or stat.S_IMODE(output.st_mode)!=0o700
        or stat.S_ISLNK(output_leaf.st_mode)
        or (output.st_dev,output.st_ino)!=(output_leaf.st_dev,output_leaf.st_ino)):
    raise SystemExit("output parent differs")
os.set_inheritable(output_fd,True)
all_fds=(wrapper_fd,receipt_fd,python_fd,guard_fd,payload_fd,validator_fd,archive_fd,output_fd)
if len(set(all_fds))!=8: raise SystemExit("retained fd closure differs")
env=dict(os.environ)
env.update({
 "SAIC_FV2_FD_CANARY_INALLOCATION_STAGE":nonce,
 "SAIC_FV2_FD_CANARY_STAGE0_WRAPPER_FD_NUMBER":str(wrapper_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER":str(receipt_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_PYTHON_FD_NUMBER":str(python_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER":str(guard_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER":str(payload_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER":str(validator_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER":str(archive_fd),
 "SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER":str(output_fd),
 "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE":str(output.st_dev),
 "SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE":str(output.st_ino),
})
os.execve("/usr/bin/bash",["/usr/bin/bash",f"/proc/self/fd/{wrapper_fd}"],env)
PY
  exit 2
fi

wrapper_fd="${SAIC_FV2_FD_CANARY_STAGE0_WRAPPER_FD_NUMBER:?}"
receipt_fd="${SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER:?}"
python_fd="${SAIC_FV2_FD_CANARY_STAGE0_PYTHON_FD_NUMBER:?}"
guard_source_fd="${SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER:?}"
payload_source_fd="${SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER:?}"
validator_fd="${SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER:?}"
archive_fd="${SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER:?}"
output_fd="${SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER:?}"
output_parent_device="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT_DEVICE:?}"
output_parent_inode="${SAIC_FV2_FD_CANARY_OUTPUT_PARENT_INODE:?}"
python_fd_path="/proc/$$/fd/${python_fd}"
[[ "$(sha256sum "${python_fd_path}" | awk '{print $1}')" == "${python_sha}" ]] || \
  fail "retained science Python fd differs"
readonly python_fd_exec_loader='import os,sys;fd=int(sys.argv[1]);os.execve(fd,[sys.argv[2],*sys.argv[3:]],os.environ)'
python_exec=(/usr/bin/python3 -I -B -c "${python_fd_exec_loader}" \
  "${python_fd}" "${python_bin}")
[[ "$(sha256sum "/proc/$$/fd/${wrapper_fd}" | awk '{print $1}')" == "${wrapper_sha}" ]] || \
  fail "executed inallocation wrapper retained fd differs"
[[ "$(stat -Lc '%d:%i:%h:%a' "/proc/$$/fd/${receipt_fd}")" == \
   "${step_receipt_device}:${step_receipt_inode}:1:444" ]] || fail "step receipt fd differs"

step_binding="$("${python_exec[@]}" -I -B - "/proc/$$/fd/${guard_source_fd}" \
  "/proc/$$/fd/${receipt_fd}" "${expected_receipt_schema}" "${expected_parent}" \
  "${SLURM_STEP_ID}" "${job_step_id}" "${expected_node}" <<'PY'
import hashlib,json,sys,types
from pathlib import Path
guard,receipt=map(Path,sys.argv[1:3]); schema,parent,step,job_step,node=sys.argv[3:]
raw_guard=guard.read_bytes(); module=types.ModuleType("inallocation_step_guard")
exec(compile(raw_guard,"retained-guard-v2","exec"),module.__dict__)
value=module._decode_sealed(receipt.read_bytes(),schema_version=schema,exact_fields={
 "schema_version","status","parent_allocation_job_id","step_id","job_step_id",
 "node","exact_srun_argv","exact_srun_argv_digest","release_manifest",
 "output_parent_identity","log_directory_identity",
 "compute_bootstrap_sha256",
 "compute_bootstrap_size_bytes",
 "release_manifest_file_sha256","release_manifest_digest","step_success",
 "parent_job_success","bootstrap_boundary","authority","receipt_digest"})
if (value.get("status")!="compute_step_bootstrap_admitted"
        or value.get("parent_allocation_job_id")!=parent or value.get("step_id")!=step
        or value.get("job_step_id")!=job_step or value.get("node")!=node
        or not isinstance(value.get("output_parent_identity"),str)
        or not isinstance(value.get("log_directory_identity"),str)
        or value.get("step_success") is not None or value.get("parent_job_success") is not None
        or hashlib.sha256(module.canonical_json_bytes(
            value["exact_srun_argv"]
        )).hexdigest()
           != value.get("exact_srun_argv_digest")
        or value.get("bootstrap_boundary")!={
          "receipt_reserved_before_srun":True,"receipt_same_inode":True,
          "receipt_opened_o_nofollow_inside_step":True,
          "wrapper_opened_o_nofollow_inside_step":True,
          "wrapper_executed_from_retained_fd":True,
          "compute_bootstrap_transported_over_srun_stdin":True,
          "compute_bootstrap_stdin_sha256_verified_inside_step":True,
          "compute_bootstrap_pathname_execution":False,
          "compute_bootstrap_interpreter":"/usr/bin/python3",
          "compute_bootstrap_interpreter_trust":"host_os_absolute_path",
          "science_python_opened_o_nofollow_inside_step":True,
          "science_python_retained_fd_prepared_for_wrapper":True,
          "receipt_success_mode":"0444"}
        or value.get("authority")!={"scientific":False,"generation":False,
          "training":False,"publication":False,"formal_job_authorized":False}):
    raise SystemExit("step launch receipt differs")
print(value["receipt_digest"]+":"+value["exact_srun_argv_digest"])
PY
)" || fail "step launch receipt rejected"
[[ "${step_binding}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}$ ]] || fail "step binding differs"
IFS=: read -r step_receipt_digest exact_srun_argv_digest <<<"${step_binding}"

probe_binding="$("${python_exec[@]}" -I -B "/proc/$$/fd/${validator_fd}" \
  --path "${probe_admission}" --sha256 "${probe_admission_sha}" \
  --digest "${probe_admission_digest}")" || fail "probe admission rejected"
[[ "${probe_binding}" == \{*\} ]] || fail "probe admission serialization differs"

output_root="${output_parent}/step-${job_step_id}"
fixture_root="${output_parent}/fd-fixture-step-${job_step_id}"
failure_receipt="${output_parent}/step-${job_step_id}.failure.json"
[[ ! -e "${output_root}" && ! -L "${output_root}" && \
   ! -e "${fixture_root}" && ! -L "${fixture_root}" && \
   ! -e "${failure_receipt}" && ! -L "${failure_receipt}" ]] || fail "fresh step namespace differs"

payload_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${payload_pid:-}" =~ ^[0-9]+$ ]]; then
    kill "${payload_pid}" 2>/dev/null || true; wait "${payload_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -m 0700 -- "${fixture_root}"
payload_logical="${fixture_root}/payload.logical"
guard_logical="${fixture_root}/guard.logical"
cp -- "/proc/$$/fd/${payload_source_fd}" "${payload_logical}"
cp -- "/proc/$$/fd/${guard_source_fd}" "${guard_logical}"
chmod 0444 -- "${payload_logical}" "${guard_logical}"
exec {payload_fd}<"${payload_logical}"
exec {guard_fd}<"${guard_logical}"
payload_fd_path="/proc/$$/fd/${payload_fd}"
guard_fd_path="/proc/$$/fd/${guard_fd}"
payload_retained="${fixture_root}/payload.original"
guard_retained="${fixture_root}/guard.original"
mv -- "${payload_logical}" "${payload_retained}"
mv -- "${guard_logical}" "${guard_retained}"
printf '%s\n' '#!/usr/bin/bash' 'exit 97' >"${payload_logical}"
printf '%s\n' 'raise SystemExit(98)' >"${guard_logical}"
chmod 0444 -- "${payload_logical}" "${guard_logical}"
[[ "$(sha256sum "${payload_fd_path}" | awk '{print $1}')" == "${payload_sha}" && \
   "$(sha256sum "${guard_fd_path}" | awk '{print $1}')" == "${guard_sha}" && \
   "$(sha256sum "${payload_logical}" | awk '{print $1}')" != "${payload_sha}" && \
   "$(sha256sum "${guard_logical}" | awk '{print $1}')" != "${guard_sha}" ]] || \
  fail "retained/decoy fixture differs"

export SAIC_FV2_FD_CANARY_GUARD_FD_PATH="${guard_fd_path}"
export SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH="${payload_fd_path}"
export SAIC_FV2_FD_CANARY_GUARD_RETAINED_LEAF="${guard_retained}"
export SAIC_FV2_FD_CANARY_GUARD_DECOY_LEAF="${guard_logical}"
export SAIC_FV2_FD_CANARY_PAYLOAD_RETAINED_LEAF="${payload_retained}"
export SAIC_FV2_FD_CANARY_PAYLOAD_DECOY_LEAF="${payload_logical}"
export SAIC_FV2_FD_CANARY_SCRATCH_PARENT="${SLURM_TMPDIR:-/tmp}"
export SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_DEVICE="${output_parent_device}"
export SAIC_FV2_FD_CANARY_COMPUTE_OUTPUT_PARENT_INODE="${output_parent_inode}"
export SAIC_FV2_FD_CANARY_COMPUTE_STEP_LAUNCH_RECEIPT_DEVICE="${step_receipt_device}"
export SAIC_FV2_FD_CANARY_COMPUTE_STEP_LAUNCH_RECEIPT_INODE="${step_receipt_inode}"
export SAIC_FV2_FD_CANARY_STEP_LAUNCH_RECEIPT_FD_NUMBER="${receipt_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_WRAPPER_FD_NUMBER="${wrapper_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_GUARD_FD_NUMBER="${guard_source_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_PAYLOAD_FD_NUMBER="${payload_source_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_PROBE_VALIDATOR_FD_NUMBER="${validator_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_SOURCE_ARCHIVE_FD_NUMBER="${archive_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_RECEIPT_FD_NUMBER="${receipt_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_PYTHON_FD_NUMBER="${python_fd}"
export SAIC_FV2_FD_CANARY_STAGE0_OUTPUT_FD_NUMBER="${output_fd}"
export SAIC_FV2_FD_CANARY_PROBE_ADMISSION_BINDING="${probe_binding}"
export SAIC_FV2_FD_CANARY_STEP_RECEIPT_DIGEST="${step_receipt_digest}"
export SAIC_FV2_FD_CANARY_EXACT_SRUN_ARGV_DIGEST="${exact_srun_argv_digest}"
export SAIC_FV2_FD_CANARY_PYTHON_FD_EXEC_LOADER="${python_fd_exec_loader}"
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0

SAIC_FV2_FD_CANARY_PAYLOAD_FD_NUMBER="${payload_fd}" \
SAIC_FV2_FD_CANARY_GUARD_FD_NUMBER="${guard_fd}" \
/usr/bin/bash -c '
  set -Eeuo pipefail
  p="${SAIC_FV2_FD_CANARY_PAYLOAD_FD_NUMBER:?}"
  g="${SAIC_FV2_FD_CANARY_GUARD_FD_NUMBER:?}"
  export SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH="/proc/$$/fd/${p}"
  export SAIC_FV2_FD_CANARY_GUARD_FD_PATH="/proc/$$/fd/${g}"
  exec /usr/bin/bash "${SAIC_FV2_FD_CANARY_PAYLOAD_FD_PATH}"
' & payload_pid=$!
payload_status=0
wait "${payload_pid}" || payload_status=$?
payload_pid=""
[[ "${payload_status}" == 0 ]] || exit "${payload_status}"
[[ -f "${output_root}/operational-evidence.json" && \
   ! -L "${output_root}/operational-evidence.json" ]] || fail "operational evidence missing"
readonly pass_sentinel="SAIC_FV2_FD_WORLD8_INALLOCATION_R13_PASS ${job_step_id}"
printf '%s\n' "${pass_sentinel}"
trap - EXIT TERM INT
exit 0
