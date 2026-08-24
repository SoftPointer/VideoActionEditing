#!/usr/bin/bash
# Zero-science compute-node probe for Bash dynamic retained descriptors.
#SBATCH --job-name=saic-bash-fd-probe1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:05:00

set -Eeuo pipefail
umask 077
export PATH=/usr/bin:/bin
export LANG=C
export LC_ALL=C

readonly bash_path=/usr/bin/bash
readonly expected_python=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly authority_json='{"formal_job_authorized":false,"generation":false,"publication":false,"scientific":false,"training":false}'

# Child payload, reached only through the descriptor inherited from the parent.
if [[ "${SAIC_BASH_FD_PROBE_PAYLOAD_MODE:-}" == 1 ]]; then
  fd="${SAIC_BASH_FD_PROBE_FD:?}"
  original="${SAIC_BASH_FD_PROBE_ORIGINAL:?}"
  decoy="${SAIC_BASH_FD_PROBE_DECOY:?}"
  expected="${SAIC_BASH_FD_PROBE_EXPECTED_SHA:?}"
  inherited="/proc/$$/fd/${fd}"
  [[ "$0" == "${inherited}" && -r "${inherited}" && \
     "$(sha256sum "$0" | awk '{print $1}')" == "${expected}" && \
     "$(stat -Lc '%d:%i' -- "$0")" == "$(stat -Lc '%d:%i' -- "${original}")" && \
     "$(stat -Lc '%d:%i' -- "$0")" != "$(stat -Lc '%d:%i' -- "${decoy}")" && \
     "$(sha256sum "${decoy}" | awk '{print $1}')" != "${expected}" ]] || exit 98
  mkdir -m 0700 -- "$(dirname "${SAIC_BASH_FD_PROBE_EVIDENCE}")"
  child_identity="$(stat -Lc '%d:%i:%h:%a:%s' -- "${inherited}")"
  original_identity="$(stat -Lc '%d:%i:%h:%a:%s' -- "${original}")"
  decoy_identity="$(stat -Lc '%d:%i:%h:%a:%s' -- "${decoy}")"
  child_sha="$(sha256sum "${inherited}" | awk '{print $1}')"
  decoy_sha="$(sha256sum "${decoy}" | awk '{print $1}')"
  "${expected_python}" -B - "${SAIC_BASH_FD_PROBE_EVIDENCE}" "${SLURM_JOB_ID}" \
    "${SAIC_BASH_FD_PROBE_BASH_SHA}" "${SAIC_BASH_FD_PROBE_BASH_VERSION_STDOUT_SHA}" \
    "${SAIC_BASH_FD_PROBE_BASH_VERSION_FIRST_LINE}" "${SAIC_BASH_FD_PROBE_BASH_IDENTITY}" \
    "${SAIC_BASH_FD_PROBE_BASH_VERSION_SIZE}" "${inherited}" "${original}" "${decoy}" \
    "${SAIC_BASH_FD_PROBE_PARENT_IDENTITY}" "${child_identity}" "${original_identity}" \
    "${decoy_identity}" "${child_sha}" "${decoy_sha}" "${SAIC_BASH_FD_PROBE_DECOY_EXIT}" \
    "${SAIC_BASH_FD_PROBE_SUBMISSION_FILE_SHA}" "${SAIC_BASH_FD_PROBE_SUBMISSION_DIGEST}" \
    "${SAIC_BASH_FD_PROBE_SUBMISSION_PATH}" "${SAIC_BASH_FD_PROBE_SUBMISSION_IDENTITY}" \
    "${SAIC_BASH_FD_PROBE_OUTPUT_IDENTITY}" \
    "${SAIC_BASH_FD_PROBE_WRAPPER_EVIDENCE_SHA}" \
    "${SAIC_BASH_FD_PROBE_MANIFEST_EVIDENCE_SHA}" \
    "${SAIC_BASH_FD_PROBE_MANIFEST_EVIDENCE_DIGEST}" \
    "${SAIC_BASH_FD_PROBE_POSTFLIGHT_EVIDENCE_SHA}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
out=Path(sys.argv[1]); (job,bsha,vsha,first,bidentity,vsize,fd,original,decoy,parent_id,child_id,original_id,decoy_id,child_sha,decoy_sha,decoy_exit,submission_sha,submission_digest,submission_path,submission_identity,output_identity,wrapper_sha,manifest_sha,manifest_digest,postflight_sha)=sys.argv[2:]
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
core={'schema_version':'saic-compute-bash-retained-fd-probe-evidence-v1','status':'job_completed_awaiting_external_slurm_admission','slurm_job_id':job,'job_success':None,'slurm_terminal_verified':False,'compute_bash':{'path':'/usr/bin/bash','identity':bidentity,'sha256':bsha,'version_stdout_sha256':vsha,'version_stdout_size':int(vsize),'version_first_line':first,'brace_fd_redirection_supported':True,'retained_fd_survives_bash_script_handoff':True,'varredir_close_option_required':False},'retained_fd':{'child_proc_fd_path':fd,'original_retained_path':original,'wrong_decoy_path':decoy,'parent_fd_identity':parent_id,'child_fd_identity':child_id,'original_retained_identity':original_id,'wrong_decoy_identity':decoy_id,'retained_payload_sha256':child_sha,'wrong_decoy_sha256':decoy_sha,'logical_leaf_replaced_after_open':True,'parent_and_child_inode_equal':parent_id==child_id==original_id,'wrong_decoy_inode_differs':decoy_id.split(':',2)[:2]!=child_id.split(':',2)[:2],'wrong_decoy_sha_differs':decoy_sha!=child_sha,'wrong_decoy_exit_code':int(decoy_exit)},'submission_binding':{'submission_receipt_path':submission_path,'submission_receipt_file_sha256':submission_sha,'submission_receipt_digest':submission_digest,'submission_receipt_identity':submission_identity,'output_parent_identity':output_identity,'release_manifest_file_sha256':manifest_sha,'release_manifest_digest':manifest_digest,'postflight_sha256':postflight_sha},'wrapper_sha256':wrapper_sha,'scientific_generation_entered':False,'authority':{'scientific':False,'generation':False,'training':False,'publication':False,'formal_job_authorized':False}}
if not (core['retained_fd']['parent_and_child_inode_equal'] and core['retained_fd']['wrong_decoy_inode_differs'] and core['retained_fd']['wrong_decoy_sha_differs'] and int(decoy_exit)==97): raise SystemExit('retained FD evidence identities differ')
value={**core,'receipt_digest':hashlib.sha256(canonical(core)).hexdigest()}; raw=canonical(value)+b'\n'
fdout=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
view=memoryview(raw)
while view:
 wrote=os.write(fdout,view)
 if wrote<=0: raise SystemExit('evidence write stalled')
 view=view[wrote:]
os.fsync(fdout); os.fchmod(fdout,0o444); os.fsync(fdout); os.close(fdout)
parent=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); os.fsync(parent); os.close(parent)
PY
  exit 0
fi

wrapper="${SAIC_BASH_FD_PROBE_WRAPPER:?}"
wrapper_sha="${SAIC_BASH_FD_PROBE_WRAPPER_SHA256:?}"
python_bin="${SAIC_BASH_FD_PROBE_PYTHON:?}"
python_sha="${SAIC_BASH_FD_PROBE_PYTHON_SHA256:?}"
output_parent="${SAIC_BASH_FD_PROBE_OUTPUT_PARENT:?}"
output_device="${SAIC_BASH_FD_PROBE_OUTPUT_DEVICE:?}"
output_inode="${SAIC_BASH_FD_PROBE_OUTPUT_INODE:?}"
submission="${SAIC_BASH_FD_PROBE_SUBMISSION:?}"
submission_device="${SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE:?}"
submission_inode="${SAIC_BASH_FD_PROBE_SUBMISSION_INODE:?}"
postflight="${SAIC_BASH_FD_PROBE_POSTFLIGHT:?}"
postflight_sha="${SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256:?}"
release_manifest="${SAIC_BASH_FD_PROBE_RELEASE_MANIFEST:?}"
release_manifest_sha="${SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256:?}"
release_manifest_digest="${SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST:?}"

fail() { echo "[saic-bash-fd-probe] ERROR: $*" >&2; exit 2; }
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "job ID differs"
for item in wrapper python_bin output_parent submission postflight release_manifest; do
  [[ "${!item}" == /* ]] || fail "${item} is not absolute"
done
for item in wrapper_sha python_sha postflight_sha release_manifest_sha release_manifest_digest; do
  [[ "${!item}" =~ ^[0-9a-f]{64}$ ]] || fail "${item} differs"
done
[[ "${python_bin}" == "${expected_python}" ]] || fail "Python path differs"
owner_uid="$(id -u)"
[[ -d "${output_parent}" && ! -L "${output_parent}" && \
   "$(stat -c '%d:%i:%u:%a' -- "${output_parent}")" == "${output_device}:${output_inode}:${owner_uid}:700" ]] || \
  fail "output parent identity differs"
submission_identity="$(stat -c '%d:%i:%h:%a' -- "${submission}" 2>/dev/null || true)"
[[ -f "${submission}" && ! -L "${submission}" && \
   ( "${submission_identity}" == "${submission_device}:${submission_inode}:1:600" || \
     "${submission_identity}" == "${submission_device}:${submission_inode}:1:444" ) ]] || \
  fail "submission identity differs before wait"
[[ -f "${wrapper}" && ! -L "${wrapper}" && "$(stat -c '%u:%h:%a' -- "${wrapper}")" == "${owner_uid}:1:444" && \
   "$(sha256sum "${wrapper}" | awk '{print $1}')" == "${wrapper_sha}" ]] || fail "logical wrapper differs"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${wrapper_sha}" ]] || fail "executed spooled wrapper differs"
[[ -f "${postflight}" && ! -L "${postflight}" && "$(stat -c '%u:%h:%a' -- "${postflight}")" == "${owner_uid}:1:444" && \
   "$(sha256sum "${postflight}" | awk '{print $1}')" == "${postflight_sha}" ]] || fail "postflight differs"
[[ -f "${release_manifest}" && ! -L "${release_manifest}" && \
   "$(stat -c '%u:%h:%a' -- "${release_manifest}")" == "${owner_uid}:1:444" && \
   "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || \
  fail "release manifest differs"
[[ -f "${python_bin}" && -x "${python_bin}" && ! -L "${python_bin}" && \
   "$(stat -c '%h' -- "${python_bin}")" == 1 && \
   "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "Python differs"

failure="${output_parent}/job-${SLURM_JOB_ID}.failure.json"
evidence="${output_parent}/job-${SLURM_JOB_ID}/operational-evidence.json"
fixture="${output_parent}/fd-fixture-job-${SLURM_JOB_ID}"
failure_stage=pre_retained_fixture
publish_failure() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${status}" != 0 && ! -e "${failure}" && ! -L "${failure}" ]]; then
    "${python_bin}" -B - "${failure}" "${SLURM_JOB_ID}" "${status}" "${failure_stage}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); job,status,stage=sys.argv[2:]
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
core={'schema_version':'saic-compute-bash-retained-fd-probe-failure-v1','status':'job_failed_before_external_admission','slurm_job_id':job,'exit_status':int(status),'failure_stage':stage,'job_success':None,'slurm_terminal_verified':False,'scientific_generation_entered':False,'authority':{'scientific':False,'generation':False,'training':False,'publication':False,'formal_job_authorized':False}}
value={**core,'receipt_digest':hashlib.sha256(canonical(core)).hexdigest()}; raw=canonical(value)+b'\n'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
view=memoryview(raw)
while view:
    wrote=os.write(fd,view)
    if wrote<=0: raise SystemExit('failure write stalled')
    view=view[wrote:]
os.fsync(fd); os.fchmod(fd,0o444); os.fsync(fd); os.close(fd)
parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); os.fsync(parent); os.close(parent)
PY
  fi
  exit "${status}"
}
trap publish_failure EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Admit only this job's terminal same-inode submission receipt.  The bounded
# wait closes the legitimate fast-start window while rejecting replacement.
submission_binding="$("${python_bin}" -B - "${submission}" "${SLURM_JOB_ID}" "${wrapper}" "${wrapper_sha}" \
  "${python_bin}" "${python_sha}" "${output_parent}" "${output_device}" "${output_inode}" \
  "${submission_device}" "${submission_inode}" "${postflight}" "${postflight_sha}" \
  "${release_manifest}" "${release_manifest_sha}" "${release_manifest_digest}" <<'PY'
import hashlib,json,os,stat,sys,time
from pathlib import Path
(path,job,wrapper,wrapper_sha,python,python_sha,output,odev,oino,rdev,rino,postflight,postflight_sha,manifest,manifest_sha,manifest_digest)=sys.argv[1:]
path=Path(path); expected_identity=(int(rdev),int(rino))
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
deadline=time.monotonic()+120
while True:
    s=path.lstat()
    if (s.st_dev,s.st_ino)!=expected_identity or not stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode) or s.st_nlink!=1: raise SystemExit('submission inode replaced')
    if stat.S_IMODE(s.st_mode)==0o444: break
    if stat.S_IMODE(s.st_mode)!=0o600 or time.monotonic()>=deadline: raise SystemExit('submission did not become ready')
    time.sleep(.05)
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW); before=os.fstat(fd); chunks=[]
while True:
    chunk=os.read(fd,1024*1024)
    if not chunk: break
    chunks.append(chunk)
after=os.fstat(fd); leaf=path.lstat(); os.close(fd); raw=b''.join(chunks)
if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 or (leaf.st_dev,leaf.st_ino,leaf.st_size,leaf.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 or before.st_uid!=os.getuid() or stat.S_IMODE(before.st_mode)!=0o444 or before.st_nlink!=1): raise SystemExit('submission changed during retained read')
v=json.loads(raw.decode('ascii')); expected_fields={'schema_version','status','submission_success','job_success','submitted_job','request','submission_boundary','inputs','outputs','authority','receipt_digest'}
if not isinstance(v,dict) or set(v)!=expected_fields or v.get('schema_version')!='saic-compute-bash-retained-fd-probe-submission-v1': raise SystemExit('submission schema differs')
u=dict(v); claimed=u.pop('receipt_digest',None)
if raw!=canonical(v)+b'\n' or claimed!=hashlib.sha256(canonical(u)).hexdigest(): raise SystemExit('submission seal differs')
b=v.get('submission_boundary',{}); submitted=v.get('submitted_job',{})
authority={'scientific':False,'generation':False,'training':False,'publication':False,'formal_job_authorized':False}
exports=['SAIC_BASH_FD_PROBE_WRAPPER','SAIC_BASH_FD_PROBE_WRAPPER_SHA256','SAIC_BASH_FD_PROBE_PYTHON','SAIC_BASH_FD_PROBE_PYTHON_SHA256','SAIC_BASH_FD_PROBE_OUTPUT_PARENT','SAIC_BASH_FD_PROBE_OUTPUT_DEVICE','SAIC_BASH_FD_PROBE_OUTPUT_INODE','SAIC_BASH_FD_PROBE_SUBMISSION','SAIC_BASH_FD_PROBE_SUBMISSION_DEVICE','SAIC_BASH_FD_PROBE_SUBMISSION_INODE','SAIC_BASH_FD_PROBE_POSTFLIGHT','SAIC_BASH_FD_PROBE_POSTFLIGHT_SHA256','SAIC_BASH_FD_PROBE_RELEASE_MANIFEST','SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_SHA256','SAIC_BASH_FD_PROBE_RELEASE_MANIFEST_DIGEST']
retained_submit_fd=b.get('retained_wrapper_fd') if isinstance(b,dict) else None
expected_boundary={'environment_replaced':True,'exact_job_export_names':exports,'export_all':False,'reservation_created_before_sbatch':True,'same_inode_retained':True,'launcher_submitted_from_retained_fd':True,'retained_wrapper_fd':retained_submit_fd,'reservation_device':expected_identity[0],'reservation_inode':expected_identity[1],'output_parent_device':int(odev),'output_parent_inode':int(oino),'success_mode':'0444'}
manifest_path=Path(manifest); manifest_before=manifest_path.lstat(); manifest_raw=manifest_path.read_bytes(); manifest_after=manifest_path.lstat()
manifest_value=json.loads(manifest_raw.decode('ascii')); manifest_unsigned=dict(manifest_value); manifest_claimed=manifest_unsigned.pop('receipt_digest',None)
release=Path(wrapper).parent; release_info=release.lstat()
if ((manifest_before.st_dev,manifest_before.st_ino,manifest_before.st_size,manifest_before.st_mtime_ns)!=(manifest_after.st_dev,manifest_after.st_ino,manifest_after.st_size,manifest_after.st_mtime_ns)
 or not stat.S_ISDIR(release_info.st_mode) or stat.S_ISLNK(release_info.st_mode) or release_info.st_uid!=os.getuid() or stat.S_IMODE(release_info.st_mode)!=0o555
 or not stat.S_ISREG(manifest_before.st_mode) or stat.S_ISLNK(manifest_before.st_mode) or manifest_before.st_uid!=os.getuid() or manifest_before.st_nlink!=1 or stat.S_IMODE(manifest_before.st_mode)!=0o444
 or hashlib.sha256(manifest_raw).hexdigest()!=manifest_sha or manifest_raw!=canonical(manifest_value)+b'\n' or manifest_claimed!=manifest_digest or manifest_claimed!=hashlib.sha256(canonical(manifest_unsigned)).hexdigest()
 or set(manifest_value)!={'schema_version','status','stem','release_root','output_parent','wrapper','postflight','executables','authority','receipt_digest'}
 or manifest_value.get('schema_version')!='saic-compute-bash-retained-fd-probe-release-v1' or manifest_value.get('status')!='sealed_before_submission'
 or manifest_value.get('stem')!=release.name or manifest_value.get('release_root')!=str(release) or manifest_value.get('output_parent')!=output
 or manifest_value.get('wrapper')!={'path':wrapper,'sha256':wrapper_sha} or manifest_value.get('postflight')!={'path':postflight,'sha256':postflight_sha}
 or manifest_value.get('executables')!={'python':python,'python_sha256':python_sha,'sacct':'/usr/bin/sacct','sacct_sha256':'fadcd62c4a3b28e3a185c8eacf23691e0bd208839aee40c82076fa9364e84f9e'}
 or manifest_value.get('authority')!=authority or set(release.iterdir())!={Path(wrapper),Path(postflight),manifest_path}): raise SystemExit('release manifest binding differs')
cluster=submitted.get('cluster') if isinstance(submitted,dict) else None
expected_stdout=(job+((';'+cluster) if cluster is not None else '')+'\n').encode('ascii') if isinstance(cluster,(str,type(None))) else b''
if (v.get('status')!='submitted' or v.get('submission_success') is not True or v.get('job_success') is not None
 or not isinstance(submitted,dict) or set(submitted)!={'job_id','cluster','stdout_sha256','stderr_sha256'} or submitted.get('job_id')!=job
 or (submitted.get('cluster') is not None and (not isinstance(submitted['cluster'],str) or not submitted['cluster'] or '\n' in submitted['cluster'] or ';' in submitted['cluster']))
 or submitted.get('stdout_sha256')!=hashlib.sha256(expected_stdout).hexdigest() or submitted.get('stderr_sha256')!=hashlib.sha256(b'').hexdigest()
 or type(retained_submit_fd) is not int or retained_submit_fd<3
 or v.get('request')!={'job_name':'saic-bash-fd-probe1','partition':'faculty','qos':'bgqos','nodes':1,'ntasks':1,'cpus_per_task':4,'memory':'8G','walltime':'00:05:00','gpu_resource_requested':'gpu:mi210:1','hold':False,'dependency':None,'scientific_generation':False}
 or v.get('submission_boundary')!=expected_boundary
 or v.get('inputs')!={'wrapper':wrapper,'wrapper_sha256':wrapper_sha,'python':python,'python_sha256':python_sha,'postflight':postflight,'postflight_sha256':postflight_sha,'release_manifest':manifest,'release_manifest_file_sha256':manifest_sha,'release_manifest_digest':manifest_digest}
 or v.get('outputs')!={'output_parent':output,'job_output_root':str(Path(output)/f'job-{job}'),'submission_receipt':str(path)}
 or v.get('authority')!=authority): raise SystemExit('own submission binding differs')
print(hashlib.sha256(raw).hexdigest()+':'+claimed+':'+str(before.st_dev)+':'+str(before.st_ino))
PY
)"
[[ "${submission_binding}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}:[0-9]+:[1-9][0-9]*$ ]] || \
  fail "own submission binding output differs"
submission_file_sha="${submission_binding%%:*}"
submission_tail="${submission_binding#*:}"
submission_digest="${submission_tail%%:*}"
submission_identity_bound="${submission_tail#*:}"
failure_stage=own_submission_admitted
[[ "${submission_identity_bound}" == "${submission_device}:${submission_inode}" && \
   "$(find "${output_parent}" -mindepth 1 -maxdepth 1 -printf '%f\n')" == submission-receipt.json ]] || \
  fail "pre-fixture output closure differs"

# The descriptor is opened by the trusted spooled wrapper before replacement.
mkdir -m 0700 -- "${fixture}"
logical="${fixture}/payload.logical"
retained="${fixture}/payload.retained"
cp -- "${wrapper}" "${logical}"
chmod 0444 -- "${logical}"

# The current interpreter—not merely a later child—is the binary under test
# for Bash brace-FD redirection.  Bind it before allocating the descriptor.
[[ "${BASH}" == "${bash_path}" && "$(realpath -e -- "/proc/$$/exe")" == "${bash_path}" ]] || \
  fail "current Bash interpreter differs"
bash_info="$(stat -Lc '%F:%d:%i:%u:%h:%a:%s' -- "${bash_path}")"
[[ "${bash_info}" =~ ^regular\ file:[0-9]+:[1-9][0-9]*:0:1:[0-9]+:[1-9][0-9]*$ ]] || fail "compute Bash identity differs"
[[ "$(realpath -e -- "${bash_path}")" == "${bash_path}" ]] || fail "compute Bash path differs"
bash_mode="$(stat -Lc '%a' -- "${bash_path}")"
(( (8#${bash_mode} & 8#022) == 0 )) || fail "compute Bash is group/world writable"
[[ "${bash_info##*:}" =~ ^[1-9][0-9]*$ ]] || fail "compute Bash size differs"
bash_sha="$(sha256sum "${bash_path}" | awk '{print $1}')"
bash_version_file="${fixture}/bash-version.stdout"
"${bash_path}" --version >"${bash_version_file}"
chmod 0444 -- "${bash_version_file}"
bash_version_stdout_sha="$(sha256sum "${bash_version_file}" | awk '{print $1}')"
bash_version_size="$(stat -c '%s' -- "${bash_version_file}")"
IFS= read -r bash_version_first_line <"${bash_version_file}"

exec {payload_fd}<"${logical}"
payload_fd_path="/proc/$$/fd/${payload_fd}"
[[ "$(stat -Lc '%d:%i' -- "${logical}")" == "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" && \
   "$(sha256sum "${payload_fd_path}" | awk '{print $1}')" == "${wrapper_sha}" ]] || fail "parent FD differs"
mv -- "${logical}" "${retained}"
printf '#!/usr/bin/env bash\nexit 97\n' >"${logical}"
chmod 0444 -- "${logical}"
[[ "$(sha256sum "${logical}" | awk '{print $1}')" != "${wrapper_sha}" && \
   "$(stat -Lc '%d:%i' -- "${retained}")" == "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" && \
   "$(stat -Lc '%d:%i' -- "${logical}")" != "$(stat -Lc '%d:%i' -- "${payload_fd_path}")" ]] || fail "replacement fixture differs"

failure_stage=child_payload_running
export SAIC_BASH_FD_PROBE_PAYLOAD_MODE=1
export SAIC_BASH_FD_PROBE_FD="${payload_fd}"
export SAIC_BASH_FD_PROBE_ORIGINAL="${retained}"
export SAIC_BASH_FD_PROBE_DECOY="${logical}"
export SAIC_BASH_FD_PROBE_PARENT_IDENTITY="$(stat -Lc '%d:%i:%h:%a:%s' -- "${payload_fd_path}")"
export SAIC_BASH_FD_PROBE_EXPECTED_SHA="${wrapper_sha}"
export SAIC_BASH_FD_PROBE_EVIDENCE="${evidence}"
export SAIC_BASH_FD_PROBE_BASH_SHA="${bash_sha}"
export SAIC_BASH_FD_PROBE_BASH_IDENTITY="${bash_info}"
export SAIC_BASH_FD_PROBE_BASH_VERSION_STDOUT_SHA="${bash_version_stdout_sha}"
export SAIC_BASH_FD_PROBE_BASH_VERSION_SIZE="${bash_version_size}"
export SAIC_BASH_FD_PROBE_BASH_VERSION_FIRST_LINE="${bash_version_first_line}"
export SAIC_BASH_FD_PROBE_SUBMISSION_FILE_SHA="${submission_file_sha}"
export SAIC_BASH_FD_PROBE_SUBMISSION_DIGEST="${submission_digest}"
export SAIC_BASH_FD_PROBE_SUBMISSION_PATH="${submission}"
export SAIC_BASH_FD_PROBE_SUBMISSION_IDENTITY="${submission_identity_bound}"
export SAIC_BASH_FD_PROBE_OUTPUT_IDENTITY="${output_device}:${output_inode}"
export SAIC_BASH_FD_PROBE_WRAPPER_EVIDENCE_SHA="${wrapper_sha}"
export SAIC_BASH_FD_PROBE_MANIFEST_EVIDENCE_SHA="${release_manifest_sha}"
export SAIC_BASH_FD_PROBE_MANIFEST_EVIDENCE_DIGEST="${release_manifest_digest}"
export SAIC_BASH_FD_PROBE_POSTFLIGHT_EVIDENCE_SHA="${postflight_sha}"
set +e
"${bash_path}" "${logical}"
decoy_exit=$?
set -e
[[ "${decoy_exit}" == 97 ]] || fail "wrong decoy execution did not return 97"
export SAIC_BASH_FD_PROBE_DECOY_EXIT="${decoy_exit}"
"${bash_path}" -c 'child_fd="/proc/$$/fd/${SAIC_BASH_FD_PROBE_FD}"; [[ -r "${child_fd}" ]] || exit 99; exec /usr/bin/bash "${child_fd}"'
failure_stage=child_payload_completed
[[ -f "${evidence}" && ! -L "${evidence}" && "$(stat -c '%h:%a' -- "${evidence}")" == 1:444 ]] || fail "evidence missing"
"${python_bin}" -B - "${evidence}" "${SLURM_JOB_ID}" "${wrapper_sha}" \
  "${submission}" "${submission_file_sha}" "${submission_digest}" \
  "${submission_identity_bound}" "${output_device}:${output_inode}" \
  "${release_manifest_sha}" "${release_manifest_digest}" "${postflight_sha}" \
  "${bash_sha}" "${bash_info}" "${bash_version_stdout_sha}" "${bash_version_size}" \
  "${bash_version_first_line}" "${retained}" "${logical}" \
  "$(stat -Lc '%d:%i:%h:%a:%s' -- "${payload_fd_path}")" "${payload_fd}" <<'PY'
import hashlib,json,re,stat,sys
from pathlib import Path
(p,job,wrapper_sha,submission,submission_sha,submission_digest,submission_identity,output_identity,manifest_sha,manifest_digest,postflight_sha,bash_sha,bash_identity,version_sha,version_size,version_first,retained,decoy,parent_identity,fd_number)=sys.argv[1:]
p=Path(p); retained=Path(retained); decoy=Path(decoy)
before=p.lstat(); raw=p.read_bytes(); after=p.lstat()
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns): raise SystemExit('evidence changed during read')
v=json.loads(raw.decode('ascii')); u=dict(v); claimed=u.pop('receipt_digest',None)
canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
expected={'schema_version','status','slurm_job_id','job_success','slurm_terminal_verified','compute_bash','retained_fd','submission_binding','wrapper_sha256','scientific_generation_entered','authority','receipt_digest'}
authority={'scientific':False,'generation':False,'training':False,'publication':False,'formal_job_authorized':False}
identity=lambda path:(lambda s:f'{s.st_dev}:{s.st_ino}:{s.st_nlink}:{stat.S_IMODE(s.st_mode):o}:{s.st_size}')(path.lstat())
retained_sha=hashlib.sha256(retained.read_bytes()).hexdigest(); decoy_sha=hashlib.sha256(decoy.read_bytes()).hexdigest()
expected_bash={'path':'/usr/bin/bash','identity':bash_identity,'sha256':bash_sha,'version_stdout_sha256':version_sha,'version_stdout_size':int(version_size),'version_first_line':version_first,'brace_fd_redirection_supported':True,'retained_fd_survives_bash_script_handoff':True,'varredir_close_option_required':False}
expected_binding={'submission_receipt_path':submission,'submission_receipt_file_sha256':submission_sha,'submission_receipt_digest':submission_digest,'submission_receipt_identity':submission_identity,'output_parent_identity':output_identity,'release_manifest_file_sha256':manifest_sha,'release_manifest_digest':manifest_digest,'postflight_sha256':postflight_sha}
fd=v.get('retained_fd',{}); child_path=fd.get('child_proc_fd_path') if isinstance(fd,dict) else None
expected_fd={'child_proc_fd_path':child_path,'original_retained_path':str(retained),'wrong_decoy_path':str(decoy),'parent_fd_identity':parent_identity,'child_fd_identity':parent_identity,'original_retained_identity':identity(retained),'wrong_decoy_identity':identity(decoy),'retained_payload_sha256':retained_sha,'wrong_decoy_sha256':decoy_sha,'logical_leaf_replaced_after_open':True,'parent_and_child_inode_equal':True,'wrong_decoy_inode_differs':True,'wrong_decoy_sha_differs':True,'wrong_decoy_exit_code':97}
if (set(v)!=expected or raw!=canonical(v)+b'\n' or claimed!=hashlib.sha256(canonical(u)).hexdigest()
 or v.get('schema_version')!='saic-compute-bash-retained-fd-probe-evidence-v1' or v.get('slurm_job_id')!=job
 or v.get('status')!='job_completed_awaiting_external_slurm_admission' or v.get('job_success') is not None
 or v.get('slurm_terminal_verified') is not False or v.get('scientific_generation_entered') is not False
 or v.get('authority')!=authority or v.get('wrapper_sha256')!=wrapper_sha
 or v.get('compute_bash')!=expected_bash or v.get('submission_binding')!=expected_binding or fd!=expected_fd
 or not isinstance(child_path,str) or not re.fullmatch(r'/proc/[1-9][0-9]*/fd/'+re.escape(fd_number),child_path)
 or retained_sha!=wrapper_sha or decoy.read_bytes()!=b'#!/usr/bin/env bash\nexit 97\n'
 or identity(retained)!=parent_identity or identity(decoy).split(':',2)[:2]==parent_identity.split(':',2)[:2]): raise SystemExit('evidence seal/schema/actuals differ')
PY
trap - EXIT TERM INT
exit 0
