#!/bin/bash
# One-shot controller for the PRE_D0 Level-B P2 00435 full-renderer product.
# It never retries and never cancels, releases, requeues, or signals job 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_bin_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly python_bin_size=31490256
readonly capacity_python=/usr/bin/python3.10
readonly capacity_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
readonly env_bin=/usr/bin/env
readonly env_sha=85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0
readonly env_size=43976
readonly base64_bin=/usr/bin/base64
readonly base64_sha=b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898
readonly base64_size=35336
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v4
readonly release_root="${experiment_root}/releases/fresh-world8-level-b-p2-00435-v3"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly attempt_root="${experiment_root}/attempts/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly controller="${launch_root}/auh_launch_action_edit_level_b_p2_00435_job140846_v4.sh"
readonly step_payload="${launch_root}/auh_action_edit_level_b_p2_00435_step_v4.sh"
readonly step_payload_sha=74f4ad83a198447246031a5b68ec6d812455dae0b6adee944e42c452bad0f0cc
readonly rank_exec="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v4.sh"
readonly rank_exec_sha=64fc0df647ab28d950d81b6735aead559d1e91216416a8e44e8ac0c3707620c8
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v4.py"
readonly bootstrap_sha=1d72a1594ab52e258f0fbac5410ea1d27e5c557a12d76e88b806b3ac99794391
readonly capacity_member="${launch_root}/action_edit_level_b_p2_00435_capacity_0817_v4.py"
readonly capacity_member_sha=87fc10c580070eef660fdfeaecf18ddd997d031a009508edbcc34a263cd6c4dc
readonly known_hosts="${launch_root}/node279_known_hosts"
readonly known_hosts_sha=376ed12f9662eba4fe41396853713c9e2ad30bc3069698016f295853ce3e4454
readonly launch_authority_core="${launch_root}/LAUNCH_AUTHORITY_CORE.json"
readonly launch_authority_core_sha=166cb80170763562c8041d80b3ed771bb1088890261b71821d807df0f998c92a
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=380b433d4be8c349bb79c8eb3914442136e153c2dccd4cb57ff25db9f7688a16
readonly renderer_sha=8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6
readonly static_preflight_sha=68edb1c3d5925d5ef26a2601b989f91777da3be4ad03d772aa5c2c7f1dde7998
readonly static_preflight_size=29205
readonly static_preflight_base64_sha=e69abf4f8829c8748b271d94ceae717c014c181f9a39207f5fb17846af5b6f59
readonly static_preflight_base64_size=38940
readonly output_mp4="${run_root}/00435ad621c44fac_p2_seed2026080821_v4.mp4"
readonly output_receipt="${output_mp4}.receipt.json"
readonly output_marker="${output_mp4}.COMMITTED.json"
readonly job_name=bernini0817-level-b-p2-00435-v4
readonly foreground_capacity_receipt="${attempt_root}/foreground-capacity-receipt.json"

fail() {
  printf 'Level-B P2 controller refused: %s\n' "$*" >&2
  exit 95
}

require_stat_value() {
  local stat_path="$1"
  local stat_format="$2"
  local stat_expected="$3"
  local stat_label="$4"
  local stat_observed
  if ! stat_observed="$(/usr/bin/stat -c "${stat_format}" "${stat_path}")"; then
    fail "${stat_label}: stat query failed"
  fi
  [[ "${stat_observed}" == "${stat_expected}" ]] || \
    fail "${stat_label}: expected ${stat_expected}, observed ${stat_observed}"
}

capture_sha256() {
  local sha_path="$1"
  local sha_output_name="$2"
  local sha_label="$3"
  local sha_observed
  if ! sha_observed="$(/usr/bin/sha256sum "${sha_path}" | /usr/bin/awk '{print $1}')"; then
    fail "${sha_label}: SHA query failed"
  fi
  [[ "${sha_observed}" =~ ^[0-9a-f]{64}$ ]] || fail "${sha_label}: SHA format differs"
  printf -v "${sha_output_name}" '%s' "${sha_observed}"
}

require_sha256() {
  local require_sha_path="$1"
  local require_sha_expected="$2"
  local require_sha_label="$3"
  local require_sha_observed
  capture_sha256 "${require_sha_path}" require_sha_observed "${require_sha_label}"
  [[ "${require_sha_observed}" == "${require_sha_expected}" ]] || \
    fail "${require_sha_label}: SHA differs"
}

capture_command_output_base64() {
  local command_output_name="$1"
  local command_label="$2"
  shift 2
  local command_frame
  local command_status
  local command_suffix
  local command_payload
  local command_sentinel=__LEVEL_B_P2_00435_V4_COMMAND_PIPESTATUS_
  if ! command_frame="$({
    set +e
    "$@" 2>&1 | "${base64_bin}" -w0
    command_status=("${PIPESTATUS[@]}")
    printf '%s%03d_%03d__' "${command_sentinel}" \
      "${command_status[0]}" "${command_status[1]}"
    exit 0
  })"; then
    fail "${command_label}: command framing failed"
  fi
  [[ "${command_frame}" =~ ${command_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || \
    fail "${command_label}: command frame suffix differs"
  command_suffix="${BASH_REMATCH[0]}"
  [[ "${BASH_REMATCH[1]}" == 000 ]] || \
    fail "${command_label}: command failed rc=${BASH_REMATCH[1]}"
  [[ "${BASH_REMATCH[2]}" == 000 ]] || \
    fail "${command_label}: command encoder failed rc=${BASH_REMATCH[2]}"
  command_payload="${command_frame%"${command_suffix}"}"
  [[ -n "${command_payload}" && "${command_payload}" != *$'\n'* \
    && "${command_payload}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || \
    fail "${command_label}: command transport base64 differs"
  printf -v "${command_output_name}" '%s' "${command_payload}"
}

decode_command_transport() {
  local command_decode_output_name="$1"
  local command_decode_payload="$2"
  local command_decode_label="$3"
  local command_decode_raw
  local command_decode_roundtrip
  if ! command_decode_raw="$(/usr/bin/printf '%s' "${command_decode_payload}" | "${base64_bin}" -d)"; then
    fail "${command_decode_label}: command transport decode failed"
  fi
  if ! command_decode_roundtrip="$(/usr/bin/printf '%s' "${command_decode_raw}" | "${base64_bin}" -w0)"; then
    fail "${command_decode_label}: command transport re-encode failed"
  fi
  [[ "${command_decode_roundtrip}" == "${command_decode_payload}" ]] || \
    fail "${command_decode_label}: command transport contains trailing bytes"
  printf -v "${command_decode_output_name}" '%s' "${command_decode_raw}"
}

publish_canonical_json_base64() {
  local publication_base64="$1"
  local publication_sha="$2"
  local publication_path="$3"
  local publication_label="$4"
  local publication_kind="${5:-json}"
  local publication_transport
  local publication_result
  require_sha256 "${env_bin}" "${env_sha}" \
    "${publication_label}: env tool before publication"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "${publication_label}: publisher Python before publication"
  require_sha256 "${base64_bin}" "${base64_sha}" \
    "${publication_label}: base64 tool before publication"
  capture_command_output_base64 publication_transport "${publication_label}" \
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    HOME=/nonexistent/bernini-level-b-p2-00435-v4-publisher \
    "${capacity_python}" -I -S -B -c '
import base64,binascii,hashlib,json,os,pathlib,re,stat,sys
def unique(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise ValueError("duplicate JSON key")
  out[key]=value
 return out
encoded,expected,destination,kind=sys.argv[1:]
assert kind in {"json","bytes"}
assert re.fullmatch(r"[A-Za-z0-9+/]+={0,2}",encoded)
raw=base64.b64decode(encoded.encode("ascii"),validate=True)
assert base64.b64encode(raw).decode("ascii")==encoded
assert re.fullmatch(r"[0-9a-f]{64}",expected) and hashlib.sha256(raw).hexdigest()==expected
if kind=="json":
 value=json.loads(raw.decode("utf-8"),object_pairs_hook=unique)
 assert raw==json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
final=pathlib.Path(destination); assert final.is_absolute()
parent=final.parent; pinfo=parent.lstat()
assert stat.S_ISDIR(pinfo.st_mode) and not parent.is_symlink()
try: final.lstat(); raise FileExistsError(destination)
except FileNotFoundError: pass
temporary=parent/("."+final.name+"."+str(os.getpid())+".tmp")
flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
fd=os.open(temporary,flags,0o444)
try:
 view=memoryview(raw)
 while view:
  written=os.write(fd,view); assert written>0; view=view[written:]
 os.fchmod(fd,0o444); os.fsync(fd)
finally: os.close(fd)
os.link(temporary,final,follow_symlinks=False)
tinfo=temporary.lstat(); finfo=final.lstat()
assert stat.S_ISREG(finfo.st_mode) and stat.S_IMODE(finfo.st_mode)==0o444
assert (tinfo.st_dev,tinfo.st_ino)==(finfo.st_dev,finfo.st_ino) and finfo.st_nlink==2
assert final.read_bytes()==raw
temporary.unlink()
dirfd=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
try: os.fsync(dirfd)
finally: os.close(dirfd)
finfo=final.lstat()
assert stat.S_ISREG(finfo.st_mode) and stat.S_IMODE(finfo.st_mode)==0o444 and finfo.st_nlink==1
assert final.read_bytes()==raw
sys.stdout.write(expected)
' "${publication_base64}" "${publication_sha}" "${publication_path}" \
    "${publication_kind}"
  require_sha256 "${env_bin}" "${env_sha}" \
    "${publication_label}: env tool after publication"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "${publication_label}: publisher Python after publication"
  require_sha256 "${base64_bin}" "${base64_sha}" \
    "${publication_label}: base64 tool after publication"
  decode_command_transport publication_result "${publication_transport}" \
    "${publication_label}"
  [[ "${publication_result}" == "${publication_sha}" ]] || \
    fail "${publication_label}: published SHA differs"
}

capture_capacity_output_base64() {
  local transport_output_name="$1"
  local transport_label="$2"
  shift 2
  local transport_frame
  local transport_status
  local transport_suffix
  local transport_payload
  local transport_sentinel=__LEVEL_B_P2_00435_V4_CONTROLLER_CAPACITY_PIPESTATUS_
  if ! transport_frame="$({
    set +e
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
      HOME=/nonexistent/bernini-level-b-p2-00435-v4-controller-capacity \
      "${capacity_python}" -I -S -B "${capacity_member}" "$@" 2>&1 | \
      "${base64_bin}" -w0
    transport_status=("${PIPESTATUS[@]}")
    printf '%s%03d_%03d__' "${transport_sentinel}" \
      "${transport_status[0]}" "${transport_status[1]}"
    exit 0
  })"; then
    fail "${transport_label}: capacity framing failed"
  fi
  [[ "${transport_frame}" =~ ${transport_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || \
    fail "${transport_label}: capacity frame suffix differs"
  transport_suffix="${BASH_REMATCH[0]}"
  [[ "${BASH_REMATCH[1]}" == 000 ]] || \
    fail "${transport_label}: capacity command failed rc=${BASH_REMATCH[1]}"
  [[ "${BASH_REMATCH[2]}" == 000 ]] || \
    fail "${transport_label}: capacity encoder failed rc=${BASH_REMATCH[2]}"
  transport_payload="${transport_frame%"${transport_suffix}"}"
  [[ -n "${transport_payload}" && "${transport_payload}" != *$'\n'* \
    && "${transport_payload}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || \
    fail "${transport_label}: capacity transport base64 differs"
  printf -v "${transport_output_name}" '%s' "${transport_payload}"
}

decode_capacity_transport() {
  local decode_output_name="$1"
  local decode_payload="$2"
  local decode_label="$3"
  local decode_raw
  local decode_roundtrip
  if ! decode_raw="$(/usr/bin/printf '%s' "${decode_payload}" | "${base64_bin}" -d)"; then
    fail "${decode_label}: capacity transport decode failed"
  fi
  if ! decode_roundtrip="$(/usr/bin/printf '%s' "${decode_raw}" | "${base64_bin}" -w0)"; then
    fail "${decode_label}: capacity transport re-encode failed"
  fi
  [[ "${decode_roundtrip}" == "${decode_payload}" ]] || \
    fail "${decode_label}: capacity transport contains trailing bytes"
  printf -v "${decode_output_name}" '%s' "${decode_raw}"
}

node_children() {
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    /usr/bin/awk -v prefix="${job_id}." \
      'index($0,prefix)==1 && substr($0,length(prefix)+1) ~ /^[0-9]+$/ {print $0}' |
    LC_ALL=C /usr/bin/sort
}

require_no_node_children() {
  local closure_label="$1"
  local closure_observed
  if ! closure_observed="$(node_children)"; then
    fail "${closure_label}: numeric child query failed"
  fi
  [[ -z "${closure_observed}" ]] || fail "${closure_label}: ${closure_observed}"
}

await_child_teardown() {
  local observed poll
  for poll in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if ! observed="$(node_children)"; then
      fail "completed-child teardown query failed"
    fi
    [[ -z "${observed}" ]] && return 0
    /usr/bin/sleep 2
  done
  fail "completed child remained visible: ${observed}"
}

write_status() {
  # These names must not shadow the module-scope readonly launch evidence.
  # Bash applies readonly attributes through dynamic scope, so v1's local
  # parent_before/parent_after/intent/log_path declarations failed exactly on
  # the child-failure finalizer.  Keep this namespace disjoint permanently.
  local status_exit_code="$1"
  local status_parent_before="$2"
  local status_parent_after="$3"
  local status_intent_path="$4"
  local status_log_path="$5"
  local status_foreground_path="$6"
  local status_foreground_sha="$7"
  local status_foreground_challenge="$8"
  local status_controller_path="$9"
  local status_controller_sha="${10}"
  local status_controller_challenge="${11}"
  local status_step_path="${12}"
  local status_parent_after_observation="${13}"
  local status_output="${attempt_root}/controller.status.json"
  local status_transport
  local status_raw
  local status_sha
  [[ ! -e "${status_output}" && ! -L "${status_output}" ]] || fail "controller status already exists"
  require_sha256 "${env_bin}" "${env_sha}" "env tool before status generation"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "publisher Python before status generation"
  capture_command_output_base64 status_transport \
    "controller status generation" \
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    HOME=/nonexistent/bernini-level-b-p2-00435-v4-status \
    "${capacity_python}" -I -S -B -c '
import hashlib,json,pathlib,stat,sys
step=pathlib.Path(sys.argv[12]); step_value={"path":str(step),"state":"absent"}
try:
 info=step.lstat()
except FileNotFoundError:
 pass
except BaseException as exc:
 step_value={"path":str(step),"state":"present_invalid","failure_class":type(exc).__name__}
else:
 step_value={"path":str(step),"state":"present_invalid","mode":stat.S_IMODE(info.st_mode),"nlink":info.st_nlink,"size":info.st_size}
 try:
  assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1 and not step.is_symlink()
  raw=step.read_bytes(); value=json.loads(raw.decode("utf-8"))
  assert raw==json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
  assert value["sample_phase"]=="step"
  challenge=value["sample_challenge"]
  assert isinstance(challenge,str) and len(challenge)==64 and all(c in "0123456789abcdef" for c in challenge)
  step_value={"path":str(step),"state":"present_parseable_identity","sha256":hashlib.sha256(raw).hexdigest(),"challenge":challenge,"phase":"step","mode":stat.S_IMODE(info.st_mode),"nlink":info.st_nlink,"size":info.st_size,"archival_validation_performed":False,"admission_authority_reasserted":False}
 except BaseException as exc:
  step_value["failure_class"]=type(exc).__name__
assert sys.argv[13] in {"observed","query_failed"}
out={"schema_version":"bernini-action-edit-level-b-p2-controller-status-v4","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v4","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":int(sys.argv[1]),"parent_state_before":sys.argv[2],"parent_state_after":sys.argv[3] if sys.argv[13]=="observed" else None,"parent_state_after_observation":sys.argv[13],"intent_path":sys.argv[4],"run_log_path":sys.argv[5],"capacity_receipts":{"foreground":{"path":sys.argv[6],"sha256":sys.argv[7],"challenge":sys.argv[8],"phase":"foreground"},"controller":{"path":sys.argv[9],"sha256":sys.argv[10],"challenge":sys.argv[11],"phase":"controller"},"step":step_value},"automatic_relaunch_authorized":False,"parent_control_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${status_exit_code}" "${status_parent_before}" "${status_parent_after}" \
  "${status_intent_path}" "${status_log_path}" "${status_foreground_path}" \
  "${status_foreground_sha}" "${status_foreground_challenge}" \
  "${status_controller_path}" "${status_controller_sha}" \
  "${status_controller_challenge}" "${status_step_path}" \
  "${status_parent_after_observation}"
  decode_command_transport status_raw "${status_transport}" \
    "controller status generation"
  require_sha256 "${env_bin}" "${env_sha}" "env tool after status generation"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "publisher Python after status generation"
  if ! status_sha="$(/usr/bin/printf '%s' "${status_raw}" | \
    /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
    fail "controller status SHA query failed"
  fi
  [[ "${status_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller status SHA differs"
  publish_canonical_json_base64 "${status_transport}" "${status_sha}" \
    "${status_output}" "controller status publication"
}

validated_product_probe() {
  local probe_label="$1"
  local probe_frame
  local probe_pipeline_status
  local probe_frame_suffix
  local probe_child_status
  local probe_encoder_status
  local probe_base64
  local probe_raw
  local probe_wrapper_transport
  local probe_sentinel=__LEVEL_B_P2_00435_V4_PRODUCT_VALIDATION_PIPESTATUS_
  require_sha256 "${bootstrap}" "${bootstrap_sha}" \
    "${probe_label} bootstrap before product validation"
  require_sha256 "${python_bin}" "${python_bin_sha}" \
    "${probe_label} renderer Python before product validation"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "${probe_label} wrapper Python before product validation"
  require_sha256 "${env_bin}" "${env_sha}" \
    "${probe_label} env tool before product validation"
  if ! probe_frame="$({
    set +e
    "${python_bin}" -I -B "${bootstrap}" validate-product 2>&1 | \
      "${base64_bin}" -w0
    probe_pipeline_status=("${PIPESTATUS[@]}")
    printf '%s%03d_%03d__' "${probe_sentinel}" \
      "${probe_pipeline_status[0]}" "${probe_pipeline_status[1]}"
    exit 0
  })"; then
    fail "${probe_label} product-validation framing failed"
  fi
  [[ "${probe_frame}" =~ ${probe_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || \
    fail "${probe_label} product-validation frame suffix differs"
  probe_frame_suffix="${BASH_REMATCH[0]}"
  probe_child_status="${BASH_REMATCH[1]}"
  probe_encoder_status="${BASH_REMATCH[2]}"
  probe_base64="${probe_frame%"${probe_frame_suffix}"}"
  [[ "${probe_child_status}" == 000 ]] || \
    fail "${probe_label} product validation failed rc=${probe_child_status}"
  [[ "${probe_encoder_status}" == 000 ]] || \
    fail "${probe_label} product-validation base64 failed rc=${probe_encoder_status}"
  [[ -n "${probe_base64}" && "${probe_base64}" != *$'\n'* ]] || \
    fail "${probe_label} product-validation base64 framing differs"
  capture_command_output_base64 probe_wrapper_transport \
    "${probe_label} product-validation canonical wrapper" \
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
    HOME=/nonexistent/bernini-level-b-p2-00435-v4-product-wrapper \
    "${capacity_python}" -I -S -B -c '
import base64,json,sys
def unique(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError("duplicate JSON key")
        out[key]=value
    return out
raw=base64.b64decode(sys.argv[1].encode("ascii"),validate=True)
value=json.loads(raw.decode("utf-8"),object_pairs_hook=unique)
canonical=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
required={"schema_version","method","authority","output_mp4","validation","receipt_claims_revalidated","receipt_inode_alias_marker_revalidated","committed_marker_required","formal_training_started","counts_as_d0","promotion_authorized"}
assert raw==canonical and set(value)==required
assert value["schema_version"]=="bernini-action-edit-level-b-p2-product-validation-v4"
assert value["method"]=="bernini-action-edit-level-b-p2-00435-bootstrap-0817-v4"
assert value["authority"]=="PRE_D0_ENGINEERING_ONLY"
assert value["output_mp4"]==sys.argv[2]
assert isinstance(value["validation"],dict)
assert value["receipt_claims_revalidated"] is True
assert value["receipt_inode_alias_marker_revalidated"] is True
assert value["committed_marker_required"] is True
assert value["formal_training_started"] is False
assert value["counts_as_d0"] is False
assert value["promotion_authorized"] is False
sys.stdout.buffer.write(raw)
' "${probe_base64}" "${output_mp4}"
  decode_command_transport probe_raw "${probe_wrapper_transport}" \
    "${probe_label} product-validation canonical wrapper"
  require_sha256 "${bootstrap}" "${bootstrap_sha}" \
    "${probe_label} bootstrap after product validation"
  require_sha256 "${python_bin}" "${python_bin_sha}" \
    "${probe_label} renderer Python after product validation"
  require_sha256 "${capacity_python}" "${capacity_python_sha}" \
    "${probe_label} wrapper Python after product validation"
  require_sha256 "${env_bin}" "${env_sha}" \
    "${probe_label} env tool after product validation"
  [[ -n "${probe_raw}" && "${probe_raw}" != *$'\n'* ]] || \
    fail "${probe_label} product-validation canonical bytes differ"
  printf '%s' "${probe_raw}"
}

for pending_sha in "${step_payload_sha}" "${rank_exec_sha}" "${bootstrap_sha}" \
  "${launch_authority_core_sha}" "${release_manifest_sha}" "${renderer_sha}" \
  "${static_preflight_sha}" "${static_preflight_base64_sha}" \
  "${capacity_member_sha}" "${known_hosts_sha}" \
  "${capacity_python_sha}" "${python_bin_sha}" "${env_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ "${PATH:-}" == /usr/bin:/bin ]] || fail "controller entry PATH differs"
[[ "${LC_ALL:-}" == C && "${LANG:-}" == C ]] || \
  fail "controller entry locale differs"
[[ "${HOME:-}" == /vast/users/guangyi.chen ]] || fail "controller entry HOME differs"
[[ "${BASH_ENV:-}" == /dev/null ]] || fail "pre-script BASH_ENV boundary differs"
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${ROCM_SMI_LIB_PATH:-}" ]] || \
  fail "pre-script inherited loader or Python environment differs"
[[ $# == 3 ]] || fail "exact foreground capacity authority argv differs"
readonly foreground_receipt_arg="$1"
readonly foreground_receipt_sha="$2"
readonly foreground_capacity_challenge="$3"
[[ "${foreground_receipt_arg}" == "${foreground_capacity_receipt}" ]] || \
  fail "foreground capacity receipt path differs"
[[ "${foreground_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "foreground capacity receipt SHA differs"
[[ "${foreground_capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "foreground capacity challenge differs"
[[ "${static_preflight_size}" =~ ^[1-9][0-9]*$ ]] || fail "static preflight size differs"
[[ "${static_preflight_base64_size}" =~ ^[1-9][0-9]*$ ]] || fail "static preflight base64 size differs"
[[ "$0" == "${controller}" ]] || fail "controller absolute path differs"
[[ -x "${controller}" && ! -L "${controller}" ]] || fail "controller file differs"
require_stat_value "${controller}" %a 555 "controller mode"
require_stat_value "${controller}" %h 1 "controller link count"
[[ -d "${launch_root}" && ! -L "${launch_root}" ]] || fail "launch root differs"
require_stat_value "${launch_root}" %a 555 "launch root mode"
readonly expected_launch_entries=$'LAUNCH_AUTHORITY_CORE.json\naction_edit_level_b_p2_00435_bootstrap_0817_v4.py\naction_edit_level_b_p2_00435_capacity_0817_v4.py\nauh_action_edit_level_b_p2_00435_rank_exec_v4.sh\nauh_action_edit_level_b_p2_00435_step_v4.sh\nauh_launch_action_edit_level_b_p2_00435_job140846_v4.sh\nnode279_known_hosts'
observed_launch_entries=
if ! observed_launch_entries="$(/usr/bin/find "${launch_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "launch exact closure query failed"
fi
readonly observed_launch_entries
[[ "${observed_launch_entries}" == "${expected_launch_entries}" ]] || fail "launch exact closure differs"
for executable in "${controller}" "${step_payload}" "${rank_exec}"; do
  [[ -x "${executable}" && ! -L "${executable}" ]] || fail "launcher executable differs"
  require_stat_value "${executable}" %a 555 "launcher executable mode"
  require_stat_value "${executable}" %h 1 "launcher executable link count"
done
for sealed_member in "${bootstrap}" "${capacity_member}" "${known_hosts}" \
  "${launch_authority_core}"; do
  [[ -f "${sealed_member}" && ! -L "${sealed_member}" ]] || fail "sealed launch member topology differs"
  require_stat_value "${sealed_member}" %a 444 "sealed launch member mode"
  require_stat_value "${sealed_member}" %h 1 "sealed launch member link count"
done
[[ -x "${base64_bin}" && ! -L "${base64_bin}" ]] || fail "base64 tool path differs"
base64_resolved=
if ! base64_resolved="$(/usr/bin/readlink -f "${base64_bin}")"; then
  fail "base64 tool canonical-path query failed"
fi
readonly base64_resolved
[[ "${base64_resolved}" == "${base64_bin}" ]] || fail "base64 tool path differs"
require_stat_value "${base64_bin}" %a 755 "base64 tool mode"
require_stat_value "${base64_bin}" %h 1 "base64 tool link count"
require_stat_value "${base64_bin}" %s "${base64_size}" "base64 tool size"
require_sha256 "${base64_bin}" "${base64_sha}" "base64 tool"
[[ -x "${env_bin}" && ! -L "${env_bin}" ]] || fail "env tool path differs"
require_stat_value "${env_bin}" %a 755 "env tool mode"
require_stat_value "${env_bin}" %h 1 "env tool link count"
require_stat_value "${env_bin}" %s "${env_size}" "env tool size"
require_sha256 "${env_bin}" "${env_sha}" "env tool"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "renderer Python differs"
require_stat_value "${python_bin}" %a 755 "renderer Python mode"
require_stat_value "${python_bin}" %h 1 "renderer Python link count"
require_stat_value "${python_bin}" %s "${python_bin_size}" "renderer Python size"
require_sha256 "${python_bin}" "${python_bin_sha}" "renderer Python"
require_sha256 "${step_payload}" "${step_payload_sha}" "step payload"
require_sha256 "${rank_exec}" "${rank_exec_sha}" "rank wrapper"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap"
require_sha256 "${capacity_member}" "${capacity_member_sha}" "capacity member"
require_sha256 "${known_hosts}" "${known_hosts_sha}" "sealed node279 known hosts"
require_sha256 "${launch_authority_core}" "${launch_authority_core_sha}" "launch core"
require_sha256 "${release_manifest}" "${release_manifest_sha}" "release manifest"
[[ -x "${capacity_python}" && ! -L "${capacity_python}" ]] || fail "capacity Python differs"
require_stat_value "${capacity_python}" %a 755 "capacity Python mode"
require_stat_value "${capacity_python}" %h 1 "capacity Python link count"
require_stat_value "${capacity_python}" %s 5937800 "capacity Python size"
require_sha256 "${capacity_python}" "${capacity_python_sha}" "capacity Python"
require_sha256 "${env_bin}" "${env_sha}" "env tool after launch-member authentication"
require_sha256 "${python_bin}" "${python_bin_sha}" \
  "renderer Python after launch-member authentication"

[[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt latch root differs"
require_stat_value "${attempt_root}" %a 700 "attempt root mode"
attempt_initial_entries=
if ! attempt_initial_entries="$(/usr/bin/find "${attempt_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "attempt freshness query failed"
fi
readonly attempt_initial_entries
[[ "${attempt_initial_entries}" == foreground-capacity-receipt.json ]] || \
  fail "attempt root must contain only the foreground capacity receipt"
[[ -f "${foreground_capacity_receipt}" && ! -L "${foreground_capacity_receipt}" ]] || \
  fail "foreground capacity receipt topology differs"
require_stat_value "${foreground_capacity_receipt}" %a 444 "foreground capacity receipt mode"
require_stat_value "${foreground_capacity_receipt}" %h 1 "foreground capacity receipt link count"
require_sha256 "${foreground_capacity_receipt}" "${foreground_receipt_sha}" \
  "foreground capacity receipt"
validated_foreground_capacity=
foreground_validation_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before foreground receipt validation"
capture_capacity_output_base64 foreground_validation_transport \
  "foreground capacity receipt validation" validate-file \
  "${foreground_capacity_receipt}" "${foreground_receipt_sha}" foreground \
  "${foreground_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after foreground receipt validation"
decode_capacity_transport validated_foreground_capacity \
  "${foreground_validation_transport}" "foreground capacity receipt validation"
readonly validated_foreground_capacity
validated_foreground_capacity_sha=
if ! validated_foreground_capacity_sha="$(printf '%s' "${validated_foreground_capacity}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "foreground capacity receipt SHA query failed"
fi
readonly validated_foreground_capacity_sha
[[ "${validated_foreground_capacity_sha}" == "${foreground_receipt_sha}" ]] || \
  fail "foreground capacity receipt canonical bytes differ"
[[ -d "${run_root}" && ! -L "${run_root}" ]] || fail "run root differs"
require_stat_value "${run_root}" %a 700 "run root mode"
run_initial_entry=
if ! run_initial_entry="$(/usr/bin/find "${run_root}" -mindepth 1 -print -quit)"; then
  fail "run freshness query failed"
fi
readonly run_initial_entry
[[ -z "${run_initial_entry}" ]] || fail "run root is not fresh"
parent_before=
if ! parent_before="$(/usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed before launch"
fi
readonly parent_before
[[ "${parent_before}" == "${expected_parent_state}" ]] || fail "parent state differs before launch: ${parent_before}"
require_no_node_children "parent already has a numeric node279 child"

# CPU-only, read/import/descriptor inspection of the exact sealed release.
# This runs before the persistent STARTED latch and before every output/log
# redirection.  Its Python audit hook denies filesystem mutation, process
# creation, and network access; the receipt is held in memory only.
readonly static_preflight_frame_sentinel=__LEVEL_B_P2_00435_V4_STATIC_PREFLIGHT_PIPESTATUS_
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap before static preflight"
require_sha256 "${python_bin}" "${python_bin_sha}" \
  "renderer Python before static preflight"
static_preflight_frame=
if ! static_preflight_frame="$(
  set +e
  "${env_bin}" -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/nonexistent/bernini-level-b-p2-00435-v4 \
    TMPDIR=/nonexistent/bernini-level-b-p2-00435-v4/tmp \
    XDG_CACHE_HOME=/nonexistent/bernini-level-b-p2-00435-v4/cache \
    HF_HOME=/nonexistent/bernini-level-b-p2-00435-v4/huggingface \
    TRANSFORMERS_CACHE=/nonexistent/bernini-level-b-p2-00435-v4/transformers \
    CUDA_VISIBLE_DEVICES='' \
    ROCR_VISIBLE_DEVICES='' \
    HIP_VISIBLE_DEVICES='' \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OPENBLAS_MAIN_FREE=1 \
    GOTOBLAS_MAIN_FREE=1 \
    VEOMNI_VERBOSITY=ERROR \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    "${python_bin}" -I -B "${bootstrap}" static-preflight 2>&1 | \
    "${base64_bin}" -w0
  static_preflight_pipeline_status=("${PIPESTATUS[@]}")
  printf '%s%03d_%03d__' "${static_preflight_frame_sentinel}" \
    "${static_preflight_pipeline_status[0]}" \
    "${static_preflight_pipeline_status[1]}"
  exit 0
)"; then
  fail "CPU static runtime preflight framing failed"
fi
readonly static_preflight_frame
[[ "${static_preflight_frame}" =~ ${static_preflight_frame_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || fail "CPU static preflight frame suffix differs"
readonly static_preflight_frame_suffix="${BASH_REMATCH[0]}"
readonly static_preflight_child_status="${BASH_REMATCH[1]}"
readonly static_preflight_encoder_status="${BASH_REMATCH[2]}"
readonly static_preflight_base64="${static_preflight_frame%"${static_preflight_frame_suffix}"}"
[[ "${static_preflight_child_status}" == 000 ]] || fail "CPU static runtime preflight failed rc=${static_preflight_child_status}"
[[ "${static_preflight_encoder_status}" == 000 ]] || fail "CPU static preflight base64 framing failed rc=${static_preflight_encoder_status}"
[[ "${static_preflight_base64}" != *$'\n'* ]] || fail "CPU static preflight base64 framing contains a newline"
[[ "${#static_preflight_base64}" == "${static_preflight_base64_size}" ]] || fail "CPU static preflight base64 size differs"
static_preflight_observed_base64_sha=
if ! static_preflight_observed_base64_sha="$(printf '%s' "${static_preflight_base64}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "CPU static preflight base64 SHA query failed"
fi
readonly static_preflight_observed_base64_sha
[[ "${static_preflight_observed_base64_sha}" == "${static_preflight_base64_sha}" ]] || fail "CPU static preflight combined stdout/stderr base64 SHA differs"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap after static preflight"
require_sha256 "${python_bin}" "${python_bin_sha}" \
  "renderer Python after static preflight"
parent_after_static_preflight=
if ! parent_after_static_preflight="$(/usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed after CPU static preflight"
fi
readonly parent_after_static_preflight
[[ "${parent_after_static_preflight}" == "${parent_before}" ]] || fail "parent changed during CPU static preflight"
require_no_node_children "numeric node279 child appeared during CPU static preflight"
attempt_after_static_preflight=
if ! attempt_after_static_preflight="$(/usr/bin/find "${attempt_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "attempt post-preflight closure query failed"
fi
readonly attempt_after_static_preflight
[[ "${attempt_after_static_preflight}" == foreground-capacity-receipt.json ]] || \
  fail "attempt root changed during CPU static preflight"
run_after_static_preflight=
if ! run_after_static_preflight="$(/usr/bin/find "${run_root}" -mindepth 1 -print -quit)"; then
  fail "run post-preflight closure query failed"
fi
readonly run_after_static_preflight
[[ -z "${run_after_static_preflight}" ]] || fail "run root changed during CPU static preflight"

# Recheck parent/child closure before the controller's independent live sample.
# From the single direct-node sample through STARTED there is no sleep or poll.
parent_before_controller_capacity=
if ! parent_before_controller_capacity="$(/usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed before controller capacity sample"
fi
readonly parent_before_controller_capacity
[[ "${parent_before_controller_capacity}" == "${parent_before}" ]] || \
  fail "parent changed before controller capacity sample"
require_no_node_children "numeric child appeared before controller capacity sample"

controller_capacity_challenge=
controller_challenge_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before controller challenge generation"
capture_capacity_output_base64 controller_challenge_transport \
  "controller capacity challenge" challenge
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after controller challenge generation"
decode_capacity_transport controller_capacity_challenge \
  "${controller_challenge_transport}" "controller capacity challenge"
readonly controller_capacity_challenge
[[ "${controller_capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "controller capacity challenge differs"
[[ "${controller_capacity_challenge}" != "${foreground_capacity_challenge}" ]] || \
  fail "controller capacity challenge reused foreground challenge"

# The caller binds the self-hash immediately around the one direct-node SSH.
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before controller direct-node sample"
controller_capacity_base64=
controller_probe_transport=
capture_capacity_output_base64 controller_probe_transport \
  "controller direct-node capacity sample" remote-probe-base64 controller \
  "${controller_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after controller direct-node sample"
decode_capacity_transport controller_capacity_base64 \
  "${controller_probe_transport}" "controller direct-node capacity sample"
readonly controller_capacity_base64
[[ -n "${controller_capacity_base64}" \
  && "${controller_capacity_base64}" != *$'\n'* \
  && "${controller_capacity_base64}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || \
  fail "controller capacity receipt base64 differs"
controller_capacity_raw=
if ! controller_capacity_raw="$(printf '%s' "${controller_capacity_base64}" | "${base64_bin}" -d)"; then
  fail "controller capacity receipt decode failed"
fi
readonly controller_capacity_raw
controller_capacity_roundtrip=
if ! controller_capacity_roundtrip="$(printf '%s' "${controller_capacity_raw}" | "${base64_bin}" -w0)"; then
  fail "controller capacity receipt re-encode failed"
fi
readonly controller_capacity_roundtrip
[[ "${controller_capacity_roundtrip}" == "${controller_capacity_base64}" ]] || \
  fail "controller capacity receipt base64 is not canonical"
controller_capacity_sha=
if ! controller_capacity_sha="$(printf '%s' "${controller_capacity_raw}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "controller capacity receipt SHA query failed"
fi
readonly controller_capacity_sha
[[ "${controller_capacity_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "controller capacity receipt SHA differs"
validated_controller_capacity=
controller_validation_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before controller receipt validation"
capture_capacity_output_base64 controller_validation_transport \
  "controller capacity receipt validation" validate-base64 \
  "${controller_capacity_base64}" "${controller_capacity_sha}" controller \
  "${controller_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after controller receipt validation"
decode_capacity_transport validated_controller_capacity \
  "${controller_validation_transport}" "controller capacity receipt validation"
readonly validated_controller_capacity
[[ "${validated_controller_capacity}" == "${controller_capacity_raw}" ]] || \
  fail "controller capacity receipt canonical bytes differ"

# Persistent atomic claim.  It is never removed, including on child failure.
readonly started="${attempt_root}/STARTED"
/usr/bin/mkdir -m 0700 "${started}" 2>/dev/null || fail "attempt was already claimed"
readonly controller_capacity_receipt="${started}/controller-capacity-receipt.json"
published_controller_capacity_sha=
controller_publish_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before controller receipt publication"
capture_capacity_output_base64 controller_publish_transport \
  "controller capacity receipt publication" publish-base64 \
  "${controller_capacity_base64}" "${controller_capacity_sha}" controller \
  "${controller_capacity_challenge}" "${controller_capacity_receipt}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after controller receipt publication"
decode_capacity_transport published_controller_capacity_sha \
  "${controller_publish_transport}" "controller capacity receipt publication"
readonly published_controller_capacity_sha
[[ "${published_controller_capacity_sha}" == "${controller_capacity_sha}" ]] || \
  fail "published controller capacity receipt SHA differs"
require_stat_value "${controller_capacity_receipt}" %a \
  444 "controller capacity receipt mode"
require_stat_value "${controller_capacity_receipt}" %h \
  1 "controller capacity receipt link count"
require_sha256 "${controller_capacity_receipt}" "${controller_capacity_sha}" \
  "controller capacity receipt"
readonly intent="${started}/intent.json"
[[ ! -e "${intent}" && ! -L "${intent}" ]] || fail "intent already exists"
intent_generation_transport=
require_sha256 "${env_bin}" "${env_sha}" "env tool before intent generation"
require_sha256 "${capacity_python}" "${capacity_python_sha}" \
  "publisher Python before intent generation"
capture_command_output_base64 intent_generation_transport \
  "dynamic canonical intent generation" \
  "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
  HOME=/nonexistent/bernini-level-b-p2-00435-v4-intent \
  "${capacity_python}" -I -S -B -c '
import json,sys
out={
 "schema_version":"bernini-action-edit-level-b-p2-attempt-intent-v4",
 "method":"bernini-action-edit-level-b-p2-00435-bootstrap-0817-v4",
 "authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v4",
 "parent_job_id":140846,"node":"auh7-1b-gpu-279",
 "job_name":"bernini0817-level-b-p2-00435-v4",
 "release_root":sys.argv[1],"release_reused_from_v3":True,
 "release_bytes_unchanged":True,"launch_root":sys.argv[2],
 "attempt_root":sys.argv[3],"run_root":sys.argv[4],"output_mp4":sys.argv[5],
 "source_video_sha256":"b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1",
 "instruction_utf8_sha256":"cfe1e51a8b8ada76c5b1d6993cfb8d55cbc1f21fb0694a14ddb9c11133f74088",
 "inference_seed":2026080821,"checkpoint_step":2,
 "checkpoint_parameter_sha256":"5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e",
 "release_manifest_sha256":sys.argv[6],"renderer_sha256":sys.argv[7],
 "bootstrap_sha256":sys.argv[8],"step_payload_sha256":sys.argv[9],
 "rank_exec_sha256":sys.argv[10],"static_preflight_stdout_sha256":sys.argv[11],
 "static_preflight_stdout_size":int(sys.argv[12]),"capacity_member_sha256":sys.argv[13],
 "node279_known_hosts_sha256":sys.argv[14],
 "capacity":{"schema_version":"bernini-action-edit-level-b-p2-capacity-receipt-v4",
  "foreground":{"path":sys.argv[15],"sha256":sys.argv[16],"challenge":sys.argv[17],"phase":"foreground"},
  "controller":{"path":sys.argv[18],"sha256":sys.argv[19],"challenge":sys.argv[20],"phase":"controller"},
  "step":{"expected_path":sys.argv[21],"phase":"step","must_be_fresh_after_srun":True},
  "independent_fresh_sample_count_required":3,"one_rocm_producer_per_sample":True,
  "gpu_use_percent_required":0,"vram_total_bytes_required":68702699520,
  "minimum_free_basis_points":9500,"per_card_not_average":True,
  "retained_inner_torch_95_percent_gate":True},
 "world_size":8,"dp_size":2,"sp_size":4,"host_memory_gib":64,"max_restarts":0,
 "committed_marker_required":True,"automatic_relaunch_authorized":False,
 "parent_control_authorized":False,"formal_training_started":False,
 "counts_as_d0":False,"promotion_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${release_root}" "${launch_root}" "${attempt_root}" "${run_root}" "${output_mp4}" \
  "${release_manifest_sha}" "${renderer_sha}" "${bootstrap_sha}" \
  "${step_payload_sha}" "${rank_exec_sha}" "${static_preflight_sha}" \
  "${static_preflight_size}" "${capacity_member_sha}" "${known_hosts_sha}" \
  "${foreground_capacity_receipt}" "${foreground_receipt_sha}" \
  "${foreground_capacity_challenge}" "${controller_capacity_receipt}" \
  "${controller_capacity_sha}" "${controller_capacity_challenge}" \
  "${started}/step-capacity-receipt.json"
intent_raw=
decode_command_transport intent_raw "${intent_generation_transport}" \
  "dynamic canonical intent generation"
require_sha256 "${env_bin}" "${env_sha}" "env tool after intent generation"
require_sha256 "${capacity_python}" "${capacity_python_sha}" \
  "publisher Python after intent generation"
readonly intent_raw
attempt_intent_sha=
if ! attempt_intent_sha="$(/usr/bin/printf '%s' "${intent_raw}" | \
  /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "dynamic canonical intent SHA query failed"
fi
readonly attempt_intent_sha
[[ "${attempt_intent_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "dynamic canonical intent SHA format differs"
publish_canonical_json_base64 "${intent_generation_transport}" \
  "${attempt_intent_sha}" "${intent}" "dynamic canonical intent publication"
[[ -f "${intent}" && ! -L "${intent}" ]] || fail "intent topology differs"
require_stat_value "${intent}" %a 444 "intent mode"
require_stat_value "${intent}" %h 1 "intent link count"
require_sha256 "${intent}" "${attempt_intent_sha}" "dynamic canonical intent"

readonly log_path="${attempt_root}/run.log"
export PATH=/usr/bin:/bin
export LC_ALL=C LANG=C
export BASH_ENV=/dev/null
unset ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export -n SHELLOPTS BASHOPTS 2>/dev/null || true
set +e
set -o noclobber
/usr/bin/srun --jobid="${job_id}" --overlap --exact --nodes=1 --ntasks=1 \
  --nodelist="${node}" --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 \
  --job-name="${job_name}" --kill-on-bad-exit=1 \
  "${step_payload}" "${launch_authority_core}" "${launch_authority_core_sha}" \
  "${intent}" "${attempt_intent_sha}" "${controller_capacity_receipt}" \
  "${controller_capacity_sha}" "${controller_capacity_challenge}" \
  "${foreground_capacity_challenge}" \
  >"${log_path}" 2>&1
readonly child_exit=$?
set +o noclobber
set -e
parent_after=
parent_after_observation=observed
if ! parent_after="$(/usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  parent_after=
  parent_after_observation=query_failed
fi
readonly parent_after parent_after_observation
write_status "${child_exit}" "${parent_before}" "${parent_after}" "${intent}" \
  "${log_path}" "${foreground_capacity_receipt}" "${foreground_receipt_sha}" \
  "${foreground_capacity_challenge}" "${controller_capacity_receipt}" \
  "${controller_capacity_sha}" "${controller_capacity_challenge}" \
  "${started}/step-capacity-receipt.json" "${parent_after_observation}"
if (( child_exit != 0 )); then
  printf 'Level-B P2 child failed rc=%s; no retry is authorized\n' "${child_exit}" >&2
  /usr/bin/tail -n 240 "${log_path}" >&2 || true
  exit "${child_exit}"
fi
[[ "${parent_after_observation}" == observed ]] || \
  fail "parent state query failed after successful child; status was recorded"
[[ "${parent_after}" == "${expected_parent_state}" ]] || fail "parent state changed after child: ${parent_after}"
await_child_teardown

readonly step_capacity_receipt="${started}/step-capacity-receipt.json"
[[ -f "${step_capacity_receipt}" && ! -L "${step_capacity_receipt}" ]] || \
  fail "step capacity receipt topology differs"
require_stat_value "${step_capacity_receipt}" %a 444 "step capacity receipt mode"
require_stat_value "${step_capacity_receipt}" %h 1 "step capacity receipt link count"
step_capacity_sha=
capture_sha256 "${step_capacity_receipt}" step_capacity_sha "step capacity receipt"
readonly step_capacity_sha
step_capacity_challenge=
step_challenge_transport=
capture_command_output_base64 step_challenge_transport \
  "step capacity challenge extraction" \
  "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
  HOME=/nonexistent/bernini-level-b-p2-00435-v4-step-challenge \
  "${capacity_python}" -I -S -B -c '
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]); raw=p.read_bytes(); value=json.loads(raw.decode("utf-8"))
assert raw==json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
challenge=value.get("sample_challenge")
assert value.get("sample_phase")=="step" and isinstance(challenge,str) and re.fullmatch(r"[0-9a-f]{64}",challenge)
sys.stdout.write(challenge)
' "${step_capacity_receipt}"
decode_command_transport step_capacity_challenge "${step_challenge_transport}" \
  "step capacity challenge extraction"
readonly step_capacity_challenge
[[ "${step_capacity_challenge}" != "${foreground_capacity_challenge}" \
  && "${step_capacity_challenge}" != "${controller_capacity_challenge}" ]] || \
  fail "step capacity challenge reused an earlier challenge"
validated_step_capacity=
step_validation_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before post-child archival validation"
capture_capacity_output_base64 step_validation_transport \
  "step capacity receipt post-child archival validation" validate-file-archival \
  "${step_capacity_receipt}" "${step_capacity_sha}" step \
  "${step_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after post-child archival validation"
decode_capacity_transport validated_step_capacity "${step_validation_transport}" \
  "step capacity receipt post-child archival validation"
readonly validated_step_capacity
validated_step_capacity_sha=
if ! validated_step_capacity_sha="$(printf '%s' "${validated_step_capacity}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "validated step capacity receipt SHA query failed"
fi
readonly validated_step_capacity_sha
[[ "${validated_step_capacity_sha}" == "${step_capacity_sha}" ]] || \
  fail "validated step capacity receipt bytes differ"

readonly expected_run_entries=$'00435ad621c44fac_p2_seed2026080821_v4.mp4\n00435ad621c44fac_p2_seed2026080821_v4.mp4.COMMITTED.json\n00435ad621c44fac_p2_seed2026080821_v4.mp4.receipt.json'
observed_run_entries=
if ! observed_run_entries="$(/usr/bin/find "${run_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "committed product closure query failed"
fi
readonly observed_run_entries
[[ "${observed_run_entries}" == "${expected_run_entries}" ]] || fail "committed product exact closure differs"
for product_file in "${output_mp4}" "${output_receipt}" "${output_marker}"; do
  [[ -f "${product_file}" && ! -L "${product_file}" ]] || fail "committed product member differs"
  require_stat_value "${product_file}" %a 444 "committed product mode"
done
require_stat_value "${output_mp4}" %h 1 "committed MP4 link count"
require_stat_value "${output_receipt}" %h 2 "committed receipt link count"
require_stat_value "${output_marker}" %h 2 "commit-marker link count"
receipt_dev_ino=
if ! receipt_dev_ino="$(/usr/bin/stat -c '%d:%i' "${output_receipt}")"; then
  fail "receipt inode identity query failed"
fi
readonly receipt_dev_ino
marker_dev_ino=
if ! marker_dev_ino="$(/usr/bin/stat -c '%d:%i' "${output_marker}")"; then
  fail "COMMITTED inode identity query failed"
fi
readonly marker_dev_ino
[[ "${receipt_dev_ino}" == "${marker_dev_ino}" ]] || fail "COMMITTED marker is not the exact receipt inode alias"
receipt_alias_sha=
capture_sha256 "${output_receipt}" receipt_alias_sha "committed receipt"
readonly receipt_alias_sha
marker_alias_sha=
capture_sha256 "${output_marker}" marker_alias_sha "COMMITTED marker"
readonly marker_alias_sha
[[ "${receipt_alias_sha}" == "${marker_alias_sha}" ]] || fail "receipt/COMMITTED alias bytes differ"

# Two complete, side-effect-free validations must agree before terminal seal.
readonly validation_a="${started}/product-validation-1.json"
readonly validation_b="${started}/product-validation-2.json"
probe_a=
if ! probe_a="$(validated_product_probe first)"; then
  fail "first committed-product validation failed"
fi
readonly probe_a
/usr/bin/sleep 2
probe_b=
if ! probe_b="$(validated_product_probe second)"; then
  fail "second committed-product validation failed"
fi
readonly probe_b
[[ "${probe_a}" == "${probe_b}" ]] || fail "independent committed-product validations differ"
validation_base64=
capture_command_output_base64 validation_base64 \
  "canonical product-validation serialization" /usr/bin/printf '%s' "${probe_a}"
readonly validation_base64
validation_sha=
if ! validation_sha="$(/usr/bin/printf '%s' "${probe_a}" | \
  /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "canonical product-validation SHA query failed"
fi
readonly validation_sha
[[ "${validation_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "canonical product-validation SHA differs"
publish_canonical_json_base64 "${validation_base64}" "${validation_sha}" \
  "${validation_a}" "first product-validation publication"
publish_canonical_json_base64 "${validation_base64}" "${validation_sha}" \
  "${validation_b}" "second product-validation publication"

# The final parent/child observation is sampled before publishing either the
# terminal claim or SUCCESS.  A late Slurm drift must leave neither authority.
require_no_node_children "numeric child appeared before terminal seal"
terminal_parent_state=
if ! terminal_parent_state="$(/usr/bin/squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed before terminal seal"
fi
readonly terminal_parent_state
[[ "${terminal_parent_state}" == "${expected_parent_state}" ]] || fail "parent state changed before terminal seal: ${terminal_parent_state}"

readonly terminal="${attempt_root}/terminal.authority.json"
terminal_generation_transport=
require_sha256 "${env_bin}" "${env_sha}" "env tool before terminal generation"
require_sha256 "${capacity_python}" "${capacity_python_sha}" \
  "publisher Python before terminal generation"
capture_command_output_base64 terminal_generation_transport \
  "terminal authority generation" \
  "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
  HOME=/nonexistent/bernini-level-b-p2-00435-v4-terminal \
  "${capacity_python}" -I -S -B -c '
import hashlib,json,pathlib,stat,sys
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def unique(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise ValueError("duplicate JSON key")
  out[key]=value
 return out
def capacity_identity(path,expected,challenge,phase):
 p=pathlib.Path(path); info=p.lstat()
 assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1 and not p.is_symlink()
 raw=p.read_bytes(); assert hashlib.sha256(raw).hexdigest()==expected
 value=json.loads(raw.decode("utf-8"),object_pairs_hook=unique)
 assert raw==json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
 assert value.get("sample_phase")==phase and value.get("sample_challenge")==challenge
 return {"path":path,"sha256":expected,"challenge":challenge,"phase":phase,"mode":stat.S_IMODE(info.st_mode),"nlink":1,"size":len(raw),"terminal_check":"posthoc_identity_only","freshness_reauthorized_at_terminal":False}
receipt=pathlib.Path(sys.argv[2]).stat(); marker=pathlib.Path(sys.argv[3]).stat()
capacity_receipts={
 "foreground":capacity_identity(sys.argv[10],sys.argv[11],sys.argv[12],"foreground"),
 "controller":capacity_identity(sys.argv[13],sys.argv[14],sys.argv[15],"controller"),
 "step":capacity_identity(sys.argv[16],sys.argv[17],sys.argv[18],"step"),
}
out={"schema_version":"bernini-action-edit-level-b-p2-terminal-authority-v4","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v4","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":0,"output_mp4":{"path":sys.argv[1],"sha256":sha(sys.argv[1])},"output_receipt":{"path":sys.argv[2],"sha256":sha(sys.argv[2])},"output_commit_marker":{"path":sys.argv[3],"sha256":sha(sys.argv[3])},"receipt_inode_alias_marker_verified":receipt.st_dev==marker.st_dev and receipt.st_ino==marker.st_ino and receipt.st_nlink==2 and marker.st_nlink==2,"intent_sha256":sha(sys.argv[4]),"controller_status_sha256":sha(sys.argv[5]),"run_log_sha256":sha(sys.argv[6]),"validation_sha256":sha(sys.argv[7]),"two_identical_full_validations":sha(sys.argv[7])==sha(sys.argv[8]),"parent_state_at_terminal":sys.argv[9],"capacity_receipts":capacity_receipts,"three_independent_capacity_gates_verified":len({sys.argv[12],sys.argv[15],sys.argv[18]})==3,"step_capacity_freshly_validated_pre_torch_by_step_and_ranks":True,"step_capacity_post_child_archival_revalidation_freshness_disabled":True,"step_capacity_post_child_archival_revalidation_is_non_admission_evidence_only":True,"post_child_archival_revalidation_admission_authority":False,"terminal_capacity_receipt_checks_are_posthoc_identity_only":True,"full40_denoise_executed":True,"full_bernini_renderer_denoise_verified":True,"offline_product_inference_completed":True,"mp4_emitted":True,"committed_marker_required":True,"formal_training_started":False,"counts_as_d0":False,"scientific_claim_authorized":False,"promotion_authorized":False,"parent_untouched":sys.argv[9]=="RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8","automatic_relaunch_authorized":False}
assert out["two_identical_full_validations"] is True
assert out["receipt_inode_alias_marker_verified"] is True
assert out["three_independent_capacity_gates_verified"] is True
unsigned=dict(out); out["terminal_digest"]=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${output_mp4}" "${output_receipt}" "${output_marker}" "${intent}" \
  "${attempt_root}/controller.status.json" "${log_path}" "${validation_a}" \
  "${validation_b}" "${terminal_parent_state}" \
  "${foreground_capacity_receipt}" "${foreground_receipt_sha}" \
  "${foreground_capacity_challenge}" "${controller_capacity_receipt}" \
  "${controller_capacity_sha}" "${controller_capacity_challenge}" \
  "${step_capacity_receipt}" "${step_capacity_sha}" \
  "${step_capacity_challenge}"
terminal_raw=
decode_command_transport terminal_raw "${terminal_generation_transport}" \
  "terminal authority generation"
require_sha256 "${env_bin}" "${env_sha}" "env tool after terminal generation"
require_sha256 "${capacity_python}" "${capacity_python_sha}" \
  "publisher Python after terminal generation"
readonly terminal_raw
terminal_sha=
if ! terminal_sha="$(/usr/bin/printf '%s' "${terminal_raw}" | \
  /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "terminal authority SHA query failed"
fi
readonly terminal_sha
[[ "${terminal_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "terminal authority SHA differs"
publish_canonical_json_base64 "${terminal_generation_transport}" "${terminal_sha}" \
  "${terminal}" "terminal authority publication"
[[ -f "${terminal}" && ! -L "${terminal}" ]] || fail "terminal authority file differs"
require_stat_value "${terminal}" %a 444 "terminal authority mode"
require_stat_value "${terminal}" %h 1 "terminal authority link count"
require_sha256 "${terminal}" "${terminal_sha}" "terminal authority"

readonly success="${attempt_root}/SUCCESS"
success_transport=
capture_command_output_base64 success_transport "SUCCESS serialization" \
  /usr/bin/printf 'LEVEL_B_P2_00435_V4_COMPLETE=true\nterminal_authority_sha256=%s\nfull40_denoise_executed=true\nfull_bernini_renderer_denoise_verified=true\noffline_product_inference_completed=true\nmp4_emitted=true\ncommitted_marker_required=true\nreceipt_inode_alias_marker_verified=true\nformal_training_started=false\ncounts_as_d0=false\nscientific_claim_authorized=false\npromotion_authorized=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "${terminal_sha}"
readonly success_transport
success_sha=
if ! success_sha="$(/usr/bin/printf '%s' "${success_transport}" | \
  "${base64_bin}" -d | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "SUCCESS SHA query failed"
fi
readonly success_sha
[[ "${success_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "SUCCESS SHA differs"
publish_canonical_json_base64 "${success_transport}" "${success_sha}" \
  "${success}" "SUCCESS publication" bytes
require_sha256 "${success}" "${success_sha}" "SUCCESS"
printf 'Level-B P2 00435 one-shot complete; parent_untouched=true promotion=false\n'
