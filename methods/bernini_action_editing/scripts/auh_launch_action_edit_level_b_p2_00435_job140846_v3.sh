#!/usr/bin/env bash
# One-shot controller for the PRE_D0 Level-B P2 00435 full-renderer product.
# It never retries and never cancels, releases, requeues, or signals job 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly base64_bin=/usr/bin/base64
readonly base64_sha=b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898
readonly base64_size=35336
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v3
readonly release_root="${experiment_root}/releases/${tag}"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly attempt_root="${experiment_root}/attempts/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly controller="${launch_root}/auh_launch_action_edit_level_b_p2_00435_job140846_v3.sh"
readonly step_payload="${launch_root}/auh_action_edit_level_b_p2_00435_step_v3.sh"
readonly step_payload_sha=f1d2714aac69180c45ef9925648e1adfb912479fe160b887e7aa2de4d98b95c6
readonly rank_exec="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v3.sh"
readonly rank_exec_sha=7a8daaf1bf0e7ad20f7881b6e2138b0667ecb6606566a3693fe3f9fcd26ce5cb
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v3.py"
readonly bootstrap_sha=0c7d0e28dedc9a22fe543faec5f5c4f4abba628445d1f8a7f72c9138ccc6fe00
readonly launch_authority_core="${launch_root}/LAUNCH_AUTHORITY_CORE.json"
readonly launch_authority_core_sha=a96604ad558ed0a720e3ce71f84fd2bda0c4a4a6191dae9ef69045197daf0e3c
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=380b433d4be8c349bb79c8eb3914442136e153c2dccd4cb57ff25db9f7688a16
readonly renderer_sha=8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6
readonly static_preflight_sha=2fb4d9d5b1e8e875025260a7287d0c00178e030ac79f1db5d8f2611fcf0618f1
readonly static_preflight_size=29205
readonly static_preflight_base64_sha=594fabf3b03a53ea28e977c6fdc2c562e3d1e209701f01814a3d77baf0fa4417
readonly static_preflight_base64_size=38940
readonly attempt_intent_sha=cf9ae4d1179556be06740f6b57608d6d0bab1055b00817c562bb288c373ca594
readonly output_mp4="${run_root}/00435ad621c44fac_p2_seed2026080821_v3.mp4"
readonly output_receipt="${output_mp4}.receipt.json"
readonly output_marker="${output_mp4}.COMMITTED.json"
readonly job_name=bernini0817-level-b-p2-00435-v3

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
  if ! stat_observed="$(stat -c "${stat_format}" "${stat_path}")"; then
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
  if ! sha_observed="$(sha256sum "${sha_path}" | awk '{print $1}')"; then
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

node_children() {
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    awk -v prefix="${job_id}." \
      'index($0,prefix)==1 && substr($0,length(prefix)+1) ~ /^[0-9]+$/ {print $0}' |
    LC_ALL=C sort
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
    sleep 2
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
  local status_output="${attempt_root}/controller.status.json"
  local status_temporary="${attempt_root}/.controller.status.$$.tmp"
  [[ ! -e "${status_output}" && ! -L "${status_output}" ]] || fail "controller status already exists"
  "${python_bin}" -I -B -c '
import json,sys
out={"schema_version":"bernini-action-edit-level-b-p2-controller-status-v3","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v3","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":int(sys.argv[1]),"parent_state_before":sys.argv[2],"parent_state_after":sys.argv[3],"intent_path":sys.argv[4],"run_log_path":sys.argv[5],"automatic_relaunch_authorized":False,"parent_control_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${status_exit_code}" "${status_parent_before}" "${status_parent_after}" \
  "${status_intent_path}" "${status_log_path}" >"${status_temporary}"
  chmod 0444 "${status_temporary}"
  mv "${status_temporary}" "${status_output}"
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
  local probe_sentinel=__LEVEL_B_P2_00435_V3_PRODUCT_VALIDATION_PIPESTATUS_
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
  if ! probe_raw="$("${python_bin}" -I -B -c '
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
assert value["schema_version"]=="bernini-action-edit-level-b-p2-product-validation-v3"
assert value["method"]=="bernini-action-edit-level-b-p2-00435-bootstrap-0817-v3"
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
' "${probe_base64}" "${output_mp4}")"; then
    fail "${probe_label} product-validation canonical claim closure differs"
  fi
  [[ -n "${probe_raw}" && "${probe_raw}" != *$'\n'* ]] || \
    fail "${probe_label} product-validation canonical bytes differ"
  printf '%s' "${probe_raw}"
}

for pending_sha in "${step_payload_sha}" "${rank_exec_sha}" "${bootstrap_sha}" \
  "${launch_authority_core_sha}" "${release_manifest_sha}" "${renderer_sha}" \
  "${static_preflight_sha}" "${static_preflight_base64_sha}" \
  "${attempt_intent_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ "${static_preflight_size}" =~ ^[1-9][0-9]*$ ]] || fail "static preflight size differs"
[[ "${static_preflight_base64_size}" =~ ^[1-9][0-9]*$ ]] || fail "static preflight base64 size differs"
[[ "$0" == "${controller}" ]] || fail "controller absolute path differs"
[[ -x "${controller}" && ! -L "${controller}" ]] || fail "controller file differs"
require_stat_value "${controller}" %a 555 "controller mode"
require_stat_value "${controller}" %h 1 "controller link count"
[[ -d "${launch_root}" && ! -L "${launch_root}" ]] || fail "launch root differs"
require_stat_value "${launch_root}" %a 555 "launch root mode"
readonly expected_launch_entries=$'LAUNCH_AUTHORITY_CORE.json\naction_edit_level_b_p2_00435_bootstrap_0817_v3.py\nauh_action_edit_level_b_p2_00435_rank_exec_v3.sh\nauh_action_edit_level_b_p2_00435_step_v3.sh\nauh_launch_action_edit_level_b_p2_00435_job140846_v3.sh'
observed_launch_entries=
if ! observed_launch_entries="$(find "${launch_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"; then
  fail "launch exact closure query failed"
fi
readonly observed_launch_entries
[[ "${observed_launch_entries}" == "${expected_launch_entries}" ]] || fail "launch exact closure differs"
for executable in "${controller}" "${step_payload}" "${rank_exec}"; do
  [[ -x "${executable}" && ! -L "${executable}" ]] || fail "launcher executable differs"
  require_stat_value "${executable}" %a 555 "launcher executable mode"
  require_stat_value "${executable}" %h 1 "launcher executable link count"
done
[[ -f "${bootstrap}" && ! -L "${bootstrap}" ]] || fail "bootstrap topology differs"
require_stat_value "${bootstrap}" %a 444 "bootstrap mode"
require_stat_value "${bootstrap}" %h 1 "bootstrap link count"
[[ -x "${base64_bin}" && ! -L "${base64_bin}" ]] || fail "base64 tool path differs"
base64_resolved=
if ! base64_resolved="$(readlink -f "${base64_bin}")"; then
  fail "base64 tool canonical-path query failed"
fi
readonly base64_resolved
[[ "${base64_resolved}" == "${base64_bin}" ]] || fail "base64 tool path differs"
require_stat_value "${base64_bin}" %a 755 "base64 tool mode"
require_stat_value "${base64_bin}" %h 1 "base64 tool link count"
require_stat_value "${base64_bin}" %s "${base64_size}" "base64 tool size"
require_sha256 "${base64_bin}" "${base64_sha}" "base64 tool"
require_sha256 "${step_payload}" "${step_payload_sha}" "step payload"
require_sha256 "${rank_exec}" "${rank_exec_sha}" "rank wrapper"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap"
require_sha256 "${launch_authority_core}" "${launch_authority_core_sha}" "launch core"
require_sha256 "${release_manifest}" "${release_manifest_sha}" "release manifest"

[[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt latch root differs"
require_stat_value "${attempt_root}" %a 700 "attempt root mode"
attempt_initial_entry=
if ! attempt_initial_entry="$(find "${attempt_root}" -mindepth 1 -print -quit)"; then
  fail "attempt freshness query failed"
fi
readonly attempt_initial_entry
[[ -z "${attempt_initial_entry}" ]] || fail "attempt root is not fresh"
[[ -d "${run_root}" && ! -L "${run_root}" ]] || fail "run root differs"
require_stat_value "${run_root}" %a 700 "run root mode"
run_initial_entry=
if ! run_initial_entry="$(find "${run_root}" -mindepth 1 -print -quit)"; then
  fail "run freshness query failed"
fi
readonly run_initial_entry
[[ -z "${run_initial_entry}" ]] || fail "run root is not fresh"
parent_before=
if ! parent_before="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed before launch"
fi
readonly parent_before
[[ "${parent_before}" == "${expected_parent_state}" ]] || fail "parent state differs before launch: ${parent_before}"
require_no_node_children "parent already has a numeric node279 child"

# CPU-only, read/import/descriptor inspection of the exact sealed release.
# This runs before the persistent STARTED latch and before every output/log
# redirection.  Its Python audit hook denies filesystem mutation, process
# creation, and network access; the receipt is held in memory only.
readonly static_preflight_frame_sentinel=__LEVEL_B_P2_00435_V3_STATIC_PREFLIGHT_PIPESTATUS_
static_preflight_frame=
if ! static_preflight_frame="$(
  set +e
  /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    LC_ALL=C \
    LANG=C \
    HOME=/nonexistent/bernini-level-b-p2-00435-v3 \
    TMPDIR=/nonexistent/bernini-level-b-p2-00435-v3/tmp \
    XDG_CACHE_HOME=/nonexistent/bernini-level-b-p2-00435-v3/cache \
    HF_HOME=/nonexistent/bernini-level-b-p2-00435-v3/huggingface \
    TRANSFORMERS_CACHE=/nonexistent/bernini-level-b-p2-00435-v3/transformers \
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
if ! static_preflight_observed_base64_sha="$(printf '%s' "${static_preflight_base64}" | sha256sum | awk '{print $1}')"; then
  fail "CPU static preflight base64 SHA query failed"
fi
readonly static_preflight_observed_base64_sha
[[ "${static_preflight_observed_base64_sha}" == "${static_preflight_base64_sha}" ]] || fail "CPU static preflight combined stdout/stderr base64 SHA differs"
parent_after_static_preflight=
if ! parent_after_static_preflight="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed after CPU static preflight"
fi
readonly parent_after_static_preflight
[[ "${parent_after_static_preflight}" == "${parent_before}" ]] || fail "parent changed during CPU static preflight"
require_no_node_children "numeric node279 child appeared during CPU static preflight"
attempt_after_static_preflight=
if ! attempt_after_static_preflight="$(find "${attempt_root}" -mindepth 1 -print -quit)"; then
  fail "attempt post-preflight closure query failed"
fi
readonly attempt_after_static_preflight
[[ -z "${attempt_after_static_preflight}" ]] || fail "attempt root changed during CPU static preflight"
run_after_static_preflight=
if ! run_after_static_preflight="$(find "${run_root}" -mindepth 1 -print -quit)"; then
  fail "run post-preflight closure query failed"
fi
readonly run_after_static_preflight
[[ -z "${run_after_static_preflight}" ]] || fail "run root changed during CPU static preflight"

# Persistent atomic claim.  It is never removed, including on child failure.
readonly started="${attempt_root}/STARTED"
mkdir -m 0700 "${started}" 2>/dev/null || fail "attempt was already claimed"
readonly intent="${started}/intent.json"
readonly intent_tmp="${started}/.intent.$$.tmp"
"${python_bin}" -I -B -c '
import json,sys
out={"schema_version":"bernini-action-edit-level-b-p2-attempt-intent-v3","method":"bernini-action-edit-level-b-p2-00435-bootstrap-0817-v3","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v3","parent_job_id":140846,"node":"auh7-1b-gpu-279","job_name":"bernini0817-level-b-p2-00435-v3","release_root":sys.argv[1],"launch_root":sys.argv[2],"attempt_root":sys.argv[3],"run_root":sys.argv[4],"output_mp4":sys.argv[5],"source_video_sha256":"b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1","instruction_utf8_sha256":"cfe1e51a8b8ada76c5b1d6993cfb8d55cbc1f21fb0694a14ddb9c11133f74088","inference_seed":2026080821,"checkpoint_step":2,"checkpoint_parameter_sha256":"5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e","release_manifest_sha256":sys.argv[6],"renderer_sha256":sys.argv[7],"bootstrap_sha256":sys.argv[8],"step_payload_sha256":sys.argv[9],"rank_exec_sha256":sys.argv[10],"static_preflight_stdout_sha256":sys.argv[11],"static_preflight_stdout_size":int(sys.argv[12]),"world_size":8,"dp_size":2,"sp_size":4,"host_memory_gib":64,"max_restarts":0,"committed_marker_required":True,"automatic_relaunch_authorized":False,"parent_control_authorized":False,"formal_training_started":False,"counts_as_d0":False,"promotion_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${release_root}" "${launch_root}" "${attempt_root}" "${run_root}" "${output_mp4}" \
  "${release_manifest_sha}" "${renderer_sha}" "${bootstrap_sha}" \
  "${step_payload_sha}" "${rank_exec_sha}" "${static_preflight_sha}" \
  "${static_preflight_size}" >"${intent_tmp}"
chmod 0444 "${intent_tmp}"
require_sha256 "${intent_tmp}" "${attempt_intent_sha}" "canonical intent"
mv "${intent_tmp}" "${intent}"
sleep 2

readonly log_path="${attempt_root}/run.log"
set +e
set -o noclobber
/usr/bin/srun --jobid="${job_id}" --overlap --exact --nodes=1 --ntasks=1 \
  --nodelist="${node}" --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 \
  --job-name="${job_name}" --kill-on-bad-exit=1 \
  "${step_payload}" "${launch_authority_core}" "${launch_authority_core_sha}" \
  "${intent}" "${attempt_intent_sha}" >"${log_path}" 2>&1
readonly child_exit=$?
set +o noclobber
set -e
parent_after=
if ! parent_after="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed after child"
fi
readonly parent_after
write_status "${child_exit}" "${parent_before}" "${parent_after}" "${intent}" "${log_path}"
if (( child_exit != 0 )); then
  printf 'Level-B P2 child failed rc=%s; no retry is authorized\n' "${child_exit}" >&2
  tail -n 240 "${log_path}" >&2 || true
  exit "${child_exit}"
fi
[[ "${parent_after}" == "${expected_parent_state}" ]] || fail "parent state changed after child: ${parent_after}"
await_child_teardown

readonly expected_run_entries=$'00435ad621c44fac_p2_seed2026080821_v3.mp4\n00435ad621c44fac_p2_seed2026080821_v3.mp4.COMMITTED.json\n00435ad621c44fac_p2_seed2026080821_v3.mp4.receipt.json'
observed_run_entries=
if ! observed_run_entries="$(find "${run_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"; then
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
if ! receipt_dev_ino="$(stat -c '%d:%i' "${output_receipt}")"; then
  fail "receipt inode identity query failed"
fi
readonly receipt_dev_ino
marker_dev_ino=
if ! marker_dev_ino="$(stat -c '%d:%i' "${output_marker}")"; then
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
sleep 2
probe_b=
if ! probe_b="$(validated_product_probe second)"; then
  fail "second committed-product validation failed"
fi
readonly probe_b
[[ "${probe_a}" == "${probe_b}" ]] || fail "independent committed-product validations differ"
printf '%s' "${probe_a}" >"${validation_a}"
printf '%s' "${probe_b}" >"${validation_b}"
chmod 0444 "${validation_a}" "${validation_b}"

# The final parent/child observation is sampled before publishing either the
# terminal claim or SUCCESS.  A late Slurm drift must leave neither authority.
require_no_node_children "numeric child appeared before terminal seal"
terminal_parent_state=
if ! terminal_parent_state="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"; then
  fail "parent state query failed before terminal seal"
fi
readonly terminal_parent_state
[[ "${terminal_parent_state}" == "${expected_parent_state}" ]] || fail "parent state changed before terminal seal: ${terminal_parent_state}"

readonly terminal="${attempt_root}/terminal.authority.json"
readonly terminal_tmp="${attempt_root}/.terminal.$$.tmp"
"${python_bin}" -I -B -c '
import hashlib,json,pathlib,sys
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt=pathlib.Path(sys.argv[2]).stat(); marker=pathlib.Path(sys.argv[3]).stat()
out={"schema_version":"bernini-action-edit-level-b-p2-terminal-authority-v3","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v3","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":0,"output_mp4":{"path":sys.argv[1],"sha256":sha(sys.argv[1])},"output_receipt":{"path":sys.argv[2],"sha256":sha(sys.argv[2])},"output_commit_marker":{"path":sys.argv[3],"sha256":sha(sys.argv[3])},"receipt_inode_alias_marker_verified":receipt.st_dev==marker.st_dev and receipt.st_ino==marker.st_ino and receipt.st_nlink==2 and marker.st_nlink==2,"intent_sha256":sha(sys.argv[4]),"controller_status_sha256":sha(sys.argv[5]),"run_log_sha256":sha(sys.argv[6]),"validation_sha256":sha(sys.argv[7]),"two_identical_full_validations":sha(sys.argv[7])==sha(sys.argv[8]),"parent_state_at_terminal":sys.argv[9],"full40_denoise_executed":True,"full_bernini_renderer_denoise_verified":True,"offline_product_inference_completed":True,"mp4_emitted":True,"committed_marker_required":True,"formal_training_started":False,"counts_as_d0":False,"scientific_claim_authorized":False,"promotion_authorized":False,"parent_untouched":sys.argv[9]=="RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8","automatic_relaunch_authorized":False}
assert out["two_identical_full_validations"] is True
assert out["receipt_inode_alias_marker_verified"] is True
unsigned=dict(out); out["terminal_digest"]=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${output_mp4}" "${output_receipt}" "${output_marker}" "${intent}" \
  "${attempt_root}/controller.status.json" "${log_path}" "${validation_a}" \
  "${validation_b}" "${terminal_parent_state}" >"${terminal_tmp}"
chmod 0444 "${terminal_tmp}"
mv "${terminal_tmp}" "${terminal}"
[[ -f "${terminal}" && ! -L "${terminal}" ]] || fail "terminal authority file differs"
require_stat_value "${terminal}" %a 444 "terminal authority mode"
require_stat_value "${terminal}" %h 1 "terminal authority link count"
terminal_sha=
capture_sha256 "${terminal}" terminal_sha "terminal authority"
readonly terminal_sha

readonly success_tmp="${attempt_root}/.SUCCESS.$$.tmp"
printf 'LEVEL_B_P2_00435_V3_COMPLETE=true\nterminal_authority_sha256=%s\nfull40_denoise_executed=true\nfull_bernini_renderer_denoise_verified=true\noffline_product_inference_completed=true\nmp4_emitted=true\ncommitted_marker_required=true\nreceipt_inode_alias_marker_verified=true\nformal_training_started=false\ncounts_as_d0=false\nscientific_claim_authorized=false\npromotion_authorized=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "${terminal_sha}" >"${success_tmp}"
chmod 0444 "${success_tmp}"
mv "${success_tmp}" "${attempt_root}/SUCCESS"
printf 'Level-B P2 00435 one-shot complete; parent_untouched=true promotion=false\n'
