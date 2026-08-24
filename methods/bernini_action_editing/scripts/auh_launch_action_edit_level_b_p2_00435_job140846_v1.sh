#!/usr/bin/env bash
# One-shot controller for the PRE_D0 Level-B P2 00435 full-renderer product.
# It never retries and never cancels, releases, requeues, or signals job 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v1
readonly release_root="${experiment_root}/releases/${tag}"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly attempt_root="${experiment_root}/attempts/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly controller="${launch_root}/auh_launch_action_edit_level_b_p2_00435_job140846_v1.sh"
readonly step_payload="${launch_root}/auh_action_edit_level_b_p2_00435_step_v1.sh"
readonly step_payload_sha=43edf5f350c63e98653a33ea4e07b3a1fae3b917207c363d240f7594a3481cb1
readonly rank_exec="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v1.sh"
readonly rank_exec_sha=6ec31d8bbc35f960d52a1aae5ac9160f1002cf86f0893094d51ae2cea1981e26
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v1.py"
readonly bootstrap_sha=e00e3ece252ca6b37e00bb717419199f2371cf59b6e4123c59eb19f69a98a5d1
readonly launch_authority_core="${launch_root}/LAUNCH_AUTHORITY_CORE.json"
readonly launch_authority_core_sha=abb79221c0f0bf4ac1745e64247539eec35a7ce38987a121df6c5b92cfbe7490
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=8cf24d45f64eed4d6bc3e02b60d68ab997ab8f56ccdfe33849e04dc7c6f684bf
readonly renderer_sha=2b807aec19c17953a890ef76b9164b786af2b7f1912c32b2c33194c15ca29eed
readonly attempt_intent_sha=e3757167323e2f228bc9f404344cb40d419c6077aa6b04b3be037cfd3ce98f80
readonly output_mp4="${run_root}/00435ad621c44fac_p2_seed2026080821.mp4"
readonly output_receipt="${output_mp4}.receipt.json"
readonly output_marker="${output_mp4}.COMMITTED.json"
readonly job_name=bernini0817-level-b-p2-00435-v1

fail() {
  printf 'Level-B P2 controller refused: %s\n' "$*" >&2
  exit 95
}

node_children() {
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    awk -v prefix="${job_id}." \
      'index($0,prefix)==1 && substr($0,length(prefix)+1) ~ /^[0-9]+$/ {print $0}' |
    LC_ALL=C sort
}

await_child_teardown() {
  local observed poll
  for poll in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    observed="$(node_children)"
    [[ -z "${observed}" ]] && return 0
    sleep 2
  done
  fail "completed child remained visible: ${observed}"
}

write_status() {
  local exit_code="$1" parent_before="$2" parent_after="$3" intent="$4" log_path="$5"
  local output="${attempt_root}/controller.status.json" temporary="${attempt_root}/.controller.status.$$.tmp"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "controller status already exists"
  "${python_bin}" -I -B -c '
import json,sys
out={"schema_version":"bernini-action-edit-level-b-p2-controller-status-v1","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v1","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":int(sys.argv[1]),"parent_state_before":sys.argv[2],"parent_state_after":sys.argv[3],"intent_path":sys.argv[4],"run_log_path":sys.argv[5],"automatic_relaunch_authorized":False,"parent_control_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${exit_code}" "${parent_before}" "${parent_after}" "${intent}" "${log_path}" >"${temporary}"
  chmod 0444 "${temporary}"
  mv "${temporary}" "${output}"
}

for pending_sha in "${step_payload_sha}" "${rank_exec_sha}" "${bootstrap_sha}" \
  "${launch_authority_core_sha}" "${release_manifest_sha}" "${renderer_sha}" \
  "${attempt_intent_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ "$0" == "${controller}" ]] || fail "controller absolute path differs"
[[ -x "${controller}" && ! -L "${controller}" ]] || fail "controller file differs"
[[ "$(stat -c %a "${controller}")" == 555 && "$(stat -c %h "${controller}")" == 1 ]] || fail "controller topology differs"
[[ -d "${launch_root}" && ! -L "${launch_root}" && "$(stat -c %a "${launch_root}")" == 555 ]] || fail "launch root differs"
readonly expected_launch_entries=$'LAUNCH_AUTHORITY_CORE.json\naction_edit_level_b_p2_00435_bootstrap_0817_v1.py\nauh_action_edit_level_b_p2_00435_rank_exec_v1.sh\nauh_action_edit_level_b_p2_00435_step_v1.sh\nauh_launch_action_edit_level_b_p2_00435_job140846_v1.sh'
readonly observed_launch_entries="$(find "${launch_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_launch_entries}" == "${expected_launch_entries}" ]] || fail "launch exact closure differs"
for executable in "${controller}" "${step_payload}" "${rank_exec}"; do
  [[ -x "${executable}" && ! -L "${executable}" ]] || fail "launcher executable differs"
  [[ "$(stat -c %a "${executable}")" == 555 && "$(stat -c %h "${executable}")" == 1 ]] || fail "launcher executable topology differs"
done
[[ -f "${bootstrap}" && ! -L "${bootstrap}" && "$(stat -c %a "${bootstrap}")" == 444 && "$(stat -c %h "${bootstrap}")" == 1 ]] || fail "bootstrap topology differs"
[[ "$(sha256sum "${step_payload}" | awk '{print $1}')" == "${step_payload_sha}" ]] || fail "step SHA differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank SHA differs"
[[ "$(sha256sum "${bootstrap}" | awk '{print $1}')" == "${bootstrap_sha}" ]] || fail "bootstrap SHA differs"
[[ "$(sha256sum "${launch_authority_core}" | awk '{print $1}')" == "${launch_authority_core_sha}" ]] || fail "launch core SHA differs"
[[ "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || fail "release manifest SHA differs"

[[ -d "${attempt_root}" && ! -L "${attempt_root}" && "$(stat -c %a "${attempt_root}")" == 700 ]] || fail "attempt latch root differs"
[[ -z "$(find "${attempt_root}" -mindepth 1 -print -quit)" ]] || fail "attempt root is not fresh"
[[ -d "${run_root}" && ! -L "${run_root}" && "$(stat -c %a "${run_root}")" == 700 ]] || fail "run root differs"
[[ -z "$(find "${run_root}" -mindepth 1 -print -quit)" ]] || fail "run root is not fresh"
readonly parent_before="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
[[ "${parent_before}" == "${expected_parent_state}" ]] || fail "parent state differs before launch: ${parent_before}"
[[ -z "$(node_children)" ]] || fail "parent already has a numeric node279 child"

# Persistent atomic claim.  It is never removed, including on child failure.
readonly started="${attempt_root}/STARTED"
mkdir -m 0700 "${started}" 2>/dev/null || fail "attempt was already claimed"
readonly intent="${started}/intent.json"
readonly intent_tmp="${started}/.intent.$$.tmp"
"${python_bin}" -I -B -c '
import json,sys
out={"schema_version":"bernini-action-edit-level-b-p2-attempt-intent-v1","method":"bernini-action-edit-level-b-p2-00435-bootstrap-0817-v1","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v1","parent_job_id":140846,"node":"auh7-1b-gpu-279","job_name":"bernini0817-level-b-p2-00435-v1","release_root":sys.argv[1],"launch_root":sys.argv[2],"attempt_root":sys.argv[3],"run_root":sys.argv[4],"output_mp4":sys.argv[5],"source_video_sha256":"b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1","instruction_utf8_sha256":"cfe1e51a8b8ada76c5b1d6993cfb8d55cbc1f21fb0694a14ddb9c11133f74088","inference_seed":2026080821,"checkpoint_step":2,"checkpoint_parameter_sha256":"5f9c31e84ab9ec4330b07d86cb1a2fc79c7aa365f4bf88a9cdffc0c244dcaa3e","release_manifest_sha256":sys.argv[6],"renderer_sha256":sys.argv[7],"bootstrap_sha256":sys.argv[8],"step_payload_sha256":sys.argv[9],"rank_exec_sha256":sys.argv[10],"world_size":8,"dp_size":2,"sp_size":4,"host_memory_gib":64,"max_restarts":0,"committed_marker_required":True,"automatic_relaunch_authorized":False,"parent_control_authorized":False,"formal_training_started":False,"counts_as_d0":False,"promotion_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${release_root}" "${launch_root}" "${attempt_root}" "${run_root}" "${output_mp4}" \
  "${release_manifest_sha}" "${renderer_sha}" "${bootstrap_sha}" \
  "${step_payload_sha}" "${rank_exec_sha}" >"${intent_tmp}"
chmod 0444 "${intent_tmp}"
[[ "$(sha256sum "${intent_tmp}" | awk '{print $1}')" == "${attempt_intent_sha}" ]] || fail "canonical intent SHA differs"
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
readonly parent_after="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
write_status "${child_exit}" "${parent_before}" "${parent_after}" "${intent}" "${log_path}"
if (( child_exit != 0 )); then
  printf 'Level-B P2 child failed rc=%s; no retry is authorized\n' "${child_exit}" >&2
  tail -n 240 "${log_path}" >&2 || true
  exit "${child_exit}"
fi
[[ "${parent_after}" == "${expected_parent_state}" ]] || fail "parent state changed after child: ${parent_after}"
await_child_teardown

readonly expected_run_entries=$'00435ad621c44fac_p2_seed2026080821.mp4\n00435ad621c44fac_p2_seed2026080821.mp4.COMMITTED.json\n00435ad621c44fac_p2_seed2026080821.mp4.receipt.json'
readonly observed_run_entries="$(find "${run_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_run_entries}" == "${expected_run_entries}" ]] || fail "committed product exact closure differs"
for product_file in "${output_mp4}" "${output_receipt}" "${output_marker}"; do
  [[ -f "${product_file}" && ! -L "${product_file}" ]] || fail "committed product member differs"
  [[ "$(stat -c %a "${product_file}")" == 444 ]] || fail "committed product mode differs"
done
[[ "$(stat -c %h "${output_mp4}")" == 1 ]] || fail "committed MP4 link count differs"
[[ "$(stat -c %h "${output_receipt}")" == 2 ]] || fail "committed receipt link count differs"
[[ "$(stat -c %h "${output_marker}")" == 2 ]] || fail "commit-marker link count differs"
[[ "$(stat -c '%d:%i' "${output_receipt}")" == "$(stat -c '%d:%i' "${output_marker}")" ]] || fail "COMMITTED marker is not the exact receipt inode alias"
[[ "$(sha256sum "${output_receipt}" | awk '{print $1}')" == "$(sha256sum "${output_marker}" | awk '{print $1}')" ]] || fail "receipt/COMMITTED alias bytes differ"

# Two complete, side-effect-free validations must agree before terminal seal.
readonly validation_a="${started}/product-validation-1.json"
readonly validation_b="${started}/product-validation-2.json"
readonly probe_a="$("${python_bin}" -I -B "${bootstrap}" validate-product)"
sleep 2
readonly probe_b="$("${python_bin}" -I -B "${bootstrap}" validate-product)"
[[ "${probe_a}" == "${probe_b}" ]] || fail "independent committed-product validations differ"
printf '%s' "${probe_a}" >"${validation_a}"
printf '%s' "${probe_b}" >"${validation_b}"
chmod 0444 "${validation_a}" "${validation_b}"

readonly terminal="${attempt_root}/terminal.authority.json"
readonly terminal_tmp="${attempt_root}/.terminal.$$.tmp"
"${python_bin}" -I -B -c '
import hashlib,json,pathlib,sys
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
receipt=pathlib.Path(sys.argv[2]).stat(); marker=pathlib.Path(sys.argv[3]).stat()
out={"schema_version":"bernini-action-edit-level-b-p2-terminal-authority-v1","authority":"PRE_D0_ENGINEERING_ONLY","tag":"fresh-world8-level-b-p2-00435-v1","parent_job_id":140846,"node":"auh7-1b-gpu-279","child_exit_code":0,"output_mp4":{"path":sys.argv[1],"sha256":sha(sys.argv[1])},"output_receipt":{"path":sys.argv[2],"sha256":sha(sys.argv[2])},"output_commit_marker":{"path":sys.argv[3],"sha256":sha(sys.argv[3])},"receipt_inode_alias_marker_verified":receipt.st_dev==marker.st_dev and receipt.st_ino==marker.st_ino and receipt.st_nlink==2 and marker.st_nlink==2,"intent_sha256":sha(sys.argv[4]),"controller_status_sha256":sha(sys.argv[5]),"run_log_sha256":sha(sys.argv[6]),"validation_sha256":sha(sys.argv[7]),"two_identical_full_validations":sha(sys.argv[7])==sha(sys.argv[8]),"full40_denoise_executed":True,"full_bernini_renderer_denoise_verified":True,"offline_product_inference_completed":True,"mp4_emitted":True,"committed_marker_required":True,"formal_training_started":False,"counts_as_d0":False,"scientific_claim_authorized":False,"promotion_authorized":False,"parent_untouched":True,"automatic_relaunch_authorized":False}
assert out["two_identical_full_validations"] is True
assert out["receipt_inode_alias_marker_verified"] is True
unsigned=dict(out); out["terminal_digest"]=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${output_mp4}" "${output_receipt}" "${output_marker}" "${intent}" \
  "${attempt_root}/controller.status.json" "${log_path}" "${validation_a}" \
  "${validation_b}" >"${terminal_tmp}"
chmod 0444 "${terminal_tmp}"
mv "${terminal_tmp}" "${terminal}"

readonly success_tmp="${attempt_root}/.SUCCESS.$$.tmp"
printf 'LEVEL_B_P2_00435_COMPLETE=true\nterminal_authority_sha256=%s\nfull40_denoise_executed=true\nfull_bernini_renderer_denoise_verified=true\noffline_product_inference_completed=true\nmp4_emitted=true\ncommitted_marker_required=true\nreceipt_inode_alias_marker_verified=true\nformal_training_started=false\ncounts_as_d0=false\nscientific_claim_authorized=false\npromotion_authorized=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "$(sha256sum "${terminal}" | awk '{print $1}')" >"${success_tmp}"
chmod 0444 "${success_tmp}"
mv "${success_tmp}" "${attempt_root}/SUCCESS"
[[ -z "$(node_children)" ]] || fail "numeric child remained after terminal seal"
[[ "$(squeue -h -j "${job_id}" -o '%T|%N|%b')" == "${expected_parent_state}" ]] || fail "parent state changed at terminal seal"
printf 'Level-B P2 00435 one-shot complete; parent_untouched=true promotion=false\n'
