#!/usr/bin/env bash
# One-shot A then B controller for the frozen PRE_D0 Level-A P2 consumer.
# It never retries and never cancels, releases, requeues, or signals parent
# allocation 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly expected_parent_state='RUNNING|auh7-1b-gpu-[246-248,279]|gres/gpu:mi210:8'
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-a-r2-p2-launchbound-v2
readonly release_root="${experiment_root}/releases/${tag}"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly attempt_root="${experiment_root}/attempts/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=f9e9f8542ec701cc9890fed919695980b989fd6d731eb914a5588edb1de4eeaa
readonly driver="${release_root}/action_edit_fresh_world8_level_a_driver_0817_v1.py"
readonly driver_sha=6435c6bb06a79cfcb407c137571404e5962e0de50e8082e7bd600e4618c05ea4
readonly consumer_sha=8bf0a9e48e0b2443a8e2f8e0744d08591226a167ac6ace45ee513481f5a97b3a
readonly product_sha=b16d8aef25b35df13e8294ef387e4d334170af65c2f43ece9894142d7cadac14
readonly controller="${launch_root}/auh_launch_fresh_world8_level_a_r2_p2_node279_job140846_v1.sh"
readonly step_payload="${launch_root}/auh_fresh_world8_level_a_r2_p2_node279_step_v1.sh"
readonly step_payload_sha=421d2dc391833f34dfce7370480b1b807dac69c6373e5f596610f13a5721dfa6
readonly rank_exec="${launch_root}/auh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh"
readonly rank_exec_sha=37a099285453265f3442da10d14b577f3e31fd60ac01b9399356d2d9966b00c8
readonly launch_authority_core="${launch_root}/LAUNCH_AUTHORITY_CORE.json"
readonly launch_authority_core_sha=f641187bb0f09a51ceb05025322bc7fe18ad5ae643ea29b39790806d5f2a879b
readonly intent_a_sha=735cf18e333651637d4aff5e421ed9ff85d175cca509afbb8500e34fe81e536b
readonly intent_b_sha=eab26aedaabfb4ff8a9e34e7bf1efd8f0157453fbe4e393e07ad51264984d072
readonly job_name_a=bernini0817-level-a-launchbound-v2-A
readonly job_name_b=bernini0817-level-a-launchbound-v2-B

fail() {
  printf 'Level-A P2 node279 controller refused: %s\n' "$*" >&2
  exit 95
}

node_children() {
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    awk -v prefix="${job_id}." \
      'index($0, prefix) == 1 && substr($0, length(prefix) + 1) ~ /^[0-9]+$/ { print $0 }' |
    LC_ALL=C sort
}

await_own_child_teardown() {
  local observation poll
  for poll in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    observation="$(node_children)"
    [[ -z "${observation}" ]] && return 0
    sleep 2
  done
  fail "completed child did not leave node279 step view: ${observation}"
}

await_two_full_receipt_validations() {
  local label="$1" receipt="$2" intent="$3" intent_sha="$4"
  local started="$5" poll first_probe second_probe first_status second_status
  local first_probe_sha second_probe_sha first_out second_out
  first_out="${started}/receipt-validation-1.json"
  second_out="${started}/receipt-validation-2.json"
  [[ ! -e "${first_out}" && ! -L "${first_out}" ]] || fail "fixed validation-1 output already exists"
  [[ ! -e "${second_out}" && ! -L "${second_out}" ]] || fail "fixed validation-2 output already exists"
  # Every bounded round performs two complete read-only probes.  Missing or
  # transient visibility, including a failed second probe, has no filesystem
  # side effect and advances to the next round; this function never invokes
  # srun and cannot relaunch either child.
  for poll in $(seq 1 60); do
    set +e
    first_probe="$("${python_bin}" -I -B "${driver}" validate-receipt \
      --release-manifest "${release_manifest}" \
      --expected-release-manifest-sha256 "${release_manifest_sha}" \
      --expected-driver-source-sha256 "${driver_sha}" \
      --attempt-label "${label}" \
      --launch-authority-core "${launch_authority_core}" \
      --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
      --attempt-intent "${intent}" \
      --expected-attempt-intent-sha256 "${intent_sha}" \
      --receipt "${receipt}" 2>/dev/null)"
    first_status=$?
    sleep 2
    second_probe="$("${python_bin}" -I -B "${driver}" validate-receipt \
      --release-manifest "${release_manifest}" \
      --expected-release-manifest-sha256 "${release_manifest_sha}" \
      --expected-driver-source-sha256 "${driver_sha}" \
      --attempt-label "${label}" \
      --launch-authority-core "${launch_authority_core}" \
      --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
      --attempt-intent "${intent}" \
      --expected-attempt-intent-sha256 "${intent_sha}" \
      --receipt "${receipt}" 2>/dev/null)"
    second_status=$?
    set -e
    (( first_status == 0 && second_status == 0 )) || continue
    first_probe_sha="$(printf '%s' "${first_probe}" | sha256sum | awk '{print $1}')"
    second_probe_sha="$(printf '%s' "${second_probe}" | sha256sum | awk '{print $1}')"
    [[ "${first_probe}" == "${second_probe}" && \
       "${first_probe_sha}" == "${second_probe_sha}" ]] || continue
    # Only one wholly successful dry round may publish the fixed pair.  The
    # driver revalidates the live receipt and uses link(2) create-only commits.
    "${python_bin}" -I -B "${driver}" publish-validation-pair \
      --release-manifest "${release_manifest}" \
      --expected-release-manifest-sha256 "${release_manifest_sha}" \
      --expected-driver-source-sha256 "${driver_sha}" \
      --attempt-label "${label}" \
      --launch-authority-core "${launch_authority_core}" \
      --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
      --attempt-intent "${intent}" \
      --expected-attempt-intent-sha256 "${intent_sha}" \
      --receipt "${receipt}" \
      --validation-json-a "${first_probe}" \
      --validation-json-b "${second_probe}" \
      --output-validation-a "${first_out}" \
      --output-validation-b "${second_out}" >/dev/null || \
      fail "create-only fixed validation-pair publication failed"
    return 0
  done
  fail "receipt did not pass one side-effect-free two-probe round in 60 x 2s"
}

run_attempt() {
  local label="$1"
  local attempt="${attempt_root}/${label}"
  local output="${run_root}/${label}"
  local parent_before child_before started intent intent_sha job_name intent_tmp log_path
  local status parent_after receipt controller_status terminal_authority success
  [[ "${label}" == A || "${label}" == B ]] || fail "attempt label differs"
  if [[ "${label}" == A ]]; then
    intent_sha="${intent_a_sha}"
    job_name="${job_name_a}"
  else
    intent_sha="${intent_b_sha}"
    job_name="${job_name_b}"
  fi
  [[ -d "${attempt}" && ! -L "${attempt}" ]] || fail "attempt ${label} latch root differs"
  [[ "$(stat -c %a "${attempt}")" == 700 ]] || fail "attempt ${label} latch mode differs"
  [[ -z "$(find "${attempt}" -mindepth 1 -print -quit)" ]] || fail "attempt ${label} is not fresh"
  [[ ! -e "${output}" ]] || fail "attempt ${label} output already exists"
  parent_before="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
  [[ "${parent_before}" == "${expected_parent_state}" ]] || fail "parent allocation state differs before ${label}: ${parent_before}"
  child_before="$(node_children)"
  [[ -z "${child_before}" ]] || fail "node279 already has a numeric child before ${label}: ${child_before}"

  # This mkdir is the attempt's persistent, atomic, one-shot claim.  Any
  # later failure leaves STARTED in place and cannot become an automatic retry.
  started="${attempt}/STARTED"
  mkdir -m 0700 "${started}" 2>/dev/null || fail "attempt ${label} was already claimed"
  intent="${started}/intent.json"
  intent_tmp="${started}/.intent.$$.tmp"
  "${python_bin}" -I -B -c '
import json,sys
keys=("attempt","job_name","release_root","launch_root","attempt_root","output_root","release_manifest_sha256","driver_sha256","consumer_sha256","product_bridge_sha256","step_payload_sha256","rank_exec_sha256")
v=dict(zip(keys,sys.argv[1:]))
out={"schema_version":"bernini-action-edit-fresh-world8-level-a-attempt-intent-v2","method":"bernini-action-edit-fresh-world8-level-a-driver-0817-v1","authority":"PRE_D0_ENGINEERING_ONLY","attempt":v["attempt"],"parent_job_id":140846,"node":"auh7-1b-gpu-279","job_name":v["job_name"],"release_root":v["release_root"],"launch_root":v["launch_root"],"attempt_root":v["attempt_root"],"output_root":v["output_root"],"checkpoint_step":2,"world_size":8,"dp_size":2,"sp_size":4,"release_manifest_sha256":v["release_manifest_sha256"],"driver_sha256":v["driver_sha256"],"consumer_sha256":v["consumer_sha256"],"product_bridge_sha256":v["product_bridge_sha256"],"step_payload_sha256":v["step_payload_sha256"],"rank_exec_sha256":v["rank_exec_sha256"],"automatic_relaunch_authorized":False,"parent_control_authorized":False}
sys.stdout.buffer.write(json.dumps(out,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())
' "${label}" "${job_name}" "${release_root}" "${launch_root}" "${attempt}" \
    "${output}" "${release_manifest_sha}" "${driver_sha}" "${consumer_sha}" \
    "${product_sha}" "${step_payload_sha}" "${rank_exec_sha}" >"${intent_tmp}"
  chmod 0444 "${intent_tmp}"
  [[ "$(sha256sum "${intent_tmp}" | awk '{print $1}')" == "${intent_sha}" ]] || \
    fail "attempt ${label} canonical intent SHA differs"
  mv "${intent_tmp}" "${intent}"
  [[ "$(stat -c %a "${intent}")" == 444 && "$(stat -c %h "${intent}")" == 1 ]] || \
    fail "attempt ${label} intent topology differs"
  # Preserve a deploy-to-node visibility boundary before the one-shot child.
  sleep 2

  printf 'Level-A P2 one-shot attempt=%s job=%s node=%s parent_untouched=true\n' \
    "${label}" "${job_id}" "${node}"
  log_path="${attempt}/run.log"
  set +e
  set -o noclobber
  /usr/bin/srun --jobid="${job_id}" --overlap --exact --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=60G --gres=gpu:mi210:8 \
    --job-name="${job_name}" --kill-on-bad-exit=1 \
    "${step_payload}" "${label}" "${launch_authority_core}" \
    "${launch_authority_core_sha}" "${intent}" "${intent_sha}" >"${log_path}" 2>&1
  status=$?
  set +o noclobber
  set -e

  parent_after="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
  controller_status="${attempt}/controller.status.json"
  "${python_bin}" -I -B "${driver}" write-controller-status \
    --release-manifest "${release_manifest}" \
    --expected-release-manifest-sha256 "${release_manifest_sha}" \
    --expected-driver-source-sha256 "${driver_sha}" \
    --attempt-label "${label}" \
    --launch-authority-core "${launch_authority_core}" \
    --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
    --attempt-intent "${intent}" --expected-attempt-intent-sha256 "${intent_sha}" \
    --child-exit "${status}" --parent-state-before "${parent_before}" \
    --parent-state-after "${parent_after}" \
    --output-controller-status "${controller_status}" >/dev/null || \
    fail "controller status sealing failed"
  if (( status != 0 )); then
    printf 'Level-A P2 attempt=%s failed rc=%s; no retry is authorized\n' "${label}" "${status}" >&2
    tail -n 240 "${log_path}" >&2 || true
    exit "${status}"
  fi
  [[ "${parent_after}" == "${expected_parent_state}" ]] || fail "parent allocation state differs after ${label}: ${parent_after}"
  await_own_child_teardown
  receipt="${output}/bundle.consumer_receipt"
  await_two_full_receipt_validations \
    "${label}" "${receipt}" "${intent}" "${intent_sha}" "${started}"
  terminal_authority="${attempt}/terminal.authority.json"
  success="${attempt}/SUCCESS"
  "${python_bin}" -I -B "${driver}" seal-attempt-terminal \
    --release-manifest "${release_manifest}" \
    --expected-release-manifest-sha256 "${release_manifest_sha}" \
    --expected-driver-source-sha256 "${driver_sha}" \
    --attempt-label "${label}" \
    --launch-authority-core "${launch_authority_core}" \
    --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
    --attempt-intent "${intent}" --expected-attempt-intent-sha256 "${intent_sha}" \
    --receipt "${receipt}" \
    --validation-a "${started}/receipt-validation-1.json" \
    --validation-b "${started}/receipt-validation-2.json" \
    --controller-status "${controller_status}" \
    --output-terminal-authority "${terminal_authority}" \
    --output-success "${success}" >/dev/null || fail "terminal authority sealing failed"
}

[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
[[ -d "${launch_root}" && ! -L "${launch_root}" ]] || fail "launch root differs"
[[ "$(stat -c %a "${launch_root}")" == 555 ]] || fail "launch root mode differs"
readonly expected_launch_entries=$'LAUNCH_AUTHORITY_CORE.json\nauh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh\nauh_fresh_world8_level_a_r2_p2_node279_step_v1.sh\nauh_launch_fresh_world8_level_a_r2_p2_node279_job140846_v1.sh'
readonly observed_launch_entries="$(find "${launch_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_launch_entries}" == "${expected_launch_entries}" ]] || fail "launch exact root closure differs"
[[ "$0" == "${controller}" ]] || fail "controller must be invoked by its frozen absolute path"
for executable in "${controller}" "${step_payload}" "${rank_exec}"; do
  [[ -x "${executable}" && ! -L "${executable}" ]] || fail "launcher executable differs: ${executable}"
  [[ "$(stat -c %a "${executable}")" == 555 ]] || fail "launcher executable mode differs: ${executable}"
  [[ "$(stat -c %h "${executable}")" == 1 ]] || fail "launcher executable link count differs: ${executable}"
done
[[ "$(sha256sum "${step_payload}" | awk '{print $1}')" == "${step_payload_sha}" ]] || fail "step payload SHA differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ -f "${launch_authority_core}" && ! -L "${launch_authority_core}" ]] || fail "launch authority core differs"
[[ "$(stat -c %a "${launch_authority_core}")" == 444 ]] || fail "launch authority core mode differs"
[[ "$(stat -c %h "${launch_authority_core}")" == 1 ]] || fail "launch authority core link count differs"
[[ "$(sha256sum "${launch_authority_core}" | awk '{print $1}')" == "${launch_authority_core_sha}" ]] || fail "launch authority core SHA differs"
[[ -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root differs"
[[ "$(stat -c %a "${release_root}")" == 555 ]] || fail "release root mode differs"
for pinned_file in "${release_manifest}" "${driver}"; do
  [[ -f "${pinned_file}" && ! -L "${pinned_file}" ]] || fail "pinned release authority differs"
  [[ "$(stat -c %a "${pinned_file}")" == 444 ]] || fail "pinned release authority mode differs"
  [[ "$(stat -c %h "${pinned_file}")" == 1 ]] || fail "pinned release authority link count differs"
done
[[ "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ "$(sha256sum "${driver}" | awk '{print $1}')" == "${driver_sha}" ]] || fail "driver SHA differs"
[[ -d "${attempt_root}" && ! -L "${attempt_root}" ]] || fail "attempt campaign root differs"
[[ "$(stat -c %a "${attempt_root}")" == 700 ]] || fail "attempt campaign root mode differs"
readonly initial_attempt_entries="$(find "${attempt_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${initial_attempt_entries}" == $'A\nB' ]] || fail "attempt A/B root closure differs"
[[ -d "${run_root}" && ! -L "${run_root}" ]] || fail "run root differs"
[[ "$(stat -c %a "${run_root}")" == 700 ]] || fail "run root mode differs"
[[ -z "$(find "${run_root}" -mindepth 1 -print -quit)" ]] || fail "run root is not fresh"
readonly parent_state_initial="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
[[ "${parent_state_initial}" == "${expected_parent_state}" ]] || fail "initial parent allocation state differs: ${parent_state_initial}"
[[ -z "$(node_children)" ]] || fail "node279 already has a numeric child"

run_attempt A
run_attempt B

readonly receipt_a="${run_root}/A/bundle.consumer_receipt"
readonly receipt_b="${run_root}/B/bundle.consumer_receipt"
readonly attempt_a="${attempt_root}/A"
readonly attempt_b="${attempt_root}/B"
readonly parity_root="${run_root}/parity"
readonly parity_receipt="${parity_root}/fresh_world8_a_b.parity_receipt"
mkdir -m 0700 "${parity_root}" || fail "parity output root claim failed"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
set -o noclobber
"${python_bin}" -I -B "${driver}" compare \
  --release-manifest "${release_manifest}" \
  --expected-release-manifest-sha256 "${release_manifest_sha}" \
  --expected-driver-source-sha256 "${driver_sha}" \
  --launch-authority-core "${launch_authority_core}" \
  --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
  --intent-a "${attempt_a}/STARTED/intent.json" \
  --expected-intent-a-sha256 "${intent_a_sha}" \
  --intent-b "${attempt_b}/STARTED/intent.json" \
  --expected-intent-b-sha256 "${intent_b_sha}" \
  --receipt-a "${receipt_a}" \
  --receipt-b "${receipt_b}" \
  --terminal-a "${attempt_a}/terminal.authority.json" \
  --terminal-b "${attempt_b}/terminal.authority.json" \
  --success-a "${attempt_a}/SUCCESS" \
  --success-b "${attempt_b}/SUCCESS" \
  --output-parity-receipt "${parity_receipt}" \
  >"${attempt_root}/compare.log" 2>&1
set +o noclobber
[[ -f "${parity_receipt}" && ! -L "${parity_receipt}" ]] || fail "A/B parity receipt is missing"
[[ "$(stat -c %a "${parity_receipt}")" == 444 ]] || fail "A/B parity receipt mode differs"
[[ "$(stat -c %h "${parity_receipt}")" == 1 ]] || fail "A/B parity receipt link count differs"
"${python_bin}" -I -B -c '
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1]); payload = path.read_bytes(); value = json.loads(payload.decode("utf-8"))
assert json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") == payload
claimed = value["receipt_digest"]; unsigned = dict(value); del unsigned["receipt_digest"]
canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == claimed
assert value["authority"] == "PRE_D0_ENGINEERING_ONLY"
assert value["fresh_world8_parity"]["exact_parity"] is True
assert value["fresh_world8_parity"]["world8_launches"] == 2
assert value["fresh_world8_parity"]["distinct_fresh_process_sessions"] == 16
assert value["full_bernini_renderer_forward_executed"] is False
assert value["offline_product_inference_completed"] is False
assert value["full40_denoise_executed"] is False
assert value["mp4_emitted"] is False
assert value["formal_training_started"] is False
assert value["counts_as_d0"] is False
assert value["scientific_claim_authorized"] is False
assert value["promotion_authorized"] is False
assert value["promotable"] is False
print("PASS sealed Level-A A/B parity receipt", hashlib.sha256(payload).hexdigest(), flush=True)
' "${parity_receipt}"
readonly parent_state_final="$(squeue -h -j "${job_id}" -o '%T|%N|%b')"
[[ "${parent_state_final}" == "${expected_parent_state}" ]] || fail "final parent allocation state differs: ${parent_state_final}"
[[ -z "$(node_children)" ]] || fail "node279 child remained after A/B"
readonly parity_sha="$(sha256sum "${parity_receipt}" | awk '{print $1}')"
readonly campaign_success_tmp="${attempt_root}/.SUCCESS.$$.tmp"
printf 'LEVEL_A_INDEPENDENT_WORLD8_A_B_COMPLETE=true\nparity_receipt_sha256=%s\nworld8_launches=2\ndistinct_fresh_process_sessions=16\ncheckpoint=P2\nconditioner_exact30_cell_only=true\nfull_bernini_renderer_forward_executed=false\noffline_product_inference_completed=false\nfull40_denoise_executed=false\nmp4_emitted=false\nformal_training_started=false\ncounts_as_d0=false\nscientific_claim_authorized=false\npromotion_authorized=false\nparent_untouched=true\nautomatic_relaunch_authorized=false\n' \
  "${parity_sha}" >"${campaign_success_tmp}"
chmod 0400 "${campaign_success_tmp}"
mv "${campaign_success_tmp}" "${attempt_root}/SUCCESS"
printf 'Level-A P2 A/B complete parity_sha256=%s parent_untouched=true promotion=false\n' "${parity_sha}"
