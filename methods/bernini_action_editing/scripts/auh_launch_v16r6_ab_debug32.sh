#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[v16r6-ab-debug32-launcher] ERROR: $*" >&2
  exit 3
}

usage() {
  echo "usage: $0 --release-root DIR --run-root DIR --variant a|b|c|d --job-id ID --node NODE" >&2
  exit 2
}

release_root=""
run_root=""
variant=""
job=""
node=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --release-root) [[ "$#" -ge 2 ]] || usage; release_root="$2"; shift 2 ;;
    --run-root) [[ "$#" -ge 2 ]] || usage; run_root="$2"; shift 2 ;;
    --variant) [[ "$#" -ge 2 ]] || usage; variant="$2"; shift 2 ;;
    --job-id) [[ "$#" -ge 2 ]] || usage; job="$2"; shift 2 ;;
    --node) [[ "$#" -ge 2 ]] || usage; node="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "${variant}" == "a" || "${variant}" == "b" || "${variant}" == "c" || "${variant}" == "d" ]] || usage
[[ "${job}" =~ ^[0-9]+$ ]] || usage
[[ "${node}" =~ ^auh[0-9]+-[0-9a-z-]+$ ]] || usage
[[ -n "${release_root}" && -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root differs"
[[ -n "${run_root}" && "${run_root}" == /vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/* ]] || fail "run root is outside the experiment namespace"

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) fail "launcher must run on an AUH login host" ;;
esac

readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly release_manifest="${release_root}/v16r6ab-release.json"
readonly source_archive="${release_root}/v16r6ab-source.tar"
readonly source_manifest="${release_root}/v16r6ab-source.manifest.json"
readonly worker="${release_root}/auh_run_v16r6_ab_debug32.sh"
readonly launcher="${release_root}/auh_launch_v16r6_ab_debug32.sh"
for path in "${release_manifest}" "${source_archive}" "${source_manifest}" "${worker}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "release input is absent: ${path}"
done
builder_leaf="$(jq -er '.source_release.builder' "${release_manifest}")"
builder="${release_root}/${builder_leaf}"
[[ -f "${builder}" && ! -L "${builder}" ]] || fail "release verifier is absent"
[[ -x "${python_bin}" ]] || fail "vace Python is absent"

jq -e \
  --arg variant "${variant}" \
  '(.schema_version == "bernini-v16r6-ab-debug32-release-v1" or
    .schema_version == "bernini-v16r6c-two-sided-debug32-release-v1" or
    .schema_version == "bernini-v16r6d-absolute-anchor-debug32-release-v1") and
   .debug_contract.sealed_manifest_row_count == 644 and
   .debug_contract.optimizer_step_budget == 32 and
   .debug_contract.exact644_training_complete == false and
   .debug_contract.terminal_full644_checkpoint == false and
   .debug_contract.scientific_claim_authorized == false and
   .preflight.new_contract_expected_test_count >= 1 and
   .preflight.v16r5_regression_expected_test_count == 8 and
   (.variants[$variant] | type == "object")' \
  "${release_manifest}" >/dev/null || fail "release semantic contract differs"

for leaf in auh_run_v16r6_ab_debug32.sh auh_launch_v16r6_ab_debug32.sh "${builder_leaf}"; do
  expected="$(jq -er --arg leaf "${leaf}" '.control_file_sha256[$leaf]' "${release_manifest}")"
  [[ "$(sha256sum -- "${release_root}/${leaf}" | awk '{print $1}')" == "${expected}" ]] || fail "control SHA differs: ${leaf}"
done
archive_sha="$(jq -er '.source_release.archive_sha256' "${release_manifest}")"
manifest_sha="$(jq -er '.source_release.manifest_sha256' "${release_manifest}")"
method_revision="$(jq -er '.source_release.content_closure_sha256' "${release_manifest}")"
[[ "$(sha256sum -- "${source_archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "source archive SHA differs"
[[ "$(sha256sum -- "${source_manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "source manifest SHA differs"
"${python_bin}" -B "${builder}" verify \
  --archive "${source_archive}" \
  --manifest "${source_manifest}" \
  --expected-archive-sha256 "${archive_sha}" \
  --expected-manifest-sha256 "${manifest_sha}"

preflight_scratch="$(mktemp -d /tmp/v16r6-ab-debug32-test.XXXXXX)"
training_srun_pid=""
cleanup() {
  local status="$?"
  set +e
  trap - EXIT
  if [[ -n "${training_srun_pid:-}" ]] && kill -0 "${training_srun_pid}" 2>/dev/null; then
    kill "${training_srun_pid}" 2>/dev/null || true
    wait "${training_srun_pid}" 2>/dev/null || true
  fi
  if [[ -n "${preflight_scratch:-}" && -d "${preflight_scratch}" && ! -L "${preflight_scratch}" && "${preflight_scratch}" == /tmp/v16r6-ab-debug32-test.* ]]; then
    rm -rf -- "${preflight_scratch}"
  fi
  exit "${status}"
}
trap cleanup EXIT
tar -xf "${source_archive}" -C "${preflight_scratch}"
method_root="${preflight_scratch}/methods/bernini_action_editing"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${method_root}"
contract_test_member="$(jq -er '.preflight.new_contract_test_member' "${release_manifest}")"
contract_test_count="$(jq -er '.preflight.new_contract_expected_test_count' "${release_manifest}")"
regression_test_member="$(jq -er '.preflight.v16r5_regression_test_member' "${release_manifest}")"
regression_test_count="$(jq -er '.preflight.v16r5_regression_expected_test_count' "${release_manifest}")"
mkdir -p "${run_root}/preflight"
new_test_log="${run_root}/preflight/v16r6${variant}_contract_tests_job${job}.log"
old_test_log="${run_root}/preflight/v16r5_regression_tests_job${job}.log"
for path in "${new_test_log}" "${old_test_log}"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] || fail "refusing to overwrite preflight artifact: ${path}"
done
"${python_bin}" -B "${preflight_scratch}/${contract_test_member}" >"${new_test_log}" 2>&1 || fail "v16r6 contract preflight tests failed"
"${python_bin}" -B "${preflight_scratch}/${regression_test_member}" >"${old_test_log}" 2>&1 || fail "v16r5 regression tests failed"
grep -Eq "^Ran ${contract_test_count} tests in " "${new_test_log}" || fail "v16r6 preflight test count differs"
grep -Eq '^OK$' "${new_test_log}" || fail "v16r6 preflight result differs"
grep -Eq "^Ran ${regression_test_count} tests in " "${old_test_log}" || fail "v16r5 regression test count differs"
grep -Eq '^OK$' "${old_test_log}" || fail "v16r5 regression result differs"
rm -rf -- "${preflight_scratch}"
preflight_scratch=""

mapfile -t allocated_nodes < <(squeue --noheader --jobs="${job}" --states=RUNNING --format='%N')
[[ "${#allocated_nodes[@]}" -eq 1 && "${allocated_nodes[0]}" == "${node}" ]] || fail "job ${job} is not RUNNING on exact node ${node}"
mapfile -t active_steps < <(
  squeue --steps --noheader --jobs="${job}" --format='%i' |
    awk '$0 !~ /\.(batch|extern)$/ && $0 ~ /^[0-9]+\.[0-9]+$/'
)
[[ "${#active_steps[@]}" -eq 0 ]] || fail "allocation already has an active compute step: ${active_steps[*]}"

mkdir -p "${run_root}/plans" "${run_root}/outputs" "${run_root}/logs"
output="${run_root}/outputs/v16r6${variant}_prefix32_job${job}"
plan="${run_root}/plans/v16r6${variant}_prefix32_job${job}.json"
log="${run_root}/logs/v16r6${variant}_prefix32_job${job}.log"
final="${run_root}/v16r6${variant}_prefix32_job${job}.final.json"
for path in "${output}" "${plan}" "${log}" "${final}"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] || fail "refusing to overwrite run artifact: ${path}"
done

jq -nS \
  --arg variant "${variant}" \
  --argjson job "${job}" \
  --arg node "${node}" \
  --arg output "${output}" \
  --arg archive_sha "${archive_sha}" \
  --arg manifest_sha "${manifest_sha}" \
  --arg revision "${method_revision}" \
  --argjson learning_rate "$(jq -er --arg variant "${variant}" '.variants[$variant].learning_rate' "${release_manifest}")" \
  --argjson target_count "$(jq -er --arg variant "${variant}" '.variants[$variant].target_module_count' "${release_manifest}")" \
  '{schema_version:"bernini-v16r6-ab-debug32-run-plan-v1",variant:$variant,job_id:$job,node:$node,output:$output,max_steps:32,learning_rate:$learning_rate,lora_target_module_count:$target_count,source_archive_sha256:$archive_sha,source_manifest_sha256:$manifest_sha,method_source_revision:$revision,exact644_training_complete:false,terminal_full644_checkpoint:false,scientific_claim_authorized:false}' \
  >"${plan}"
plan_sha="$(sha256sum -- "${plan}" | awk '{print $1}')"

echo "[v16r6-ab-debug32-launcher] launching variant ${variant} exact32 diagnostic on ${job}/${node}"
srun \
  --jobid="${job}" --overlap --exact --kill-on-bad-exit=1 \
  --job-name="v16r6${variant}-debug32" \
  --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --gres=gpu:mi210:8 --mem=256G --nodelist="${node}" \
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES=0,1,2,3 \
  V16R6_EXPECTED_JOB="${job}" V16R6_EXPECTED_NODE="${node}" \
  V16R6_VARIANT="${variant}" V16R6_RELEASE_ROOT="${release_root}" \
  V16R6_PLAN="${plan}" V16R6_PLAN_SHA256="${plan_sha}" \
  bash "${worker}" >"${log}" 2>&1 &
training_srun_pid="$!"

training_step=""
for _ in $(seq 1 240); do
  mapfile -t candidates < <(
    squeue --steps --noheader --jobs="${job}" --format='%i|%j|%N' |
      awk -F'|' -v wanted="v16r6${variant}-debug32" -v wanted_node="${node}" \
        '$2 == wanted && $3 == wanted_node {print $1}'
  )
  if [[ "${#candidates[@]}" -eq 1 && "${candidates[0]}" =~ ^${job}\.[0-9]+$ ]]; then
    training_step="${candidates[0]}"
    break
  fi
  kill -0 "${training_srun_pid}" 2>/dev/null || break
  sleep 0.5
done
[[ -n "${training_step}" ]] || fail "could not resolve the owned training Slurm step"

while kill -0 "${training_srun_pid}" 2>/dev/null; do
  mapfile -t concurrent_steps < <(
    squeue --steps --noheader --jobs="${job}" --format='%i|%j|%N' |
      awk -F'|' -v owned="${training_step}" \
        '$1 ~ /^[0-9]+\.[0-9]+$/ && $1 != owned {print $1 "|" $2 "|" $3}'
  )
  [[ "${#concurrent_steps[@]}" -eq 0 ]] || fail "concurrent step appeared in allocation: ${concurrent_steps[*]}"
  mapfile -t colocated_jobs < <(
    squeue --noheader --nodes="${node}" --states=RUNNING --format='%i|%N' |
      awk -F'|' -v owned_job="${job}" '$1 != owned_job {print}'
  )
  [[ "${#colocated_jobs[@]}" -eq 0 ]] || fail "another running job appeared on exclusive node: ${colocated_jobs[*]}"
  sleep 5
done

set +e
wait "${training_srun_pid}"
training_exit="$?"
set -e
training_srun_pid=""
[[ "${training_exit}" -eq 0 ]] || fail "training Slurm step exited ${training_exit}"

receipt="${output}/checkpoint-00000032/receipt.json"
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "exact32 receipt is absent after worker success"
receipt_sha="$(sha256sum -- "${receipt}" | awk '{print $1}')"
log_sha="$(sha256sum -- "${log}" | awk '{print $1}')"
new_test_log_sha="$(sha256sum -- "${new_test_log}" | awk '{print $1}')"
old_test_log_sha="$(sha256sum -- "${old_test_log}" | awk '{print $1}')"
jq -nS \
  --arg variant "${variant}" --argjson job "${job}" --arg node "${node}" \
  --arg step "${training_step}" \
  --arg output "${output}" --arg plan "${plan}" --arg plan_sha "${plan_sha}" \
  --arg receipt "${receipt}" --arg receipt_sha "${receipt_sha}" \
  --arg log "${log}" --arg log_sha "${log_sha}" \
  --arg new_test_log "${new_test_log}" --arg new_test_log_sha "${new_test_log_sha}" \
  --arg old_test_log "${old_test_log}" --arg old_test_log_sha "${old_test_log_sha}" \
  --argjson contract_test_count "${contract_test_count}" --argjson regression_test_count "${regression_test_count}" \
  '{schema_version:"bernini-v16r6-ab-debug32-run-final-v1",status:"debug32_process_complete_non_exact644",variant:$variant,job_id:$job,node:$node,slurm_step:$step,exclusive_node_monitor_interval_seconds:5,output:$output,global_step:32,max_steps:32,plan:$plan,plan_sha256:$plan_sha,receipt:$receipt,receipt_sha256:$receipt_sha,log:$log,log_sha256:$log_sha,v16r6_contract_tests:{passed:true,count:$contract_test_count,log:$new_test_log,log_sha256:$new_test_log_sha},v16r5_regression_tests:{passed:true,count:$regression_test_count,log:$old_test_log,log_sha256:$old_test_log_sha},exact644_training_complete:false,terminal_full644_checkpoint:false,scientific_claim_authorized:false}' \
  >"${final}"
echo "[v16r6-ab-debug32-launcher] variant ${variant} exact32 debug complete; no exact644 claim"
