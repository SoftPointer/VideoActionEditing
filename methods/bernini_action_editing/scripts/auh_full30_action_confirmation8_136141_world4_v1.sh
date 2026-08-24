#!/usr/bin/env bash
# BOX-EXP-009 no-diagnostic confirmation exact8 on retained 136141/gpu299.
#
# This launcher is generation-only.  It makes one disposable fit-r13 physical
# safe_open/r10-r13-parity smoke and then four serial WORLD4 shards, two formal
# candidates per shard (action, incomplete).  The release has no diagnostic
# task or CLI and never creates an optimizer.

set -Eeuo pipefail
umask 077

fail() { echo "[full30-confirmation8-136141-v1] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirm="${C8_CONFIRM:?explicit launch confirmation required}"
readonly run_root="${C8_RUN_ROOT:?set fresh run root}"
readonly method_root="${C8_METHOD_ROOT:?set extracted release method root}"
readonly method_archive="${C8_METHOD_ARCHIVE:?set release archive}"
readonly method_archive_sha="${C8_METHOD_ARCHIVE_SHA256:?pin release archive}"
readonly method_manifest="${C8_METHOD_MANIFEST:?set release manifest}"
readonly method_manifest_sha="${C8_METHOD_MANIFEST_SHA256:?pin release manifest}"
readonly seed1_spec="${C8_SEED1_SPEC:?set seed1 reserve4 spec}"
readonly seed2_spec="${C8_SEED2_SPEC:?set seed2 reserve4 spec}"
readonly python_bin="${C8_PYTHON_BIN:?set frozen Python}"
readonly bernini_root="${C8_BERNINI_ROOT:?set Bernini root}"
readonly veomni_root="${C8_VEOMNI_ROOT:?set VeOmni root}"
readonly checkpoint="${C8_CHECKPOINT:?set Bernini checkpoint}"
readonly checkpoint_manifest="${C8_CHECKPOINT_MANIFEST:?set checkpoint manifest}"
readonly method_revision="${C8_METHOD_REVISION:?set content revision}"
readonly master_port="${C8_MASTER_PORT:?set four-port base}"
readonly r10_receipt="${C8_R10_COMPILE_SMOKE_RECEIPT:?bind r10 receipt}"
readonly r10_receipt_sha="${C8_R10_COMPILE_SMOKE_RECEIPT_SHA256:?pin r10 receipt}"
readonly r10_log="${C8_R10_GENERATION_LOG:?bind r10 log}"
readonly r10_log_sha="${C8_R10_GENERATION_LOG_SHA256:?pin r10 log}"

readonly holder_job=136141
readonly holder_node=auh7-1b-gpu-299
readonly launch_confirmation=launch-approved-BOX-EXP-009-confirmation-action-anchor-exact8-136141

case "${1:-}" in
  "") readonly role=parent ;;
  __child) [[ $# == 1 ]] || fail "child takes no extra arguments"; readonly role=child ;;
  *) fail "launcher arguments differ" ;;
esac

[[ "${confirm}" == "${launch_confirmation}" ]] || fail "launch confirmation differs"
[[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65532 )) || fail "four-port range differs"
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method revision differs"
for digest in method_archive_sha method_manifest_sha r10_receipt_sha r10_log_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} is not SHA-256"
done
for name in run_root method_root method_archive method_manifest seed1_spec seed2_spec python_bin bernini_root veomni_root checkpoint checkpoint_manifest r10_receipt r10_log; do
  value="${!name}"
  [[ "${value}" == /* && "${value}" != / ]] || fail "${name} must be a scoped absolute path"
done

readonly controller="${method_root}/full30_action_confirmation8_136141_controller_v1.py"
readonly plan_tool="${method_root}/full30_action_confirmation8_136141_plan_v1.py"
readonly generator="${method_root}/full30_action_confirmation8_136141_generator_v1.py"
readonly resource="${method_root}/tools/reserve4_fixed_generation_sp4_136141_confirmation8_specialized_v1.py"
readonly release_builder="${method_root}/tools/build_full30_action_confirmation8_136141_release_v1.py"
readonly launcher="${method_root}/scripts/auh_full30_action_confirmation8_136141_world4_v1.sh"

for path in "${method_archive}" "${method_manifest}" "${seed1_spec}" "${seed2_spec}" "${python_bin}" "${checkpoint_manifest}" "${r10_receipt}" "${r10_log}" "${controller}" "${plan_tool}" "${generator}" "${resource}" "${release_builder}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
for path in "${method_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed directory differs: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "release archive SHA differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ "$(sha256_file "${r10_receipt}")" == "${r10_receipt_sha}" ]] || fail "r10 receipt SHA differs"
[[ "$(sha256_file "${r10_log}")" == "${r10_log_sha}" ]] || fail "r10 log SHA differs"
[[ "$(sha256_file "${resource}")" == be722e4020040ba446f290f07378e870e2d3c1a4228ec997c3447770fcb53d5d ]] || fail "136141 resource specialization differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
"${python_bin}" -B "${release_builder}" audit \
  --archive "${method_archive}" --expected-archive-sha256 "${method_archive_sha}" \
  --manifest "${method_manifest}" --expected-manifest-sha256 "${method_manifest_sha}" >/dev/null || fail "release audit failed"

numbered_steps() { squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}'; }

assert_topology_text() {
  local topology="$1" rows row_count xgmi pcie numa0 numa1
  rows="$(awk '$1~/^GPU[0-7]$/ && NF==9 {ok=1; for(i=2;i<=9;i++)if($i!~/^(0|XGMI|PCIE)$/)ok=0; if(ok)print}' <<<"${topology}")"
  row_count="$(awk 'NF{n++}END{print n+0}' <<<"${rows}")"
  xgmi="$(awk '{for(i=1;i<=NF;i++)if($i=="XGMI")n++}END{print n+0}' <<<"${rows}")"
  pcie="$(awk '{for(i=1;i<=NF;i++)if($i=="PCIE")n++}END{print n+0}' <<<"${rows}")"
  numa0="$(awk '/GPU\[[0-3]\].*Topology.*Numa Node:/ && $NF==0{n++}END{print n+0}' <<<"${topology}")"
  numa1="$(awk '/GPU\[[4-7]\].*Topology.*Numa Node:/ && $NF==1{n++}END{print n+0}' <<<"${topology}")"
  [[ "${row_count}" == 8 && "${xgmi}" == 24 && "${pcie}" == 32 && "${numa0}" == 4 && "${numa1}" == 4 ]] || fail "node is not two exact XGMI4/NUMA islands"
}

assert_all8_idle_local() {
  local snapshot count memory_count busy
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || fail "all8 physical GPUs are not idle"
}

if [[ "${role}" == child ]]; then
  [[ "${SLURM_JOB_ID:?Slurm child required}" == "${holder_job}" ]] || fail "child holder differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered step required}" =~ ^[0-9]+$ ]] || fail "child step differs"
  [[ "$(numbered_steps)" == "${holder_job}.${SLURM_STEP_ID}" ]] || fail "child is not the sole numbered step"
  case "${SLURM_STEP_GPUS:-}" in 0,1,2,3,4,5,6,7|0-7) ;; *) fail "child did not receive exact physical GPUs 0-7" ;; esac
  unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES
  assert_all8_idle_local
  assert_topology_text "$(rocm-smi --showtopo)"
  "${python_bin}" -B "${controller}" validate-runtime \
    --controller-plan "${C8_CONTROLLER_PLAN}" \
    --expected-controller-plan-sha256 "${C8_CONTROLLER_PLAN_SHA256}" >/dev/null || fail "child runtime validation failed"

  readonly scratch_parent="${SLURM_TMPDIR:-/tmp}"
  [[ -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" && "$(readlink -f -- "${scratch_parent}")" == "${scratch_parent}" ]] || fail "node-local scratch differs"
  readonly scratch_fstype="$(stat -f -c '%T' -- "${scratch_parent}")"
  case "${scratch_fstype}" in ext2/ext3|xfs|tmpfs) ;; *) fail "COMGR scratch is not node-local" ;; esac
  task_scratch="$(mktemp -d -- "${scratch_parent}/confirmation8-${SLURM_JOB_ID}-${SLURM_STEP_ID}.XXXXXXXX")"
  readonly task_scratch task_scratch_identity="$(stat -c '%d:%i:%u:%a' -- "${task_scratch}")"
  host_memory_monitor_pid=""
  cleanup_child() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${host_memory_monitor_pid}" ]]; then kill -TERM "${host_memory_monitor_pid}" 2>/dev/null || true; wait "${host_memory_monitor_pid}" 2>/dev/null || true; fi
    if [[ -d "${task_scratch}" && ! -L "${task_scratch}" && "$(stat -c '%d:%i:%u:%a' -- "${task_scratch}")" == "${task_scratch_identity}" ]]; then
      find "${task_scratch}" -xdev -depth -mindepth 1 -delete || exit 70
      rmdir "${task_scratch}" || exit 70
    else
      exit 70
    fi
    exit "${status}"
  }
  trap cleanup_child EXIT INT TERM HUP
  export GADP_NODE_LOCAL_SCRATCH="${task_scratch}" GADP_NODE_LOCAL_SCRATCH_FSTYPE="${scratch_fstype}" TMPDIR="${task_scratch}"
  readonly model_load_lock="${task_scratch}/renderer-load.lock"
  (umask 0377; : >"${model_load_lock}")
  chmod 0400 "${model_load_lock}"
  export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1 NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}" NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1

  readonly journal="${run_root}/logs/host-cgroup-memory-current-samples.bin"
  readonly monitor_start="${run_root}/logs/host-cgroup-memory-monitor-start.json"
  readonly monitor_stop="${task_scratch}/host-memory-monitor-stop"
  "${python_bin}" -B "${resource}" host-memory-monitor \
    --sample-journal "${journal}" --start-receipt-output "${monitor_start}" \
    --stop-path "${monitor_stop}" --supervisor-pid "$$" \
    --slurm-job-id "${SLURM_JOB_ID}" --slurm-step-id "${SLURM_STEP_ID}" &
  host_memory_monitor_pid=$!
  export GADP_HOST_MEMORY_SAMPLE_JOURNAL="${journal}" GADP_HOST_MEMORY_MONITOR_START_RECEIPT="${monitor_start}"
  export GADP_HOST_MEMORY_MONITOR_PID="${host_memory_monitor_pid}" GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID="$$"
  monitor_ready=false
  for _ in $(seq 1 2000); do
    kill -0 "${host_memory_monitor_pid}" 2>/dev/null || fail "host monitor exited before readiness"
    if [[ -f "${monitor_start}" ]] && "${python_bin}" -B "${resource}" assert-host-memory-monitor-live >/dev/null 2>&1; then monitor_ready=true; break; fi
    sleep 0.01
  done
  [[ "${monitor_ready}" == true ]] || fail "10ms host monitor did not become live"

  readonly smoke_plan_root="${run_root}/resource-smoke-plan"
  smoke_plan_json="$("${python_bin}" -B "${resource}" build-plan --seed1-spec "${seed1_spec}" --seed2-spec "${seed2_spec}" --split fit --output-dir "${smoke_plan_root}")"
  readonly smoke_plan="$("${python_bin}" -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_path"])' "${smoke_plan_json}")"
  readonly smoke_plan_sha="$("${python_bin}" -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_file_sha256"])' "${smoke_plan_json}")"
  readonly smoke_receipt="${run_root}/logs/resource-compile-smoke-receipt.json"
  readonly smoke_host_gate="${run_root}/logs/resource-compile-smoke-host-gate.json"
  export ROCR_VISIBLE_DEVICES=0,1,2,3
  "${python_bin}" -B "${resource}" smoke-sp4 \
    --plan "${smoke_plan}" --expected-plan-sha256 "${smoke_plan_sha}" \
    --python "${python_bin}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --method-source-revision "${method_revision}" --method-source-archive-sha256 "${method_archive_sha}" \
    --master-port "${master_port}" --receipt-output "${smoke_receipt}" \
    --host-memory-gate-output "${smoke_host_gate}" \
    --r10-compile-smoke-receipt "${r10_receipt}" --expected-r10-compile-smoke-receipt-sha256 "${r10_receipt_sha}" \
    --r10-generation-log "${r10_log}" --expected-r10-generation-log-sha256 "${r10_log_sha}" || fail "physical safe_open/r10 compile smoke failed"
  readonly smoke_receipt_sha="$(sha256_file "${smoke_receipt}")"

  run_admission_shard() {
    local seed_slot="$1" group_id="$2" visible="$3" port="$4" output
    "${python_bin}" -B "${resource}" assert-host-memory-monitor-live >/dev/null || fail "host monitor boundary failed"
    assert_all8_idle_local
    export ROCR_VISIBLE_DEVICES="${visible}"
    output="${run_root}/generation/${seed_slot}-${group_id}"
    "${python_bin}" -B "${generator}" run-sp4 \
      --plan "${C8_EXACT8_PLAN}" --expected-plan-sha256 "${C8_EXACT8_PLAN_SHA256}" \
      --resource-contract "${resource}" --resource-compile-smoke-receipt "${smoke_receipt}" \
      --expected-resource-compile-smoke-receipt-sha256 "${smoke_receipt_sha}" \
      --seed-slot "${seed_slot}" --group-id "${group_id}" \
      --python "${python_bin}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${method_revision}" --method-source-archive-sha256 "${method_archive_sha}" \
      --master-port "${port}" --output-dir "${output}" || fail "${seed_slot}/${group_id} exact2 generation failed"
    "${python_bin}" -B "${resource}" assert-host-memory-monitor-live >/dev/null || fail "post-shard host monitor boundary failed"
  }

  run_admission_shard seed1 sp4-a 0,1,2,3 "${master_port}"
  run_admission_shard seed1 sp4-b 4,5,6,7 "$((master_port + 1))"
  run_admission_shard seed2 sp4-a 0,1,2,3 "$((master_port + 2))"
  run_admission_shard seed2 sp4-b 4,5,6,7 "$((master_port + 3))"
  unset ROCR_VISIBLE_DEVICES
  "${python_bin}" -B "${generator}" audit-exact8 \
    --plan "${C8_EXACT8_PLAN}" --expected-plan-sha256 "${C8_EXACT8_PLAN_SHA256}" \
    --generation-root "${run_root}/generation/seed1-sp4-a" \
    --generation-root "${run_root}/generation/seed1-sp4-b" \
    --generation-root "${run_root}/generation/seed2-sp4-a" \
    --generation-root "${run_root}/generation/seed2-sp4-b" \
    --output "${run_root}/generation-exact8-audit.json" \
    --gap-output "${run_root}/generation-exact8-gap.json" >/dev/null || fail "formal exact8 audit failed"

  mkdir -m 0700 "${monitor_stop}"
  set +e
  wait "${host_memory_monitor_pid}"
  monitor_status=$?
  set -e
  host_memory_monitor_pid=""
  [[ "${monitor_status}" == 0 ]] || fail "host monitor failed status=${monitor_status}"
  readonly monitor_start_sha="$(sha256_file "${monitor_start}")"
  "${python_bin}" -B "${controller}" seal-terminal-host-gate \
    --resource-contract "${resource}" --monitor-start-receipt "${monitor_start}" \
    --expected-monitor-start-receipt-sha256 "${monitor_start_sha}" \
    --monitor-exit-status "${monitor_status}" \
    --output "${run_root}/logs/terminal-confirmation8-host-gate.json" >/dev/null || fail "terminal exact8 host gate failed"
  [[ -z "$(jobs -pr)" ]] || fail "background process remained"
  exit 0
fi

[[ $# == 0 ]] || fail "parent launcher takes no arguments"
[[ ! -e "${run_root}" && ! -L "${run_root}" && -d "$(dirname -- "${run_root}")" ]] || fail "run root must be fresh"
parent_state="$(squeue -j "${holder_job}" -h -o '%T|%N|%u' | sort -u)"
[[ "${parent_state}" == "RUNNING|${holder_node}|guangyi.chen" ]] || fail "retained parent 136141/gpu299 differs"
[[ -z "$(numbered_steps)" ]] || fail "holder already has a numbered child"
mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/generation"

exact8_json="$("${python_bin}" -B "${plan_tool}" build-plan \
  --seed1-spec "${seed1_spec}" --seed2-spec "${seed2_spec}" --split confirmation \
  --output-dir "${run_root}/exact8-plan")"
readonly exact8_plan="$("${python_bin}" -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_path"])' "${exact8_json}")"
readonly exact8_plan_sha="$("${python_bin}" -B -c 'import json,sys; print(json.loads(sys.argv[1])["plan_file_sha256"])' "${exact8_json}")"
readonly controller_plan="${run_root}/controller-plan.json"
"${python_bin}" -B "${controller}" plan \
  --method-root "${method_root}" --release-manifest "${method_manifest}" \
  --expected-release-manifest-sha256 "${method_manifest_sha}" \
  --exact8-plan "${exact8_plan}" --expected-exact8-plan-sha256 "${exact8_plan_sha}" \
  --output "${controller_plan}" >/dev/null || fail "controller plan failed"
readonly controller_plan_sha="$(sha256_file "${controller_plan}")"

child_pid=""
cleanup_parent() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid}" ]]; then kill -TERM "${child_pid}" 2>/dev/null || true; wait "${child_pid}" 2>/dev/null || true; fi
  exit "${status}"
}
trap cleanup_parent EXIT INT TERM HUP
set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --immediate=5 \
  --cpus-per-task=32 --mem=60G --gpus-per-task=8 --gpu-bind=none --gres-flags=enforce-binding \
  env C8_CONTROLLER_PLAN="${controller_plan}" C8_CONTROLLER_PLAN_SHA256="${controller_plan_sha}" \
      C8_EXACT8_PLAN="${exact8_plan}" C8_EXACT8_PLAN_SHA256="${exact8_plan_sha}" \
      bash "${launcher}" __child >"${run_root}/logs/confirmation8-generation.log" 2>&1 &
child_pid=$!
wait "${child_pid}"
status=$?
child_pid=""
set -e
if (( status != 0 )); then tail -n 240 "${run_root}/logs/confirmation8-generation.log" >&2 || true; fail "all8 serial exact8 child failed status=${status}"; fi
[[ -f "${run_root}/generation-exact8-audit.json" && -f "${run_root}/logs/terminal-confirmation8-host-gate.json" ]] || fail "exact8 outputs are incomplete"

printf '%s\n' \
  'schema=bernini-full30-action-confirmation8-launch-status-v1' \
  'experiment_id=BOX-EXP-009' \
  'dataset=confirmation_action_anchor_exact8' \
  'formal_candidate_count=8' \
  'formal_branch_order=action,incomplete' \
  'inference_steps_per_clip=40' \
  'optimizer_steps=0' \
  'formal_world4_invocations=8' \
  'compile_smoke_world4_invocations=1' \
  'diagnostic_task_count=0' \
  'diagnostic_generation_allowed=false' \
  'host_memory_limit_gib=60' \
  'host_sampled_safe_ceiling_gib=56' \
  'host_sample_interval_ns=10000000' \
  'gpu_peak_reserved_limit_gib=52' \
  'world_size=4' \
  'xgmi4_island_topology_required=true' \
  't5_rank_gpu_residency=true' \
  'physical_safe_open=true' \
  'r10_r13_parity=true' \
  'independent_full81_review=pending' \
  'same_state_materializer_threshold_gate=pending' \
  'optimizer_created=false' \
  'optimizer_authorized=false' \
  'training_performed=false' \
  'parent_136141_cancelled_released_or_requeued=false' >"${run_root}/launcher.status"
echo 'BOX_EXP_009_CONFIRMATION8_136141_GENERATION_PENDING_REVIEW_AND_MATERIALIZER candidate_count=8 diagnostics=0 optimizer=false'
