#!/usr/bin/env bash
# Sealed reserve4 confirmation40 authoring-media generation on 136309/gpu280.
#
# One numbered Slurm child owns all eight GPUs and 60 GiB host memory.  Inside
# it, one disposable full-native40 compile smoke runs before four sealed
# run-sp4 shard runners execute synchronously in exact order: seed1-a,
# seed1-b, seed2-a, seed2-b.  The formal shards make forty WORLD4=DP1xSP4
# model invocations.  Each runner sees one XGMI4 island through
# ROCR_VISIBLE_DEVICES; no model invocation is backgrounded and at most one
# model replica exists.  Within each WORLD4 invocation, checkpoint loads are
# serialized through one authenticated node-local flock until each model is
# GPU-resident and host deserialization arenas are trimmed.  A mandatory
# WORLD4 barrier then holds every rank before source/tokenizer setup or native
# sampling.  Ranks never map
# to dog/human/action family.
# Outputs remain pending external blind review.  The shipped train_lora.py is
# an inference import dependency, but the two allowed release entrypoints
# forbid all training, optimizer, review certification, Phi, and P/O operations.

set -Eeuo pipefail
umask 077

fail() { echo "[generic-action-confirmation40-r3] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirm="${GADP_CONFIRM:?controller confirmation required}"
readonly phase="${GADP_PHASE:?bind data phase}"
readonly split="${GADP_SPLIT:?bind analysis split}"
readonly run_root="${GADP_RUN_ROOT:?set run root}"
readonly master_port="${GADP_MASTER_PORT:?set master port}"
readonly controller_plan="${GADP_CONTROLLER_PLAN:?set controller plan}"
readonly controller_plan_sha="${GADP_CONTROLLER_PLAN_SHA256:?pin controller plan}"
readonly upstream_plan="${GADP_UPSTREAM_PLAN:?set upstream plan}"
readonly upstream_plan_sha="${GADP_UPSTREAM_PLAN_SHA256:?pin upstream plan}"
readonly method_root="${GADP_METHOD_ROOT:?set released method root}"
readonly method_archive="${GADP_METHOD_ARCHIVE:?set method archive}"
readonly method_archive_sha="${GADP_METHOD_ARCHIVE_SHA256:?pin method archive}"
readonly method_manifest="${GADP_METHOD_MANIFEST:?set method manifest}"
readonly method_manifest_sha="${GADP_METHOD_MANIFEST_SHA256:?pin method manifest}"
readonly method_revision="${GADP_METHOD_REVISION:?pin method content revision}"
readonly python_bin="${GADP_PYTHON_BIN:?set Python executable}"
readonly python_sha="${GADP_PYTHON_SHA256:?pin Python executable}"
readonly controller_sha="${GADP_CONTROLLER_SHA256:?pin controller}"
readonly launcher_sha="${GADP_LAUNCHER_SHA256:?pin launcher}"
readonly generator_sha="${GADP_GENERATOR_SHA256:?pin generator}"
readonly resource_contract_sha="${GADP_RESOURCE_CONTRACT_SHA256:?pin r13 resource contract}"
readonly r10_compile_smoke_receipt="${GADP_R10_COMPILE_SMOKE_RECEIPT:?bind r10 compile-smoke receipt}"
readonly r10_compile_smoke_receipt_sha="${GADP_R10_COMPILE_SMOKE_RECEIPT_SHA256:?pin r10 compile-smoke receipt}"
readonly r10_generation_log="${GADP_R10_GENERATION_LOG:?bind r10 generation log}"
readonly r10_generation_log_sha="${GADP_R10_GENERATION_LOG_SHA256:?pin r10 generation log}"

readonly holder_job=136309
readonly holder_node=auh7-1b-gpu-280
readonly holder_user=guangyi.chen
readonly launch_confirmation=launch-approved-generic-action-confirmation40-generation-136309
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${1:-}" in
  "") readonly role=parent ;;
  __child)
    [[ $# == 1 ]] || fail "child takes no additional arguments"
    readonly role=child
    ;;
  *) fail "launcher arguments differ" ;;
esac

[[ "${confirm}" == "${launch_confirmation}" ]] || fail "launch confirmation differs"
[[ "${phase}" == generation ]] || fail "only confirmation generation is present"
[[ "${split}" == confirmation ]] || fail "confirmation is absent from this release"
[[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65532 )) || fail "four-port range differs"
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method content revision differs"
for digest_name in controller_plan_sha upstream_plan_sha method_archive_sha method_manifest_sha python_sha controller_sha launcher_sha generator_sha resource_contract_sha r10_compile_smoke_receipt_sha r10_generation_log_sha; do
  [[ "${!digest_name}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest_name} is not a SHA-256"
done
for path_name in run_root controller_plan upstream_plan method_root method_archive method_manifest python_bin r10_compile_smoke_receipt r10_generation_log; do
  value="${!path_name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${path_name} path differs"
done
[[ -d "${run_root}" && ! -L "${run_root}" && "$(readlink -f -- "${run_root}")" == "${run_root}" ]] || fail "run root differs"
[[ -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "method root differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" && "$(readlink -f -- "${python_bin}")" == "${python_bin}" ]] || fail "Python executable differs"

readonly controller="${method_root}/generic_action_confirmation_data_prep_controller_v1.py"
readonly launcher="${method_root}/scripts/auh_generic_action_confirmation_data_prep_136309_world4_v1.sh"
readonly generator="${method_root}/tools/reserve4_confirmation_generation_sp4_v1.py"
readonly resource_contract="${method_root}/tools/reserve4_fixed_generation_sp4_v1.py"
readonly release_builder="${method_root}/tools/build_generic_action_confirmation_release_v1.py"
for path in "${controller_plan}" "${upstream_plan}" "${method_archive}" "${method_manifest}" "${controller}" "${launcher}" "${generator}" "${resource_contract}" "${release_builder}" "${checkpoint_manifest}" "${r10_compile_smoke_receipt}" "${r10_generation_log}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
[[ "$(sha256_file "${controller_plan}")" == "${controller_plan_sha}" ]] || fail "controller plan SHA differs"
[[ "$(sha256_file "${upstream_plan}")" == "${upstream_plan_sha}" ]] || fail "upstream plan SHA differs"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "method archive SHA differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "method manifest SHA differs"
[[ "$(sha256_file "${python_bin}")" == "${python_sha}" ]] || fail "Python SHA differs"
[[ "$(sha256_file "${controller}")" == "${controller_sha}" ]] || fail "controller SHA differs"
[[ "$(sha256_file "${launcher}")" == "${launcher_sha}" ]] || fail "launcher SHA differs"
[[ "$(sha256_file "${generator}")" == "${generator_sha}" ]] || fail "generator SHA differs"
[[ "$(sha256_file "${resource_contract}")" == "${resource_contract_sha}" ]] || fail "r13 resource contract SHA differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(sha256_file "${r10_compile_smoke_receipt}")" == "${r10_compile_smoke_receipt_sha}" ]] || fail "r10 compile-smoke receipt SHA differs"
[[ "$(sha256_file "${r10_generation_log}")" == "${r10_generation_log_sha}" ]] || fail "r10 generation log SHA differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
"${python_bin}" -B "${release_builder}" audit \
  --archive "${method_archive}" --manifest "${method_manifest}" \
  --expected-archive-sha256 "${method_archive_sha}" \
  --expected-manifest-sha256 "${method_manifest_sha}" >/dev/null || fail "release audit failed"
"${python_bin}" -B "${controller}" validate-runtime \
  --method-root "${method_root}" \
  --method-archive "${method_archive}" --expected-method-archive-sha256 "${method_archive_sha}" \
  --method-manifest "${method_manifest}" --expected-method-manifest-sha256 "${method_manifest_sha}" \
  --python-bin "${python_bin}" --expected-python-sha256 "${python_sha}" >/dev/null || fail "live runtime tree differs from release"
"${python_bin}" -B "${controller}" validate-launch-environment \
  --plan "${controller_plan}" --expected-plan-sha256 "${controller_plan_sha}" >/dev/null || fail "launcher environment differs from sealed plan"

numbered_steps() { squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}'; }

assert_topology_text() {
  local topology="$1" rows row_count xgmi pcie numa0 numa1
  rows="$(awk '$1~/^GPU[0-7]$/ && NF==9 {ok=1; for(i=2;i<=9;i++)if($i!~/^(0|XGMI|PCIE)$/)ok=0; if(ok)print}' <<<"${topology}")"
  row_count="$(awk 'NF {n++} END {print n+0}' <<<"${rows}")"
  xgmi="$(awk '{for(i=1;i<=NF;i++)if($i=="XGMI")n++}END{print n+0}' <<<"${rows}")"
  pcie="$(awk '{for(i=1;i<=NF;i++)if($i=="PCIE")n++}END{print n+0}' <<<"${rows}")"
  numa0="$(awk '/GPU\[[0-3]\].*Topology.*Numa Node:/ && $NF==0{n++}END{print n+0}' <<<"${topology}")"
  numa1="$(awk '/GPU\[[4-7]\].*Topology.*Numa Node:/ && $NF==1{n++}END{print n+0}' <<<"${topology}")"
  [[ "${row_count}" == 8 && "${xgmi}" == 24 && "${pcie}" == 32 && "${numa0}" == 4 && "${numa1}" == 4 ]] || fail "node is not two exact XGMI4/NUMA islands"
}

assert_child_inventory() {
  local physical only_step snapshot count memory_count busy topology identity unique_rows bus_rows
  local identity_raw topology_raw physical_inventory
  [[ "${SLURM_JOB_ID:?Slurm child required}" == "${holder_job}" ]] || fail "child holder differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered child required}" =~ ^[0-9]+$ ]] || fail "child step differs"
  only_step="$(numbered_steps)"
  [[ "${only_step}" == "${holder_job}.${SLURM_STEP_ID}" ]] || fail "child is not the holder's only numbered step: ${only_step}"
  physical="${SLURM_STEP_GPUS:-}"
  case "${physical}" in
    0,1,2,3,4,5,6,7|0-7) ;;
    *) fail "Slurm did not grant exact physical GPUs 0-7: ${physical:-missing}" ;;
  esac
  unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL ROCR_VISIBLE_DEVICES
  snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || fail "all-eight child inventory is not idle exact8"
  topology="$(rocm-smi --showtopo)"
  assert_topology_text "${topology}"
  identity="$(rocm-smi --showuniqueid --showbus)"
  unique_rows="$(awk '/Unique ID:/ {print $NF}' <<<"${identity}")"
  bus_rows="$(awk '/PCI Bus:/ {print $NF}' <<<"${identity}")"
  [[ "$(awk 'NF{n++}END{print n+0}' <<<"${unique_rows}")" == 8 ]] || fail "GPU unique-ID inventory is not exact8"
  [[ "$(sort -u <<<"${unique_rows}" | awk 'NF{n++}END{print n+0}')" == 8 ]] || fail "GPU unique IDs are not distinct"
  [[ "$(awk 'NF{n++}END{print n+0}' <<<"${bus_rows}")" == 8 ]] || fail "GPU PCI-bus inventory is not exact8"
  [[ "$(sort -u <<<"${bus_rows}" | awk 'NF{n++}END{print n+0}')" == 8 ]] || fail "GPU PCI buses are not distinct"
  identity_raw="${run_root}/logs/gpu-identity-bus.raw.txt"
  topology_raw="${run_root}/logs/gpu-topology.raw.txt"
  physical_inventory="${run_root}/logs/gpu-physical-inventory.json"
  [[ ! -e "${identity_raw}" && ! -L "${identity_raw}" && ! -e "${topology_raw}" && ! -L "${topology_raw}" && ! -e "${physical_inventory}" && ! -L "${physical_inventory}" ]] || fail "physical GPU inventory outputs are not fresh"
  printf '%s\n' "${identity}" >"${identity_raw}"
  printf '%s\n' "${topology}" >"${topology_raw}"
  chmod 0400 "${identity_raw}" "${topology_raw}"
  "${python_bin}" -B "${controller}" seal-physical-gpu-inventory \
    --identity-input "${identity_raw}" --topology-input "${topology_raw}" \
    --output "${physical_inventory}" >/dev/null || fail "physical-index/PCI/unique-ID/XGMI inventory seal failed"
}

if [[ "${role}" == child ]]; then
  assert_child_inventory
  readonly scratch_parent="${SLURM_TMPDIR:-/tmp}"
  [[ "${scratch_parent}" == /* && "${scratch_parent}" != / && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" && "$(readlink -f -- "${scratch_parent}")" == "${scratch_parent}" ]] || fail "node-local scratch parent differs"
  readonly scratch_fstype="$(stat -f -c '%T' -- "${scratch_parent}")"
  case "${scratch_fstype}" in
    ext2/ext3|xfs|tmpfs) ;;
    *) fail "COMGR scratch parent is not an allowed node-local filesystem: ${scratch_fstype}" ;;
  esac
  task_scratch="$(mktemp -d -- "${scratch_parent}/generic-action-confirmation40-${SLURM_JOB_ID}-${SLURM_STEP_ID}.XXXXXXXX")"
  readonly task_scratch
  readonly task_scratch_identity="$(stat -c '%d:%i:%u:%a' -- "${task_scratch}")"
  cleanup_task_scratch() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${host_memory_monitor_pid:-}" ]]; then
      kill -TERM "${host_memory_monitor_pid}" 2>/dev/null || true
      wait "${host_memory_monitor_pid}" 2>/dev/null || true
      host_memory_monitor_pid=""
    fi
    if [[ -d "${task_scratch}" && ! -L "${task_scratch}" && "$(stat -c '%d:%i:%u:%a' -- "${task_scratch}")" == "${task_scratch_identity}" && "$(dirname -- "${task_scratch}")" == "${scratch_parent}" ]]; then
      find "${task_scratch}" -xdev -depth -mindepth 1 -delete || exit 70
      rmdir "${task_scratch}" || exit 70
    else
      exit 70
    fi
    exit "${status}"
  }
  trap cleanup_task_scratch EXIT INT TERM HUP
  export GADP_NODE_LOCAL_SCRATCH="${task_scratch}"
  export GADP_NODE_LOCAL_SCRATCH_FSTYPE="${scratch_fstype}"
  export TMPDIR="${task_scratch}"
  readonly model_load_lock="${task_scratch}/renderer-load.lock"
  [[ ! -e "${model_load_lock}" && ! -L "${model_load_lock}" ]] || fail "model-load lock is not fresh"
  (umask 0377; : >"${model_load_lock}")
  chmod 0400 "${model_load_lock}"
  [[ -f "${model_load_lock}" && ! -L "${model_load_lock}" && "$(stat -c '%s:%h:%u:%a' -- "${model_load_lock}")" == "0:1:$(id -u):400" && "$(dirname -- "${model_load_lock}")" == "${task_scratch}" ]] || fail "node-local model-load lock identity differs"
  export NATIVE_SERIALIZED_HOST_LOAD_REQUIRED=1
  export NATIVE_V_AXIS_LOAD_LOCK="${model_load_lock}"
  export NATIVE_T2V_KEEP_T5_ON_RANK_GPU_REQUIRED=1
  readonly host_memory_sample_journal="${run_root}/logs/host-cgroup-memory-current-samples.bin"
  readonly host_memory_monitor_start_receipt="${run_root}/logs/host-cgroup-memory-monitor-start-receipt.json"
  readonly host_memory_compile_gate="${run_root}/logs/host-cgroup-memory-compile-smoke-gate.json"
  readonly host_memory_terminal_receipt="${run_root}/logs/host-cgroup-memory-terminal-receipt.json"
  readonly physical_smoke_receipt="${run_root}/logs/compile-smoke-physical-tensor-evidence.json"
  readonly host_memory_stop_path="${task_scratch}/host-memory-monitor-stop"
  for path in "${host_memory_sample_journal}" "${host_memory_monitor_start_receipt}" "${host_memory_compile_gate}" "${host_memory_terminal_receipt}" "${physical_smoke_receipt}" "${host_memory_stop_path}"; do
    [[ ! -e "${path}" && ! -L "${path}" ]] || fail "host memory monitor output is not fresh: ${path}"
  done
  host_memory_monitor_pid=""
  "${python_bin}" -B "${resource_contract}" host-memory-monitor \
    --sample-journal "${host_memory_sample_journal}" \
    --start-receipt-output "${host_memory_monitor_start_receipt}" \
    --stop-path "${host_memory_stop_path}" \
    --supervisor-pid "$$" --slurm-job-id "${SLURM_JOB_ID}" \
    --slurm-step-id "${SLURM_STEP_ID}" &
  host_memory_monitor_pid=$!
  export GADP_HOST_MEMORY_SAMPLE_JOURNAL="${host_memory_sample_journal}"
  export GADP_HOST_MEMORY_MONITOR_START_RECEIPT="${host_memory_monitor_start_receipt}"
  export GADP_HOST_MEMORY_MONITOR_PID="${host_memory_monitor_pid}"
  export GADP_HOST_MEMORY_MONITOR_SUPERVISOR_PID="$$"
  export GADP_PHYSICAL_SMOKE_RECEIPT_OUTPUT="${physical_smoke_receipt}"
  host_monitor_ready=false
  for _ in $(seq 1 2000); do
    kill -0 "${host_memory_monitor_pid}" 2>/dev/null || fail "host memory monitor exited before readiness"
    if [[ -f "${host_memory_monitor_start_receipt}" && ! -L "${host_memory_monitor_start_receipt}" ]] && \
       "${python_bin}" -B "${resource_contract}" assert-host-memory-monitor-live >/dev/null 2>&1; then
      host_monitor_ready=true
      break
    fi
    sleep 0.01
  done
  [[ "${host_monitor_ready}" == true ]] || fail "10ms host memory monitor did not become live"
  assert_only_host_memory_monitor() {
    local jobs_now
    jobs_now="$(jobs -pr)"
    [[ "${jobs_now}" == "${host_memory_monitor_pid}" ]] || fail "background process set differs from the sole host memory monitor: ${jobs_now}"
    kill -0 "${host_memory_monitor_pid}" 2>/dev/null || fail "host memory monitor is not live"
    "${python_bin}" -B "${resource_contract}" assert-host-memory-monitor-live >/dev/null || fail "host memory monitor health replay failed"
  }
  assert_only_host_memory_monitor
  readonly physical_inventory="${run_root}/logs/gpu-physical-inventory.json"
  readonly physical_inventory_sha="$(sha256_file "${physical_inventory}")"
  readonly all8_runtime_mapping="${run_root}/logs/all8-rocm-runtime-mapping.json"
  "${python_bin}" -B "${controller}" observe-gpu-mapping \
    --output "${all8_runtime_mapping}" --expected-count 8 \
    --expected-rocr unset --physical-inventory "${physical_inventory}" \
    --expected-physical-inventory-sha256 "${physical_inventory_sha}" >/dev/null || fail "all8 physical-index/PCI/unique-ID/HIP observation failed"
  readonly all8_runtime_mapping_sha="$(sha256_file "${all8_runtime_mapping}")"
  readonly all8_runtime_observation_digest="$("${python_bin}" -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["observation_digest"])' "${all8_runtime_mapping}")"
  [[ "${all8_runtime_observation_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "all8 runtime mapping digest differs"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0
  export GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export PYTHONPATH="${method_root}"
  completed_shards=0
  compile_smoke_receipt_sha=""

  run_compile_smoke() {
    local visible mapping mapping_sha join_json visible_snapshot visible_count visible_memory busy receipt
    [[ "${completed_shards}" == 0 ]] || fail "compile smoke must precede formal40"
    assert_only_host_memory_monitor
    [[ -z "$(find "${run_root}/generation" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "formal generation exists before compile smoke"
    [[ -z "$(ss -H -ltn "sport = :${master_port}")" ]] || fail "compile-smoke master port is occupied"
    visible=0,1,2,3
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export ROCR_VISIBLE_DEVICES="${visible}"
    mapping="${run_root}/logs/compile-smoke-rocm-runtime-mapping.json"
    "${python_bin}" -B "${controller}" observe-gpu-mapping \
      --output "${mapping}" --expected-count 4 --expected-rocr "${visible}" \
      --physical-inventory "${physical_inventory}" \
      --expected-physical-inventory-sha256 "${physical_inventory_sha}" >/dev/null || \
      fail "compile smoke physical-index/PCI/unique-ID/HIP observation failed"
    mapping_sha="$(sha256_file "${mapping}")"
    join_json="$("${python_bin}" -B "${controller}" validate-gpu-mapping \
      --all8-mapping "${all8_runtime_mapping}" \
      --expected-all8-mapping-sha256 "${all8_runtime_mapping_sha}" \
      --observed-mapping "${mapping}" \
      --expected-observed-mapping-sha256 "${mapping_sha}" \
      --expected-physical-indices 0,1,2,3)" || fail "compile smoke actual ROCr mapping differs"
    "${python_bin}" -B -c 'import json,sys; value=json.loads(sys.argv[1]); assert value["mapping_exact_physical_set_verified"] is True and value["cross_island_visibility_rejected"] is True' "${join_json}" || fail "compile smoke physical-island proof differs"
    visible_snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
    visible_count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${visible_snapshot}")"
    visible_memory="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${visible_snapshot}")"
    busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${visible_snapshot}")"
    [[ "${visible_count}" == 8 && "${visible_memory}" == 8 && -z "${busy}" ]] || fail "all8 physical inventory is not idle before compile smoke"
    receipt="${run_root}/logs/compile-smoke-receipt.json"
    "${python_bin}" -B "${generator}" smoke-sp4 \
      --plan "${upstream_plan}" --expected-plan-sha256 "${upstream_plan_sha}" \
      --python "${python_bin}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${method_revision}" --method-source-archive-sha256 "${method_archive_sha}" \
      --master-port "${master_port}" --receipt-output "${receipt}" || fail "full native40 confirmation compile smoke failed"
    [[ -z "$(find "${run_root}/generation" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail "compile smoke polluted formal generation"
    [[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "compile smoke receipt is absent"
    compile_smoke_receipt_sha="$(sha256_file "${receipt}")"
    [[ "${compile_smoke_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "compile smoke receipt SHA differs"
    [[ -f "${physical_smoke_receipt}" && ! -L "${physical_smoke_receipt}" ]] || fail "physical safe_open smoke receipt is absent"
    "${python_bin}" -B "${controller}" seal-compile-host-memory-gate \
      --output "${host_memory_compile_gate}" >/dev/null || fail "compile-smoke host memory gate failed"
    [[ -f "${host_memory_compile_gate}" && ! -L "${host_memory_compile_gate}" ]] || fail "compile-smoke host memory gate is absent"
    assert_only_host_memory_monitor
  }

  run_sealed_shard() {
    local ordinal="$1" seed_slot="$2" group_id="$3" visible="$4" port="$5"
    local output visible_snapshot visible_count visible_memory busy mapping mapping_sha mapping_digest join_json logical_mapping_digest binding binding_sha
    [[ "${completed_shards}" == "$((ordinal - 1))" ]] || fail "shard order/concurrency state differs"
    assert_only_host_memory_monitor
    [[ -z "$(ss -H -ltn "sport = :${port}")" ]] || fail "shard master port is occupied: ${port}"
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export ROCR_VISIBLE_DEVICES="${visible}"
    mapping="${run_root}/logs/${ordinal}-${seed_slot}-${group_id}-rocm-runtime-mapping.json"
    "${python_bin}" -B "${controller}" observe-gpu-mapping \
      --output "${mapping}" --expected-count 4 --expected-rocr "${visible}" \
      --physical-inventory "${physical_inventory}" \
      --expected-physical-inventory-sha256 "${physical_inventory_sha}" >/dev/null || \
      fail "${seed_slot}/${group_id} physical-index/PCI/unique-ID/HIP observation failed"
    mapping_sha="$(sha256_file "${mapping}")"
    mapping_digest="$("${python_bin}" -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["observation_digest"])' "${mapping}")"
    [[ "${mapping_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${seed_slot}/${group_id} runtime mapping digest differs"
    join_json="$("${python_bin}" -B "${controller}" validate-gpu-mapping \
      --all8-mapping "${all8_runtime_mapping}" \
      --expected-all8-mapping-sha256 "${all8_runtime_mapping_sha}" \
      --observed-mapping "${mapping}" \
      --expected-observed-mapping-sha256 "${mapping_sha}" \
      --expected-physical-indices "${visible}")" || fail "${seed_slot}/${group_id} actual ROCr mapping differs"
    logical_mapping_digest="$("${python_bin}" -B -c 'import json,sys; print(json.loads(sys.argv[1])["observed_logical_to_physical_order_digest"])' "${join_json}")"
    [[ "${logical_mapping_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${seed_slot}/${group_id} observed logical-to-physical digest differs"
    visible_snapshot="$(rocm-smi --showuse --showmemuse --showpids)"
    visible_count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${visible_snapshot}")"
    visible_memory="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${visible_snapshot}")"
    busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${visible_snapshot}")"
    # rocm-smi is a physical-management view and may ignore the ROCr visibility
    # filter.  The all-eight allocation must remain idle here; the immediately
    # preceding torch/HIP UUID+bus join is the sealed logical-view exact4 check.
    [[ "${visible_count}" == 8 && "${visible_memory}" == 8 && -z "${busy}" ]] || fail "${seed_slot}/${group_id} all8 physical inventory is not idle"
    output="${run_root}/generation/${seed_slot}-${group_id}"
    "${python_bin}" -B "${generator}" run-sp4 \
      --plan "${upstream_plan}" --expected-plan-sha256 "${upstream_plan_sha}" \
      --seed-slot "${seed_slot}" --group-id "${group_id}" \
      --compile-smoke-receipt "${run_root}/logs/compile-smoke-receipt.json" \
      --expected-compile-smoke-receipt-sha256 "${compile_smoke_receipt_sha}" \
      --python "${python_bin}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${method_revision}" --method-source-archive-sha256 "${method_archive_sha}" \
      --master-port "${port}" --output-dir "${output}"
    assert_only_host_memory_monitor
    binding="${run_root}/logs/${ordinal}-${seed_slot}-${group_id}-physical-binding.json"
    "${python_bin}" -B "${controller}" seal-gpu-binding-receipt \
      --run-root "${run_root}" --ordinal "${ordinal}" \
      --seed-slot "${seed_slot}" --group-id "${group_id}" \
      --output "${binding}" >/dev/null || fail "${seed_slot}/${group_id} binding receipt seal failed"
    binding_sha="$(sha256_file "${binding}")"
    printf -v "binding_${ordinal}_path" '%s' "${binding}"
    printf -v "binding_${ordinal}_sha" '%s' "${binding_sha}"
    completed_shards=$((completed_shards + 1))
  }

  run_compile_smoke
  "${python_bin}" -B "${controller}" seal-gpu-admission-receipt \
    --run-root "${run_root}" --output "${run_root}/gpu-admission-receipt.json" >/dev/null || fail "pre-formal GPU/host admission receipt seal failed"
  assert_only_host_memory_monitor
  run_sealed_shard 1 seed1 sp4-a 0,1,2,3 "${master_port}"
  run_sealed_shard 2 seed1 sp4-b 4,5,6,7 "$((master_port + 1))"
  run_sealed_shard 3 seed2 sp4-a 0,1,2,3 "$((master_port + 2))"
  run_sealed_shard 4 seed2 sp4-b 4,5,6,7 "$((master_port + 3))"
  [[ "${completed_shards}" == 4 ]] || fail "four-shard serial completion differs"
  assert_only_host_memory_monitor
  mkdir -m 0700 "${host_memory_stop_path}"
  set +e
  wait "${host_memory_monitor_pid}"
  host_memory_monitor_status=$?
  set -e
  host_memory_monitor_pid=""
  [[ "${host_memory_monitor_status}" == 0 ]] || fail "host memory monitor failed status=${host_memory_monitor_status}"
  "${python_bin}" -B "${resource_contract}" seal-terminal-host-memory-gate \
    --output "${host_memory_terminal_receipt}" \
    --monitor-exit-status "${host_memory_monitor_status}" >/dev/null || \
    fail "bound-supervisor terminal host memory gate seal failed"
  [[ -f "${host_memory_terminal_receipt}" && ! -L "${host_memory_terminal_receipt}" ]] || fail "terminal host memory receipt is absent"
  "${python_bin}" -B "${controller}" validate-terminal-host-cgroup-memory-receipt \
    --run-root "${run_root}" --require-live-child-cgroup >/dev/null || \
    fail "terminal host sampled-current/cgroup-binding gate failed"
  [[ -z "$(jobs -pr)" ]] || fail "background process remained after clean host monitor stop"
  exit 0
fi

[[ $# == 0 ]] || fail "parent launcher takes no arguments"
[[ ! -e "${run_root}/logs" && ! -L "${run_root}/logs" ]] || fail "log root must be fresh"
[[ ! -e "${run_root}/generation" && ! -L "${run_root}/generation" ]] || fail "generation output root must be fresh"
[[ ! -e "${run_root}/cache" && ! -L "${run_root}/cache" ]] || fail "NFS cache root is forbidden"
[[ ! -e "${run_root}/gpu-admission-receipt.json" && ! -L "${run_root}/gpu-admission-receipt.json" ]] || fail "GPU admission receipt must be fresh"
[[ ! -e "${run_root}/logs/host-cgroup-memory-terminal-receipt.json" && ! -L "${run_root}/logs/host-cgroup-memory-terminal-receipt.json" ]] || fail "terminal host cgroup memory receipt must be fresh"
[[ ! -e "${run_root}/generation-closure-receipt.json" && ! -L "${run_root}/generation-closure-receipt.json" ]] || fail "generation closure receipt must be fresh"
[[ ! -e "${run_root}/controller-completion.json" && ! -L "${run_root}/controller-completion.json" ]] || fail "completion must be fresh"

assert_parent_running() {
  local record
  record="$(scontrol show job -o "${holder_job}")"
  [[ "${record}" == *"JobId=${holder_job} "* && "${record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
  [[ "${record}" == *"UserId=${holder_user}"* && "${record}" == *"NodeList=${holder_node}"* ]] || fail "holder owner/node differs"
  [[ "${record}" == *"NumCPUs=64"* && "${record}" == *"AllocTRES=cpu=64,mem=64G,"* && "${record}" == *"gres/gpu:mi210=8"* ]] || fail "holder 64GiB/8GPU resource closure differs"
}
assert_idle_once() {
  local steps snapshot count memory_count busy kfd_owners
  assert_parent_running
  steps="$(numbered_steps)"; [[ -z "${steps}" ]] || fail "holder has a numbered child: ${steps}"
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if((v+0)!=0)print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || fail "holder GPUs are not idle exact8"
  kfd_owners="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'fuser /dev/kfd 2>/dev/null || true')"
  [[ -z "${kfd_owners//[[:space:]]/}" ]] || fail "holder has workload /dev/kfd owners: ${kfd_owners}"
}
assert_idle_twice() { assert_idle_once; sleep 2; assert_idle_once; }
assert_topology() {
  local topology
  topology="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showtopo')"
  assert_topology_text "${topology}"
}
assert_port_free() {
  local port="$1" found
  found="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${port}'")"
  [[ -z "${found}" ]] || fail "master port is occupied: ${port}"
}

assert_idle_twice
assert_topology
for port in "${master_port}" "$((master_port + 1))" "$((master_port + 2))" "$((master_port + 3))"; do assert_port_free "${port}"; done
[[ -z "$(numbered_steps)" ]] || fail "late numbered child appeared before output creation"
mkdir -m 0700 "${run_root}/logs" "${run_root}/generation"

child_pid=""
cleanup_child() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup_child EXIT INT TERM HUP

assert_idle_twice
for port in "${master_port}" "$((master_port + 1))" "$((master_port + 2))" "$((master_port + 3))"; do assert_port_free "${port}"; done
[[ -z "$(numbered_steps)" ]] || fail "late numbered child appeared"
set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --immediate=5 \
  --cpus-per-task=32 --mem=60G --gpus-per-task=8 --gpu-bind=none --gres-flags=enforce-binding \
  env \
    GADP_CONFIRM="${confirm}" GADP_PHASE="${phase}" GADP_SPLIT="${split}" \
    GADP_RUN_ROOT="${run_root}" GADP_MASTER_PORT="${master_port}" \
    GADP_CONTROLLER_PLAN="${controller_plan}" GADP_CONTROLLER_PLAN_SHA256="${controller_plan_sha}" \
    GADP_UPSTREAM_PLAN="${upstream_plan}" GADP_UPSTREAM_PLAN_SHA256="${upstream_plan_sha}" \
    GADP_METHOD_ROOT="${method_root}" GADP_METHOD_ARCHIVE="${method_archive}" GADP_METHOD_ARCHIVE_SHA256="${method_archive_sha}" \
    GADP_METHOD_MANIFEST="${method_manifest}" GADP_METHOD_MANIFEST_SHA256="${method_manifest_sha}" GADP_METHOD_REVISION="${method_revision}" \
    GADP_PYTHON_BIN="${python_bin}" GADP_PYTHON_SHA256="${python_sha}" \
    GADP_CONTROLLER_SHA256="${controller_sha}" GADP_LAUNCHER_SHA256="${launcher_sha}" GADP_GENERATOR_SHA256="${generator_sha}" \
    GADP_RESOURCE_CONTRACT_SHA256="${resource_contract_sha}" \
    GADP_R10_COMPILE_SMOKE_RECEIPT="${r10_compile_smoke_receipt}" GADP_R10_COMPILE_SMOKE_RECEIPT_SHA256="${r10_compile_smoke_receipt_sha}" \
    GADP_R10_GENERATION_LOG="${r10_generation_log}" GADP_R10_GENERATION_LOG_SHA256="${r10_generation_log_sha}" \
    bash "${launcher}" __child >"${run_root}/logs/generation-confirmation-all8-serial4.log" 2>&1 &
child_pid=$!
wait "${child_pid}"; status=$?
child_pid=""
set -e
if (( status != 0 )); then tail -n 240 "${run_root}/logs/generation-confirmation-all8-serial4.log" >&2 || true; fail "all8 serial-four-shard child failed status=${status}"; fi
[[ -f "${run_root}/logs/host-cgroup-memory-terminal-receipt.json" && ! -L "${run_root}/logs/host-cgroup-memory-terminal-receipt.json" ]] || fail "terminal host cgroup memory receipt missing"
printf 'slurm_child_gpus=8 world_size_per_model=4 compile_smoke_world4_invocation_count=1 formal_run_sp4_shard_process_count=4 formal_world4_model_invocation_count=40 total_native_model_invocation_count=41 host_cgroup_current_pid_leaf_identity_bound=true host_cgroup_leaf_memory_max_inherited=true host_cgroup_governing_ancestor_nearest_finite=true host_cgroup_governing_scope_exact_slurm_step_user=true host_cgroup_memory_max_exact_gib=60 host_sampled_current_safe_ceiling_gib=56 host_sample_interval_ns=10000000 host_max_sample_gap_ns=100000000 host_live_tail_max_age_ns=100000000 host_monitor_started_before_compile_smoke=true host_monitor_wait_exit_status_zero=true terminal_gate_after_bound_supervisor_wait=true host_monitor_clean_terminal_after_formal40=true host_oom=0 host_oom_kill=0 t2v_rank_gpu_memory_limit_gib=52 compile_smoke_world4_gpu_peaks_below_limit=true child_exit=0 all_model_invocations_strictly_serial=true per_rank_node_local_cache=true nfs_comgr_tmp_rejected=true concurrent_model_replicas=1 rank_action_family_partition=false\n' >"${run_root}/logs/generation-confirmation-all8-serial4.status"

"${python_bin}" -B "${generator}" audit \
  --plan "${upstream_plan}" --expected-plan-sha256 "${upstream_plan_sha}" \
  --generation-root "${run_root}/generation/seed1-sp4-a" \
  --generation-root "${run_root}/generation/seed1-sp4-b" \
  --generation-root "${run_root}/generation/seed2-sp4-a" \
  --generation-root "${run_root}/generation/seed2-sp4-b" \
  --output "${run_root}/generation-audit.json" \
  --gap-output "${run_root}/generation-gap-after-run.json" >/dev/null || fail "confirmation generation audit failed"
[[ -f "${run_root}/generation-audit.json" && ! -L "${run_root}/generation-audit.json" ]] || fail "confirmation generation audit receipt missing"
[[ -f "${run_root}/logs/compile-smoke-receipt.json" && ! -L "${run_root}/logs/compile-smoke-receipt.json" ]] || fail "compile smoke receipt missing"
[[ -f "${run_root}/gpu-admission-receipt.json" && ! -L "${run_root}/gpu-admission-receipt.json" ]] || fail "GPU admission receipt missing"
"${python_bin}" -B "${controller}" seal-generation-closure \
  --plan "${upstream_plan}" --expected-plan-sha256 "${upstream_plan_sha}" \
  --generation-root "${run_root}/generation/seed1-sp4-a" \
  --generation-root "${run_root}/generation/seed1-sp4-b" \
  --generation-root "${run_root}/generation/seed2-sp4-a" \
  --generation-root "${run_root}/generation/seed2-sp4-b" \
  --generation-audit "${run_root}/generation-audit.json" \
  --output "${run_root}/generation-closure-receipt.json" >/dev/null || fail "exact confirmation40 generation directory/receipt closure failed"
[[ -f "${run_root}/generation-closure-receipt.json" && ! -L "${run_root}/generation-closure-receipt.json" ]] || fail "generation closure receipt missing"

printf 'schema=bernini-generic-action-confirmation40-launcher-status-v3\nphase=generation\nsplit=confirmation\nholder_job=%s\nholder_node=%s\nslurm_child_gpus=8\nnumbered_slurm_children=1\ncompile_smoke_world4_invocation_count=1\ncompile_smoke_full_native_steps=40\ncompile_smoke_disposable=true\ncompile_smoke_precedes_formal40=true\nformal_run_sp4_shard_process_count=4\nformal_world4_model_invocation_count=40\ntotal_native_model_invocation_count=41\nall_model_invocations_strictly_serial=true\nper_rank_node_local_cache=true\nnfs_comgr_tmp_rejected=true\nserialized_world4_host_checkpoint_load=true\nmodel_load_lock_node_local=true\nmodel_load_lock_held_through_gpu_move_and_malloc_trim=true\nworld4_all_renderer_loads_complete_barrier_before_source_tokenizer_setup=true\nresource_lifecycle_receipt_proves_all_four_load_complete_before_first_sampling=true\ncompile_smoke_asserts_world4_load_completion_ordering=true\nt2v_text_encoder_rank_gpu_residency_required=true\nt2v_text_encoder_exact_cpu_offload_suppressed_once_per_rank=true\nt2v_text_encoder_retired_only_with_renderer=true\nt2v_rank_gpu_memory_limit_gib=52\ncompile_smoke_per_rank_gpu_peak_allocated_reserved_required=true\ncompile_smoke_all_rank_gpu_peak_reserved_strictly_below_limit=true\nhost_cgroup_sample_monitor_started_before_compile_smoke=true\nhost_cgroup_current_pid_leaf_identity_bound=true\nhost_cgroup_leaf_memory_max_inherited=true\nhost_cgroup_governing_ancestor_nearest_finite=true\nhost_cgroup_governing_scope_exact_slurm_step_user=true\nhost_cgroup_sample_interval_ns=10000000\nhost_cgroup_max_sample_gap_ns=100000000\nhost_live_tail_max_age_ns=100000000\nhost_cgroup_memory_max_exactly_60_gib=true\nhost_sampled_current_safe_ceiling_gib=56\ncompile_smoke_host_sampled_peak_strictly_below_56_gib=true\ncompile_smoke_host_monitor_alive_before_formal40=true\ncompile_smoke_zero_oom_and_oom_kill_before_formal40=true\nformal_candidate_boundary_host_monitor_checks_required=true\nterminal_host_sampled_current_receipt_required=true\nterminal_host_sampled_peak_strictly_below_56_gib=true\nterminal_host_monitor_wait_exit_status_zero=true\nterminal_gate_created_after_bound_supervisor_wait=true\nterminal_host_monitor_clean_exit=true\nterminal_zero_oom_and_oom_kill=true\nr10_smoke_authority_derived_from_pinned_receipt_and_log=true\nr10_smoke_mp4_whole_file_sha256_exact_required=true\nr10_smoke_gaussian_tensor_identity_exact_required=true\nr10_smoke_clean_latent_generated_identity_exact_required=true\ncurrent_smoke_physical_safetensors_safe_open_required=true\ncurrent_smoke_exact_single_tensor_key_required=true\ncurrent_smoke_exact_safetensors_metadata_required=true\ncurrent_smoke_tensor_identity_recomputed_from_physical_values=true\ncurrent_smoke_physical_identity_bound_to_receipt_all_rank_generated_and_r10=true\nsafetensors_container_sha256_cross_process_equivalence_required=false\nt2v_vae_load_deferred_until_rank0_post_sampling=true\nworld4_renderer_retirement_barrier_before_rank_zero_vae_load=true\nphysical_index_pci_unique_id_join_replayed=true\nper_shard_observed_uuid_pci_bus_join_before_model_forward=true\nhip_logical_order_is_observation_only=true\nper_shard_exact_physical_set_verified=true\nlogical_order_permutation_allowed=true\ncross_island_shard_visibility_rejected=true\ngeneration_directory_exact_member_closure=true\nconcurrent_model_replicas=1\nsealed_shard_order=seed1-sp4-a-confirmation,seed1-sp4-b-confirmation,seed2-sp4-a-confirmation,seed2-sp4-b-confirmation\nrank_action_family_partition=false\nprobe_receipt_pinned=false\ndynamic_probe_is_release_authority=false\nindependent_blind_review_present=false\nconfirmation_authorized=true\nphi_authorized=false\ngenerated_media_is_editor_input_or_target=false\noptimizer_created=false\noptimizer_authorized=false\nparent_released_or_cancelled=false\n' \
  "${holder_job}" "${holder_node}" >"${run_root}/launcher.status"
echo "GENERIC_ACTION_CONFIRMATION40_GENERATION_R3_COMPLETE candidate_count=40 compile_smoke=1 shard_runners=4 serial_native_invocations=41 serialized_host_load=true world4_load_completion_barrier=true t5_rank_gpu_residency=true t5_world4_gpu_peak_reserved_lt_52g=true host_cgroup_current_pid_leaf_identity_bound=true host_cgroup_leaf_memory_max_inherited=true host_cgroup_governing_ancestor_nearest_finite=true host_cgroup_governing_scope_exact_slurm_step_user=true host_memory_max_exact_60g=true host_sampled_current_peak_lt_56g=true host_sample_interval_ns=10000000 host_max_sample_gap_ns=100000000 host_live_tail_max_age_ns=100000000 host_oom=0 host_oom_kill=0 host_monitor_wait_exit_status_zero=true terminal_gate_after_bound_supervisor_wait=true host_monitor_clean_terminal_after_formal40=true r10_smoke_mp4_gaussian_latent_byte_parity_required=true r10_artifact_parity=true r10_mp4_tensor_parity=true current_physical_safe_open=true exact_single_key_metadata=true safetensors_container_sha_equivalence=false load_ordering_receipt=true world4_retirement_barrier=true rank0_deferred_vae=true exact_closure=true review=false phi=false optimizer=false"
