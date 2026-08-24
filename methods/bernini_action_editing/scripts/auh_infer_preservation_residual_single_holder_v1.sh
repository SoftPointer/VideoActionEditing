#!/usr/bin/env bash
# Reserve all8 only after a completed training bundle is available; retain parent.

set -Eeuo pipefail
umask 077

holder_job="${PRESERVATION_INFER_HOLDER_JOB:?set holder job}"
holder_node="${PRESERVATION_INFER_HOLDER_NODE:?set holder node}"
method_root="${PRESERVATION_INFER_METHOD_ROOT:?set method root}"
training_bundle="${PRESERVATION_INFER_TRAINING_BUNDLE:?set training bundle}"
run_root="${PRESERVATION_INFER_RUN_ROOT:?set run root}"
runtime_revision="${PRESERVATION_INFER_RUNTIME_REVISION:?set revision}"
runtime_archive_sha="${PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256:?set archive SHA}"
base_port="${PRESERVATION_INFER_BASE_PORT:?set base port}"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python

fail() { echo "[preservation-infer-holder] ERROR: $*" >&2; exit 2; }
case "${holder_job}:${holder_node}" in
  135407:auh7-1b-gpu-260|135411:auh7-1b-gpu-214) ;;
  *) fail "holder allowlist differs" ;;
esac
[[ -d "${method_root}" && -d "${training_bundle}" && ! -L "${training_bundle}" ]] || fail "input directory differs"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh"
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ && "${runtime_archive_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "source binding differs"
[[ "${base_port}" =~ ^[0-9]+$ ]] && (( base_port >= 1024 && base_port <= 65533 )) || fail "port differs"
job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobState=RUNNING"* && "${job_record}" == *"NodeList=${holder_node}"* && "${job_record}" == *"UserId=guangyi.chen"* ]] || fail "holder differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder has active child"
assert_idle() {
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && -z "${busy}" ]] || fail "holder GPUs are not idle exact8"
}
assert_idle
sleep 2
assert_idle

adapter="${training_bundle}/adapter.safetensors"
receipt="${training_bundle}/receipt.json"
[[ -f "${adapter}" && -f "${receipt}" ]] || fail "training bundle incomplete"
adapter_sha="$(sha256sum "${adapter}" | awk '{print $1}')"
receipt_sha="$(sha256sum "${receipt}" | awk '{print $1}')"
registry="${method_root}/assets/self_guided_action_field_core2_v1.json"
registry_sha="$(sha256sum "${registry}" | awk '{print $1}')"
launcher_sha="$(sha256sum "${method_root}/scripts/auh_infer_preservation_residual_single_holder_v1.sh" "${method_root}/scripts/auh_infer_preservation_residual_exec_v1.sh" | sha256sum | awk '{print $1}')"

mkdir -m 0700 "${run_root}" "${run_root}/logs"
output_root="${run_root}/outputs"
set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=64 --mem=60G --gres=gpu:mi210:8 \
  env PYTHONPATH="${method_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
    PRESERVATION_INFER_METHOD_ROOT="${method_root}" PRESERVATION_INFER_PYTHON_BIN="${python_bin}" \
    PRESERVATION_INFER_REGISTRY="${registry}" PRESERVATION_INFER_REGISTRY_SHA256="${registry_sha}" \
    PRESERVATION_INFER_TRAINING_BUNDLE="${training_bundle}" PRESERVATION_INFER_ADAPTER_SHA256="${adapter_sha}" \
    PRESERVATION_INFER_RECEIPT_SHA256="${receipt_sha}" PRESERVATION_INFER_OUTPUT_ROOT="${output_root}" \
    PRESERVATION_INFER_RUNTIME_REVISION="${runtime_revision}" PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256="${runtime_archive_sha}" \
    PRESERVATION_INFER_LAUNCHER_SHA256="${launcher_sha}" PRESERVATION_INFER_DOG_PORT="${base_port}" \
    PRESERVATION_INFER_HUMAN_PORT="$((base_port + 1))" \
    bash "${method_root}/scripts/auh_infer_preservation_residual_exec_v1.sh" \
  >"${run_root}/logs/infer.log" 2>&1
status=$?
set -e
printf 'child_exit=%s\nparent_not_released=true\n' "${status}" >"${run_root}/controller.status"
(( status == 0 )) || { tail -n 180 "${run_root}/logs/infer.log" >&2 || true; exit "${status}"; }
printf 'COMPLETE holder=%s node=%s parent_retained=true\n' "${holder_job}" "${holder_node}" >"${run_root}/controller.COMPLETE"
