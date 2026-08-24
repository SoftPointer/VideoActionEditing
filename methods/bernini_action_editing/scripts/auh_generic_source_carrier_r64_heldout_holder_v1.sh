#!/usr/bin/env bash
# Run one preservation-only R64 held-out decode in an existing retained holder.
# Never cancels, releases, requeues, or signals the parent allocation.

set -Eeuo pipefail
umask 077

fail() { echo "[r64-heldout] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly confirmation="${R64_EVAL_CONFIRM:?explicit confirmation required}"
readonly holder_job="${R64_EVAL_HOLDER_JOB:?set holder job}"
readonly holder_node="${R64_EVAL_HOLDER_NODE:?set holder node}"
readonly method_root="${R64_EVAL_METHOD_ROOT:?set exact extracted method root}"
readonly method_manifest="${R64_EVAL_METHOD_MANIFEST:?set release manifest}"
readonly method_manifest_sha="${R64_EVAL_METHOD_MANIFEST_SHA256:?pin release manifest}"
readonly training_receipt="${R64_EVAL_TRAINING_RECEIPT:?set R64 receipt}"
readonly training_receipt_sha="${R64_EVAL_TRAINING_RECEIPT_SHA256:?pin R64 receipt}"
readonly r64_checkpoint="${R64_EVAL_CHECKPOINT:?set R64 checkpoint}"
readonly r64_checkpoint_sha="${R64_EVAL_CHECKPOINT_SHA256:?pin R64 checkpoint}"
readonly source_manifest="${R64_EVAL_SOURCE_MANIFEST:?set source manifest}"
readonly source_manifest_sha="${R64_EVAL_SOURCE_MANIFEST_SHA256:?pin source manifest}"
readonly run_root="${R64_EVAL_RUN_ROOT:?set fresh run root}"
readonly master_port="${R64_EVAL_MASTER_PORT:?set one free port}"
readonly runtime_revision="${R64_EVAL_RUNTIME_REVISION:?set content revision}"
readonly runtime_closure_sha="${R64_EVAL_RUNTIME_CLOSURE_SHA256:?set release closure SHA}"
readonly launcher_sha="${R64_EVAL_LAUNCHER_SHA256:?pin launcher}"
readonly python_bin="${R64_EVAL_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
readonly holder_user=guangyi.chen
readonly launch_token=launch-approved-r64-heldout-preservation-only
readonly expected_receipt_sha=0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f
readonly expected_checkpoint_sha=b037496df99ea01d5a7e3fa509aac4c451806a6e47ecb7a1070529abde249726
readonly expected_source_sha=128064fd335c4e48c567217c6e7bae43555a904875625c9d1e21178e6f7fcc3d
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${1:-}" in
  "") readonly role=parent ;;
  __child) readonly role=child ;;
  *) fail "launcher accepts only its internal __child role" ;;
esac
[[ "${confirmation}" == "${launch_token}" ]] || fail "launch confirmation differs"
case "${holder_job}:${holder_node}" in
  136309:auh7-1b-gpu-280|136141:auh7-1b-gpu-299) ;;
  *) fail "holder is outside the user-provided allowlist" ;;
esac
[[ "${training_receipt_sha}" == "${expected_receipt_sha}" ]] || fail "R64 receipt pin differs"
[[ "${r64_checkpoint_sha}" == "${expected_checkpoint_sha}" ]] || fail "R64 checkpoint pin differs"
[[ "${source_manifest_sha}" == "${expected_source_sha}" ]] || fail "source manifest pin differs"
for digest in method_manifest_sha training_receipt_sha r64_checkpoint_sha source_manifest_sha runtime_closure_sha launcher_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
for name in method_root method_manifest training_receipt r64_checkpoint source_manifest run_root python_bin; do
  value="${!name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${name} path differs"
done
[[ -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "method root differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for path in "${method_manifest}" "${training_receipt}" "${r64_checkpoint}" "${source_manifest}" "${checkpoint_manifest}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "release manifest bytes differ"
[[ "$(sha256_file "${training_receipt}")" == "${training_receipt_sha}" ]] || fail "R64 receipt bytes differ"
[[ "$(sha256_file "${r64_checkpoint}")" == "${r64_checkpoint_sha}" ]] || fail "R64 checkpoint bytes differ"
[[ "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "source manifest bytes differ"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha}" ]] || fail "base checkpoint manifest differs"

readonly runner="${method_root}/infer_generic_source_carrier_r64_heldout_v1.py"
readonly contract_source="${method_root}/generic_source_carrier_r64_heldout_contract_v1.py"
readonly html_builder="${method_root}/tools/build_generic_source_carrier_r64_heldout_html_v1.py"
readonly release_builder="${method_root}/tools/build_generic_source_carrier_r64_heldout_release_v1.py"
readonly source_preflight="${method_root}/tools/preflight_generic_source_carrier_r64_heldout_sources_v1.py"
readonly rank_exec="${method_root}/scripts/auh_generic_source_carrier_r64_heldout_rank_exec_v1.sh"
readonly launcher="${method_root}/scripts/auh_generic_source_carrier_r64_heldout_holder_v1.sh"
for path in "${runner}" "${contract_source}" "${html_builder}" "${release_builder}" "${source_preflight}" "${rank_exec}" "${launcher}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "runtime member differs: ${path}"
done
[[ -x "${rank_exec}" && -x "${launcher}" ]] || fail "runtime launchers are not executable"
[[ "$(sha256_file "${launcher}")" == "${launcher_sha}" ]] || fail "launcher bytes differ"
release_identity="$(PYTHONPATH="${method_root}" "${python_bin}" -B -c \
  'from pathlib import Path; import sys; from tools import build_generic_source_carrier_r64_heldout_release_v1 as r; v=r.validate_executed_release(method_root=Path(sys.argv[1]), manifest_path=Path(sys.argv[2]), expected_manifest_sha256=sys.argv[3]); print("|".join((v["content_closure_sha1"],v["content_closure_sha256"],v["component_pins"]["launcher_sha256"])))' \
  "${method_root}" "${method_manifest}" "${method_manifest_sha}")" \
  || fail "executed release closure differs"
IFS='|' read -r verified_revision verified_closure_sha verified_launcher_sha <<<"${release_identity}"
[[ "${runtime_revision}" == "${verified_revision}" ]] || fail "runtime revision is not the executed release"
[[ "${runtime_closure_sha}" == "${verified_closure_sha}" ]] || fail "runtime closure is not the executed release"
[[ "${launcher_sha}" == "${verified_launcher_sha}" ]] || fail "launcher pin is not the executed release"

assert_parent() {
  local record steps
  record="$(scontrol show job -o "${holder_job}")"
  [[ "${record}" == *"JobId=${holder_job} "* && "${record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
  [[ "${record}" == *"UserId=${holder_user}"* && "${record}" == *"NodeList=${holder_node}"* ]] || fail "holder owner/node differs"
  steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')"
  [[ -z "${steps}" ]] || fail "holder already has a numbered child: ${steps}"
}
assert_idle() {
  local snapshot count memory_count busy
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  memory_count="$(awk '/GPU Memory Allocated \(VRAM%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && "${memory_count}" == 8 && -z "${busy}" ]] || fail "holder GPU inventory is not idle exact8"
}

if [[ "${role}" == child ]]; then
  shift
  [[ $# == 0 ]] || fail "unexpected child arguments"
  [[ "${SLURM_JOB_ID:?Slurm child required}" == "${holder_job}" ]] || fail "child job differs"
  [[ "$(hostname -s)" == "${holder_node}" ]] || fail "child node differs"
  [[ "${SLURM_STEP_ID:?numbered child required}" =~ ^[0-9]+$ ]] || fail "child step differs"
  child_steps="$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')"
  [[ "${child_steps}" == "${holder_job}.${SLURM_STEP_ID}" ]] || fail "child is not the only numbered holder step"
  physical_gpus="${SLURM_STEP_GPUS:-}"
  case "${physical_gpus}" in 0,1,2,3|4,5,6,7) ;; *) fail "child lacks one contiguous XGMI4 island" ;; esac
  unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${physical_gpus}"
  export PYTHONPATH="${method_root}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export R64_HELDOUT_CACHE_TOKEN="r64-heldout-${runtime_revision:0:10}"
  export R64_HELDOUT_PYTHON_BIN="${python_bin}"
  export NATIVE_V_AXIS_LOAD_LOCK="${run_root}/renderer-load.lock"
  readonly source_preflight_receipt="${run_root}/source-media-preflight.json"
  [[ ! -e "${source_preflight_receipt}" && ! -L "${source_preflight_receipt}" ]] \
    || fail "source-media preflight receipt path is not fresh"
  "${python_bin}" -B "${source_preflight}" \
    --source-manifest "${source_manifest}" \
    --expected-source-manifest-sha256 "${source_manifest_sha}" \
    --output-receipt "${source_preflight_receipt}"
  [[ -f "${source_preflight_receipt}" && ! -L "${source_preflight_receipt}" \
     && "$(stat -c '%a' -- "${source_preflight_receipt}")" == 400 ]] \
    || fail "source-media preflight receipt differs"
  exec "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${master_port}" --no_python \
    "${rank_exec}" "${runner}" \
    --training-receipt "${training_receipt}" \
    --expected-training-receipt-sha256 "${training_receipt_sha}" \
    --expected-r64-checkpoint-sha256 "${r64_checkpoint_sha}" \
    --source-manifest "${source_manifest}" \
    --expected-source-manifest-sha256 "${source_manifest_sha}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --base-checkpoint "${base_checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --runtime-source-revision "${runtime_revision}" \
    --runtime-source-closure-sha256 "${runtime_closure_sha}" \
    --launcher-source-sha256 "${launcher_sha}" \
    --output-dir "${run_root}/evaluation"
fi

[[ $# == 0 ]] || fail "unexpected launcher arguments"
[[ ! -e "${run_root}" && ! -L "${run_root}" && "$(realpath -m -- "${run_root}")" == "${run_root}" ]] || fail "run root must be fresh"
assert_parent
assert_idle
sleep 2
assert_parent
assert_idle
[[ -z "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${master_port}'")" ]] || fail "master port is occupied"
mkdir -m 0700 "${run_root}" "${run_root}/logs"
: >"${run_root}/renderer-load.lock"
chmod 0400 "${run_root}/renderer-load.lock"
assert_parent

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --immediate=5 \
  --cpus-per-task=32 --mem=60G --gpus-per-task=4 --gpu-bind=none --gres-flags=enforce-binding \
  env R64_EVAL_CONFIRM="${confirmation}" R64_EVAL_HOLDER_JOB="${holder_job}" R64_EVAL_HOLDER_NODE="${holder_node}" \
    R64_EVAL_METHOD_ROOT="${method_root}" R64_EVAL_METHOD_MANIFEST="${method_manifest}" R64_EVAL_METHOD_MANIFEST_SHA256="${method_manifest_sha}" \
    R64_EVAL_TRAINING_RECEIPT="${training_receipt}" R64_EVAL_TRAINING_RECEIPT_SHA256="${training_receipt_sha}" \
    R64_EVAL_CHECKPOINT="${r64_checkpoint}" R64_EVAL_CHECKPOINT_SHA256="${r64_checkpoint_sha}" \
    R64_EVAL_SOURCE_MANIFEST="${source_manifest}" R64_EVAL_SOURCE_MANIFEST_SHA256="${source_manifest_sha}" \
    R64_EVAL_RUN_ROOT="${run_root}" R64_EVAL_MASTER_PORT="${master_port}" \
    R64_EVAL_RUNTIME_REVISION="${runtime_revision}" R64_EVAL_RUNTIME_CLOSURE_SHA256="${runtime_closure_sha}" \
    R64_EVAL_LAUNCHER_SHA256="${launcher_sha}" R64_EVAL_PYTHON_BIN="${python_bin}" \
    bash "${launcher}" __child >"${run_root}/logs/decode.log" 2>&1
status=$?
set -e
printf 'holder_job=%s\nholder_node=%s\nchild_exit=%s\ncomplete_action_result=false\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${status}" >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 240 "${run_root}/logs/decode.log" >&2 || true
  exit "${status}"
fi
[[ -f "${run_root}/evaluation/receipt.json" && ! -L "${run_root}/evaluation/receipt.json" ]] || fail "evaluation receipt missing"
"${python_bin}" -B "${html_builder}" --input-dir "${run_root}/evaluation" \
  --runtime-source-revision "${runtime_revision}" \
  --runtime-source-closure-sha256 "${runtime_closure_sha}" \
  --launcher-sha256 "${launcher_sha}" \
  >"${run_root}/logs/html.log" 2>&1 || fail "HTML builder failed"
[[ -f "${run_root}/evaluation/index.html" && -f "${run_root}/evaluation/html_receipt.json" ]] || fail "HTML packet incomplete"
printf 'R64_HELDOUT_PRESERVATION_COMPLETE holder=%s node=%s pairs=8 mp4=16 exact81=true exact40=true action_result=false html=%s parent_retained=true\n' \
  "${holder_job}" "${holder_node}" "${run_root}/evaluation/index.html" \
  >"${run_root}/controller.COMPLETE"
echo "R64_HELDOUT_PRESERVATION_COMPLETE output=${run_root}/evaluation action_result=false parent_retained=true"
