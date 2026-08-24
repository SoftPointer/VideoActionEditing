#!/usr/bin/env bash
# One WORLD8 DP2/SP4 visual-context scope inside a retained holder: structural
# parity, four-microbatch backward feasibility, or formal exact80.
# The numbered child may finish; this script never releases/cancels the parent.
# This is training-only: it verifies checkpoint presence/loadability receipts,
# but does not decode fixed sentinels or publish an HTML/review COMPLETE marker.

set -Eeuo pipefail
umask 077

fail() { echo "[clean-source-visual-context-holder] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly holder_job="${CSVC_HOLDER_JOB:?wrapper must bind holder job}"
readonly holder_node="${CSVC_HOLDER_NODE:?wrapper must bind holder node}"
readonly memory_input_kind="${CSVC_MEMORY_INPUT_KIND:?wrapper must bind arm}"
readonly execution_scope="${CSVC_EXECUTION_SCOPE:?wrapper must bind execution scope}"
readonly run_root="${CSVC_RUN_ROOT:?set fresh arm run root}"
readonly method_root="${CSVC_METHOD_ROOT:?set sealed method root}"
readonly source_manifest="${CSVC_SOURCE_ONLY_MANIFEST:?set source-only 64/16/8 manifest}"
readonly source_manifest_sha="${CSVC_SOURCE_ONLY_MANIFEST_SHA256:?set manifest SHA}"
readonly stage_a_admission="${CSVC_STAGE_A_ADMISSION:-}"
readonly stage_a_admission_sha="${CSVC_STAGE_A_ADMISSION_SHA256:-}"
readonly formal_pair_admission="${CSVC_FORMAL_PAIR_ADMISSION:-}"
readonly formal_pair_admission_sha="${CSVC_FORMAL_PAIR_ADMISSION_SHA256:-}"
readonly expected_initial_parameter_digest="${CSVC_EXPECTED_INITIAL_PARAMETER_DIGEST:-}"
readonly preflight_pair_receipt="${CSVC_PREFLIGHT_PAIR_RECEIPT:-}"
readonly preflight_pair_receipt_sha="${CSVC_PREFLIGHT_PAIR_RECEIPT_SHA256:-}"
readonly method_revision="${CSVC_METHOD_REVISION:?set source revision}"
readonly method_archive="${CSVC_METHOD_ARCHIVE:?set sealed source archive}"
readonly method_archive_sha="${CSVC_METHOD_ARCHIVE_SHA256:?set source archive SHA}"
readonly method_manifest="${CSVC_METHOD_MANIFEST:?set sealed source manifest}"
readonly method_manifest_sha="${CSVC_METHOD_MANIFEST_SHA256:?set source manifest SHA}"
readonly master_port="${CSVC_MASTER_PORT:?set arm master port}"

readonly holder_user=guangyi.chen
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly expected_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly expected_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly expected_checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

case "${holder_job}:${holder_node}:${memory_input_kind}" in
  135980:auh7-1b-gpu-239:clean_source) ;;
  135981:auh7-1b-gpu-234:same_noise_forward_noised_source) ;;
  136140:auh7-1b-gpu-215:clean_source) ;;
  136140:auh7-1b-gpu-215:same_noise_forward_noised_source) ;;
  *) fail "holder/node/arm is outside the registered bindings" ;;
esac
case "${execution_scope}" in
  formal-exact80)
    [[ -n "${stage_a_admission}" && -n "${stage_a_admission_sha}" && -n "${formal_pair_admission}" && -n "${formal_pair_admission_sha}" && -n "${expected_initial_parameter_digest}" ]] || fail "formal exact80 requires decoded Stage-A, shared pair admission, and preflighted initialization"
    [[ -z "${preflight_pair_receipt}" && -z "${preflight_pair_receipt_sha}" ]] || fail "formal exact80 must not consume smoke-only pair inputs"
    ;;
  structural-parity-preflight)
    [[ -z "${stage_a_admission}" && -z "${stage_a_admission_sha}" && -z "${formal_pair_admission}" && -z "${formal_pair_admission_sha}" && -z "${expected_initial_parameter_digest}" && -z "${preflight_pair_receipt}" && -z "${preflight_pair_receipt_sha}" ]] || fail "structural preflight must not consume Stage-A/pair admission/initial digest"
    ;;
  backward-feasibility-preflight)
    [[ -z "${stage_a_admission}" && -z "${stage_a_admission_sha}" && -z "${formal_pair_admission}" && -z "${formal_pair_admission_sha}" && -z "${expected_initial_parameter_digest}" ]] || fail "backward feasibility must remain pre-Stage-A and pre-optimizer"
    [[ -n "${preflight_pair_receipt}" && -n "${preflight_pair_receipt_sha}" ]] || fail "backward feasibility requires the prior paired structural receipt"
    ;;
  *) fail "execution scope differs" ;;
esac
[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
for name in run_root method_root source_manifest method_archive method_manifest; do
  value="${!name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${name} path differs"
done
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
for path in "${source_manifest}" "${checkpoint_manifest}" "${method_archive}" "${method_manifest}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
for digest in source_manifest_sha method_archive_sha method_manifest_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method revision differs"
[[ "$(sha256_file "${source_manifest}")" == "${source_manifest_sha}" ]] || fail "source-only manifest SHA differs"
stage_a_args=()
backward_args=()
if [[ "${execution_scope}" == formal-exact80 ]]; then
  [[ "${stage_a_admission}" == /vast/users/guangyi.chen/* && "${stage_a_admission}" != / ]] || fail "Stage-A admission path differs"
  [[ -f "${stage_a_admission}" && ! -L "${stage_a_admission}" && "$(readlink -f -- "${stage_a_admission}")" == "${stage_a_admission}" ]] || fail "Stage-A admission input differs"
  [[ "${stage_a_admission_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "Stage-A admission SHA differs"
  [[ "$(sha256_file "${stage_a_admission}")" == "${stage_a_admission_sha}" ]] || fail "Stage-A admission SHA differs"
  stage_a_args=(--stage-a-admission "${stage_a_admission}" --expected-stage-a-admission-sha256 "${stage_a_admission_sha}")
  [[ "${formal_pair_admission}" == /vast/users/guangyi.chen/* && "${formal_pair_admission}" != / ]] || fail "formal pair admission path differs"
  [[ -f "${formal_pair_admission}" && ! -L "${formal_pair_admission}" && "$(readlink -f -- "${formal_pair_admission}")" == "${formal_pair_admission}" ]] || fail "formal pair admission input differs"
  [[ "${formal_pair_admission_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "formal pair admission SHA differs"
  [[ "${expected_initial_parameter_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "preflighted initial parameter digest differs"
  [[ "$(sha256_file "${formal_pair_admission}")" == "${formal_pair_admission_sha}" ]] || fail "formal pair admission SHA differs"
  stage_a_args+=(--formal-pair-admission "${formal_pair_admission}" --expected-formal-pair-admission-sha256 "${formal_pair_admission_sha}" --expected-initial-parameter-digest "${expected_initial_parameter_digest}")
fi
if [[ "${execution_scope}" == backward-feasibility-preflight ]]; then
  [[ "${preflight_pair_receipt}" == /vast/users/guangyi.chen/* && "${preflight_pair_receipt}" != / ]] || fail "preflight pair receipt path differs"
  [[ -f "${preflight_pair_receipt}" && ! -L "${preflight_pair_receipt}" && "$(readlink -f -- "${preflight_pair_receipt}")" == "${preflight_pair_receipt}" ]] || fail "preflight pair receipt input differs"
  [[ "${preflight_pair_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "preflight pair receipt SHA differs"
  [[ "$(sha256_file "${preflight_pair_receipt}")" == "${preflight_pair_receipt_sha}" ]] || fail "preflight pair receipt SHA differs"
  backward_args=(--preflight-pair-receipt "${preflight_pair_receipt}" --expected-preflight-pair-receipt-sha256 "${preflight_pair_receipt_sha}")
fi
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || fail "checkpoint manifest SHA differs"
[[ "$(sha256_file "${method_archive}")" == "${method_archive_sha}" ]] || fail "method archive SHA differs"
[[ "$(sha256_file "${method_manifest}")" == "${method_manifest_sha}" ]] || fail "method manifest SHA differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -f "${method_root}/train_clean_source_visual_context_stage_b_v1.py" ]] || fail "Stage-B runner missing"
readonly rank_exec_source="${method_root}/scripts/auh_preservation_rank_cache_exec_v1.sh"
[[ -f "${rank_exec_source}" && ! -L "${rank_exec_source}" ]] || fail "rank cache runner missing"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh"

job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "holder owner/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder already has a numbered child"

assert_idle() {
  local snapshot count busy
  snapshot="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" 'rocm-smi --showuse --showmemuse --showpids')"
  count="$(awk '/GPU use \(%\)/ {n++} END {print n+0}' <<<"${snapshot}")"
  busy="$(awk '/GPU use \(%\)/ || /GPU Memory Allocated \(VRAM%\)/ {v=$NF; gsub(/[^0-9]/,"",v); if ((v+0)!=0) print}' <<<"${snapshot}")"
  [[ "${count}" == 8 && -z "${busy}" ]] || fail "holder GPU inventory is not idle exact8"
}
assert_idle
sleep 2
assert_idle
[[ -z "$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${holder_node}" "ss -H -ltn 'sport = :${master_port}'")" ]] || fail "master port occupied"

mkdir -m 0700 "${run_root}" "${run_root}/logs"
readonly training_output="${run_root}/training"
readonly checkpoint_output="${run_root}/checkpoints"
readonly rank_exec="${run_root}/rank_exec.sh"
cp -- "${rank_exec_source}" "${rank_exec}"
chmod 0500 "${rank_exec}"
[[ "$(sha256_file "${rank_exec}")" == "$(sha256_file "${rank_exec_source}")" ]] || fail "rank cache scratch copy differs"

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=64 --mem=64G --gres=gpu:mi210:8 \
  env BERNINI_HELDOUT_RANK_CACHE_TOKEN="csvc-${memory_input_kind}-${method_revision:0:10}" \
    BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
    "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
      --master_addr=127.0.0.1 --master_port="${master_port}" --no_python "${rank_exec}" \
      "${method_root}/train_clean_source_visual_context_stage_b_v1.py" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-only-manifest "${source_manifest}" \
      --expected-source-only-manifest-sha256 "${source_manifest_sha}" \
      "${stage_a_args[@]}" \
      "${backward_args[@]}" \
      --output "${training_output}" --checkpoint-output-root "${checkpoint_output}" \
      --mode clean-source-visual-context-stage-b-v1 --execution-scope "${execution_scope}" \
      --parallel-topology world8-dp2-sp4 \
      --memory-input-kind "${memory_input_kind}" --optimizer-steps 80 \
      --expected-bernini-commit "${expected_bernini_commit}" \
      --expected-veomni-commit "${expected_veomni_commit}" \
      --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha}" \
      --expected-checkpoint-content-manifest-sha256 "${expected_checkpoint_manifest_sha}" \
      --method-source-revision "${method_revision}" \
      --method-source-archive "${method_archive}" \
      --method-source-archive-sha256 "${method_archive_sha}" \
      --method-source-manifest "${method_manifest}" \
      --method-source-manifest-sha256 "${method_manifest_sha}" \
      --ack-upstream-training-use-forbidden --ack-user-authorized-exploratory-training \
  >"${run_root}/logs/train.log" 2>&1
status=$?
set -e

printf 'holder_job=%s\nholder_node=%s\nmemory_input_kind=%s\nexecution_scope=%s\nchild_exit=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${memory_input_kind}" "${execution_scope}" "${status}" \
  >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 200 "${run_root}/logs/train.log" >&2 || true
  exit "${status}"
fi
[[ -f "${training_output}/receipt.json" && ! -L "${training_output}/receipt.json" ]] || fail "training receipt missing"
if [[ "${execution_scope}" == formal-exact80 ]]; then
  for step in 00000000 00000020 00000040 00000060 00000080; do
    checkpoint_file="${checkpoint_output}/checkpoint_step_${step}.pt"
    [[ -f "${checkpoint_file}" && ! -L "${checkpoint_file}" ]] || fail "immutable cadence checkpoint ${step} missing"
  done
  printf 'optimizer_steps=80\ncheckpoint_steps=0,20,40,60,80\ndecode_chain_ready=true\ndecoded_checkpoint_inference_executed=false\nhtml_review_generated=false\nreview_complete=false\n' >>"${run_root}/controller.status"
  printf 'TRAINING_COMPLETE_CSVC_STAGE_B holder=%s node=%s arm=%s steps=80 decoded_review=false parent_retained=true\n' \
    "${holder_job}" "${holder_node}" "${memory_input_kind}" \
    >"${run_root}/controller.TRAINING_COMPLETE"
  echo "TRAINING_COMPLETE_CSVC_STAGE_B output=${run_root} decoded_review=false"
elif [[ "${execution_scope}" == structural-parity-preflight ]]; then
  [[ ! -e "${checkpoint_output}" && ! -L "${checkpoint_output}" ]] || fail "structural preflight created checkpoint root"
  "${python_bin}" -B -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); a=r["authority"]; assert r["execution_scope"]=="structural-parity-preflight"; assert a["optimizer_constructed"] is False; assert a["backward_executed"] is False; assert a["optimizer_step_count"]==0; assert a["checkpoint_written"] is False; assert a["checkpoint_root_created"] is False' "${training_output}/receipt.json" || fail "structural preflight authority receipt differs"
  printf 'optimizer_constructed=false\noptimizer_steps=0\nbackward_executed=false\ncheckpoint_written=false\nstep0_bit_exact_parity=true\nreview_complete=false\n' >>"${run_root}/controller.status"
  printf 'PREFLIGHT_COMPLETE_CSVC_STAGE_B holder=%s node=%s arm=%s optimizer=false checkpoint=false parent_retained=true\n' \
    "${holder_job}" "${holder_node}" "${memory_input_kind}" \
    >"${run_root}/controller.PREFLIGHT_COMPLETE"
  echo "PREFLIGHT_COMPLETE_CSVC_STAGE_B output=${run_root} optimizer=false checkpoint=false"
else
  [[ ! -e "${checkpoint_output}" && ! -L "${checkpoint_output}" ]] || fail "backward feasibility created checkpoint root"
  "${python_bin}" -B -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); a=r["authority"]; b=r["backward_feasibility"]; assert r["execution_scope"]=="backward-feasibility-preflight"; assert a["optimizer_constructed"] is False; assert a["optimizer_step_count"]==0; assert a["parameters_changed"] is False; assert a["checkpoint_written"] is False; assert b["microbatches_per_dp_arm"]==4; assert b["logical_records"]==8; assert b["parameters"]["unchanged"] is True' "${training_output}/receipt.json" || fail "backward feasibility authority receipt differs"
  printf 'optimizer_constructed=false\noptimizer_steps=0\nbackward_microbatches_per_dp_arm=4\ndp2_sp4_gradient_sync=true\nparameters_changed=false\ncheckpoint_written=false\nreview_complete=false\n' >>"${run_root}/controller.status"
  printf 'BACKWARD_PREFLIGHT_COMPLETE_CSVC_STAGE_B holder=%s node=%s arm=%s optimizer=false checkpoint=false parent_retained=true\n' \
    "${holder_job}" "${holder_node}" "${memory_input_kind}" \
    >"${run_root}/controller.BACKWARD_PREFLIGHT_COMPLETE"
  echo "BACKWARD_PREFLIGHT_COMPLETE_CSVC_STAGE_B output=${run_root} optimizer=false checkpoint=false"
fi
