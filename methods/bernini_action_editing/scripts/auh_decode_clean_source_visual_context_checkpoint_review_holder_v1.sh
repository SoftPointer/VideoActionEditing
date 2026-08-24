#!/usr/bin/env bash
# Decode the fixed four-sentinel grid at 0/20/40/60/80 after exact80 training.
# Every child uses WORLD4/SP4, runs serially inside the retained parent holder,
# and returns GPUs to that holder.  This script never cancels/releases it.

set -Eeuo pipefail
umask 077

fail() { echo "[csvc-checkpoint-review] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly requested_source="${BASH_SOURCE[0]}"
[[ "${requested_source}" == /* && -f "${requested_source}" && ! -L "${requested_source}" ]] || \
  fail "controller must be invoked as an absolute plain file"
readonly controller_source="$(readlink -f -- "${requested_source}")"
readonly holder_job="${CSVC_HOLDER_JOB:?bind retained Stage-B holder job}"
readonly holder_node="${CSVC_HOLDER_NODE:?bind retained Stage-B holder node}"
readonly memory_input_kind="${CSVC_MEMORY_INPUT_KIND:?bind trained memory arm}"
readonly run_root="${CSVC_RUN_ROOT:?bind completed Stage-B run root}"
readonly method_root="${CSVC_METHOD_ROOT:?bind sealed method root}"
readonly review_manifest="${CSVC_REVIEW_MANIFEST:?bind fixed heldout review manifest}"
readonly review_manifest_sha="${CSVC_REVIEW_MANIFEST_SHA256:?bind review manifest SHA}"
readonly runtime_revision="${CSVC_REVIEW_RUNTIME_REVISION:?bind runtime source revision}"
readonly expected_runtime_closure="${CSVC_REVIEW_RUNTIME_CLOSURE_SHA256:?bind runtime closure SHA}"
readonly expected_controller_sha="${CSVC_REVIEW_CONTROLLER_SHA256:?bind controller SHA}"
readonly python_bin="${CSVC_REVIEW_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
readonly master_port_base="${CSVC_REVIEW_MASTER_PORT_BASE:-30240}"
readonly holder_user=guangyi.chen
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly expected_checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly expected_checkpoint_manifest_sha=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly expected_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly expected_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d

case "${holder_job}:${holder_node}:${memory_input_kind}" in
  135980:auh7-1b-gpu-239:clean_source) ;;
  135981:auh7-1b-gpu-234:same_noise_forward_noised_source) ;;
  *) fail "holder/node/arm lies outside the registered Stage-B pair" ;;
esac
for name in run_root method_root review_manifest python_bin bernini_root veomni_root base_checkpoint checkpoint_manifest; do
  value="${!name}"
  [[ "${value}" == /* && "${value}" =~ ^/[A-Za-z0-9._/-]+$ && "${value}" != / ]] || fail "${name} path differs"
done
for digest in review_manifest_sha expected_runtime_closure expected_controller_sha; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ "${master_port_base}" =~ ^[1-9][0-9]*$ ]] && (( master_port_base >= 1024 && master_port_base <= 65495 )) || fail "master port base differs"
[[ -d "${run_root}" && ! -L "${run_root}" && "$(readlink -f -- "${run_root}")" == "${run_root}" ]] || fail "run root differs"
[[ -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "method root differs"
for path in "${review_manifest}" "${checkpoint_manifest}" "${controller_source}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "sealed input differs: ${path}"
done
for path in "${python_bin}"; do [[ -x "${path}" && ! -L "${path}" ]] || fail "Python differs"; done
for path in "${bernini_root}" "${veomni_root}" "${base_checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "runtime root differs: ${path}"
done
[[ "$(sha256_file "${controller_source}")" == "${expected_controller_sha}" ]] || fail "controller source SHA differs"
[[ "$(sha256_file "${review_manifest}")" == "${review_manifest_sha}" ]] || fail "review manifest SHA differs"
[[ "$(sha256_file "${checkpoint_manifest}")" == "${expected_checkpoint_manifest_sha}" ]] || fail "base checkpoint manifest SHA differs"

readonly runner="${method_root}/infer_clean_source_visual_context_checkpoint_review_v1.py"
readonly route_runtime="${method_root}/clean_source_visual_context_checkpoint_decode_runtime_v1.py"
readonly review_contract="${method_root}/clean_source_visual_context_checkpoint_review_contract_v1.py"
readonly html_builder="${method_root}/tools/build_clean_source_visual_context_checkpoint_review_html_v1.py"
readonly rank_exec="${run_root}/rank_exec.sh"
for path in "${runner}" "${route_runtime}" "${review_contract}" "${html_builder}" "${rank_exec}"; do
  [[ -f "${path}" && ! -L "${path}" && "$(readlink -f -- "${path}")" == "${path}" ]] || fail "review runtime member differs: ${path}"
done
[[ -x "${rank_exec}" ]] || fail "rank cache runner is not executable"
runtime_closure="$(${python_bin} -B -c 'import hashlib,json,sys; rows=[{"name":p.rsplit("/",1)[-1],"sha256":hashlib.sha256(open(p,"rb").read()).hexdigest()} for p in sys.argv[1:]]; print(hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest())' "${runner}" "${route_runtime}" "${review_contract}" "${html_builder}")"
[[ "${runtime_closure}" == "${expected_runtime_closure}" ]] || fail "review runtime source closure differs"

readonly training_marker="${run_root}/controller.TRAINING_COMPLETE"
readonly training_receipt="${run_root}/training/receipt.json"
[[ -f "${training_marker}" && ! -L "${training_marker}" ]] || fail "exact80 TRAINING_COMPLETE marker is absent"
[[ -f "${training_receipt}" && ! -L "${training_receipt}" ]] || fail "training receipt is absent"
grep -Fq 'decoded_review=false' "${training_marker}" || fail "training marker is not the training-only handoff"
readonly training_receipt_sha="$(sha256_file "${training_receipt}")"
readonly shard_root="${run_root}/checkpoint-review-shards"
readonly html_output="${run_root}/checkpoint-review-html"
readonly load_lock="${run_root}/checkpoint-review-renderer-load.lock"
[[ ! -e "${shard_root}" && ! -L "${shard_root}" ]] || fail "checkpoint review shard root must be fresh"
[[ ! -e "${html_output}" && ! -L "${html_output}" ]] || fail "checkpoint review HTML output must be fresh"
mkdir -m 0700 "${shard_root}" "${run_root}/logs/checkpoint-review"
: >"${load_lock}"
chmod 0400 "${load_lock}"

job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "parent holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "parent holder owner/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder already has a numbered child"

readonly steps=(0 20 40 60 80)
for ordinal in "${!steps[@]}"; do
  step="${steps[${ordinal}]}"
  printf -v step_name '%08d' "${step}"
  output="${shard_root}/step-${step_name}"
  port=$((master_port_base + ordinal))
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "checkpoint step ${step} output is not fresh"
  srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
    --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=32 --mem=64G --gres=gpu:mi210:4 \
    env BERNINI_HELDOUT_RANK_CACHE_TOKEN="csvc-review-${memory_input_kind}-${runtime_revision:0:10}-${step_name}" \
      BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
      NATIVE_V_AXIS_LOAD_LOCK="${load_lock}" \
      "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="${port}" --no_python "${rank_exec}" \
        "${runner}" \
        --review-manifest "${review_manifest}" \
        --expected-review-manifest-sha256 "${review_manifest_sha}" \
        --training-receipt "${training_receipt}" \
        --expected-training-receipt-sha256 "${training_receipt_sha}" \
        --checkpoint-step "${step}" \
        --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
        --base-checkpoint "${base_checkpoint}" \
        --checkpoint-content-manifest "${checkpoint_manifest}" \
        --expected-checkpoint-content-manifest-sha256 "${expected_checkpoint_manifest_sha}" \
        --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha}" \
        --output-dir "${output}" --runtime-source-revision "${runtime_revision}" \
        --runtime-source-closure-sha256 "${runtime_closure}" \
        --launcher-source-sha256 "${expected_controller_sha}" \
        --expected-bernini-commit "${expected_bernini_commit}" \
        --expected-veomni-commit "${expected_veomni_commit}" \
    >"${run_root}/logs/checkpoint-review/step-${step_name}.log" 2>&1
  [[ -f "${output}/receipt.json" && ! -L "${output}/receipt.json" ]] || fail "checkpoint step ${step} receipt missing"
done

"${python_bin}" -B "${html_builder}" \
  --manifest "${review_manifest}" \
  --expected-manifest-sha256 "${review_manifest_sha}" \
  --shard-root "${shard_root}" --output-dir "${html_output}" \
  >"${run_root}/logs/checkpoint-review/html-builder.log" 2>&1
[[ -f "${html_output}/index.html" && -f "${html_output}/evidence.json" ]] || fail "self-contained checkpoint HTML is incomplete"
printf 'DECODE_REVIEW_COMPLETE_CSVC_STAGE_B holder=%s node=%s arm=%s checkpoints=0,20,40,60,80 html=%s parent_retained=true\n' \
  "${holder_job}" "${holder_node}" "${memory_input_kind}" "${html_output}" \
  >"${run_root}/controller.DECODE_REVIEW_COMPLETE"
printf 'decoded_checkpoint_inference_executed=true\nhtml_review_generated=true\nreview_complete=true\nparent_not_released=true\n' \
  >>"${run_root}/controller.status"
echo "DECODE_REVIEW_COMPLETE_CSVC_STAGE_B output=${html_output} parent_retained=true"
