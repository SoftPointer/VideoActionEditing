#!/usr/bin/env bash
# Run one WORLD8 DP2/SP4 preservation-residual arm inside an existing holder.
# The numbered child step may finish; the parent allocation is never signalled,
# cancelled, released, requeued, or otherwise modified by this controller.

set -Eeuo pipefail
umask 077

fail() { echo "[preservation-residual-single-holder] ERROR: $*" >&2; exit 2; }

readonly holder_job="${PRESERVATION_HOLDER_JOB:?set holder job}"
readonly holder_node="${PRESERVATION_HOLDER_NODE:?set holder node}"
readonly adapter_rank="${PRESERVATION_ADAPTER_RANK:?set adapter rank}"
readonly optimizer_steps="${PRESERVATION_OPTIMIZER_STEPS:-40}"
readonly run_root="${PRESERVATION_RUN_ROOT:?set fresh run root}"
readonly method_root="${PRESERVATION_METHOD_ROOT:?set staged method root}"
readonly materialized="${PRESERVATION_MATERIALIZED_ROOT:?set sealed source-self dataset}"
readonly spec_sha="${PRESERVATION_SPEC_SHA256:?set source-self spec SHA}"
readonly method_revision="${PRESERVATION_METHOD_REVISION:?set git revision}"
readonly method_archive_sha="${PRESERVATION_METHOD_ARCHIVE_SHA256:?set staged source digest}"
readonly method_manifest_sha="${PRESERVATION_METHOD_MANIFEST_SHA256:?set staged manifest digest}"
readonly master_port="${PRESERVATION_MASTER_PORT:?set master port}"

readonly holder_user=guangyi.chen
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly expected_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly expected_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly expected_checkpoint_tree_sha=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca

case "${holder_job}:${holder_node}" in
  135407:auh7-1b-gpu-260|135411:auh7-1b-gpu-214) ;;
  *) fail "holder job/node is outside the two-arm allowlist" ;;
esac
case "${adapter_rank}" in 2|8) ;; *) fail "adapter rank must be 8 or 2" ;; esac
case "${optimizer_steps}" in 20|40) ;; *) fail "optimizer steps must be checkpoint 20 or final 40" ;; esac
[[ "${master_port}" =~ ^[1-9][0-9]*$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
[[ "${run_root}" == /vast/users/guangyi.chen/* && "${run_root}" != / ]] || fail "unsafe run root"
[[ "${method_root}" == /vast/users/guangyi.chen/* && -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
[[ -d "${materialized}" && ! -L "${materialized}" ]] || fail "materialized dataset differs"
[[ -x "${python_bin}" && -f "${method_root}/train_preservation_residual_v1.py" ]] || fail "runtime source differs"
for digest in spec_sha method_archive_sha method_manifest_sha; do [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"; done
[[ "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method revision differs"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh"

job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "holder ownership/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder already has a numbered child"

assert_idle() {
  local snapshot busy count
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
readonly rank_exec="${method_root}/scripts/auh_preservation_rank_cache_exec_v1.sh"
chmod 0500 "${rank_exec}"
checkpoint_args=()
if (( optimizer_steps == 40 )); then
  readonly checkpoint_output_root="${run_root}/checkpoints"
  [[ ! -e "${checkpoint_output_root}" && ! -L "${checkpoint_output_root}" ]] || fail "checkpoint root must be fresh"
  checkpoint_args=(--checkpoint-output-root "${checkpoint_output_root}")
fi

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=64 --mem=60G --gres=gpu:mi210:8 \
  env BERNINI_HELDOUT_RANK_CACHE_TOKEN="preservation-r${adapter_rank}-${method_revision:0:10}" \
    BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODELING_BACKEND=hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
    "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
      --master_addr=127.0.0.1 --master_port="${master_port}" --no_python "${rank_exec}" \
      "${method_root}/train_preservation_residual_v1.py" \
      --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
      --dataset-root "${materialized}" --expected-materialization-spec-sha256 "${spec_sha}" \
      --output "${training_output}" "${checkpoint_args[@]}" \
      --mode preservation-residual-v1 --parallel-topology world8-dp2-sp4 \
      --adapter-block-scope early-mid-0-22 --adapter-rank "${adapter_rank}" --optimizer-steps "${optimizer_steps}" \
      --expected-bernini-commit "${expected_bernini_commit}" --expected-veomni-commit "${expected_veomni_commit}" \
      --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha}" \
      --method-source-revision "${method_revision}" --method-source-revision-kind git-commit \
      --method-source-archive-sha256 "${method_archive_sha}" --method-source-manifest-sha256 "${method_manifest_sha}" \
      --ack-upstream-training-use-forbidden --ack-forward-noising-is-not-inversion \
  >"${run_root}/logs/train.log" 2>&1
status=$?
set -e

printf 'holder_job=%s\nholder_node=%s\nadapter_rank=%s\noptimizer_steps=%s\nchild_exit=%s\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${adapter_rank}" "${optimizer_steps}" "${status}" \
  >"${run_root}/controller.status"
if (( status != 0 )); then
  tail -n 160 "${run_root}/logs/train.log" >&2 || true
  exit "${status}"
fi
[[ -f "${training_output}/receipt.json" ]] || fail "training receipt missing"
if (( optimizer_steps == 40 )); then
  for step in 00000000 00000020; do
    checkpoint_bundle="${checkpoint_output_root}/step-${step}"
    [[ -d "${checkpoint_bundle}" && ! -L "${checkpoint_bundle}" ]] || fail "cadence checkpoint missing"
    for name in adapter.safetensors optimizer.pt history.json receipt.json; do
      [[ -f "${checkpoint_bundle}/${name}" && ! -L "${checkpoint_bundle}/${name}" ]] || fail "cadence checkpoint artifact missing"
    done
  done
  printf 'continuous_exact40=true\ncheckpoint_steps=0,20,40\n' >>"${run_root}/controller.status"
fi
if (( optimizer_steps == 20 )); then
  case "${adapter_rank}" in
    8) readonly expected_prefix_parameter_sha=20af97615bf51aba46c59795f21330a5563426826043faa8ad5626ad17c5f42a ;;
    2) readonly expected_prefix_parameter_sha=2a5f775212796fbe7836f206ef3a0e9f49dced7c544b08323cba599f6900ffc9 ;;
  esac
  observed_prefix_parameter_sha="$(
    "${python_bin}" -I -S -B -c \
      'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["final_adapter_sha256"])' \
      "${training_output}/receipt.json"
  )"
  [[ "${observed_prefix_parameter_sha}" == "${expected_prefix_parameter_sha}" ]] || \
    fail "exact20 parameter digest does not reproduce the historical exact40 prefix"
  printf 'historical_exact20_parameter_sha256=%s\nexact20_prefix_reproduced=true\n' \
    "${observed_prefix_parameter_sha}" >>"${run_root}/controller.status"
fi
printf 'COMPLETE_PRESERVATION_RESIDUAL holder=%s node=%s rank=%s steps=%s parent_retained=true\n' \
  "${holder_job}" "${holder_node}" "${adapter_rank}" "${optimizer_steps}" \
  >"${run_root}/controller.COMPLETE"
echo "COMPLETE_PRESERVATION_RESIDUAL output=${run_root}"
