#!/usr/bin/env bash
# Run one frozen-vs-SEER heldout pair as two-node static WORLD4 inference.

set -Eeuo pipefail
umask 077

job0="${BERNINI_HELDOUT_JOB0:?set first allocation}"
job1="${BERNINI_HELDOUT_JOB1:?set second allocation}"
node0="${BERNINI_HELDOUT_NODE0:?set first node}"
node1="${BERNINI_HELDOUT_NODE1:?set second node}"
iid="${BERNINI_HELDOUT_IID:?set heldout IID}"
source_archive="${BERNINI_HELDOUT_METHOD_SOURCE_ARCHIVE:?set method archive}"
source_archive_sha256="${BERNINI_HELDOUT_METHOD_SOURCE_ARCHIVE_SHA256:?set archive SHA}"
source_revision="${BERNINI_HELDOUT_METHOD_SOURCE_REVISION:?set source revision}"
multinode_runner="${BERNINI_HELDOUT_MULTINODE_RUNNER:?set frozen multinode runner}"
rank_cache_exec="${BERNINI_HELDOUT_RANK_CACHE_EXEC:?set frozen rank cache worker}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set base checkpoint}"
adapter_checkpoint="${BERNINI_ACTION_ADAPTER_CHECKPOINT:?set adapter checkpoint}"
output_root="${BERNINI_HELDOUT_OUTPUT_ROOT:?set fresh output root}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

readonly expected_runner_sha256="b58b173e07aa43a14dd6507dc9cf2685ef75f45bfe89cf1456a4dfdec6bb1467"
readonly expected_rank_cache_exec_sha256="f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5"
readonly expected_spec_sha256="82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c"

fail() { echo "[heldout-multinode] ERROR: $*" >&2; exit 2; }

for value in "${job0}" "${job1}"; do [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "job differs"; done
[[ "${job0}" != "${job1}" && "${node0}" != "${node1}" ]] || fail "two distinct allocations/nodes required"
case "${iid}" in
  99cde432839f4240|6ea45d35943742bb|311c82f83eca4a7f|6d346c38cf504493) ;;
  *) fail "IID differs" ;;
esac
for path in source_archive multinode_runner rank_cache_exec bernini_root veomni_root checkpoint adapter_checkpoint output_root python_bin; do
  [[ "${!path}" == /* && "${!path}" != / ]] || fail "${path} must be absolute"
done
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "archive differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive SHA differs"
[[ -f "${multinode_runner}" && ! -L "${multinode_runner}" ]] || fail "runner differs"
[[ "$(sha256sum "${multinode_runner}" | awk '{print $1}')" == "${expected_runner_sha256}" ]] || fail "runner SHA differs"
[[ -x "${rank_cache_exec}" && ! -L "${rank_cache_exec}" ]] || fail "rank cache worker differs"
[[ "$(sha256sum "${rank_cache_exec}" | awk '{print $1}')" == "${expected_rank_cache_exec_sha256}" ]] || fail "rank cache worker SHA differs"
[[ -x "${python_bin}" ]] || fail "Python differs"
[[ -d "${bernini_root}" && -d "${veomni_root}" && -d "${checkpoint}" && -d "${adapter_checkpoint}" ]] || fail "model roots differ"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output must be fresh"
[[ -d "${output_root%/*}" && -w "${output_root%/*}" ]] || fail "output parent differs"

check_allocation() {
  local job="$1" node="$2" record steps
  record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobState=RUNNING"* && "${record}" == *"NodeList=${node}"* && "${record}" == *"gres/gpu:mi210=8"* ]] || fail "allocation ${job}/${node} differs"
  steps="$(squeue -s -j "${job}" -h -o '%i' | grep -E '[.][0-9]+$' || true)"
  [[ -z "${steps}" ]] || fail "allocation ${job} has numbered step: ${steps}"
}
check_allocation "${job0}" "${node0}"
check_allocation "${job1}" "${node1}"

mkdir -m 0700 "${output_root}"
source_root="${output_root}/source"
results_root="${output_root}/results"
logs_root="${output_root}/logs"
mkdir -m 0700 "${source_root}" "${results_root}" "${logs_root}"
tar --delay-directory-restore --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
spec="${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json"
[[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${expected_spec_sha256}" ]] || fail "spec differs"
"${python_bin}" -B "${multinode_runner}" --spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}" inspect --verify-files

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1 SLURM_EXPORT_ENV=ALL

children=()
cleanup() {
  local pid
  for pid in "${children[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${children[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup INT TERM HUP EXIT

run_node() {
  local arm="$1" port="$2" job="$3" node="$4" node_rank="$5"
  local cache_token="${iid}-${arm}-n${node_rank}"
  local adaptation=()
  [[ "${arm}" == trained_adapter ]] && adaptation=(--adapter-checkpoint "${adapter_checkpoint}")
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --exclusive --exact \
    --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2 \
    env BERNINI_HELDOUT_RANK_CACHE_TOKEN="${cache_token}" BERNINI_HELDOUT_PYTHON_BIN="${python_bin}" \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      "${python_bin}" -B "${multinode_runner}" \
        --spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}" run-arm \
        --iid "${iid}" --arm "${arm}" --method-root "${method_root}" --python-bin "${python_bin}" \
        --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
        "${adaptation[@]}" --trained-infer-runner infer_seer_scoped_lora.py \
        --output-root "${results_root}" --master-port "${port}" \
        --torchrun-nnodes 2 --torchrun-nproc-per-node 2 --torchrun-node-rank "${node_rank}" \
        --torchrun-master-addr "${node0}" --torchrun-worker-prefix "${rank_cache_exec}" \
        --method-source-revision "${source_revision}" \
        --method-source-archive-sha256 "${source_archive_sha256}"
}

wait_pair() {
  local p0="$1" p1="$2" rc0=0 rc1=0 done0=0 done1=0
  while (( done0 == 0 || done1 == 0 )); do
    if (( done0 == 0 )) && ! kill -0 "${p0}" 2>/dev/null; then wait "${p0}" || rc0=$?; done0=1; fi
    if (( done1 == 0 )) && ! kill -0 "${p1}" 2>/dev/null; then wait "${p1}" || rc1=$?; done1=1; fi
    if (( rc0 != 0 || rc1 != 0 )); then
      (( done0 == 0 )) && kill "${p0}" 2>/dev/null || true
      (( done1 == 0 )) && kill "${p1}" 2>/dev/null || true
    fi
    (( done0 == 0 || done1 == 0 )) && sleep 1
  done
  (( rc0 == 0 && rc1 == 0 )) || return 1
}

run_arm_pair() {
  local arm="$1" port="$2" p0 p1
  echo "[heldout-multinode] START iid=${iid} arm=${arm} world=2x2"
  run_node "${arm}" "${port}" "${job0}" "${node0}" 0 >"${logs_root}/${arm}-node0.log" 2>&1 & p0=$!
  run_node "${arm}" "${port}" "${job1}" "${node1}" 1 >"${logs_root}/${arm}-node1.log" 2>&1 & p1=$!
  children=("${p0}" "${p1}")
  if ! wait_pair "${p0}" "${p1}"; then
    tail -n 120 "${logs_root}/${arm}-node0.log" >&2 || true
    tail -n 120 "${logs_root}/${arm}-node1.log" >&2 || true
    fail "${arm} multinode inference failed"
  fi
  children=()
}

run_arm_pair frozen_base 29441
run_arm_pair trained_adapter 29442

verify_args=(--spec "${spec}" --expected-spec-sha256 "${expected_spec_sha256}" verify-pair --iid "${iid}" --adapter-checkpoint "${adapter_checkpoint}" --output-root "${results_root}")
command -v ffmpeg >/dev/null 2>&1 && verify_args+=(--ffmpeg "$(command -v ffmpeg)")
"${python_bin}" -B "${multinode_runner}" "${verify_args[@]}" >"${output_root}/paired-verification.json"
trap - INT TERM HUP EXIT
echo "[heldout-multinode] PASS_GENERATION_NOT_METHOD_SUCCESS iid=${iid}"
