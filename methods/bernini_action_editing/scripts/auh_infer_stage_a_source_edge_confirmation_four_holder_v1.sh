#!/usr/bin/env bash
# Four-source confirmation for exactly one externally authorized A1 cell.
#
# Each retained eight-GPU holder receives one WORLD4/SP4 child with 60 GiB.
# Four sentinels execute concurrently.  This controller never releases a
# parent allocation; after all strict shard gates pass it builds the
# self-contained HTML review in the run root.
set -Eeuo pipefail
umask 077

fail() { echo "[stage-a-confirmation-v1] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly -a sentinel_ids=(
  animal-dog-pick
  human-runner-jump
  hand-object-blueprint-roll
  emitter-fireworks-explode
)
readonly -a parent_jobs=(136007 136008 136009 136010)
readonly -a parent_nodes=(
  auh7-1b-gpu-215
  auh7-1b-gpu-261
  auh7-1b-gpu-262
  auh7-1b-gpu-228
)

assert_parent() {
  local job="$1" node="$2" record
  [[ "${job}" =~ ^[0-9]+$ && "${node}" =~ ^auh7-1b-gpu-[0-9]+$ ]] || fail "parent identity format differs"
  record="$(scontrol show job -o "${job}")"
  [[ "${record}" == *"JobId=${job} "* && "${record}" == *"JobState=RUNNING"* && "${record}" == *"NodeList=${node}"* ]] || fail "parent ${job}/${node} is not the requested running holder"
  [[ "${record}" == *"UserId=guangyi.chen"* ]] || fail "parent ${job} owner differs"
  [[ "${record}" == *"TresPerNode=gres/gpu:mi210:8"* ]] || fail "parent ${job} is not an eight-MI210 holder"
  [[ -z "$(squeue -s -j "${job}" -h -o '%i' | awk '/[.][0-9]+$/{print}')" ]] || fail "parent ${job} already has a numbered child"
}

if [[ "${1:-}" == __exec ]]; then
  shift
  [[ $# == 1 ]] || fail "child usage: __exec SENTINEL_ID"
  readonly sentinel_id="$1"
  case " ${sentinel_ids[*]} " in *" ${sentinel_id} "*) ;; *) fail "unknown sentinel" ;; esac
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numbered Slurm child required"

  method_root="${STAGE_A_CONFIRMATION_METHOD_ROOT:?set sealed method root}"
  output_dir="${STAGE_A_CONFIRMATION_OUTPUT_DIR:?set fresh sentinel output}"
  manifest="${STAGE_A_CONFIRMATION_MANIFEST:?set confirmation manifest}"
  manifest_sha="${STAGE_A_CONFIRMATION_MANIFEST_SHA256:?set confirmation manifest SHA}"
  runtime_revision="${STAGE_A_CONFIRMATION_RUNTIME_REVISION:?set source revision}"
  bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
  veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
  checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
  checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
  python_bin="${STAGE_A_CONFIRMATION_PYTHON_BIN:?set Python}"
  master_port="${STAGE_A_CONFIRMATION_MASTER_PORT:?set master port}"

  for value in "${method_root}" "${output_dir}" "${manifest}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${python_bin}"; do
    [[ "${value}" == /* ]] || fail "all child paths must be absolute"
  done
  [[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
  [[ ! -e "${output_dir}" && ! -L "${output_dir}" && -d "$(dirname "${output_dir}")" ]] || fail "sentinel output must be fresh"
  [[ -f "${manifest}" && ! -L "${manifest}" && "$(sha256_file "${manifest}")" == "${manifest_sha}" ]] || fail "confirmation manifest bytes differ"
  [[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
  [[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ && "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "revision/SHA format differs"
  [[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"
  [[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"

  runtime_path="${method_root}/infer_schedule_block_source_edge_localization_v2.py"
  contract_path="${method_root}/stage_a_source_edge_confirmation_contract_v1.py"
  validator_path="${method_root}/tools/validate_stage_a_source_edge_confirmation_shard_v1.py"
  html_builder_path="${method_root}/tools/build_stage_a_source_edge_confirmation_review_html_v1.py"
  manifest_builder_path="${method_root}/tools/materialize_stage_a_source_edge_confirmation_manifest_v1.py"
  launcher_path="${method_root}/scripts/auh_infer_stage_a_source_edge_confirmation_four_holder_v1.sh"
  closure_files=(
    "${runtime_path}"
    "${contract_path}"
    "${validator_path}"
    "${html_builder_path}"
    "${manifest_builder_path}"
    "${method_root}/tools/build_schedule_block_source_edge_pilot_review_html_v2.py"
    "${method_root}/tools/build_schedule_block_source_edge_formal_review_html_v2.py"
    "${method_root}/schedule_block_source_edge_ablation_v2.py"
    "${method_root}/schedule_block_causal_policy_v1.py"
    "${method_root}/clean_source_visual_context_checkpoint_review_contract_v1.py"
    "${method_root}/clean_source_visual_context_adapter_v1.py"
    "${method_root}/clean_source_visual_context_training_v1.py"
    "${method_root}/source_kv_replay.py"
    "${method_root}/native_i_axis_guidance.py"
    "${method_root}/tri_branch_unipc.py"
    "${method_root}/infer_native_v_axis_exact81_probe_v1.py"
    "${method_root}/native_v_axis_guidance_v1.py"
    "${method_root}/infer_native_identity_generation_canary.py"
    "${method_root}/infer_orderless_source_frame_set_noise_canary.py"
    "${method_root}/orderless_source_frame_set_noise.py"
    "${method_root}/infer_lora.py"
    "${method_root}/train_lora.py"
    "${method_root}/infer_source_kv_carrier_oracle.py"
    "${method_root}/infer_source_value_residual_oracle.py"
    "${method_root}/inference_sigma_strata.py"
    "${method_root}/source_kv_route_batches.py"
    "${method_root}/source_value_residual.py"
    "${launcher_path}"
    "${method_root}/tests/test_schedule_block_source_edge_ablation_v2.py"
    "${method_root}/tests/test_infer_schedule_block_source_edge_localization_v2.py"
    "${method_root}/tests/test_stage_a_source_edge_confirmation_contract_v1.py"
    "${method_root}/tests/test_infer_stage_a_source_edge_confirmation_v1.py"
    "${method_root}/tests/test_build_stage_a_source_edge_confirmation_review_html_v1.py"
    "${method_root}/tests/test_auh_infer_stage_a_source_edge_confirmation_four_holder_v1.py"
  )
  for path in "${closure_files[@]}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || fail "closure file absent: ${path}"
  done
  launcher_sha="$(sha256_file "${launcher_path}")"
  closure_sha="$(sha256sum "${closure_files[@]}" | sha256sum | awk '{print $1}')"

  for test_file in \
    test_schedule_block_source_edge_ablation_v2.py \
    test_infer_schedule_block_source_edge_localization_v2.py \
    test_stage_a_source_edge_confirmation_contract_v1.py \
    test_infer_stage_a_source_edge_confirmation_v1.py \
    test_build_stage_a_source_edge_confirmation_review_html_v1.py \
    test_auh_infer_stage_a_source_edge_confirmation_four_holder_v1.py
  do
    [[ -f "${method_root}/tests/${test_file}" && ! -L "${method_root}/tests/${test_file}" ]] || fail "test absent: ${test_file}"
    PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
  done

  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0 NCCL_IB_DISABLE=1

  scratch_parent="${SLURM_TMPDIR:-/tmp}"
  [[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
  task_scratch="$(mktemp -d "${scratch_parent%/}/stage-a-confirm-${sentinel_id}-${SLURM_JOB_ID}.XXXXXX")"
  touch "${task_scratch}/renderer-load.lock"
  chmod 0600 "${task_scratch}/renderer-load.lock"
  export NATIVE_V_AXIS_LOAD_LOCK="${task_scratch}/renderer-load.lock"
  mkdir -p "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton" "${task_scratch}/xdg" "${task_scratch}/pycache" "${task_scratch}/tmp"
  export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${task_scratch}/triton"
  export XDG_CACHE_HOME="${task_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
  export TMPDIR="${task_scratch}/tmp"
  cleanup() {
    local status=$?
    trap - EXIT TERM INT
    case "${task_scratch:-}" in "${scratch_parent%/}/stage-a-confirm-${sentinel_id}-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
    if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
      chmod -R u+w -- "${task_scratch}"
      rm -rf -- "${task_scratch}"
    fi
    [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || status=2
    exit "${status}"
  }
  trap cleanup EXIT

  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --confirmation-manifest "${manifest}" \
    --expected-confirmation-manifest-sha256 "${manifest_sha}" \
    --sentinel-id "${sentinel_id}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --output-dir "${output_dir}" \
    --runtime-source-revision "${runtime_revision}" \
    --runtime-source-closure-sha256 "${closure_sha}" \
    --launcher-source-sha256 "${launcher_sha}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}"

  "${python_bin}" -B "${validator_path}" \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${manifest_sha}" \
    --sentinel-id "${sentinel_id}" \
    --output-dir "${output_dir}"
  echo "[stage-a-confirmation-v1] PASS sentinel=${sentinel_id} output=${output_dir}"
  exit 0
fi

[[ "${1:-}" == run && $# == 1 ]] || fail "usage: $0 run"
method_root="${STAGE_A_CONFIRMATION_METHOD_ROOT:?set sealed method root}"
run_root="${STAGE_A_CONFIRMATION_RUN_ROOT:?set fresh run root}"
manifest="${STAGE_A_CONFIRMATION_MANIFEST:?set confirmation manifest}"
manifest_sha="${STAGE_A_CONFIRMATION_MANIFEST_SHA256:?set confirmation manifest SHA}"
runtime_revision="${STAGE_A_CONFIRMATION_RUNTIME_REVISION:?set source revision}"
python_bin="${STAGE_A_CONFIRMATION_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
base_port="${STAGE_A_CONFIRMATION_BASE_PORT:-31840}"

for value in "${method_root}" "${run_root}" "${manifest}" "${python_bin}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}"; do
  [[ "${value}" == /* ]] || fail "all controller paths must be absolute"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ && "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "revision/SHA format differs"
[[ "${base_port}" =~ ^[0-9]+$ ]] && (( base_port >= 1024 && base_port <= 65532 )) || fail "base port differs"
[[ "$(sha256_file "${manifest}")" == "${manifest_sha}" ]] || fail "confirmation manifest bytes differ"
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"
PYTHONPATH="${method_root}" "${python_bin}" -B -c \
  'import sys; import stage_a_source_edge_confirmation_contract_v1 as c; c.load_manifest(sys.argv[1], expected_file_sha256=sys.argv[2], verify_files=True)' \
  "${manifest}" "${manifest_sha}"
for index in 0 1 2 3; do
  assert_parent "${parent_jobs[$index]}" "${parent_nodes[$index]}"
done
mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/outputs"

launch_child() {
  local index="$1" sentinel_id="${sentinel_ids[$1]}" job="${parent_jobs[$1]}" node="${parent_nodes[$1]}" port="$((base_port + $1))"
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --ntasks=1 \
    --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=32 --mem=60G --gres=gpu:mi210:4 \
    env STAGE_A_CONFIRMATION_METHOD_ROOT="${method_root}" \
      STAGE_A_CONFIRMATION_OUTPUT_DIR="${run_root}/outputs/${sentinel_id}" \
      STAGE_A_CONFIRMATION_MANIFEST="${manifest}" \
      STAGE_A_CONFIRMATION_MANIFEST_SHA256="${manifest_sha}" \
      STAGE_A_CONFIRMATION_RUNTIME_REVISION="${runtime_revision}" \
      STAGE_A_CONFIRMATION_PYTHON_BIN="${python_bin}" \
      STAGE_A_CONFIRMATION_MASTER_PORT="${port}" \
      BERNINI_OFFICIAL_ROOT="${bernini_root}" \
      BERNINI_VEOMNI_ROOT="${veomni_root}" \
      BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
      BERNINI_CHECKPOINT_CONTENT_MANIFEST="${checkpoint_manifest}" \
      bash "${method_root}/scripts/auh_infer_stage_a_source_edge_confirmation_four_holder_v1.sh" __exec "${sentinel_id}"
}

declare -a child_pids=() child_status=(0 0 0 0)
set +e
for index in 0 1 2 3; do
  launch_child "${index}" >"${run_root}/logs/${sentinel_ids[$index]}.log" 2>&1 &
  child_pids[$index]=$!
done
for index in 0 1 2 3; do
  wait "${child_pids[$index]}"
  child_status[$index]=$?
done
set -e
for index in 0 1 2 3; do
  printf '%s_exit=%s parent_job=%s parent_node=%s\n' "${sentinel_ids[$index]}" "${child_status[$index]}" "${parent_jobs[$index]}" "${parent_nodes[$index]}" >>"${run_root}/controller.status"
  if (( child_status[$index] != 0 )); then
    tail -n 200 "${run_root}/logs/${sentinel_ids[$index]}.log" >&2 || true
    fail "one or more confirmation children failed; all parent holders remain retained"
  fi
done
printf 'parents_retained=true\nstage_b_admission=false\n' >>"${run_root}/controller.status"

"${python_bin}" -B "${method_root}/tools/build_stage_a_source_edge_confirmation_review_html_v1.py" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha}" \
  --run-root "${run_root}" \
  --output-dir "${run_root}/review"
printf 'COMPLETE parents_retained=true sentinels=4 outputs=56 stage_b_admission=false review=%s\n' "${run_root}/review/index.html" >"${run_root}/controller.COMPLETE"
echo "[stage-a-confirmation-v1] COMPLETE outputs=56 parents_retained=true review=${run_root}/review/index.html"
