#!/usr/bin/env bash
# Frozen Stage-A decoded target-query -> source-K/V localization.
#
# Controller mode uses two retained parent allocations but requests only one
# WORLD4/SP4 child in each: dog and human run in parallel.  It never invokes
# scancel, never releases either parent and writes COMPLETE only after both
# fail-closed child receipts pass.  The default calibration is the single
# preregistered C0 cell s16 x blocks0-7 (14 videos/family, 28 total).  Setting
# STAGE_A_SCHEDULE_INDICES / STAGE_A_BLOCK_BANDS creates an explicit shard;
# the Python receipt can never claim cells outside that shard.
set -Eeuo pipefail
umask 077

fail() { echo "[stage-a-source-edge-v2] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly authoring_sha256=204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c
readonly bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d

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
  [[ $# == 1 && ( "$1" == dog || "$1" == human ) ]] || fail "child usage: __exec dog|human"
  readonly family="$1"
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numbered Slurm child required"

  method_root="${STAGE_A_METHOD_ROOT:?set method root}"
  output_dir="${STAGE_A_OUTPUT_DIR:?set fresh family output}"
  runtime_revision="${STAGE_A_RUNTIME_REVISION:?set source revision}"
  schedules="${STAGE_A_SCHEDULE_INDICES:-16}"
  bands="${STAGE_A_BLOCK_BANDS:-early}"
  bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
  veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
  checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
  checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
  python_bin="${STAGE_A_PYTHON_BIN:?set Python}"
  master_port="${STAGE_A_MASTER_PORT:?set master port}"

  for value in "${method_root}" "${output_dir}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${python_bin}"; do
    [[ "${value}" == /* ]] || fail "all child paths must be absolute"
  done
  [[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
  [[ ! -e "${output_dir}" && ! -L "${output_dir}" && -d "$(dirname "${output_dir}")" ]] || fail "family output must be fresh"
  [[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
  [[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
  [[ "${master_port}" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || fail "master port differs"

  runtime_path="${method_root}/infer_schedule_block_source_edge_localization_v2.py"
  core_path="${method_root}/schedule_block_source_edge_ablation_v2.py"
  authoring_path="${method_root}/assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
  launcher_path="${method_root}/scripts/auh_infer_schedule_block_source_edge_two_holder_v2.sh"
  closure_files=(
    "${runtime_path}"
    "${core_path}"
    "${method_root}/schedule_block_causal_policy_v1.py"
    "${method_root}/source_kv_replay.py"
    "${method_root}/native_i_axis_guidance.py"
    "${method_root}/tri_branch_unipc.py"
    "${method_root}/infer_native_v_axis_exact81_probe_v1.py"
    "${method_root}/native_v_axis_guidance_v1.py"
    "${method_root}/infer_native_identity_generation_canary.py"
    "${method_root}/infer_orderless_source_frame_set_noise_canary.py"
    "${method_root}/infer_lora.py"
    "${method_root}/train_lora.py"
    "${method_root}/infer_source_kv_carrier_oracle.py"
    "${method_root}/infer_source_value_residual_oracle.py"
    "${method_root}/inference_sigma_strata.py"
    "${authoring_path}"
    "${launcher_path}"
    "${method_root}/tests/test_schedule_block_causal_policy_v1.py"
    "${method_root}/tests/test_schedule_block_source_edge_ablation_v2.py"
    "${method_root}/tests/test_infer_schedule_block_source_edge_localization_v2.py"
    "${method_root}/tests/test_auh_infer_schedule_block_source_edge_two_holder_v2.py"
  )
  for path in "${closure_files[@]}"; do
    [[ -f "${path}" && ! -L "${path}" ]] || fail "closure file absent: ${path}"
  done
  [[ "$(sha256_file "${authoring_path}")" == "${authoring_sha256}" ]] || fail "authoring authority differs"
  [[ "$(sha256_file "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"
  launcher_sha="$(sha256_file "${launcher_path}")"
  closure_sha="$(sha256sum "${closure_files[@]}" | sha256sum | awk '{print $1}')"

  for test_file in \
    test_schedule_block_causal_policy_v1.py \
    test_schedule_block_source_edge_ablation_v2.py \
    test_infer_schedule_block_source_edge_localization_v2.py \
    test_auh_infer_schedule_block_source_edge_two_holder_v2.py
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
  task_scratch="$(mktemp -d "${scratch_parent%/}/stage-a-edge-${family}-${SLURM_JOB_ID}.XXXXXX")"
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
    case "${task_scratch:-}" in "${scratch_parent%/}/stage-a-edge-${family}-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
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
    --authoring-spec "${authoring_path}" \
    --expected-authoring-spec-sha256 "${authoring_sha256}" \
    --family "${family}" \
    --schedule-indices "${schedules}" \
    --block-bands "${bands}" \
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

  "${python_bin}" -I -S -B - "${output_dir}/receipt.json" "${family}" "${schedules}" "${bands}" "${runtime_revision}" "${closure_sha}" "${launcher_sha}" <<'PY'
import hashlib, json
from pathlib import Path
import sys

path = Path(sys.argv[1])
family, schedules, bands, revision, closure_sha, launcher_sha = sys.argv[2:]
value = json.loads(path.read_text(encoding="ascii"))
unsigned = dict(value)
declared = unsigned.pop("receipt_digest", None)
raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
expected_count = 8 + len(schedules.split(",")) * len(bands.split(",")) * 6
interpretation = value.get("interpretation", {})
sampling = value.get("sampling", {})
shard = value.get("shard", {})
if not (
    hashlib.sha256(raw).hexdigest() == declared
    and value.get("schema_version") == "bernini-schedule-block-source-edge-decoded-runtime-v2"
    and shard.get("family") == family
    and shard.get("candidate_count") == expected_count
    and len(value.get("candidates", [])) == expected_count
    and len(value.get("outputs", {})) == expected_count
    and value.get("runtime_source", {}).get("revision") == revision
    and value.get("runtime_source", {}).get("closure_sha256") == closure_sha
    and value.get("runtime_source", {}).get("launcher_sha256") == launcher_sha
    and sampling.get("exact40") is True
    and sampling.get("exact81") is True
    and sampling.get("same_initial_gaussian_all_candidates") is True
    and sampling.get("source_on_native_parity_bit_exact") is True
    and interpretation.get("score_computed") is False
    and interpretation.get("reward_computed") is False
    and interpretation.get("ranking_performed") is False
    and interpretation.get("selection_performed") is False
    and interpretation.get("training_performed") is False
    and interpretation.get("optimizer_present") is False
    and interpretation.get("parameter_update") is False
):
    raise SystemExit("decoded Stage-A child receipt gate failed")
for artifact in value["outputs"].values():
    candidate = Path(artifact["path"])
    observed_sha = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None
    if not (
        candidate.is_file()
        and observed_sha == artifact.get("sha256")
        and artifact.get("frame_count") == 81
        and artifact.get("fps") == 25
    ):
        raise SystemExit("decoded Stage-A MP4 closure differs")
print(json.dumps({"passed": True, "family": family, "candidate_count": expected_count, "receipt_digest": declared}, sort_keys=True))
PY
  echo "[stage-a-source-edge-v2] PASS family=${family} schedules=${schedules} bands=${bands} output=${output_dir}"
  exit 0
fi

[[ "${1:-}" == run && $# == 1 ]] || fail "usage: $0 run"
method_root="${STAGE_A_METHOD_ROOT:?set sealed method root}"
run_root="${STAGE_A_RUN_ROOT:?set fresh run root}"
runtime_revision="${STAGE_A_RUNTIME_REVISION:?set full source revision}"
dog_job="${STAGE_A_DOG_JOB:?set retained dog job}"
dog_node="${STAGE_A_DOG_NODE:?set retained dog node}"
human_job="${STAGE_A_HUMAN_JOB:?set retained human job}"
human_node="${STAGE_A_HUMAN_NODE:?set retained human node}"
schedules="${STAGE_A_SCHEDULE_INDICES:-16}"
bands="${STAGE_A_BLOCK_BANDS:-early}"
python_bin="${STAGE_A_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
base_port="${STAGE_A_BASE_PORT:-31640}"

for value in "${method_root}" "${run_root}" "${python_bin}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}"; do
  [[ "${value}" == /* ]] || fail "all controller paths must be absolute"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ "${base_port}" =~ ^[0-9]+$ ]] && (( base_port >= 1024 && base_port <= 65534 )) || fail "base port differs"
[[ "${dog_job}" != "${human_job}" && "${dog_node}" != "${human_node}" ]] || fail "pilot requires two independent retained holders"
[[ "${run_root}" != / && "$(realpath -m -- "${run_root}")" == "${run_root}" && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root must be fresh canonical"
assert_parent "${dog_job}" "${dog_node}"
assert_parent "${human_job}" "${human_node}"
mkdir -m 0700 "${run_root}" "${run_root}/logs" "${run_root}/outputs"

launch_child() {
  local family="$1" job="$2" node="$3" port="$4"
  srun --jobid="${job}" --nodelist="${node}" --nodes=1 --ntasks=1 \
    --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=32 --mem=60G --gres=gpu:mi210:4 \
    env STAGE_A_METHOD_ROOT="${method_root}" \
      STAGE_A_OUTPUT_DIR="${run_root}/outputs/${family}" \
      STAGE_A_RUNTIME_REVISION="${runtime_revision}" \
      STAGE_A_SCHEDULE_INDICES="${schedules}" \
      STAGE_A_BLOCK_BANDS="${bands}" \
      STAGE_A_PYTHON_BIN="${python_bin}" \
      STAGE_A_MASTER_PORT="${port}" \
      BERNINI_OFFICIAL_ROOT="${bernini_root}" \
      BERNINI_VEOMNI_ROOT="${veomni_root}" \
      BERNINI_ACTION_CHECKPOINT="${checkpoint}" \
      BERNINI_CHECKPOINT_CONTENT_MANIFEST="${checkpoint_manifest}" \
      bash "${method_root}/scripts/auh_infer_schedule_block_source_edge_two_holder_v2.sh" __exec "${family}"
}

set +e
launch_child dog "${dog_job}" "${dog_node}" "${base_port}" >"${run_root}/logs/dog.log" 2>&1 &
dog_pid=$!
launch_child human "${human_job}" "${human_node}" "$((base_port + 1))" >"${run_root}/logs/human.log" 2>&1 &
human_pid=$!
wait "${dog_pid}"; dog_status=$?
wait "${human_pid}"; human_status=$?
set -e
printf 'dog_exit=%s\nhuman_exit=%s\ndog_parent=%s\nhuman_parent=%s\nparents_not_released=true\n' "${dog_status}" "${human_status}" "${dog_job}" "${human_job}" >"${run_root}/controller.status"
if (( dog_status != 0 || human_status != 0 )); then
  tail -n 160 "${run_root}/logs/dog.log" >&2 || true
  tail -n 160 "${run_root}/logs/human.log" >&2 || true
  fail "one or both Stage-A family children failed; parent holders were retained"
fi
candidate_count=$(( 2 * (8 + $(awk -F, '{print NF}' <<<"${schedules}") * $(awk -F, '{print NF}' <<<"${bands}") * 6) ))
printf 'COMPLETE dog_parent=%s human_parent=%s parents_retained=true candidates=%s schedules=%s bands=%s\n' "${dog_job}" "${human_job}" "${candidate_count}" "${schedules}" "${bands}" >"${run_root}/controller.COMPLETE"
echo "[stage-a-source-edge-v2] COMPLETE candidates=${candidate_count} parents_retained=true output=${run_root}"
