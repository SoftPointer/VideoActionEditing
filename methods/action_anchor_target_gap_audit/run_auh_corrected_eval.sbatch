#!/usr/bin/env bash
#SBATCH --job-name=mev-gap16-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00

set -Eeuo pipefail
umask 077

repo_root="${ACTION_GAP_REPO_ROOT:?set ACTION_GAP_REPO_ROOT}"
manifest="${ACTION_GAP_MANIFEST:?set ACTION_GAP_MANIFEST}"
experiment_root="${ACTION_GAP_V2_ROOT:?set ACTION_GAP_V2_ROOT}"
mode="${ACTION_GAP_V2_MODE:-full}"
smoke_prefix="${ACTION_GAP_V2_SMOKE_PREFIX:-81533c9e56ec}"
smoke_role="${ACTION_GAP_V2_SMOKE_ROLE:-}"
shard_index="${ACTION_GAP_V2_SHARD_INDEX:-}"
num_shards="${ACTION_GAP_V2_NUM_SHARDS:-2}"
repair_prefixes="${ACTION_GAP_V2_REPAIR_PREFIXES:-}"
rescore_prefixes="${ACTION_GAP_V2_RESCORE_PREFIXES:-}"
python_qwen="${ACTION_GAP_QWEN_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12}"
python_intern="${ACTION_GAP_INTERNVIDEO_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
internvideo_extra_site="${ACTION_GAP_INTERNVIDEO_EXTRA_SITE:-${experiment_root}/vendor/transformers-4.51.3}"
qwen_model="${ACTION_GAP_QWEN_MODEL:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct}"
internvideo_model="${ACTION_GAP_INTERNVIDEO_MODEL:-${experiment_root}/vendor/InternVideo2_CLIP_S-1f9fca1}"
ffmpeg_bin="${ACTION_GAP_FFMPEG:-/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}"
module="methods.action_anchor_target_gap_audit.corrected_eval"
contracts="${repo_root}/methods/action_anchor_target_gap_audit/manual_action_contracts_v2.json"
controls_root="${experiment_root}/controls"
controls="${controls_root}/control_manifest.json"

fail() { echo "[action-gap-v2] ERROR: $*" >&2; exit 2; }
[[ "${mode}" == "smoke" || "${mode}" == "intern-smoke" || "${mode}" == "intern-full" || "${mode}" == "qwen-shard" || "${mode}" == "qwen-repair" || "${mode}" == "qwen-rescore" || "${mode}" == "finalize" || "${mode}" == "full" ]] || fail "unsupported mode ${mode}"
for file in "${manifest}" "${contracts}"; do [[ -f "${file}" && ! -L "${file}" ]] || fail "missing plain input ${file}"; done
for directory in "${repo_root}" "${qwen_model}" "${internvideo_model}"; do [[ -d "${directory}" && ! -L "${directory}" ]] || fail "missing plain directory ${directory}"; done
for executable in "${python_qwen}" "${python_intern}" "${ffmpeg_bin}"; do [[ -x "${executable}" ]] || fail "missing executable ${executable}"; done
[[ "${experiment_root}" == /* && "${experiment_root}" != "/" ]] || fail "unsafe experiment root"
[[ "${experiment_root}" != /vast/users/guangyi.chen/dataset/MEV/MEV* ]] || fail "protected MEV output refused"

mkdir -p "${experiment_root}/logs" "${experiment_root}/qwen_v2/full" "${experiment_root}/qwen_v2/smoke" "${experiment_root}/qwen_v2/repairs" "${experiment_root}/internvideo2"
export PYTHONPATH="${repo_root}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="/tmp/action-gap-v2-${SLURM_JOB_ID}/hf" TORCH_HOME="/tmp/action-gap-v2-${SLURM_JOB_ID}/torch"
export TMPDIR="/tmp/action-gap-v2-${SLURM_JOB_ID}/tmp" PYTHONHASHSEED=0
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

if [[ ! -f "${controls}" ]]; then
  "${python_qwen}" -B -m "${module}" build-controls \
    --manifest "${manifest}" --contracts "${contracts}" --ffmpeg "${ffmpeg_bin}" \
    --output-dir "${controls_root}" >"${experiment_root}/logs/build-controls.log" 2>&1
fi

if [[ "${mode}" == "intern-smoke" ]]; then
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 \
    "${python_intern}" -B -m "${module}" internvideo-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${internvideo_model}" --device cuda --extra-site-packages "${internvideo_extra_site}" --pair-prefix "${smoke_prefix}" \
      --output "${experiment_root}/internvideo2/smoke-${smoke_prefix}.json" \
      >"${experiment_root}/logs/internvideo-smoke-${smoke_prefix}.log" 2>&1
  echo "[action-gap-v2] INTERNVIDEO SMOKE COMPLETE prefix=${smoke_prefix} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

if [[ "${mode}" == "intern-full" ]]; then
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 \
    "${python_intern}" -B -m "${module}" internvideo-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${internvideo_model}" --device cuda --extra-site-packages "${internvideo_extra_site}" \
      --output "${experiment_root}/internvideo2_summary.json" \
      >"${experiment_root}/logs/internvideo-v2.log" 2>&1
  echo "[action-gap-v2] INTERNVIDEO FULL COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

if [[ "${mode}" == "qwen-shard" ]]; then
  [[ "${num_shards}" =~ ^[1-9][0-9]*$ ]] || fail "qwen-shard requires a positive shard count"
  [[ "${shard_index}" =~ ^[0-9]+$ ]] || fail "qwen-shard requires a nonnegative shard index"
  (( shard_index < num_shards )) || fail "shard index must be smaller than shard count"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${python_qwen}" -B -m "${module}" qwen-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${qwen_model}" --shard-index "${shard_index}" --num-shards "${num_shards}" --num-frames 32 \
      --resume \
      --output "${experiment_root}/qwen_v2/full/qwen-v2-shard-${shard_index}.jsonl" \
      >"${experiment_root}/logs/qwen-v2-worker-${shard_index}.log" 2>&1
  echo "[action-gap-v2] QWEN SHARD COMPLETE shard=${shard_index}/${num_shards} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

if [[ "${mode}" == "qwen-repair" ]]; then
  [[ -n "${repair_prefixes}" ]] || fail "qwen-repair requires comma-separated repair prefixes"
  IFS=',' read -r -a repair_array <<<"${repair_prefixes}"
  pair_args=()
  for prefix in "${repair_array[@]}"; do
    [[ "${prefix}" =~ ^[0-9a-f]{12}$ ]] || fail "invalid repair prefix ${prefix}"
    pair_args+=(--pair-prefix "${prefix}")
  done
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${python_qwen}" -B -m "${module}" qwen-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${qwen_model}" --shard-index 0 --num-shards 1 "${pair_args[@]}" --num-frames 32 \
      --output "${experiment_root}/qwen_v2/repairs/qwen-v2-repair-${SLURM_JOB_ID}.jsonl" \
      >"${experiment_root}/logs/qwen-v2-repair-${SLURM_JOB_ID}.log" 2>&1
  echo "[action-gap-v2] QWEN REPAIR COMPLETE prefixes=${repair_prefixes} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

if [[ "${mode}" == "qwen-rescore" ]]; then
  [[ -n "${rescore_prefixes}" ]] || fail "qwen-rescore requires comma-separated rescore prefixes"
  IFS=',' read -r -a rescore_array <<<"${rescore_prefixes}"
  pair_args=()
  for prefix in "${rescore_array[@]}"; do
    [[ "${prefix}" =~ ^[0-9a-f]{12}$ ]] || fail "invalid rescore prefix ${prefix}"
    pair_args+=(--pair-prefix "${prefix}")
  done
  record_args=()
  for record_file in "${experiment_root}"/qwen_v2/full/qwen-v2-shard-*.jsonl; do
    [[ -f "${record_file}" ]] || fail "missing full shard record ${record_file}"
    record_args+=(--records "${record_file}")
  done
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${python_qwen}" -B -m "${module}" qwen-rescore-contract \
      --contracts "${contracts}" --model "${qwen_model}" \
      "${record_args[@]}" "${pair_args[@]}" \
      --output-dir "${experiment_root}/qwen_v2/repairs/rescore-${SLURM_JOB_ID}" \
      >"${experiment_root}/logs/qwen-v2-rescore-${SLURM_JOB_ID}.log" 2>&1
  echo "[action-gap-v2] QWEN RESCORE COMPLETE prefixes=${rescore_prefixes} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

finalize_outputs() {
  "${python_qwen}" -B -m "${module}" qwen-summarize \
    --contracts "${contracts}" --records-dir "${experiment_root}/qwen_v2/full" \
    --output "${experiment_root}/qwen_v2_summary.json"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 \
    "${python_intern}" -B -m "${module}" internvideo-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${internvideo_model}" --device cuda --extra-site-packages "${internvideo_extra_site}" \
      --output "${experiment_root}/internvideo2_summary.json" \
      >"${experiment_root}/logs/internvideo-v2.log" 2>&1
  sha256sum "${manifest}" "${contracts}" "${controls}" \
    "${experiment_root}/qwen_v2_summary.json" "${experiment_root}/internvideo2_summary.json" \
    >"${experiment_root}/SHA256SUMS.v2"
}

if [[ "${mode}" == "finalize" ]]; then
  finalize_outputs
  echo "[action-gap-v2] FINALIZE COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

if [[ "${mode}" == "smoke" ]]; then
  role_args=()
  if [[ -n "${smoke_role}" ]]; then role_args=(--role "${smoke_role}"); fi
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0,1,2,3 \
    "${python_qwen}" -B -m "${module}" qwen-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${qwen_model}" --shard-index 0 --num-shards 1 --pair-prefix "${smoke_prefix}" \
      "${role_args[@]}" \
      --num-frames 32 --output "${experiment_root}/qwen_v2/smoke/qwen-v2-shard-smoke-${smoke_prefix}.jsonl" \
      >"${experiment_root}/logs/qwen-smoke-${smoke_prefix}.log" 2>&1
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 \
    "${python_intern}" -B -m "${module}" internvideo-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${internvideo_model}" --device cuda --extra-site-packages "${internvideo_extra_site}" --pair-prefix "${smoke_prefix}" \
      --output "${experiment_root}/internvideo2/smoke-${smoke_prefix}.json" \
      >"${experiment_root}/logs/internvideo-smoke-${smoke_prefix}.log" 2>&1
  echo "[action-gap-v2] SMOKE COMPLETE prefix=${smoke_prefix} job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
  exit 0
fi

run_qwen_worker() {
  local index="$1" devices="$2"
  env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES="${devices}" \
    "${python_qwen}" -B -m "${module}" qwen-evaluate \
      --manifest "${manifest}" --contracts "${contracts}" --controls "${controls}" \
      --model "${qwen_model}" --shard-index "${index}" --num-shards 2 --num-frames 32 \
      --output "${experiment_root}/qwen_v2/full/qwen-v2-shard-${index}.jsonl"
}

run_qwen_worker 0 0,1,2,3 >"${experiment_root}/logs/qwen-v2-worker-0.log" 2>&1 & pid_a=$!
run_qwen_worker 1 4,5,6,7 >"${experiment_root}/logs/qwen-v2-worker-1.log" 2>&1 & pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
(( status == 0 )) || fail "Qwen v2 evaluation failed"

finalize_outputs
echo "[action-gap-v2] FULL COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
