#!/usr/bin/env bash
#SBATCH --job-name=mev-action-full
#SBATCH --account=test-acc
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:mi210:8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/logs/%x-%j.err

set -Eeuo pipefail
umask 077

project=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction
source_root=/vast/users/guangyi.chen/dataset/MEV/MEV
model=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct
python_bin=/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12
input=${project}/metadata/qwen_audit_queue.jsonl
output_root=${MEV_AUDIT_OUTPUT_ROOT:?export a fresh absolute MEV_AUDIT_OUTPUT_ROOT}
workers=8

fail() { printf '[mev-action-full] %s\n' "$*" >&2; exit 2; }
[[ "${output_root}" == "${project}"/runs/full_* ]] || fail 'output root must be project/runs/full_*'
[[ "${output_root}" != "${source_root}"* ]] || fail 'output overlaps protected source'
[[ "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || fail 'full job must own exactly four nodes'
[[ -x "${python_bin}" && -f "${model}/config.json" ]] || fail 'Qwen runtime is unavailable'
[[ -f "${input}" && ! -L "${input}" ]] || fail 'full input is absent or unsafe'
rows=$(wc -l <"${input}")
[[ "${rows}" == 13093 ]] || fail "frozen full queue must contain 13093 rows, observed ${rows}"

mkdir -p "${output_root}" "${output_root}/control"
contract=${output_root}/control/run_contract.json
if [[ ! -e "${contract}" ]]; then
  "${python_bin}" - "${contract}" "${input}" "${rows}" "${model}" "${SLURM_JOB_ID}" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path
path, input_s, rows, model, job_id = sys.argv[1:]
input_p = Path(input_s).resolve(strict=True)
value = {
    "schema_version": "mev-action-edit-qwen-full-run-contract-v1",
    "phase": "full",
    "input": str(input_p),
    "input_sha256": hashlib.sha256(input_p.read_bytes()).hexdigest(),
    "rows": int(rows),
    "model": str(Path(model).resolve(strict=True)),
    "slurm_job_id": int(job_id),
    "nodes": 4,
    "gpus": "32xMI210",
    "topology": "eight_persistent_workers_x_four_mi210",
    "audit_schema": "mev-action-edit-pair-audit-v5",
    "source_root_read_only": True,
    "video_copying_performed": False,
}
value["contract_digest"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
target = Path(path)
with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = handle.name
os.link(temporary, target)
os.unlink(temporary)
PY
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${project}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SLURM_EXPORT_ENV=ALL

"${python_bin}" "${project}/verify_source_unchanged.py" \
  --build-summary "${project}/metadata/build_summary.json" \
  --events "${source_root}/metadata/events.csv" \
  >"${output_root}/control/source_preflight.json"

srun --overlap --exact --nodes=4 --ntasks=8 --ntasks-per-node=2 \
  --cpus-per-task=28 --mem=0 --gpus-per-task=4 --gpu-bind=none \
  --kill-on-bad-exit=1 /bin/bash --noprofile --norc -c '
    set -Eeuo pipefail
    project=$1; python_bin=$2; input=$3; output_root=$4; model=$5; workers=$6
    worker=${SLURM_PROCID}
    local_worker=${SLURM_LOCALID}
    if (( local_worker == 0 )); then visible=0,1,2,3;
    elif (( local_worker == 1 )); then visible=4,5,6,7;
    else exit 91; fi
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="${visible}"
    cache=/tmp/mev-action-full-${SLURM_JOB_ID}-worker-${worker}
    mkdir -p "${cache}"/{tmp,xdg,torch,torch-kernels,miopen,miopen-custom}
    export TMPDIR="${cache}/tmp"
    export XDG_CACHE_HOME="${cache}/xdg"
    export TORCH_HOME="${cache}/torch"
    export PYTORCH_KERNEL_CACHE_PATH="${cache}/torch-kernels"
    export MIOPEN_USER_DB_PATH="${cache}/miopen"
    export MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom"
    exec "${python_bin}" "${project}/audit_pairs_qwen.py" run-worker \
      --input "${input}" --output-root "${output_root}" --model "${model}" \
      --worker-index "${worker}" --num-workers "${workers}" --nframes 8 \
      --tile-width 256 --mosaic-columns 4 --max-new-tokens 768 --attempts 2
  ' mev-full-worker "${project}" "${python_bin}" "${input}" "${output_root}" "${model}" "${workers}"

"${python_bin}" "${project}/audit_pairs_qwen.py" verify \
  --input "${input}" --output-root "${output_root}" --min-ok 0 \
  --output "${output_root}/control/verification.json"
"${python_bin}" "${project}/finalize_metadata.py" \
  --queue "${input}" --audit-root "${output_root}" \
  --output-root "${output_root}/final_metadata"
"${python_bin}" "${project}/verify_source_unchanged.py" \
  --build-summary "${project}/metadata/build_summary.json" \
  --events "${source_root}/metadata/events.csv" \
  >"${output_root}/control/source_postflight.json"

printf '{"schema_version":"mev-action-edit-qwen-done-v1","phase":"full","rows":%s,"slurm_job_id":%s}\n' \
  "${rows}" "${SLURM_JOB_ID}" >"${output_root}/DONE.json"
