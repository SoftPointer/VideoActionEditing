#!/usr/bin/env bash
#SBATCH --job-name=mev-action-shard
#SBATCH --account=test-acc
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
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
output_root=${MEV_AUDIT_OUTPUT_ROOT:?export the shared fresh full-run output root}
worker_base=${MEV_WORKER_BASE:?export one of 0, 2, 4, 6}
workers=8

fail() { printf '[mev-action-shard] %s\n' "$*" >&2; exit 2; }
[[ "${output_root}" == "${project}"/runs/full_* ]] || fail 'output root must be project/runs/full_*'
[[ "${output_root}" != "${source_root}"* ]] || fail 'output overlaps protected source'
[[ "${worker_base}" =~ ^(0|2|4|6)$ ]] || fail 'worker base must be one of 0, 2, 4, 6'
[[ "${SLURM_JOB_NUM_NODES:-}" == 1 ]] || fail 'shard job must own exactly one node'
[[ -x "${python_bin}" && -f "${model}/config.json" ]] || fail 'Qwen runtime is unavailable'
[[ -f "${input}" && ! -L "${input}" ]] || fail 'full input is absent or unsafe'
rows=$(wc -l <"${input}")
[[ "${rows}" == 13093 ]] || fail "frozen full queue must contain 13093 rows, observed ${rows}"

mkdir -p "${output_root}" "${output_root}/control"
contract=${output_root}/control/shard_${worker_base}_contract.json
"${python_bin}" - "${contract}" "${input}" "${rows}" "${model}" "${SLURM_JOB_ID}" "${worker_base}" "${project}" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path

path, input_s, rows, model, job_id, worker_base, project_s = sys.argv[1:]
input_p = Path(input_s).resolve(strict=True)
project = Path(project_s).resolve(strict=True)
code_sha256 = {}
for name in ("audit_pairs_qwen.py", "common.py", "docs/QWEN_AUDIT_POLICY.md"):
    code_sha256[name] = hashlib.sha256((project / name).read_bytes()).hexdigest()
invariants = {
    "schema_version": "mev-action-edit-qwen-full-shard-contract-v1",
    "phase": "full",
    "input": str(input_p),
    "input_sha256": hashlib.sha256(input_p.read_bytes()).hexdigest(),
    "rows": int(rows),
    "model": str(Path(model).resolve(strict=True)),
    "worker_base": int(worker_base),
    "worker_indices": [int(worker_base), int(worker_base) + 1],
    "num_workers": 8,
    "topology": "two_persistent_workers_x_four_mi210",
    "audit_schema": "mev-action-edit-pair-audit-v5",
    "code_sha256": code_sha256,
    "source_root_read_only": True,
    "video_copying_performed": False,
}
target = Path(path)
if target.exists():
    prior = json.loads(target.read_text(encoding="utf-8"))
    for key, expected in invariants.items():
        if prior.get(key) != expected:
            raise SystemExit(f"existing shard contract mismatch for {key}")
    raise SystemExit(0)
value = dict(invariants)
value["initial_slurm_job_id"] = int(job_id)
value["contract_digest"] = hashlib.sha256(
    json.dumps(invariants, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = handle.name
try:
    os.link(temporary, target)
finally:
    os.unlink(temporary)
PY

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
  >"${output_root}/control/source_preflight_shard_${worker_base}_job_${SLURM_JOB_ID}.json"

srun --overlap --exact --nodes=1 --ntasks=2 --ntasks-per-node=2 \
  --cpus-per-task=28 --mem=0 --gpus-per-task=4 --gpu-bind=none \
  --kill-on-bad-exit=1 /bin/bash --noprofile --norc -c '
    set -Eeuo pipefail
    project=$1; python_bin=$2; input=$3; output_root=$4; model=$5; workers=$6; worker_base=$7
    local_worker=${SLURM_LOCALID}
    worker=$((worker_base + local_worker))
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
  ' mev-full-shard "${project}" "${python_bin}" "${input}" "${output_root}" "${model}" "${workers}" "${worker_base}"

printf '{"schema_version":"mev-action-edit-qwen-shard-done-v1","worker_base":%s,"worker_indices":[%s,%s],"slurm_job_id":%s}\n' \
  "${worker_base}" "${worker_base}" "$((worker_base + 1))" "${SLURM_JOB_ID}" \
  >"${output_root}/control/shard_${worker_base}_DONE.json"
