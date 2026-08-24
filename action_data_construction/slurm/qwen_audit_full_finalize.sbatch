#!/usr/bin/env bash
#SBATCH --job-name=mev-action-finalize
#SBATCH --account=test-acc
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/logs/%x-%j.out
#SBATCH --error=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/logs/%x-%j.err

set -Eeuo pipefail
umask 077

project=/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction
source_root=/vast/users/guangyi.chen/dataset/MEV/MEV
python_bin=/vast/users/guangyi.chen/anaconda3/envs/qwen/bin/python3.12
input=${project}/metadata/qwen_audit_queue.jsonl
output_root=${MEV_AUDIT_OUTPUT_ROOT:?export the shared full-run output root}

fail() { printf '[mev-action-finalize] %s\n' "$*" >&2; exit 2; }
[[ "${output_root}" == "${project}"/runs/full_* ]] || fail 'output root must be project/runs/full_*'
[[ "${output_root}" != "${source_root}"* ]] || fail 'output overlaps protected source'
[[ -x "${python_bin}" && -f "${input}" ]] || fail 'runtime or queue unavailable'
rows=$(wc -l <"${input}")
[[ "${rows}" == 13093 ]] || fail "frozen full queue must contain 13093 rows, observed ${rows}"

for base in 0 2 4 6; do
  [[ -f "${output_root}/control/shard_${base}_contract.json" ]] || fail "missing shard ${base} contract"
  [[ -f "${output_root}/control/shard_${base}_DONE.json" ]] || fail "missing shard ${base} completion receipt"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${project}"

"${python_bin}" - "${output_root}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
contracts = [json.loads((root / "control" / f"shard_{base}_contract.json").read_text()) for base in (0, 2, 4, 6)]
for field in ("input", "input_sha256", "rows", "model", "num_workers", "audit_schema", "code_sha256"):
    values = {json.dumps(contract[field], sort_keys=True) for contract in contracts}
    if len(values) != 1:
        raise SystemExit(f"shard contracts disagree on {field}")
if sorted(worker for contract in contracts for worker in contract["worker_indices"]) != list(range(8)):
    raise SystemExit("shard contracts do not cover worker indices 0..7 exactly once")
PY

"${python_bin}" "${project}/audit_pairs_qwen.py" verify \
  --input "${input}" --output-root "${output_root}" --min-ok 0 \
  --output "${output_root}/control/verification.json"

# Source integrity is checked before any final metadata is released.
"${python_bin}" "${project}/verify_source_unchanged.py" \
  --build-summary "${project}/metadata/build_summary.json" \
  --events "${source_root}/metadata/events.csv" \
  >"${output_root}/control/source_postflight.json"

"${python_bin}" "${project}/finalize_metadata.py" \
  --queue "${input}" --audit-root "${output_root}" \
  --output-root "${output_root}/final_metadata"

printf '{"schema_version":"mev-action-edit-qwen-done-v1","phase":"full","rows":%s,"slurm_job_id":%s,"source_unchanged":true}\n' \
  "${rows}" "${SLURM_JOB_ID}" >"${output_root}/DONE.json"
