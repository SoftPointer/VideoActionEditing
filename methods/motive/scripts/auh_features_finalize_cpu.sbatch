#!/usr/bin/env bash
#SBATCH --job-name=motive-features-merge
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 for every job.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
task_count="${MOTIVE_FEATURE_ARRAY_COUNT:-8}"
task_base="${output_root}/features_array/tasks"
merged_root="${output_root}/features_merged"
merged_feature_dir="${merged_root}/features"
candidate_output="${output_root}/feature_candidates"
array_manifest_dir="${output_root}/features_array/manifests"

if ! [[ "${task_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[motive-features-merge] invalid task count: ${task_count}" >&2
  exit 2
fi
for required in \
  "${code_root}/motive/cascade.py" \
  "${task_base}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-features-merge] missing required path: ${required}" >&2
    exit 2
  fi
done

link_verified() {
  local source_path="$1"
  local destination_path="$2"
  if [[ -e "${destination_path}" || -L "${destination_path}" ]]; then
    if [[ "$(readlink -f "${destination_path}")" != "$(readlink -f "${source_path}")" ]]; then
      echo "[motive-features-merge] conflicting destination: ${destination_path}" >&2
      exit 2
    fi
    return
  fi
  ln -s "${source_path}" "${destination_path}"
}

mkdir -p "${merged_feature_dir}"
first_source_marker="${array_manifest_dir}/task-000.jsonl.source"
if [[ ! -s "${first_source_marker}" ]]; then
  echo "[motive-features-merge] missing source marker: ${first_source_marker}" >&2
  exit 2
fi
IFS='|' read -r source_digest marker_count marker_index < "${first_source_marker}"
if [[ "${marker_count}" != "${task_count}" || "${marker_index}" != "0" ]]; then
  echo "[motive-features-merge] task-000 source marker is incompatible" >&2
  exit 2
fi

feature_contract="$(
  "${python_bin}" - \
    "${task_base}" \
    "${array_manifest_dir}" \
    "${task_count}" \
    "${source_digest}" \
    "${MOTIVE_FEATURE_ALLOW_ERRORS:-0}" <<'PY'
import json
import sys
from pathlib import Path

task_base = Path(sys.argv[1])
manifest_dir = Path(sys.argv[2])
task_count = int(sys.argv[3])
source_digest = sys.argv[4]
allow_errors = sys.argv[5] == "1"
contracts = set()
total_errors = 0
for index in range(task_count):
    tag = f"{index:03d}"
    summary_path = task_base / f"task-{tag}" / "stage" / "feature_summary.json"
    marker_path = manifest_dir / f"task-{tag}.jsonl.source"
    if not summary_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"task {tag} summary/source marker is missing")
    expected_marker = f"{source_digest}|{task_count}|{index}"
    if marker_path.read_text(encoding="utf-8").strip() != expected_marker:
        raise RuntimeError(f"task {tag} source provenance changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = (
        str(summary.get("stage_version") or ""),
        str(summary.get("config_digest") or ""),
        str(summary.get("implementation_digest") or ""),
    )
    if any(not value for value in contract):
        raise RuntimeError(f"task {tag} lacks feature contract provenance")
    contracts.add(contract)
    total_errors += int(summary["errors"])
if len(contracts) != 1:
    raise RuntimeError(f"feature tasks have mixed contracts: {sorted(contracts)}")
if total_errors and not allow_errors:
    raise RuntimeError(
        f"feature tasks contain {total_errors} errors; retry or set "
        "MOTIVE_FEATURE_ALLOW_ERRORS=1"
    )
stage_version, config_digest, implementation_digest = contracts.pop()
print(
    "|".join(
        (stage_version, config_digest, implementation_digest, str(total_errors))
    )
)
PY
)"
IFS='|' read -r feature_stage_version feature_config_digest \
  feature_implementation_digest feature_total_errors <<< "${feature_contract}"

merge_marker="${merged_root}/array.source"
expected_merge_marker="${source_digest}|${task_count}|${feature_stage_version}|${feature_config_digest}|${feature_implementation_digest}"
if [[ -s "${merge_marker}" ]]; then
  if [[ "$(<"${merge_marker}")" != "${expected_merge_marker}" ]]; then
    echo "[motive-features-merge] merged feature provenance changed" >&2
    exit 2
  fi
elif find "${merged_feature_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "[motive-features-merge] unproven files exist in ${merged_feature_dir}" >&2
  exit 2
else
  printf '%s\n' "${expected_merge_marker}" > "${merge_marker}"
fi

for ((task_index = 0; task_index < task_count; task_index++)); do
  task_tag="$(printf '%03d' "${task_index}")"
  stage_root="${task_base}/task-${task_tag}/stage"
  feature_dir="${stage_root}/features"
  summary_path="${stage_root}/feature_summary.json"
  task_source_marker="${array_manifest_dir}/task-${task_tag}.jsonl.source"
  if [[ ! -s "${summary_path}" || ! -d "${feature_dir}" ]]; then
    echo "[motive-features-merge] task ${task_tag} is incomplete" >&2
    exit 2
  fi
  if [[ ! -s "${task_source_marker}" ]] \
    || [[ "$(<"${task_source_marker}")" != "${source_digest}|${task_count}|${task_index}" ]]; then
    echo "[motive-features-merge] task ${task_tag} source provenance changed" >&2
    exit 2
  fi
  task_errors="$(
    "${python_bin}" -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["errors"]))' \
      "${summary_path}"
  )"
  if [[ "${task_errors}" != "0" && "${MOTIVE_FEATURE_ALLOW_ERRORS:-0}" != "1" ]]; then
    echo "[motive-features-merge] task ${task_tag} contains ${task_errors} errors" >&2
    exit 2
  fi
  found=0
  for source_path in "${feature_dir}"/shard-*; do
    [[ -e "${source_path}" ]] || continue
    found=1
    source_name="$(basename "${source_path}")"
    source_suffix="${source_name#shard-}"
    destination="${merged_feature_dir}/shard-t${task_tag}-s${source_suffix}"
    link_verified "$(readlink -f "${source_path}")" "${destination}"
  done
  if [[ "${found}" != "1" ]]; then
    echo "[motive-features-merge] task ${task_tag} has no feature shards" >&2
    exit 2
  fi
done

"${python_bin}" - \
  "${merged_root}/provenance.json" \
  "${source_digest}" \
  "${task_count}" \
  "${feature_stage_version}" \
  "${feature_config_digest}" \
  "${feature_implementation_digest}" \
  "${feature_total_errors}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "source_manifest_sha256": sys.argv[2],
    "task_count": int(sys.argv[3]),
    "stage_version": sys.argv[4],
    "config_digest": sys.argv[5],
    "implementation_digest": sys.argv[6],
    "errors": int(sys.argv[7]),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY

cd "${repo_root}"
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
"${python_bin}" -m motive.cascade export \
  --feature-dir "${merged_feature_dir}" \
  --output-dir "${candidate_output}" \
  --keep-decisions auto_keep review \
  --qwen-budget "${MOTIVE_QWEN_BUDGET:-8000}" \
  --qwen-decisions auto_keep review \
  --qwen-auto-keep-fraction "${MOTIVE_QWEN_AUTO_KEEP_FRACTION:-0.20}"

echo "[motive-features-merge] tasks=${task_count}"
echo "[motive-features-merge] merged_features=${merged_feature_dir}"
echo "[motive-features-merge] qwen_input=${candidate_output}/qwen_queue.jsonl"
