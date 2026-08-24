#!/usr/bin/env bash
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00
#SBATCH --job-name=motive-r10b-qwen-audit

set -euo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
queue_dir="${MOTIVE_R10B_QUEUE_DIR:?set MOTIVE_R10B_QUEUE_DIR}"
data_root="${MOTIVE_R10B_DATA_ROOT:?set MOTIVE_R10B_DATA_ROOT}"
qwen_model="${MOTIVE_R10B_QWEN_MODEL:?set MOTIVE_R10B_QWEN_MODEL}"
output_dir="${MOTIVE_R10B_OUTPUT_DIR:?set MOTIVE_R10B_OUTPUT_DIR}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:?set MOTIVE_R10B_PYTHON_BIN}"

nframes="12"
max_pixels="589824"
max_new_tokens="512"
attn_implementation="sdpa"
visual_input="mosaic"

require_absolute_path() {
  local label="$1"
  local path="$2"
  if [[ "${path}" != /* || "${path}" == "/" ]]; then
    echo "${label} must be a non-root absolute path: ${path}" >&2
    exit 2
  fi
}

require_sha256() {
  local label="$1"
  local digest="$2"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${label} must be one lowercase SHA-256" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "queue_dir:${queue_dir}" \
  "data_root:${data_root}" \
  "qwen_model:${qwen_model}" \
  "output_dir:${output_dir}" \
  "python_bin:${python_bin}"; do
  require_absolute_path "${binding%%:*}" "${binding#*:}"
done
require_sha256 "source_tree_sha256" "${source_tree_sha256}"

if [[ -L "${source_snapshot}" || ! -d "${source_snapshot}" ]]; then
  echo "source snapshot must be one non-symlink directory: ${source_snapshot}" >&2
  exit 2
fi
if [[ -L "${queue_dir}" || ! -d "${queue_dir}" ]]; then
  echo "queue must be one non-symlink directory: ${queue_dir}" >&2
  exit 2
fi
if [[ -L "${data_root}" || ! -d "${data_root}" ]]; then
  echo "data root must be one non-symlink directory: ${data_root}" >&2
  exit 2
fi
if [[ -L "${qwen_model}" || ! -d "${qwen_model}" ]]; then
  echo "Qwen model must be one non-symlink directory: ${qwen_model}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "refusing to reuse R10B Qwen audit output: ${output_dir}" >&2
  exit 2
fi
output_parent="${output_dir%/*}"
if [[ ! -d "${output_parent}" || ! -w "${output_parent}" ]]; then
  echo "output parent must be an existing writable directory: ${output_parent}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${source_snapshot}:${source_snapshot}/methods/motive"

task_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-qwen-audit-${SLURM_JOB_ID:-manual}"
mkdir -p \
  "${task_scratch}/hf" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom"
export HF_HOME="${task_scratch}/hf"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"

record_binding() {
  printf 'motive_r10b_qwen_audit.%s=%s\n' "$1" "$2"
}

record_binding "source_snapshot" "${source_snapshot}"
record_binding "source_tree_sha256" "${source_tree_sha256}"
record_binding "queue_dir" "${queue_dir}"
record_binding "data_root" "${data_root}"
record_binding "qwen_model" "${qwen_model}"
record_binding "output_dir" "${output_dir}"
record_binding "python_bin" "${python_bin}"
record_binding "nframes" "${nframes}"
record_binding "max_pixels" "${max_pixels}"
record_binding "max_new_tokens" "${max_new_tokens}"
record_binding "attn_implementation" "${attn_implementation}"
record_binding "visual_input" "${visual_input}"
record_binding "videos_copied" "0"
record_binding "videos_rendered" "0"
record_binding "optimizer_steps" "0"
record_binding "representation_gate_passed" "false"
record_binding "renderer_probe_authorized" "false"
record_binding "generation_authorized" "false"
record_binding "training_authorized" "false"

verify_bound_inputs() {
  local stage="$1"
  record_binding "input_verification_stage" "${stage}"
  "${python_bin}" \
    "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
    verify \
    --snapshot "${source_snapshot}" \
    --expected-tree-sha256 "${source_tree_sha256}"
  "${python_bin}" - \
    "${queue_dir}" \
    "${data_root}" \
    "${qwen_model}" <<'PY'
from pathlib import Path
import json
import re
import sys

from motive.r10b_bernini_pilot_manifest import _load_queue_commit
from motive.r10b_family_qwen_audit import (
    PROMPT_CONTRACT_SHA256,
    _resolve_media,
)

queue_raw, data_raw, model_raw = sys.argv[1:]
queue = Path(queue_raw).resolve(strict=True)
data = Path(data_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
rows, summary, files = _load_queue_commit(queue)
if not rows:
    raise SystemExit("Qwen queue is empty")
qwen = summary.get("qwen_audit", {})
model_id = qwen.get("qwen_model_id")
if (
    not isinstance(model_id, str)
    or not model_id
    or qwen.get("qwen_prompt_sha256") != PROMPT_CONTRACT_SHA256
):
    raise SystemExit("queue Qwen binding differs")
if "@" not in model_id:
    raise SystemExit("queue Qwen model ID must bind one revision with @")
revision = model_id.rsplit("@", 1)[1]
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise SystemExit("queue Qwen revision must be one lowercase Git commit")
if model.name != revision:
    raise SystemExit("Qwen model snapshot directory/revision differs")
config_path = model / "config.json"
if config_path.is_symlink():
    config_path = config_path.resolve(strict=True)
if not config_path.is_file():
    raise SystemExit("Qwen model config.json is unavailable")
config = json.loads(config_path.read_text(encoding="utf-8"))
model_type = str(config.get("model_type", "")).lower()
if "qwen" not in model_type or "vl" not in model_type:
    raise SystemExit(f"Qwen model is not one VL checkpoint: {model_type}")
for row in rows:
    if row.get("media_binding", {}).get("data_root") != str(data):
        raise SystemExit(f"iid={row.get('iid')} data-root binding differs")
    _resolve_media(row, data)
print(f"motive_r10b_qwen_audit.verified_queue_rows={len(rows)}")
print(
    "motive_r10b_qwen_audit.verified_queue_files="
    + ",".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
)
print(f"motive_r10b_qwen_audit.verified_qwen_model_id={model_id}")
print(f"motive_r10b_qwen_audit.verified_data_root={data}")
PY
}

verify_bound_inputs "before"

"${python_bin}" -m motive.r10b_family_qwen_audit \
  --queue-dir "${queue_dir}" \
  --data-root "${data_root}" \
  --model "${qwen_model}" \
  --output-dir "${output_dir}" \
  --nframes "${nframes}" \
  --max-pixels "${max_pixels}" \
  --max-new-tokens "${max_new_tokens}" \
  --attn-implementation "${attn_implementation}"

if [[ -L "${output_dir}" || ! -d "${output_dir}" ]]; then
  echo "Qwen auditor did not publish one non-symlink output directory" >&2
  exit 4
fi

"${python_bin}" -m motive.r10b_family_qwen_audit \
  --output-dir "${output_dir}" \
  --validate-only

verify_bound_inputs "after"

"${python_bin}" - \
  "${output_dir}" \
  "${queue_dir}" \
  "${qwen_model}" \
  "${nframes}" \
  "${max_pixels}" \
  "${max_new_tokens}" \
  "${attn_implementation}" <<'PY'
from pathlib import Path
import json
import sys

(
    output_raw,
    queue_raw,
    model_raw,
    nframes,
    max_pixels,
    max_new_tokens,
    attn_implementation,
) = sys.argv[1:]
output = Path(output_raw).resolve(strict=True)
queue = Path(queue_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
names = {path.name for path in output.iterdir()}
expected_names = {"records.jsonl", "adapters.jsonl", "summary.json", "done.json"}
if names != expected_names:
    raise SystemExit(
        f"Qwen audit output files differ: expected={expected_names}, observed={names}"
    )
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
done = json.loads((output / "done.json").read_text(encoding="utf-8"))
expected_runtime = {
    "nframes": int(nframes),
    "max_pixels": int(max_pixels),
    "max_new_tokens": int(max_new_tokens),
    "attn_implementation": attn_implementation,
}
runtime = summary.get("runtime", {})
for key, expected in expected_runtime.items():
    if runtime.get(key) != expected:
        raise SystemExit(f"published Qwen runtime contract differs: {key}")
implementation = runtime.get("implementation", {})
if (
    implementation.get("visual_input") != "mosaic"
    or not isinstance(implementation.get("family_audit_sha256"), str)
    or not isinstance(implementation.get("qwen_filter_sha256"), str)
):
    raise SystemExit("published Qwen implementation binding differs")
if summary.get("queue", {}).get("path") != str(queue):
    raise SystemExit("published Qwen queue binding differs")
if summary.get("model", {}).get("path") != str(model):
    raise SystemExit("published Qwen model binding differs")
closed_summary = {
    "videos_copied": 0,
    "videos_rendered": 0,
    "formal_evidence": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
for key, expected in closed_summary.items():
    if summary.get(key) != expected:
        raise SystemExit(f"published Qwen safety gate differs: {key}")
for key in (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "training_authorized",
):
    if done.get(key) is not False:
        raise SystemExit(f"published Qwen done gate is open: {key}")
print(f"motive_r10b_qwen_audit.output_rows={summary.get('rows')}")
print("motive_r10b_qwen_audit.status=validated_routing_evidence_only")
PY
