#!/usr/bin/env bash
#SBATCH --job-name=motive-r10b-qwen-v2-packed
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=02:00:00

set -Eeuo pipefail

source_snapshot="${MOTIVE_R10B_SOURCE_SNAPSHOT:?set MOTIVE_R10B_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_R10B_SOURCE_TREE_SHA256:?set MOTIVE_R10B_SOURCE_TREE_SHA256}"
full_queue_dir="${MOTIVE_R10B_FULL_QUEUE_DIR:?set MOTIVE_R10B_FULL_QUEUE_DIR}"
data_root="${MOTIVE_R10B_DATA_ROOT:?set MOTIVE_R10B_DATA_ROOT}"
qwen_model="${MOTIVE_R10B_QWEN_MODEL:?set MOTIVE_R10B_QWEN_MODEL}"
output_root="${MOTIVE_R10B_OUTPUT_ROOT:?set MOTIVE_R10B_OUTPUT_ROOT}"
python_bin="${MOTIVE_R10B_PYTHON_BIN:?set MOTIVE_R10B_PYTHON_BIN}"

shard_count=8
split_strategy="round_robin"
nframes="${MOTIVE_R10B_NFRAMES:-12}"
max_pixels="${MOTIVE_R10B_MAX_PIXELS:-294912}"
max_new_tokens=512
attn_implementation="sdpa"

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

require_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${label} must be one positive integer: ${value}" >&2
    exit 2
  fi
}

for binding in \
  "source_snapshot:${source_snapshot}" \
  "full_queue_dir:${full_queue_dir}" \
  "data_root:${data_root}" \
  "qwen_model:${qwen_model}" \
  "output_root:${output_root}" \
  "python_bin:${python_bin}"; do
  require_absolute_path "${binding%%:*}" "${binding#*:}"
done
require_sha256 "source_tree_sha256" "${source_tree_sha256}"
require_positive_integer "MOTIVE_R10B_NFRAMES" "${nframes}"
require_positive_integer "MOTIVE_R10B_MAX_PIXELS" "${max_pixels}"

if [[ -L "${source_snapshot}" || ! -d "${source_snapshot}" ]]; then
  echo "source snapshot must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${full_queue_dir}" || ! -d "${full_queue_dir}" ]]; then
  echo "full queue must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${data_root}" || ! -d "${data_root}" ]]; then
  echo "data root must be one non-symlink directory" >&2
  exit 2
fi
if [[ -L "${qwen_model}" || ! -d "${qwen_model}" ]]; then
  echo "Qwen model must be one non-symlink directory" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  echo "refusing to reuse packed output root: ${output_root}" >&2
  exit 2
fi
output_parent="${output_root%/*}"
if [[ ! -d "${output_parent}" || ! -w "${output_parent}" ]]; then
  echo "output parent must be an existing writable directory" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-}" != "${shard_count}" ]]; then
  echo "packed job requires exactly 8 real Slurm tasks" >&2
  exit 2
fi

code_root="${source_snapshot}/methods/motive"
for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${code_root}/motive/attribution.py" \
  "${code_root}/motive/qwen_filter.py" \
  "${code_root}/motive/r10b_bernini_retrieval_audit.py" \
  "${code_root}/motive/r10b_bernini_tangent_extract.py" \
  "${code_root}/motive/r10b_family_qwen_audit.py" \
  "${code_root}/motive/r10b_lucy_tangent_extract.py" \
  "${code_root}/motive/r10b_qwen_audit_shards.py" \
  "${code_root}/motive/r10b_bernini_pilot_manifest.py" \
  "${code_root}/motive/r10b_bernini_prompt_variants.py" \
  "${code_root}/motive/r10b_tangent_core.py" \
  "${full_queue_dir}/qwen_audit_queue.jsonl" \
  "${full_queue_dir}/summary.json" \
  "${full_queue_dir}/done.json" \
  "${qwen_model}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required packed input: ${required}" >&2
    exit 2
  fi
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHONPATH="${source_snapshot}:${code_root}"

job_scratch="${SLURM_TMPDIR:-/tmp}/motive-r10b-qwen-v2-${SLURM_JOB_ID:-manual}"
mkdir -p "${job_scratch}"

record_binding() {
  printf 'motive_r10b_qwen_v2_packed.%s=%s\n' "$1" "$2"
}

record_binding "source_snapshot" "${source_snapshot}"
record_binding "source_tree_sha256" "${source_tree_sha256}"
record_binding "full_queue_dir" "${full_queue_dir}"
record_binding "data_root" "${data_root}"
record_binding "qwen_model" "${qwen_model}"
record_binding "output_root" "${output_root}"
record_binding "python_bin" "${python_bin}"
record_binding "shard_count" "${shard_count}"
record_binding "split_strategy" "${split_strategy}"
record_binding "nframes" "${nframes}"
record_binding "max_pixels" "${max_pixels}"
record_binding "max_new_tokens" "${max_new_tokens}"
record_binding "attn_implementation" "${attn_implementation}"
record_binding "qwen_visual_calls_per_row" "1"
record_binding "qwen_text_calls_per_row" "0"
record_binding "videos_copied" "0"
record_binding "videos_rendered" "0"
record_binding "optimizer_steps" "0"
record_binding "representation_gate_passed" "false"
record_binding "renderer_probe_authorized" "false"
record_binding "generation_authorized" "false"
record_binding "training_authorized" "false"
record_binding "downstream_submitted" "false"

verify_bound_inputs() {
  local stage="$1"
  record_binding "input_verification_stage" "${stage}"
  "${python_bin}" \
    "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
    verify \
    --snapshot "${source_snapshot}" \
    --expected-tree-sha256 "${source_tree_sha256}"
  "${python_bin}" - \
    "${full_queue_dir}" \
    "${data_root}" \
    "${qwen_model}" \
    "${model_inventory_sha256}" <<'PY'
from pathlib import Path
import json
import re
import sys

from motive.r10b_bernini_pilot_manifest import _load_queue_commit
from motive.r10b_family_qwen_audit import (
    PROMPT_CONTRACT_SHA256,
    _model_inventory,
    _resolve_media,
)

queue_raw, data_raw, model_raw, expected_inventory_sha256 = sys.argv[1:]
queue = Path(queue_raw).resolve(strict=True)
data = Path(data_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
rows, summary, files = _load_queue_commit(queue)
if len(rows) < 8:
    raise SystemExit("full Qwen queue must contain at least eight rows")
qwen = summary.get("qwen_audit", {})
model_id = qwen.get("qwen_model_id")
if (
    not isinstance(model_id, str)
    or not model_id
    or qwen.get("qwen_prompt_sha256") != PROMPT_CONTRACT_SHA256
):
    raise SystemExit("full queue Qwen binding differs")
if "@" not in model_id:
    raise SystemExit("Qwen model ID must bind one revision with @")
revision = model_id.rsplit("@", 1)[1]
if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
    raise SystemExit("Qwen revision must be one lowercase Git commit")
if model.name != revision:
    raise SystemExit("Qwen snapshot directory/revision differs")
config_path = model / "config.json"
if config_path.is_symlink():
    config_path = config_path.resolve(strict=True)
if not config_path.is_file():
    raise SystemExit("Qwen config.json is unavailable")
config = json.loads(config_path.read_text(encoding="utf-8"))
model_type = str(config.get("model_type", "")).lower()
if "qwen" not in model_type or "vl" not in model_type:
    raise SystemExit(f"Qwen checkpoint is not one VL model: {model_type}")
inventory = _model_inventory(model)
large_threshold = 32 * 1024 * 1024
for item in inventory.get("files", []):
    if item.get("bytes", -1) <= large_threshold:
        continue
    if item.get("symlink") is not True or re.fullmatch(
        r"[0-9a-f]{64}",
        str(item.get("resolved_blob_name", "")),
    ) is None:
        raise SystemExit(
            "large Qwen checkpoint files must be immutable HF blob symlinks: "
            + str(item.get("path"))
        )
if inventory.get("sha256") != expected_inventory_sha256:
    raise SystemExit("Qwen model inventory changed after initial binding")
for row in rows:
    if row.get("media_binding", {}).get("data_root") != str(data):
        raise SystemExit(f"iid={row.get('iid')} data-root binding differs")
    _resolve_media(row, data)
print(f"motive_r10b_qwen_v2_packed.verified_queue_rows={len(rows)}")
print(
    "motive_r10b_qwen_v2_packed.verified_queue_files="
    + ",".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
)
print(f"motive_r10b_qwen_v2_packed.verified_model_id={model_id}")
print(
    "motive_r10b_qwen_v2_packed.verified_model_inventory_sha256="
    + inventory["sha256"]
)
print(
    "motive_r10b_qwen_v2_packed.verified_media_sha256_rows="
    + str(len(rows))
)
PY
}

model_inventory_digest() {
  "${python_bin}" - "${qwen_model}" <<'PY'
from pathlib import Path
import re
import sys

from motive.r10b_family_qwen_audit import _model_inventory

model = Path(sys.argv[1]).resolve(strict=True)
inventory = _model_inventory(model)
large_threshold = 32 * 1024 * 1024
for item in inventory.get("files", []):
    if item.get("bytes", -1) <= large_threshold:
        continue
    if item.get("symlink") is not True or re.fullmatch(
        r"[0-9a-f]{64}",
        str(item.get("resolved_blob_name", "")),
    ) is None:
        raise SystemExit(
            "large Qwen checkpoint files must be immutable HF blob symlinks: "
            + str(item.get("path"))
        )
digest = inventory.get("sha256")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("Qwen checkpoint inventory digest differs")
print(digest)
PY
}

probe_bound_gpu() {
  local probe_path="$1"
  "${MOTIVE_PACKED_PYTHON}" - \
    "${probe_path}" \
    "${SLURM_PROCID:?SLURM_PROCID is required}" \
    "${SLURM_LOCALID:?SLURM_LOCALID is required}" \
    "${SLURM_NTASKS:?SLURM_NTASKS is required}" <<'PY'
from pathlib import Path
import json
import os
import sys
import tempfile

import torch

probe_path = Path(sys.argv[1])
rank = int(sys.argv[2])
local_rank = int(sys.argv[3])
slurm_ntasks = int(sys.argv[4])
if rank not in range(8) or local_rank not in range(8) or slurm_ntasks != 8:
    raise SystemExit("packed GPU probe Slurm rank topology differs")
if os.path.lexists(probe_path):
    raise SystemExit(f"refusing to reuse GPU probe: {probe_path}")
if not probe_path.parent.is_dir():
    raise SystemExit(f"GPU probe parent is unavailable: {probe_path.parent}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is unavailable on packed Qwen rank")
device_count = torch.cuda.device_count()
if device_count != 1:
    raise SystemExit(
        f"packed Qwen rank must see exactly one GPU, observed={device_count}"
    )
device_name = str(torch.cuda.get_device_name(0))
if "MI210" not in device_name.upper():
    raise SystemExit(f"packed Qwen rank GPU is not MI210: {device_name}")
properties = torch.cuda.get_device_properties(0)
device_uuid = getattr(properties, "uuid", None)
if device_uuid is None and hasattr(torch.cuda, "get_device_uuid"):
    try:
        device_uuid = torch.cuda.get_device_uuid(0)
    except Exception:
        device_uuid = None
if device_uuid is not None:
    device_uuid = str(device_uuid)
tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
tensor_value = float((tensor * tensor).sum().item())
torch.cuda.synchronize(0)
if str(tensor.device) != "cuda:0" or tensor_value != 14.0:
    raise SystemExit(
        "cuda:0 tensor execution differs: "
        f"device={tensor.device}, value={tensor_value}"
    )
visibility_names = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_ID",
    "SLURMD_NODENAME",
    "HOSTNAME",
)
payload = {
    "schema_version": "motive-r10b-qwen-v2-gpu-probe-v1",
    "status": "passed",
    "rank": rank,
    "local_rank": local_rank,
    "slurm_ntasks": slurm_ntasks,
    "device_count": device_count,
    "device_index": 0,
    "device_name": device_name,
    "device_uuid": device_uuid,
    "tensor_probe": {
        "device": str(tensor.device),
        "sum_of_squares": tensor_value,
        "passed": True,
    },
    "visibility": {
        name: os.environ.get(name) for name in visibility_names
    },
}
raw = (
    json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
descriptor, temporary_raw = tempfile.mkstemp(
    prefix=f".{probe_path.name}.tmp.",
    dir=str(probe_path.parent),
)
temporary = Path(temporary_raw)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, probe_path)
    directory_descriptor = os.open(probe_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    if temporary.exists():
        temporary.unlink()
PY
}

require_complete_qwen_audit() {
  local summary_path="$1"
  local label="$2"
  "${MOTIVE_PACKED_PYTHON}" - \
    "${summary_path}" \
    "${label}" \
    "${MOTIVE_PACKED_NFRAMES:?MOTIVE_PACKED_NFRAMES is required}" \
    "${MOTIVE_PACKED_MAX_PIXELS:?MOTIVE_PACKED_MAX_PIXELS is required}" <<'PY'
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1]).resolve(strict=True)
label = sys.argv[2]
expected_nframes = int(sys.argv[3])
expected_max_pixels = int(sys.argv[4])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
counts = {}
for field in (
    "rows",
    "successful_rows",
    "schema_error_rows",
    "generation_error_rows",
):
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"{label} {field} differs")
    counts[field] = value
if (
    counts["successful_rows"]
    + counts["schema_error_rows"]
    + counts["generation_error_rows"]
    != counts["rows"]
):
    raise SystemExit(f"{label} Qwen outcome counts do not close")
if summary.get("status") != "complete":
    raise SystemExit(
        f"{label} Qwen status is not complete: {summary.get('status')}"
    )
if counts["generation_error_rows"] != 0:
    raise SystemExit(f"{label} has Qwen generation errors")
runtime = summary.get("runtime", {})
if (
    runtime.get("nframes") != expected_nframes
    or runtime.get("max_pixels") != expected_max_pixels
):
    raise SystemExit(
        f"{label} Qwen temporal/image runtime binding differs"
    )
execution = summary.get("runtime", {}).get("backend_execution")
if (
    not isinstance(execution, dict)
    or execution.get("schema_version")
    != "motive-r10b-qwen-backend-execution-v1"
    or execution.get("mode") != "production_local_qwen"
    or execution.get("production_backend") is not True
    or execution.get("test_backend") is not False
    or execution.get("inspection_performed") is not True
    or execution.get("verified_after_model_load") is not True
    or execution.get("cuda_available") is not True
    or execution.get("device_count") != 1
    or execution.get("current_device") != 0
    or execution.get("model_device") != "cuda:0"
    or execution.get("parameter_devices") != ["cuda:0"]
    or execution.get("cpu_offload_detected") is not False
    or execution.get("disk_offload_detected") is not False
    or execution.get("meta_offload_detected") is not False
    or execution.get("cuda_only") is not True
):
    raise SystemExit(
        f"{label} lacks production cuda:0-only backend execution evidence"
    )
print(
    f"motive_r10b_qwen_v2_packed.{label}.successful_rows="
    f"{counts['successful_rows']}"
)
print(
    f"motive_r10b_qwen_v2_packed.{label}.schema_error_rows="
    f"{counts['schema_error_rows']}"
)
print(
    f"motive_r10b_qwen_v2_packed.{label}.generation_error_rows=0"
)
PY
}

export -f probe_bound_gpu
export -f require_complete_qwen_audit

model_inventory_sha256="$(model_inventory_digest)"
require_sha256 "model_inventory_sha256" "${model_inventory_sha256}"
record_binding "model_inventory_sha256" "${model_inventory_sha256}"
verify_bound_inputs "before"

mkdir "${output_root}"
shard_queues="${output_root}/shard_queues"
shard_audits="${output_root}/shard_audits"
gpu_probes="${output_root}/gpu_probes"
merged_audit="${output_root}/merged_audit"
pilot_manifest="${output_root}/pilot_manifest"
prompt_variants="${output_root}/prompt_variants"
pipeline_summary="${output_root}/pipeline_summary.json"

cd "${code_root}"

"${python_bin}" -m motive.r10b_qwen_audit_shards split \
  --full-queue-dir "${full_queue_dir}" \
  --output-root "${shard_queues}" \
  --shard-count "${shard_count}" \
  --strategy "${split_strategy}"

"${python_bin}" -m motive.r10b_qwen_audit_shards validate-split \
  --shard-queues-root "${shard_queues}" \
  --full-queue-dir "${full_queue_dir}"

mkdir "${shard_audits}"
mkdir "${gpu_probes}"
export MOTIVE_PACKED_PYTHON="${python_bin}"
export MOTIVE_PACKED_SHARD_QUEUES="${shard_queues}"
export MOTIVE_PACKED_SHARD_AUDITS="${shard_audits}"
export MOTIVE_PACKED_GPU_PROBES="${gpu_probes}"
export MOTIVE_PACKED_DATA_ROOT="${data_root}"
export MOTIVE_PACKED_QWEN_MODEL="${qwen_model}"
export MOTIVE_PACKED_SCRATCH="${job_scratch}"
export MOTIVE_PACKED_NFRAMES="${nframes}"
export MOTIVE_PACKED_MAX_PIXELS="${max_pixels}"
export MOTIVE_PACKED_MAX_NEW_TOKENS="${max_new_tokens}"
export MOTIVE_PACKED_ATTN_IMPLEMENTATION="${attn_implementation}"

# Every Slurm rank owns a distinct immutable queue shard and performs real
# visual Qwen inference on its bound MI210.  No controller-only GPU rank exists.
srun \
  --kill-on-bad-exit=1 \
  --ntasks=8 \
  --ntasks-per-node=8 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -c '
    set -Eeuo pipefail
    if [[ ! "${SLURM_PROCID:-}" =~ ^[0-7]$ ]]; then
      echo "invalid packed rank: ${SLURM_PROCID:-unset}" >&2
      exit 3
    fi
    printf -v shard_name "shard_%03d" "${SLURM_PROCID}"
    shard_queue="${MOTIVE_PACKED_SHARD_QUEUES}/${shard_name}"
    shard_audit="${MOTIVE_PACKED_SHARD_AUDITS}/${shard_name}"
    if [[ ! -d "${shard_queue}" || -e "${shard_audit}" ]]; then
      echo "${shard_name} queue/output precondition differs" >&2
      exit 3
    fi
    rank_scratch="${MOTIVE_PACKED_SCRATCH}/${shard_name}"
    mkdir -p \
      "${rank_scratch}/hf" \
      "${rank_scratch}/miopen-user" \
      "${rank_scratch}/miopen-custom"
    export HF_HOME="${rank_scratch}/hf"
    export MIOPEN_USER_DB_PATH="${rank_scratch}/miopen-user"
    export MIOPEN_CUSTOM_CACHE_DIR="${rank_scratch}/miopen-custom"
    probe_path="${MOTIVE_PACKED_GPU_PROBES}/${shard_name}.json"
    probe_bound_gpu "${probe_path}"
    "${MOTIVE_PACKED_PYTHON}" -m motive.r10b_family_qwen_audit \
      --queue-dir "${shard_queue}" \
      --data-root "${MOTIVE_PACKED_DATA_ROOT}" \
      --model "${MOTIVE_PACKED_QWEN_MODEL}" \
      --output-dir "${shard_audit}" \
      --nframes "${MOTIVE_PACKED_NFRAMES}" \
      --max-pixels "${MOTIVE_PACKED_MAX_PIXELS}" \
      --max-new-tokens "${MOTIVE_PACKED_MAX_NEW_TOKENS}" \
      --attn-implementation "${MOTIVE_PACKED_ATTN_IMPLEMENTATION}"
    "${MOTIVE_PACKED_PYTHON}" -m motive.r10b_family_qwen_audit \
      --output-dir "${shard_audit}" \
      --validate-only
    require_complete_qwen_audit \
      "${shard_audit}/summary.json" \
      "${shard_name}"
    printf "motive_r10b_qwen_v2_packed.%s.qwen_visual_calls_per_row=1\n" "${shard_name}"
    printf "motive_r10b_qwen_v2_packed.%s.qwen_text_calls_per_row=0\n" "${shard_name}"
  '

for shard_index in 0 1 2 3 4 5 6 7; do
  printf -v shard_name "shard_%03d" "${shard_index}"
  if [[ ! -d "${shard_audits}/${shard_name}" ]]; then
    echo "missing completed audit shard: ${shard_name}" >&2
    exit 4
  fi
done

"${python_bin}" -m motive.r10b_qwen_audit_shards merge \
  --full-queue-dir "${full_queue_dir}" \
  --shard-queues-root "${shard_queues}" \
  --shard-audits-root "${shard_audits}" \
  --output-dir "${merged_audit}"

"${python_bin}" -m motive.r10b_qwen_audit_shards validate-merge \
  --output-dir "${merged_audit}"

require_complete_qwen_audit \
  "${merged_audit}/summary.json" \
  "merged_audit"

"${python_bin}" -m motive.r10b_bernini_pilot_manifest finalize \
  --queue-dir "${full_queue_dir}" \
  --audit-records "${merged_audit}/adapters.jsonl" \
  --output-dir "${pilot_manifest}"

controlled_pilot_commit_digest="$("${python_bin}" - \
  "${pilot_manifest}" <<'PY'
import re
import sys

from motive.r10b_bernini_retrieval_audit import (
    validate_controlled_pilot_commit,
)

result = validate_controlled_pilot_commit(sys.argv[1])
if (
    result.get("status") != "VALID"
    or result.get("representation_gate_passed") is not False
    or result.get("renderer_probe_authorized") is not False
    or result.get("training_authorized") is not False
):
    raise SystemExit("controlled-pilot commit validator gates differ")
digest = result.get("commit_digest")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("controlled-pilot commit digest differs")
print(digest)
PY
)"
require_sha256 \
  "controlled_pilot_commit_digest" \
  "${controlled_pilot_commit_digest}"
record_binding \
  "controlled_pilot_commit_digest" \
  "${controlled_pilot_commit_digest}"

balanced="$("${python_bin}" -c \
  'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("balanced_pilot_ready"); print("true" if value is True else "false")' \
  "${pilot_manifest}/summary.json")"
if [[ "${balanced}" == "true" ]]; then
  "${python_bin}" -m motive.r10b_bernini_prompt_variants build \
    --pilot-dir "${pilot_manifest}" \
    --output-dir "${prompt_variants}"
  "${python_bin}" -m motive.r10b_bernini_prompt_variants validate \
    --pilot-dir "${pilot_manifest}" \
    --variant-dir "${prompt_variants}"
  record_binding "prompt_variants" "built_after_balanced_gate"
elif [[ "${balanced}" == "false" ]]; then
  if [[ -e "${prompt_variants}" || -L "${prompt_variants}" ]]; then
    echo "unbalanced pilot must not publish prompt variants" >&2
    exit 4
  fi
  record_binding "prompt_variants" "not_built_unbalanced_pilot"
else
  echo "invalid balanced_pilot_ready value: ${balanced}" >&2
  exit 4
fi

verify_bound_inputs "after"

"${python_bin}" - \
  "${output_root}" \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${full_queue_dir}" \
  "${data_root}" \
  "${qwen_model}" \
  "${python_bin}" \
  "${balanced}" \
  "${model_inventory_sha256}" \
  "${controlled_pilot_commit_digest}" \
  "${nframes}" \
  "${max_pixels}" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

(
    output_raw,
    snapshot_raw,
    source_tree_sha256,
    queue_raw,
    data_raw,
    model_raw,
    python_raw,
    balanced_raw,
    model_inventory_sha256,
    controlled_pilot_commit_digest,
    nframes_raw,
    max_pixels_raw,
) = sys.argv[1:]
output = Path(output_raw).resolve(strict=True)
snapshot = Path(snapshot_raw).resolve(strict=True)
queue = Path(queue_raw).resolve(strict=True)
data = Path(data_raw).resolve(strict=True)
model = Path(model_raw).resolve(strict=True)
python_bin = Path(python_raw).resolve(strict=True)
balanced = balanced_raw == "true"
nframes = int(nframes_raw)
max_pixels = int(max_pixels_raw)
if nframes <= 0 or max_pixels <= 0:
    raise SystemExit("packed Qwen runtime values must be positive")

def require_sha256(value, *, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SystemExit(f"{field} must be lowercase SHA-256")
    return value

require_sha256(
    model_inventory_sha256,
    field="model_inventory_sha256",
)
require_sha256(
    controlled_pilot_commit_digest,
    field="controlled_pilot_commit_digest",
)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def canonical_digest(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def tree_record(root, *, excluded_relative_paths=()):
    excluded = set(excluded_relative_paths)
    rows = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise SystemExit(f"published output contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SystemExit(f"published output is not regular: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        rows.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256(path),
            }
        )
        total_bytes += size
    raw = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return {
        "path": str(root),
        "files": len(rows),
        "bytes": total_bytes,
        "tree_sha256": hashlib.sha256(raw).hexdigest(),
    }

def atomic_write_new(path, payload):
    if os.path.lexists(path):
        raise SystemExit(f"refusing to reuse pipeline commit file: {path}")
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=str(path.parent),
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()

required_names = {
    "gpu_probes",
    "shard_queues",
    "shard_audits",
    "merged_audit",
    "pilot_manifest",
}
if balanced:
    required_names.add("prompt_variants")
observed_names = {path.name for path in output.iterdir()}
if observed_names != required_names:
    raise SystemExit(
        f"packed output topology differs: expected={required_names}, observed={observed_names}"
    )
for name in required_names:
    path = output / name
    if path.is_symlink() or not path.is_dir():
        raise SystemExit(f"packed artifact root is not one directory: {path}")

visibility_names = {
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_JOB_ID",
    "SLURMD_NODENAME",
    "HOSTNAME",
}
probe_root = output / "gpu_probes"
expected_probe_names = {
    f"shard_{rank:03d}.json" for rank in range(8)
}
observed_probe_names = {path.name for path in probe_root.iterdir()}
if observed_probe_names != expected_probe_names:
    raise SystemExit(
        "GPU probe closure differs: "
        f"expected={expected_probe_names}, observed={observed_probe_names}"
    )
gpu_probe_evidence = []
for rank in range(8):
    path = probe_root / f"shard_{rank:03d}.json"
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"GPU probe is not one regular file: {path}")
    probe = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "status",
        "rank",
        "local_rank",
        "slurm_ntasks",
        "device_count",
        "device_index",
        "device_name",
        "device_uuid",
        "tensor_probe",
        "visibility",
    }
    if set(probe) != expected_fields:
        raise SystemExit(f"rank={rank} GPU probe schema differs")
    if (
        probe.get("schema_version")
        != "motive-r10b-qwen-v2-gpu-probe-v1"
        or probe.get("status") != "passed"
        or probe.get("rank") != rank
        or probe.get("local_rank") != rank
        or probe.get("slurm_ntasks") != 8
        or probe.get("device_count") != 1
        or probe.get("device_index") != 0
        or "MI210" not in str(probe.get("device_name", "")).upper()
    ):
        raise SystemExit(f"rank={rank} GPU probe identity differs")
    device_uuid = probe.get("device_uuid")
    if device_uuid is not None and (
        not isinstance(device_uuid, str) or not device_uuid
    ):
        raise SystemExit(f"rank={rank} GPU UUID differs")
    tensor_probe = probe.get("tensor_probe")
    if (
        not isinstance(tensor_probe, dict)
        or set(tensor_probe)
        != {"device", "sum_of_squares", "passed"}
        or tensor_probe.get("device") != "cuda:0"
        or tensor_probe.get("sum_of_squares") != 14.0
        or tensor_probe.get("passed") is not True
    ):
        raise SystemExit(f"rank={rank} cuda:0 tensor probe differs")
    visibility = probe.get("visibility")
    if (
        not isinstance(visibility, dict)
        or set(visibility) != visibility_names
        or any(
            value is not None and not isinstance(value, str)
            for value in visibility.values()
        )
        or visibility.get("SLURM_PROCID") != str(rank)
        or visibility.get("SLURM_LOCALID") != str(rank)
        or visibility.get("SLURM_NTASKS") != "8"
    ):
        raise SystemExit(f"rank={rank} Slurm GPU visibility differs")
    gpu_probe_evidence.append(
        {
            "rank": rank,
            "local_rank": rank,
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": sha256(path),
            "device_name": probe["device_name"],
            "device_uuid": device_uuid,
            "visibility": visibility,
        }
    )

merged_summary = json.loads(
    (output / "merged_audit" / "summary.json").read_text(encoding="utf-8")
)
pilot_summary = json.loads(
    (output / "pilot_manifest" / "summary.json").read_text(encoding="utf-8")
)
prompt_contract = merged_summary.get("prompt_contract", {})
if (
    prompt_contract.get("qwen_visual_calls_per_row") != 1
    or prompt_contract.get("qwen_text_calls_per_row") != 0
    or prompt_contract.get("stage2") != "deterministic_python_no_model_call"
):
    raise SystemExit("merged Qwen visual/text-call contract differs")
backend_execution = merged_summary.get("runtime", {}).get(
    "backend_execution"
)
merged_runtime = merged_summary.get("runtime", {})
if (
    merged_runtime.get("nframes") != nframes
    or merged_runtime.get("max_pixels") != max_pixels
):
    raise SystemExit("merged Qwen temporal/image runtime binding differs")
if (
    not isinstance(backend_execution, dict)
    or backend_execution.get("schema_version")
    != "motive-r10b-qwen-backend-execution-v1"
    or backend_execution.get("mode") != "production_local_qwen"
    or backend_execution.get("production_backend") is not True
    or backend_execution.get("test_backend") is not False
    or backend_execution.get("inspection_performed") is not True
    or backend_execution.get("verified_after_model_load") is not True
    or backend_execution.get("cuda_available") is not True
    or backend_execution.get("device_count") != 1
    or backend_execution.get("current_device") != 0
    or backend_execution.get("model_device") != "cuda:0"
    or backend_execution.get("parameter_devices") != ["cuda:0"]
    or backend_execution.get("cpu_offload_detected") is not False
    or backend_execution.get("disk_offload_detected") is not False
    or backend_execution.get("meta_offload_detected") is not False
    or backend_execution.get("cuda_only") is not True
):
    raise SystemExit(
        "merged Qwen backend execution is not production cuda:0-only"
    )
qwen_outcome_counts = {}
for field in (
    "rows",
    "successful_rows",
    "schema_error_rows",
    "generation_error_rows",
):
    value = merged_summary.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"merged Qwen {field} differs")
    qwen_outcome_counts[field] = value
if (
    qwen_outcome_counts["successful_rows"]
    + qwen_outcome_counts["schema_error_rows"]
    + qwen_outcome_counts["generation_error_rows"]
    != qwen_outcome_counts["rows"]
    or merged_summary.get("status") != "complete"
    or qwen_outcome_counts["generation_error_rows"] != 0
):
    raise SystemExit("merged Qwen audit is not generation-complete")
if (
    merged_summary.get("model", {})
    .get("inventory", {})
    .get("sha256")
    != model_inventory_sha256
):
    raise SystemExit("merged Qwen model inventory binding differs")
for field in (
    "formal_evidence",
    "representation_gate_passed",
    "renderer_probe_authorized",
    "generation_authorized",
    "training_authorized",
):
    if merged_summary.get(field) is not False:
        raise SystemExit(f"merged Qwen gate is open: {field}")
if pilot_summary.get("balanced_pilot_ready") is not balanced:
    raise SystemExit("pilot balanced gate differs")

queue_summary = json.loads((queue / "summary.json").read_text(encoding="utf-8"))
model_id = queue_summary.get("qwen_audit", {}).get("qwen_model_id")
revision = model_id.rsplit("@", 1)[1]
config_path = model / "config.json"
if config_path.is_symlink():
    config_path = config_path.resolve(strict=True)
outputs = {
    name: tree_record(output / name)
    for name in sorted(required_names)
}
summary = {
    "schema_version": "motive-r10b-qwen-v2-packed-pipeline-v1",
    "status": "ready_for_commit",
    "inputs": {
        "source_snapshot": {
            "path": str(snapshot),
            "tree_sha256": source_tree_sha256,
        },
        "full_queue": tree_record(queue),
        "data_root": {
            "path": str(data),
            "media_sha256_verified_before": True,
            "media_sha256_verified_after": True,
        },
        "qwen_model": {
            "path": str(model),
            "model_id": model_id,
            "revision": revision,
            "config_sha256": sha256(config_path),
            "inventory_sha256": model_inventory_sha256,
            "revision_verified_before": True,
            "revision_verified_after": True,
        },
        "python": str(python_bin),
    },
    "execution": {
        "nodes": 1,
        "real_gpu_tasks": len(gpu_probe_evidence),
        "mi210_gpus": len(gpu_probe_evidence),
        "gpu_probe_evidence": gpu_probe_evidence,
        "queue_shards": 8,
        "split_strategy": "round_robin",
        "nframes": nframes,
        "max_pixels": max_pixels,
        "max_new_tokens": 512,
        "attn_implementation": "sdpa",
        "qwen_visual_calls_per_row": 1,
        "qwen_text_calls_per_row": 0,
        "stage2": "deterministic_python_no_model_call",
        "backend_execution": backend_execution,
        "qwen_rows": qwen_outcome_counts["rows"],
        "successful_rows": qwen_outcome_counts["successful_rows"],
        "schema_error_rows": qwen_outcome_counts["schema_error_rows"],
        "generation_error_rows": qwen_outcome_counts[
            "generation_error_rows"
        ],
    },
    "outputs": outputs,
    "controlled_pilot_commit_digest": controlled_pilot_commit_digest,
    "balanced_pilot_ready": balanced,
    "prompt_variants_materialized": balanced,
    "videos_copied": 0,
    "videos_rendered": 0,
    "optimizer_steps": 0,
    "formal_evidence": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
    "downstream_submitted": False,
    "authorization": {
        "formal_evidence": False,
        "representation_promoted": False,
        "renderer_probe_authorized": False,
        "generation_authorized": False,
        "training_authorized": False,
    },
}
payload = (
    json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
summary_path = output / "pipeline_summary.json"
atomic_write_new(summary_path, payload)
published = json.loads(summary_path.read_text(encoding="utf-8"))
if published != summary:
    raise SystemExit("pipeline summary round-trip differs")
summary_record = {
    "bytes": len(payload),
    "sha256": hashlib.sha256(payload).hexdigest(),
}
summary_topology = required_names | {"pipeline_summary.json"}
if {path.name for path in output.iterdir()} != summary_topology:
    raise SystemExit("packed topology differs after pipeline summary")
committed_output_tree = tree_record(output)
commit_binding = {
    "pipeline_summary": summary_record,
    "output_tree_without_done": committed_output_tree,
    "controlled_pilot_commit_digest": controlled_pilot_commit_digest,
}
done = {
    "schema_version": "motive-r10b-qwen-v2-packed-pipeline-done-v1",
    "status": "complete",
    "pipeline_summary": summary_record,
    "output_tree_without_done": committed_output_tree,
    "commit_digest": canonical_digest(commit_binding),
    "controlled_pilot_commit_digest": controlled_pilot_commit_digest,
    "balanced_pilot_ready": balanced,
    "nframes": nframes,
    "max_pixels": max_pixels,
    "real_gpu_tasks_verified": len(gpu_probe_evidence),
    "production_backend_cuda_only": (
        backend_execution.get("cuda_only") is True
    ),
    "successful_rows": qwen_outcome_counts["successful_rows"],
    "schema_error_rows": qwen_outcome_counts["schema_error_rows"],
    "generation_error_rows": qwen_outcome_counts[
        "generation_error_rows"
    ],
    "authorization": {
        "formal_evidence": False,
        "representation_promoted": False,
        "renderer_probe_authorized": False,
        "generation_authorized": False,
        "training_authorized": False,
    },
}
done_payload = (
    json.dumps(done, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
).encode("utf-8")
done_path = output / "pipeline_done.json"
atomic_write_new(done_path, done_payload)
final_topology = summary_topology | {"pipeline_done.json"}
if {path.name for path in output.iterdir()} != final_topology:
    raise SystemExit("final packed output closure differs")
if done_path.is_symlink() or not done_path.is_file():
    raise SystemExit("pipeline done is not one regular file")
if json.loads(done_path.read_text(encoding="utf-8")) != done:
    raise SystemExit("pipeline done round-trip differs")
if sha256(summary_path) != done["pipeline_summary"]["sha256"]:
    raise SystemExit("pipeline done summary digest differs")
if (
    tree_record(
        output,
        excluded_relative_paths={"pipeline_done.json"},
    )
    != done["output_tree_without_done"]
):
    raise SystemExit("pipeline done output-tree binding differs")
if done["commit_digest"] != canonical_digest(commit_binding):
    raise SystemExit("pipeline done commit digest differs")
print(
    "motive_r10b_qwen_v2_packed.pipeline_summary_sha256="
    + summary_record["sha256"]
)
print(
    "motive_r10b_qwen_v2_packed.pipeline_done_sha256="
    + hashlib.sha256(done_payload).hexdigest()
)
print(
    "motive_r10b_qwen_v2_packed.status="
    + ("balanced_prompt_variants_ready" if balanced else "unbalanced_no_prompt_variants")
)
PY

record_binding "status" "complete_routing_evidence_only"
