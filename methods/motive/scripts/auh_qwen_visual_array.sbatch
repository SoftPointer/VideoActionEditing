#!/usr/bin/env bash
#SBATCH --job-name=motive-qwen-visual
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=1-00:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
data_root="${MOTIVE_DATA_ROOT:-${repo_root}/data/goku/subject_movement/extracted}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${QWEN_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
model_path="${QWEN_MODEL_PATH:-/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
input_manifest="${MOTIVE_QWEN_INPUT:-${output_root}/feature_candidates/qwen_queue.jsonl}"
shard_index="${SLURM_ARRAY_TASK_ID:-${MOTIVE_QWEN_SHARD_INDEX:-}}"
shard_count="${MOTIVE_QWEN_NUM_SHARDS:-8}"

if [[ -z "${shard_index}" ]]; then
  echo "[motive-qwen] submit as a Slurm array or set MOTIVE_QWEN_SHARD_INDEX" >&2
  exit 2
fi
if ! [[ "${shard_index}" =~ ^[0-9]+$ && "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[motive-qwen] invalid shard index/count: ${shard_index}/${shard_count}" >&2
  exit 2
fi
if (( shard_count != 8 || shard_index >= shard_count )); then
  echo "[motive-qwen] this launcher requires exactly eight shards, got ${shard_index}/${shard_count}" >&2
  exit 2
fi
for required in \
  "${code_root}/motive/qwen_filter.py" \
  "${data_root}" \
  "${input_manifest}" \
  "${model_path}/config.json" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-qwen] missing required path: ${required}" >&2
    exit 2
  fi
done

shard_tag="$(printf '%03d' "${shard_index}")"
qwen_root="${output_root}/qwen_visual"
shard_input="${qwen_root}/manifests/shard-${shard_tag}.jsonl"
source_marker="${shard_input}.source"
shard_output="${qwen_root}/shards/qwen-${shard_tag}.jsonl"
source_digest="$(sha256sum "${input_manifest}" | awk '{print $1}')"
source_rows="$(awk 'END {print NR + 0}' "${input_manifest}")"
legacy_provenance="${MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE:-0}"
if [[ "${legacy_provenance}" != "0" && "${legacy_provenance}" != "1" ]]; then
  echo "[motive-qwen] MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$(dirname "${shard_input}")"
temporary_expected="${shard_input}.expected.tmp.${SLURM_JOB_ID:-manual}.${shard_index}"
temporary_marker="${source_marker}.tmp.${SLURM_JOB_ID:-manual}.${shard_index}"
cleanup_manifest_temporaries() {
  rm -f -- "${temporary_expected}" "${temporary_marker}"
}
trap cleanup_manifest_temporaries EXIT
# GNU awk reserves `index` as a builtin function name. Recompute the expected
# byte stream on every launch so a marker and shard cannot drift together.
awk -v shard="${shard_index}" -v count="${shard_count}" \
  '((NR - 1) % count) == shard' "${input_manifest}" > "${temporary_expected}"
expected_shard_digest="$(sha256sum "${temporary_expected}" | awk '{print $1}')"
expected_shard_rows="$(awk 'END {print NR + 0}' "${temporary_expected}")"
verified_source_digest="$(sha256sum "${input_manifest}" | awk '{print $1}')"
verified_source_rows="$(awk 'END {print NR + 0}' "${input_manifest}")"
if [[ "${verified_source_digest}" != "${source_digest}" || "${verified_source_rows}" != "${source_rows}" ]]; then
  echo "[motive-qwen] source input changed while materializing shard=${shard_index}" >&2
  exit 2
fi

if [[ -e "${shard_input}" && -e "${source_marker}" ]]; then
  if [[ ! -f "${shard_input}" || ! -f "${source_marker}" ]]; then
    echo "[motive-qwen] shard input/marker must be regular files" >&2
    exit 2
  fi
  if ! cmp -s "${temporary_expected}" "${shard_input}"; then
    echo "[motive-qwen] existing shard bytes differ from source partition: ${shard_input}" >&2
    exit 2
  fi
elif [[ -e "${shard_input}" || -e "${source_marker}" ]]; then
  echo "[motive-qwen] incomplete shard input exists: ${shard_input}" >&2
  exit 2
else
  mv "${temporary_expected}" "${shard_input}"
  printf \
    '{"partition":"line_modulo_v1","schema_version":"motive-qwen-shard-manifest-v2","shard_count":%s,"shard_index":%s,"shard_rows":%s,"shard_sha256":"%s","source_rows":%s,"source_sha256":"%s"}\n' \
    "${shard_count}" \
    "${shard_index}" \
    "${expected_shard_rows}" \
    "${expected_shard_digest}" \
    "${source_rows}" \
    "${source_digest}" > "${temporary_marker}"
  mv "${temporary_marker}" "${source_marker}"
fi

shard_digest="$(sha256sum "${shard_input}" | awk '{print $1}')"
shard_rows="$(awk 'END {print NR + 0}' "${shard_input}")"
if [[ "${shard_digest}" != "${expected_shard_digest}" || "${shard_rows}" != "${expected_shard_rows}" ]]; then
  echo "[motive-qwen] shard SHA256/row count changed after materialization" >&2
  exit 2
fi
marker_kind="$("${python_bin}" - \
  "${source_marker}" \
  "${source_digest}" \
  "${source_rows}" \
  "${shard_count}" \
  "${shard_index}" \
  "${shard_digest}" \
  "${shard_rows}" \
  "${legacy_provenance}" <<'PY'
import json
import re
import sys
from pathlib import Path

(
    marker_path,
    source_sha256,
    source_rows,
    shard_count,
    shard_index,
    shard_sha256,
    shard_rows,
    allow_legacy,
) = sys.argv[1:]
text = Path(marker_path).read_text(encoding="utf-8").strip()
expected = {
    "partition": "line_modulo_v1",
    "schema_version": "motive-qwen-shard-manifest-v2",
    "shard_count": int(shard_count),
    "shard_index": int(shard_index),
    "shard_rows": int(shard_rows),
    "shard_sha256": shard_sha256,
    "source_rows": int(source_rows),
    "source_sha256": source_sha256,
}
try:
    marker = json.loads(text)
except json.JSONDecodeError:
    marker = None
if isinstance(marker, dict):
    if marker != expected:
        raise RuntimeError(
            f"shard marker mismatch: expected={expected!r} actual={marker!r}"
        )
    print("v2")
else:
    if allow_legacy != "1":
        raise RuntimeError(
            "legacy shard marker refused; set "
            "MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE=1 only for an audited "
            "in-flight legacy run"
        )
    legacy_expected = f"{source_sha256}|{shard_count}|{shard_index}"
    if not re.fullmatch(r"[0-9a-f]{64}\|[1-9][0-9]*\|[0-9]+", text):
        raise RuntimeError("malformed legacy shard marker")
    if text != legacy_expected:
        raise RuntimeError(
            f"legacy shard marker mismatch: expected={legacy_expected!r} "
            f"actual={text!r}"
        )
    print("legacy-v1")
PY
)"
rm -f -- "${temporary_expected}" "${temporary_marker}"
trap - EXIT
if [[ "${marker_kind}" == "legacy-v1" ]]; then
  echo "[motive-qwen] WARNING: accepting audited legacy shard marker for shard=${shard_index}" >&2
fi
mkdir -p "$(dirname "${shard_output}")"

cache_parent="${MOTIVE_JOB_CACHE_PARENT:-${SLURM_TMPDIR:-/tmp}}"
cache_root="${cache_parent}/motive-qwen-${SLURM_JOB_ID:-manual}-${shard_tag}"
mkdir -p \
  "${cache_root}/tmp" \
  "${cache_root}/xdg" \
  "${cache_root}/huggingface" \
  "${cache_root}/torch" \
  "${cache_root}/torch-kernels" \
  "${cache_root}/inductor" \
  "${cache_root}/triton" \
  "${cache_root}/miopen" \
  "${cache_root}/miopen-custom"

export TMPDIR="${cache_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"
export HF_HOME="${cache_root}/huggingface"
export TORCH_HOME="${cache_root}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${cache_root}/torch-kernels"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
export TRITON_CACHE_DIR="${cache_root}/triton"
export MIOPEN_USER_DB_PATH="${cache_root}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${cache_root}/miopen-custom"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"

visible_devices="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${visible_devices}" != "1" ]]; then
  echo "[motive-qwen] expected exactly one Slurm-visible GPU, got ${visible_devices}" >&2
  exit 2
fi

cd "${repo_root}"
echo "[motive-qwen] shard=${shard_index}/${shard_count} input=${shard_input}"
echo "[motive-qwen] model=${model_path} cache=${cache_root}"
qwen_error_args=()
if [[ "${MOTIVE_QWEN_ALLOW_ERRORS:-0}" == "1" ]]; then
  qwen_error_args+=(--allow-errors)
fi
qwen_limit_args=()
if [[ -n "${MOTIVE_QWEN_MAX_SAMPLES:-}" ]]; then
  if ! [[ "${MOTIVE_QWEN_MAX_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[motive-qwen] MOTIVE_QWEN_MAX_SAMPLES must be a positive integer" >&2
    exit 2
  fi
  qwen_limit_args+=(--max-samples "${MOTIVE_QWEN_MAX_SAMPLES}")
fi
repair_attempts="${MOTIVE_QWEN_REPAIR_ATTEMPTS:-1}"
if ! [[ "${repair_attempts}" =~ ^[0-9]+$ ]]; then
  echo "[motive-qwen] MOTIVE_QWEN_REPAIR_ATTEMPTS must be non-negative" >&2
  exit 2
fi
schema_failure_policy="${MOTIVE_QWEN_SCHEMA_FAILURE_POLICY:-error}"
if [[ "${schema_failure_policy}" != "error" && "${schema_failure_policy}" != "uncertain" ]]; then
  echo "[motive-qwen] invalid schema failure policy: ${schema_failure_policy}" >&2
  exit 2
fi
exec "${python_bin}" -m motive.qwen_filter \
  --input "${shard_input}" \
  --output "${shard_output}" \
  --model "${model_path}" \
  --mode visual \
  --root "${data_root}" \
  --resume \
  --max-new-tokens "${MOTIVE_QWEN_MAX_NEW_TOKENS:-512}" \
  --repair-attempts "${repair_attempts}" \
  --schema-failure-policy "${schema_failure_policy}" \
  --nframes 6 \
  --max-pixels "${MOTIVE_QWEN_MAX_PIXELS:-147456}" \
  --visual-input mosaic \
  --attn-implementation sdpa \
  --execution-shard-index "${shard_index}" \
  --execution-shard-count "${shard_count}" \
  "${qwen_limit_args[@]}" \
  "${qwen_error_args[@]}"
