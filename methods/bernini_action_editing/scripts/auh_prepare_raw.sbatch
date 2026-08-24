#!/usr/bin/env bash
#SBATCH --job-name=bern-r13-raw644
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
qwen_passed="${BERNINI_ACTION_QWEN_PASSED:?set BERNINI_ACTION_QWEN_PASSED}"
wan_root="${BERNINI_ACTION_WAN_ROOT:?set BERNINI_ACTION_WAN_ROOT}"
natural_root="${BERNINI_ACTION_NATURAL_ROOT:?set BERNINI_ACTION_NATURAL_ROOT}"
output_root="${BERNINI_ACTION_RAW_OUTPUT:?set BERNINI_ACTION_RAW_OUTPUT}"
expected_rows="${BERNINI_ACTION_EXPECTED_ROWS:-644}"
expected_strict_rows="${BERNINI_ACTION_EXPECTED_STRICT_ROWS:-359}"
expected_terminal_rows="${BERNINI_ACTION_EXPECTED_TERMINAL_ROWS:-673}"
expected_error_rows="${BERNINI_ACTION_EXPECTED_ERROR_ROWS:-29}"
expected_natural_summary_sha256="${BERNINI_ACTION_EXPECTED_NATURAL_SUMMARY_SHA256:-3cc2267f47d204bd9654eed6c1c3b35cf60ed243fb88835afcd761da82a555bd}"
expected_natural_manifest_sha256="${BERNINI_ACTION_EXPECTED_NATURAL_MANIFEST_SHA256:-29036f2232af0dfad92e2ae740477285cc7471bc69c54e53c1c8b254ecf3da76}"
expected_natural_iid_set_sha256="${BERNINI_ACTION_EXPECTED_NATURAL_IID_SET_SHA256:-4909037868c6f2c3f9697595995fba770025cae7a215669f36e9e7ae00b67685}"
expected_qwen_index_sha256="${BERNINI_ACTION_EXPECTED_QWEN_INDEX_SHA256:-a1b8b912f2c5e1b00b91c26ddc1641698acbdf2a5e351a2e4f8eee9513ce2c17}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() {
  echo "[bernini-raw] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -d "${qwen_passed}" && ! -L "${qwen_passed}" ]] || fail "invalid Qwen passed root"
[[ -d "${wan_root}" && ! -L "${wan_root}" ]] || fail "invalid Wan root"
[[ -d "${natural_root}" && ! -L "${natural_root}" ]] || fail "invalid natural-label root"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${expected_rows}" =~ ^[1-9][0-9]*$ ]] || fail "expected rows must be positive"
[[ "${expected_strict_rows}" =~ ^[0-9]+$ ]] || fail "expected strict rows must be non-negative"
[[ "${expected_terminal_rows}" =~ ^[1-9][0-9]*$ ]] || fail "expected terminal rows must be positive"
[[ "${expected_error_rows}" =~ ^[0-9]+$ ]] || fail "expected error rows must be non-negative"
[[ "${expected_natural_summary_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid natural summary hash"
[[ "${expected_natural_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid natural manifest hash"
[[ "${expected_natural_iid_set_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid natural IID-set hash"
[[ "${expected_qwen_index_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid Qwen index hash"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "create-only output root exists"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

task_scratch="${SLURM_TMPDIR:-/tmp}/bernini-raw-${SLURM_JOB_ID}"
mkdir -p "${task_scratch}/source" "${output_root}"
tar -xf "${source_archive}" -C "${task_scratch}/source"
bernini_method="${task_scratch}/source/methods/bernini_action_editing"
omni_method="${task_scratch}/source/methods/omnivideo2_action_editing"
[[ -f "${omni_method}/tools/build_action_preview_manifest.py" ]] || fail "archive lacks preview joiner"
[[ -f "${bernini_method}/tools/build_renderer_dataset.py" ]] || fail "archive lacks raw adapter"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-raw] source_revision=${source_revision}"
echo "[bernini-raw] source_archive_sha256=${source_archive_sha256}"

"${python_bin}" "${omni_method}/tools/build_action_preview_manifest.py" \
  --qwen-passed-dir "${qwen_passed}" \
  --wan-root "${wan_root}" \
  --natural-root "${natural_root}" \
  --instruction-source natural \
  --selection-policy natural_release_all \
  --output-manifest "${output_root}/preview_manifest.jsonl" \
  --summary-output "${output_root}/preview_summary.json"

"${python_bin}" "${bernini_method}/tools/build_renderer_dataset.py" \
  --preview-manifest "${output_root}/preview_manifest.jsonl" \
  --output-parquet "${output_root}/raw.parquet" \
  --receipt-path "${output_root}/raw.receipt.json" \
  --acknowledge-preview-only \
  --acknowledge-broader-natural-release

"${python_bin}" - \
  "${output_root}" \
  "${expected_rows}" \
  "${expected_strict_rows}" \
  "${expected_terminal_rows}" \
  "${expected_error_rows}" \
  "${expected_natural_summary_sha256}" \
  "${expected_natural_manifest_sha256}" \
  "${expected_natural_iid_set_sha256}" \
  "${expected_qwen_index_sha256}" \
  "${source_revision}" \
  "${source_archive_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
expected_strict = int(sys.argv[3])
expected_terminal = int(sys.argv[4])
expected_errors = int(sys.argv[5])
preview = json.loads((root / "preview_summary.json").read_text(encoding="utf-8"))
raw = json.loads((root / "raw.receipt.json").read_text(encoding="utf-8"))
if preview.get("preview_rows") != expected or raw.get("sample_count") != expected:
    raise SystemExit(
        f"row count differs: preview={preview.get('preview_rows')} "
        f"raw={raw.get('sample_count')} expected={expected}"
    )
if (
    preview.get("selection_policy") != "natural_release_all"
    or preview.get("selection_eligible_rows") != expected
    or preview.get("strict_gate_eligible_rows") != expected_strict
    or preview.get("natural_release_ok_rows") != expected
    or preview.get("natural_release_expected_rows") != expected_terminal
    or preview.get("natural_release_error_rows") != expected_errors
    or preview.get("natural_release_summary_sha256") != sys.argv[6]
    or preview.get("natural_release_manifest_sha256") != sys.argv[7]
    or preview.get("natural_release_iid_set_sha256") != sys.argv[8]
    or preview.get("qwen_passed_index_sha256") != sys.argv[9]
    or raw.get("experimental_inclusion_policy") != "natural_release_all"
    or raw.get("broader_natural_release_inclusion_acknowledged") is not True
    or raw.get("strict_selection_rows") != expected_strict
    or raw.get("non_strict_selection_rows") != expected - expected_strict
):
    raise SystemExit("natural-release inclusion policy/counts differ")
if (
    preview.get("natural_release_manifest_sha256")
    != raw.get("natural_release", {}).get("manifest_sha256")
):
    raise SystemExit("natural release manifest binding differs")
if not (
    preview.get("preview_only") is True
    and preview.get("training_authorized") is False
    and raw.get("preview_only") is True
    and raw.get("training_authorized") is False
    and raw.get("production_claim_forbidden") is True
):
    raise SystemExit("authorization/claim state differs")
done = {
    "schema_version": "bernini-r-action-raw-job-done-v2",
    "complete": True,
    "sample_count": expected,
    "strict_selection_rows": expected_strict,
    "non_strict_selection_rows": expected - expected_strict,
    "experimental_inclusion_policy": "natural_release_all",
    "natural_release_manifest_sha256": preview[
        "natural_release_manifest_sha256"
    ],
    "natural_release_summary_sha256": sys.argv[6],
    "natural_release_iid_set_sha256": sys.argv[8],
    "qwen_passed_index_sha256": sys.argv[9],
    "source_revision": sys.argv[10],
    "source_archive_sha256": sys.argv[11],
    "preview_manifest_sha256": hashlib.sha256(
        (root / "preview_manifest.jsonl").read_bytes()
    ).hexdigest(),
    "raw_parquet_sha256": hashlib.sha256((root / "raw.parquet").read_bytes()).hexdigest(),
    "raw_receipt_sha256": hashlib.sha256(
        (root / "raw.receipt.json").read_bytes()
    ).hexdigest(),
    "preview_only": True,
    "production_claim_forbidden": True,
}
done["job_done_digest"] = hashlib.sha256(
    json.dumps(
        done,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
(root / "job_done.json").write_text(
    json.dumps(done, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(done, sort_keys=True))
PY

echo "[bernini-raw] PASS"
