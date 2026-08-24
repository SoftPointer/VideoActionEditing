#!/usr/bin/env bash
#
# Submit exactly one fresh Goku action-anchor v16 smoke allocation.
#
# This helper is intentionally fail closed.  A submission attempt leaves a
# durable pending/raw receipt before its job id is interpreted, so an
# interrupted caller cannot silently submit the same named run twice.

set -Eeuo pipefail
umask 077

die() {
  echo "[goku-v16-submit] $*" >&2
  exit 2
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    die "required environment variable is unset: ${name}"
  fi
}

require_absolute_nonroot() {
  local label="$1"
  local value="$2"
  if [[ "${value}" != /* || "${value}" == "/" ]]; then
    die "${label} must be a non-root absolute path: ${value}"
  fi
}

require_sha256() {
  local label="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    die "${label} is not a lowercase SHA-256 digest: ${value}"
  fi
}

require_env MOTIVE_V16_RUN_ROOT
require_env MOTIVE_V16_SOURCE_SNAPSHOT
require_env MOTIVE_V16_SOURCE_TREE_SHA256
require_env MOTIVE_V16_SELECTED
require_env MOTIVE_V16_SELECTED_SHA256
require_env MOTIVE_V16_SMOKE_GOLD
require_env MOTIVE_V16_SMOKE_GOLD_SHA256
require_env MOTIVE_V16_MODEL_CLOSURE
require_env MOTIVE_V16_MODEL_CLOSURE_SHA256
require_env MOTIVE_V16_QWEN_MODEL
require_env MOTIVE_V16_PYTHON_BIN
require_env MOTIVE_V16_SUBMISSION_CONTRACT
require_env MOTIVE_V16_SUBMISSION_CONTRACT_SHA256
require_env MOTIVE_V16_SBATCH_SHA256
require_env MOTIVE_V16_JOB_NAME

run_root="${MOTIVE_V16_RUN_ROOT}"
source_snapshot="${MOTIVE_V16_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_V16_SOURCE_TREE_SHA256}"
selected="${MOTIVE_V16_SELECTED}"
selected_sha256="${MOTIVE_V16_SELECTED_SHA256}"
smoke_gold="${MOTIVE_V16_SMOKE_GOLD}"
smoke_gold_sha256="${MOTIVE_V16_SMOKE_GOLD_SHA256}"
model_closure="${MOTIVE_V16_MODEL_CLOSURE}"
model_closure_sha256="${MOTIVE_V16_MODEL_CLOSURE_SHA256}"
qwen_model="${MOTIVE_V16_QWEN_MODEL}"
python_bin="${MOTIVE_V16_PYTHON_BIN}"
submission_contract="${MOTIVE_V16_SUBMISSION_CONTRACT}"
submission_contract_sha256="${MOTIVE_V16_SUBMISSION_CONTRACT_SHA256}"
sbatch_sha256="${MOTIVE_V16_SBATCH_SHA256}"
job_name="${MOTIVE_V16_JOB_NAME}"

for binding in \
  "run root:${run_root}" \
  "source snapshot:${source_snapshot}" \
  "selected input:${selected}" \
  "smoke gold:${smoke_gold}" \
  "model closure:${model_closure}" \
  "Qwen model:${qwen_model}" \
  "Python:${python_bin}" \
  "submission contract:${submission_contract}"; do
  require_absolute_nonroot "${binding%%:*}" "${binding#*:}"
done
require_sha256 "source tree SHA-256" "${source_tree_sha256}"
require_sha256 "selected SHA-256" "${selected_sha256}"
require_sha256 "smoke gold SHA-256" "${smoke_gold_sha256}"
require_sha256 "model closure SHA-256" "${model_closure_sha256}"
require_sha256 "submission contract SHA-256" "${submission_contract_sha256}"
require_sha256 "sbatch SHA-256" "${sbatch_sha256}"
if [[ "${smoke_gold_sha256}" != "b99972b81139e7a3193e6589efdf8de38075102cf14f312e3e1e73dfc3d626df" ]]; then
  die "smoke gold differs from the v16 source-level trust anchor"
fi
if [[ "${model_closure_sha256}" != "395236b156d85409ca40643683b47b1badb28602df0ef41e519e50f9a60f6c05" ]]; then
  die "model closure differs from the v16 source-level trust anchor"
fi
if [[ ! "${job_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  die "job name contains unsupported characters: ${job_name}"
fi

for directory in "${run_root}" "${source_snapshot}" "${qwen_model}"; do
  if [[ -L "${directory}" || ! -d "${directory}" ]]; then
    die "required directory is missing or a symlink: ${directory}"
  fi
  if [[ "$(readlink -f "${directory}")" != "${directory}" ]]; then
    die "directory is not canonical: ${directory}"
  fi
done
for file in \
  "${selected}" \
  "${smoke_gold}" \
  "${model_closure}" \
  "${submission_contract}" \
  "${python_bin}"; do
  if [[ -L "${file}" || ! -f "${file}" ]]; then
    die "required file is missing or a symlink: ${file}"
  fi
  if [[ "$(readlink -f "${file}")" != "${file}" ]]; then
    die "file is not canonical: ${file}"
  fi
done
expected_smoke_gold="${source_snapshot}/methods/motive/audits/goku_action_v16_smoke_gold.json"
if [[ "${smoke_gold}" != "${expected_smoke_gold}" ]]; then
  die "smoke gold must use its canonical source-snapshot path"
fi
expected_model_closure="${source_snapshot}/methods/motive/audits/qwen3_vl_32b_instruct_model_closure.json"
if [[ "${model_closure}" != "${expected_model_closure}" ]]; then
  die "model closure must use its canonical source-snapshot path"
fi
if [[ "${selected}" != "${run_root}/input/selected_smoke.jsonl" ]]; then
  die "selected input must use its canonical fresh-run path"
fi
if [[ "${submission_contract}" != "${run_root}/submission_contract.json" ]]; then
  die "submission contract must use its canonical fresh-run path"
fi
if [[ ! -x "${python_bin}" ]]; then
  die "Python is not executable: ${python_bin}"
fi

snapshot_verifier="${source_snapshot}/methods/motive/scripts/action_source_snapshot.py"
sbatch_script="${source_snapshot}/methods/motive/scripts/auh_goku_action_anchor_qwen.sbatch"
for file in "${snapshot_verifier}" "${sbatch_script}"; do
  if [[ -L "${file}" || ! -f "${file}" ]]; then
    die "snapshot executable is missing or a symlink: ${file}"
  fi
done

actual_selected_sha256="$(sha256sum "${selected}" | awk '{print $1}')"
actual_smoke_gold_sha256="$(sha256sum "${smoke_gold}" | awk '{print $1}')"
actual_model_closure_sha256="$(sha256sum "${model_closure}" | awk '{print $1}')"
actual_contract_sha256="$(sha256sum "${submission_contract}" | awk '{print $1}')"
actual_sbatch_sha256="$(sha256sum "${sbatch_script}" | awk '{print $1}')"
[[ "${actual_selected_sha256}" == "${selected_sha256}" ]] ||
  die "selected input SHA-256 differs"
[[ "${actual_smoke_gold_sha256}" == "${smoke_gold_sha256}" ]] ||
  die "smoke gold SHA-256 differs"
[[ "${actual_model_closure_sha256}" == "${model_closure_sha256}" ]] ||
  die "model closure SHA-256 differs"
[[ "${actual_contract_sha256}" == "${submission_contract_sha256}" ]] ||
  die "submission contract SHA-256 differs"
[[ "${actual_sbatch_sha256}" == "${sbatch_sha256}" ]] ||
  die "sbatch SHA-256 differs"

"${python_bin}" -c '
import json
import hashlib
import sys
from pathlib import Path

contract_path = Path(sys.argv[1])
value = json.loads(contract_path.read_text(encoding="utf-8"))
gold = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
model_closure = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
if gold.get("schema_version") != "goku-action-v16-smoke-gold-v1":
    raise SystemExit("smoke gold schema is not v16")
gold_selected = gold.get("selected_smoke")
if not isinstance(gold_selected, dict):
    raise SystemExit("smoke gold selected_smoke is missing")
if gold_selected.get("sha256") != sys.argv[3]:
    raise SystemExit("selected SHA differs from smoke gold")
if model_closure.get("schema_version") != "motive-qwen-model-closure-v1":
    raise SystemExit("model closure schema differs")
if model_closure.get("model_id") != "Qwen/Qwen3-VL-32B-Instruct":
    raise SystemExit("model closure model ID differs")
if model_closure.get("revision") != "Qwen3-VL-32B-Instruct":
    raise SystemExit("model closure revision differs")
if model_closure.get("file_count") != 54:
    raise SystemExit("model closure file count differs")
if model_closure.get("total_bytes") != 66726522473:
    raise SystemExit("model closure total bytes differs")
expected_keys = {
    "schema_version",
    "selected",
    "smoke_gold",
    "model_closure",
    "source_snapshot",
    "model",
    "runtime",
    "outputs",
}
if not isinstance(value, dict) or set(value) != expected_keys:
    raise SystemExit("submission contract is not closed v16")
if value["schema_version"] != "motive-goku-action-v16-submission-contract-v1":
    raise SystemExit("submission contract schema is not v16")
expected_selected = {
    "path": sys.argv[2],
    "sha256": sys.argv[3],
    "rows": gold_selected.get("rows"),
}
expected_gold = {"path": sys.argv[4], "sha256": sys.argv[5]}
expected_model_closure = {
    "path": sys.argv[6],
    "sha256": sys.argv[7],
    "file_count": model_closure.get("file_count"),
    "total_bytes": model_closure.get("total_bytes"),
}
if value["selected"] != expected_selected:
    raise SystemExit("submission selected binding differs")
if value["smoke_gold"] != expected_gold:
    raise SystemExit("submission smoke-gold binding differs")
if value["model_closure"] != expected_model_closure:
    raise SystemExit("submission model-closure binding differs")
if model_closure.get("model_path") != sys.argv[8]:
    raise SystemExit("model closure path differs from exported Qwen model")
closure_files = model_closure.get("files")
if not isinstance(closure_files, list):
    raise SystemExit("model closure files are missing")
config_bindings = [
    item
    for item in closure_files
    if isinstance(item, dict) and item.get("relative_path") == "config.json"
]
if len(config_bindings) != 1:
    raise SystemExit("model closure config binding is not unique")
closure_config_sha256 = config_bindings[0].get("sha256")
if (
    not isinstance(closure_config_sha256, str)
    or len(closure_config_sha256) != 64
    or any(character not in "0123456789abcdef"
           for character in closure_config_sha256)
):
    raise SystemExit("model closure config SHA-256 is invalid")
config_path = Path(sys.argv[8]) / "config.json"
expected_model = {
    "path": sys.argv[8],
    "config_path": str(config_path),
    "config_sha256": closure_config_sha256,
}
if value["model"] != expected_model:
    raise SystemExit("submission model binding differs")
if not config_path.is_file():
    raise SystemExit("exported Qwen model config is missing")
actual_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
if actual_config_sha256 != closure_config_sha256:
    raise SystemExit("exported Qwen model config SHA-256 differs")
if value["source_snapshot"].get("path") != sys.argv[9]:
    raise SystemExit("submission source-snapshot path differs")
if value["source_snapshot"].get("tree_sha256") != sys.argv[10]:
    raise SystemExit("submission source-tree binding differs")
if "approval_path" in value["runtime"]:
    raise SystemExit("submission exposes a forbidden approval interface")
expected_runtime = {
    "num_shards": 8,
    "max_samples": None,
    "max_new_tokens": 1536,
    "nframes": 12,
    "max_pixels": 589824,
    "attn_implementation": "sdpa",
    "allow_download": False,
    "repair_attempts": 1,
    "final_seed": 260730,
    "allow_partial": True,
}
canonical = lambda item: json.dumps(
    item,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
if canonical(value["runtime"]) != canonical(expected_runtime):
    raise SystemExit("submission runtime differs from frozen smoke")
if value["outputs"] != {
    "qwen_root": sys.argv[11] + "/qwen8",
    "final_output": sys.argv[11] + "/final",
}:
    raise SystemExit("submission output binding differs")
' \
  "${submission_contract}" \
  "${selected}" \
  "${selected_sha256}" \
  "${smoke_gold}" \
  "${smoke_gold_sha256}" \
  "${model_closure}" \
  "${model_closure_sha256}" \
  "${qwen_model}" \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${run_root}" ||
  die "submission contract failed strict v16 binding validation"

"${python_bin}" "${snapshot_verifier}" verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}" >/dev/null

qwen_root="${run_root}/qwen8"
final_output="${run_root}/final"
logs_dir="${run_root}/logs"
if [[ ! -d "${logs_dir}" || -L "${logs_dir}" ]]; then
  die "fresh run must contain one real logs directory: ${logs_dir}"
fi

for forbidden in \
  "${qwen_root}" \
  "${final_output}" \
  "${run_root}/jobs.tsv" \
  "${run_root}/qwen_submission.raw" \
  "${run_root}/qwen_submission.raw.pending" \
  "${run_root}/submission_intent.env" \
  "${run_root}/completion_receipt.json" \
  "${run_root}/acceptance_contract.json" \
  "${run_root}/acceptance_result.json"; do
  if [[ -e "${forbidden}" || -L "${forbidden}" ]]; then
    die "refusing pre-existing run artifact: ${forbidden}"
  fi
done

# The timestamped job name is a durable idempotency key.  Refuse both active
# jobs and accounting history.  A failed accounting lookup is itself unsafe.
active_name_rows="$(
  timeout 20s squeue --me -h -n "${job_name}" -o '%i'
)" || die "could not audit active Slurm jobs named ${job_name}"
if [[ -n "${active_name_rows//[[:space:]]/}" ]]; then
  die "an active job already uses job name ${job_name}"
fi
active_work_rows="$(
  timeout 20s squeue --me -h -o '%i|%Z'
)" || die "could not audit active Slurm work directories"
while IFS='|' read -r active_job_id active_work_dir; do
  [[ -z "${active_job_id}" ]] && continue
  if [[ "${active_work_dir}" == "${run_root}" ]]; then
    die "active job ${active_job_id} already uses run root ${run_root}"
  fi
done <<<"${active_work_rows}"
accounting_rows="$(
  timeout 20s sacct -X -n --name "${job_name}" --starttime 2026-07-30 \
    --format=JobIDRaw 2>/dev/null
)" || die "could not audit Slurm accounting for ${job_name}"
if [[ -n "${accounting_rows//[[:space:]]/}" ]]; then
  die "Slurm accounting already contains job name ${job_name}"
fi

{
  printf 'schema=motive-goku-action-v16-submission-intent-v1\n'
  printf 'job_name=%s\n' "${job_name}"
  printf 'submission_contract=%s\n' "${submission_contract}"
  printf 'submission_contract_sha256=%s\n' "${submission_contract_sha256}"
  printf 'selected=%s\n' "${selected}"
  printf 'selected_sha256=%s\n' "${selected_sha256}"
  printf 'smoke_gold=%s\n' "${smoke_gold}"
  printf 'smoke_gold_sha256=%s\n' "${smoke_gold_sha256}"
  printf 'model_closure=%s\n' "${model_closure}"
  printf 'model_closure_sha256=%s\n' "${model_closure_sha256}"
  printf 'source_snapshot=%s\n' "${source_snapshot}"
  printf 'source_tree_sha256=%s\n' "${source_tree_sha256}"
  printf 'qwen_root=%s\n' "${qwen_root}"
  printf 'final_output=%s\n' "${final_output}"
  printf 'generation_authorized=false\n'
  printf 'wan_generation_authorized=false\n'
  printf 'production_eligible=false\n'
  printf 'authorization_interface_available=false\n'
} >"${run_root}/submission_intent.env"
chmod 0400 "${run_root}/submission_intent.env"

export_string="$(
  printf '%s' \
    "PATH=/usr/bin:/bin," \
    "SLURM_EXPORT_ENV=ALL," \
    "MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT=${source_snapshot}," \
    "MOTIVE_GOKU_ACTION_SELECTED=${selected}," \
    "MOTIVE_GOKU_ACTION_QWEN_MODEL=${qwen_model}," \
    "MOTIVE_GOKU_ACTION_QWEN_OUTPUT=${qwen_root}," \
    "MOTIVE_GOKU_ACTION_FINAL_OUTPUT=${final_output}," \
    "MOTIVE_GOKU_ACTION_PYTHON_BIN=${python_bin}," \
    "MOTIVE_GOKU_ACTION_QWEN_NFRAMES=12," \
    "MOTIVE_GOKU_ACTION_QWEN_MAX_PIXELS=589824," \
    "MOTIVE_GOKU_ACTION_QWEN_MAX_NEW_TOKENS=1536," \
    "MOTIVE_GOKU_ACTION_FINAL_SEED=260730," \
    "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL=1"
)"

# The batch job itself starts from the explicit minimal environment above.
# SLURM_EXPORT_ENV=ALL only asks nested srun steps to inherit exports created
# inside that already-sanitized job (not the caller's login environment).
#
# This is the only sbatch invocation in the helper.  Redirection creates the
# pending receipt before sbatch starts; interruption therefore leaves a
# durable artifact that blocks a second attempt.
set +e
sbatch \
  --parsable \
  --job-name="${job_name}" \
  --chdir="${run_root}" \
  --output="${logs_dir}/qwen-%j.out" \
  --error="${logs_dir}/qwen-%j.err" \
  --export="${export_string}" \
  "${sbatch_script}" >"${run_root}/qwen_submission.raw.pending" 2>&1
submit_status=$?
set -e
chmod 0400 "${run_root}/qwen_submission.raw.pending"
if (( submit_status != 0 )); then
  die "sbatch failed; preserved qwen_submission.raw.pending"
fi

submission_raw="$(
  tr -d '\r\n' <"${run_root}/qwen_submission.raw.pending"
)"
job_id="${submission_raw%%;*}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  die "sbatch returned an invalid job id; preserved pending receipt"
fi
mv "${run_root}/qwen_submission.raw.pending" \
  "${run_root}/qwen_submission.raw"

{
  printf 'job_id\tjob_name\tstage\tstatus\tsubmission_contract_sha256\n'
  printf '%s\t%s\tqwen8_and_finalize\tsubmitted\t%s\n' \
    "${job_id}" "${job_name}" "${submission_contract_sha256}"
} >"${run_root}/jobs.tsv"
chmod 0400 "${run_root}/qwen_submission.raw" "${run_root}/jobs.tsv"

echo "[goku-v16-submit] submitted job_id=${job_id} job_name=${job_name}"
