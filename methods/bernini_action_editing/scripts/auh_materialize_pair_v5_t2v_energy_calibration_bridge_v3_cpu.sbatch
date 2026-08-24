#!/usr/bin/env bash
#SBATCH --job-name=pair5-cal-bridge-v3
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail

# Receipt/scalar-only postflight.  No GPU is requested and no model is loaded.
method_root="${PAIR_V5_CAL_BRIDGE_METHOD_ROOT:?set PAIR_V5_CAL_BRIDGE_METHOD_ROOT}"
python_bin="${PAIR_V5_CAL_BRIDGE_PYTHON_BIN:?set PAIR_V5_CAL_BRIDGE_PYTHON_BIN}"
root_spec="${PAIR_V5_CAL_BRIDGE_ROOT_SPEC:?set PAIR_V5_CAL_BRIDGE_ROOT_SPEC}"
root_spec_sha256="${PAIR_V5_CAL_BRIDGE_ROOT_SPEC_SHA256:?set PAIR_V5_CAL_BRIDGE_ROOT_SPEC_SHA256}"
bank_output="${PAIR_V5_CAL_BRIDGE_BANK_OUTPUT_DIR:?set PAIR_V5_CAL_BRIDGE_BANK_OUTPUT_DIR}"
bank_receipt="${PAIR_V5_CAL_BRIDGE_BANK_RECEIPT:?set PAIR_V5_CAL_BRIDGE_BANK_RECEIPT}"
bank_receipt_sha256="${PAIR_V5_CAL_BRIDGE_BANK_RECEIPT_SHA256:?set PAIR_V5_CAL_BRIDGE_BANK_RECEIPT_SHA256}"
score_root="${PAIR_V5_CAL_BRIDGE_SCORE_ROOT:?set PAIR_V5_CAL_BRIDGE_SCORE_ROOT}"
sp4_a_group_sha256="${PAIR_V5_CAL_BRIDGE_SP4_A_GROUP_SHA256:?set PAIR_V5_CAL_BRIDGE_SP4_A_GROUP_SHA256}"
sp4_b_group_sha256="${PAIR_V5_CAL_BRIDGE_SP4_B_GROUP_SHA256:?set PAIR_V5_CAL_BRIDGE_SP4_B_GROUP_SHA256}"
label_manifest="${PAIR_V5_CAL_BRIDGE_LABEL_MANIFEST:?set PAIR_V5_CAL_BRIDGE_LABEL_MANIFEST}"
label_manifest_sha256="${PAIR_V5_CAL_BRIDGE_LABEL_MANIFEST_SHA256:?set PAIR_V5_CAL_BRIDGE_LABEL_MANIFEST_SHA256}"
output_dir="${PAIR_V5_CAL_BRIDGE_OUTPUT_DIR:?set PAIR_V5_CAL_BRIDGE_OUTPUT_DIR}"
acknowledgement="${PAIR_V5_CAL_BRIDGE_ACKNOWLEDGEMENT:?set PAIR_V5_CAL_BRIDGE_ACKNOWLEDGEMENT}"

bridge_source_sha256="${PAIR_V5_CAL_BRIDGE_SOURCE_SHA256:?set PAIR_V5_CAL_BRIDGE_SOURCE_SHA256}"
author_source_sha256="${PAIR_V5_CAL_BRIDGE_AUTHOR_SOURCE_SHA256:?set PAIR_V5_CAL_BRIDGE_AUTHOR_SOURCE_SHA256}"
calibrator_source_sha256="${PAIR_V5_CAL_BRIDGE_CALIBRATOR_SOURCE_SHA256:?set PAIR_V5_CAL_BRIDGE_CALIBRATOR_SOURCE_SHA256}"
scorer_source_sha256="${PAIR_V5_CAL_BRIDGE_SCORER_SOURCE_SHA256:?set PAIR_V5_CAL_BRIDGE_SCORER_SOURCE_SHA256}"

fail() { echo "[pair5-cal-bridge-v3] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

for name in method_root python_bin root_spec bank_output bank_receipt score_root label_manifest output_dir; do
  require_absolute "${!name}" "${name}"
done
for name in root_spec_sha256 bank_receipt_sha256 sp4_a_group_sha256 sp4_b_group_sha256 label_manifest_sha256 bridge_source_sha256 author_source_sha256 calibrator_source_sha256 scorer_source_sha256; do
  require_sha256 "${!name}" "${name}"
done
[[ "${acknowledgement}" == "I_REVIEWED_THE_SEALED_40_ROW_DETACHED_LABEL_MANIFEST" ]] || \
  fail "explicit reviewed-label acknowledgement differs"

method_root="$(realpath -e -- "${method_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
root_spec="$(realpath -e -- "${root_spec}")"
bank_output="$(realpath -e -- "${bank_output}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
score_root="$(realpath -e -- "${score_root}")"
label_manifest="$(realpath -e -- "${label_manifest}")"

[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -f "${root_spec}" && ! -L "${root_spec}" ]] || fail "root spec differs"
[[ -d "${bank_output}" && ! -L "${bank_output}" ]] || fail "bank output differs"
[[ -f "${bank_receipt}" && ! -L "${bank_receipt}" ]] || fail "bank receipt differs"
[[ "$(dirname -- "${bank_receipt}")" == "${bank_output}" ]] || fail "bank receipt is outside bank output"
[[ -d "${score_root}" && ! -L "${score_root}" ]] || fail "score root differs"
[[ -f "${label_manifest}" && ! -L "${label_manifest}" ]] || fail "label manifest differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output directory must be fresh"
[[ -d "$(dirname -- "${output_dir}")" ]] || fail "output parent is absent"

bridge_source="${method_root}/materialize_pair_v5_t2v_energy_calibration_bridge_v3.py"
author_source="${method_root}/tools/author_pair_v5_core4_event_labels_v3.py"
calibrator_source="${method_root}/pair_v5_t2v_energy_calibration_v3.py"
scorer_source="${method_root}/score_pair_v5_t2v_energy_bank_v3.py"
for source in "${bridge_source}" "${author_source}" "${calibrator_source}" "${scorer_source}"; do
  [[ -f "${source}" && ! -L "${source}" ]] || fail "required method source differs: ${source}"
done
[[ "$(sha256sum "${bridge_source}" | awk '{print $1}')" == "${bridge_source_sha256}" ]] || fail "bridge source hash differs"
[[ "$(sha256sum "${author_source}" | awk '{print $1}')" == "${author_source_sha256}" ]] || fail "author source hash differs"
[[ "$(sha256sum "${calibrator_source}" | awk '{print $1}')" == "${calibrator_source_sha256}" ]] || fail "calibrator source hash differs"
[[ "$(sha256sum "${scorer_source}" | awk '{print $1}')" == "${scorer_source_sha256}" ]] || fail "scorer source hash differs"
[[ "$(sha256sum "${root_spec}" | awk '{print $1}')" == "${root_spec_sha256}" ]] || fail "root spec hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${label_manifest}" | awk '{print $1}')" == "${label_manifest_sha256}" ]] || fail "label manifest hash differs"

sp4_a_group="${score_root}/sp4-a/pair-v5-t2v-global-energy-sp4-a-v4.json"
sp4_b_group="${score_root}/sp4-b/pair-v5-t2v-global-energy-sp4-b-v4.json"
[[ -f "${sp4_a_group}" && ! -L "${sp4_a_group}" ]] || fail "sp4-a group receipt differs"
[[ -f "${sp4_b_group}" && ! -L "${sp4_b_group}" ]] || fail "sp4-b group receipt differs"
[[ "$(sha256sum "${sp4_a_group}" | awk '{print $1}')" == "${sp4_a_group_sha256}" ]] || fail "sp4-a group hash differs"
[[ "$(sha256sum "${sp4_b_group}" | awk '{print $1}')" == "${sp4_b_group_sha256}" ]] || fail "sp4-b group hash differs"

export ROCR_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${method_root}:${method_root}/tools"

"${python_bin}" -B "${bridge_source}" \
  --root-spec "${root_spec}" \
  --expected-root-spec-sha256 "${root_spec_sha256}" \
  --bank-output-dir "${bank_output}" \
  --bank-receipt "${bank_receipt}" \
  --expected-bank-receipt-sha256 "${bank_receipt_sha256}" \
  --score-root "${score_root}" \
  --expected-sp4-a-score-group-sha256 "${sp4_a_group_sha256}" \
  --expected-sp4-b-score-group-sha256 "${sp4_b_group_sha256}" \
  --detached-label-manifest "${label_manifest}" \
  --expected-detached-label-manifest-sha256 "${label_manifest_sha256}" \
  --output-dir "${output_dir}" \
  --acknowledge-reviewed-label-manifest

"${python_bin}" -B - "${output_dir}" <<'PY'
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
receipt_path = root / "pair-v5-t2v-energy-calibration-bridge-receipt-v3.json"
if not receipt_path.is_file() or receipt_path.is_symlink():
    raise SystemExit("bridge receipt is absent")
receipt = json.loads(receipt_path.read_text(encoding="ascii"))
if (
    receipt.get("candidate_count") != 40
    or len(list((root / "event_audits").glob("*.json"))) != 40
    or len(list((root / "score_rows").glob("*.json"))) != 40
    or receipt.get("confirmation_samples_consumed_by_optimizer") is not False
    or receipt.get("training_performed") is not False
    or receipt.get("scientific_action_editing_claim") is not False
):
    raise SystemExit("bridge output closure differs")
print(
    json.dumps(
        {
            "bridge_receipt_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "bridge_receipt_digest": receipt["receipt_digest"],
            "optimizer_authorized": receipt["optimizer_authorized"],
            "candidate_count": 40,
        },
        sort_keys=True,
    )
)
PY

echo "[pair5-cal-bridge-v3] PASS output=${output_dir}"
