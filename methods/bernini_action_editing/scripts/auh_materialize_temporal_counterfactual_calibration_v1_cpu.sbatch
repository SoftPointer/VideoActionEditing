#!/usr/bin/env bash
#SBATCH --job-name=temporal-cf-cal-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# AUH's available faculty QOSes require at least one GPU GRES even for a
# receipt-only job.  The device is deliberately hidden below and never used.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=01:00:00

set -Eeuo pipefail

# CPU receipt-only postflight for temporal scorer job 131237.  It is designed
# for a Slurm ``dependency=afterok:131237`` after both group receipts exist;
# file hashes are either externally presealed or explicitly recorded at runtime.
# One scheduler-required GPU is allocated but hidden; no model, media, tensor
# artifact, optimizer, training process, or accelerator computation is used.

method_root="${TEMPORAL_CF_CAL_METHOD_ROOT:?set TEMPORAL_CF_CAL_METHOD_ROOT}"
python_bin="${TEMPORAL_CF_CAL_PYTHON_BIN:?set TEMPORAL_CF_CAL_PYTHON_BIN}"
root_spec="${TEMPORAL_CF_CAL_ROOT_SPEC:?set TEMPORAL_CF_CAL_ROOT_SPEC}"
root_spec_sha256="${TEMPORAL_CF_CAL_ROOT_SPEC_SHA256:?set TEMPORAL_CF_CAL_ROOT_SPEC_SHA256}"
bank_receipt="${TEMPORAL_CF_CAL_BANK_RECEIPT:?set TEMPORAL_CF_CAL_BANK_RECEIPT}"
bank_receipt_sha256="${TEMPORAL_CF_CAL_BANK_RECEIPT_SHA256:?set TEMPORAL_CF_CAL_BANK_RECEIPT_SHA256}"
score_root="${TEMPORAL_CF_CAL_SCORE_ROOT:?set TEMPORAL_CF_CAL_SCORE_ROOT}"
sp4_a_group_sha256="${TEMPORAL_CF_CAL_SP4_A_GROUP_SHA256:-}"
sp4_b_group_sha256="${TEMPORAL_CF_CAL_SP4_B_GROUP_SHA256:-}"
audit_bridge_receipt="${TEMPORAL_CF_CAL_AUDIT_BRIDGE_RECEIPT:?set TEMPORAL_CF_CAL_AUDIT_BRIDGE_RECEIPT}"
audit_bridge_sha256="${TEMPORAL_CF_CAL_AUDIT_BRIDGE_SHA256:?set TEMPORAL_CF_CAL_AUDIT_BRIDGE_SHA256}"
event_audit_root="${TEMPORAL_CF_CAL_EVENT_AUDIT_ROOT:?set TEMPORAL_CF_CAL_EVENT_AUDIT_ROOT}"
label_manifest="${TEMPORAL_CF_CAL_LABEL_MANIFEST:?set TEMPORAL_CF_CAL_LABEL_MANIFEST}"
label_manifest_sha256="${TEMPORAL_CF_CAL_LABEL_MANIFEST_SHA256:?set TEMPORAL_CF_CAL_LABEL_MANIFEST_SHA256}"
preregistration="${TEMPORAL_CF_CAL_PREREGISTRATION:?set TEMPORAL_CF_CAL_PREREGISTRATION}"
preregistration_sha256="${TEMPORAL_CF_CAL_PREREGISTRATION_SHA256:?set TEMPORAL_CF_CAL_PREREGISTRATION_SHA256}"
calibrator_archive="${TEMPORAL_CF_CAL_CALIBRATOR_ARCHIVE:?set TEMPORAL_CF_CAL_CALIBRATOR_ARCHIVE}"
calibrator_revision="${TEMPORAL_CF_CAL_CALIBRATOR_REVISION:?set TEMPORAL_CF_CAL_CALIBRATOR_REVISION}"
calibrator_archive_sha256="${TEMPORAL_CF_CAL_CALIBRATOR_ARCHIVE_SHA256:?set TEMPORAL_CF_CAL_CALIBRATOR_ARCHIVE_SHA256}"
output_dir="${TEMPORAL_CF_CAL_OUTPUT_DIR:?set TEMPORAL_CF_CAL_OUTPUT_DIR}"
acknowledgement="${TEMPORAL_CF_CAL_ACKNOWLEDGEMENT:?set TEMPORAL_CF_CAL_ACKNOWLEDGEMENT}"

materializer_source_sha256="${TEMPORAL_CF_CAL_MATERIALIZER_SOURCE_SHA256:?set TEMPORAL_CF_CAL_MATERIALIZER_SOURCE_SHA256}"

readonly required_root_spec_sha256=a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95
readonly required_bank_receipt_sha256=8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59
readonly required_audit_bridge_sha256=5401ca7075ef1a5818251d9fbdad150131eb770f0822ef93666e2d5677d7b45f
readonly required_label_manifest_sha256=9246504e97e1ee46c2cdcf7dfac0f41364dca40f26e5c26f28f0968d0443808d
readonly required_preregistration_sha256=d6df656bfe65aff6966d68bb002ecff071320b4af12e526564781771cb74bd25
readonly required_calibrator_revision=f7ffa5921026b991b0b74592ad9bdce6e81bf76d
readonly required_calibrator_archive_sha256=55a7ae5f4555b290a17339e6bd09c2f6fff39962535ce012f0581b7f63ded8f2
readonly required_calibrator_source_sha256=18fbcb381c894188832423d7af5099e4dc299c05bcf99215b9bf725979e918e1
readonly required_contract_source_sha256=7a2edf4b8a04a11ba35358c833aa7feafbf1c137134cde5e018caf152cdc0dd7
readonly required_scorer_source_sha256=fe3b67d7540da838689782113fed307e4b1045cf6162d26f170445a1bc81293e
readonly required_event_validator_source_sha256=1bd1c5fd2f41c28093c3209eef278200305c06f464ec8a613e95551c8642fac1
readonly required_audit_bridge_source_sha256=4d94e022b3ad0917731df39ab92db9806a3feb81bb491b3107b8ff9544296618
readonly required_label_author_source_sha256=6ae3c0198d828f4a18fd90f4d3e5f60509edf95676efc4845a7aeeb83a72c25b

fail() { echo "[temporal-cf-cal-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be lowercase SHA-1"; }

for name in method_root python_bin root_spec bank_receipt score_root audit_bridge_receipt event_audit_root label_manifest preregistration calibrator_archive output_dir; do
  require_absolute "${!name}" "${name}"
done
for name in root_spec_sha256 bank_receipt_sha256 audit_bridge_sha256 label_manifest_sha256 preregistration_sha256 calibrator_archive_sha256 materializer_source_sha256; do
  require_sha256 "${!name}" "${name}"
done
for name in sp4_a_group_sha256 sp4_b_group_sha256; do
  [[ -z "${!name}" ]] || require_sha256 "${!name}" "${name}"
done
require_sha1 "${calibrator_revision}" "calibrator_revision"
[[ "${root_spec_sha256}" == "${required_root_spec_sha256}" ]] || fail "root spec authority differs"
[[ "${bank_receipt_sha256}" == "${required_bank_receipt_sha256}" ]] || fail "bank receipt authority differs"
[[ "${audit_bridge_sha256}" == "${required_audit_bridge_sha256}" ]] || fail "audit bridge authority differs"
[[ "${label_manifest_sha256}" == "${required_label_manifest_sha256}" ]] || fail "label manifest authority differs"
[[ "${preregistration_sha256}" == "${required_preregistration_sha256}" ]] || fail "preregistration authority differs"
[[ "${calibrator_revision}" == "${required_calibrator_revision}" ]] || fail "calibrator revision differs"
[[ "${calibrator_archive_sha256}" == "${required_calibrator_archive_sha256}" ]] || fail "calibrator archive authority differs"
[[ "${acknowledgement}" == "I_REVIEWED_THE_SEALED_40_ROW_DETACHED_LABEL_MANIFEST" ]] || fail "reviewed-label acknowledgement differs"

method_root="$(realpath -e -- "${method_root}")"
python_bin="$(realpath -e -- "${python_bin}")"
root_spec="$(realpath -e -- "${root_spec}")"
bank_receipt="$(realpath -e -- "${bank_receipt}")"
score_root="$(realpath -e -- "${score_root}")"
audit_bridge_receipt="$(realpath -e -- "${audit_bridge_receipt}")"
event_audit_root="$(realpath -e -- "${event_audit_root}")"
label_manifest="$(realpath -e -- "${label_manifest}")"
preregistration="$(realpath -e -- "${preregistration}")"
calibrator_archive="$(realpath -e -- "${calibrator_archive}")"

[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "method root differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for name in root_spec bank_receipt audit_bridge_receipt label_manifest preregistration calibrator_archive; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in score_root event_audit_root; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output directory must be fresh"
[[ -d "$(dirname -- "${output_dir}")" && ! -L "$(dirname -- "${output_dir}")" ]] || fail "output parent differs"

materializer_source="${method_root}/materialize_temporal_counterfactual_calibration_v1.py"
calibrator_source="${method_root}/temporal_counterfactual_calibration_v1.py"
contract_source="${method_root}/temporal_counterfactual_contract_v1.py"
scorer_source="${method_root}/temporal_counterfactual_action_scorer_v1.py"
event_validator_source="${method_root}/pair_v5_t2v_energy_calibration_v3.py"
audit_bridge_source="${method_root}/materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3.py"
label_author_source="${method_root}/tools/author_pair_v5_core4_event_labels_d541801_v3.py"
for name in materializer_source calibrator_source contract_source scorer_source event_validator_source audit_bridge_source label_author_source; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "required source differs: ${!name}"
done
[[ "$(sha256sum "${materializer_source}" | awk '{print $1}')" == "${materializer_source_sha256}" ]] || fail "materializer source changed"
[[ "$(sha256sum "${calibrator_source}" | awk '{print $1}')" == "${required_calibrator_source_sha256}" ]] || fail "calibrator source changed"
[[ "$(sha256sum "${contract_source}" | awk '{print $1}')" == "${required_contract_source_sha256}" ]] || fail "contract source changed"
[[ "$(sha256sum "${scorer_source}" | awk '{print $1}')" == "${required_scorer_source_sha256}" ]] || fail "scorer source changed"
[[ "$(sha256sum "${event_validator_source}" | awk '{print $1}')" == "${required_event_validator_source_sha256}" ]] || fail "event validator source changed"
[[ "$(sha256sum "${audit_bridge_source}" | awk '{print $1}')" == "${required_audit_bridge_source_sha256}" ]] || fail "audit bridge source changed"
[[ "$(sha256sum "${label_author_source}" | awk '{print $1}')" == "${required_label_author_source_sha256}" ]] || fail "label author source changed"

[[ "$(sha256sum "${root_spec}" | awk '{print $1}')" == "${root_spec_sha256}" ]] || fail "root spec hash differs"
[[ "$(sha256sum "${bank_receipt}" | awk '{print $1}')" == "${bank_receipt_sha256}" ]] || fail "bank receipt hash differs"
[[ "$(sha256sum "${audit_bridge_receipt}" | awk '{print $1}')" == "${audit_bridge_sha256}" ]] || fail "audit bridge hash differs"
[[ "$(sha256sum "${label_manifest}" | awk '{print $1}')" == "${label_manifest_sha256}" ]] || fail "label manifest hash differs"
[[ "$(sha256sum "${preregistration}" | awk '{print $1}')" == "${preregistration_sha256}" ]] || fail "preregistration hash differs"
[[ "$(sha256sum "${calibrator_archive}" | awk '{print $1}')" == "${calibrator_archive_sha256}" ]] || fail "calibrator archive hash differs"
[[ "$(git get-tar-commit-id <"${calibrator_archive}")" == "${calibrator_revision}" ]] || fail "calibrator archive git revision differs"

sp4_a_group="${score_root}/sp4-a/temporal-counterfactual-action-score-sp4-a-v1.json"
sp4_b_group="${score_root}/sp4-b/temporal-counterfactual-action-score-sp4-b-v1.json"
[[ -f "${sp4_a_group}" && ! -L "${sp4_a_group}" ]] || fail "sp4-a group receipt differs"
[[ -f "${sp4_b_group}" && ! -L "${sp4_b_group}" ]] || fail "sp4-b group receipt differs"
score_group_hash_authority=presealed_external
if [[ -z "${sp4_a_group_sha256}" && -z "${sp4_b_group_sha256}" ]]; then
  sp4_a_group_sha256="$(sha256sum "${sp4_a_group}" | awk '{print $1}')"
  sp4_b_group_sha256="$(sha256sum "${sp4_b_group}" | awk '{print $1}')"
  score_group_hash_authority=observed_afterok_runtime
  echo "[temporal-cf-cal-v1] OBSERVED_AFTEROK group=sp4-a file_sha256=${sp4_a_group_sha256}"
  echo "[temporal-cf-cal-v1] OBSERVED_AFTEROK group=sp4-b file_sha256=${sp4_b_group_sha256}"
elif [[ -z "${sp4_a_group_sha256}" || -z "${sp4_b_group_sha256}" ]]; then
  fail "group hashes must be both presealed or both omitted"
fi
[[ "$(sha256sum "${sp4_a_group}" | awk '{print $1}')" == "${sp4_a_group_sha256}" ]] || fail "sp4-a group hash differs"
[[ "$(sha256sum "${sp4_b_group}" | awk '{print $1}')" == "${sp4_b_group_sha256}" ]] || fail "sp4-b group hash differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${method_root}:${method_root}/tools"

"${python_bin}" -B "${materializer_source}" \
  --root-spec "${root_spec}" \
  --expected-root-spec-sha256 "${root_spec_sha256}" \
  --bank-receipt "${bank_receipt}" \
  --expected-bank-receipt-sha256 "${bank_receipt_sha256}" \
  --score-root "${score_root}" \
  --expected-sp4-a-score-group-sha256 "${sp4_a_group_sha256}" \
  --expected-sp4-b-score-group-sha256 "${sp4_b_group_sha256}" \
  --score-group-file-sha256-authority "${score_group_hash_authority}" \
  --audit-bridge-receipt "${audit_bridge_receipt}" \
  --expected-audit-bridge-sha256 "${audit_bridge_sha256}" \
  --event-audit-root "${event_audit_root}" \
  --detached-label-manifest "${label_manifest}" \
  --expected-detached-label-manifest-sha256 "${label_manifest_sha256}" \
  --preregistration "${preregistration}" \
  --expected-preregistration-sha256 "${preregistration_sha256}" \
  --calibrator-source-archive "${calibrator_archive}" \
  --calibrator-source-revision "${calibrator_revision}" \
  --calibrator-source-archive-sha256 "${calibrator_archive_sha256}" \
  --expected-calibrator-source-sha256 "${required_calibrator_source_sha256}" \
  --expected-materializer-source-sha256 "${materializer_source_sha256}" \
  --output-dir "${output_dir}" \
  --acknowledge-reviewed-label-manifest

# Independent-process proof: re-open all 40 scores, all 40 detached audits,
# both groups, the preregistration, and the three output files; deterministically
# reproduce calibration and require byte-semantic dictionary equality.
"${python_bin}" -B "${materializer_source}" \
  --root-spec "${root_spec}" \
  --expected-root-spec-sha256 "${root_spec_sha256}" \
  --bank-receipt "${bank_receipt}" \
  --expected-bank-receipt-sha256 "${bank_receipt_sha256}" \
  --score-root "${score_root}" \
  --expected-sp4-a-score-group-sha256 "${sp4_a_group_sha256}" \
  --expected-sp4-b-score-group-sha256 "${sp4_b_group_sha256}" \
  --score-group-file-sha256-authority "${score_group_hash_authority}" \
  --audit-bridge-receipt "${audit_bridge_receipt}" \
  --expected-audit-bridge-sha256 "${audit_bridge_sha256}" \
  --event-audit-root "${event_audit_root}" \
  --detached-label-manifest "${label_manifest}" \
  --expected-detached-label-manifest-sha256 "${label_manifest_sha256}" \
  --preregistration "${preregistration}" \
  --expected-preregistration-sha256 "${preregistration_sha256}" \
  --calibrator-source-archive "${calibrator_archive}" \
  --calibrator-source-revision "${calibrator_revision}" \
  --calibrator-source-archive-sha256 "${calibrator_archive_sha256}" \
  --expected-calibrator-source-sha256 "${required_calibrator_source_sha256}" \
  --expected-materializer-source-sha256 "${materializer_source_sha256}" \
  --output-dir "${output_dir}" \
  --acknowledge-reviewed-label-manifest \
  --verify-existing-output

"${python_bin}" -B - "${output_dir}" <<'PY'
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "temporal-counterfactual-preregistration-v1.json",
    "temporal-counterfactual-calibration-v1.json",
    "temporal-counterfactual-calibration-materialization-v1-receipt.json",
}
if {entry.name for entry in root.iterdir()} != expected or any(entry.is_symlink() for entry in root.iterdir()):
    raise SystemExit("materialized output closure differs")
receipt_path = root / "temporal-counterfactual-calibration-materialization-v1-receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="ascii"))
calibration = json.loads((root / "temporal-counterfactual-calibration-v1.json").read_text(encoding="ascii"))
if (
    receipt.get("candidate_count") != 40
    or type(receipt.get("optimizer_authorized")) is not bool
    or receipt.get("optimizer_authorized") is not calibration.get("optimizer_authorized")
    or receipt.get("failure_reasons") != calibration.get("failure_reasons")
    or receipt.get("training_performed") is not False
    or receipt.get("confirmation_samples_consumed_by_optimizer") is not False
    or receipt.get("scientific_action_editing_claim") is not False
):
    raise SystemExit("materialized receipt summary differs")
print(json.dumps({
    "materialization_file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "materialization_receipt_digest": receipt["receipt_digest"],
    "optimizer_authorized": receipt["optimizer_authorized"],
    "failure_reasons": receipt["failure_reasons"],
    "candidate_count": 40,
    "training_performed": False,
}, sort_keys=True))
PY

echo "[temporal-cf-cal-v1] PASS exact40_replay=true output=${output_dir}"
