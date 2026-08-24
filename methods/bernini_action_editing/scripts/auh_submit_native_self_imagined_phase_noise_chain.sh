#!/usr/bin/env bash
set -Eeuo pipefail

# Login-side submission of the one-step dual-WORLD4 engineering canary only.
# Exact40 is deliberately not chained: the current proposal semantics,
# source-only factorial arm, and pairwise intervention-strength gates are not
# closed.  A successful call-path job is not scientific authorization.

compute_launcher="${PHASE_NOISE_COMPUTE_LAUNCHER:?set PHASE_NOISE_COMPUTE_LAUNCHER}"
output_root="${PHASE_NOISE_CHAIN_OUTPUT_ROOT:?set PHASE_NOISE_CHAIN_OUTPUT_ROOT}"
source_archive="${PHASE_NOISE_SOURCE_ARCHIVE:?set PHASE_NOISE_SOURCE_ARCHIVE}"
source_archive_sha256="${PHASE_NOISE_SOURCE_ARCHIVE_SHA256:?set PHASE_NOISE_SOURCE_ARCHIVE_SHA256}"
source_revision="${PHASE_NOISE_SOURCE_REVISION:?set PHASE_NOISE_SOURCE_REVISION}"

fail() {
  echo "[submit-self-imagined-phase-noise] ERROR: $*" >&2
  exit 2
}

for required_name in \
  PHASE_NOISE_FACTOR_MANIFEST PHASE_NOISE_FACTOR_MANIFEST_SHA256 \
  PHASE_NOISE_BANK_RECEIPT PHASE_NOISE_BANK_RECEIPT_SHA256 \
  PHASE_NOISE_BANK_OUTPUT_ROOT PHASE_NOISE_EXECUTION_GROUP \
  PHASE_NOISE_CONDITION_MODE PHASE_NOISE_SOURCE_VIDEO PHASE_NOISE_PYTHON_BIN \
  BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT \
  BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ -n "${!required_name:-}" ]] || fail "set ${required_name}"
done

for value_name in compute_launcher output_root source_archive; do
  [[ "${!value_name}" == /* ]] || fail "${value_name} must be absolute"
  case "${!value_name}" in *$'\n'*|*',') fail "${value_name} contains unsafe export characters" ;; esac
done
for required_name in \
  PHASE_NOISE_FACTOR_MANIFEST PHASE_NOISE_BANK_RECEIPT PHASE_NOISE_BANK_OUTPUT_ROOT \
  PHASE_NOISE_SOURCE_VIDEO PHASE_NOISE_PYTHON_BIN BERNINI_OFFICIAL_ROOT \
  BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ "${!required_name}" == /* ]] || fail "${required_name} must be absolute"
  case "${!required_name}" in *$'\n'*|*',') fail "${required_name} contains unsafe export characters" ;; esac
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${PHASE_NOISE_FACTOR_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 must be lowercase"
[[ "${PHASE_NOISE_BANK_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 must be lowercase"
[[ "${PHASE_NOISE_EXECUTION_GROUP}" == "sp4-a" || "${PHASE_NOISE_EXECUTION_GROUP}" == "sp4-b" ]] || fail "execution group is invalid"
[[ "${PHASE_NOISE_CONDITION_MODE}" == "r2v5" || "${PHASE_NOISE_CONDITION_MODE}" == "rv2v4" ]] || fail "condition mode is invalid"

compute_launcher="$(realpath -e -- "${compute_launcher}")"
source_archive="$(realpath -e -- "${source_archive}")"
submitter_path="$(realpath -e -- "$0")"
[[ -f "${compute_launcher}" && ! -L "${compute_launcher}" ]] || fail "compute launcher is not plain"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${submitter_path}" && ! -L "${submitter_path}" ]] || fail "submitter is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"

compute_member="methods/bernini_action_editing/scripts/auh_infer_native_self_imagined_phase_noise_dual4.sbatch"
submitter_member="methods/bernini_action_editing/scripts/auh_submit_native_self_imagined_phase_noise_chain.sh"
archived_compute_sha="$(tar -xOf "${source_archive}" "${compute_member}" | sha256sum | awk '{print $1}')"
archived_submitter_sha="$(tar -xOf "${source_archive}" "${submitter_member}" | sha256sum | awk '{print $1}')"
[[ "$(sha256sum "${compute_launcher}" | awk '{print $1}')" == "${archived_compute_sha}" ]] || fail "compute launcher differs from archive"
[[ "$(sha256sum "${submitter_path}" | awk '{print $1}')" == "${archived_submitter_sha}" ]] || fail "submitter differs from archive"

[[ "${output_root}" != / ]] || fail "chain root cannot be filesystem root"
output_basename="${output_root##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "chain root basename is unsafe"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "chain root must be canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "refusing reused chain root"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "chain parent must exist"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "chain parent is noncanonical"
mkdir -- "${output_root}"

canary_output="${output_root}/canary-step1"
canary_export="ALL,PHASE_NOISE_OUTPUT_DIR=${canary_output},PHASE_NOISE_NUM_INFERENCE_STEPS=1"
canary_raw="$(sbatch \
  --parsable \
  --job-name=bernini-phase-canary1 \
  --export="${canary_export}" \
  "${compute_launcher}")"
canary_job="${canary_raw%%;*}"
[[ "${canary_job}" =~ ^[0-9]+$ ]] || fail "invalid canary job id: ${canary_raw}"

submission_receipt="${output_root}/submission.tsv"
(
  printf 'stage\tjob_id\tdependency\toutput\n'
  printf 'canary-step1\t%s\tnone\t%s\n' "${canary_job}" "${canary_output}"
  printf 'exact40\tNOT_SUBMITTED\tSCIENTIFIC_GATE_CLOSED\tNOT_CREATED\n'
) >"${submission_receipt}"
chmod 0444 -- "${submission_receipt}"

echo "[submit-self-imagined-phase-noise] canary_job=${canary_job} output=${canary_output}"
echo "[submit-self-imagined-phase-noise] exact40=NOT_SUBMITTED scientific_gate=CLOSED"
echo "[submit-self-imagined-phase-noise] receipt=${submission_receipt}"
