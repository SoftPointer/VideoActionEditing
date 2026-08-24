#!/usr/bin/env bash
set -Eeuo pipefail

# Login-side submission only.  Schedule the dual-WORLD4 one-step engineering
# canary, then the matched exact40 stage with afterok:<canary>.  The compute
# launcher never submits recursively.

compute_launcher="${ROLE_REBINDING_COMPUTE_LAUNCHER:?set ROLE_REBINDING_COMPUTE_LAUNCHER}"
output_root="${ROLE_REBINDING_CHAIN_OUTPUT_ROOT:?set ROLE_REBINDING_CHAIN_OUTPUT_ROOT}"
source_archive="${ROLE_REBINDING_SOURCE_ARCHIVE:?set ROLE_REBINDING_SOURCE_ARCHIVE}"
source_archive_sha256="${ROLE_REBINDING_SOURCE_ARCHIVE_SHA256:?set ROLE_REBINDING_SOURCE_ARCHIVE_SHA256}"
source_revision="${ROLE_REBINDING_SOURCE_REVISION:?set ROLE_REBINDING_SOURCE_REVISION}"

fail() {
  echo "[submit-native-role-rebinding] ERROR: $*" >&2
  exit 2
}

for required_name in \
  ROLE_REBINDING_FACTOR_MANIFEST ROLE_REBINDING_FACTOR_MANIFEST_SHA256 \
  ROLE_REBINDING_BANK_RECEIPT ROLE_REBINDING_BANK_RECEIPT_SHA256 \
  ROLE_REBINDING_BANK_OUTPUT_ROOT ROLE_REBINDING_SOURCE_VIDEO \
  ROLE_REBINDING_WRONG_SOURCE_VIDEO \
  ROLE_REBINDING_PYTHON_BIN BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT \
  BERNINI_ACTION_CHECKPOINT BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ -n "${!required_name:-}" ]] || fail "set ${required_name}"
done

for value_name in compute_launcher output_root source_archive; do
  [[ "${!value_name}" == /* ]] || fail "${value_name} must be absolute"
  case "${!value_name}" in *$'\n'*|*,*) fail "${value_name} has unsafe export bytes" ;; esac
done
for required_name in \
  ROLE_REBINDING_FACTOR_MANIFEST ROLE_REBINDING_BANK_RECEIPT \
  ROLE_REBINDING_BANK_OUTPUT_ROOT ROLE_REBINDING_SOURCE_VIDEO \
  ROLE_REBINDING_WRONG_SOURCE_VIDEO \
  ROLE_REBINDING_PYTHON_BIN BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT \
  BERNINI_ACTION_CHECKPOINT BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ "${!required_name}" == /* ]] || fail "${required_name} must be absolute"
  case "${!required_name}" in *$'\n'*|*,*) fail "${required_name} has unsafe export bytes" ;; esac
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${ROLE_REBINDING_FACTOR_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 must be lowercase"
[[ "${ROLE_REBINDING_BANK_RECEIPT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "bank receipt SHA-256 must be lowercase"

compute_launcher="$(realpath -e -- "${compute_launcher}")"
source_archive="$(realpath -e -- "${source_archive}")"
submitter_path="$(realpath -e -- "$0")"
[[ -f "${compute_launcher}" && ! -L "${compute_launcher}" ]] || fail "compute launcher is not plain"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${submitter_path}" && ! -L "${submitter_path}" ]] || fail "submitter is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"

compute_member="methods/bernini_action_editing/scripts/auh_infer_native_action_scaffold_identity_rebinding_dual4.sbatch"
submitter_member="methods/bernini_action_editing/scripts/auh_submit_native_action_scaffold_identity_rebinding_chain.sh"
archived_compute_sha="$(tar -xOf "${source_archive}" "${compute_member}" | sha256sum | awk '{print $1}')"
archived_submitter_sha="$(tar -xOf "${source_archive}" "${submitter_member}" | sha256sum | awk '{print $1}')"
[[ "$(sha256sum "${compute_launcher}" | awk '{print $1}')" == "${archived_compute_sha}" ]] || fail "compute launcher differs from archive"
[[ "$(sha256sum "${submitter_path}" | awk '{print $1}')" == "${archived_submitter_sha}" ]] || fail "submitter differs from archive"

[[ "${output_root}" != / ]] || fail "chain output cannot be filesystem root"
output_basename="${output_root##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output root must be canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "refusing reused output root"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must exist"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "output parent is noncanonical"
mkdir -- "${output_root}"

canary_output="${output_root}/canary-step1"
exact40_output="${output_root}/exact40"
canary_raw="$(sbatch \
  --parsable \
  --job-name=bernini-rebind-canary1 \
  --export="ALL,ROLE_REBINDING_OUTPUT_DIR=${canary_output},ROLE_REBINDING_NUM_INFERENCE_STEPS=1" \
  "${compute_launcher}")"
canary_job="${canary_raw%%;*}"
[[ "${canary_job}" =~ ^[0-9]+$ ]] || fail "invalid one-step job id: ${canary_raw}"

exact_raw="$(sbatch \
  --parsable \
  --job-name=bernini-rebind-exact40 \
  --dependency="afterok:${canary_job}" \
  --export="ALL,ROLE_REBINDING_OUTPUT_DIR=${exact40_output},ROLE_REBINDING_NUM_INFERENCE_STEPS=40" \
  "${compute_launcher}")"
exact_job="${exact_raw%%;*}"
[[ "${exact_job}" =~ ^[0-9]+$ ]] || fail "invalid exact40 job id: ${exact_raw}"

submission_receipt="${output_root}/submission.tsv"
(
  printf 'stage\tjob_id\tdependency\toutput\n'
  printf 'canary-step1\t%s\tnone\t%s\n' "${canary_job}" "${canary_output}"
  printf 'exact40\t%s\tafterok:%s\t%s\n' "${exact_job}" "${canary_job}" "${exact40_output}"
) >"${submission_receipt}"
chmod 0444 -- "${submission_receipt}"

echo "[submit-native-role-rebinding] canary_job=${canary_job} output=${canary_output}"
echo "[submit-native-role-rebinding] exact40_job=${exact_job} dependency=afterok:${canary_job} output=${exact40_output}"
echo "[submit-native-role-rebinding] receipt=${submission_receipt}"
