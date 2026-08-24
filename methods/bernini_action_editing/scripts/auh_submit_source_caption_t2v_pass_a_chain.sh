#!/usr/bin/env bash
set -Eeuo pipefail

# Login-side submission only.  The exact40 render is scheduled after the
# one-step engineering job succeeds.  afterok establishes call-path/OOM health,
# not semantic validity: exact40 remains rejected until an independent reviewer
# accepts every branch at both preregistered seeds.

compute_launcher="${PASS_A_COMPUTE_LAUNCHER:?set PASS_A_COMPUTE_LAUNCHER}"
output_root="${PASS_A_CHAIN_OUTPUT_ROOT:?set PASS_A_CHAIN_OUTPUT_ROOT}"
source_archive="${PASS_A_SOURCE_ARCHIVE:?set PASS_A_SOURCE_ARCHIVE}"
source_archive_sha256="${PASS_A_SOURCE_ARCHIVE_SHA256:?set PASS_A_SOURCE_ARCHIVE_SHA256}"
source_revision="${PASS_A_SOURCE_REVISION:?set PASS_A_SOURCE_REVISION}"

fail() {
  echo "[submit-bernini-pass-a] ERROR: $*" >&2
  exit 2
}

for required_name in \
  PASS_A_SOURCE_VIDEO PASS_A_SOURCE_SHA256 PASS_A_PYTHON_BIN \
  BERNINI_OFFICIAL_ROOT BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT \
  BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ -n "${!required_name:-}" ]] || fail "set ${required_name}"
done

for value_name in compute_launcher output_root source_archive; do
  [[ "${!value_name}" == /* ]] || fail "${value_name} must be absolute"
  case "${!value_name}" in
    *$'\n'*|*,*) fail "${value_name} contains an unsafe export character" ;;
  esac
done
for required_name in \
  PASS_A_SOURCE_VIDEO PASS_A_PYTHON_BIN BERNINI_OFFICIAL_ROOT \
  BERNINI_VEOMNI_ROOT BERNINI_ACTION_CHECKPOINT \
  BERNINI_CHECKPOINT_CONTENT_MANIFEST
do
  [[ "${!required_name}" == /* ]] || fail "${required_name} must be absolute"
  case "${!required_name}" in
    *$'\n'*|*,*) fail "${required_name} contains an unsafe export character" ;;
  esac
done

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "archive SHA-256 must be lowercase"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision must be full lowercase SHA-1"
[[ "${PASS_A_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "source SHA-256 must be lowercase"

compute_launcher="$(realpath -e -- "${compute_launcher}")"
source_archive="$(realpath -e -- "${source_archive}")"
submitter_path="$(realpath -e -- "$0")"
[[ -f "${compute_launcher}" && ! -L "${compute_launcher}" ]] || fail "compute launcher is not plain"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ -f "${submitter_path}" && ! -L "${submitter_path}" ]] || fail "submitter is not plain"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"

compute_member="methods/bernini_action_editing/scripts/auh_source_caption_t2v_pass_a_dual4.sbatch"
submitter_member="methods/bernini_action_editing/scripts/auh_submit_source_caption_t2v_pass_a_chain.sh"
archived_compute_sha="$(tar -xOf "${source_archive}" "${compute_member}" | sha256sum | awk '{print $1}')"
archived_submitter_sha="$(tar -xOf "${source_archive}" "${submitter_member}" | sha256sum | awk '{print $1}')"
[[ "$(sha256sum "${compute_launcher}" | awk '{print $1}')" == "${archived_compute_sha}" ]] || fail "compute launcher differs from source archive"
[[ "$(sha256sum "${submitter_path}" | awk '{print $1}')" == "${archived_submitter_sha}" ]] || fail "submitter differs from source archive"

[[ "${output_root}" != / ]] || fail "chain output root cannot be root"
output_basename="${output_root##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "chain output basename is unsafe"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "chain output root must be canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "refusing to reuse chain output root"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "chain output parent must exist and be plain"
[[ "$(realpath -e -- "${output_parent}")" == "${output_parent}" ]] || fail "chain output parent is noncanonical"
mkdir -- "${output_root}"

canary_output="${output_root}/canary-step1"
exact40_output="${output_root}/exact40"
canary_raw="$(sbatch \
  --parsable \
  --job-name=bern-pass-a-step1 \
  --export="ALL,PASS_A_OUTPUT_DIR=${canary_output},PASS_A_NUM_INFERENCE_STEPS=1" \
  "${compute_launcher}")"
canary_job="${canary_raw%%;*}"
[[ "${canary_job}" =~ ^[0-9]+$ ]] || fail "canary sbatch returned invalid job id: ${canary_raw}"

exact_raw="$(sbatch \
  --parsable \
  --job-name=bern-pass-a-exact40 \
  --dependency="afterok:${canary_job}" \
  --export="ALL,PASS_A_OUTPUT_DIR=${exact40_output},PASS_A_NUM_INFERENCE_STEPS=40" \
  "${compute_launcher}")"
exact_job="${exact_raw%%;*}"
[[ "${exact_job}" =~ ^[0-9]+$ ]] || fail "exact40 sbatch returned invalid job id: ${exact_raw}"

submission_receipt="${output_root}/submission.tsv"
submission_temporary="${output_root}/.submission.tsv.$$.tmp"
(
  printf 'stage\tjob_id\tdependency\toutput\tsemantic_status\n'
  printf 'canary-step1\t%s\tnone\t%s\tengineering_only\n' "${canary_job}" "${canary_output}"
  printf 'exact40\t%s\tafterok:%s\t%s\tpending_independent_complete_2x4_manual_qualification\n' \
    "${exact_job}" "${canary_job}" "${exact40_output}"
) >"${submission_temporary}"
chmod 0444 -- "${submission_temporary}"
mv -- "${submission_temporary}" "${submission_receipt}"

echo "[submit-bernini-pass-a] canary_job=${canary_job} output=${canary_output}"
echo "[submit-bernini-pass-a] exact40_job=${exact_job} dependency=afterok:${canary_job} output=${exact40_output}"
echo "[submit-bernini-pass-a] exact40_semantic_status=pending_independent_complete_2x4_manual_qualification"
echo "[submit-bernini-pass-a] receipt=${submission_receipt}"
