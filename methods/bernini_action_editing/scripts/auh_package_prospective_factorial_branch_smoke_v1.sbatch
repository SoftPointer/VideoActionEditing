#!/usr/bin/env bash
#SBATCH --job-name=fact7-html
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --account=test-acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -Eeuo pipefail
umask 077

source_archive="${FACTORIAL_BRANCH_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${FACTORIAL_BRANCH_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${FACTORIAL_BRANCH_SOURCE_REVISION:?set source revision}"
manifest_sha256="${FACTORIAL_BRANCH_MANIFEST_SHA256:?set manifest SHA-256}"
released_cells="${FACTORIAL_BRANCH_RELEASED_CELLS:?set sorted comma-separated source:seed cells}"
output_root="${FACTORIAL_BRANCH_OUTPUT_ROOT:?set completed release output root}"
python_bin="${FACTORIAL_BRANCH_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() { echo "[factorial-branch-package] ERROR: $*" >&2; exit 2; }

[[ "${source_archive}" == /* && -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ && "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"
[[ "${output_root}" == /* && -d "${output_root}" && ! -L "${output_root}" ]] || fail "output root differs"
[[ -f "${output_root}/release.receipt.json" && ! -L "${output_root}/release.receipt.json" ]] || fail "release receipt missing"
[[ ! -e "${output_root}/index.html" && ! -L "${output_root}/index.html" ]] || fail "HTML already exists"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/factorial-html-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local rc=$?
  trap - EXIT
  case "${task_scratch}" in
    "${scratch_parent%/}/factorial-html-${SLURM_JOB_ID}."*) chmod -R u+w "${task_scratch}" 2>/dev/null || true; rm -rf "${task_scratch}" ;;
    *) rc=2 ;;
  esac
  exit "${rc}"
}
trap cleanup EXIT

tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}"
method_root="${task_scratch}/methods/bernini_action_editing"
manifest="${method_root}/assets/prospective_factorial_branch_manifest_v1.json"
builder="${method_root}/tools/build_prospective_factorial_branch_review_html_v1.py"
[[ -f "${manifest}" && -f "${builder}" ]] || fail "packaging source closure differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest hash differs"

IFS=',' read -r -a cell_values <<<"${released_cells}"
[[ "${#cell_values[@]}" -eq 2 ]] || fail "review packet requires exactly two cells"
cell_args=()
for cell in "${cell_values[@]}"; do cell_args+=(--cell "${cell}"); done

"${python_bin}" -B "${builder}" \
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
  "${cell_args[@]}" --output-root "${output_root}" --output "${output_root}/index.html" \
  >"${output_root}/html-build.receipt.json"

packet="${output_root%/*}/review_packet.tar.gz"
packet_sha="${packet}.sha256"
[[ ! -e "${packet}" && ! -L "${packet}" && ! -e "${packet_sha}" && ! -L "${packet_sha}" ]] || fail "review archive already exists"
tar -C "${output_root%/*}" -czf "${packet}" "${output_root##*/}"
sha256sum "${packet}" >"${packet_sha}"
echo "FACTORIAL_BRANCH_REVIEW_HTML_PACKAGED_NOT_TRAINING_TARGET"
