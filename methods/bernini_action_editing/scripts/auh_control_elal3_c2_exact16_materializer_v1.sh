#!/usr/bin/env bash
# External one-holder controller.  The submitting command must literal-pin this
# file's SHA through ELAL3_C2_CONTROLLER_SHA256; this controller then pins the
# launcher and both deterministic release artifacts before handing off.

set -Eeuo pipefail
umask 0027
fail() { echo "[elal3-c2-exact16-controller] ERROR: $*" >&2; exit 2; }

expected_launcher_sha="f16973abb8d3ccd41136af07ec8ae6014708b27bba8341ae39572c793ff42430"
expected_archive_sha="143e99cfbbafe470f008a3be6cf3a23412ddc0fe3d7e5b41f161c7faa097fce6"
expected_manifest_sha="47a8c1ef2dd1805da91af4eed65868ff668dbec7950449cbeec0bf6814e3f687"

release_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c2_exact16_materializer_release_20260817_r3"
output_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c2_exact16_modelbound_node226_20260817_retry2"
launcher="${release_root}/auh_run_elal3_c2_exact16_materializer_release_v1.sh"
archive="${release_root}/source.tar"
manifest="${release_root}/source.manifest.json"
controller="$(readlink -f "$0")"

[[ "${SLURM_JOB_ID:-}:${HOSTNAME%%.*}" == "141620:auh7-1b-gpu-226" ]] || fail "only holder 141620/node226 is authorized"
[[ "${ELAL3_C2_CONTROLLER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "external submitter did not literal-pin controller SHA"
[[ "$(sha256sum "${controller}" | awk '{print $1}')" == "${ELAL3_C2_CONTROLLER_SHA256}" ]] || fail "controller self file differs from submit literal"
[[ -d "${release_root}" && ! -L "${release_root}" && "$(stat -c '%a' "${release_root}")" == "555" ]] || fail "release root type/mode differs"
for binding in \
  "${launcher}:${expected_launcher_sha}" \
  "${archive}:${expected_archive_sha}" \
  "${manifest}:${expected_manifest_sha}"; do
  path="${binding%%:*}"; expected="${binding##*:}"
  [[ -f "${path}" && ! -L "${path}" && "$(stat -c '%a:%h' "${path}")" == "444:1" ]] || fail "release file mode/link differs: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "release file SHA differs: ${path}"
done
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root is not fresh"

export ELAL3_C2_SOURCE_ARCHIVE="${archive}"
export ELAL3_C2_SOURCE_MANIFEST="${manifest}"
export ELAL3_C2_OUTPUT_ROOT="${output_root}"
export ELAL3_C2_CONTROLLER_PATH="${controller}"
exec bash "${launcher}"
