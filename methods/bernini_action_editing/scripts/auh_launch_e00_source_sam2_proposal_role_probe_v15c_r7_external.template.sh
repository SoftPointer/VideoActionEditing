#!/usr/bin/env bash
set -euo pipefail
umask 077

# LOCAL VERIFICATION TEMPLATE ONLY.  The r7 release explicitly sets
# observer_execution_authorized=false and contains no worker-launch command.

readonly REPO_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
readonly BOOTSTRAP="${REPO_ROOT}/methods/bernini_action_editing/tools/v15c_r7_external_bootstrap.py"
readonly RELEASE="${REPO_ROOT}/methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r7_release.json"
readonly PYTHON_AUTHORITY="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly BOOTSTRAP_SHA256="f99b529e2b9cc1e53f080bf395f9b81465ebabe7f3f9ec35b7d1d67c521db500"
readonly RELEASE_SHA256="6df2385dea259c1e39050c6acc0498925f9b1a92adb12795ab76c68f0a29c6a9"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "v15c-r7 local verifier: $*" >&2
  exit 1
}

[[ "$#" -eq 0 ]] || fail "this template accepts no worker/run arguments"
[[ "${BOOTSTRAP_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "bootstrap pin is not materialized"
[[ "${RELEASE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "release pin is not materialized"
[[ "${PYTHON_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "Python pin differs"
[[ -f "${BOOTSTRAP}" && ! -L "${BOOTSTRAP}" ]] || fail "bootstrap differs"
[[ -f "${RELEASE}" && ! -L "${RELEASE}" ]] || fail "release differs"
[[ -f "${PYTHON_AUTHORITY}" && ! -L "${PYTHON_AUTHORITY}" ]] || fail "Python authority differs"
read -r observed_bootstrap ignored < <(/usr/bin/sha256sum "${BOOTSTRAP}")
read -r observed_release ignored < <(/usr/bin/sha256sum "${RELEASE}")
read -r observed_python ignored < <(/usr/bin/sha256sum "${PYTHON_AUTHORITY}")
[[ "${observed_bootstrap}" == "${BOOTSTRAP_SHA256}" ]] || fail "bootstrap bytes differ"
[[ "${observed_release}" == "${RELEASE_SHA256}" ]] || fail "release bytes differ"
[[ "${observed_python}" == "${PYTHON_SHA256}" ]] || fail "Python bytes differ"

exec 7<"${BOOTSTRAP}"
exec 8<"${PYTHON_AUTHORITY}"
exec /usr/bin/env -i \
  HOME=/nonexistent/v15c-r7-local-verifier \
  LANG=C \
  LC_ALL=C \
  PATH=/vast/users/guangyi.chen/anaconda3/envs/vace/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/vast/users/guangyi.chen/anaconda3/envs/vace/lib \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONHASHSEED=0 \
  TMPDIR=/tmp \
  V15C_R7_LOCAL_VERIFY_ONLY=1 \
  /bin/bash -c '
set -euo pipefail
python_authority="$1"
shift
[[ -r /proc/self/fd/7 && -r /proc/self/fd/8 ]]
exec -a "${python_authority}" /proc/self/fd/8 -I -S -B /proc/self/fd/7 "$@"
' v15c-r7-python "${PYTHON_AUTHORITY}" \
  --trusted-bootstrap-fd 7 \
  --expected-bootstrap-sha256 "${BOOTSTRAP_SHA256}" \
  --trusted-python-fd 8 \
  --python-authority "${PYTHON_AUTHORITY}" \
  --expected-python-sha256 "${PYTHON_SHA256}" \
  verify-release \
  --root "${REPO_ROOT}" \
  --release-manifest "${RELEASE}" \
  --release-sha256 "${RELEASE_SHA256}"
