#!/usr/bin/env bash
set -euo pipefail
umask 077

# External trust-root template.  The caller supplies only a fresh run id; every
# executable/manifest trust anchor below is a literal, independently reportable
# SHA.  Do not replace these literals with values read from the package.

readonly EXPECTED_PARENT_JOB="143808"
readonly EXPECTED_NODE="auh7-1b-gpu-292"
readonly REPO_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
readonly RUN_BASE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job143808_v15c_r6"
readonly BOOTSTRAP="${REPO_ROOT}/methods/bernini_action_editing/tools/v15c_r6_external_bootstrap.py"
readonly RELEASE="${REPO_ROOT}/methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r6_release.json"
readonly PYTHON_BIN="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly BOOTSTRAP_SHA256="8a29b635d929dda86c46e3f383a4c961058e08169c03f0540abea63fbfe08b8b"
readonly RELEASE_SHA256="3339150722c781eafe135a0e71ab4d1ef4167324e84125f97bf29cf66fffb86e"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "v15c-r6 external launch template: $*" >&2
  exit 1
}

sha_file() {
  local observed remainder
  read -r observed remainder < <(/usr/bin/sha256sum "$1")
  [[ "${observed}" =~ ^[0-9a-f]{64}$ ]] || fail "sha256sum output differs"
  printf '%s\n' "${observed}"
}

[[ "$#" -eq 1 ]] || fail "usage: $0 <fresh-run-id>"
readonly RUN_ID="$1"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || fail "run id differs"
readonly RUN_ROOT="${RUN_BASE}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root exists"
[[ -f "${BOOTSTRAP}" && ! -L "${BOOTSTRAP}" ]] || fail "bootstrap is not regular"
[[ -f "${RELEASE}" && ! -L "${RELEASE}" ]] || fail "release is not regular"
[[ -f "${PYTHON_BIN}" && ! -L "${PYTHON_BIN}" ]] || fail "Python is not regular"
[[ "$(sha_file "${BOOTSTRAP}")" == "${BOOTSTRAP_SHA256}" ]] || fail "bootstrap pin differs"
[[ "$(sha_file "${RELEASE}")" == "${RELEASE_SHA256}" ]] || fail "release pin differs"
[[ "$(sha_file "${PYTHON_BIN}")" == "${PYTHON_SHA256}" ]] || fail "Python pin differs"
[[ "$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%T')" == "RUNNING" ]] || fail "parent is not running"
scontrol show hostnames "$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%N')" | \
  grep -Fx "${EXPECTED_NODE}" >/dev/null || fail "node is not allocated"

# The compute-node shell opens Python and bootstrap once, hashes those exact
# descriptors, and executes both through /proc/self/fd.  A nested Bash
# exec -a helper supplies PYTHON_BIN as argv[0], so CPython reports the frozen
# canonical sys.executable while the kernel still executes the verified FD.
srun --jobid="${EXPECTED_PARENT_JOB}" --overlap --exact --nodes=1 --ntasks=1 \
  --nodelist="${EXPECTED_NODE}" --gpus-per-task=1 \
  /bin/bash -c '
set -euo pipefail
bootstrap="$1"
bootstrap_sha="$2"
release="$3"
release_sha="$4"
python_bin="$5"
python_sha="$6"
source_root="$7"
run_root="$8"
job="$9"
node="${10}"
exec 8<"${python_bin}"
exec 9<"${bootstrap}"
read -r observed_bootstrap ignored < <(/usr/bin/sha256sum /proc/self/fd/9)
read -r observed_release ignored < <(/usr/bin/sha256sum "${release}")
read -r observed_python ignored < <(/usr/bin/sha256sum /proc/self/fd/8)
[[ "${observed_bootstrap}" == "${bootstrap_sha}" ]]
[[ "${observed_release}" == "${release_sha}" ]]
[[ "${observed_python}" == "${python_sha}" ]]
clean_env=(
  /usr/bin/env -i
  HOME=/nonexistent/v15c-r6-bootstrap
  LANG=C
  LC_ALL=C
  PATH=/vast/users/guangyi.chen/anaconda3/envs/vace/bin:/opt/rocm/bin:/usr/bin:/bin
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/vast/users/guangyi.chen/anaconda3/envs/vace/lib
  PYTHONDONTWRITEBYTECODE=1
  PYTHONNOUSERSITE=1
  PYTHONHASHSEED=0
  SLURM_JOB_ID="${job}"
)
for key in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HSA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  value="${!key:-}"
  if [[ -n "${value}" ]]; then
    [[ "${value}" =~ ^[A-Za-z0-9_,.:-]{1,256}$ ]]
    clean_env+=("${key}=${value}")
  fi
done
exec "${clean_env[@]}" /bin/bash -c "
set -euo pipefail
python_authority=\"\$1\"
python_fd=\"\$2\"
bootstrap_fd=\"\$3\"
shift 3
[[ -r \"\${python_fd}\" && -r \"\${bootstrap_fd}\" ]]
exec -a \"\${python_authority}\" \"\${python_fd}\" -I -S -B \"\${bootstrap_fd}\" \"\$@\"
" v15c-r6-python "${python_bin}" /proc/self/fd/8 /proc/self/fd/9 \
  --source-root "${source_root}" \
  --release-manifest "${release}" \
  --expected-release-sha256 "${release_sha}" \
  --expected-bootstrap-sha256 "${bootstrap_sha}" \
  --bootstrap-source-path "${bootstrap}" \
  --trusted-bootstrap-fd 9 \
  --python-bin "${python_bin}" \
  --trusted-python-fd 8 \
  --expected-python-sha256 "${python_sha}" \
  --run-root "${run_root}" \
  --expected-job "${job}" \
  --expected-node "${node}"
' v15c-r6-bootstrap \
  "${BOOTSTRAP}" "${BOOTSTRAP_SHA256}" "${RELEASE}" "${RELEASE_SHA256}" \
  "${PYTHON_BIN}" "${PYTHON_SHA256}" "${REPO_ROOT}" "${RUN_ROOT}" \
  "${EXPECTED_PARENT_JOB}" "${EXPECTED_NODE}"
