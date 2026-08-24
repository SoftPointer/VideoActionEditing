#!/usr/bin/env bash
set -euo pipefail
umask 077

# External trust-root template.  The caller supplies only a fresh run id; every
# executable/manifest trust anchor below is a literal, independently reportable
# SHA.  Do not replace these literals with values read from the package.

readonly EXPECTED_PARENT_JOB="143808"
readonly EXPECTED_NODE="auh7-1b-gpu-292"
readonly REPO_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
readonly RUN_BASE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job143808_v15c_r4"
readonly BOOTSTRAP="${REPO_ROOT}/methods/bernini_action_editing/tools/v15c_r4_external_bootstrap.py"
readonly RELEASE="${REPO_ROOT}/methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r4_release.json"
readonly PYTHON_BIN="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
readonly BOOTSTRAP_SHA256="a36374f5deb4b4a4e16148042e0e2bb0561e638a7f075783678dfdc50db1d7c1"
readonly RELEASE_SHA256="15755f505d0d898c0b43cfb36c5b2e67c063ec9227cbf9641f9411165bc9fae2"
readonly PYTHON_SHA256="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "v15c-r4 external launch template: $*" >&2
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

# The compute-node shell opens bootstrap once, hashes that descriptor, and asks
# Python to execute /proc/self/fd/9.  Thus bootstrap check and use name the same
# inode.  Python starts isolated (-I -S -B); bootstrap later constructs the
# explicit clean environment used by every authenticated package member.
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
  HOME=/nonexistent/v15c-r4-bootstrap
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
exec "${clean_env[@]}" /proc/self/fd/8 -I -S -B /proc/self/fd/9 \
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
' v15c-r4-bootstrap \
  "${BOOTSTRAP}" "${BOOTSTRAP_SHA256}" "${RELEASE}" "${RELEASE_SHA256}" \
  "${PYTHON_BIN}" "${PYTHON_SHA256}" "${REPO_ROOT}" "${RUN_ROOT}" \
  "${EXPECTED_PARENT_JOB}" "${EXPECTED_NODE}"
