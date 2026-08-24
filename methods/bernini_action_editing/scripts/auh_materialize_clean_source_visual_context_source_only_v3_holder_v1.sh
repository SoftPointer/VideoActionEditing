#!/usr/bin/env bash
# Materialize the sealed physical index-0-only 64/16/8 split on the clean-arm
# holder.  This CPU step never starts a model/optimizer and never releases the
# parent allocation.  The strict single-actor heldout eight are reserved first.

set -Eeuo pipefail
umask 077

fail() { echo "[csvc-source-only-v3] ERROR: $*" >&2; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

readonly holder_job=135980
readonly holder_node=auh7-1b-gpu-239
readonly holder_user=guangyi.chen
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly method_root="${CSVC_METHOD_ROOT:?set sealed method root}"
readonly release_root="${CSVC_SOURCE_ONLY_RELEASE_ROOT:?set fresh source-only release root}"

for name in method_root release_root; do
  value="${!name}"
  [[ "${value}" == /vast/users/guangyi.chen/* && "${value}" != / ]] || fail "${name} path differs"
done
[[ -d "${method_root}" && ! -L "${method_root}" && "$(readlink -f -- "${method_root}")" == "${method_root}" ]] || fail "method root differs"
[[ -f "${method_root}/clean_source_visual_context_training_v1.py" ]] || fail "materializer missing"
[[ ! -e "${release_root}" && ! -L "${release_root}" ]] || fail "release root must be fresh"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"

job_record="$(scontrol show job -o "${holder_job}")"
[[ "${job_record}" == *"JobId=${holder_job} "* && "${job_record}" == *"JobState=RUNNING"* ]] || fail "holder is not RUNNING"
[[ "${job_record}" == *"UserId=${holder_user}"* && "${job_record}" == *"NodeList=${holder_node}"* ]] || fail "holder owner/node differs"
[[ -z "$(squeue -s -j "${holder_job}" -h -o '%i' | awk '/[.][0-9]+$/ {print}')" ]] || fail "holder already has a numbered child"

mkdir -m 0700 "${release_root}" "${release_root}/logs"
readonly manifest="${release_root}/source_only_split_v3.json"
readonly materialized="${release_root}/physical_source_posterior_index0"

set +e
srun --jobid="${holder_job}" --nodelist="${holder_node}" --nodes=1 --ntasks=1 \
  --exclusive --exact --kill-on-bad-exit=1 --cpus-per-task=16 --mem=48G \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH="${method_root}" \
    "${python_bin}" -B "${method_root}/clean_source_visual_context_training_v1.py" \
      build-manifest --output "${manifest}" --materialization-root "${materialized}" \
  >"${release_root}/logs/materialize.log" 2>&1
status=$?
set -e
if (( status != 0 )); then
  tail -n 200 "${release_root}/logs/materialize.log" >&2 || true
  exit "${status}"
fi

[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "manifest missing"
manifest_sha="$(sha256_file "${manifest}")"
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA differs"
"${python_bin}" -B "${method_root}/clean_source_visual_context_training_v1.py" \
  audit-manifest --manifest "${manifest}" \
  --ack-upstream-training-use-forbidden \
  --ack-user-authorized-exploratory-training \
  >"${release_root}/logs/audit.log" 2>&1 || fail "materialized manifest audit failed"

printf 'holder_job=%s\nholder_node=%s\nmanifest=%s\nmanifest_sha256=%s\nschema=v3\nphysical_files=88\nheldout_strict_single_actor=8\noptimizer_constructed=false\nparent_not_released=true\n' \
  "${holder_job}" "${holder_node}" "${manifest}" "${manifest_sha}" \
  >"${release_root}/MATERIALIZATION_COMPLETE"
echo "MATERIALIZATION_COMPLETE manifest=${manifest} sha256=${manifest_sha} parent_retained=true"
