#!/usr/bin/env bash
set -euo pipefail
umask 077

# v15c-r3 observer only. Usage requires the independently reported release-file
# SHA, which is the trust anchor and avoids a self-referential launcher pin.

readonly EXPECTED_PARENT_JOB="143808"
readonly EXPECTED_NODE="auh7-1b-gpu-292"
readonly REPO_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
readonly REPO_METHOD_ROOT="${REPO_ROOT}/methods/bernini_action_editing"
readonly PYTHON_BIN="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python"
readonly SOURCE_VIDEO="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/2f183dbf9e7a4d2e/source.mp4"
readonly SAM2_CHECKPOINT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/outputs/lucy_long/mask_annotator_runtime_63512cbb/VACE-Annotators/sam2/sam2.1_hiera_large.pt"
readonly SAM2_CONFIG="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
readonly R6_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-owned-role-locator-v15b-e00-sp4-r6-null64-67fd8211-ff71de79-r2/output"
readonly RUN_BASE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job143808_v15c_r3"
readonly RELEASE_REL="methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r3_release.json"
readonly REPO_RELEASE="${REPO_ROOT}/${RELEASE_REL}"
readonly REPO_FINALIZER="${REPO_METHOD_ROOT}/finalize_source_sam2_proposal_role_probe_v15c_r3.py"
readonly PYTHON_SHA="8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a"

fail() {
  echo "v15c-r3 sealed launcher: $*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

verify_input_pins() {
  [[ "$(sha_file "${PYTHON_BIN}")" == "${PYTHON_SHA}" ]] || fail "Python pin differs"
  [[ "$(sha_file "${SOURCE_VIDEO}")" == "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de" ]] || fail "source pin differs"
  [[ "$(sha_file "${SAM2_CHECKPOINT}")" == "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318" ]] || fail "checkpoint pin differs"
  [[ "$(sha_file "${SAM2_CONFIG}")" == "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107" ]] || fail "config pin differs"
  [[ "$(sha_file "${R6_ROOT}/e00_v15b_r6_probe_receipt.json")" == "8f081c990edd84a64ca35e78ca1de3d4ea6cf4b80bfcdec70bf54c51dc9ed959" ]] || fail "r6 receipt pin differs"
  [[ "$(sha_file "${R6_ROOT}/e00_v15b_r6_affinity.safetensors")" == "2535193d41a3405460bd152cd77bc61db7ef8ea6ba7cefd98f514f0787acc553" ]] || fail "r6 tensor pin differs"
}

sanitize_python_environment() {
  unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PYTHONINSPECT
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONNOUSERSITE=1
}

verify_release() {
  local snapshot="$1"
  local release_sha="$2"
  "${PYTHON_BIN}" -E -s -B \
    "${snapshot}/methods/bernini_action_editing/finalize_source_sam2_proposal_role_probe_v15c_r3.py" \
    verify-release \
    --root "${snapshot}" \
    --release-manifest "${snapshot}/${RELEASE_REL}" \
    --expected-sha256 "${release_sha}"
}

bootstrap_worker() {
  local run_root="$1"
  local release_sha="$2"
  [[ "${SLURM_JOB_ID:-}" == "${EXPECTED_PARENT_JOB}" ]] || fail "parent job differs"
  [[ "$(hostname -s)" == "${EXPECTED_NODE}" ]] || fail "node differs"
  [[ "${release_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "release SHA format differs"
  [[ ! -e "${run_root}/sealed_code_snapshot" ]] || fail "snapshot is not fresh"
  [[ ! -e "${run_root}/COMPLETE.manifest.json" ]] || fail "COMPLETE already exists"
  sanitize_python_environment
  verify_input_pins
  [[ "$(sha_file "${REPO_RELEASE}")" == "${release_sha}" ]] || fail "release trust anchor differs"
  "${PYTHON_BIN}" -E -s -B "${REPO_FINALIZER}" verify-release \
    --root "${REPO_ROOT}" --release-manifest "${REPO_RELEASE}" \
    --expected-sha256 "${release_sha}"

  local snapshot="${run_root}/sealed_code_snapshot"
  mkdir -m 0700 "${snapshot}"
  while IFS= read -r relative; do
    [[ -n "${relative}" ]] || fail "empty release member"
    mkdir -p "${snapshot}/$(dirname "${relative}")"
    chmod 0700 "${snapshot}/$(dirname "${relative}")"
    install -m 0600 "${REPO_ROOT}/${relative}" "${snapshot}/${relative}"
  done < <("${PYTHON_BIN}" -E -s -B -c '
import json,sys
p=json.load(open(sys.argv[1],"r",encoding="utf-8"))
members=p.get("members")
if type(members) is not list or len(members)!=8:
    raise SystemExit(17)
for row in members:
    path=row.get("path") if type(row) is dict else None
    if not isinstance(path,str) or not path:
        raise SystemExit(18)
    print(path)
' "${REPO_RELEASE}")
  mkdir -p "${snapshot}/$(dirname "${RELEASE_REL}")"
  chmod 0700 "${snapshot}/$(dirname "${RELEASE_REL}")"
  install -m 0600 "${REPO_RELEASE}" "${snapshot}/${RELEASE_REL}"
  find "${snapshot}" -type d -exec chmod 0700 {} +
  verify_release "${snapshot}" "${release_sha}"
  exec bash "${snapshot}/methods/bernini_action_editing/scripts/auh_launch_e00_source_sam2_proposal_role_probe_v15c_r3_sealed.sh" \
    --sealed-worker "${run_root}" "${release_sha}"
}

sealed_worker() {
  local run_root="$1"
  local release_sha="$2"
  local snapshot="${run_root}/sealed_code_snapshot"
  local method_root="${snapshot}/methods/bernini_action_editing"
  local spec="${method_root}/assets/e00_source_sam2_proposal_role_probe_v15c.json"
  local materializer="${method_root}/materialize_source_sam2_proposal_tracks_v15c.py"
  local runner="${method_root}/run_source_object_proposal_role_probe_v15c.py"
  local postflight="${method_root}/postflight_source_sam2_proposal_role_probe_v15c_r3.py"
  local overlay="${method_root}/tools/build_source_object_proposal_role_v15c_r3_review.py"
  local finalizer="${method_root}/finalize_source_sam2_proposal_role_probe_v15c_r3.py"
  [[ "${SLURM_JOB_ID:-}" == "${EXPECTED_PARENT_JOB}" ]] || fail "worker job differs"
  [[ "$(hostname -s)" == "${EXPECTED_NODE}" ]] || fail "worker node differs"
  [[ ! -e "${run_root}/tracks" && ! -e "${run_root}/result.json" ]] || fail "outputs are not fresh"
  [[ ! -e "${run_root}/postflight.json" && ! -e "${run_root}/review" ]] || fail "outputs are not fresh"
  [[ ! -e "${run_root}/COMPLETE.manifest.json" ]] || fail "COMPLETE is not fresh"
  sanitize_python_environment

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B -c '
import sys,torch
ok=(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="AMD Instinct MI210")
if not ok:
    raise SystemExit("exact one-GPU MI210 gate failed")
'

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${materializer}" \
    --spec "${spec}" --source-video "${SOURCE_VIDEO}" \
    --checkpoint "${SAM2_CHECKPOINT}" --config-authority "${SAM2_CONFIG}" \
    --output-dir "${run_root}/tracks" \
    >"${run_root}/materializer.stdout.jsonl" 2>"${run_root}/materializer.stderr.log"

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${runner}" \
    --spec "${spec}" \
    --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
    --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --track-tensors "${run_root}/tracks/phase_coverage.safetensors" \
    --output-json "${run_root}/result.json" \
    >"${run_root}/runner.stdout.jsonl" 2>"${run_root}/runner.stderr.log"

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${postflight}" \
    --spec "${spec}" --source-video "${SOURCE_VIDEO}" \
    --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
    --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --track-tensors "${run_root}/tracks/phase_coverage.safetensors" \
    --result-json "${run_root}/result.json" \
    --output-json "${run_root}/postflight.json" \
    >"${run_root}/postflight.stdout.jsonl" 2>"${run_root}/postflight.stderr.log"

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${overlay}" \
    --source-video "${SOURCE_VIDEO}" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --result-json "${run_root}/result.json" \
    --postflight-json "${run_root}/postflight.json" \
    --output-dir "${run_root}/review" \
    >"${run_root}/overlay.stdout.jsonl" 2>"${run_root}/overlay.stderr.log"

  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${finalizer}" complete \
    --run-root "${run_root}" --snapshot "${snapshot}" \
    --release-manifest "${snapshot}/${RELEASE_REL}" --release-sha256 "${release_sha}" \
    --job-id "${SLURM_JOB_ID}" --node "$(hostname -s)" --gpu-index "0" \
    --gpu-name "AMD Instinct MI210" \
    --source "${SOURCE_VIDEO}" --checkpoint "${SAM2_CHECKPOINT}" \
    --config "${SAM2_CONFIG}" \
    --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
    --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors"
  verify_release "${snapshot}" "${release_sha}"
  verify_input_pins
  "${PYTHON_BIN}" -E -s -B "${finalizer}" verify-complete --run-root "${run_root}"
  echo "v15c-r3 observer-only COMPLETE: ${run_root}"
}

if [[ "${1:-}" == "--bootstrap-worker" ]]; then
  [[ "$#" -eq 3 ]] || fail "bootstrap arity differs"
  bootstrap_worker "$2" "$3"
elif [[ "${1:-}" == "--sealed-worker" ]]; then
  [[ "$#" -eq 3 ]] || fail "worker arity differs"
  sealed_worker "$2" "$3"
else
  [[ "$#" -eq 2 ]] || fail "usage: $0 <fresh-run-id> <release-manifest-sha256>"
  readonly RUN_ID="$1"
  readonly RELEASE_SHA="$2"
  [[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$ ]] || fail "run id differs"
  [[ "${RELEASE_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "release SHA differs"
  readonly RUN_ROOT="${RUN_BASE}/${RUN_ID}"
  [[ ! -e "${RUN_ROOT}" ]] || fail "run root exists"
  [[ "$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%T')" == "RUNNING" ]] || fail "parent is not running"
  scontrol show hostnames "$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%N')" | grep -Fx "${EXPECTED_NODE}" >/dev/null || fail "node is not allocated"
  mkdir -m 0700 "${RUN_ROOT}"
  srun --jobid="${EXPECTED_PARENT_JOB}" --overlap --exact --nodes=1 --ntasks=1 \
    --nodelist="${EXPECTED_NODE}" --gpus-per-task=1 \
    bash "$0" --bootstrap-worker "${RUN_ROOT}" "${RELEASE_SHA}"
fi
