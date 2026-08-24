#!/usr/bin/env bash
set -euo pipefail

# Observer-only v15c-r2 launcher.  It creates one fresh child step inside the
# retained parent allocation and never changes the parent allocation state.

readonly EXPECTED_PARENT_JOB="143808"
readonly EXPECTED_NODE="auh7-1b-gpu-292"
readonly REPO_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit"
readonly METHOD_ROOT="${REPO_ROOT}/methods/bernini_action_editing"
readonly PYTHON_BIN="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python"
readonly SOURCE_VIDEO="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/2f183dbf9e7a4d2e/source.mp4"
readonly SAM2_CHECKPOINT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/outputs/lucy_long/mask_annotator_runtime_63512cbb/VACE-Annotators/sam2/sam2.1_hiera_large.pt"
readonly SAM2_CONFIG="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
readonly R6_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-owned-role-locator-v15b-e00-sp4-r6-null64-67fd8211-ff71de79-r2/output"
readonly RUN_BASE="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job143808_v15c_r2"

readonly CORE="${METHOD_ROOT}/source_object_proposal_role_probe_v15c.py"
readonly MATERIALIZER="${METHOD_ROOT}/materialize_source_sam2_proposal_tracks_v15c.py"
readonly RUNNER="${METHOD_ROOT}/run_source_object_proposal_role_probe_v15c.py"
readonly POSTFLIGHT="${METHOD_ROOT}/postflight_source_sam2_proposal_role_probe_v15c_r2.py"
readonly OVERLAY="${METHOD_ROOT}/tools/build_source_object_proposal_role_v15c_r2_review.py"
readonly SPEC="${METHOD_ROOT}/assets/e00_source_sam2_proposal_role_probe_v15c.json"

verify_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    echo "sealed member SHA differs: ${path}" >&2
    return 1
  }
}

worker() {
  local run_root="$1"
  [[ "${SLURM_JOB_ID:-}" == "${EXPECTED_PARENT_JOB}" ]]
  [[ "$(hostname -s)" == "${EXPECTED_NODE}" ]]
  [[ ! -e "${run_root}/tracks" ]]
  [[ ! -e "${run_root}/result.json" ]]
  [[ ! -e "${run_root}/postflight.json" ]]
  [[ ! -e "${run_root}/review" ]]

  verify_sha "${CORE}" "195115abe027e84c2950988bcb4e5323f07e93c80d3c20a31c3c6c291d3fa6f5"
  verify_sha "${MATERIALIZER}" "d2cf3da4ac2afe15a10d12d923d47fa5658cffde7f23dea7de04c0f7b991ee2d"
  verify_sha "${RUNNER}" "a7295a950af7d987be59a7f4f0efd1dc2deaed3fb71afadf7e3521431a70507e"
  verify_sha "${POSTFLIGHT}" "1bd40ec68a36e69ccde64aca1cfabdf3085dce841bb8f109dea3642c2abc3d86"
  verify_sha "${OVERLAY}" "6df164af32177e9374ecfc544f748bb9516cab4bec43072ce54a1ec64da9599a"
  verify_sha "${SPEC}" "5ad0a90804eeb68523cd94cd2790b32fe370e23fcf0d02441d5dc082c361a59b"
  verify_sha "${SOURCE_VIDEO}" "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de"
  verify_sha "${SAM2_CHECKPOINT}" "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
  verify_sha "${SAM2_CONFIG}" "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
  verify_sha "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" "8f081c990edd84a64ca35e78ca1de3d4ea6cf4b80bfcdec70bf54c51dc9ed959"
  verify_sha "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" "2535193d41a3405460bd152cd77bc61db7ef8ea6ba7cefd98f514f0787acc553"

  "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; assert torch.cuda.get_device_name(0)=="AMD Instinct MI210"'

  "${PYTHON_BIN}" "${MATERIALIZER}" \
    --spec "${SPEC}" \
    --source-video "${SOURCE_VIDEO}" \
    --checkpoint "${SAM2_CHECKPOINT}" \
    --config-authority "${SAM2_CONFIG}" \
    --output-dir "${run_root}/tracks" \
    >"${run_root}/materializer.stdout.jsonl" 2>"${run_root}/materializer.stderr.log"

  "${PYTHON_BIN}" "${RUNNER}" \
    --spec "${SPEC}" \
    --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
    --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --track-tensors "${run_root}/tracks/phase_coverage.safetensors" \
    --output-json "${run_root}/result.json" \
    >"${run_root}/runner.stdout.jsonl" 2>"${run_root}/runner.stderr.log"

  "${PYTHON_BIN}" "${POSTFLIGHT}" \
    --spec "${SPEC}" \
    --source-video "${SOURCE_VIDEO}" \
    --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
    --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --track-tensors "${run_root}/tracks/phase_coverage.safetensors" \
    --result-json "${run_root}/result.json" \
    --output-json "${run_root}/postflight.json" \
    >"${run_root}/postflight.stdout.jsonl" 2>"${run_root}/postflight.stderr.log"

  "${PYTHON_BIN}" "${OVERLAY}" \
    --source-video "${SOURCE_VIDEO}" \
    --track-receipt "${run_root}/tracks/track_receipt.json" \
    --result-json "${run_root}/result.json" \
    --postflight-json "${run_root}/postflight.json" \
    --output-dir "${run_root}/review" \
    >"${run_root}/overlay.stdout.jsonl" 2>"${run_root}/overlay.stderr.log"

  "${PYTHON_BIN}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["route_authorized"] is False and p["training_authorized"] is False and p["decode_authorized"] is False; q=json.load(open(sys.argv[2])); assert q["human_audit_action"]=="reject_only" and q["human_audit_may_authorize_route"] is False' "${run_root}/result.json" "${run_root}/postflight.json"
  echo "observer-only v15c-r2 complete: ${run_root}"
}

if [[ "${1:-}" == "--node-worker" ]]; then
  [[ "$#" -eq 2 ]]
  worker "$2"
  exit 0
fi

[[ "$#" -eq 1 ]] || {
  echo "usage: $0 <fresh-run-id>" >&2
  exit 2
}
readonly RUN_ID="$1"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$ ]]
readonly RUN_ROOT="${RUN_BASE}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]]

job_state="$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%T')"
job_nodes="$(squeue -h -j "${EXPECTED_PARENT_JOB}" -o '%N')"
[[ "${job_state}" == "RUNNING" ]]
scontrol show hostnames "${job_nodes}" | grep -Fx "${EXPECTED_NODE}" >/dev/null
mkdir -p "${RUN_ROOT}"

srun \
  --jobid="${EXPECTED_PARENT_JOB}" \
  --overlap \
  --exact \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${EXPECTED_NODE}" \
  --gpus-per-task=1 \
  "$0" --node-worker "${RUN_ROOT}"
