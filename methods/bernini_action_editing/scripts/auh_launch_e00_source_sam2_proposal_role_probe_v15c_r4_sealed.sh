#!/usr/bin/env bash
set -euo pipefail
umask 077

# This launcher has no controller/bootstrap mode.  It may be entered only from
# the externally SHA-pinned stdlib bootstrap after exact-tree snapshot sealing.

readonly EXPECTED_PARENT_JOB="143808"
readonly EXPECTED_NODE="auh7-1b-gpu-292"
readonly PYTHON_BIN="/proc/self/fd/8"
readonly SOURCE_VIDEO="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/2f183dbf9e7a4d2e/source.mp4"
readonly SAM2_CHECKPOINT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/outputs/lucy_long/mask_annotator_runtime_63512cbb/VACE-Annotators/sam2/sam2.1_hiera_large.pt"
readonly SAM2_CONFIG="/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
readonly R6_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-owned-role-locator-v15b-e00-sp4-r6-null64-67fd8211-ff71de79-r2/output"
readonly RELEASE_REL="methods/bernini_action_editing/assets/e00_source_sam2_proposal_role_probe_v15c_r4_release.json"

fail() {
  echo "v15c-r4 sealed worker: $*" >&2
  exit 1
}

[[ "${1:-}" == "--sealed-worker" && "$#" -eq 7 ]] || \
  fail "entry requires externally authenticated sealed-worker arguments"
[[ "${V15C_R4_EXTERNAL_BOOTSTRAP:-}" == "1" ]] || fail "external bootstrap marker absent"
[[ -r "${PYTHON_BIN}" ]] || fail "authenticated Python descriptor absent"
readonly RUN_ROOT="$2"
readonly SNAPSHOT="$3"
readonly RELEASE_SHA="$4"
readonly BOOTSTRAP_RECEIPT="$5"
readonly BOOTSTRAP_SHA="$6"
readonly PYTHON_SHA="$7"
readonly METHOD_ROOT="${SNAPSHOT}/methods/bernini_action_editing"
readonly SPEC="${METHOD_ROOT}/assets/e00_source_sam2_proposal_role_probe_v15c.json"
readonly RELEASE="${SNAPSHOT}/${RELEASE_REL}"
readonly MATERIALIZER="${METHOD_ROOT}/materialize_source_sam2_proposal_tracks_v15c.py"
readonly RUNNER="${METHOD_ROOT}/run_source_object_proposal_role_probe_v15c.py"
readonly POSTFLIGHT="${METHOD_ROOT}/postflight_source_sam2_proposal_role_probe_v15c_r3.py"
readonly OVERLAY="${METHOD_ROOT}/tools/build_source_object_proposal_role_v15c_r3_review.py"
readonly FINALIZER="${METHOD_ROOT}/finalize_source_sam2_proposal_role_probe_v15c_r4.py"

[[ "${SLURM_JOB_ID:-}" == "${EXPECTED_PARENT_JOB}" ]] || fail "parent job differs"
[[ "$(/bin/hostname -s)" == "${EXPECTED_NODE}" ]] || fail "node differs"
[[ "${RELEASE_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "release SHA format differs"
[[ "${BOOTSTRAP_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bootstrap SHA format differs"
[[ "${PYTHON_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "Python SHA format differs"
[[ "${RUN_ROOT}/sealed_code_snapshot" == "${SNAPSHOT}" ]] || fail "snapshot placement differs"
[[ -f "${BOOTSTRAP_RECEIPT}" && ! -L "${BOOTSTRAP_RECEIPT}" ]] || fail "bootstrap receipt differs"
[[ ! -e "${RUN_ROOT}/tracks" && ! -e "${RUN_ROOT}/result.json" ]] || fail "track/result output is not fresh"
[[ ! -e "${RUN_ROOT}/postflight.json" && ! -e "${RUN_ROOT}/review" ]] || fail "postflight/review output is not fresh"
[[ ! -e "${RUN_ROOT}/COMPLETE.manifest.json" ]] || fail "COMPLETE is not fresh"

readonly -a RUNTIME_ARGS=(
  --run-root "${RUN_ROOT}"
  --snapshot "${SNAPSHOT}"
  --release-manifest "${RELEASE}"
  --release-sha256 "${RELEASE_SHA}"
  --bootstrap-receipt "${BOOTSTRAP_RECEIPT}"
  --bootstrap-sha256 "${BOOTSTRAP_SHA}"
  --python-sha256 "${PYTHON_SHA}"
  --source "${SOURCE_VIDEO}"
  --checkpoint "${SAM2_CHECKPOINT}"
  --config "${SAM2_CONFIG}"
  --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json"
  --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors"
)

verify_runtime() {
  "${PYTHON_BIN}" -E -s -B "${FINALIZER}" verify-runtime "${RUNTIME_ARGS[@]}"
}

# First package-internal Python execution.  The external bootstrap already
# authenticated FINALIZER and every dependency before entering this launcher.
verify_runtime
"${PYTHON_BIN}" -E -s -B -c '
import torch
if not (
    torch.cuda.is_available()
    and torch.cuda.device_count() == 1
    and torch.cuda.get_device_name(0) == "AMD Instinct MI210"
):
    raise SystemExit("exact one-GPU MI210 gate failed")
'

verify_runtime
"${PYTHON_BIN}" -E -s -B "${MATERIALIZER}" \
  --spec "${SPEC}" --source-video "${SOURCE_VIDEO}" \
  --checkpoint "${SAM2_CHECKPOINT}" --config-authority "${SAM2_CONFIG}" \
  --output-dir "${RUN_ROOT}/tracks" \
  >"${RUN_ROOT}/materializer.stdout.jsonl" 2>"${RUN_ROOT}/materializer.stderr.log"

verify_runtime
"${PYTHON_BIN}" -E -s -B "${RUNNER}" \
  --spec "${SPEC}" \
  --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
  --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
  --track-receipt "${RUN_ROOT}/tracks/track_receipt.json" \
  --track-tensors "${RUN_ROOT}/tracks/phase_coverage.safetensors" \
  --output-json "${RUN_ROOT}/result.json" \
  >"${RUN_ROOT}/runner.stdout.jsonl" 2>"${RUN_ROOT}/runner.stderr.log"

verify_runtime
"${PYTHON_BIN}" -E -s -B "${POSTFLIGHT}" \
  --spec "${SPEC}" --source-video "${SOURCE_VIDEO}" \
  --r6-receipt "${R6_ROOT}/e00_v15b_r6_probe_receipt.json" \
  --r6-tensors "${R6_ROOT}/e00_v15b_r6_affinity.safetensors" \
  --track-receipt "${RUN_ROOT}/tracks/track_receipt.json" \
  --track-tensors "${RUN_ROOT}/tracks/phase_coverage.safetensors" \
  --result-json "${RUN_ROOT}/result.json" \
  --output-json "${RUN_ROOT}/postflight.json" \
  >"${RUN_ROOT}/postflight.stdout.jsonl" 2>"${RUN_ROOT}/postflight.stderr.log"

verify_runtime
"${PYTHON_BIN}" -E -s -B "${OVERLAY}" \
  --source-video "${SOURCE_VIDEO}" \
  --track-receipt "${RUN_ROOT}/tracks/track_receipt.json" \
  --result-json "${RUN_ROOT}/result.json" \
  --postflight-json "${RUN_ROOT}/postflight.json" \
  --output-dir "${RUN_ROOT}/review" \
  >"${RUN_ROOT}/overlay.stdout.jsonl" 2>"${RUN_ROOT}/overlay.stderr.log"

verify_runtime
"${PYTHON_BIN}" -E -s -B "${FINALIZER}" complete \
  "${RUNTIME_ARGS[@]}" \
  --job-id "${SLURM_JOB_ID}" --node "$(/bin/hostname -s)" \
  --gpu-index "0" --gpu-name "AMD Instinct MI210"

"${PYTHON_BIN}" -E -s -B "${FINALIZER}" verify-complete --run-root "${RUN_ROOT}"
echo "v15c-r4 observer-only COMPLETE (reject-only, ROUTE_NO_GO): ${RUN_ROOT}"
