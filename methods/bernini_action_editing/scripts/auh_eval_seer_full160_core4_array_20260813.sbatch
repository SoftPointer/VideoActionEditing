#!/usr/bin/env bash
# Decode the sealed SEER core4 with the completed full160 adapter.
# Prepared for deployment only: submit after checkpoint-00000160 postflight.
#SBATCH --job-name=seer160-eval256
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --exclusive
#SBATCH --exclude=auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

fail() { echo "[seer160-eval256] ERROR: $*" >&2; exit 2; }

run_id="${SEER_FULL160_EVAL_RUN_ID:?set fresh run ID}"
[[ "${run_id}" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]] || fail "run ID differs"

exp=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
stage="${exp}/staging/seer-full160-eval-overlay-v1"
training_archive="${exp}/staging/seer-event-erasure-method-r1.tar"
training_archive_sha=ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822
training_revision=6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a
overlay_archive="${stage}/seer-full160-eval-overlay-v1.tar"
overlay_archive_sha="${SEER_FULL160_OVERLAY_ARCHIVE_SHA256:?set frozen overlay archive SHA}"
overlay_manifest="${stage}/seer-full160-eval-overlay-v1.manifest.json"
overlay_manifest_sha="${SEER_FULL160_OVERLAY_MANIFEST_SHA256:?set frozen overlay manifest SHA}"
source_binder="${stage}/bind_seer_full160_eval_source_v1.py"
source_binder_sha="${SEER_FULL160_SOURCE_BINDER_SHA256:?set frozen source binder SHA}"
overlay_builder="${stage}/tools/build_seer_full160_eval_overlay_v1.py"
overlay_builder_sha="${SEER_FULL160_OVERLAY_BUILDER_SHA256:?set frozen overlay builder SHA}"
postflight="${stage}/postflight_seer_full160_core4_eval_v1.py"
postflight_sha="${SEER_FULL160_POSTFLIGHT_SHA256:?set frozen postflight SHA}"
postflight_launcher="${stage}/auh_postflight_seer_full160_core4_eval_20260813.sbatch"
postflight_launcher_sha="${SEER_FULL160_POSTFLIGHT_LAUNCHER_SHA256:?set frozen postflight launcher SHA}"
verdict_reducer="${stage}/seer_full160_review_verdict_v1.py"
verdict_reducer_sha="${SEER_FULL160_VERDICT_REDUCER_SHA256:?set frozen verdict reducer SHA}"
launcher_sha="${SEER_FULL160_EVAL_LAUNCHER_SHA256:?set this launcher SHA}"
launcher_path="${stage}/auh_eval_seer_full160_core4_array_20260813.sbatch"
spec_sha=82fbe0f042d86f8d54aa254ce72a384e70aa5bdc3c1ac66d5422037cd4b4051c
adapter="${exp}/runs/seer-same-state-fm-160step-r1/checkpoint-00000160"
bernini=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
output_root="${exp}/heldout/full160-core4-${run_id}"
logs_root="${exp}/logs/eval-full160-core4-${run_id}"
results_root="${output_root}/results"
node_cache_root="${SLURM_TMPDIR:-/tmp}/seer160-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

[[ "${SLURM_NNODES:?}" == 1 && "${SLURM_NTASKS:?}" == 1 ]] || fail "Slurm topology differs"
[[ "${SLURM_ARRAY_TASK_COUNT:?}" == 2 ]] || fail "array contract differs"
[[ "${overlay_archive_sha}" =~ ^[0-9a-f]{64}$ && "${overlay_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "overlay pins differ"
[[ "${source_binder_sha}" =~ ^[0-9a-f]{64}$ && "${overlay_builder_sha}" =~ ^[0-9a-f]{64}$ && "${postflight_sha}" =~ ^[0-9a-f]{64}$ && "${postflight_launcher_sha}" =~ ^[0-9a-f]{64}$ && "${verdict_reducer_sha}" =~ ^[0-9a-f]{64}$ && "${launcher_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "runtime pins differ"
[[ "${overlay_archive_sha}" != "${training_archive_sha}" ]] || fail "overlay cannot impersonate training archive"
[[ "$(sha256sum "${training_archive}" | awk '{print $1}')" == "${training_archive_sha}" ]] || fail "training archive differs"
[[ "$(sha256sum "${overlay_archive}" | awk '{print $1}')" == "${overlay_archive_sha}" ]] || fail "overlay archive differs"
[[ "$(sha256sum "${overlay_manifest}" | awk '{print $1}')" == "${overlay_manifest_sha}" ]] || fail "overlay manifest differs"
[[ "$(sha256sum "${source_binder}" | awk '{print $1}')" == "${source_binder_sha}" ]] || fail "source binder differs"
[[ "$(sha256sum "${overlay_builder}" | awk '{print $1}')" == "${overlay_builder_sha}" ]] || fail "overlay builder differs"
[[ "$(sha256sum "${postflight}" | awk '{print $1}')" == "${postflight_sha}" ]] || fail "postflight differs"
[[ "$(sha256sum "${postflight_launcher}" | awk '{print $1}')" == "${postflight_launcher_sha}" ]] || fail "postflight launcher differs"
[[ "$(sha256sum "${verdict_reducer}" | awk '{print $1}')" == "${verdict_reducer_sha}" ]] || fail "verdict reducer differs"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_sha}" ]] || fail "launcher differs"
[[ -f "${adapter}/adapter/adapter_model.safetensors" && -f "${adapter}/receipt.json" ]] || fail "full160 adapter incomplete"
"${python_bin}" -B - "${adapter}/receipt.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert type(r.get("global_step")) is int and r["global_step"] == 160
assert type(r.get("max_steps")) is int and r["max_steps"] == 160
assert r.get("immutable_contract",{}).get("value",{}).get("method_source_archive_sha256") == "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822"
assert r.get("immutable_contract",{}).get("value",{}).get("method_source_revision") == "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a"
PY

mkdir -m 0700 -p "${logs_root}"
run_marker="${output_root}/.array-job-id"
if mkdir -m 0700 "${output_root}" 2>/dev/null; then
  mkdir -m 0700 "${results_root}"
  printf '%s\n' "${SLURM_ARRAY_JOB_ID}" >"${run_marker}"
  chmod 0400 "${run_marker}"
else
  for _ in {1..60}; do
    [[ -f "${run_marker}" ]] && break
    sleep 1
  done
fi
[[ -f "${run_marker}" && "$(<"${run_marker}")" == "${SLURM_ARRAY_JOB_ID}" ]] || fail "output root belongs to another array job"
[[ -d "${results_root}" && ! -L "${results_root}" ]] || fail "shared results root differs"
[[ ! -e "${node_cache_root}" ]] || fail "node cache must be fresh"
mkdir -m 0700 "${node_cache_root}"
cleanup() {
  find "${node_cache_root}" -xdev -mindepth 1 -delete 2>/dev/null || true
  rmdir "${node_cache_root}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${SLURM_ARRAY_TASK_ID}" in
  0) iids=(99cde432839f4240 311c82f83eca4a7f) ;;
  1) iids=(6ea45d35943742bb 6d346c38cf504493) ;;
  *) fail "unexpected array index" ;;
esac

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"${python_bin}" -B -c 'import torch; assert torch.cuda.is_available(); assert torch.version.hip; assert torch.cuda.device_count() >= 4'

run_case() {
  local iid="$1"
  local case_root="${output_root}/${iid}"
  local base_root="${case_root}/base-source"
  local runtime_root="${case_root}/runtime-source"
  local method_root="${runtime_root}/methods/bernini_action_editing"
  local runner="${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
  local trained_runner="infer_seer_same_state_full160_lora.py"
  local spec="${method_root}/assets/self_generated_action_lora_heldout_core4_v1.json"
  local results="${results_root}"
  local cache_root="${node_cache_root}/${iid}"
  local binding_receipt="${case_root}/eval-source-binding.json"
  local case_binding_receipt="${case_root}/eval-execution-binding.json"
  [[ ! -e "${case_root}" ]] || fail "case output must be fresh: ${iid}"
  mkdir -m 0700 "${case_root}" "${base_root}" "${cache_root}"
  tar --delay-directory-restore --no-same-owner --no-same-permissions -xf "${training_archive}" -C "${base_root}"
  cp -R "${base_root}" "${runtime_root}"
  chmod u+w "${method_root}"
  for overlay_name in \
    infer_seer_same_state_lora.py \
    run_self_generated_action_lora_heldout_core4_v1.py; do
    chmod u+w "${method_root}/${overlay_name}"
  done
  tar --no-same-owner --no-same-permissions -xf "${overlay_archive}" -C "${runtime_root}"
  chmod 0400 \
    "${method_root}/infer_seer_same_state_full160_lora.py" \
    "${method_root}/infer_seer_same_state_lora.py" \
    "${method_root}/run_self_generated_action_lora_heldout_core4_v1.py"
  chmod u-w "${method_root}"
  "${python_bin}" -B "${source_binder}" bind \
    --training-archive "${training_archive}" \
    --overlay-archive "${overlay_archive}" \
    --overlay-manifest "${overlay_manifest}" \
    --expected-overlay-archive-sha256 "${overlay_archive_sha}" \
    --expected-overlay-manifest-sha256 "${overlay_manifest_sha}" \
    --base-root "${base_root}" --runtime-root "${runtime_root}" \
    --output "${binding_receipt}" >"${case_root}/eval-source-binding.stdout.json"
  "${python_bin}" -B "${source_binder}" verify-receipt --receipt "${binding_receipt}" >/dev/null
  [[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${spec_sha}" ]] || fail "heldout spec differs"
  "${python_bin}" -B "${runner}" --spec "${spec}" \
    --expected-spec-sha256 "${spec_sha}" inspect --verify-files

  run_arm() {
    local arm="$1" port="$2"
    local cache="${cache_root}/${arm}"
    local adaptation=()
    mkdir -m 0700 -p "${cache}/tmp" "${cache}/xdg" \
      "${cache}/torch-extensions" "${cache}/triton" \
      "${cache}/miopen-user" "${cache}/miopen-custom"
    if [[ "${arm}" == trained_adapter ]]; then
      adaptation=(--adapter-checkpoint "${adapter}")
    fi
    env \
      HOME="${cache}" TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg" \
      TORCH_EXTENSIONS_DIR="${cache}/torch-extensions" \
      TRITON_CACHE_DIR="${cache}/triton" \
      MIOPEN_USER_DB_PATH="${cache}/miopen-user" \
      MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom" \
      "${python_bin}" -B "${runner}" \
        --spec "${spec}" --expected-spec-sha256 "${spec_sha}" \
        run-arm --iid "${iid}" --arm "${arm}" \
        --method-root "${method_root}" --python-bin "${python_bin}" \
        --bernini-root "${bernini}" --veomni-root "${veomni}" \
        --checkpoint "${checkpoint}" "${adaptation[@]}" \
        --trained-infer-runner "${trained_runner}" \
        --output-root "${results}" --master-port "${port}" \
        --method-source-revision "${training_revision}" \
        --method-source-archive-sha256 "${training_archive_sha}"
  }

  echo "START method=full160 iid=${iid} arm=frozen_base"
  run_arm frozen_base 29611 >"${case_root}/frozen-base.log" 2>&1
  echo "START method=full160 iid=${iid} arm=trained_adapter"
  run_arm trained_adapter 29612 >"${case_root}/trained-adapter.log" 2>&1
  verify_args=(
    --spec "${spec}" --expected-spec-sha256 "${spec_sha}"
    verify-pair --iid "${iid}" --adapter-checkpoint "${adapter}"
    --output-root "${results}"
  )
  command -v ffmpeg >/dev/null 2>&1 && verify_args+=(--ffmpeg "$(command -v ffmpeg)")
  "${python_bin}" -B "${runner}" "${verify_args[@]}" \
    >"${case_root}/paired-verification.json"
  "${python_bin}" -B "${source_binder}" finalize-case \
    --source-binding "${binding_receipt}" \
    --paired-receipt "${results}/${iid}/paired-receipt.json" \
    --output "${case_binding_receipt}" \
    >"${case_root}/eval-execution-binding.stdout.json"
  "${python_bin}" -B "${source_binder}" verify-case \
    --receipt "${case_binding_receipt}" >/dev/null
  [[ -f "${binding_receipt}" && -f "${case_binding_receipt}" && -f "${results}/${iid}/paired-receipt.json" ]] || fail "case evidence closure incomplete"
  echo "PASS_GENERATION_NOT_METHOD_SUCCESS method=full160 iid=${iid}"
}

for iid in "${iids[@]}"; do
  run_case "${iid}"
done
printf 'COMPLETE method=full160 array_job=%s array_task=%s node=%s\n' \
  "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}" \
  >"${output_root}/task-${SLURM_ARRAY_TASK_ID}.COMPLETE"
