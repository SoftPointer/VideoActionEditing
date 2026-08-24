#!/usr/bin/env bash
# Execute dog/human serially as WORLD4/SP4 groups inside one all8 step.
#
# The holder has only 64 GiB host RAM.  Two simultaneous model populations
# reached 53--59 GiB MaxRSS and were killed by the Slurm cgroup.  A complete
# torchrun process tree now exits before the next cell starts, bounding host
# residency while leaving the parent all8 allocation intact.

set -Eeuo pipefail
umask 077

method_root="${PRESERVATION_INFER_METHOD_ROOT:?set method root}"
python_bin="${PRESERVATION_INFER_PYTHON_BIN:?set Python}"
registry="${PRESERVATION_INFER_REGISTRY:?set registry}"
registry_sha="${PRESERVATION_INFER_REGISTRY_SHA256:?set registry SHA}"
training_bundle="${PRESERVATION_INFER_TRAINING_BUNDLE:?set training bundle}"
adapter_sha="${PRESERVATION_INFER_ADAPTER_SHA256:?set adapter SHA}"
receipt_sha="${PRESERVATION_INFER_RECEIPT_SHA256:?set receipt SHA}"
output_root="${PRESERVATION_INFER_OUTPUT_ROOT:?set output root}"
runtime_revision="${PRESERVATION_INFER_RUNTIME_REVISION:?set runtime revision}"
runtime_archive_sha="${PRESERVATION_INFER_RUNTIME_ARCHIVE_SHA256:?set runtime archive SHA}"
launcher_sha="${PRESERVATION_INFER_LAUNCHER_SHA256:?set launcher SHA}"
dog_port="${PRESERVATION_INFER_DOG_PORT:?set dog port}"
human_port="${PRESERVATION_INFER_HUMAN_PORT:?set human port}"

bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/source_ea900d5/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

fail() { echo "[preservation-infer-exec] ERROR: $*" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "Slurm step differs"
[[ -f "${method_root}/infer_preservation_residual_action_canary_v1.py" ]] || fail "inference runtime missing"
[[ -x "${python_bin}" && -f "${registry}" && -d "${training_bundle}" ]] || fail "sealed input differs"
[[ -f "${checkpoint_manifest}" ]] || fail "checkpoint manifest missing"
for value in "${registry_sha}" "${adapter_sha}" "${receipt_sha}" "${runtime_archive_sha}" "${launcher_sha}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA differs"
done
[[ "${runtime_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "runtime revision differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output must be fresh"
mkdir -m 0700 "${output_root}" "${output_root}/logs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
scratch="$(mktemp -d "${scratch_parent%/}/preservation-infer-${SLURM_JOB_ID}-${SLURM_STEP_ID}.XXXXXXXX")"
touch "${scratch}/renderer-load.lock"
chmod 0600 "${scratch}/renderer-load.lock"
export PRESERVATION_INFER_LOAD_LOCK="${scratch}/renderer-load.lock"
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then
    find "${scratch}" -xdev -depth -mindepth 1 -delete
    rmdir "${scratch}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

launch_group() (
  set -Eeuo pipefail
  label="$1"
  visible="$2"
  port="$3"
  cache="${scratch}/${label}"
  mkdir -m 0700 "${cache}"
  for leaf in home tmp xdg torch triton pycache miopen-user miopen-custom; do mkdir -m 0700 "${cache}/${leaf}"; done
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export HOME="${cache}/home" TMPDIR="${cache}/tmp" XDG_CACHE_HOME="${cache}/xdg"
  export TORCH_EXTENSIONS_DIR="${cache}/torch" TRITON_CACHE_DIR="${cache}/triton"
  export PYTHONPYCACHEPREFIX="${cache}/pycache" MIOPEN_USER_DB_PATH="${cache}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom"
  exec "${python_bin}" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${port}" \
    "${method_root}/infer_preservation_residual_action_canary_v1.py" \
    --registry "${registry}" --expected-registry-sha256 "${registry_sha}" --cell-id "${label}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --training-bundle "${training_bundle}" --expected-adapter-sha256 "${adapter_sha}" \
    --expected-training-receipt-sha256 "${receipt_sha}" \
    --output-dir "${output_root}/${label}" --runtime-source-revision "${runtime_revision}" \
    --runtime-source-archive-sha256 "${runtime_archive_sha}" --launcher-source-sha256 "${launcher_sha}"
)

dog_status=0
human_status=0
launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/logs/dog.log" 2>&1 || dog_status=$?
if [[ "${dog_status}" == 0 ]]; then
  launch_group human 0,1,2,3 "${human_port}" >"${output_root}/logs/human.log" 2>&1 || human_status=$?
else
  human_status=125
fi
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "dog=${dog_status} human=${human_status}"
for cell in dog human; do
  [[ -f "${output_root}/${cell}/native-rv2v.mp4" && -f "${output_root}/${cell}/preservation-residual.mp4" && -f "${output_root}/${cell}/receipt.json" ]] || fail "${cell} output closure differs"
done
printf 'COMPLETE_PRESERVATION_INFERENCE output=%s\n' "${output_root}" >"${output_root}/INFERENCE.COMPLETE"
