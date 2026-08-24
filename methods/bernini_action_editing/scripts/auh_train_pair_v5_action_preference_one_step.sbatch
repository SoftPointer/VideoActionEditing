#!/usr/bin/env bash
#SBATCH --job-name=pair5-dpo-1step
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() { echo "[pair-v5-dpo-one-step] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

output="${PAIR_V5_TRAIN_OUTPUT:?set a fresh absolute PAIR_V5_TRAIN_OUTPUT}"
manifest="${PAIR_V5_PREFERENCE_MANIFEST:?set the sealed preference manifest}"
manifest_sha256="${PAIR_V5_PREFERENCE_MANIFEST_SHA256:?set its raw SHA-256}"
source_archive="${PAIR_V5_SOURCE_ARCHIVE:?set the immutable method source archive}"
source_archive_sha256="${PAIR_V5_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${PAIR_V5_SOURCE_REVISION:?set the full method source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
python_bin="${PAIR_V5_PYTHON_BIN:?set vace Python executable}"
frozen_cio="${PAIR_V5_FROZEN_CIO_ADAPTER:-}"
frozen_cio_sha256="${PAIR_V5_FROZEN_CIO_ADAPTER_SHA256:-}"
learning_rate="${PAIR_V5_LEARNING_RATE:-1e-6}"
beta="${PAIR_V5_DPO_BETA:-1000}"
seed="${PAIR_V5_TRAIN_SEED:-20260808}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in output manifest source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  absolute "${!name}" "${name}"
done
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "manifest SHA-256 differs"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be a nonnegative decimal"
[[ -z "${frozen_cio}" && -z "${frozen_cio_sha256}" ]] || {
  [[ -n "${frozen_cio}" && -n "${frozen_cio_sha256}" ]] || fail "CIO path/SHA must be paired"
  absolute "${frozen_cio}" "frozen CIO adapter"
  [[ "${frozen_cio_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "frozen CIO SHA differs"
}

manifest="$(realpath -e -- "${manifest}")"
source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -z "${frozen_cio}" ]] || frozen_cio="$(realpath -e -- "${frozen_cio}")"

for path in manifest source_archive checkpoint_manifest python_bin; do
  [[ -f "${!path}" && ! -L "${!path}" ]] || fail "${path} must be a plain file"
done
for path in bernini_root veomni_root checkpoint; do
  [[ -d "${!path}" && ! -L "${!path}" ]] || fail "${path} must be a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest bytes differ"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"
if [[ -n "${frozen_cio}" ]]; then
  [[ -f "${frozen_cio}" && ! -L "${frozen_cio}" ]] || fail "CIO adapter must be a plain file"
  [[ "$(sha256sum "${frozen_cio}" | awk '{print $1}')" == "${frozen_cio_sha256}" ]] || fail "CIO bytes differ"
fi

[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe output"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output is create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent is a symlink"

(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-dpo-${SLURM_JOB_ID}.XXXXXX")"
mkdir -- "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" "${task_scratch}/triton" \
  "${task_scratch}/pycache" "${task_scratch}/tmp"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-dpo-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source archive differs"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged source archive revision differs"
mkdir -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for relative in \
  train_pair_v5_action_preference.py \
  pair_v5_action_adapter.py \
  pair_v5_action_energy_calibration.py \
  pair_v5_candidate_evaluator_packet.py \
  pair_v5_flow_dpo.py \
  pair_v5_native_bridge.py \
  pair_v5_native_rollout_spec.py \
  pair_v5_phase_conjunctive_energy.py \
  pair_v5_safe_pareto.py \
  source_self_native_ref_contrastive_v3.py \
  source_self_native_rv2v_guidance.py \
  source_self_native_target_adapter.py \
  source_self_runtime.py \
  infer_native_identity_generation_canary.py \
  infer_source_kv_carrier_oracle.py \
  infer_lora.py \
  train_lora.py \
  tools/materialize_vae.py \
  scripts/auh_train_pair_v5_action_preference_one_step.sbatch \
  tests/test_train_pair_v5_action_preference.py \
  tests/test_pair_v5_candidate_evaluator_packet.py \
  tests/test_pair_v5_action_energy_calibration.py \
  tests/test_pair_v5_safe_pareto.py \
  tests/test_auh_train_pair_v5_action_preference_launcher.py; do
  [[ -f "${method_root}/${relative}" && ! -L "${method_root}/${relative}" ]] || fail "archive lacks ${relative}"
done
find "${method_root}" -type f -exec chmod a-w -- {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

for pattern in \
  test_train_pair_v5_action_preference.py \
  test_pair_v5_flow_dpo.py \
  test_pair_v5_action_adapter.py \
  test_pair_v5_native_bridge.py \
  test_pair_v5_candidate_evaluator_packet.py \
  test_pair_v5_action_energy_calibration.py \
  test_pair_v5_safe_pareto.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
    -s "${method_root}/tests" -p "${pattern}"
done

cio_args=()
if [[ -n "${frozen_cio}" ]]; then
  cio_args+=(--frozen-cio-adapter "${frozen_cio}" --expected-frozen-cio-adapter-sha256 "${frozen_cio_sha256}")
fi

echo "[pair-v5-dpo-one-step] allocation=8xMI210 topology=WORLD8/DP2xSP4 exact81=true refs=0,27,53,80 optimizer_steps=1"
echo "[pair-v5-dpo-one-step] student_inputs=source_video,source_refs,complete_caption,fresh_gaussian target=false proposal=false donor=false mask=false flow=false pose=false track=false trajectory=false"

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "${method_root}/train_pair_v5_action_preference.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --preference-manifest "${manifest}" \
  --expected-preference-manifest-sha256 "${manifest_sha256}" \
  --output "${output}" \
  --max-steps 1 \
  --learning-rate "${learning_rate}" \
  --beta "${beta}" \
  --seed "${seed}" \
  --num-frames 81 \
  "${cio_args[@]}" \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-exploratory-no-action-editing-claim

[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "extracted method tree changed"

"${python_bin}" -B - "${output}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
expected={"adapter.safetensors","optimizer.pt","history.json","receipt.json"}
if root.is_symlink() or {p.name for p in root.iterdir()} != expected:
    raise SystemExit("training artifact closure differs")
receipt=json.loads((root/"receipt.json").read_text(encoding="ascii"))
declared=receipt.pop("receipt_digest",None)
payload=json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest()!=declared:
    raise SystemExit("run receipt digest differs")
if not (
    receipt.get("schema_version")=="bernini-pair-v5-action-preference-training-receipt-v1"
    and receipt.get("complete") is True
    and receipt.get("optimizer_steps")==1
    and receipt.get("distributed",{}).get("all_eight_gpus_used") is True
    and receipt.get("native_rv2v4",{}).get("same_fresh_epsilon_and_sigma_for_chosen_rejected") is True
    and receipt.get("native_rv2v4",{}).get("recorded_rollout_noise_loaded") is False
    and receipt.get("semantic_action_editing_success") is False
    and receipt.get("scientific_generalization_claim_authorized") is False
    and receipt.get("long_training_automatically_submitted") is False
):
    raise SystemExit("run/scientific contract differs")
if receipt.get("method_source_revision")!=sys.argv[2] or receipt.get("method_source_archive_sha256")!=sys.argv[3]:
    raise SystemExit("source binding differs")
for name in expected-{"receipt.json"}:
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=receipt["artifacts"][name]:
        raise SystemExit(f"artifact hash differs: {name}")
print("PAIR_V5_ACTION_PREFERENCE_ONE_STEP_STRONG_AUDIT_OK")
PY

echo "[pair-v5-dpo-one-step] PASS output=${output} action_success_claim=false"
