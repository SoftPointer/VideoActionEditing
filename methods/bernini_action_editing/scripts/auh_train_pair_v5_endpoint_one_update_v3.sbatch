#!/usr/bin/env bash
#SBATCH --job-name=pair5-endpoint-v3
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00

set -Eeuo pipefail
umask 027

fail() { echo "[pair5-endpoint-v3] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

output="${PAIR_V5_ENDPOINT_V3_OUTPUT:?set fresh output}"
manifest="${PAIR_V5_ENDPOINT_V3_MANIFEST:?set endpoint v3 manifest}"
manifest_sha="${PAIR_V5_ENDPOINT_V3_MANIFEST_SHA256:?set manifest SHA-256}"
generation_round="${PAIR_V5_ENDPOINT_V3_GENERATION_ROUND:?set generation round}"
parent_policy_digest="${PAIR_V5_ENDPOINT_V3_PARENT_POLICY_DIGEST:?set parent policy digest}"
source_archive="${PAIR_V5_ENDPOINT_V3_SOURCE_ARCHIVE:?set immutable source archive}"
source_archive_sha="${PAIR_V5_ENDPOINT_V3_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${PAIR_V5_ENDPOINT_V3_SOURCE_REVISION:?set source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${PAIR_V5_ENDPOINT_V3_PYTHON_BIN:?set vace Python}"
sigma_index="${PAIR_V5_ENDPOINT_V3_SIGMA_INDEX:-20}"
gradient_accumulation="${PAIR_V5_ENDPOINT_V3_GRADIENT_ACCUMULATION:-1}"
learning_rate="${PAIR_V5_ENDPOINT_V3_LEARNING_RATE:-1e-6}"
beta="${PAIR_V5_ENDPOINT_V3_BETA:-1000}"
max_grad_norm="${PAIR_V5_ENDPOINT_V3_MAX_GRAD_NORM:-1.0}"
seed="${PAIR_V5_ENDPOINT_V3_SEED:-20260808}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in output manifest source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  absolute "${!name}" "${name}"
done
for name in manifest_sha parent_policy_digest source_archive_sha; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${generation_round}" =~ ^[0-9]+$ ]] || fail "generation round differs"
[[ "${sigma_index}" =~ ^[0-9]+$ && "${sigma_index}" -le 37 ]] || fail "sigma index differs"
[[ "${gradient_accumulation}" =~ ^[1-4]$ ]] || fail "gradient accumulation differs"

manifest="$(realpath -e -- "${manifest}")"
source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "manifest bytes differ"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha}" ]] || fail "source archive differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe output"
mkdir -p -- "$(dirname -- "${output}")"
[[ "$(realpath -m -- "${output}")" == "${output}" && ! -e "${output}" && ! -L "${output}" ]] || fail "output is not fresh/canonical"
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

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || fail "scratch differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair5-endpoint-v3-${SLURM_JOB_ID}.XXXXXX")"
mkdir -- "${task_scratch}/source" "${task_scratch}/tmp" "${task_scratch}/pycache" "${task_scratch}/triton"
export TMPDIR="${task_scratch}/tmp" PYTHONPYCACHEPREFIX="${task_scratch}/pycache" TRITON_CACHE_DIR="${task_scratch}/triton"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair5-endpoint-v3-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

tar --no-same-owner --no-same-permissions -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for relative in \
  train_pair_v5_endpoint_one_update_v3.py pair_v5_endpoint_one_update_v3.py \
  pair_v5_action_adapter.py pair_v5_flow_dpo.py pair_v5_native_bridge.py \
  source_self_native_ref_contrastive_v3.py source_self_native_rv2v_guidance.py \
  source_self_native_target_adapter.py source_self_runtime.py train_lora.py \
  infer_native_identity_generation_canary.py infer_source_kv_carrier_oracle.py \
  tests/test_pair_v5_endpoint_one_update_v3.py \
  tests/test_train_pair_v5_endpoint_one_update_v3.py \
  scripts/auh_train_pair_v5_endpoint_one_update_v3.sbatch; do
  [[ -f "${method_root}/${relative}" && ! -L "${method_root}/${relative}" ]] || fail "archive lacks ${relative}"
done

common_args=(
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}" --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha}"
  --expected-generation-round "${generation_round}"
  --expected-parent-policy-digest "${parent_policy_digest}"
  --output "${output}" --sigma-index "${sigma_index}"
  --gradient-accumulation-steps "${gradient_accumulation}"
  --learning-rate "${learning_rate}" --beta "${beta}" --max-grad-norm "${max_grad_norm}" --seed "${seed}"
  --expected-bernini-commit "${expected_bernini_commit}" --expected-veomni-commit "${expected_veomni_commit}"
  --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha}"
  --ack-experimental-no-action-success-claim
)

echo "[pair5-endpoint-v3] preflight before model/optimizer"
PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/train_pair_v5_endpoint_one_update_v3.py" "${common_args[@]}" --preflight-only
echo "[pair5-endpoint-v3] topology=WORLD8/DP2xSP4 exact81=true optimizer_updates=1"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "${method_root}/train_pair_v5_endpoint_one_update_v3.py" "${common_args[@]}"

[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "manifest changed"
"${python_bin}" -B - "${output}" "${generation_round}" "${parent_policy_digest}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
receipt=json.loads((root/'receipt.json').read_text(encoding='ascii'))
declared=receipt.pop('receipt_digest')
raw=json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
assert hashlib.sha256(raw).hexdigest()==declared
assert receipt['complete'] is True
assert receipt['optimizer_update_count']==1 and receipt['optimizer_step_indices']==[0]
assert receipt['generation_round']==int(sys.argv[2])
assert receipt['parent_policy_digest']==sys.argv[3]
assert receipt['static_rollout_reused_for_multiple_steps'] is False
assert receipt['fresh_next_round_rollout_required'] is True
assert receipt['distributed']['all_eight_gpus_used'] is True
assert receipt['semantic_action_editing_success'] is False
print('PAIR_V5_ENDPOINT_ONE_UPDATE_V3_STRONG_AUDIT_OK')
PY
echo "[pair5-endpoint-v3] PASS one-update only; fresh next-round rollout required"
