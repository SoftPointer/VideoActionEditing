#!/usr/bin/env bash
#SBATCH --job-name=pair5-native-dpo-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=48:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() { echo "[pair-v5-native-dpo-v2] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

output="${PAIR_V5_DPO_OUTPUT:?set fresh absolute PAIR_V5_DPO_OUTPUT}"
manifest="${PAIR_V5_DPO_MANIFEST:?set sealed safe-Pareto manifest}"
manifest_sha256="${PAIR_V5_DPO_MANIFEST_SHA256:?set manifest raw SHA-256}"
cio_adapter="${PAIR_V5_DPO_CIO_ADAPTER:?set frozen CIO adapter}"
cio_adapter_sha256="${PAIR_V5_DPO_CIO_ADAPTER_SHA256:?set frozen CIO adapter SHA-256}"
cio_receipt="${PAIR_V5_DPO_CIO_RECEIPT:?set frozen CIO training receipt}"
cio_receipt_sha256="${PAIR_V5_DPO_CIO_RECEIPT_SHA256:?set frozen CIO receipt SHA-256}"
source_archive="${PAIR_V5_DPO_SOURCE_ARCHIVE:?set immutable method source archive}"
source_archive_sha256="${PAIR_V5_DPO_SOURCE_ARCHIVE_SHA256:?set source archive SHA-256}"
source_revision="${PAIR_V5_DPO_SOURCE_REVISION:?set full method source revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${PAIR_V5_DPO_PYTHON_BIN:?set vace Python executable}"
schedule_steps="${PAIR_V5_DPO_SCHEDULE_STEPS:-40}"
gradient_accumulation="${PAIR_V5_DPO_GRADIENT_ACCUMULATION:-1}"
learning_rate="${PAIR_V5_DPO_LEARNING_RATE:-1e-6}"
beta="${PAIR_V5_DPO_BETA:-1000}"
max_grad_norm="${PAIR_V5_DPO_MAX_GRAD_NORM:-1.0}"
seed="${PAIR_V5_DPO_SEED:-20260808}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in output manifest cio_adapter cio_receipt source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  absolute "${!name}" "${name}"
done
for digest in manifest_sha256 cio_adapter_sha256 cio_receipt_sha256 source_archive_sha256; do
  [[ "${!digest}" =~ ^[0-9a-f]{64}$ ]] || fail "${digest} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${schedule_steps}" =~ ^[0-9]+$ && "${schedule_steps}" -ge 40 && $((schedule_steps % 40)) -eq 0 ]] || fail "schedule steps must be positive exact40 cycles"
[[ "${gradient_accumulation}" =~ ^[0-9]+$ && "${gradient_accumulation}" -ge 1 && "${gradient_accumulation}" -le 16 ]] || fail "gradient accumulation differs"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed differs"

manifest="$(realpath -e -- "${manifest}")"
cio_adapter="$(realpath -e -- "${cio_adapter}")"
cio_receipt="$(realpath -e -- "${cio_receipt}")"
source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for path in manifest cio_adapter cio_receipt source_archive checkpoint_manifest python_bin; do
  [[ -f "${!path}" && ! -L "${!path}" ]] || fail "${path} must be a plain file"
done
for path in bernini_root veomni_root checkpoint; do
  [[ -d "${!path}" && ! -L "${!path}" ]] || fail "${path} must be a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest bytes differ"
[[ "$(sha256sum "${cio_adapter}" | awk '{print $1}')" == "${cio_adapter_sha256}" ]] || fail "CIO adapter bytes differ"
[[ "$(sha256sum "${cio_receipt}" | awk '{print $1}')" == "${cio_receipt_sha256}" ]] || fail "CIO receipt bytes differ"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest differs"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-native-dpo-v2-${SLURM_JOB_ID}.XXXXXX")"
mkdir -- "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" "${task_scratch}/triton" \
  "${task_scratch}/pycache" "${task_scratch}/tmp" "${task_scratch}/source"
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
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-native-dpo-v2-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
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
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "staged source revision differs"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for relative in \
  train_pair_v5_native_flow_dpo_v2.py \
  train_pair_v5_action_preference.py \
  pair_v5_action_adapter.py \
  pair_v5_action_energy_calibration.py \
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
  tests/test_train_pair_v5_native_flow_dpo_v2.py \
  scripts/auh_train_pair_v5_native_flow_dpo_v2.sbatch; do
  [[ -f "${method_root}/${relative}" && ! -L "${method_root}/${relative}" ]] || fail "archive lacks ${relative}"
done
find "${method_root}" -type f -exec chmod a-w -- {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
  -s "${method_root}/tests" -p 'test_train_pair_v5_native_flow_dpo_v2.py'

common_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --pair-manifest "${manifest}"
  --expected-pair-manifest-sha256 "${manifest_sha256}"
  --frozen-cio-adapter "${cio_adapter}"
  --expected-frozen-cio-adapter-sha256 "${cio_adapter_sha256}"
  --frozen-cio-receipt "${cio_receipt}"
  --expected-frozen-cio-receipt-sha256 "${cio_receipt_sha256}"
  --output "${output}"
  --max-schedule-steps "${schedule_steps}"
  --gradient-accumulation-steps "${gradient_accumulation}"
  --learning-rate "${learning_rate}"
  --beta "${beta}"
  --max-grad-norm "${max_grad_norm}"
  --seed "${seed}"
  --num-frames 81
  --expected-bernini-commit "${expected_bernini_commit}"
  --expected-veomni-commit "${expected_veomni_commit}"
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
  --method-source-revision "${source_revision}"
  --method-source-archive-sha256 "${source_archive_sha256}"
  --ack-experimental-no-action-success-claim
)

echo "[pair-v5-native-dpo-v2] preflight before renderer/model/optimizer"
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/train_pair_v5_native_flow_dpo_v2.py" \
  "${common_args[@]}" --preflight-only

echo "[pair-v5-native-dpo-v2] allocation=8xMI210 topology=WORLD8/DP2xSP4 exact81=true schedule=${schedule_steps} low38_39=zero-update-audit"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "${method_root}/train_pair_v5_native_flow_dpo_v2.py" \
  "${common_args[@]}"

[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest changed"
[[ "$(sha256sum "${cio_adapter}" | awk '{print $1}')" == "${cio_adapter_sha256}" ]] || fail "CIO adapter changed"
[[ "$(sha256sum "${cio_receipt}" | awk '{print $1}')" == "${cio_receipt_sha256}" ]] || fail "CIO receipt changed"
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "extracted method tree changed"

"${python_bin}" -B - "${output}" "${schedule_steps}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
steps=int(sys.argv[2])
expected={"adapter.safetensors","optimizer.pt","history.json","receipt.json"}
if root.is_symlink() or {p.name for p in root.iterdir()} != expected:
    raise SystemExit("training artifact closure differs")
receipt=json.loads((root/"receipt.json").read_text(encoding="ascii"))
declared=receipt.pop("receipt_digest",None)
payload=json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest()!=declared:
    raise SystemExit("run receipt digest differs")
updates=(steps//40)*38
exact=receipt.get("exact40",{})
if not (
    receipt.get("complete") is True
    and receipt.get("schedule_steps")==steps
    and receipt.get("optimizer_updates")==updates
    and exact.get("complete_exact40_dynamic_coverage") is True
    and exact.get("indices_38_39_flow_dpo_constructed") is False
    and exact.get("indices_38_39_optimizer_step_called") is False
    and receipt.get("distributed",{}).get("all_eight_gpus_used") is True
    and receipt.get("semantic_action_editing_success") is False
    and receipt.get("scientific_generalization_claim_authorized") is False
):
    raise SystemExit("run/scientific contract differs")
for name in expected-{"receipt.json"}:
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=receipt["artifacts"][name]:
        raise SystemExit(f"artifact hash differs: {name}")
print("PAIR_V5_NATIVE_FLOW_DPO_V2_STRONG_AUDIT_OK")
PY

echo "[pair-v5-native-dpo-v2] PASS output=${output} action_success_claim=false"
