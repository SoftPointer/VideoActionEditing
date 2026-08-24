#!/usr/bin/env bash
#SBATCH --job-name=pair6-scaid
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${PAIR_V6_SOURCE_ARCHIVE:?set PAIR_V6_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V6_SOURCE_ARCHIVE_SHA256:?set PAIR_V6_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V6_SOURCE_REVISION:?set PAIR_V6_SOURCE_REVISION}"
manifest="${PAIR_V6_SCAID_MANIFEST:?set PAIR_V6_SCAID_MANIFEST}"
manifest_sha256="${PAIR_V6_SCAID_MANIFEST_SHA256:?set PAIR_V6_SCAID_MANIFEST_SHA256}"
evidence="${PAIR_V6_CAGD_V3_EVIDENCE:?set PAIR_V6_CAGD_V3_EVIDENCE}"
evidence_sha256="${PAIR_V6_CAGD_V3_EVIDENCE_SHA256:?set PAIR_V6_CAGD_V3_EVIDENCE_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${PAIR_V6_SCAID_OUTPUT_DIR:?set PAIR_V6_SCAID_OUTPUT_DIR}"
python_bin="${PAIR_V6_PYTHON_BIN:?set PAIR_V6_PYTHON_BIN}"
schedule_steps="${PAIR_V6_SCAID_SCHEDULE_STEPS:-40}"
learning_rate="${PAIR_V6_SCAID_LR:-1e-6}"
smoke_high_sigma_only="${PAIR_V6_SCAID_HIGH_SIGMA_SMOKE_ONLY:-0}"
smoke_receipt="${PAIR_V6_SCAID_HIGH_SIGMA_SMOKE_RECEIPT:-}"
smoke_receipt_sha256="${PAIR_V6_SCAID_HIGH_SIGMA_SMOKE_RECEIPT_SHA256:-}"

fail() { echo "[pair-v6-scaid] ERROR: $*" >&2; exit 2; }
for value in "${source_archive}" "${manifest}" "${evidence}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_dir}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for digest in "${source_archive_sha256}" "${manifest_sha256}" "${evidence_sha256}" "${checkpoint_manifest_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${schedule_steps}" =~ ^[0-9]+$ ]] || fail "schedule steps must be decimal"
[[ "${smoke_high_sigma_only}" == 0 || "${smoke_high_sigma_only}" == 1 ]] || fail "smoke flag must be 0 or 1"
if [[ "${smoke_high_sigma_only}" == 1 ]]; then
  (( schedule_steps == 1 )) || fail "high-sigma smoke requires exactly one step"
else
  (( schedule_steps >= 40 && schedule_steps % 40 == 0 )) || fail "schedule must be complete exact40 cycles"
  [[ -n "${smoke_receipt}" && "${smoke_receipt}" == /* ]] || fail "full run requires absolute high-sigma smoke receipt"
  [[ "${smoke_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "full run smoke receipt SHA-256 differs"
fi

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
evidence="$(realpath -e -- "${evidence}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
if [[ "${smoke_high_sigma_only}" == 0 ]]; then
  smoke_receipt="$(realpath -e -- "${smoke_receipt}")"
  [[ -f "${smoke_receipt}" && ! -L "${smoke_receipt}" ]] || fail "smoke receipt differs"
  [[ "$(sha256sum "${smoke_receipt}" | awk '{print $1}')" == "${smoke_receipt_sha256}" ]] || fail "smoke receipt hash differs"
fi
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "manifest differs"
[[ -f "${evidence}" && ! -L "${evidence}" ]] || fail "evidence differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint content manifest differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive revision differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest hash differs"
[[ "$(sha256sum "${evidence}" | awk '{print $1}')" == "${evidence_sha256}" ]] || fail "evidence hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint content manifest hash differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" && "${output_dir}" != / ]] || fail "output must be fresh non-root"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v6-scaid-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v6-scaid-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;; *) status=2 ;; esac
  exit "${status}"
}
trap cleanup EXIT
mkdir -p -- "${task_scratch}/source" "${task_scratch}/cache" "${task_scratch}/tmp"
cp -- "${source_archive}" "${task_scratch}/source.tar"

"${python_bin}" -B - "${task_scratch}/source.tar" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/train_pair_v6_scaid.py",
    "methods/bernini_action_editing/pair_v6_scaid_source_coordinate.py",
    "methods/bernini_action_editing/validate_pair_v5_cagd_evidence_v3.py",
    "methods/bernini_action_editing/finalize_pair_v5_t2v_cagd_v3.py",
    "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
    "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
    "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
    "methods/bernini_action_editing/mace_candidate_action_energy.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/inference_sigma_strata.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_native_target_adapter.py",
    "methods/bernini_action_editing/source_self_runtime.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/train_pair_v5_action_preference.py",
    "methods/bernini_action_editing/train_pair_v5_native_flow_dpo_v2.py",
    "methods/bernini_action_editing/train_pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/train_lora.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("unsafe source archive member")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks SCAID closure: {missing}")
PY
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
PYTHONPATH="${method_root}" "${python_bin}" -B -c \
  'import pair_v6_scaid_source_coordinate; import train_pair_v6_scaid'
export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export TMPDIR="${task_scratch}/tmp"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

master_port=$((30000 + SLURM_JOB_ID % 10000))
smoke_args=()
if [[ "${smoke_high_sigma_only}" == 1 ]]; then
  smoke_args+=(--smoke-high-sigma-only)
else
  smoke_args+=(
    --high-sigma-smoke-receipt "${smoke_receipt}"
    --expected-high-sigma-smoke-receipt-sha256 "${smoke_receipt_sha256}"
  )
fi
echo "[pair-v6-scaid] allocation=8xMI210 topology=DP2xSP4 frames=81 schedule=${schedule_steps} high_smoke=${smoke_high_sigma_only} low_zero=38,39"
"${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${method_root}/train_pair_v6_scaid.py" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
  --cagd-validator-evidence "${evidence}" \
  --expected-cagd-validator-evidence-sha256 "${evidence_sha256}" \
  --output "${output_dir}" --max-schedule-steps "${schedule_steps}" \
  "${smoke_args[@]}" \
  --learning-rate "${learning_rate}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-experimental-no-action-success-claim

[[ -f "${output_dir}/receipt.json" && -f "${output_dir}/adapter.safetensors" ]] || fail "completed artifacts absent"
echo "PAIR_V6_SCAID_DP2SP4_EXACT81_EXACT40_OK"
