#!/usr/bin/env bash
#SBATCH --job-name=saic-typed-sc
#SBATCH --partition=faculty
# QOS is intentionally selected at submission: sbatch --qos=gtqos ...
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${SAIC_TYPED_SOURCE_ARCHIVE:?set SAIC_TYPED_SOURCE_ARCHIVE}"
source_archive_sha256="${SAIC_TYPED_SOURCE_ARCHIVE_SHA256:?set SAIC_TYPED_SOURCE_ARCHIVE_SHA256}"
source_revision="${SAIC_TYPED_SOURCE_REVISION:?set SAIC_TYPED_SOURCE_REVISION}"
manifest="${SAIC_TYPED_SCAID_MANIFEST:?set SAIC_TYPED_SCAID_MANIFEST}"
manifest_sha256="${SAIC_TYPED_SCAID_MANIFEST_SHA256:?set SAIC_TYPED_SCAID_MANIFEST_SHA256}"
evidence="${SAIC_TYPED_CAGD_V3_EVIDENCE:?set SAIC_TYPED_CAGD_V3_EVIDENCE}"
evidence_sha256="${SAIC_TYPED_CAGD_V3_EVIDENCE_SHA256:?set SAIC_TYPED_CAGD_V3_EVIDENCE_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${SAIC_TYPED_OUTPUT_DIR:?set SAIC_TYPED_OUTPUT_DIR}"
python_bin="${SAIC_TYPED_PYTHON_BIN:?set SAIC_TYPED_PYTHON_BIN}"
schedule_steps="${SAIC_TYPED_SCHEDULE_STEPS:-40}"
learning_rate="${SAIC_TYPED_LR:-1e-6}"
smoke_only="${SAIC_TYPED_HIGH_SIGMA_SMOKE_ONLY:-0}"

fail() { echo "[saic-typed-sc] ERROR: $*" >&2; exit 2; }
for value in "${source_archive}" "${manifest}" "${evidence}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_dir}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for digest in "${source_archive_sha256}" "${manifest_sha256}" "${evidence_sha256}" "${checkpoint_manifest_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${schedule_steps}" =~ ^[0-9]+$ ]] || fail "schedule steps must be decimal"
[[ "${smoke_only}" == 0 || "${smoke_only}" == 1 ]] || fail "smoke flag must be zero or one"
if [[ "${smoke_only}" == 1 ]]; then
  (( schedule_steps == 1 )) || fail "high-sigma smoke requires one cell"
else
  (( schedule_steps >= 40 && schedule_steps % 40 == 0 )) || fail "training requires complete exact40 cycles"
fi

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
evidence="$(realpath -e -- "${evidence}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "manifest differs"
[[ -f "${evidence}" && ! -L "${evidence}" ]] || fail "evidence differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha256}" ]] || fail "manifest hash differs"
[[ "$(sha256sum "${evidence}" | awk '{print $1}')" == "${evidence_sha256}" ]] || fail "evidence hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" && "${output_dir}" != / ]] || fail "output must be fresh and non-root"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-typed-sc-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/saic-typed-sc-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;;
    *) status=2 ;;
  esac
  exit "${status}"
}
trap cleanup EXIT TERM INT
mkdir -p -- "${task_scratch}/source" "${task_scratch}/cache" "${task_scratch}/tmp"
cp -- "${source_archive}" "${task_scratch}/source.tar"

"${python_bin}" -B - "${task_scratch}/source.tar" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/train_saic_typed_source_coordinate_v1.py",
    "methods/bernini_action_editing/saic_typed_action_operator_v1.py",
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
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_pair_v5_action_preference.py",
    "methods/bernini_action_editing/train_pair_v5_native_flow_dpo_v2.py",
    "methods/bernini_action_editing/train_pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/pair_v5_action_energy_calibration.py",
    "methods/bernini_action_editing/pair_v5_candidate_evaluator_packet.py",
    "methods/bernini_action_editing/pair_v5_flow_dpo.py",
    "methods/bernini_action_editing/pair_v5_native_rollout_spec.py",
    "methods/bernini_action_editing/pair_v5_safe_pareto.py",
    "methods/bernini_action_editing/pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
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
    raise SystemExit(f"source archive lacks typed pilot closure: {missing}")
PY
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
PYTHONPATH="${method_root}" "${python_bin}" -B -c \
  'import saic_typed_action_operator_v1; import train_saic_typed_source_coordinate_v1'

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export TMPDIR="${task_scratch}/tmp"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

master_port=$((30000 + SLURM_JOB_ID % 10000))
smoke_args=()
if [[ "${smoke_only}" == 1 ]]; then
  smoke_args+=(--smoke-high-sigma-only)
fi
echo "[saic-typed-sc] allocation=8xMI210 topology=WORLD8/DP2xSP4 frames=exact81 schedule=${schedule_steps} low_base=38,39 event_gate=none inverse=none"
"${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${method_root}/train_saic_typed_source_coordinate_v1.py" \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --manifest "${manifest}" --expected-manifest-sha256 "${manifest_sha256}" \
  --cagd-validator-evidence "${evidence}" \
  --expected-cagd-validator-evidence-sha256 "${evidence_sha256}" \
  --output "${output_dir}" --max-schedule-steps "${schedule_steps}" \
  --learning-rate "${learning_rate}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  "${smoke_args[@]}" \
  --ack-typed-source-coordinate-pilot-no-semantic-success-claim

[[ -f "${output_dir}/receipt.json" && -f "${output_dir}/operator.pt" && -f "${output_dir}/history.json" ]] || fail "completed typed artifacts are absent"
echo "SAIC_TYPED_SOURCE_COORDINATE_WORLD8_EXACT81_OK"
