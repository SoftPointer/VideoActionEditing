#!/usr/bin/env bash
#SBATCH --job-name=oasis-phasea-preflight
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# This allocation validates the exact WORLD8/DP2xSP4 source/evidence plan.  It
# intentionally performs zero Bernini forwards: independent final-clean
# exact40 candidates and the registered re-noise scorer are not wired yet.

set -Eeuo pipefail
umask 027

fail() { echo "[oasis-phase-a-preflight] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

[[ "${OASIS_ACK_FROZEN_NO_TRAINING_NO_SUCCESS:-NO}" == YES ]] || \
  fail "set OASIS_ACK_FROZEN_NO_TRAINING_NO_SUCCESS=YES"
[[ "${OASIS_ACK_BACKEND_NOT_EXECUTED:-NO}" == YES ]] || \
  fail "set OASIS_ACK_BACKEND_NOT_EXECUTED=YES"

source_archive="${OASIS_PHASE_A_SOURCE_ARCHIVE:?set immutable git archive}"
source_archive_sha256="${OASIS_PHASE_A_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${OASIS_PHASE_A_SOURCE_REVISION:?set archive git revision}"
manifest="${OASIS_PHASE_A_MANIFEST:?set source-only Phase-A manifest}"
manifest_sha256="${OASIS_PHASE_A_MANIFEST_SHA256:?set manifest SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
checkpoint_tree_sha256="${BERNINI_CHECKPOINT_TREE_SHA256:?set checkpoint tree SHA-256}"
output="${OASIS_PHASE_A_PREFLIGHT_OUTPUT:?set fresh preflight output}"
python_bin="${OASIS_PHASE_A_PYTHON_BIN:?set vace Python executable}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

for name in source_archive manifest bernini_root veomni_root checkpoint checkpoint_manifest output python_bin; do
  absolute "${!name}" "${name}"
done
for name in source_archive_sha256 manifest_sha256 checkpoint_manifest_sha256 checkpoint_tree_sha256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${checkpoint_manifest_sha256}" == "${expected_checkpoint_manifest_sha256}" ]] || \
  fail "checkpoint content manifest is outside preregistration"
[[ "${checkpoint_tree_sha256}" == "${expected_checkpoint_tree_sha256}" ]] || \
  fail "checkpoint tree is outside preregistration"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for name in source_archive manifest checkpoint_manifest python_bin; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ ! -e "${output}" && ! -L "${output}" && "${output}" != / ]] || \
  fail "output must be fresh and non-root"

for pair in \
  "${source_archive}:${source_archive_sha256}" \
  "${manifest}:${manifest_sha256}" \
  "${checkpoint_manifest}:${checkpoint_manifest_sha256}"; do
  path="${pair%%:*}"
  digest="${pair##*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${digest}" ]] || \
    fail "sealed input SHA differs: ${path}"
done
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || \
  fail "source archive revision differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "${scratch_parent%/}/oasis-phasea-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/oasis-phasea-${SLURM_JOB_ID}."*) chmod -R u+w -- "${task_scratch}"; rm -rf -- "${task_scratch}" ;;
    *) status=2 ;;
  esac
  exit "${status}"
}
trap cleanup EXIT TERM INT
mkdir -p -- "${task_scratch}/source" "${task_scratch}/cache" "${task_scratch}/tmp"
cp -- "${source_archive}" "${task_scratch}/source.tar"

"${python_bin}" -B - "${task_scratch}/source.tar" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/oasis_phase_a_core.py",
    "methods/bernini_action_editing/oasis_phase_a_manifest.py",
    "methods/bernini_action_editing/infer_oasis_phase_a_frozen_controller.py",
    "methods/bernini_action_editing/infer_oasis_phase_a_noise_bank.py",
    "methods/bernini_action_editing/motion_null_appearance_noise.py",
    "methods/bernini_action_editing/scripts/auh_infer_oasis_phase_a_noise_bank_dual4.sbatch",
    "methods/bernini_action_editing/tools/build_oasis_phase_a_manifest.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_core.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_manifest.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_noise_bank.py",
    "methods/bernini_action_editing/tests/test_auh_oasis_phase_a_launcher.py",
    "methods/bernini_action_editing/tests/test_auh_oasis_phase_a_noise_bank_launcher.py",
    "methods/bernini_action_editing/tests/test_motion_null_appearance_noise.py",
    "methods/bernini_action_editing/mace_candidate_action_energy.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_native_target_adapter.py",
    "methods/bernini_action_editing/source_self_runtime.py",
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
    raise SystemExit(f"source archive lacks OASIS Phase-A closure: {missing}")
PY
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest \
  "${method_root}/tests/test_oasis_phase_a_core.py" \
  "${method_root}/tests/test_oasis_phase_a_manifest.py" \
  "${method_root}/tests/test_auh_oasis_phase_a_launcher.py"

export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export TMPDIR="${task_scratch}/tmp"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL

master_port=$((30000 + SLURM_JOB_ID % 10000))
echo "[oasis-phase-a-preflight] allocation=8xMI210 topology=WORLD8/DP2xSP4 exact81=true exact40=true low38_39=base-off final_clean_probe=true arbitrary_state_oracle=NO-GO runtime=false"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="${master_port}" \
  "${method_root}/infer_oasis_phase_a_frozen_controller.py" \
  --manifest "${manifest}" \
  --expected-manifest-sha256 "${manifest_sha256}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --output "${output}" \
  --preflight-only \
  --ack-frozen-no-training-no-success-claim \
  --ack-bernini-controller-backend-not-yet-executed

for pair in \
  "${source_archive}:${source_archive_sha256}" \
  "${manifest}:${manifest_sha256}" \
  "${checkpoint_manifest}:${checkpoint_manifest_sha256}"; do
  path="${pair%%:*}"
  digest="${pair##*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${digest}" ]] || \
    fail "sealed input changed during preflight: ${path}"
done
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "read-only extracted method tree changed"

"${python_bin}" -B - "${output}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
if root.is_symlink() or {item.name for item in root.iterdir()} != {"preflight-receipt.json"}:
    raise SystemExit("OASIS preflight artifact closure differs")
path = root / "preflight-receipt.json"
receipt = json.loads(path.read_text(encoding="ascii"))
declared = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest() != declared:
    raise SystemExit("OASIS preflight receipt digest differs")
boundary = receipt.get("boundary_contract", {})
if not (
    receipt.get("preflight_complete") is True
    and receipt.get("controller_runtime_executed") is False
    and receipt.get("frozen_oracle_rollouts_executed") == 0
    and receipt.get("training_performed") is False
    and receipt.get("optimizer_constructed") is False
    and receipt.get("parameter_mutation_performed") is False
    and receipt.get("scientific_action_editing_success_claim") is False
    and boundary.get("frozen_oracle_execution_authorized") is False
    and boundary.get("model_forward_count") == 0
    and boundary.get("native_scheduler_step_count") == 0
):
    raise SystemExit("OASIS preflight/no-execution closure differs")
print("BERNINI_OASIS_PHASE_A_DP2SP4_PREFLIGHT_ONLY_OK")
PY

echo "[oasis-phase-a-preflight] COMPLETE output=${output} controller_runtime=false rollouts=0 training=false success_claim=false"
