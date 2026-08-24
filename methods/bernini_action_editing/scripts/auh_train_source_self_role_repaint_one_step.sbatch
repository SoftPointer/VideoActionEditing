#!/usr/bin/env bash
#SBATCH --job-name=bern-ss-repaint1
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

fail() {
  echo "[bernini-source-self-repaint] ERROR: $*" >&2
  exit 2
}

run_root="${BERNINI_SSR_RUN_ROOT:?set fresh absolute BERNINI_SSR_RUN_ROOT}"
spec="${BERNINI_SSR_SPEC:?set absolute BERNINI_SSR_SPEC}"
spec_sha256="${BERNINI_SSR_SPEC_SHA256:?set BERNINI_SSR_SPEC_SHA256}"
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_CHECKPOINT:?set BERNINI_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
python_bin="${BERNINI_SSR_PYTHON_BIN:?set BERNINI_SSR_PYTHON_BIN to vace Python}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in run_root spec source_archive source_repository bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  value="${!name}"
  [[ "${value}" == /* ]] || fail "${name} must be absolute"
done
[[ "${spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid spec SHA-256"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ -f "${spec}" && ! -L "${spec}" ]] || fail "spec must be a plain file"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive must be a plain file"
[[ -z "$(find "${spec}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable spec must be read-only"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable source archive must be read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "source repository is invalid"
[[ -d "${bernini_root}/.git" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be a plain executable"
[[ "${run_root}" != / && "${run_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe run root"
[[ "$(realpath -m -- "${run_root}")" == "${run_root}" ]] || fail "run root must be canonical"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root is create-only"
mkdir -p -- "$(dirname -- "${run_root}")"
[[ ! -L "$(dirname -- "${run_root}")" ]] || fail "run parent must not be a symlink"

[[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${spec_sha256}" ]] || fail "spec hash mismatch"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}")" == commit ]] || fail "source revision is not a commit"
[[ "$(git -C "${bernini_root}" rev-parse HEAD)" == "${expected_bernini_commit}" ]] || fail "Bernini revision mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${expected_veomni_commit}" ]] || fail "VeOmni revision mismatch"
[[ -z "$(git -C "${bernini_root}" status --porcelain --untracked-files=no)" ]] || fail "Bernini tracked source is dirty"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid SLURM job id"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-ssr-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$?
  trap - EXIT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-ssr-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-source-self-repaint] refusing unsafe scratch cleanup" >&2; exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

archive_copy="${task_scratch}/method.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive differs"
git_archive="${task_scratch}/git-method.tar"
git -C "${source_repository}" archive --format=tar --output="${git_archive}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${git_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive is not the declared commit subtree"

mkdir -p -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  source_self_role_repaint.py \
  source_self_runtime.py \
  train_source_self_role_repaint.py \
  tools/materialize_source_self_role_repaint.py \
  tools/materialize_ramp_motion_analogy_vae.py \
  tools/materialize_vae.py \
  train_lora.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive is missing ${required}"
done
find "${method_root}" -type f -exec chmod a-w {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

spec_copy="${task_scratch}/materialization-spec.json"
cp -- "${spec}" "${spec_copy}"
chmod a-w -- "${spec_copy}"
[[ "$(sha256sum "${spec_copy}" | awk '{print $1}')" == "${spec_sha256}" ]] || fail "staged spec differs"

mkdir -- "${run_root}"
materialized="${run_root}/materialized"
training="${run_root}/one_step"

PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/tools/materialize_source_self_role_repaint.py" \
  --spec "${spec_copy}" \
  --expected-spec-sha256 "${spec_sha256}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --expected-checkpoint-content-manifest-sha256 "${expected_checkpoint_manifest_sha256}" \
  --output "${materialized}" \
  --device cuda:0

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${method_root}/train_source_self_role_repaint.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --dataset-root "${materialized}" \
  --expected-materialization-spec-sha256 "${spec_sha256}" \
  --output "${training}" \
  --mode engineering-canary \
  --rho 0 \
  --adapter-block-scope early-mid-0-22 \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-upstream-training-use-forbidden

[[ "$(sha256sum "${spec}" | awk '{print $1}')" == "${spec_sha256}" ]] || fail "durable spec changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable archive changed"
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "extracted source changed"

"${python_bin}" -B - "${run_root}" "${spec_sha256}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from safetensors import safe_open

root = Path(sys.argv[1]).resolve(strict=True)
spec_sha, revision, archive_sha = sys.argv[2:5]
if {path.name for path in root.iterdir()} != {"materialized", "one_step"}:
    raise SystemExit("run-root artifact closure differs")
training = root / "one_step"
expected = {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}
if {path.name for path in training.iterdir()} != expected:
    raise SystemExit("training artifact closure differs")
receipt = json.loads((training / "receipt.json").read_text(encoding="utf-8"))
declared = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest() != declared:
    raise SystemExit("training receipt digest differs")
if receipt.get("complete") is not True or receipt.get("optimizer_steps") != 1:
    raise SystemExit("one-step completion contract differs")
if receipt.get("rho") != 0.0 or receipt.get("long_training_automatically_submitted") is not False:
    raise SystemExit("rho/long-training gate differs")
if receipt.get("method_source_revision") != revision or receipt.get("method_source_archive_sha256") != archive_sha:
    raise SystemExit("training source binding differs")
dataset_receipt = json.loads((root / "materialized" / "receipt.json").read_text(encoding="utf-8"))
if dataset_receipt.get("spec", {}).get("sha256") != spec_sha:
    raise SystemExit("materialized spec binding differs")
if (
    dataset_receipt.get("schema_version")
    != "bernini-source-self-role-repaint-dataset-receipt-v2"
    or dataset_receipt.get("independent_vae_encode_calls_per_row") != 6
    or dataset_receipt.get("all_six_calls_share_one_pinned_vae_identity") is not True
    or dataset_receipt.get("paired_dataset_accessed") is not False
    or dataset_receipt.get("prior_posterior_accessed") is not False
    or dataset_receipt.get("target_video_path_present") is not False
    or dataset_receipt.get("references_from_video_posterior_slice") is not False
):
    raise SystemExit("raw-source six-call materialization contract differs")
adapter_path = training / "adapter.safetensors"
if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != receipt["artifacts"]["adapter.safetensors"]:
    raise SystemExit("adapter hash differs")
with safe_open(str(adapter_path), framework="pt", device="cpu") as opened:
    metadata = dict(opened.metadata() or {})
    keys = list(opened.keys())
if metadata.get("schema_version") != "bernini-source-self-role-repaint-adapter-v1":
    raise SystemExit("adapter schema metadata differs")
if json.loads(metadata.get("block_indices_json", "null")) != list(range(23)):
    raise SystemExit("adapter block-range metadata differs")
if (
    metadata.get("conditional_base_schema")
    != "bernini-source-self-conditional-base-v1"
    or metadata.get("conditional_base_rho_hex") != float(0.0).hex()
    or metadata.get("inference_requires_identical_conditional_base_rho") != "true"
):
    raise SystemExit("adapter conditional-base metadata differs")
if not keys or "role_embedding.weight" not in keys:
    raise SystemExit("adapter tensor closure lacks role embedding")
PY

touch "${run_root}/ONE_STEP_ONLY_SCIENTIFIC_GATE_CLOSED"
chmod a-w -- "${run_root}/ONE_STEP_ONLY_SCIENTIFIC_GATE_CLOSED"
echo "[bernini-source-self-repaint] PASS one-step-only output=${run_root}"
