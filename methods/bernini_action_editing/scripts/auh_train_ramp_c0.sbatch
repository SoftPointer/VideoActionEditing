#!/usr/bin/env bash
#SBATCH --job-name=bern-ramp-c0
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
  echo "[bernini-ramp-c0] ERROR: $*" >&2
  exit 2
}

mode="${BERNINI_RAMP_C0_MODE:?set BERNINI_RAMP_C0_MODE=engineering-canary or afterok-c0}"
output="${BERNINI_RAMP_C0_OUTPUT:?set fresh absolute BERNINI_RAMP_C0_OUTPUT}"
pair_config="${BERNINI_RAMP_C0_PAIR_CONFIG:?set BERNINI_RAMP_C0_PAIR_CONFIG}"
pair_config_sha256="${BERNINI_RAMP_C0_PAIR_CONFIG_SHA256:?set BERNINI_RAMP_C0_PAIR_CONFIG_SHA256}"
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_CHECKPOINT:?set BERNINI_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
python_bin="${BERNINI_RAMP_C0_PYTHON_BIN:?set BERNINI_RAMP_C0_PYTHON_BIN to vace Python}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

case "${mode}" in
  engineering-canary) expected_steps=1 ;;
  afterok-c0) expected_steps=16 ;;
  *) fail "BERNINI_RAMP_C0_MODE must be engineering-canary or afterok-c0" ;;
esac

for name in output pair_config source_archive source_repository bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  value="${!name}"
  [[ "${value}" == /* ]] || fail "${name} must be absolute"
done
[[ "${pair_config_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid pair-config SHA-256"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ -f "${pair_config}" && ! -L "${pair_config}" ]] || fail "pair config must be a plain file"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method source archive must be a plain file"
[[ -z "$(find "${pair_config}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable pair config must be read-only"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable method archive must be read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "method source repository is invalid"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be a plain executable"
[[ "${output}" != / && "${output}" != *.json && "${output}" != *.pt ]] || fail "unsafe output path"
[[ "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe output basename"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "refusing to reuse output"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent must not be a symlink"

[[ "$(sha256sum "${pair_config}" | awk '{print $1}')" == "${pair_config_sha256}" ]] || fail "pair config hash mismatch"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method source archive hash mismatch"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}")" == commit ]] || fail "method revision is not a commit"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${expected_veomni_commit}" ]] || fail "VeOmni revision mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content manifest verification failed"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-ramp-c0-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local status=$?
  trap - EXIT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-ramp-c0-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-ramp-c0] refusing unsafe scratch cleanup" >&2; exit 2 ;;
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
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged method archive differs"
git_archive="${task_scratch}/git-commit-method.tar"
git -C "${source_repository}" archive --format=tar --output="${git_archive}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${git_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive bytes are not the declared commit"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if not path.parts[:2] == ("methods", "bernini_action_editing"):
            raise SystemExit(f"archive escaped method subtree: {member.name}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"non-plain archive member: {member.name}")
PY

mkdir -p -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  train_ramp_c0.py \
  ramp_target_row_lora.py \
  ramp_same_state_route_objective.py \
  mdr_exact_motion_analogy.py \
  train_lora.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive is missing ${required}"
done
find "${method_root}" -type f -exec chmod a-w {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

pair_copy="${task_scratch}/pair-config.json"
cp -- "${pair_config}" "${pair_copy}"
chmod a-w -- "${pair_copy}"
[[ "$(sha256sum "${pair_copy}" | awk '{print $1}')" == "${pair_config_sha256}" ]] || fail "staged pair config differs"

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${method_root}/train_ramp_c0.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --pair-config "${pair_copy}" \
  --expected-pair-config-sha256 "${pair_config_sha256}" \
  --output "${output}" \
  --mode "${mode}" \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-upstream-training-use-forbidden

[[ "$(sha256sum "${pair_config}" | awk '{print $1}')" == "${pair_config_sha256}" ]] || fail "durable pair config changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "extracted method source changed"

"${python_bin}" -B - "${output}" "${mode}" "${expected_steps}" "${pair_config_sha256}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
mode, steps = sys.argv[2], int(sys.argv[3])
pair_sha, revision, archive_sha = sys.argv[4:7]
expected = {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}
entries = list(root.iterdir())
if any(not path.is_file() or path.is_symlink() for path in entries):
    raise SystemExit("output contains a non-plain artifact")
actual = {path.name for path in entries}
if actual != expected:
    raise SystemExit(f"output artifact set differs: {actual}")
receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
declared = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
if hashlib.sha256(payload).hexdigest() != declared:
    raise SystemExit("run receipt digest differs")
if receipt.get("complete") is not True or receipt.get("mode") != mode or receipt.get("optimizer_steps") != steps:
    raise SystemExit("run completion/mode/step contract differs")
if receipt.get("pair_config", {}).get("sha256") != pair_sha:
    raise SystemExit("receipt pair-config binding differs")
if receipt.get("method_source_revision") != revision or receipt.get("method_source_archive_sha256") != archive_sha:
    raise SystemExit("receipt source binding differs")
for name, digest in receipt.get("artifacts", {}).items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f"artifact digest differs: {name}")
PY

echo "[bernini-ramp-c0] PASS mode=${mode} steps=${expected_steps} output=${output}"
