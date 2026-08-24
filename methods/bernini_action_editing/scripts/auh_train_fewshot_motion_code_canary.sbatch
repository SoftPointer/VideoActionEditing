#!/usr/bin/env bash
#SBATCH --job-name=bern-epmc-train
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

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
preview_manifest="${BERNINI_EPMC_PREVIEW_MANIFEST:?set BERNINI_EPMC_PREVIEW_MANIFEST}"
vae_index="${BERNINI_EPMC_VAE_INDEX:?set BERNINI_EPMC_VAE_INDEX}"
mode="${BERNINI_EPMC_MODE:?set BERNINI_EPMC_MODE=smoke or formal}"
output_dir="${BERNINI_EPMC_TRAIN_OUTPUT_DIR:?set BERNINI_EPMC_TRAIN_OUTPUT_DIR}"
python_bin="${BERNINI_EPMC_PYTHON_BIN:?set BERNINI_EPMC_PYTHON_BIN to the vace Python with PyAV 13.1.0}"

expected_config_sha256="a46d18fce025b0cd3b30a6505514b817cd5d96c43d305d6202405a952eed2446"
expected_preview_manifest_sha256="49506e003f86f319ebe8a5e843d19c88cef75e84cd4250968da283bb19252e47"
expected_vae_index_sha256="d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
expected_trainer_sha256="53424538b0685e5e9a226d5b9729bf61546116fb4d853298fe9475ce06cb4851"
expected_trainer_contract_test_sha256="e2f7e6c1b4b7b1795b75240eabe3acf443ace27b6622e6897c3e2cda0f7bc49f"
expected_episode_parallel_sha256="220fc85c24fe9cf781e5908cfc548e5ca4acf54ad87a5c00bab7b2cdbefa70ec"
expected_episode_parallel_test_sha256="a37d2242516bcb911db311c5c96660410c1cb66163bb1e7515b20a67bbc75415"
expected_ffprobe_helper_sha256="c2739db02e1e3abedebd5b7d84dc253432884dc797d4e16102df8362f158c23f"
expected_pyav_version="13.1.0"
episode_probe_iid="841b5e0080a1441d"
episode_probe_source_sha256="5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_content_file_count=23
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
num_frames=81
k_shot=2
steps_per_support=50
learning_rate="0.05"
max_grad_norm="1.0"
training_seed=20260808
proposal_seed=2027
fixed_sigma_index=20
held_sigma_index=32
full_target_fm_weight="0.0"
expected_world_size=8
expected_ulysses_size=4
expected_data_parallel_size=2
expected_within_support_gradient_sync="ulysses_group_all_reduce_sum_then_divide_by_4"

fail() {
  echo "[bernini-epmc-train] ERROR: $*" >&2
  exit 2
}

validate_ffprobe_shim() {
  local phase="$1"
  local actual_sha256
  local actual_pyav_version
  local resolved
  [[ "${ffprobe_bin}" == "${task_scratch}/bin/ffprobe" ]] || fail "ffprobe must be the task-private PyAV shim"
  [[ -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" && -x "${ffprobe_bin}" ]] || fail "ffprobe must be a plain executable non-symlink"
  [[ "$(stat -c '%a' "${ffprobe_bin}")" == 555 ]] || fail "ffprobe PyAV shim mode must remain 0555"
  actual_sha256="$(sha256sum "${ffprobe_bin}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "${expected_ffprobe_helper_sha256}" ]] || fail "ffprobe PyAV shim hash mismatch"
  [[ "${BERNINI_EPMC_PYTHON_BIN}" == "${python_bin}" ]] || fail "ffprobe runtime Python binding differs"
  [[ "${BERNINI_EPMC_PYTHON_BIN}" == /* && -f "${BERNINI_EPMC_PYTHON_BIN}" && -x "${BERNINI_EPMC_PYTHON_BIN}" ]] || fail "ffprobe runtime Python must resolve from an absolute executable file"
  actual_pyav_version="$("${BERNINI_EPMC_PYTHON_BIN}" -B -c 'import av; print(av.__version__)')"
  [[ "${actual_pyav_version}" == "${expected_pyav_version}" ]] || fail "PyAV version mismatch"
  resolved="$(command -v ffprobe || true)"
  [[ "${resolved}" == "${ffprobe_bin}" ]] || fail "PATH ffprobe resolution is not the pinned executable"
  echo "[bernini-epmc-train] media_probe_phase=${phase} media_probe_backend=pyav_ffprobe_shim ffprobe_bin=${ffprobe_bin} ffprobe_helper_sha256=${actual_sha256} pyav_version=${actual_pyav_version}"
}

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not a plain file"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_content_manifest_sha256}" ]] || fail "checkpoint manifest digest mismatch"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint manifest file count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains a non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_content_file_count}" ]] || fail "checkpoint non-cache file set differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content hash differs"
}

method_tree_digest() {
  "${python_bin}" -B - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
if root.is_symlink():
    raise SystemExit("method root is a symlink")
root = root.resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"method source contains symlink: {relative}")
    if stat.S_ISDIR(mode):
        if path.name == "__pycache__":
            raise SystemExit("method source contains __pycache__")
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"method source contains non-regular entry: {relative}")
    if path.suffix in (".pyc", ".pyo"):
        raise SystemExit(f"method source contains bytecode: {relative}")
    if mode & 0o222:
        raise SystemExit(f"method source file is writable: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    rows.append({"path": relative, "sha256": digest.hexdigest()})
payload = json.dumps(
    rows,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

output_tree_digest() {
  "${python_bin}" -B - "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
rows = []
for path in sorted(root.iterdir(), key=lambda item: item.name):
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise SystemExit(f"output contains non-regular artifact: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    rows.append({"name": path.name, "sha256": digest.hexdigest()})
payload = json.dumps(
    rows,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

case "${mode}" in
  smoke)
    mode_args=(--engineering-smoke)
    receipt_name="smoke.receipt.json"
    expected_output_count=2
    ;;
  formal)
    mode_args=()
    receipt_name="receipt.json"
    expected_output_count=5
    ;;
  *) fail "BERNINI_EPMC_MODE must be exactly smoke or formal" ;;
esac

for value_name in source_archive source_repository bernini_root bernini_archive veomni_root checkpoint preview_manifest vae_index output_dir python_bin; do
  value="${!value_name}"
  [[ "${value}" == /* ]] || fail "${value_name} must be an absolute path"
done
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid durable method source archive"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "durable method archive must be read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -f "${preview_manifest}" && ! -L "${preview_manifest}" ]] || fail "invalid preview manifest"
[[ -f "${vae_index}" && ! -L "${vae_index}" ]] || fail "invalid VAE index"
[[ "${preview_manifest}" != "${vae_index}" ]] || fail "preview manifest and VAE index paths must differ"
[[ "${python_bin}" == /* && -f "${python_bin}" && -x "${python_bin}" ]] || fail "BERNINI_EPMC_PYTHON_BIN must resolve from an absolute executable file"
export BERNINI_EPMC_PYTHON_BIN="${python_bin}"
[[ "$("${BERNINI_EPMC_PYTHON_BIN}" -B -c 'import av; print(av.__version__)')" == "${expected_pyav_version}" ]] || fail "BERNINI_EPMC_PYTHON_BIN must provide pinned PyAV 13.1.0"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method revision"
[[ "${output_dir}" != / ]] || fail "output cannot be filesystem root"
output_name="${output_dir##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9_-]+$ ]] || fail "output basename contains unsupported characters or a suffix"
[[ "${output_name}" =~ (^|[-_])${mode}($|[-_]) ]] || fail "output basename must explicitly contain the smoke/formal mode label"
canonical_output_dir="$(realpath -m -- "${output_dir}")"
[[ "${canonical_output_dir}" == "${output_dir}" ]] || fail "output must be canonical and contain no symlink traversal"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse training output"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent must be a non-symlink directory"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${preview_manifest}" | awk '{print $1}')" == "${expected_preview_manifest_sha256}" ]] || fail "preview manifest hash mismatch"
[[ "$(sha256sum "${vae_index}" | awk '{print $1}')" == "${expected_vae_index_sha256}" ]] || fail "VAE index hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni revision mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid SLURM_JOB_ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-epmc-train-${mode}-${SLURM_JOB_ID}.XXXXXX")"

cleanup() {
  local exit_status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  if [[ -n "${task_scratch:-}" ]]; then
    case "${task_scratch}" in
      "${scratch_parent%/}/bernini-epmc-train-${mode}-${SLURM_JOB_ID}."*) ;;
      *) echo "[bernini-epmc-train] refusing unsafe scratch cleanup: ${task_scratch}" >&2; cleanup_failed=1 ;;
    esac
    if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
      chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
      rm -rf -- "${task_scratch}" || cleanup_failed=1
    fi
    [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" != 0 && "${exit_status}" == 0 ]]; then
    exit_status=2
  fi
  if [[ "${exit_status}" == 0 ]]; then
    echo "[bernini-epmc-train] PASS mode=${mode} cleanup_verified=true output=${output_dir}"
  else
    echo "[bernini-epmc-train] EXIT mode=${mode} status=${exit_status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/input" \
  "${task_scratch}/bin" \
  "${task_scratch}/pycache-prefix" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache-prefix"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

staged_preview_manifest="${task_scratch}/input/preview_manifest.jsonl"
staged_vae_index="${task_scratch}/input/vae_index.jsonl"
cp -- "${preview_manifest}" "${staged_preview_manifest}"
cp -- "${vae_index}" "${staged_vae_index}"
chmod a-w -- "${staged_preview_manifest}" "${staged_vae_index}"
[[ "$(sha256sum "${staged_preview_manifest}" | awk '{print $1}')" == "${expected_preview_manifest_sha256}" ]] || fail "staged preview manifest hash differs"
[[ "$(sha256sum "${staged_vae_index}" | awk '{print $1}')" == "${expected_vae_index_sha256}" ]] || fail "staged VAE index hash differs"

archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod a-w -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive hash differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method revision is not a commit"
git -C "${source_repository}" archive --format=tar -o "${archive_from_revision}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive bytes are not the declared commit"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        allowed_parent = path.parts in (("methods",), ("methods", "bernini_action_editing"))
        allowed_child = path.parts[:2] == ("methods", "bernini_action_editing")
        if not (allowed_parent or allowed_child):
            raise SystemExit(f"archive member escaped method subtree: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  train_fewshot_motion_code.py \
  fewshot_episode_io.py \
  fewshot_episode_parallel.py \
  inference_sigma_strata.py \
  motion_residual.py \
  train_lora.py \
  train_prior_tangent_lora.py \
  prior_guided_tangent.py \
  train_delta_lora.py \
  tri_branch_unipc.py \
  counterfactual_proposal_motion_rebinding.py \
  counterfactual_proposal_motion_branch.py \
  counterfactual_proposal_motion_runtime.py \
  fewshot_proposal_motion_carrier.py \
  fewshot_motion_branch.py \
  fewshot_privileged_motion_code.py \
  fewshot_teacher_objective.py \
  infer_fewshot_motion_code.py \
  infer_counterfactual_proposal_motion_oracle.py \
  infer_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_route_batches.py \
  source_kv_replay.py \
  source_value_residual.py \
  tools/ffprobe_pyav_compat.py \
  configs/epmc_sit_turn_head_k2_v1.json \
  scripts/auh_train_fewshot_motion_code_canary.sbatch \
  tests/test_train_fewshot_motion_code_contract.py \
  tests/test_fewshot_episode_parallel.py \
  tests/test_fewshot_proposal_motion_carrier.py \
  tests/test_ffprobe_pyav_compat.py \
  tests/test_auh_train_fewshot_motion_code_canary_launcher.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "extracted method tree contains a symlink"
ffprobe_helper="${method_root}/tools/ffprobe_pyav_compat.py"
[[ -f "${ffprobe_helper}" && ! -L "${ffprobe_helper}" ]] || fail "archive lacks plain PyAV ffprobe helper"
[[ "$(sha256sum "${ffprobe_helper}" | awk '{print $1}')" == "${expected_ffprobe_helper_sha256}" ]] || fail "source PyAV ffprobe helper hash differs"
ffprobe_bin="${task_scratch}/bin/ffprobe"
[[ ! -e "${ffprobe_bin}" && ! -L "${ffprobe_bin}" ]] || fail "task-private ffprobe destination already exists"
cp -- "${ffprobe_helper}" "${ffprobe_bin}"
chmod 0555 -- "${ffprobe_bin}"
[[ "$(sha256sum "${ffprobe_bin}" | awk '{print $1}')" == "${expected_ffprobe_helper_sha256}" ]] || fail "copied PyAV ffprobe helper hash differs"
export PATH="${task_scratch}/bin${PATH:+:${PATH}}"
hash -r
validate_ffprobe_shim installed
trainer_path="${method_root}/train_fewshot_motion_code.py"
trainer_contract_test_path="${method_root}/tests/test_train_fewshot_motion_code_contract.py"
episode_parallel_path="${method_root}/fewshot_episode_parallel.py"
episode_parallel_test_path="${method_root}/tests/test_fewshot_episode_parallel.py"
[[ "$(sha256sum "${trainer_path}" | awk '{print $1}')" == "${expected_trainer_sha256}" ]] || fail "final-stable trainer hash differs"
[[ "$(sha256sum "${trainer_contract_test_path}" | awk '{print $1}')" == "${expected_trainer_contract_test_sha256}" ]] || fail "final-stable trainer contract test hash differs"
[[ "$(sha256sum "${episode_parallel_path}" | awk '{print $1}')" == "${expected_episode_parallel_sha256}" ]] || fail "final-stable episode-parallel module hash differs"
[[ "$(sha256sum "${episode_parallel_test_path}" | awk '{print $1}')" == "${expected_episode_parallel_test_sha256}" ]] || fail "final-stable episode-parallel test hash differs"
k2_config="${method_root}/configs/epmc_sit_turn_head_k2_v1.json"
[[ "$(sha256sum "${k2_config}" | awk '{print $1}')" == "${expected_config_sha256}" ]] || fail "exact K2 config hash differs"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "archive lacks checkpoint content manifest"
find "${method_root}" -type f -exec chmod a-w -- {} +
chmod a-w -- "${archive_from_revision}"
[[ -z "$(find "${method_root}" -type d -name __pycache__ -print -quit)" ]] || fail "extracted method tree contains pycache"
validate_checkpoint_content "${checkpoint_manifest}"
method_tree_digest_pre="$(method_tree_digest "${method_root}")"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-epmc-train] mode=${mode} explicit_mode_label=true engineering_smoke=$([[ "${mode}" == smoke ]] && echo true || echo false) posthoc_heldout_eval=false"
echo "[bernini-epmc-train] revision=${source_revision} archive_sha256=${source_archive_sha256} method_tree_sha256=${method_tree_digest_pre}"
echo "[bernini-epmc-train] trainer_sha256=${expected_trainer_sha256} trainer_contract_test_sha256=${expected_trainer_contract_test_sha256} episode_parallel_sha256=${expected_episode_parallel_sha256} episode_parallel_test_sha256=${expected_episode_parallel_test_sha256}"
echo "[bernini-epmc-train] frames=81 K=2 steps_per_support=${steps_per_support} smoke_steps_per_support=1 smoke_total_optimizer_steps=2 formal_steps_per_support=${steps_per_support} GPUs=8xMI210 world_size=${expected_world_size} data_parallel_size=${expected_data_parallel_size} Ulysses=${expected_ulysses_size} support_parallel=true"
echo "[bernini-epmc-train] config_sha256=${expected_config_sha256} preview_sha256=${expected_preview_manifest_sha256} vae_index_sha256=${expected_vae_index_sha256}"
echo "[bernini-epmc-train] preview_only=true experimental_ack=true production_claim=false scientific_claim=false"
echo "[bernini-epmc-train] output=${output_dir} output_reuse=false expected_output_count=${expected_output_count}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_train_fewshot_motion_code_contract.py \
  test_fewshot_episode_parallel.py \
  test_fewshot_proposal_motion_carrier.py \
  test_ffprobe_pyav_compat.py \
  test_auh_train_fewshot_motion_code_canary_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

validate_ffprobe_shim pre_torchrun
episode_source_path="$("${python_bin}" -B - \
  "${k2_config}" \
  "${staged_preview_manifest}" \
  "${episode_probe_iid}" \
  "${episode_probe_source_sha256}" <<'PY'
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
preview_path = Path(sys.argv[2])
iid = sys.argv[3]
expected_sha256 = sys.argv[4]

config = json.loads(config_path.read_text(encoding="utf-8"))
config_rows = [row for row in config.get("rows", []) if row.get("iid") == iid]
if len(config_rows) != 1:
    raise SystemExit("probe IID is not unique in the K2 config")
if config_rows[0].get("source_video_sha256") != expected_sha256:
    raise SystemExit("probe source SHA256 differs in the K2 config")

preview_rows = []
with preview_path.open("r", encoding="utf-8") as handle:
    for line_number, raw_line in enumerate(handle, start=1):
        if not raw_line.strip():
            raise SystemExit(f"blank preview manifest line: {line_number}")
        row = json.loads(raw_line)
        if row.get("iid") == iid:
            preview_rows.append(row)
if len(preview_rows) != 1:
    raise SystemExit("probe IID is not unique in the preview manifest")
row = preview_rows[0]
if row.get("source_video_sha256") != expected_sha256:
    raise SystemExit("probe source SHA256 differs in the preview manifest")
source = row.get("source_video_path")
if not isinstance(source, str) or not source or not Path(source).is_absolute():
    raise SystemExit("probe source path must be a non-empty absolute path")
print(source)
PY
)" || fail "could not resolve pinned episode source for real media probe"
[[ -f "${episode_source_path}" && ! -L "${episode_source_path}" ]] || fail "pinned episode source is not a plain file"
[[ "$(sha256sum "${episode_source_path}" | awk '{print $1}')" == "${episode_probe_source_sha256}" ]] || fail "pinned episode source hash differs before real media probe"
PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${episode_source_path}" "${num_frames}" <<'PY'
from fractions import Fraction
from pathlib import Path
import sys

from fewshot_episode_io import _probe_video_metadata

source = Path(sys.argv[1])
expected_frames = int(sys.argv[2])
metadata = _probe_video_metadata(source)
if metadata.frame_count != expected_frames:
    raise SystemExit(
        f"real episode source frame count differs: {metadata.frame_count} != {expected_frames}"
    )
if metadata.fps != Fraction(25, 1):
    raise SystemExit(f"real episode source FPS differs: {metadata.fps} != 25")
print(
    "[bernini-epmc-train] "
    f"episode_source_probe=true frame_count={metadata.frame_count} "
    f"fps={metadata.fps.numerator}/{metadata.fps.denominator} source={source}"
)
PY
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${method_root}/train_fewshot_motion_code.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --k2-config "${k2_config}" \
  --expected-k2-config-sha256 "${expected_config_sha256}" \
  --preview-manifest "${staged_preview_manifest}" \
  --vae-index "${staged_vae_index}" \
  --output "${output_dir}" \
  --num-frames "${num_frames}" \
  --k-shot "${k_shot}" \
  --steps-per-support "${steps_per_support}" \
  --learning-rate "${learning_rate}" \
  --max-grad-norm "${max_grad_norm}" \
  --seed "${training_seed}" \
  --proposal-seed "${proposal_seed}" \
  --fixed-sigma-index "${fixed_sigma_index}" \
  --held-sigma-index "${held_sigma_index}" \
  --full-target-fm-weight "${full_target_fm_weight}" \
  --ack-preview-experimental-only \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  "${mode_args[@]}"
validate_ffprobe_shim post_torchrun

[[ -d "${output_dir}" && ! -L "${output_dir}" ]] || fail "training output directory missing"
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq "${expected_output_count}" ]] || fail "mode-specific output file count differs"
[[ -z "$(find "${output_dir}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]] || fail "training output contains a non-file entry"
if [[ "${mode}" == smoke ]]; then
  for name in diagnostics.pt smoke.receipt.json; do
    [[ -f "${output_dir}/${name}" && ! -L "${output_dir}/${name}" ]] || fail "smoke output lacks plain ${name}"
  done
else
  for name in diagnostics.pt training_go_receipt.json prototype.safetensors prototype.receipt.json receipt.json; do
    [[ -f "${output_dir}/${name}" && ! -L "${output_dir}/${name}" ]] || fail "formal output lacks plain ${name}"
  done
fi

audit_receipt() {
  PYTHONPATH="${method_root}" "${python_bin}" -B - \
    --mode "${mode}" \
    --receipt "${output_dir}/${receipt_name}" \
    --output "${output_dir}" \
    --method-root "${method_root}" \
    --config "${k2_config}" \
    --config-sha256 "${expected_config_sha256}" \
    --preview "${staged_preview_manifest}" \
    --preview-sha256 "${expected_preview_manifest_sha256}" \
    --vae-index "${staged_vae_index}" \
    --vae-index-sha256 "${expected_vae_index_sha256}" \
    --revision "${source_revision}" \
    --archive-sha256 "${source_archive_sha256}" \
    --checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --world-size "${expected_world_size}" \
    --ulysses-size "${expected_ulysses_size}" \
    --data-parallel-size "${expected_data_parallel_size}" \
    --steps-per-support "${steps_per_support}" \
    --within-support-gradient-sync "${expected_within_support_gradient_sync}" <<'PY'
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
parser.add_argument("--receipt", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--method-root", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--config-sha256", required=True)
parser.add_argument("--preview", required=True)
parser.add_argument("--preview-sha256", required=True)
parser.add_argument("--vae-index", required=True)
parser.add_argument("--vae-index-sha256", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--archive-sha256", required=True)
parser.add_argument("--checkpoint-tree-sha256", required=True)
parser.add_argument("--world-size", type=int, required=True)
parser.add_argument("--ulysses-size", type=int, required=True)
parser.add_argument("--data-parallel-size", type=int, required=True)
parser.add_argument("--steps-per-support", type=int, required=True)
parser.add_argument("--within-support-gradient-sync", required=True)
args = parser.parse_args()

def canonical_sha(value, *, ensure_ascii=False):
    payload = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def plain_file(path_value):
    path = Path(path_value)
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise RuntimeError(f"not a plain artifact: {path}")
    return path

def file_sha(path_value):
    path = plain_file(path_value)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def validate_bound(value, expected_sha=None):
    if not isinstance(value, dict):
        raise RuntimeError("bound artifact is not an object")
    path = plain_file(value.get("path"))
    digest = file_sha(path)
    if digest != value.get("sha256") or (
        expected_sha is not None and digest != expected_sha
    ):
        raise RuntimeError(f"bound artifact hash differs: {path}")
    info = path.stat()
    if (
        info.st_size != value.get("size_bytes")
        or info.st_dev != value.get("device")
        or info.st_ino != value.get("inode")
        or info.st_mtime_ns != value.get("mtime_ns")
    ):
        raise RuntimeError(f"bound artifact identity differs: {path}")

output = Path(args.output)
receipt = json.loads(plain_file(args.receipt).read_text(encoding="utf-8"))
expected_files = (
    {"diagnostics.pt", "smoke.receipt.json"}
    if args.mode == "smoke"
    else {
        "diagnostics.pt",
        "training_go_receipt.json",
        "prototype.safetensors",
        "prototype.receipt.json",
        "receipt.json",
    }
)
if {path.name for path in output.iterdir()} != expected_files:
    raise SystemExit("mode-specific output file set differs")

episode = receipt.get("episode_audit", {})
rows = episode.get("rows", [])
expected_rows = (
    (
        "support",
        "841b5e0080a1441d",
        "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
        "c26970c7d263587db5e5f96b98b30072d7f4b30bc242f9ef581e5780d7c9f070",
        "51b30d41032fa443e667261af2ffeb8e9e1875338c41655c8f5e8e9ecd37fdc1",
    ),
    (
        "support",
        "7262dd490cbf42c5",
        "f2807db8eda17c9657fe33552d1c58b208af08f88394396a5b4acedbec4f3548",
        "1bc20ee874314a60914a71554f109d38dfc4b02c706f9eabb0dc46e92b2879bc",
        "23924dea163e45ed1aaae7ebce36eb689304bf630200c9662766437a2141c69a",
    ),
    (
        "heldout",
        "7b88a1ca1f804f41",
        "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
        "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015",
        "a84fc017ce59de301a9b31208a358a1559f24c50c4f14c2e8d7990a8f2ad9e44",
    ),
)
episode_checks = (
    episode.get("schema_version") == "bernini-epmc-exact-micro-program-k2-audit-v1",
    episode.get("complete") is True,
    episode.get("experimental_training_acknowledged") is True,
    episode.get("preview_only") is True,
    episode.get("training_authorized") is False,
    episode.get("training_use_forbidden") is True,
    episode.get("production_claim_forbidden") is True,
    episode.get("scientific_claim_authorized") is False,
    episode.get("role_order") == ["support", "support", "heldout"],
    episode.get("exact_rgb_frames") == 81,
    episode.get("exact_fps") == 25,
    episode.get("latent_phases") == 21,
    episode.get("iid_disjoint") is True,
    episode.get("source_video_sha256_disjoint") is True,
    episode.get("group_id_disjoint") is True,
    episode.get("selected_artifact_hashes_verified") is True,
    episode.get("selected_media_probed") is True,
    len(rows) == 3,
)
if not all(episode_checks):
    raise SystemExit("episode audit contract differs")
episode_candidate = dict(episode)
episode_digest = episode_candidate.pop("audit_digest", None)
if episode_digest != canonical_sha(episode_candidate, ensure_ascii=False):
    raise SystemExit("episode audit digest differs")
validate_bound(episode.get("config"), args.config_sha256)
validate_bound(episode.get("preview_manifest"), args.preview_sha256)
validate_bound(episode.get("vae_index"), args.vae_index_sha256)
if episode["config"]["path"] != args.config:
    raise SystemExit("episode config path differs")
if episode["preview_manifest"]["path"] != args.preview:
    raise SystemExit("episode preview path differs")
if episode["vae_index"]["path"] != args.vae_index:
    raise SystemExit("episode VAE index path differs")

for row, expected in zip(rows, expected_rows):
    role, iid, source_sha, target_sha, parquet_sha = expected
    if not (
        row.get("role") == role
        and row.get("iid") == iid
        and row.get("source_video_sha256") == source_sha
        and row.get("target_video_sha256") == target_sha
        and row.get("source_video_frames") == 81
        and row.get("target_video_frames") == 81
        and row.get("source_video_fps") == "25"
        and row.get("target_video_fps") == "25"
        and row.get("bucket_hw") == [480, 496]
        and row.get("posterior_parameters_shape") == [1, 32, 21, 60, 62]
    ):
        raise SystemExit(f"selected row contract differs: {iid}")
    validate_bound(row.get("source_video"), source_sha)
    validate_bound(row.get("target_video"), target_sha)
    validate_bound(row.get("vae_parquet"), parquet_sha)
    validate_bound(row.get("vae_sample_receipt"))

receipt_candidate = dict(receipt)
receipt_digest = receipt_candidate.pop("receipt_sha256", None)
if receipt_digest != canonical_sha(receipt_candidate, ensure_ascii=False):
    raise SystemExit("run receipt digest differs")
common_checks = (
    receipt.get("schema_version") == "bernini-epmc-k2-code-inversion-receipt-v2",
    receipt.get("method") == "k2-privileged-epmc-code-inversion",
    receipt.get("preview_only") in (None, True),
)
if not all(common_checks):
    raise SystemExit("run receipt common contract differs")

expected_support_iids = ["841b5e0080a1441d", "7262dd490cbf42c5"]
expected_support_assignments = [
    {
        "support_index": 1,
        "iid": expected_support_iids[0],
        "dp_rank": 0,
        "sp_ranks": [0, 1, 2, 3],
    },
    {
        "support_index": 2,
        "iid": expected_support_iids[1],
        "dp_rank": 1,
        "sp_ranks": [4, 5, 6, 7],
    },
]
expected_distributed = {
    "world_size": args.world_size,
    "ulysses_size": args.ulysses_size,
    "data_parallel_size": args.data_parallel_size,
    "support_parallel": True,
    "sp_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "dp_groups": [[0, 4], [1, 5], [2, 6], [3, 7]],
    "support_assignments": expected_support_assignments,
    "within_support_gradient_sync": args.within_support_gradient_sync,
    "cross_support_gradient_sync": False,
    "k2_exchange": "data_parallel_lane_all_gather_then_world_digest_consensus",
    "backend": "nccl/rccl",
}
if not (
    args.world_size == 8
    and args.ulysses_size == 4
    and args.data_parallel_size == 2
    and args.within_support_gradient_sync
    == "ulysses_group_all_reduce_sum_then_divide_by_4"
    and receipt.get("distributed") == expected_distributed
):
    raise SystemExit("exact WORLD8 DP2 Ulysses4 receipt differs")

sha256_pattern = re.compile(r"[0-9a-f]{64}")
support = receipt.get("support")
if not isinstance(support, list) or len(support) != 2:
    raise SystemExit("run receipt must contain exactly two support results")
expected_support_steps = 1 if args.mode == "smoke" else args.steps_per_support
for observed, expected in zip(support, expected_support_assignments):
    if not isinstance(observed, dict) or any(
        observed.get(key) != expected[key]
        for key in ("support_index", "iid", "dp_rank", "sp_ranks")
    ) or not (
        observed.get("steps") == expected_support_steps
        and observed.get("gradient_sync") == args.within_support_gradient_sync
        and isinstance(observed.get("tied_36d_code_sha256"), str)
        and sha256_pattern.fullmatch(observed["tied_36d_code_sha256"]) is not None
    ):
        raise SystemExit("support result assignment differs")
support_code_sha256 = [item["tied_36d_code_sha256"] for item in support]

def validate_consensus(value, expected_keys, *, label):
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise SystemExit(f"{label} consensus key set differs")
    if any(
        not isinstance(item, str) or sha256_pattern.fullmatch(item) is None
        for item in value.values()
    ):
        raise SystemExit(f"{label} consensus digest differs")

smoke_consensus_keys = {"code", "receipt", "diagnostic", "probe"}
formal_consensus_keys = smoke_consensus_keys | {
    "held_control", "prototype", "decision"
}
validate_consensus(
    receipt.get("k2_exchange_consensus"),
    smoke_consensus_keys if args.mode == "smoke" else formal_consensus_keys,
    label="run",
)

diagnostics_path = output / "diagnostics.pt"
if file_sha(diagnostics_path) != receipt.get("diagnostics_sha256", receipt.get("diagnostics_file_sha256")):
    raise SystemExit("diagnostics hash differs")

if args.mode == "smoke":
    checks = (
        receipt.get("engineering_smoke") is True,
        receipt.get("support_count") == 2,
        receipt.get("optimizer_steps") == 2,
        receipt.get("optimizer_steps_per_support") == 1,
        receipt.get("representability_gate") == "NOT_EVALUATED_ENGINEERING_SMOKE",
        receipt.get("prototype_published") is False,
        receipt.get("single_step_apg_surrogate") is True,
        receipt.get("full_trajectory_render") is False,
        receipt.get("full_target_flow_matching_weight") == 0.0,
        len(receipt.get("support", [])) == 2,
    )
    if not all(checks):
        raise SystemExit("smoke receipt contract differs")
else:
    checks = (
        receipt.get("engineering_smoke") is False,
        receipt.get("representability_gate") in ("GO", "NO_GO"),
        receipt.get("prototype_frozen_before_heldout_deserialization") is True,
        receipt.get("heldout_pre_run_audit") == "content-opaque hash and media-metadata audit only",
        receipt.get("heldout_target_latent_deserialized") is False,
        receipt.get("posthoc_heldout") is None,
        len(receipt.get("support", [])) == 2,
        receipt.get("model", {}).get("checkpoint_tree_sha256") == args.checkpoint_tree_sha256,
        receipt.get("model", {}).get("base_frozen") is True,
        receipt.get("model", {}).get("cpmr_clone_frozen") is True,
        receipt.get("model", {}).get("gradient_checkpointing") is False,
        receipt.get("model", {}).get("trainable_dimension_per_support") == 36,
        receipt.get("objective", {}).get("single_step_apg_surrogate") is True,
        receipt.get("objective", {}).get("full_trajectory_render") is False,
        receipt.get("objective", {}).get("full_target_flow_matching_weight") == 0.0,
        receipt.get("inference_conditions") == ["source_video", "edit_instruction"],
        receipt.get("preview_only") is True,
        receipt.get("training_authorized") is False,
        receipt.get("training_use_forbidden") is True,
        receipt.get("production_claim_forbidden") is True,
        receipt.get("scientific_claim_authorized") is False,
        receipt.get("method_source_revision") == args.revision,
        receipt.get("method_source_archive_sha256") == args.archive_sha256,
    )
    if not all(checks):
        raise SystemExit("formal run receipt contract differs")

    go_path = output / "training_go_receipt.json"
    go_receipt = json.loads(plain_file(go_path).read_text(encoding="utf-8"))
    go_candidate = dict(go_receipt)
    go_digest = go_candidate.pop("receipt_sha256", None)
    if go_digest != canonical_sha(go_candidate, ensure_ascii=False):
        raise SystemExit("training gate receipt digest differs")
    if not (
        go_receipt.get("schema_version") == "bernini-epmc-k2-representability-gate-v1"
        and go_receipt.get("representability_gate") == receipt.get("representability_gate")
        and go_receipt.get("support_iids") == expected_support_iids
        and go_receipt.get("support_tied_36d_code_sha256") == support_code_sha256
        and all(
            isinstance(value, str) and sha256_pattern.fullmatch(value) is not None
            for value in go_receipt["support_tied_36d_code_sha256"]
        )
        and go_receipt.get("single_step_apg_surrogate") is True
        and go_receipt.get("full_trajectory_render") is False
        and go_receipt.get("video_quality_claim") is False
        and go_receipt.get("full_target_flow_matching_weight") == 0.0
        and go_receipt.get("reference_gradient_probe_support_index") == 1
        and go_receipt.get("distributed") == expected_distributed
    ):
        raise SystemExit("training gate receipt contract differs")
    probes = go_receipt.get("gradient_probes")
    if not isinstance(probes, list) or len(probes) != 2 or any(
        not isinstance(item, dict) for item in probes
    ) or [
        (
            item.get("iid"),
            item.get("support_index"),
            item.get("dp_rank"),
            item.get("sp_ranks"),
            item.get("family"),
            item.get("passed"),
            item.get("gradient_sync"),
        )
        for item in probes
    ] != [
        (
            expected_support_iids[0], 1, 0, [0, 1, 2, 3], "phase_only", True,
            args.within_support_gradient_sync,
        ),
        (
            expected_support_iids[0], 1, 0, [0, 1, 2, 3], "block_only", True,
            args.within_support_gradient_sync,
        ),
    ]:
        raise SystemExit("support-1-only reference gradient probes differ")
    held_controls = go_receipt.get("held_noise_controls")
    if not isinstance(held_controls, list) or len(held_controls) != 2 or any(
        not isinstance(item, dict) for item in held_controls
    ) or [item.get("iid") for item in held_controls] != expected_support_iids:
        raise SystemExit("exact two-support held-noise controls differ")
    for item in held_controls:
        if not (
            item.get("held_noise") is True
            and item.get("zero_code_raw_noop_byte_exact") is True
            and item.get("full_target_flow_matching_weight") == 0.0
            and item.get("single_step_apg_surrogate") is True
            and isinstance(
                item.get("correct_strictly_better_than_zero_reverse_shuffle"), bool
            )
            and isinstance(item.get("losses"), dict)
            and set(item["losses"]) == {"zero", "correct", "reverse", "shuffle"}
        ):
            raise SystemExit("held-noise control contract differs")
    teacher_go = go_receipt.get("teacher_go", {})
    if not (
        teacher_go.get("loss_aggregation")
        == "sort_by_iid_then_arithmetic_mean_across_two_support_held_noise_cells_before_ratios"
        and teacher_go.get("loss_aggregation_iids")
        == sorted(expected_support_iids)
    ):
        raise SystemExit("held-noise loss aggregation contract differs")
    validate_consensus(
        go_receipt.get("k2_exchange_consensus"),
        formal_consensus_keys,
        label="training gate",
    )
    if go_receipt.get("k2_exchange_consensus") != receipt.get("k2_exchange_consensus"):
        raise SystemExit("training gate/run consensus evidence differs")

    state_path = output / "prototype.safetensors"
    companion_path = output / "prototype.receipt.json"
    state_sha = file_sha(state_path)
    companion_sha = file_sha(companion_path)
    if not (
        receipt.get("prototype_companion_file_sha256") == companion_sha
        and receipt.get("training_go_receipt_file_sha256") == file_sha(go_path)
        and receipt.get("diagnostics_file_sha256") == file_sha(diagnostics_path)
    ):
        raise SystemExit("formal artifact hash binding differs")
    sys.path.insert(0, args.method_root)
    import infer_fewshot_motion_code as inference

    bundle = inference.load_prototype_bundle(
        state_path,
        companion_path,
        expected_state_sha256=state_sha,
        expected_receipt_sha256=companion_sha,
        allow_no_go_diagnostic=receipt.get("representability_gate") == "NO_GO",
    )
    if bundle.representability_gate != receipt.get("representability_gate"):
        raise SystemExit("prototype/run representability gate differs")
PY
}

receipt_path="${output_dir}/${receipt_name}"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "mode receipt missing"
receipt_sha_round1="$(sha256sum "${receipt_path}" | awk '{print $1}')"
output_tree_round1="$(output_tree_digest "${output_dir}")"
audit_receipt || fail "result receipt audit round 1 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed after audit round 1"
[[ "$(output_tree_digest "${output_dir}")" == "${output_tree_round1}" ]] || fail "output tree changed after audit round 1"
receipt_sha_round2="$(sha256sum "${receipt_path}" | awk '{print $1}')"
[[ "${receipt_sha_round2}" == "${receipt_sha_round1}" ]] || fail "receipt hash closure differs"
audit_receipt || fail "result receipt audit round 2 failed"
[[ "$(sha256sum "${receipt_path}" | awk '{print $1}')" == "${receipt_sha_round2}" ]] || fail "receipt changed after audit round 2"
[[ "$(output_tree_digest "${output_dir}")" == "${output_tree_round1}" ]] || fail "output tree changed after audit round 2"

method_tree_digest_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_digest_post}" == "${method_tree_digest_pre}" ]] || fail "extracted method tree changed during training"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable method archive changed"
[[ "$(sha256sum "${preview_manifest}" | awk '{print $1}')" == "${expected_preview_manifest_sha256}" ]] || fail "preview manifest changed"
[[ "$(sha256sum "${staged_preview_manifest}" | awk '{print $1}')" == "${expected_preview_manifest_sha256}" ]] || fail "staged preview manifest changed"
[[ "$(sha256sum "${vae_index}" | awk '{print $1}')" == "${expected_vae_index_sha256}" ]] || fail "VAE index changed"
[[ "$(sha256sum "${staged_vae_index}" | awk '{print $1}')" == "${expected_vae_index_sha256}" ]] || fail "staged VAE index changed"
[[ "$(sha256sum "${trainer_path}" | awk '{print $1}')" == "${expected_trainer_sha256}" ]] || fail "final-stable trainer changed"
[[ "$(sha256sum "${trainer_contract_test_path}" | awk '{print $1}')" == "${expected_trainer_contract_test_sha256}" ]] || fail "final-stable trainer contract test changed"
[[ "$(sha256sum "${episode_parallel_path}" | awk '{print $1}')" == "${expected_episode_parallel_sha256}" ]] || fail "final-stable episode-parallel module changed"
[[ "$(sha256sum "${episode_parallel_test_path}" | awk '{print $1}')" == "${expected_episode_parallel_test_sha256}" ]] || fail "final-stable episode-parallel test changed"
validate_checkpoint_content "${checkpoint_manifest}"
validate_ffprobe_shim final
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq "${expected_output_count}" ]] || fail "output count changed after audits"

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-epmc-train] COMPLETE mode=${mode} receipt=${receipt_path} receipt_sha256=${receipt_sha_round2} output_tree_sha256=${output_tree_round1}"
