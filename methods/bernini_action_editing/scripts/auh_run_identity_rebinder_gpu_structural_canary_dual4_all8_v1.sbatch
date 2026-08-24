#!/usr/bin/env bash
#SBATCH --job-name=graft-ir-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${GRAFT_REBINDER_SOURCE_ARCHIVE:?set GRAFT_REBINDER_SOURCE_ARCHIVE}"
closure_manifest="${GRAFT_REBINDER_RUNTIME_CLOSURE_MANIFEST:?set GRAFT_REBINDER_RUNTIME_CLOSURE_MANIFEST}"
python_bin="${GRAFT_REBINDER_PYTHON_BIN:?set GRAFT_REBINDER_PYTHON_BIN}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
cell_spec="${GRAFT_REBINDER_CELL_SPEC:?set GRAFT_REBINDER_CELL_SPEC}"
output_root="${GRAFT_REBINDER_OUTPUT_ROOT:?set GRAFT_REBINDER_OUTPUT_ROOT}"
launcher_source="${GRAFT_REBINDER_LAUNCHER_SOURCE:?set GRAFT_REBINDER_LAUNCHER_SOURCE}"

runner_sha256="${GRAFT_REBINDER_RUNNER_SHA256:?set GRAFT_REBINDER_RUNNER_SHA256}"
rebinder_sha256="${GRAFT_REBINDER_CORE_SHA256:?set GRAFT_REBINDER_CORE_SHA256}"
native_probe_sha256="${GRAFT_NATIVE_FIELD_PROBE_SHA256:?set GRAFT_NATIVE_FIELD_PROBE_SHA256}"
cell_spec_sha256="${GRAFT_REBINDER_CELL_SPEC_SHA256:?set GRAFT_REBINDER_CELL_SPEC_SHA256}"
launcher_sha256="${GRAFT_REBINDER_LAUNCHER_SHA256:?set GRAFT_REBINDER_LAUNCHER_SHA256}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
source_archive_sha256="${GRAFT_REBINDER_SOURCE_ARCHIVE_SHA256:?set GRAFT_REBINDER_SOURCE_ARCHIVE_SHA256}"
closure_manifest_sha256="${GRAFT_REBINDER_RUNTIME_CLOSURE_MANIFEST_SHA256:?set GRAFT_REBINDER_RUNTIME_CLOSURE_MANIFEST_SHA256}"
learning_rate="${GRAFT_REBINDER_LEARNING_RATE:-1e-3}"

readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831

fail() { echo "[graft-ir-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

for name in source_archive closure_manifest python_bin bernini_root veomni_root checkpoint checkpoint_manifest cell_spec output_root launcher_source; do
  require_absolute "${!name}" "${name}"
done
for name in runner_sha256 rebinder_sha256 native_probe_sha256 cell_spec_sha256 launcher_sha256 checkpoint_manifest_sha256 source_archive_sha256 closure_manifest_sha256; do
  require_sha256 "${!name}" "${name}"
done
[[ "${checkpoint_manifest_sha256}" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest authority differs"
[[ "${learning_rate}" =~ ^(0\.[0-9]+|[0-9]+(\.[0-9]+)?[eE]-[0-9]+)$ ]] || fail "learning rate syntax differs"

source_archive="$(realpath -e -- "${source_archive}")"
closure_manifest="$(realpath -e -- "${closure_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
cell_spec="$(realpath -e -- "${cell_spec}")"
launcher_source="$(realpath -e -- "${launcher_source}")"
output_root="$(realpath -m -- "${output_root}")"

[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain source/checkpoint directory required: ${path}"
done
for path in "${source_archive}" "${closure_manifest}" "${checkpoint_manifest}" "${cell_spec}" "${launcher_source}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain input file required: ${path}"
done

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ "$(sha256sum "${closure_manifest}" | awk '{print $1}')" == "${closure_manifest_sha256}" ]] || fail "runtime closure manifest bytes differ"
[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "cell registry bytes differ"
[[ "$(sha256sum "${launcher_source}" | awk '{print $1}')" == "${launcher_sha256}" ]] || fail "launcher bytes differ"
executed_launcher="$(realpath -e -- "$0")"
[[ -f "${executed_launcher}" && ! -L "${executed_launcher}" ]] || fail "executed launcher differs"
[[ "$(sha256sum "${executed_launcher}" | awk '{print $1}')" == "${launcher_sha256}" ]] || fail "executed launcher bytes differ from sealed source"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest bytes differ"
[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]] || fail "output root must be fresh and non-root"
[[ -d "$(dirname -- "${output_root}")" && ! -L "$(dirname -- "${output_root}")" ]] || fail "output parent differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/graft-ir-v1-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(stat -Lc '%d:%i' -- "${task_scratch}")"
[[ "${task_scratch_identity}" =~ ^[0-9]+:[0-9]+$ ]] || fail "scratch dev:ino capture failed"
dog_pid=""; human_pid=""

cleanup() {
  local status=$? cleanup_failed=0 observed=""
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/graft-ir-v1-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    observed="$(stat -Lc '%d:%i' -- "${task_scratch}" 2>/dev/null || true)"
    [[ "${observed}" == "${task_scratch_identity}" ]] || cleanup_failed=1
  else
    cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" == 0 ]]; then
    observed="$(stat -Lc '%d:%i' -- "${task_scratch}" 2>/dev/null || true)"
    [[ "${observed}" == "${task_scratch_identity}" ]] || cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" == 0 ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" == 0 ]]; then
    observed="$(stat -Lc '%d:%i' -- "${task_scratch}" 2>/dev/null || true)"
    [[ "${observed}" == "${task_scratch_identity}" ]] || cleanup_failed=1
  fi
  if [[ "${cleanup_failed}" == 0 ]]; then
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[graft-ir-v1] PASS topology=dog:GPU0-3/SP4,human:GPU4-7/SP4 structural_only=true output=${output_root}"
  else
    echo "[graft-ir-v1] EXIT status=${status} cleanup_devino_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true
  done
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

cp -- "${source_archive}" "${task_scratch}/runtime-source.tar"
cp -- "${closure_manifest}" "${task_scratch}/runtime-closure.json"
[[ "$(sha256sum "${task_scratch}/runtime-source.tar" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "scratch source archive copy differs"
[[ "$(sha256sum "${task_scratch}/runtime-closure.json" | awk '{print $1}')" == "${closure_manifest_sha256}" ]] || fail "scratch closure manifest copy differs"
chmod a-w -- "${task_scratch}/runtime-source.tar" "${task_scratch}/runtime-closure.json"

"${python_bin}" -B - \
  "${task_scratch}/runtime-source.tar" \
  "${task_scratch}/runtime-closure.json" \
  "${runner_sha256}" "${rebinder_sha256}" "${native_probe_sha256}" <<'PY'
import hashlib
import json
from pathlib import PurePosixPath
import re
import sys
import tarfile

archive, manifest_path = sys.argv[1], sys.argv[2]
runner_sha, rebinder_sha, probe_sha = sys.argv[3:6]
sha = re.compile(r"[0-9a-f]{64}")
with open(manifest_path, "r", encoding="ascii") as handle:
    manifest = json.load(handle)
if (
    not isinstance(manifest, dict)
    or set(manifest) != {"schema_version", "root", "selection", "files"}
    or manifest.get("schema_version") != "bernini-graft-runtime-python-closure-v2"
    or manifest.get("root") != "methods/bernini_action_editing"
    or manifest.get("selection")
    != "explicit_runtime_dependency_closure_no_symlink_no_pycache_v1"
    or not isinstance(manifest.get("files"), dict)
    or not manifest["files"]
):
    raise SystemExit("runtime closure manifest schema differs")
declared = {}
for relative, digest in manifest["files"].items():
    path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath("/")
    if (
        not isinstance(relative, str)
        or not relative
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".py"
        or "__pycache__" in path.parts
        or path.as_posix() != relative
        or not isinstance(digest, str)
        or sha.fullmatch(digest) is None
    ):
        raise SystemExit(f"unsafe closure member: {relative!r}")
    declared[relative] = digest
required = {
    "run_identity_rebinder_gpu_structural_canary_v1.py": runner_sha,
    "identity_rebinder_v1.py": rebinder_sha,
    "graft_native_v2v_field_probe_v1.py": probe_sha,
}
if any(declared.get(name) != digest for name, digest in required.items()):
    raise SystemExit("direct runtime source hashes differ from closure manifest")
lazy_required = {
    "tools/materialize_vae.py",
    "tools/build_renderer_dataset.py",
}
if not lazy_required.issubset(declared):
    raise SystemExit(
        f"runtime closure lacks infer_lora lazy imports: "
        f"{sorted(lazy_required-set(declared))}"
    )
expected = {
    f"methods/bernini_action_editing/{relative}" for relative in declared
}
observed = set()
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
        ):
            raise SystemExit("source archive contains an unsafe member")
        if member.isfile():
            observed.add(normalized)
        elif not member.isdir():
            raise SystemExit("source archive contains an unsupported member type")
if observed != expected:
    raise SystemExit(
        f"source archive/closure set differs: missing={sorted(expected-observed)[:8]}, "
        f"extra={sorted(observed-expected)[:8]}"
    )
PY

mkdir -- "${task_scratch}/source-tree"
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/runtime-source.tar" \
  -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_identity_rebinder_gpu_structural_canary_v1.py"
rebinder="${method_root}/identity_rebinder_v1.py"
native_probe="${method_root}/graft_native_v2v_field_probe_v1.py"

"${python_bin}" -B - "${method_root}" "${task_scratch}/runtime-closure.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

root, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
declared = manifest["files"]
observed = {}
for path in root.rglob("*.py"):
    relative = path.relative_to(root).as_posix()
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"extracted Python source is not plain: {relative}")
    if "__pycache__" not in path.relative_to(root).parts:
        observed[relative] = path
if set(observed) != set(declared):
    raise SystemExit("extracted runtime source set differs")
for relative in sorted(declared):
    digest = hashlib.sha256()
    descriptor = os.open(
        observed[relative], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    if digest.hexdigest() != declared[relative]:
        raise SystemExit(f"extracted source hash differs: {relative}")
PY
chmod -R a-w -- "${method_root}"
[[ -f "${runner}" && ! -L "${runner}" ]] || fail "sealed runner absent"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == "${runner_sha256}" ]] || fail "sealed runner bytes differ"
[[ "$(sha256sum "${rebinder}" | awk '{print $1}')" == "${rebinder_sha256}" ]] || fail "sealed IdentityRebinder bytes differ"
[[ "$(sha256sum "${native_probe}" | awk '{print $1}')" == "${native_probe_sha256}" ]] || fail "sealed field probe bytes differ"

# infer_lora resolves its video decoder lazily inside _load_source.  Importing
# only the runner cannot expose a missing tools/materialize_vae.py, so prove
# this dynamic edge before either WORLD4 group is launched.
if ! "${python_bin}" -B - "${method_root}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
from tools import build_renderer_dataset, materialize_vae

expected = {
    "tools/build_renderer_dataset.py": build_renderer_dataset,
    "tools/materialize_vae.py": materialize_vae,
}
for relative, module in expected.items():
    observed = Path(module.__file__).resolve(strict=True)
    required = (root / relative).resolve(strict=True)
    if observed != required or observed.is_symlink():
        raise SystemExit(f"sealed lazy import resolved outside archive: {relative}")
for name in ("_decode_exact_video", "source_aspect_bucket", "_resize_video"):
    if not callable(getattr(materialize_vae, name, None)):
        raise SystemExit(f"sealed materialize_vae API absent: {name}")
PY
then
  fail "sealed infer_lora lazy-import preflight failed"
fi

mkdir -p -- "${task_scratch}/dog/cache" "${task_scratch}/dog/tmp" \
  "${task_scratch}/human/cache" "${task_scratch}/human/tmp"
mkdir -- "${output_root}"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((30000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_group() (
  set -Eeuo pipefail
  local cell_id="$1" visible_gpus="$2" master_port="$3"
  local group_scratch="${task_scratch}/${cell_id}"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/cache/triton"
  export XDG_CACHE_HOME="${group_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/cache/pycache"
  export TMPDIR="${group_scratch}/tmp"
  mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"
  "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${runner}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
    --cell-spec "${cell_spec}" \
    --expected-cell-spec-sha256 "${cell_spec_sha256}" \
    --cell-id "${cell_id}" \
    --output-dir "${output_root}/${cell_id}" \
    --expected-runner-sha256 "${runner_sha256}" \
    --expected-rebinder-sha256 "${rebinder_sha256}" \
    --expected-native-field-probe-sha256 "${native_probe_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}" \
    --runtime-closure-manifest "${task_scratch}/runtime-closure.json" \
    --expected-runtime-closure-manifest-sha256 "${closure_manifest_sha256}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --expected-bernini-commit "${required_bernini_commit}" \
    --expected-veomni-commit "${required_veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${required_checkpoint_tree_sha256}" \
    --learning-rate "${learning_rate}" \
    --ack-structural-only-no-semantic-claim
)

echo "[graft-ir-v1] allocation=8xMI210 dog=WORLD4/SP4@0-3 human=WORLD4/SP4@4-7 exact81 V-like shared_step structural surrogate"
run_group dog 0,1,2,3 "${dog_port}" >"${output_root}/dog.launch.log" 2>&1 &
dog_pid=$!
run_group human 4,5,6,7 "${human_port}" >"${output_root}/human.launch.log" 2>&1 &
human_pid=$!

set +e
wait "${dog_pid}"; dog_status=$?
wait "${human_pid}"; human_status=$?
set -e
(( dog_status == 0 )) || fail "dog WORLD4/SP4 failed; inspect ${output_root}/dog.launch.log"
(( human_status == 0 )) || fail "human WORLD4/SP4 failed; inspect ${output_root}/human.launch.log"

for cell_id in dog human; do
  receipt="${output_root}/${cell_id}/receipt.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "${cell_id} receipt absent"
  [[ "$(stat -Lc '%a' -- "${receipt}")" == 444 ]] || fail "${cell_id} receipt mode differs"
  "${python_bin}" -B - "${receipt}" "${cell_id}" <<'PY'
import json
from pathlib import Path
import stat
import sys

path, cell = Path(sys.argv[1]), sys.argv[2]
parent_before = path.parent.lstat()
value = json.loads(path.read_text(encoding="ascii"))
parent_after = path.parent.lstat()
declared_parent = value.get("output_directory_identity", {})
if (
    path.parent.is_symlink()
    or not stat.S_ISDIR(parent_before.st_mode)
    or not stat.S_ISDIR(parent_after.st_mode)
    or (int(parent_before.st_dev), int(parent_before.st_ino))
    != (int(parent_after.st_dev), int(parent_after.st_ino))
    or [int(parent_after.st_dev), int(parent_after.st_ino)]
    != [declared_parent.get("st_dev"), declared_parent.get("st_ino")]
):
    raise SystemExit("receipt parent dev:ino differs")
if value.get("pass") is not True or value.get("complete") is not True:
    raise SystemExit("structural receipt did not pass")
if value.get("cell", {}).get("cell_id") != cell:
    raise SystemExit("receipt cell differs")
claims = value.get("claim_scope", {})
if (
    claims.get("structural_canary") is not True
    or claims.get("official_sampler_parity") is not False
    or claims.get("semantic_action_success") is not False
    or claims.get("native_binder_used") is not False
):
    raise SystemExit("receipt claim boundary differs")
materialization = value.get("identity_materialization", {})
if (
    materialization.get("raw_full_source_atlas_encode_calls_per_rank") != 4
    or materialization.get("full_source_vae_encode_calls_per_rank") != 1
    or materialization.get("atlas_encoder_optimizer_target") is not True
):
    raise SystemExit("atlas/VAE materialization contract differs")
closure = value.get("provenance", {}).get("sealed_runtime_python_closure", {})
if (
    closure.get("source_archive_sha256") is None
    or closure.get("exact_file_set_verified") is not True
    or closure.get("every_file_sha256_verified") is not True
    or not isinstance(closure.get("files"), dict)
    or not closure["files"]
):
    raise SystemExit("sealed runtime closure evidence differs")
PY
done

echo "GRAFT_IDENTITY_REBINDER_VLIKE_STRUCTURAL_DUAL4_ALL8_OK"
