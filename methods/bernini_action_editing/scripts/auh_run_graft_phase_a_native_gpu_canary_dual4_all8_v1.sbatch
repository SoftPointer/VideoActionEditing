#!/bin/bash -p
#SBATCH --job-name=graft-pa-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

case "$-" in
  *p*) ;;
  *) echo "[graft-pa-v1] ERROR: Bash privileged mode is required" >&2; exit 2 ;;
esac

set -Eeuo pipefail
umask 077
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "[graft-pa-v1] ERROR: Bash 5.1+ is required for fail-fast wait -n -p" >&2
  exit 2
fi
# The formal submission wrapper replaces the environment before sbatch.  The
# launcher repeats the fail-close boundary before reading any task input.
for poison_name in ${!LD_*} ${!PYTHON*} ${!SBATCH_*}; do
  unset "${poison_name}"
done
unset BASH_ENV ENV CDPATH GLOBIGNORE
export PATH=/usr/bin:/bin

source_archive="${GRAFT_PHASE_A_SOURCE_ARCHIVE:?set GRAFT_PHASE_A_SOURCE_ARCHIVE}"
closure_manifest="${GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST:?set GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST}"
python_bin="${GRAFT_PHASE_A_PYTHON_BIN:?set GRAFT_PHASE_A_PYTHON_BIN}"
python_sha256="${GRAFT_PHASE_A_PYTHON_SHA256:?set GRAFT_PHASE_A_PYTHON_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
cell_spec="${GRAFT_PHASE_A_CELL_SPEC:?set GRAFT_PHASE_A_CELL_SPEC}"
output_root="${GRAFT_PHASE_A_OUTPUT_ROOT:?set GRAFT_PHASE_A_OUTPUT_ROOT}"
launcher_source="${GRAFT_PHASE_A_LAUNCHER_SOURCE:?set GRAFT_PHASE_A_LAUNCHER_SOURCE}"

runner_sha256="${GRAFT_PHASE_A_RUNNER_SHA256:?set GRAFT_PHASE_A_RUNNER_SHA256}"
closure_core_sha256="${GRAFT_PHASE_A_CLOSURE_CORE_SHA256:?set GRAFT_PHASE_A_CLOSURE_CORE_SHA256}"
rebinder_sha256="${GRAFT_PHASE_A_REBINDER_SHA256:?set GRAFT_PHASE_A_REBINDER_SHA256}"
cell_spec_sha256="${GRAFT_PHASE_A_CELL_SPEC_SHA256:?set GRAFT_PHASE_A_CELL_SPEC_SHA256}"
launcher_sha256="${GRAFT_PHASE_A_LAUNCHER_SHA256:?set GRAFT_PHASE_A_LAUNCHER_SHA256}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
source_archive_sha256="${GRAFT_PHASE_A_SOURCE_ARCHIVE_SHA256:?set GRAFT_PHASE_A_SOURCE_ARCHIVE_SHA256}"
closure_manifest_sha256="${GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST_SHA256:?set GRAFT_PHASE_A_RUNTIME_CLOSURE_MANIFEST_SHA256}"

readonly required_source_commit=00f7aba4bd58e67778542273cfca82fb785b648c
readonly required_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly required_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly required_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly required_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly required_cell_spec_sha256=2c822aaf5a98cad3d582ec4973bd9e1df54615bbc3cb5e55217ef65f8742b382
readonly required_runner_sha256=e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a
readonly required_closure_core_sha256=36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a
readonly required_rebinder_sha256=1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0

fail() { echo "[graft-pa-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }

for name in source_archive closure_manifest python_bin bernini_root veomni_root checkpoint checkpoint_manifest cell_spec output_root launcher_source; do
  require_absolute "${!name}" "${name}"
done
for name in runner_sha256 closure_core_sha256 rebinder_sha256 cell_spec_sha256 launcher_sha256 checkpoint_manifest_sha256 source_archive_sha256 closure_manifest_sha256 python_sha256; do
  require_sha256 "${!name}" "${name}"
done
[[ "${runner_sha256}" == "${required_runner_sha256}" ]] || fail "runner digest differs from commit ${required_source_commit}"
[[ "${closure_core_sha256}" == "${required_closure_core_sha256}" ]] || fail "Phase-A closure digest differs from commit ${required_source_commit}"
[[ "${rebinder_sha256}" == "${required_rebinder_sha256}" ]] || fail "IdentityRebinder digest differs from commit ${required_source_commit}"
[[ "${cell_spec_sha256}" == "${required_cell_spec_sha256}" ]] || fail "cell spec authority differs"
[[ "${checkpoint_manifest_sha256}" == "${required_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest authority differs"

source_archive="$(/usr/bin/realpath -e -- "${source_archive}")"
closure_manifest="$(/usr/bin/realpath -e -- "${closure_manifest}")"
python_bin="$(/usr/bin/realpath -e -- "${python_bin}")"
bernini_root="$(/usr/bin/realpath -e -- "${bernini_root}")"
veomni_root="$(/usr/bin/realpath -e -- "${veomni_root}")"
checkpoint="$(/usr/bin/realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(/usr/bin/realpath -e -- "${checkpoint_manifest}")"
cell_spec="$(/usr/bin/realpath -e -- "${cell_spec}")"
launcher_source="$(/usr/bin/realpath -e -- "${launcher_source}")"
output_root="$(/usr/bin/realpath -m -- "${output_root}")"

[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain source/checkpoint directory required: ${path}"
done
for path in "${source_archive}" "${closure_manifest}" "${checkpoint_manifest}" "${cell_spec}" "${launcher_source}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain input file required: ${path}"
done
[[ "$(/usr/bin/stat -Lc '%a' -- "${source_archive}")" == 444 ]] || fail "source archive must be released 0444"
[[ "$(/usr/bin/stat -Lc '%a' -- "${closure_manifest}")" == 444 ]] || fail "closure manifest must be released 0444"
[[ "$(/usr/bin/stat -Lc '%a' -- "${launcher_source}")" == 555 ]] || fail "launcher source must be released 0555"
[[ ! -e "${output_root}" && ! -L "${output_root}" && "${output_root}" != / ]] || fail "output root must be fresh and non-root"
output_parent="$(/usr/bin/dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

# Enter a newly-created task-owned directory before the first Python process
# starts.  Python isolated mode removes implicit cwd imports, while this cwd
# boundary also protects torchrun's worker interpreters, which do not inherit
# the parent interpreter's -I flag as an interpreter flag.
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(/usr/bin/mktemp -d "${scratch_parent%/}/graft-pa-v1-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${task_scratch}")"
[[ "${task_scratch_identity}" =~ ^[0-9]+:[0-9]+$ ]] || fail "scratch dev:ino capture failed"
builtin cd -- "${task_scratch}"
task_scratch_cwd="$(/bin/pwd -P)"
[[ "${task_scratch_cwd}" == "${task_scratch}" ]] || fail "fresh task scratch cwd differs"
[[ "$(/usr/bin/stat -Lc '%d:%i' -- .)" == "${task_scratch_identity}" ]] || fail "fresh task scratch cwd identity differs"
dog_pid=""; human_pid=""

cleanup() {
  local status=$? observed="" scratch_identity_stable=false scratch_cwd_stable=false
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/graft-pa-v1-${SLURM_JOB_ID}."*) ;;
    *) ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    observed="$(/usr/bin/stat -Lc '%d:%i' -- "${task_scratch}" 2>/dev/null || true)"
    [[ "${observed}" == "${task_scratch_identity}" ]] && scratch_identity_stable=true
  fi
  if [[ "$(/bin/pwd -P 2>/dev/null || true)" == "${task_scratch_cwd}" && "$(/usr/bin/stat -Lc '%d:%i' -- . 2>/dev/null || true)" == "${task_scratch_identity}" ]]; then
    scratch_cwd_stable=true
  fi
  if [[ "${status}" == 0 ]]; then
    echo "[graft-pa-v1] PASS dog=GPU0-3/WORLD4/SP4 human=GPU4-7/WORLD4/SP4 output=${output_root} scratch_preserved=${task_scratch} scratch_identity_stable=${scratch_identity_stable} scratch_cwd_stable=${scratch_cwd_stable}"
  else
    echo "[graft-pa-v1] EXIT status=${status} scratch_preserved=${task_scratch} scratch_identity_stable=${scratch_identity_stable} scratch_cwd_stable=${scratch_cwd_stable} output_retained=true" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1" child=""
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

file_sha256() {
  "${python_bin}" -I -S -B - "$1" <<'PY'
import hashlib, os, stat, sys
fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
digest = hashlib.sha256()
try:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("not plain")
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SystemExit("changed while hashing")
finally:
    os.close(fd)
print(digest.hexdigest())
PY
}

[[ "$(file_sha256 "${source_archive}")" == "${source_archive_sha256}" ]] || fail "source archive bytes differ"
[[ "$(file_sha256 "${closure_manifest}")" == "${closure_manifest_sha256}" ]] || fail "runtime closure manifest bytes differ"
[[ "$(file_sha256 "${cell_spec}")" == "${cell_spec_sha256}" ]] || fail "cell spec bytes differ"
[[ "$(file_sha256 "${checkpoint_manifest}")" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest bytes differ"
[[ "$(file_sha256 "${launcher_source}")" == "${launcher_sha256}" ]] || fail "launcher source bytes differ"
[[ "$(file_sha256 "${python_bin}")" == "${python_sha256}" ]] || fail "configured Python bytes differ"
executed_launcher="$(/usr/bin/realpath -e -- "$0")"
[[ -f "${executed_launcher}" && ! -L "${executed_launcher}" ]] || fail "executed launcher differs"
[[ "$(file_sha256 "${executed_launcher}")" == "${launcher_sha256}" ]] || fail "executed launcher bytes differ from sealed source"

audit_vendor_tree() {
  local root="$1" expected_commit="$2" label="$3" status="" candidate="" relative=""
  local symlink_scan="${task_scratch}/vendor-symlink-scan.$$.nul"
  local import_scan="${task_scratch}/vendor-import-scan.$$.nul"
  [[ "$(/usr/bin/git -C "${root}" rev-parse HEAD)" == "${expected_commit}" ]] || fail "${label} commit differs"
  if ! /usr/bin/find "${root}" -path "${root}/.git" -prune -o -type l -print0 >"${symlink_scan}"; then
    fail "${label} symlink scan failed"
  fi
  while IFS= read -r -d '' candidate; do
    relative="${candidate#"${root}/"}"
    fail "${label} has symlink outside .git: ${relative}"
  done <"${symlink_scan}"
  status="$(/usr/bin/git -C "${root}" status --porcelain=v1 --untracked-files=all)" \
    || fail "${label} git status audit failed"
  [[ -z "${status}" ]] || fail "${label} has tracked changes or untracked shadow files"
  if ! /usr/bin/find "${root}" -path "${root}/.git" -prune -o \( -name '*.py' -o -name '*.pyc' -o -name '*.so' -o -name '*.pth' \) -print0 >"${import_scan}"; then
    fail "${label} import-node scan failed"
  fi
  while IFS= read -r -d '' candidate; do
    relative="${candidate#"${root}/"}"
    [[ -f "${candidate}" && ! -L "${candidate}" ]] \
      || fail "${label} has non-plain import-capable node: ${relative}"
    /usr/bin/git -C "${root}" ls-files --error-unmatch -- "${relative}" >/dev/null 2>&1 \
      || fail "${label} has ignored/untracked import-capable shadow: ${relative}"
  done <"${import_scan}"
}
audit_vendor_tree "${bernini_root}" "${required_bernini_commit}" Bernini
audit_vendor_tree "${veomni_root}" "${required_veomni_commit}" VeOmni

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 KMP_USE_SHM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

/usr/bin/cp -- "${source_archive}" "${task_scratch}/runtime-source.tar"
/usr/bin/cp -- "${closure_manifest}" "${task_scratch}/runtime-closure.json"
[[ "$(file_sha256 "${task_scratch}/runtime-source.tar")" == "${source_archive_sha256}" ]] || fail "scratch archive copy differs"
[[ "$(file_sha256 "${task_scratch}/runtime-closure.json")" == "${closure_manifest_sha256}" ]] || fail "scratch manifest copy differs"
/usr/bin/chmod 0444 -- "${task_scratch}/runtime-source.tar" "${task_scratch}/runtime-closure.json"
/usr/bin/mkdir -- "${task_scratch}/source-tree"

# This is the complete local Python runtime closure reachable from the pinned
# Phase-A runner.  The map is fixed to git object bytes at 00f7aba, including
# infer_lora's dynamic materialize_vae -> build_renderer_dataset edge.
"${python_bin}" -I -S -B - \
  "${task_scratch}/runtime-source.tar" \
  "${task_scratch}/runtime-closure.json" \
  "${task_scratch}/source-tree" \
  "${source_archive_sha256}" "${closure_manifest_sha256}" <<'PY'
import ast, hashlib, json, os, re, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive, manifest_path, destination = map(Path, sys.argv[1:4])
archive_sha, manifest_sha = sys.argv[4:6]
commit = "00f7aba4bd58e67778542273cfca82fb785b648c"
root_name = "methods/bernini_action_editing"
expected = {
    "graft_phase_a_native_training_closure_v1.py": "36861e8670fc77d65b469f86aa472a10d453f9ba5fc227a62303329c4c38409a",
    "identity_rebinder_v1.py": "1f954f7446bdd4c9e465d44b045c8fe40d34dfa9c2aad6585256f5b4fb29dde0",
    "infer_lora.py": "ce0bb91aa1850fa4568b6441cc4be4f41db8b8dcfc2afe2d9fcc76a6fca2ebe4",
    "infer_native_i_axis_exact81_canary.py": "c3373d819534ac1e08270356b3311e9ef115390932bdbaa6dca64984ab618c58",
    "infer_native_identity_generation_canary.py": "a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334",
    "infer_orderless_source_frame_set_noise_canary.py": "f1e0182b6f9f6e2c4da4838b4f74f1ee629541887684a07424ec912e9050921c",
    "infer_source_kv_carrier_oracle.py": "fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604",
    "infer_source_value_residual_oracle.py": "40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8",
    "inference_sigma_strata.py": "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3",
    "native_i_axis_guidance.py": "6a7cb379403aca0f6010f14477300ad6af6d3ba8e4581ba1c5890e6ff6fed931",
    "orderless_source_frame_set_noise.py": "b5ad305798289fcf9112ce004f0305c5fea7f206c9cd9e6b331908c2f9af7bd0",
    "run_graft_phase_a_native_gpu_canary_v1.py": "e0b69442be284e091bad8d36a205bffe8bd314082188bfa55da72f4c2640945a",
    "source_kv_replay.py": "45b43426dc7825dbd61280154fc35161c60476ec5cb9e53bc0225f3809c759f3",
    "source_kv_route_batches.py": "7f3ae0d27747ad58b3b195c712884641012eb836bb59963896c58518b8b5731e",
    "source_value_residual.py": "420cadf3cb2824b2bf5a809c55086d81351db19f31743b0b77a957adf219e124",
    "train_lora.py": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "tri_branch_unipc.py": "58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

if digest(archive) != archive_sha or digest(manifest_path) != manifest_sha:
    raise SystemExit("copied archive/manifest hash differs")
raw_manifest = manifest_path.read_bytes()
try:
    manifest = json.loads(raw_manifest.decode("ascii"))
except Exception as error:
    raise SystemExit(f"closure manifest is not ASCII JSON: {error}")
canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
if raw_manifest != canonical:
    raise SystemExit("closure manifest is not canonical ASCII JSON newline")
if (
    not isinstance(manifest, dict)
    or set(manifest) != {"schema_version", "root", "selection", "source_git_commit", "files"}
    or manifest.get("schema_version") != "bernini-graft-phase-a-runtime-python-closure-v1"
    or manifest.get("root") != root_name
    or manifest.get("selection") != "commit-00f7aba-explicit-runtime-import-closure-v1"
    or manifest.get("source_git_commit") != commit
    or manifest.get("files") != expected
):
    raise SystemExit("runtime closure schema/file map differs from committed bytes")

prefix = root_name + "/"
seen = {}
with tarfile.open(archive, "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        normalized = path.as_posix().lstrip("./")
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("runtime archive contains unsafe member")
        if member.name != normalized or not member.isfile() or not normalized.startswith(prefix):
            raise SystemExit("runtime archive must contain exactly 19 regular file members and no directory members")
        relative = normalized[len(prefix):]
        if relative in seen or relative not in expected:
            raise SystemExit("runtime archive file set/duplicate differs")
        extracted = handle.extractfile(member)
        if extracted is None:
            raise SystemExit("runtime archive member cannot be read")
        payload = extracted.read()
        if hashlib.sha256(payload).hexdigest() != expected[relative]:
            raise SystemExit(f"committed runtime member bytes differ: {relative}")
        seen[relative] = payload
if set(seen) != set(expected):
    raise SystemExit(f"runtime archive lacks committed dependencies: {sorted(set(expected)-set(seen))}")

for relative, payload in sorted(seen.items()):
    target = destination / root_name / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(fd, payload[offset:])
            if wrote <= 0:
                raise SystemExit("runtime extraction made no progress")
            offset += wrote
        os.fchmod(fd, 0o444)
    finally:
        os.close(fd)

# Recompute the static local-import closure from the extracted committed bytes.
reachable = set()
queue = ["run_graft_phase_a_native_gpu_canary_v1.py"]
while queue:
    relative = queue.pop(0)
    if relative in reachable:
        continue
    reachable.add(relative)
    tree = ast.parse(seen[relative], filename=relative)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            candidates = (name.replace(".", "/") + ".py", name.split(".")[0] + ".py")
            for candidate in candidates:
                if candidate in expected and candidate not in reachable:
                    queue.append(candidate)
static_expected = set(expected) - {"tools/materialize_vae.py", "tools/build_renderer_dataset.py"}
if reachable != static_expected:
    raise SystemExit(f"static import closure differs: missing={sorted(static_expected-reachable)}, extra={sorted(reachable-static_expected)}")
if b"from tools import materialize_vae" not in seen["infer_lora.py"] or b"from tools import build_renderer_dataset as raw_builder" not in seen["tools/materialize_vae.py"]:
    raise SystemExit("dynamic materialize_vae/build_renderer_dataset import edge differs")
for relative, payload in seen.items():
    compile(payload, str(destination / root_name / relative), "exec")
runtime_root = destination / root_name
for directory in sorted(
    (path for path in runtime_root.rglob("*") if path.is_dir()),
    key=lambda path: len(path.parts), reverse=True,
):
    os.chmod(directory, 0o555)
os.chmod(runtime_root, 0o555)
PY

method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/run_graft_phase_a_native_gpu_canary_v1.py"
closure_core="${method_root}/graft_phase_a_native_training_closure_v1.py"
rebinder="${method_root}/identity_rebinder_v1.py"
/usr/bin/chmod -R a-w -- "${method_root}"
[[ "$(/usr/bin/stat -Lc '%a' -- "${method_root}")" == 555 ]] || fail "sealed runtime root mode differs"
[[ "$(/usr/bin/stat -Lc '%a' -- "${runner}")" == 444 ]] || fail "sealed runner mode differs"
[[ "$(file_sha256 "${runner}")" == "${runner_sha256}" ]] || fail "sealed runner bytes differ"
[[ "$(file_sha256 "${closure_core}")" == "${closure_core_sha256}" ]] || fail "sealed Phase-A closure bytes differ"
[[ "$(file_sha256 "${rebinder}")" == "${rebinder_sha256}" ]] || fail "sealed IdentityRebinder bytes differ"

# Import and dynamic video-runtime preflight happen before either WORLD4 group.
if ! "${python_bin}" -I -B - "${method_root}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import run_graft_phase_a_native_gpu_canary_v1 as runner
from tools import build_renderer_dataset, materialize_vae
for relative, module in {
    "run_graft_phase_a_native_gpu_canary_v1.py": runner,
    "tools/build_renderer_dataset.py": build_renderer_dataset,
    "tools/materialize_vae.py": materialize_vae,
}.items():
    observed = Path(module.__file__).resolve(strict=True)
    expected = (root / relative).resolve(strict=True)
    if observed != expected or observed.is_symlink():
        raise SystemExit(f"sealed import escaped archive: {relative}")
for name in ("_decode_exact_video", "source_aspect_bucket", "_resize_video"):
    if not callable(getattr(materialize_vae, name, None)):
        raise SystemExit(f"materialize_vae API absent: {name}")
if runner.ACTIVE_SCHEDULE_INDEX != 33 or runner.FRAME_COUNT != 81 or runner.WORLD_SIZE != 4 or runner.SP_SIZE != 4:
    raise SystemExit("runner fixed geometry/schedule constants differ")
PY
then
  fail "sealed Phase-A import/lazy-edge preflight failed"
fi

# Authenticate the two source cells, their exact media bytes, and exact81/25
# geometry before opening either NCCL WORLD4 process group.  This is a
# non-vendor preflight through the same pinned vace Python/decord decoder used
# by materialize_vae; the official runner re-authenticates and decodes each
# source again inside its own receipt boundary.
"${python_bin}" -I -B - "${cell_spec}" "${method_root}" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path

spec_path, method_root = Path(sys.argv[1]), Path(sys.argv[2]).resolve(strict=True)
sys.path.insert(0, str(method_root))
from tools import materialize_vae
root = json.loads(spec_path.read_text(encoding="utf-8"))
expected = {
    "dog": ("7b88a1ca1f804f41", "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed", 2026080825),
    "human": ("a35b590961d24694", "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed", 2026080827),
}
cells = root.get("cells")
if not isinstance(cells, list) or [row.get("cell_id") for row in cells if isinstance(row, dict)] != ["dog", "human"]:
    raise SystemExit("source cell order/schema differs")
for row in cells:
    iid, digest, seed = expected[row["cell_id"]]
    if row.get("source_iid") != iid or row.get("source_video_sha256") != digest or row.get("seeds", [None])[0] != seed:
        raise SystemExit("source cell identity/seed differs")
    path = Path(row.get("source_video", ""))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SystemExit("source video is not an absolute plain file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    h = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode): raise SystemExit("source video is not plain")
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            h.update(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise SystemExit("source video changed while hashing")
    finally:
        os.close(fd)
    if h.hexdigest() != digest:
        raise SystemExit("source video bytes differ")
    frames, fps, _ = materialize_vae._decode_exact_video(path)
    if int(frames.shape[0]) != 81 or abs(float(fps) - 25.0) > 1.0e-3:
        raise SystemExit("source video is not exact81/25")
PY

# Hash every non-cache checkpoint file from the pinned content manifest.  The
# same function is called again after both model processes exit, so model load
# cannot silently turn the preflight identity into the only observation.
checkpoint_content_snapshot() {
  "${python_bin}" -I -B - "${method_root}" "${checkpoint}" "${checkpoint_manifest}" "${checkpoint_manifest_sha256}" "${required_checkpoint_tree_sha256}" <<'PY'
import hashlib, json, sys
from pathlib import Path
method_root, checkpoint, manifest = map(Path, sys.argv[1:4])
manifest_sha, expected_tree = sys.argv[4:6]
sys.path.insert(0, str(method_root.resolve(strict=True)))
import infer_source_kv_carrier_oracle as audit
import infer_lora as legacy
validated_checkpoint, _ = legacy.trainer.validate_checkpoint(checkpoint)
if validated_checkpoint != checkpoint.resolve(strict=True) or legacy.trainer.CHECKPOINT_TREE_SHA256 != expected_tree:
    raise SystemExit("checkpoint structural tree authority differs")
receipt = audit.validate_checkpoint_content(
    validated_checkpoint,
    manifest.resolve(strict=True),
    expected_manifest_sha256=manifest_sha,
)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
print(hashlib.sha256(payload).hexdigest())
PY
}
checkpoint_content_pre_sha256="$(checkpoint_content_snapshot)"
require_sha256 "${checkpoint_content_pre_sha256}" checkpoint_content_pre_sha256

# Nothing above this line may open a WORLD4 process group.
/usr/bin/mkdir -- "${output_root}"
output_root_identity="$(/usr/bin/stat -Lc '%d:%i' -- "${output_root}")"
[[ "${output_root_identity}" =~ ^[0-9]+:[0-9]+$ ]] || fail "output root dev:ino capture failed"
/usr/bin/mkdir -p -- "${task_scratch}/dog/cache" "${task_scratch}/dog/tmp" "${task_scratch}/human/cache" "${task_scratch}/human/tmp"

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
  /usr/bin/mkdir -p -- "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"
  exec "${python_bin}" -I -B -m torch.distributed.run \
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
    --expected-phase-a-closure-sha256 "${closure_core_sha256}" \
    --expected-rebinder-sha256 "${rebinder_sha256}" \
    --expected-bernini-commit "${required_bernini_commit}" \
    --expected-veomni-commit "${required_veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${required_checkpoint_tree_sha256}" \
    --ack-wiring-fm-gradient-only-no-training-claim
)

echo "[graft-pa-v1] allocation=8xMI210 dog=WORLD4/SP4@0-3 human=WORLD4/SP4@4-7 exact81/25 index33 source-only no-op"
run_group dog 0,1,2,3 "${dog_port}" >"${output_root}/dog.launch.log" 2>&1 &
dog_pid=$!
run_group human 4,5,6,7 "${human_port}" >"${output_root}/human.launch.log" 2>&1 &
human_pid=$!

finished_pid=""
set +e
wait -n -p finished_pid "${dog_pid}" "${human_pid}"
first_status=$?
set -e
if (( first_status != 0 )); then
  kill "${dog_pid}" "${human_pid}" 2>/dev/null || true
  set +e
  wait "${dog_pid}" 2>/dev/null
  wait "${human_pid}" 2>/dev/null
  set -e
  fail "first completed WORLD4 arm failed (pid=${finished_pid},status=${first_status}); sibling terminated; output and scratch retained"
fi
if [[ "${finished_pid}" == "${dog_pid}" ]]; then
  remaining_pid="${human_pid}"; remaining_cell=human
elif [[ "${finished_pid}" == "${human_pid}" ]]; then
  remaining_pid="${dog_pid}"; remaining_cell=dog
else
  kill "${dog_pid}" "${human_pid}" 2>/dev/null || true
  set +e; wait "${dog_pid}" 2>/dev/null; wait "${human_pid}" 2>/dev/null; set -e
  fail "wait -n -p returned an unknown child; both arms terminated"
fi
set +e
wait "${remaining_pid}"
remaining_status=$?
set -e
(( remaining_status == 0 )) || fail "${remaining_cell} WORLD4/SP4 failed after sibling completed; output and scratch retained"

audit_vendor_tree "${bernini_root}" "${required_bernini_commit}" Bernini-post-run
audit_vendor_tree "${veomni_root}" "${required_veomni_commit}" VeOmni-post-run
checkpoint_content_post_sha256="$(checkpoint_content_snapshot)"
require_sha256 "${checkpoint_content_post_sha256}" checkpoint_content_post_sha256
[[ "${checkpoint_content_post_sha256}" == "${checkpoint_content_pre_sha256}" ]] || fail "checkpoint content identity changed across model load"

"${python_bin}" -I -S -B - \
  "${output_root}" "${output_root_identity}" \
  "${runner_sha256}" "${closure_core_sha256}" "${rebinder_sha256}" \
  "${source_archive_sha256}" "${closure_manifest_sha256}" "${launcher_sha256}" \
  "${cell_spec_sha256}" "${checkpoint_manifest_sha256}" \
  "${required_source_commit}" "${required_bernini_commit}" "${required_veomni_commit}" "${required_checkpoint_tree_sha256}" \
  "${checkpoint_content_pre_sha256}" "${checkpoint_content_post_sha256}" <<'PY'
# BEGIN GRAFT_PHASE_A_PARENT_VALIDATOR_V1
import hashlib, json, math, os, re, stat, sys
from pathlib import Path

SHA256 = re.compile(r"[0-9a-f]{64}")
CHILD_SCHEMA = "bernini-graft-phase-a-native-gpu-canary-v1"
PARENT_SCHEMA = "bernini-graft-phase-a-native-dual4-all8-parent-v1"

def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def object_digest(value, field):
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if not isinstance(claimed, str) or SHA256.fullmatch(claimed) is None or hashlib.sha256(canonical_bytes(unsigned)).hexdigest() != claimed:
        raise ValueError(f"{field} differs")
    return claimed

def tensor_finite(identity, *, dtype=None):
    if not isinstance(identity, dict) or identity.get("finite") is not True or not isinstance(identity.get("shape"), list):
        return False
    if dtype is not None and identity.get("dtype") != dtype:
        return False
    return all(isinstance(identity.get(name), str) and SHA256.fullmatch(identity[name]) for name in ("raw_sha256", "content_sha256"))

def validate_child(value, cell):
    if not isinstance(value, dict) or value.get("schema_version") != CHILD_SCHEMA or value.get("pass") is not True or value.get("complete") is not True:
        raise ValueError(f"{cell} child pass/schema differs")
    digest = object_digest(value, "receipt_digest")
    expected_cell = {
        "dog": ("7b88a1ca1f804f41", "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed", 2026080825),
        "human": ("a35b590961d24694", "6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed", 2026080827),
    }[cell]
    cell_row = value.get("cell", {})
    if (
        cell_row.get("cell_id") != cell or cell_row.get("source_iid") != expected_cell[0]
        or cell_row.get("source_video_sha256") != expected_cell[1]
        or cell_row.get("noise_base_seed") != expected_cell[2]
        or cell_row.get("frame_count") != 81 or cell_row.get("latent_phases") != 21
        or cell_row.get("source_metadata", {}).get("frame_count") != 81
        or float(cell_row.get("source_metadata", {}).get("fps", -1.0)) != 25.0
    ):
        raise ValueError(f"{cell} exact81/25 source cell differs")
    source_only = value.get("source_only_phase_a", {})
    noop = source_only.get("canonical_noop_r2v", {})
    if (
        source_only.get("source_video_used") is not True or source_only.get("target_video_used") is not False
        or source_only.get("generated_proposal_used") is not False or source_only.get("source_retelling_used") is not False
        or source_only.get("proposal_selection_used") is not False or source_only.get("phase_b_only") is not True
        or noop.get("source_only") is not True or noop.get("action_instruction_used") is not False
        or noop.get("guidance_mode") != "v2v_apg"
    ):
        raise ValueError(f"{cell} canonical source-only no-op differs")
    selected = value.get("schedule", {}).get("selected_coordinate", {})
    if selected.get("schedule_index") != 33 or selected.get("schedule_steps") != 40 or selected.get("timestep_dtype") != "torch.int64":
        raise ValueError(f"{cell} exact40 index33 differs")
    native = value.get("native_phase_a", {})
    local = native.get("local_receipt", {})
    core = local.get("phase_core_receipt", {})
    locality = native.get("world4_locality", {})
    shard_n = locality.get("local_shard_rows_N")
    if (
        type(shard_n) is not int or shard_n <= 0
        or locality.get("observed_local_target_rows") != [0, 0, shard_n, shard_n]
        or locality.get("replay_native_pack_leaf_all_ranks") is not True
        or locality.get("zero_target_ranks_adapter_absent_or_zero") is not True
        or locality.get("target_ranks_output_projection_local_grad_nonzero") is not True
        or locality.get("target_ranks_qkv_and_atlas_local_grad_exact_zero") is not True
    ):
        raise ValueError(f"{cell} SP4 ownership/pack leaf differs")
    if (
        native.get("vendor_apg_leaf_vjp") is not True
        or native.get("flow_matching_objective") != "same_source_noop_velocity_mean_mse"
        or core.get("guidance_mode") != "v2v_apg"
        or core.get("raw_output_dtype") != "torch.bfloat16"
        or core.get("raw_cotangent_dtype") != "torch.bfloat16"
        or core.get("per_branch_raw_replay_exact") != [True, True]
        or core.get("replay_visual_pack_detached_leaf") is not True
        or core.get("replay_pack_gradient_cleared_after_each_branch") is not True
        or not tensor_finite(local.get("flow_matching_loss"), dtype="torch.float32")
    ):
        raise ValueError(f"{cell} FM/APG/BF16 replay parity differs")
    gradients = native.get("synchronized_gradients", {})
    gate = native.get("zero_initialized_gradient_gate", {})
    if (
        gradients.get("all_rank_exact_after_sync") is not True
        or gradients.get("reduction") != "SUM_then_divide_by_WORLD4"
        or not math.isfinite(float.fromhex(gradients.get("total_l2_float64_hex", "nan")))
        or float.fromhex(gradients.get("total_l2_float64_hex", "0x0p+0")) <= 0.0
        or gate.get("gate") != "output_projection_only_nonzero"
        or gate.get("output_projection_nonzero") is not True
        or gate.get("query_key_value_exact_zero") is not True
        or gate.get("external_atlas_encoder_exact_zero") is not True
    ):
        raise ValueError(f"{cell} global output gradient gate differs")
    closure = value.get("parameter_closure", {})
    if (
        closure.get("trainable_bytes_unchanged") is not True
        or closure.get("frozen_base_bytes_unchanged") is not True
        or closure.get("frozen_base_gradients_all_none") is not True
        or closure.get("frozen_base_sha256_before") != closure.get("frozen_base_sha256_after")
    ):
        raise ValueError(f"{cell} frozen base closure differs")
    authority = value.get("authority", {})
    if authority.get("wiring_canary") is not True or authority.get("flow_matching_gradient_canary") is not True:
        raise ValueError(f"{cell} child wiring evidence absent")
    for name in (
        "semantic_success", "action_success", "quality_success", "semantic_action_success",
        "visual_quality_success", "beneficial_training_evidence", "training_positive",
        "training_run", "optimizer_created", "optimizer_step", "parameters_updated",
        "scientific_claim_authorized", "production_claim_authorized", "full_sampler_executed",
        "full_sampler_parity",
    ):
        if authority.get(name) is not False:
            raise ValueError(f"{cell} forbidden authority differs: {name}")
    return {"cell_id": cell, "receipt_digest": digest, "local_shard_rows_N": shard_n, "observed_local_target_rows": [0, 0, shard_n, shard_n], "runner_wiring_canary": True, "runner_flow_matching_gradient_canary": True}

def build_parent_receipt(children, provenance, output_identity):
    if (
        not isinstance(provenance.get("checkpoint_content_pre_sha256"), str)
        or SHA256.fullmatch(provenance["checkpoint_content_pre_sha256"]) is None
        or provenance.get("checkpoint_content_post_sha256")
        != provenance.get("checkpoint_content_pre_sha256")
    ):
        raise ValueError("checkpoint content pre/post model-load identity differs")
    rows = [validate_child(children[cell], cell) for cell in ("dog", "human")]
    expected_child_provenance = {
        "runner_sha256": provenance.get("runner_sha256"),
        "phase_a_closure_sha256": provenance.get("phase_a_closure_sha256"),
        "identity_rebinder_sha256": provenance.get("identity_rebinder_sha256"),
        "bernini_commit": provenance.get("bernini_commit"),
        "veomni_commit": provenance.get("veomni_commit"),
        "checkpoint_tree_sha256": provenance.get("checkpoint_tree_sha256"),
    }
    for cell in ("dog", "human"):
        observed = children[cell].get("provenance", {})
        if any(observed.get(name) != expected for name, expected in expected_child_provenance.items()):
            raise ValueError(f"{cell} child provenance differs from launcher authority")
        checkpoint_identity = observed.get("checkpoint_content", {}).get("identity", {})
        if (
            checkpoint_identity.get("manifest_sha256_computed")
            != provenance.get("checkpoint_content_manifest_sha256")
            or checkpoint_identity.get("manifest_sha256_expected")
            != provenance.get("checkpoint_content_manifest_sha256")
        ):
            raise ValueError(f"{cell} checkpoint content manifest provenance differs")
    authority = {
        "semantic_success": False, "action_success": False, "quality_success": False,
        "semantic_action_success": False, "visual_quality_success": False,
        "beneficial_training_evidence": False, "training_positive": False,
        "training_run": False, "optimizer_created": False, "optimizer_step": False,
        "parameters_updated": False, "scientific_claim_authorized": False,
        "production_claim_authorized": False, "full_sampler_executed": False,
        "full_sampler_parity": False,
    }
    if any(value is not False for value in authority.values()):
        raise ValueError("parent authority is not all false")
    unsigned = {
        "schema_version": PARENT_SCHEMA, "complete": True, "pass": True,
        "topology": {"allocation": "single_node_8xMI210", "groups_concurrent": True, "dog": {"physical_gpus": [0,1,2,3], "world_size": 4, "sp_size": 4}, "human": {"physical_gpus": [4,5,6,7], "world_size": 4, "sp_size": 4}},
        "cells": rows,
        "verified_evidence": {"source_exact81_25_nonvendor_decord_preflight": True, "runner_reauthenticated_source": True, "vendor_trees_tracked_clean_without_untracked_shadow_pre_and_post": True, "vendor_trees_symlink_free_outside_git_and_import_nodes_plain_pre_and_post": True, "checkpoint_content_manifest_full_rehash_pre_and_post_model_load": True, "schedule_index": 33, "canonical_noop": True, "sp4_ownership_pattern": "[0,0,N,N]", "fm_loss_finite_fp32": True, "vendor_apg_leaf_vjp": True, "bf16_raw_replay_exact": True, "native_pack_leaf_gradient_all_ranks": True, "global_output_projection_gradient_nonzero": True, "qkv_and_atlas_gradient_exact_zero_at_zero_init": True, "frozen_base_bytes_unchanged_and_gradients_none": True},
        "provenance": provenance,
        "output_directory_identity": {"st_dev": output_identity[0], "st_ino": output_identity[1], "created_fresh": True},
        "authority": authority,
        "receipt_publication": {"canonical_ascii_json_newline": True, "create_only_O_EXCL": True, "directory_fd_O_DIRECTORY_O_NOFOLLOW_retained": True, "mode": "0444", "post_write_reread_byte_exact": True, "atomic_replace_used": False},
    }
    unsigned["parent_receipt_digest"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    return unsigned
# END GRAFT_PHASE_A_PARENT_VALIDATOR_V1

def read_canonical_child(root_fd, output, cell):
    cell_path = output / cell
    cell_fd = os.open(cell, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    try:
        directory_before = os.fstat(cell_fd)
        path_before = cell_path.lstat()
        if cell_path.is_symlink() or not stat.S_ISDIR(path_before.st_mode) or (directory_before.st_dev, directory_before.st_ino) != (path_before.st_dev, path_before.st_ino):
            raise SystemExit(f"child output directory identity differs: {cell}")
        receipt_fd = os.open("receipt.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cell_fd)
        try:
            before = os.fstat(receipt_fd)
            if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444:
                raise SystemExit(f"child receipt is not plain 0444: {cell}")
            chunks = []
            while True:
                block = os.read(receipt_fd, 1024 * 1024)
                if not block: break
                chunks.append(block)
            raw = b"".join(chunks)
            after = os.fstat(receipt_fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise SystemExit(f"child receipt changed while reading: {cell}")
        finally:
            os.close(receipt_fd)
        value = json.loads(raw.decode("ascii"))
        if raw != canonical_bytes(value) + b"\n":
            raise SystemExit(f"child receipt is not canonical ASCII JSON newline: {cell}")
        directory_after = os.fstat(cell_fd)
        declared = value.get("output_directory_identity", {})
        if (
            (directory_before.st_dev, directory_before.st_ino) != (directory_after.st_dev, directory_after.st_ino)
            or [int(directory_after.st_dev), int(directory_after.st_ino)]
            != [declared.get("st_dev"), declared.get("st_ino")]
        ):
            raise SystemExit(f"child receipt/output directory binding differs: {cell}")
        return value
    finally:
        os.close(cell_fd)

output = Path(sys.argv[1])
expected_identity = tuple(int(item) for item in sys.argv[2].split(":"))
provenance_names = ("runner_sha256", "phase_a_closure_sha256", "identity_rebinder_sha256", "source_archive_sha256", "runtime_closure_manifest_sha256", "launcher_sha256", "cell_spec_sha256", "checkpoint_content_manifest_sha256", "source_git_commit", "bernini_commit", "veomni_commit", "checkpoint_tree_sha256", "checkpoint_content_pre_sha256", "checkpoint_content_post_sha256")
provenance = dict(zip(provenance_names, sys.argv[3:]))
fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    info = os.fstat(fd)
    path_info = output.lstat()
    observed_identity = (int(info.st_dev), int(info.st_ino))
    if observed_identity != expected_identity or observed_identity != (int(path_info.st_dev), int(path_info.st_ino)) or output.is_symlink():
        raise SystemExit("retained output root identity differs")
    children = {cell: read_canonical_child(fd, output, cell) for cell in ("dog", "human")}
    parent = build_parent_receipt(children, provenance, expected_identity)
    payload = canonical_bytes(parent) + b"\n"
    receipt_fd = os.open("all8-receipt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444, dir_fd=fd)
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(receipt_fd, payload[offset:])
            if wrote <= 0: raise SystemExit("parent receipt write made no progress")
            offset += wrote
        os.fchmod(receipt_fd, 0o444); os.fsync(receipt_fd)
        receipt_info = os.fstat(receipt_fd)
        if not stat.S_ISREG(receipt_info.st_mode) or stat.S_IMODE(receipt_info.st_mode) != 0o444 or receipt_info.st_size != len(payload):
            raise SystemExit("parent receipt mode/size differs")
    finally:
        os.close(receipt_fd)
    verify_fd = os.open("all8-receipt.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        observed = b""
        while True:
            block = os.read(verify_fd, 1024 * 1024)
            if not block: break
            observed += block
        if observed != payload: raise SystemExit("parent receipt reread differs")
    finally:
        os.close(verify_fd)
    os.fsync(fd)
finally:
    os.close(fd)
print(canonical_bytes(parent).decode("ascii"), flush=True)
PY

[[ "$(/usr/bin/stat -Lc '%a' -- "${output_root}/all8-receipt.json")" == 444 ]] || fail "parent receipt mode differs"
echo "GRAFT_PHASE_A_NATIVE_DUAL4_ALL8_OK"
