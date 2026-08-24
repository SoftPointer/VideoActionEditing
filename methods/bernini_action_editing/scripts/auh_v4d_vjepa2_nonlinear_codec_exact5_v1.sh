#!/usr/bin/env bash
# SEALED controller for one ordinary exact1 MI210 child step for the burned-
# development v4-D V-JEPA2 nonlinear temporal codec exact5 run.  It never
# submits a Slurm allocation; it can only enter the fixed live holder below.

set -Eeuo pipefail
umask 077

readonly tag=v4d-vjepa2-nonlinear-codec-exact5-v1
readonly release_sealed=true
readonly placeholder_sha256=0000000000000000000000000000000000000000000000000000000000000000
readonly expected_release_tree_sha256=a764287484d24ccfaf20e645c2b456d010dcb9992537f73a38bc3707ac465f81
readonly expected_runtime_sha256=20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc
readonly expected_runtime_test_sha256=5bd6076c160394832acbc63cd06a2df3acb1c69344448b5ab642ab0bfaaf967e

readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_v4c_runtime_sha256=d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_v4a_runtime_sha256=e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
readonly expected_v2_runtime_sha256=46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
readonly expected_feature_authority_sha256=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233

readonly expected_feature_receipt_sha256=895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a
readonly expected_v4c_frontier_receipt_sha256=8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9
readonly expected_v4a_receipt_sha256=568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2

readonly expected_job=143812
readonly expected_node=auh7-1b-gpu-293
readonly expected_schema=semantic-anchor-vjepa2-nonlinear-temporal-codec-exact5-receipt-v4d
readonly expected_status=V4D_VJEPA2_EXACT5_NONLINEAR_TEMPORAL_CODEC_COMPLETE_BURNED_DEVELOPMENT

readonly -a release_relative_files=(
  methods/bernini_action_editing/semantic_action_cvae_canary_v1.py
  methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
  methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
  methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
  methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
)
readonly -a release_expected_shas=(
  "${expected_feature_authority_sha256}"
  "${expected_v2_runtime_sha256}"
  "${expected_v4a_runtime_sha256}"
  "${expected_extractor_sha256}"
  "${expected_v4c_runtime_sha256}"
  "${expected_runtime_sha256}"
  "${expected_runtime_test_sha256}"
)

fail() {
  printf '[%s] ERROR: %s\n' "${tag}" "$*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

if [[ "${1:-}" == --draft-audit ]]; then
  [[ $# -eq 1 ]] || fail "--draft-audit takes no additional arguments"
  printf '{"controller":"%s","intentional_no_go":false,"launch_performed":false,"release_sealed":true,"release_tree_pin_placeholder":false,"runtime_pin_final":true,"runtime_test_pin_final":true,"detached_controller_authority":true,"job":143812,"node":"auh7-1b-gpu-293"}\n' "${tag}"
  exit 0
fi

[[ "${release_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: detached exact7 release tree is not sealed"
[[ "${expected_release_tree_sha256}" != "${placeholder_sha256}" ]] || fail "release tree SHA is placeholder"

require_plain_file() {
  local fn_path=$1 fn_expected_sha=$2 fn_expected_mode=$3 fn_label=$4 fn_expected_size=${5:-}
  [[ "${fn_path}" == /* && -f "${fn_path}" && ! -L "${fn_path}" ]] || fail "${fn_label} is not an absolute plain file"
  [[ "$(stat -c '%a:%h' "${fn_path}")" == "${fn_expected_mode}:1" ]] || fail "${fn_label} mode/link differs"
  [[ "$(sha256_file "${fn_path}")" == "${fn_expected_sha}" ]] || fail "${fn_label} SHA differs"
  if [[ -n "${fn_expected_size}" ]]; then
    [[ "$(stat -c %s "${fn_path}")" == "${fn_expected_size}" ]] || fail "${fn_label} size differs"
  fi
}

require_release() {
  local fn_root=$1 fn_index fn_path fn_listing fn_expected_listing fn_directories fn_expected_directories
  [[ "${fn_root}" == /* && -d "${fn_root}" && ! -L "${fn_root}" \
      && "${fn_root}" == "$(readlink -f -- "${fn_root}")" ]] || fail "release root differs"
  [[ "$(stat -c %a "${fn_root}")" == 555 ]] || fail "release root is not mode0555"
  [[ -z "$(find "${fn_root}" -type l -print -quit)" ]] || fail "release contains a symlink"
  fn_listing="$(cd "${fn_root}" && find . -type f -printf '%P\n' | LC_ALL=C sort)"
  fn_expected_listing="$(printf '%s\n' "${release_relative_files[@]}" | LC_ALL=C sort)"
  [[ "${fn_listing}" == "${fn_expected_listing}" ]] || fail "release is not exact7 files"
  fn_directories="$(cd "${fn_root}" && find . -mindepth 1 -type d -printf '%P\n' | LC_ALL=C sort)"
  fn_expected_directories=$'methods\nmethods/bernini_action_editing\nmethods/bernini_action_editing/tests'
  [[ "${fn_directories}" == "${fn_expected_directories}" ]] || fail "release directory membership differs"
  [[ -z "$(find "${fn_root}" -mindepth 1 ! -type d ! -type f ! -type l -print -quit)" ]] || \
    fail "release contains a special member"
  for fn_index in 0 1 2 3 4 5 6; do
    fn_path="${fn_root}/${release_relative_files[${fn_index}]}"
    require_plain_file "${fn_path}" "${release_expected_shas[${fn_index}]}" 444 \
      "release file ${release_relative_files[${fn_index}]}"
  done
  [[ -z "$(find "${fn_root}" -type d ! -perm 0555 -print -quit)" ]] || fail "release directory mode differs"
}

require_sealed_json() {
  require_plain_file "$1" "$2" 444 "$3"
}

sealed_tree_digest() {
  "$1" -I -S -B - "$2" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
root = Path(sys.argv[1])
if not root.is_absolute() or root.is_symlink():
    raise SystemExit("tree root differs")
root = root.resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit("tree symlink: " + relative)
    if stat.S_ISDIR(info.st_mode):
        if stat.S_IMODE(info.st_mode) != 0o555:
            raise SystemExit("tree directory mode: " + relative)
        continue
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
        raise SystemExit("tree file envelope: " + relative)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest(), "size_bytes": info.st_size})
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

input_snapshot() {
  local fn_python_bin=$1 fn_feature_root=$2 fn_v4c_frontier_receipt=$3 fn_v4a_receipt=$4
  "${fn_python_bin}" -I -S -B - \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

feature_root = Path(sys.argv[1]); authorities = [
    (feature_root / "feature_extraction_receipt.json", sys.argv[2], "feature"),
    (Path(sys.argv[3]), sys.argv[4], "v4c"),
    (Path(sys.argv[5]), sys.argv[6], "v4a"),
]
rows = []
def bind(path, expected, label, parse=False):
    if not path.is_absolute() or path.is_symlink() or str(path) != str(path.resolve(strict=True)):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1:
        raise SystemExit(label + " seal differs")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read()
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )
    digest = hashlib.sha256(raw)
    if digest.hexdigest() != expected:
        raise SystemExit(label + " SHA differs")
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " single-FD identity differs")
    rows.append({"label": label, "path": str(path), "sha256": expected, "size_bytes": before.st_size})
    return json.loads(raw) if parse else None

feature = None
for path, expected, label in authorities:
    value = bind(path, expected, label, parse=label == "feature")
    if label == "feature":
        feature = value
if not isinstance(feature, dict) or not isinstance(feature.get("shards"), list) or len(feature["shards"]) != 6:
    raise SystemExit("feature exact6 receipt differs")
for index, shard in enumerate(feature["shards"]):
    if not isinstance(shard, dict) or shard.get("index") != index:
        raise SystemExit("feature shard order differs")
    bind(Path(shard["path"]), shard["sha256"], "feature-shard-" + str(index))
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

snapshot_authorities() {
  [[ $# -eq 7 ]] || fail "snapshot authority argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4c_frontier_receipt=$4 fn_v4a_receipt=$5
  local fn_controller_source=$6 fn_expected_controller_sha=$7 fn_release_digest
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python
  require_release "${fn_release_root}"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller
  require_sealed_json "${fn_feature_root}/feature_extraction_receipt.json" "${expected_feature_receipt_sha256}" feature-receipt
  require_sealed_json "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
  require_sealed_json "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
  fn_release_digest="$(sealed_tree_digest "${fn_python_bin}" "${fn_release_root}")"
  [[ "${fn_release_digest}" == "${expected_release_tree_sha256}" ]] || fail "release tree SHA differs"
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python-post-snapshot
  printf '%s:%s:%s\n' \
    "${fn_release_digest}" \
    "$(input_snapshot "${fn_python_bin}" "${fn_feature_root}" "${fn_v4c_frontier_receipt}" "${fn_v4a_receipt}")" \
    "${fn_expected_controller_sha}:${expected_python_sha256}"
}

fresh_cache() {
  local fn_prefix=$1 fn_parent=/tmp fn_cache
  [[ "${fn_parent}" == /* && -d "${fn_parent}" && ! -L "${fn_parent}" && -w "${fn_parent}" ]] || fail "scratch parent differs"
  case "${fn_parent}" in /tmp|/tmp/*) ;; *) fail "scratch parent is not node-local /tmp" ;; esac
  fn_cache="$(mktemp -d "${fn_parent%/}/${fn_prefix}.XXXXXX")"
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "fresh cache creation failed"
  printf '%s\n' "${fn_cache}"
}

cleanup_cache() {
  local fn_cache=$1
  if [[ ! "${fn_cache}" =~ ^/tmp/${tag}-[A-Za-z0-9-]+\.[A-Za-z0-9]{6}$ \
        && ! "${fn_cache}" =~ ^/tmp/v4d-miopen-[0-9]+-[0-9]+-[0-9]+\.[A-Za-z0-9]{6}$ ]]; then
    fail "unsafe cache cleanup target"
  fi
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup identity differs"
  chmod -R u+w -- "${fn_cache}"
  rm -rf -- "${fn_cache}"
  [[ ! -e "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup failed"
}

gpu_gate() {
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4
  "${fn_python_bin}" -P -B - "${fn_job}" "${fn_node}" "${fn_cache}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, subprocess, sys, time, uuid

job, node, cache_text = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True))
        or cache != Path("/tmp") and Path("/tmp") not in cache.parents):
    raise SystemExit("fresh node-local cache root differs")
expected_cache_dirs = {
    "tmp", "xdg", "hf", "pycache", "torch", "user-db",
    "custom", "triton", "inductor",
}
actual_cache_dirs = set()
for path in cache.rglob("*"):
    relative = path.relative_to(cache).as_posix()
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise SystemExit("fresh cache pre-torch envelope differs: " + relative)
    actual_cache_dirs.add(relative)
if actual_cache_dirs != expected_cache_dirs:
    raise SystemExit("fresh cache pre-torch exact9 empty-directory closure differs")
expected_cache_env = {
    "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
    "XDG_CACHE_HOME": cache / "xdg", "HF_HOME": cache / "hf",
    "PYTHONPYCACHEPREFIX": cache / "pycache", "TORCH_HOME": cache / "torch",
    "MIOPEN_USER_DB_PATH": cache / "user-db",
    "MIOPEN_CUSTOM_CACHE_DIR": cache / "custom",
    "TRITON_CACHE_DIR": cache / "triton",
    "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
}
if any(os.environ.get(key) != str(value) for key, value in expected_cache_env.items()):
    raise SystemExit("fresh cache environment binding differs before torch import")

import torch

expected = {
    "SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
    "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
}
if any(os.environ.get(key) != value for key, value in expected.items()):
    raise SystemExit("Slurm exact1 task gate differs")
actual_node = socket.gethostname().split(".", 1)[0]
if actual_node != node or os.environ.get("SLURM_GPUS_ON_NODE") != "1":
    raise SystemExit("Slurm node/GPU count gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
physical_text = os.environ.get("SLURM_STEP_GPUS", "")
if not re.fullmatch(r"[0-9]+", step) or not re.fullmatch(r"[0-9]+", physical_text):
    raise SystemExit("numbered step/physical GPU authority differs")
if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("frozen torch/ROCm runtime differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("logical exact1 ROCm gate differs")
properties = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name(0)
logical_uuid = str(getattr(properties, "uuid", ""))
if name != "AMD Instinct MI210" or not logical_uuid:
    raise SystemExit("logical MI210 name/UUID gate differs")
try:
    decoded_logical_uuid = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (ValueError, UnicodeDecodeError) as error:
    raise SystemExit("logical MI210 UUID does not decode to an HSA unique ID") from error
if re.fullmatch(r"[0-9a-f]{16}", decoded_logical_uuid) is None:
    raise SystemExit("decoded logical MI210 unique ID differs")

probe_env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX"):
    probe_env.pop(key, None)
inventory = None
last_output = ""
for _ in range(3):
    probe = subprocess.run(
        ["rocm-smi", "--showuniqueid", "--json"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=probe_env,
    )
    last_output = probe.stdout + probe.stderr
    if probe.returncode == 0:
        try:
            candidate = json.loads(probe.stdout)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and len(candidate) == 1:
            inventory = candidate
            break
    time.sleep(0.5)
if inventory is None:
    raise SystemExit("single-device rocm-smi unique-ID probe failed: " + last_output)
inventory_card, row = next(iter(inventory.items()))
if not isinstance(row, dict):
    raise SystemExit("single-device rocm-smi row differs")
unique = [str(value) for key, value in row.items() if "unique" in key.lower()]
if len(unique) != 1 or not unique[0]:
    raise SystemExit("physical MI210 UUID gate differs")
def normalized(value):
    value = value.lower().replace("gpu-", "").replace("0x", "")
    return "".join(character for character in value if character in "0123456789abcdef")
if normalized(unique[0]) != decoded_logical_uuid:
    raise SystemExit("logical/physical MI210 UUID join differs")
print("V4D_GPU_GATE=" + json.dumps({
    "schema_version": "v4d-vjepa2-exact1-gpu-gate-v1",
    "job_id": job, "step_id": step, "node": actual_node,
    "ntasks": 1, "nnodes": 1, "logical_device_count": 1,
    "slurm_step_gpu_token": int(physical_text), "device_name": name,
    "logical_uuid": logical_uuid,
    "decoded_logical_unique_id": decoded_logical_uuid,
    "rocm_inventory_card": inventory_card, "physical_uuid": unique[0],
    "torch": str(torch.__version__), "torch_hip": str(torch.version.hip),
    "rocr_visible_devices_from_slurm": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "hip_visible_devices_from_slurm": os.environ.get("HIP_VISIBLE_DEVICES"),
    "cuda_visible_devices_from_slurm": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
    "miopen_user_db_path": os.environ["MIOPEN_USER_DB_PATH"],
    "miopen_custom_cache_dir": os.environ["MIOPEN_CUSTOM_CACHE_DIR"],
    "xdg_cache_home": os.environ["XDG_CACHE_HOME"],
}, sort_keys=True), flush=True)
PY
}

run_child() {
  [[ $# -eq 10 ]] || fail "internal child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4c_frontier_receipt=$4 fn_v4a_receipt=$5
  local fn_output=$6 fn_job=$7 fn_node=$8 fn_expected_controller_sha=$9 fn_pre_snapshot=${10}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_fold
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "child allocation binding differs"
  [[ ! -e "${fn_output}" && ! -L "${fn_output}" ]] || fail "receipt output is not fresh"
  for fn_fold in 0 1 2 3 4; do
    [[ ! -e "${fn_output%.json}.selected_fold${fn_fold}.pt" && ! -L "${fn_output%.json}.selected_fold${fn_fold}.pt" ]] || \
      fail "selected checkpoint output is not fresh"
  done
  [[ "$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" "${fn_feature_root}" "${fn_v4c_frontier_receipt}" "${fn_v4a_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")" == "${fn_pre_snapshot}" ]] || \
    fail "child pre-run authority differs"
  fn_runtime="${fn_release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py"
  fn_cache="$(fresh_cache "v4d-miopen-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  export TMPDIR="${fn_cache}/tmp" TMP="${fn_cache}/tmp" TEMP="${fn_cache}/tmp"
  export XDG_CACHE_HOME="${fn_cache}/xdg" HF_HOME="${fn_cache}/hf"
  export PYTHONPYCACHEPREFIX="${fn_cache}/pycache" TORCH_HOME="${fn_cache}/torch"
  export MIOPEN_USER_DB_PATH="${fn_cache}/user-db" MIOPEN_CUSTOM_CACHE_DIR="${fn_cache}/custom"
  export TRITON_CACHE_DIR="${fn_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${fn_cache}/inductor"
  mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PYTHONPYCACHEPREFIX}" \
    "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}"
  "${fn_python_bin}" -P -B "${fn_runtime}" run-exact5 \
    --feature-root "${fn_feature_root}" \
    --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
    --v4a-receipt "${fn_v4a_receipt}" \
    --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
    --v4c-frontier-receipt "${fn_v4c_frontier_receipt}" \
    --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
    --device cuda:0 --output "${fn_output}"
  fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" "${fn_feature_root}" "${fn_v4c_frontier_receipt}" "${fn_v4a_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || fail "code/input authority changed during child"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  "${fn_python_bin}" -I -S -B - "${fn_cache}" <<'PY'
from pathlib import Path
import json, sys
path = Path(sys.argv[1])
if path.exists() or path.is_symlink():
    raise SystemExit("fresh cache survived cleanup")
print("V4D_CACHE_CLEANED=" + json.dumps({
    "fresh_cache_root": str(path), "absent_after_child_cleanup": True,
}, sort_keys=True), flush=True)
PY
}

if [[ "${1:-}" == __run_child ]]; then
  shift
  run_child "$@"
  exit 0
fi

[[ $# -eq 7 ]] || fail "usage: $0 RELEASE PYTHON FEATURE_ROOT V4C_FRONTIER_RECEIPT V4A_RECEIPT FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256"
readonly release_root=$1
readonly python_bin=$2
readonly feature_root=$3
readonly v4c_frontier_receipt=$4
readonly v4a_receipt=$5
readonly run_root=$6
readonly expected_controller_sha=$7
readonly controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
readonly runtime="${release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py"
readonly test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d

[[ "${expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller SHA argument differs"
require_plain_file "${controller_source}" "${expected_controller_sha}" 555 detached-controller
[[ "${controller_source}" != "${release_root%/}/"* ]] || fail "controller must remain detached from release"
require_plain_file "${python_bin}" "${expected_python_sha256}" 755 python
require_release "${release_root}"
require_sealed_json "${feature_root}/feature_extraction_receipt.json" "${expected_feature_receipt_sha256}" feature-receipt
require_sealed_json "${v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
require_sealed_json "${v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
[[ "${run_root}" == /* && "${run_root}" != / && ! -e "${run_root}" && ! -L "${run_root}" ]] || \
  fail "run root is not fresh/absolute/safe"
[[ "${run_root}" == "$(readlink -m -- "${run_root}")" ]] || fail "run root or ancestor is not canonical"
readonly run_parent="$(dirname -- "${run_root}")"
[[ -d "${run_parent}" && ! -L "${run_parent}" && -w "${run_parent}" \
    && "${run_parent}" == "$(readlink -f -- "${run_parent}")" ]] || fail "run parent differs"

# No persistent run output exists until every CPU preflight succeeds.
preflight_cache="$(fresh_cache "${tag}-preflight")"
readonly preflight_cache
trap 'cleanup_cache "${preflight_cache}"' EXIT
export TMPDIR="${preflight_cache}/tmp" TMP="${preflight_cache}/tmp" TEMP="${preflight_cache}/tmp"
export XDG_CACHE_HOME="${preflight_cache}/xdg" HF_HOME="${preflight_cache}/hf"
export PYTHONPYCACHEPREFIX="${preflight_cache}/pycache" TORCH_HOME="${preflight_cache}/torch"
export MIOPEN_USER_DB_PATH="${preflight_cache}/user-db" MIOPEN_CUSTOM_CACHE_DIR="${preflight_cache}/custom"
export TRITON_CACHE_DIR="${preflight_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${preflight_cache}/inductor"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PYTHONPYCACHEPREFIX}" \
  "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export PYTHONPATH="${release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
readonly authority_pre="$(snapshot_authorities "${python_bin}" "${release_root}" "${feature_root}" "${v4c_frontier_receipt}" "${v4a_receipt}" "${controller_source}" "${expected_controller_sha}")"
(
  cd "${release_root}"
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    "${python_bin}" -P -B -m unittest "${test_module}"
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    "${python_bin}" -O -P -B -m unittest "${test_module}"
  "${python_bin}" -P -B -m py_compile "${release_relative_files[@]}"
  "${python_bin}" -P -B "${runtime}" --help
  run_help="$("${python_bin}" -P -B "${runtime}" run-exact5 --help)"
  for token in --feature-root --expected-feature-receipt-sha256 \
    --v4c-frontier-receipt --expected-v4c-frontier-receipt-sha256 \
    --v4a-receipt --expected-v4a-receipt-sha256 \
    --device --output; do
    grep -F -- "${token}" <<<"${run_help}" >/dev/null || fail "run-exact5 help misses ${token}"
  done
)
[[ "$(snapshot_authorities "${python_bin}" "${release_root}" "${feature_root}" "${v4c_frontier_receipt}" "${v4a_receipt}" "${controller_source}" "${expected_controller_sha}")" == "${authority_pre}" ]] || \
  fail "CPU preflight changed code/input authority"
trap - EXIT
cleanup_cache "${preflight_cache}"
unset TMPDIR TMP TEMP XDG_CACHE_HOME HF_HOME PYTHONPYCACHEPREFIX TORCH_HOME \
  MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR

projection="$(squeue -h -j "${expected_job}" -w "${expected_node}" -o '%T|%u' | tr -d ' ')"
[[ "${projection}" == "RUNNING|guangyi.chen" ]] || fail "holder ${expected_job}/${expected_node} differs"

# First persistent mutation: a fresh run directory after all gates above.
mkdir "${run_root}"
mkdir "${run_root}/logs"
chmod 0700 "${run_root}" "${run_root}/logs"
readonly output="${run_root}/receipt.json"
readonly stdout_log="${run_root}/logs/codec.stdout"
readonly stderr_log="${run_root}/logs/codec.stderr"
rc=0
(
  set -o noclobber
  exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${expected_job}" --nodelist="${expected_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=48G --gres=gpu:mi210:1 \
      --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${controller_source}" __run_child "${release_root}" "${python_bin}" \
        "${feature_root}" "${v4c_frontier_receipt}" "${v4a_receipt}" "${output}" \
        "${expected_job}" "${expected_node}" "${expected_controller_sha}" "${authority_pre}" \
      >"${stdout_log}" 2>"${stderr_log}"
) || rc=$?
[[ -f "${stdout_log}" && ! -L "${stdout_log}" && -f "${stderr_log}" && ! -L "${stderr_log}" ]] || \
  fail "create-only scientific logs are absent"
chmod 0444 "${stdout_log}" "${stderr_log}"
(( rc == 0 )) || fail "v4-D exact1 child failed with status ${rc}; incomplete run remains fail-closed"
[[ "$(snapshot_authorities "${python_bin}" "${release_root}" "${feature_root}" "${v4c_frontier_receipt}" "${v4a_receipt}" "${controller_source}" "${expected_controller_sha}")" == "${authority_pre}" ]] || \
  fail "post-run code/input authority differs"

postflight_cache="$(fresh_cache "${tag}-postflight")"
readonly postflight_cache
trap 'cleanup_cache "${postflight_cache}"' EXIT
export TMPDIR="${postflight_cache}/tmp" TMP="${postflight_cache}/tmp" TEMP="${postflight_cache}/tmp"
export XDG_CACHE_HOME="${postflight_cache}/xdg" HF_HOME="${postflight_cache}/hf"
export PYTHONPYCACHEPREFIX="${postflight_cache}/pycache" TORCH_HOME="${postflight_cache}/torch"
export MIOPEN_USER_DB_PATH="${postflight_cache}/user-db" MIOPEN_CUSTOM_CACHE_DIR="${postflight_cache}/custom"
export TRITON_CACHE_DIR="${postflight_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${postflight_cache}/inductor"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PYTHONPYCACHEPREFIX}" \
  "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
[[ "$(snapshot_authorities "${python_bin}" "${release_root}" "${feature_root}" "${v4c_frontier_receipt}" "${v4a_receipt}" "${controller_source}" "${expected_controller_sha}")" == "${authority_pre}" ]] || \
  fail "final pre-seal code/input/python/controller authority differs"

env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
"${python_bin}" -P -B - \
  "${run_root}" "${expected_schema}" "${expected_status}" \
  "${expected_runtime_sha256}" "${expected_feature_receipt_sha256}" \
  "${expected_v4c_frontier_receipt_sha256}" "${expected_v4a_receipt_sha256}" \
  "${authority_pre}" "${expected_job}" "${expected_node}" \
  "${feature_root}/feature_extraction_receipt.json" \
  "${v4c_frontier_receipt}" "${v4a_receipt}" "${expected_controller_sha}" \
  "${postflight_cache}" <<'PY'
from pathlib import Path
import hashlib, io, json, os, re, stat, sys, uuid

root_argument = Path(sys.argv[1])
if (not root_argument.is_absolute() or root_argument.is_symlink()
        or str(root_argument) != str(root_argument.resolve(strict=True))):
    raise SystemExit("run root identity differs")
root = root_argument
schema, status = sys.argv[2], sys.argv[3]
runtime_sha, feature_sha, v4c_sha, v4a_sha, authority = sys.argv[4:9]
job, node = sys.argv[9], sys.argv[10]
feature_receipt_path = str(Path(sys.argv[11]).resolve(strict=True))
v4c_receipt_path = str(Path(sys.argv[12]).resolve(strict=True))
v4a_receipt_path = str(Path(sys.argv[13]).resolve(strict=True))
controller_sha = sys.argv[14]
postflight_cache = Path(sys.argv[15])
if (not postflight_cache.is_absolute() or postflight_cache.is_symlink()
        or str(postflight_cache) != str(postflight_cache.resolve(strict=True))
        or Path("/tmp") not in postflight_cache.parents):
    raise SystemExit("postflight fresh cache root differs")
expected_cache_dirs = {
    "tmp", "xdg", "hf", "pycache", "torch", "user-db",
    "custom", "triton", "inductor",
}
if ({path.relative_to(postflight_cache).as_posix()
     for path in postflight_cache.rglob("*")} != expected_cache_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700
               for path in postflight_cache.rglob("*"))):
    raise SystemExit("postflight pre-torch exact9 empty cache differs")
expected_cache_env = {
    "TMPDIR": postflight_cache / "tmp", "TMP": postflight_cache / "tmp",
    "TEMP": postflight_cache / "tmp", "XDG_CACHE_HOME": postflight_cache / "xdg",
    "HF_HOME": postflight_cache / "hf",
    "PYTHONPYCACHEPREFIX": postflight_cache / "pycache",
    "TORCH_HOME": postflight_cache / "torch",
    "MIOPEN_USER_DB_PATH": postflight_cache / "user-db",
    "MIOPEN_CUSTOM_CACHE_DIR": postflight_cache / "custom",
    "TRITON_CACHE_DIR": postflight_cache / "triton",
    "TORCHINDUCTOR_CACHE_DIR": postflight_cache / "inductor",
}
if any(os.environ.get(key) != str(value) for key, value in expected_cache_env.items()):
    raise SystemExit("postflight cache environment differs before torch import")

import torch
if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("postflight frozen torch/ROCm runtime differs")

checkpoint_relatives = [f"receipt.selected_fold{fold}.pt" for fold in range(5)]
preseal_files = set(checkpoint_relatives) | {
    "receipt.json", "logs/codec.stdout", "logs/codec.stderr",
}

def exact_tree(expected_files, *, root_mode, directory_mode):
    root_info = root.lstat()
    if (not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != root_mode):
        raise SystemExit("run root mode/type differs")
    files, directories = set(), set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit("run tree contains symlink: " + relative)
        if stat.S_ISDIR(info.st_mode):
            directories.add(relative)
            if stat.S_IMODE(info.st_mode) != directory_mode:
                raise SystemExit("run directory mode differs: " + relative)
        elif stat.S_ISREG(info.st_mode):
            if stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
                raise SystemExit("run file seal differs: " + relative)
            files.add(relative)
        else:
            raise SystemExit("run tree member type differs: " + relative)
    if directories != {"logs"} or files != set(expected_files):
        raise SystemExit("run tree exact member closure differs")

def read_sealed_once(path, label):
    if not path.is_absolute() or path.parent not in (root, root / "logs"):
        raise SystemExit(label + " path boundary differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1):
        raise SystemExit(label + " pre-open seal differs")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit("O_NOFOLLOW is unavailable")
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444
                or info.st_nlink != 1 or (before.st_dev, before.st_ino)
                != (info.st_dev, info.st_ino)):
            raise SystemExit(label + " FD seal differs")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != info.st_size:
            raise SystemExit(label + " FD size differs")
        os.lseek(fd, 0, os.SEEK_SET)
        second_chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            second_chunks.append(chunk)
        raw_second = b"".join(second_chunks)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )
    if (stat.S_ISLNK(after.st_mode)
            or not (identity(before) == identity(info)
                    == identity(closed) == identity(after))
            or raw_second != raw):
        raise SystemExit(label + " path/FD identity changed")
    return raw, info, hashlib.sha256(raw).hexdigest()

def parse_json(raw, label):
    def reject_constant(value):
        raise ValueError("non-finite JSON constant " + value)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key " + key)
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(label + " strict JSON differs") from error

exact_tree(preseal_files, root_mode=0o700, directory_mode=0o700)
receipt_path = root / "receipt.json"
receipt_raw, receipt_info, receipt_sha = read_sealed_once(receipt_path, "receipt")
receipt = parse_json(receipt_raw, "receipt")
if not isinstance(receipt, dict):
    raise SystemExit("receipt root differs")
unsigned = dict(receipt)
embedded = unsigned.pop("receipt_digest", None)
canonical = json.dumps(
    unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
if embedded != hashlib.sha256(canonical).hexdigest():
    raise SystemExit("receipt self-digest differs")
if receipt.get("schema_version") != schema or receipt.get("status") != status:
    raise SystemExit("receipt schema/status differs")
implementation = receipt.get("implementation")
if (not isinstance(implementation, dict)
        or implementation.get("implementation_sha256") != runtime_sha):
    raise SystemExit("runtime receipt source binding differs")
feature = receipt.get("feature_authority")
if (not isinstance(feature, dict)
        or feature.get("feature_receipt_sha256") != feature_sha
        or feature.get("feature_receipt_path") != feature_receipt_path
        or feature.get("unique_original_iids") != 644
        or feature.get("family_count") != 28):
    raise SystemExit("feature receipt join differs")
upstream = receipt.get("upstream_authorities")
if (not isinstance(upstream, dict)
        or upstream.get("v4c_frontier_receipt_file_sha256") != v4c_sha
        or upstream.get("v4c_frontier_receipt_path") != v4c_receipt_path
        or upstream.get("v4a_receipt_file_sha256") != v4a_sha
        or upstream.get("v4a_receipt_path") != v4a_receipt_path):
    raise SystemExit("v4-A/v4-C receipt authority join differs")
model_contract = receipt.get("model_contract")
if (not isinstance(model_contract, dict)
        or model_contract.get("actual_code_numel") != 384
        or model_contract.get("exact_trainable_parameter_count") != 143360
        or model_contract.get("raw_input_skip_or_side_channel") is not False
        or model_contract.get("latent_scale_or_rotation_gauge_fixed") is not False):
    raise SystemExit("sole-B384 codec contract differs")
fixed_comparator = receipt.get("fixed_comparator_authority")
if (not isinstance(fixed_comparator, dict)
        or fixed_comparator.get("fixed_comparator_name") != "tucker_b0384_t04_r096"
        or fixed_comparator.get("v4c_oof_was_burned_before_v4d") is not True
        or fixed_comparator.get("clip_pca_b384_was_descriptively_higher_in_v4c") is not True
        or fixed_comparator.get("clip_pca_used_to_select_tucker_rank_or_mapping") is not False
        or fixed_comparator.get("rank_and_mapping_frozen_before_v4d_oof") is not True):
    raise SystemExit("fixed Tucker comparator authority differs")
training_contract = receipt.get("training_contract")
if (not isinstance(training_contract, dict)
        or training_contract.get("family_or_transform_labels_enter_loss_or_model_input") is not False
        or training_contract.get("family_metadata_used_for_inner_split") is not True
        or training_contract.get("transform_metadata_used_for_inner_split") is not False
        or training_contract.get("fixed_full_budget_no_early_stop") is not True
        or training_contract.get("oof_selection") is not False):
    raise SystemExit("training/metadata boundary differs")
folds = receipt.get("folds")
if (not isinstance(folds, list) or len(folds) != 5
        or [row.get("fold_index") for row in folds] != list(range(5))):
    raise SystemExit("exact5 fold closure differs")
scope = receipt.get("qualification_scope")
expected_scope_keys = {
    "temporal_codec_development_gate", "latent_metric_qualified",
    "action_representation_qualified", "scientific_confirmation_claimed",
    "identity_disentanglement_qualified", "identity_preservation_qualified",
    "vae_necessary", "generation_qualified", "prior_qualified",
    "prior_generation_qualified",
    "renderer_qualified", "video_editing_qualified", "inference_authorized",
    "web_evaluation_authorized", "training_authorized",
    "full644_refit_authorized", "video_model_training_performed",
    "postselection_all644_refit_authorized_or_performed",
}
if (not isinstance(scope, dict) or set(scope) != expected_scope_keys
        or type(scope.get("temporal_codec_development_gate")) is not bool
        or scope.get("vae_necessary") is not None):
    raise SystemExit("qualification scope schema differs")
for key in expected_scope_keys - {
    "temporal_codec_development_gate", "vae_necessary",
}:
    if scope.get(key) is not False:
        raise SystemExit("fail-closed qualification differs: " + key)
descriptive = receipt.get("descriptive_scope")
if descriptive != {
    "fold_local_model_fit_performed": True,
    "fresh_confirmation_requires_new_external_group_disjoint_data": True,
}:
    raise SystemExit("descriptive scope differs")

checkpoint_manifest = receipt.get("selected_fold_checkpoint_artifacts")
artifacts = checkpoint_manifest.get("artifacts") if isinstance(checkpoint_manifest, dict) else None
if (not isinstance(artifacts, list) or len(artifacts) != 5
        or checkpoint_manifest.get("count") != 5
        or checkpoint_manifest.get("all_create_only_mode0444_nlink1") is not True
        or checkpoint_manifest.get("artifacts_reverified_after_receipt_write_by_command_before_success_return") is not True):
    raise SystemExit("selected checkpoint receipt closure differs")
artifact_rows = []
for fold, artifact in enumerate(artifacts):
    expected_path = root / checkpoint_relatives[fold]
    path = Path(artifact.get("path", "")) if isinstance(artifact, dict) else Path("")
    if path != expected_path or str(path) != str(path.resolve(strict=True)):
        raise SystemExit("selected checkpoint path differs")
    raw, info, digest = read_sealed_once(path, f"selected checkpoint {fold}")
    if (digest != artifact.get("file_sha256")
            or info.st_size != artifact.get("size_bytes")
            or artifact.get("outer_fold") != fold
            or artifact.get("implementation_sha256") != runtime_sha
            or artifact.get("physical_identity") != {
                "device": info.st_dev, "inode": info.st_ino,
                "size_bytes": info.st_size,
            }
            or artifact.get("single_fd_pre_post_sha256_exact") is not True
            or artifact.get("semantic_metadata_state_replay_verified") is not True
            or artifact.get("caller_model_reloaded_from_sealed_artifact_before_oof") is not True):
        raise SystemExit("selected checkpoint seal/hash/join differs")
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise SystemExit("selected checkpoint safe load differs") from error
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    metadata_unsigned = dict(metadata) if isinstance(metadata, dict) else {}
    metadata_digest = metadata_unsigned.pop("metadata_digest", None)
    model_fit_iids = metadata.get("model_fit_ordered_iids") if isinstance(metadata, dict) else None
    state_semantic = {}
    if isinstance(state_dict, dict):
        for name, tensor in state_dict.items():
            if (not isinstance(name, str) or type(tensor) is not torch.Tensor
                    or not bool(torch.isfinite(tensor).all())):
                raise SystemExit("selected checkpoint state tensor differs")
            value = tensor.detach().to(device="cpu").contiguous().clone()
            tensor_digest = hashlib.sha256()
            tensor_digest.update(json.dumps(
                {"dtype": str(value.dtype), "shape": list(value.shape)},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("ascii"))
            tensor_digest.update(bytes(value.untyped_storage()))
            state_semantic[name] = tensor_digest.hexdigest()
    state_sha = hashlib.sha256(json.dumps(
        {name: state_semantic[name] for name in sorted(state_semantic)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()
    if (set(payload) != {"metadata", "state_dict"} or not isinstance(metadata, dict)
            or metadata.get("schema_version") != "semantic-anchor-vjepa2-nonlinear-temporal-codec-selected-fold-checkpoint-v4d"
            or metadata.get("outer_fold") != fold
            or metadata.get("selected_step") != artifact.get("selected_step")
            or metadata.get("model_state_sha256") != artifact.get("model_state_sha256")
            or state_sha != artifact.get("model_state_sha256")
            or metadata_digest != artifact.get("metadata_digest")
            or metadata_digest != hashlib.sha256(json.dumps(
                metadata_unsigned, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")).hexdigest()
            or metadata.get("full_budget_steps_executed") != 1200
            or metadata.get("refit_artifact") is not False
            or metadata.get("inference_authorized") is not False
            or not isinstance(model_fit_iids, list) or not model_fit_iids
            or len(model_fit_iids) != metadata.get("model_fit_original_count")
            or len(set(model_fit_iids)) != len(model_fit_iids)
            or hashlib.sha256(json.dumps(
                model_fit_iids, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")).hexdigest() != metadata.get("model_fit_iid_digest")
            or folds[fold].get("model_fit_ordered_iids") != model_fit_iids
            or folds[fold].get("model_fit_original_count") != len(model_fit_iids)
            or folds[fold].get("model_fit_iid_digest")
                != metadata.get("model_fit_iid_digest")
            or folds[fold].get("selected_checkpoint_artifact") != artifact
            or not isinstance(state_dict, dict)):
        raise SystemExit("selected checkpoint safe envelope differs")
    artifact_rows.append({
        "path": path.relative_to(root).as_posix(), "sha256": digest,
        "size_bytes": info.st_size,
    })

stdout_path = root / "logs/codec.stdout"
stderr_path = root / "logs/codec.stderr"
stdout_raw, stdout_info, stdout_sha = read_sealed_once(stdout_path, "stdout log")
stderr_raw, stderr_info, stderr_sha = read_sealed_once(stderr_path, "stderr log")
try:
    lines = stdout_raw.decode("utf-8").splitlines()
    stderr_raw.decode("utf-8")
except UnicodeDecodeError as error:
    raise SystemExit("scientific log UTF-8 differs") from error
gates = [parse_json(line.split("=", 1)[1].encode("utf-8"), "GPU gate")
         for line in lines if line.startswith("V4D_GPU_GATE=")]
cleanups = [parse_json(line.split("=", 1)[1].encode("utf-8"), "cache cleanup")
            for line in lines if line.startswith("V4D_CACHE_CLEANED=")]
results = [parse_json(line.encode("utf-8"), "runtime result")
           for line in lines if line.startswith("{")]
if len(gates) != 1 or len(cleanups) != 1 or len(results) != 1:
    raise SystemExit("stdout gate/result closure differs")
gate, result = gates[0], results[0]
logical_uuid = gate.get("logical_uuid")
decoded_unique = gate.get("decoded_logical_unique_id")
physical_uuid = gate.get("physical_uuid")
try:
    decoded_from_uuid = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (AttributeError, ValueError, UnicodeDecodeError) as error:
    raise SystemExit("sealed logical GPU UUID differs") from error
normalized_physical = "".join(
    character for character in str(physical_uuid).lower()
    .replace("gpu-", "").replace("0x", "")
    if character in "0123456789abcdef"
)
if (gate.get("schema_version") != "v4d-vjepa2-exact1-gpu-gate-v1"
        or gate.get("job_id") != job or gate.get("node") != node
        or gate.get("ntasks") != 1 or gate.get("nnodes") != 1
        or gate.get("logical_device_count") != 1
        or gate.get("device_name") != "AMD Instinct MI210"
        or re.fullmatch(r"[0-9]+", str(gate.get("step_id"))) is None
        or type(gate.get("slurm_step_gpu_token")) is not int
        or gate.get("slurm_step_gpu_token") < 0
        or gate.get("torch") != "2.7.1+rocm6.3"
        or not isinstance(gate.get("torch_hip"), str) or not gate.get("torch_hip")
        or decoded_from_uuid != decoded_unique
        or re.fullmatch(r"[0-9a-f]{16}", str(decoded_unique)) is None
        or normalized_physical != decoded_unique
        or not isinstance(gate.get("rocm_inventory_card"), str)
        or not gate.get("rocm_inventory_card")
        or re.fullmatch(
            rf"/tmp/v4d-miopen-{re.escape(job)}-{re.escape(str(gate.get('step_id')))}-0\.[A-Za-z0-9]{{6}}",
            str(gate.get("fresh_cache_root")),
        ) is None
        or gate.get("fresh_cache_pre_torch_exact9_empty_directories") is not True
        or gate.get("miopen_user_db_path")
            != gate.get("fresh_cache_root") + "/user-db"
        or gate.get("miopen_custom_cache_dir")
            != gate.get("fresh_cache_root") + "/custom"
        or gate.get("xdg_cache_home") != gate.get("fresh_cache_root") + "/xdg"
        or cleanups[0] != {
            "fresh_cache_root": gate.get("fresh_cache_root"),
            "absent_after_child_cleanup": True,
        }):
    raise SystemExit("stdout exact1 MI210 authority differs")
if (result.get("receipt") != str(receipt_path)
        or result.get("receipt_sha256") != receipt_sha
        or result.get("receipt_digest") != embedded
        or result.get("decoded_temporal_codec_development_gate")
            is not scope["temporal_codec_development_gate"]
        or result.get("latent_metric_qualified") is not False
        or result.get("selected_checkpoint_artifacts_reverified_after_receipt_write") is not True
        or result.get("feature_receipt_six_shards_v4a_v4c_reverified_after_receipt_write") is not True):
    raise SystemExit("stdout/receipt result join differs")
artifact_rows.extend((
    {"path": "receipt.json", "sha256": receipt_sha,
     "size_bytes": receipt_info.st_size},
    {"path": "logs/codec.stdout", "sha256": stdout_sha,
     "size_bytes": stdout_info.st_size},
    {"path": "logs/codec.stderr", "sha256": stderr_sha,
     "size_bytes": stderr_info.st_size},
))
if ({row["path"] for row in artifact_rows} != preseal_files
        or len(artifact_rows) != 8):
    raise SystemExit("single-FD exact8 artifact manifest differs")
exact_tree(preseal_files, root_mode=0o700, directory_mode=0o700)

seal = {
    "schema_version": "v4d-vjepa2-nonlinear-codec-exact5-run-seal-v1",
    "status": "V4D_EXACT5_ARTIFACTS_SEALED_BURNED_DEVELOPMENT",
    "authority_snapshot_pre_post": authority,
    "detached_controller_sha256": controller_sha,
    "receipt_sha256": receipt_sha,
    "temporal_codec_development_gate": scope["temporal_codec_development_gate"],
    "fresh_node_local_miopen_cache_created_before_torch_and_removed_after_child": True,
    "fresh_postflight_cache_verified_before_postflight_torch_import": True,
    "preseal_exact8_verified_single_fd_each": True,
    "postseal_target_exact9": True,
    "artifacts": sorted(artifact_rows, key=lambda row: row["path"]),
    "latent_metric_qualified": False,
    "action_representation_qualified": False,
    "scientific_confirmation_claimed": False,
    "identity_disentanglement_qualified": False,
    "identity_preservation_qualified": False,
    "prior_qualified": False,
    "prior_generation_qualified": False,
    "generation_qualified": False,
    "renderer_qualified": False,
    "video_editing_qualified": False,
    "training_authorized": False,
    "renderer_authorized": False,
    "inference_authorized": False,
    "web_evaluation_authorized": False,
    "video_editing_authorized": False,
    "full644_refit_authorized": False,
}
seal["receipt_digest"] = hashlib.sha256(json.dumps(
    seal, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
seal_raw = (json.dumps(
    seal, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
) + "\n").encode("ascii")
seal_sha = hashlib.sha256(seal_raw).hexdigest()
seal_path = root / "seal.json"
flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
if not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("O_NOFOLLOW is unavailable for seal")
fd = os.open(seal_path, flags | os.O_NOFOLLOW, 0o444)
try:
    os.fchmod(fd, 0o444)
    os.chmod(root / "logs", 0o555)
    os.chmod(root, 0o555)
    offset = 0
    while offset < len(seal_raw):
        written = os.write(fd, seal_raw[offset:])
        if written <= 0:
            raise SystemExit("seal write failed")
        offset += written
    os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    readback = b""
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        readback += chunk
    seal_info = os.fstat(fd)
    if (readback != seal_raw or hashlib.sha256(readback).hexdigest() != seal_sha
            or not stat.S_ISREG(seal_info.st_mode)
            or stat.S_IMODE(seal_info.st_mode) != 0o444
            or seal_info.st_nlink != 1 or seal_info.st_size != len(seal_raw)):
        raise SystemExit("final seal same-FD write/readback differs")
finally:
    os.close(fd)
seal_link = seal_path.lstat()
if (stat.S_ISLNK(seal_link.st_mode)
        or (seal_link.st_dev, seal_link.st_ino) != (seal_info.st_dev, seal_info.st_ino)):
    raise SystemExit("final seal path/FD identity differs")
exact_tree(preseal_files | {"seal.json"}, root_mode=0o555, directory_mode=0o555)
final_bindings = {
    row["path"]: (row["sha256"], row["size_bytes"])
    for row in artifact_rows
}
final_bindings["seal.json"] = (seal_sha, len(seal_raw))
if set(final_bindings) != preseal_files | {"seal.json"}:
    raise SystemExit("final exact9 binding membership differs")
for relative in sorted(final_bindings):
    raw, info, digest = read_sealed_once(root / relative, "final exact9 " + relative)
    expected_digest, expected_size = final_bindings[relative]
    if (digest != expected_digest or info.st_size != expected_size
            or len(raw) != expected_size):
        raise SystemExit("final exact9 rehash differs: " + relative)
print(json.dumps({
    "artifact_count": 9, "exact8_to_exact9_verified": True,
    "receipt_sha256": receipt_sha, "seal_sha256": seal_sha,
}, sort_keys=True))
PY
trap - EXIT
cleanup_cache "${postflight_cache}"
unset TMPDIR TMP TEMP XDG_CACHE_HOME HF_HOME PYTHONPYCACHEPREFIX TORCH_HOME \
  MIOPEN_USER_DB_PATH MIOPEN_CUSTOM_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
printf 'V4D_EXACT5_PASS run_root=%s\n' "${run_root}"
