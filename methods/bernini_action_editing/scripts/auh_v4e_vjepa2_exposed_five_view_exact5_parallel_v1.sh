#!/usr/bin/env bash
# SEALED executable controller for five independent ordinary exact1 MI210
# v4-E fold workers followed by one CPU-only aggregate.  This controller never
# submits an allocation and has no automatic holder/node fallback.
#
# The normal entry point requires the exact sealed runtime/test/release/Python
# pins and receipt postflight below before any persistent run-root mutation.

set -Eeuo pipefail
umask 077

readonly tag=v4e-vjepa2-exposed-five-view-exact5-parallel-v1
readonly release_sealed=true
readonly python_pin_sealed=true
readonly controller_contract_complete=true
readonly placeholder_sha256=0000000000000000000000000000000000000000000000000000000000000000

# Sealed pins.  Do not replace these independently: runtime, tests, exact
# release tree, CLI, fold schema, aggregate schema, and postflight are frozen
# as one executable contract.
readonly expected_release_tree_sha256=82b2341b8a14bb7d637b015914ae9507ddc1f205e5f6530ef32d85444c8690d4
readonly expected_runtime_sha256=4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a
readonly expected_runtime_test_sha256=a2da53be053d6ad5814c1f16c506bb0313d5c1b9360569858fcba86c8a2458a7
readonly expected_fold_schema=semantic-anchor-vjepa2-multiview-global-codec-fold-receipt-v4e
readonly expected_aggregate_schema=semantic-anchor-vjepa2-multiview-global-codec-exact5-receipt-v4e
readonly expected_aggregate_status=V4E_VJEPA2_EXPOSED_FIVE_VIEW_GLOBAL_CODEC_COMPLETE_BURNED_DEVELOPMENT
readonly expected_checkpoint_schema=semantic-anchor-vjepa2-multiview-global-codec-selected-fold-checkpoint-v4e

readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_feature_authority_sha256=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly expected_v2_runtime_sha256=46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
readonly expected_v4a_runtime_sha256=e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_v4c_runtime_sha256=d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef
readonly expected_v4d_runtime_sha256=20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc

readonly expected_feature_receipt_sha256=895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a
readonly expected_v4a_receipt_sha256=568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2
readonly expected_v4a_receipt_digest=f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86
readonly expected_v4c_frontier_receipt_sha256=8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9
readonly expected_v4c_frontier_receipt_digest=376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17
readonly expected_v4d_receipt_sha256=53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f
readonly expected_v4d_receipt_digest=45d2ae7c45f1db8ccee9b14ba8a7543cfd1ff0d311128472ae116d6befa92f9c

# Frozen no-fallback map after accounting for live four-GPU steps on nodes
# 233/268/292.  Node 292 is intentionally unused by v4-E.
readonly -a worker_folds=(0 1 2 3 4)
readonly -a worker_jobs=(143808 143812 143811 143808 143808)
readonly -a worker_nodes=(
  auh7-1b-gpu-315
  auh7-1b-gpu-293
  auh7-1b-gpu-306
  auh7-1b-gpu-233
  auh7-1b-gpu-268
)
readonly worker_cpus=8
readonly worker_memory=12G
readonly preflight_job=143811
readonly preflight_node=auh7-1b-gpu-306
readonly preflight_cpus=4
readonly preflight_memory=4G
readonly aggregate_job=143811
readonly aggregate_node=auh7-1b-gpu-306
readonly aggregate_cpus=8
readonly aggregate_memory=12G

readonly runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py
readonly runtime_test_relative=methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py
readonly -a release_relative_files=(
  methods/bernini_action_editing/semantic_action_cvae_canary_v1.py
  methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
  methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
  methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
  "${runtime_relative}"
  "${runtime_test_relative}"
)
readonly -a release_expected_shas=(
  "${expected_feature_authority_sha256}"
  "${expected_v2_runtime_sha256}"
  "${expected_v4a_runtime_sha256}"
  "${expected_extractor_sha256}"
  "${expected_v4c_runtime_sha256}"
  "${expected_v4d_runtime_sha256}"
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
  printf '%s\n' \
    '{"controller":"v4e-vjepa2-exposed-five-view-exact5-parallel-v1","intentional_no_go":false,"launch_performed":false,"release_sealed":true,"python_pin_sealed":true,"controller_contract_complete":true,"runtime_pin_final":true,"runtime_test_pin_final":true,"release_tree_content_pin_candidate":true,"fixed_no_fallback":true,"preflight_request":{"job":143811,"node":"auh7-1b-gpu-306","nodes":1,"ntasks":1,"cpus":4,"memory":"4G","gres":"none","overlap":true,"exact":true,"before_first_persistent_mkdir":true},"worker_request":{"nodes":1,"ntasks":1,"cpus":8,"memory":"12G","gres":"gpu:mi210:1","overlap":true,"exact":true},"aggregate_request":{"job":143811,"node":"auh7-1b-gpu-306","nodes":1,"ntasks":1,"cpus":8,"memory":"12G","gres":"none","overlap":true,"exact":true},"fold_mapping":[{"fold":0,"job":143808,"node":"auh7-1b-gpu-315"},{"fold":1,"job":143812,"node":"auh7-1b-gpu-293"},{"fold":2,"job":143811,"node":"auh7-1b-gpu-306"},{"fold":3,"job":143808,"node":"auh7-1b-gpu-233"},{"fold":4,"job":143808,"node":"auh7-1b-gpu-268"}],"runtime_cli":{"train":"train-fold --fold-index N --fold-root ROOT","aggregate":"aggregate --fold-root ROOT repeated exactly five times --output RECEIPT"},"fold_exact_files":["fold.json","selected.pt"],"final_layout":"launch-plan.json + logs/{fold0..4,aggregate}.{stdout,stderr} + fold0..4 exact2 dirs + aggregate/receipt.json + seal.json","launch_or_remote_action_performed":false}'
  exit 0
fi

# These sealed execution gates precede argument validation and every command
# that could inspect an authority or mutate state.
[[ "${release_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-E release/runtime/tests/receipt contract is not sealed"
[[ "${python_pin_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: pinned AUH Python entity is not sealed"
[[ "${controller_contract_complete}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-E controller postflight contract is incomplete"
[[ "${expected_release_tree_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_test_sha256}" != "${placeholder_sha256}" ]] || \
  fail "INTENTIONAL NO-GO: v4-E source/tree pins are placeholders"

# Everything below is the frozen executable contract.

require_plain_file() {
  local fn_path=$1 fn_expected_sha=$2 fn_expected_mode=$3 fn_label=$4 fn_expected_size=${5:-}
  [[ "${fn_path}" == /* && -f "${fn_path}" && ! -L "${fn_path}" ]] || \
    fail "${fn_label} is not an absolute plain file"
  [[ "$(stat -c '%a:%h' "${fn_path}")" == "${fn_expected_mode}:1" ]] || \
    fail "${fn_label} mode/link differs"
  [[ "$(sha256_file "${fn_path}")" == "${fn_expected_sha}" ]] || \
    fail "${fn_label} SHA differs"
  if [[ -n "${fn_expected_size}" ]]; then
    [[ "$(stat -c %s "${fn_path}")" == "${fn_expected_size}" ]] || \
      fail "${fn_label} size differs"
  fi
}

require_release() {
  local fn_root=$1 fn_index fn_path fn_listing fn_expected_listing
  local fn_directories fn_expected_directories
  [[ "${fn_root}" == /* && -d "${fn_root}" && ! -L "${fn_root}" \
      && "${fn_root}" == "$(readlink -f -- "${fn_root}")" ]] || \
    fail "release root differs"
  [[ "$(stat -c %a "${fn_root}")" == 555 ]] || fail "release root is not mode0555"
  [[ -z "$(find "${fn_root}" -type l -print -quit)" ]] || fail "release contains a symlink"
  fn_listing="$(cd "${fn_root}" && find . -type f -printf '%P\n' | LC_ALL=C sort)"
  fn_expected_listing="$(printf '%s\n' "${release_relative_files[@]}" | LC_ALL=C sort)"
  [[ "${fn_listing}" == "${fn_expected_listing}" ]] || fail "release is not exact8 files"
  fn_directories="$(cd "${fn_root}" && find . -mindepth 1 -type d -printf '%P\n' | LC_ALL=C sort)"
  fn_expected_directories=$'methods\nmethods/bernini_action_editing\nmethods/bernini_action_editing/tests'
  [[ "${fn_directories}" == "${fn_expected_directories}" ]] || \
    fail "release directory membership differs"
  [[ -z "$(find "${fn_root}" -mindepth 1 ! -type d ! -type f ! -type l -print -quit)" ]] || \
    fail "release contains a special member"
  for fn_index in 0 1 2 3 4 5 6 7; do
    fn_path="${fn_root}/${release_relative_files[${fn_index}]}"
    require_plain_file "${fn_path}" "${release_expected_shas[${fn_index}]}" 444 \
      "release file ${release_relative_files[${fn_index}]}"
  done
  [[ -z "$(find "${fn_root}" -type d ! -perm 0555 -print -quit)" ]] || \
    fail "release directory mode differs"
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
  local fn_python_bin=$1 fn_feature_root=$2 fn_v4a_receipt=$3
  local fn_v4c_frontier_receipt=$4 fn_v4d_receipt=$5
  "${fn_python_bin}" -I -S -B - \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" \
    "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

feature_root = Path(sys.argv[1])
authorities = [
    (feature_root / "feature_extraction_receipt.json", sys.argv[2], "feature"),
    (Path(sys.argv[3]), sys.argv[4], "v4a"),
    (Path(sys.argv[5]), sys.argv[6], "v4c"),
    (Path(sys.argv[7]), sys.argv[8], "v4d"),
]
rows = []
def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mode,
            value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
def bind(path, expected, label, parse=False):
    if not path.is_absolute() or path.is_symlink() or str(path) != str(path.resolve(strict=True)):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1:
        raise SystemExit(label + " seal differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " single-FD identity differs")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(label + " SHA differs")
    rows.append({"label": label, "path": str(path), "sha256": expected,
                 "size_bytes": before.st_size})
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
  [[ $# -eq 8 ]] || fail "snapshot authority argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_controller_source=$7
  local fn_expected_controller_sha=$8 fn_release_digest fn_input_digest
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python
  require_release "${fn_release_root}"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller
  require_sealed_json "${fn_feature_root}/feature_extraction_receipt.json" \
    "${expected_feature_receipt_sha256}" feature-receipt
  require_sealed_json "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
  require_sealed_json "${fn_v4c_frontier_receipt}" \
    "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
  require_sealed_json "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" v4d-receipt
  if ! fn_release_digest="$(sealed_tree_digest "${fn_python_bin}" "${fn_release_root}")"; then
    fail "release tree snapshot failed"
  fi
  [[ "${fn_release_digest}" == "${expected_release_tree_sha256}" ]] || \
    fail "release tree SHA differs"
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python-post-snapshot
  if ! fn_input_digest="$(input_snapshot "${fn_python_bin}" "${fn_feature_root}" \
       "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" "${fn_v4d_receipt}")"; then
    fail "input authority snapshot failed"
  fi
  [[ "${fn_input_digest}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "input authority snapshot digest differs"
  printf '%s:%s:%s\n' \
    "${fn_release_digest}" \
    "${fn_input_digest}" \
    "${fn_expected_controller_sha}:${expected_python_sha256}"
}

fresh_cache() {
  local fn_prefix=$1 fn_parent=/tmp fn_cache
  [[ "${fn_parent}" == /* && -d "${fn_parent}" && ! -L "${fn_parent}" && -w "${fn_parent}" ]] || \
    fail "scratch parent differs"
  fn_cache="$(mktemp -d "${fn_parent%/}/${fn_prefix}.XXXXXX")"
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "fresh cache creation failed"
  printf '%s\n' "${fn_cache}"
}

activate_cache() {
  local fn_cache=$1
  export TMPDIR="${fn_cache}/tmp" TMP="${fn_cache}/tmp" TEMP="${fn_cache}/tmp"
  export XDG_CACHE_HOME="${fn_cache}/xdg" HF_HOME="${fn_cache}/hf"
  export PYTHONPYCACHEPREFIX="${fn_cache}/pycache" TORCH_HOME="${fn_cache}/torch"
  export MIOPEN_USER_DB_PATH="${fn_cache}/user-db" MIOPEN_CUSTOM_CACHE_DIR="${fn_cache}/custom"
  export TRITON_CACHE_DIR="${fn_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${fn_cache}/inductor"
  mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PYTHONPYCACHEPREFIX}" \
    "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  chmod 0700 "${fn_cache}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" \
    "${PYTHONPYCACHEPREFIX}" "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" \
    "${MIOPEN_CUSTOM_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
}

cleanup_cache() {
  local fn_cache=$1
  [[ "${fn_cache}" =~ ^/tmp/${tag}-[A-Za-z0-9-]+\.[A-Za-z0-9]{6}$ \
      || "${fn_cache}" =~ ^/tmp/v4e-miopen-[0-9]+-[0-9]+-[0-9]+-fold[0-4]\.[A-Za-z0-9]{6}$ ]] || \
    fail "unsafe cache cleanup target"
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup identity differs"
  chmod -R u+w -- "${fn_cache}"
  rm -rf -- "${fn_cache}"
  [[ ! -e "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup failed"
}

gpu_gate() {
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_fold=$5
  "${fn_python_bin}" -P -B - "${fn_job}" "${fn_node}" "${fn_cache}" "${fn_fold}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, subprocess, sys, time, uuid

job, node, cache_text, fold = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True))
        or Path("/tmp") not in cache.parents):
    raise SystemExit("fresh node-local cache root differs")
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if ({path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("fresh cache pre-torch exact9 empty-directory closure differs")
expected_env = {
    "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
    "XDG_CACHE_HOME": cache / "xdg", "HF_HOME": cache / "hf",
    "PYTHONPYCACHEPREFIX": cache / "pycache", "TORCH_HOME": cache / "torch",
    "MIOPEN_USER_DB_PATH": cache / "user-db",
    "MIOPEN_CUSTOM_CACHE_DIR": cache / "custom",
    "TRITON_CACHE_DIR": cache / "triton",
    "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
}

if any(os.environ.get(key) != str(value) for key, value in expected_env.items()):
    raise SystemExit("fresh cache environment binding differs before torch import")

import torch

expected_slurm = {"SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
                  "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
                  "SLURM_CPUS_PER_TASK": "8"}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
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
    decoded = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (ValueError, UnicodeDecodeError) as error:
    raise SystemExit("logical MI210 UUID does not decode to HSA unique ID") from error
if re.fullmatch(r"[0-9a-f]{16}", decoded) is None:
    raise SystemExit("decoded logical MI210 unique ID differs")
probe_env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX"):
    probe_env.pop(key, None)
inventory = None
last_output = ""
for _ in range(3):
    probe = subprocess.run(["rocm-smi", "--showuniqueid", "--json"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=probe_env)
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
card, row = next(iter(inventory.items()))
unique = [str(value) for key, value in row.items() if "unique" in key.lower()]
if len(unique) != 1 or not unique[0]:
    raise SystemExit("physical MI210 UUID gate differs")
def normalized(value):
    value = value.lower().replace("gpu-", "").replace("0x", "")
    return "".join(character for character in value if character in "0123456789abcdef")
if normalized(unique[0]) != decoded:
    raise SystemExit("logical/physical MI210 UUID join differs")
print("V4E_GPU_GATE=" + json.dumps({
    "schema_version": "v4e-vjepa2-exact1-fold-gpu-gate-v1",
    "fold_index": int(fold), "job_id": job, "step_id": step,
    "node": actual_node, "ntasks": 1, "nnodes": 1,
    "cpus_per_task": 8,
    "logical_device_count": 1, "slurm_step_gpu_token": int(physical_text),
    "device_name": name, "logical_uuid": logical_uuid,
    "decoded_logical_unique_id": decoded, "rocm_inventory_card": card,
    "physical_uuid": unique[0], "torch": str(torch.__version__),
    "torch_hip": str(torch.version.hip), "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True), flush=True)
PY
}

cpu_step_gate() {
  [[ $# -eq 7 ]] || fail "CPU step gate argument count differs"
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_role=$5 fn_schema=$6
  local fn_cpus=$7
  "${fn_python_bin}" -I -S -B - "${fn_job}" "${fn_node}" "${fn_cache}" \
    "${fn_role}" "${fn_schema}" "${fn_cpus}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, sys

job, node, cache_text, role, schema, cpus = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True))
        or Path("/tmp") not in cache.parents):
    raise SystemExit("CPU step fresh cache root differs")
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if ({path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("CPU step fresh cache exact9 closure differs")
expected_env = {
    "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
    "XDG_CACHE_HOME": cache / "xdg", "HF_HOME": cache / "hf",
    "PYTHONPYCACHEPREFIX": cache / "pycache", "TORCH_HOME": cache / "torch",
    "MIOPEN_USER_DB_PATH": cache / "user-db",
    "MIOPEN_CUSTOM_CACHE_DIR": cache / "custom",
    "TRITON_CACHE_DIR": cache / "triton",
    "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
}
if any(os.environ.get(key) != str(value) for key, value in expected_env.items()):
    raise SystemExit("CPU step cache environment differs")
expected_slurm = {
    "SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
    "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_CPUS_PER_TASK": cpus,
}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
    raise SystemExit("CPU exact1 Slurm task gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
if re.fullmatch(r"[0-9]+", step) is None:
    raise SystemExit("CPU step must have a numbered Slurm step ID")
if socket.gethostname().split(".", 1)[0] != node:
    raise SystemExit("CPU step node differs")
if os.environ.get("SLURM_GPUS_ON_NODE") not in (None, "", "0"):
    raise SystemExit("CPU step unexpectedly received a GPU")
if os.environ.get("SLURM_STEP_GPUS") not in (None, ""):
    raise SystemExit("CPU step GPU token differs")
if any(os.environ.get(key) is not None for key in (
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)):
    raise SystemExit("CPU step inherited a GPU visibility variable")
print("V4E_CPU_STEP_GATE=" + json.dumps({
    "schema_version": schema, "role": role, "job_id": job,
    "step_id": step, "node": node,
    "ntasks": 1, "nnodes": 1, "cpus_per_task": int(cpus),
    "gres": "none", "gpu_visibility_variables_absent": True,
    "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True), flush=True)
PY
}

verify_fold_exact2() {
  [[ $# -eq 8 ]] || fail "verify fold argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_fold_root=$7 fn_fold=$8
  "${fn_python_bin}" -I -S -B - \
    "${fn_fold_root}" "${fn_fold}" "${expected_fold_schema}" \
    "${expected_aggregate_status}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py" \
    "${expected_v4c_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py" \
    "${expected_extractor_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py" \
    "${expected_v4a_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py" \
    "${expected_v4d_runtime_sha256}" \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" \
    "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, math, os, stat, sys

root = Path(sys.argv[1]); fold = int(sys.argv[2]); schema = sys.argv[3]
status = sys.argv[4]
implementation = {
    "implementation_path": str(Path(sys.argv[5]).resolve(strict=True)),
    "implementation_sha256": sys.argv[6],
    "v4c_implementation_path": str(Path(sys.argv[7]).resolve(strict=True)),
    "v4c_implementation_sha256": sys.argv[8],
    "extractor_implementation_path": str(Path(sys.argv[9]).resolve(strict=True)),
    "extractor_implementation_sha256": sys.argv[10],
    "v4a_implementation_path": str(Path(sys.argv[11]).resolve(strict=True)),
    "v4a_implementation_sha256": sys.argv[12],
    "v4d_implementation_path": str(Path(sys.argv[13]).resolve(strict=True)),
    "v4d_implementation_sha256": sys.argv[14],
}
feature_root = str(Path(sys.argv[15]).resolve(strict=True)); feature_sha = sys.argv[16]
v4a_path = str(Path(sys.argv[17]).resolve(strict=True)); v4a_sha = sys.argv[18]
v4c_path = str(Path(sys.argv[19]).resolve(strict=True)); v4c_sha = sys.argv[20]
v4d_path = str(Path(sys.argv[21]).resolve(strict=True)); v4d_sha = sys.argv[22]
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or stat.S_IMODE(root.lstat().st_mode) != 0o555):
    raise SystemExit("fold root seal differs")
paths = list(root.iterdir())
if {path.name for path in paths} != {"fold.json", "selected.pt"}:
    raise SystemExit("fold exact2 closure differs")
identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode,
                          value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)

def read_sealed(path, label):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size <= 0):
        raise SystemExit(label + " pre-open seal differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor); digest = hashlib.sha256(); chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk: break
            digest.update(chunk); chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " single-FD identity differs")
    return b"".join(chunks), before, digest.hexdigest()

def pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result

def nonfinite(value):
    raise ValueError("non-finite JSON constant: " + value)

def object_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

receipt_raw, receipt_info, receipt_sha = read_sealed(root / "fold.json", "fold receipt")
checkpoint_raw, checkpoint_info, checkpoint_sha = read_sealed(
    root / "selected.pt", "selected checkpoint"
)
receipt = json.loads(receipt_raw, object_pairs_hook=pairs, parse_constant=nonfinite)
unsigned = dict(receipt) if type(receipt) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
fold_row = receipt.get("fold") if type(receipt) is dict else None
artifact = fold_row.get("selected_checkpoint_artifact") if type(fold_row) is dict else None
training = fold_row.get("training") if type(fold_row) is dict else None
fit_iids = fold_row.get("model_fit_ordered_iids") if type(fold_row) is dict else None
oof_iids = fold_row.get("oof_ordered_iids") if type(fold_row) is dict else None
inner_iids = training.get("inner_validation_ordered_iids") if type(training) is dict else None
scope = receipt.get("qualification_scope") if type(receipt) is dict else None
false_scope = {
    "latent_metric_qualified", "action_representation_qualified",
    "identity_disentanglement_qualified", "identity_preservation_qualified",
    "prior_qualified", "prior_generation_qualified", "generation_qualified",
    "renderer_qualified", "video_editing_qualified", "inference_authorized",
    "web_evaluation_authorized", "full644_refit_authorized",
}
expected_oof = (131, 127, 128, 129, 129)
if (
    type(receipt) is not dict or receipt.get("schema_version") != schema
    or receipt.get("status") != status
    or receipt.get("authority") != "burned_exposed_known_transform_development_fold_only"
    or receipt_digest != object_sha(unsigned)
    or receipt.get("implementation") != implementation
    or receipt.get("fold_root") != str(root)
    or receipt.get("feature_authority", {}).get("feature_root") != feature_root
    or receipt.get("feature_authority", {}).get("feature_receipt_sha256") != feature_sha
    or receipt.get("upstream_authorities", {}).get("v4a_receipt_path") != v4a_path
    or receipt.get("upstream_authorities", {}).get("v4a_receipt_sha256") != v4a_sha
    or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_path") != v4c_path
    or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_sha256") != v4c_sha
    or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_path") != v4d_path
    or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_sha256") != v4d_sha
    or type(fold_row) is not dict or fold_row.get("fold_index") != fold
    or type(fit_iids) is not list or len(set(fit_iids)) != len(fit_iids)
    or len(fit_iids) != fold_row.get("model_fit_original_count")
    or object_sha(fit_iids) != fold_row.get("model_fit_iid_digest")
    or type(inner_iids) is not list or len(set(inner_iids)) != len(inner_iids)
    or len(inner_iids) != fold_row.get("inner_validation_original_count")
    or object_sha(inner_iids) != fold_row.get("inner_validation_iid_digest")
    or type(oof_iids) is not list or len(set(oof_iids)) != len(oof_iids)
    or len(oof_iids) != expected_oof[fold]
    or len(oof_iids) != fold_row.get("oof_original_count")
    or object_sha(oof_iids) != fold_row.get("oof_iid_digest")
    or len(set(fit_iids + inner_iids + oof_iids)) != 644
    or receipt.get("oof_evidence_count") != expected_oof[fold]
    or receipt.get("oof_evidence_sha256") != object_sha(receipt.get("oof_evidence"))
    or type(artifact) is not dict
    or artifact.get("path") != str((root / "selected.pt").resolve(strict=True))
    or artifact.get("file_sha256") != checkpoint_sha
    or artifact.get("size_bytes") != checkpoint_info.st_size
    or artifact.get("mode_octal") != "0444" or artifact.get("nlink") != 1
    or artifact.get("physical_identity") != {
        "device": checkpoint_info.st_dev, "inode": checkpoint_info.st_ino,
        "size_bytes": checkpoint_info.st_size,
    }
    or artifact.get("outer_fold") != fold
    or artifact.get("selected_step") != training.get("selected_step")
    or artifact.get("model_state_sha256") != training.get("selected_state_sha256")
    or artifact.get("implementation_sha256") != implementation["implementation_sha256"]
    or artifact.get("single_fd_pre_post_sha256_exact") is not True
    or artifact.get("semantic_metadata_state_replay_verified") is not True
    or artifact.get("fresh_reload_strict_state_verified") is not True
    or artifact.get("fresh_reload_output_bit_exact") is not True
    or artifact.get("caller_model_reloaded_from_sealed_artifact_before_oof") is not True
    or type(scope) is not dict
    or scope.get("exposed_five_view_codec_development_gate") is not None
    or scope.get("aggregate_gate_evaluated") is not False
    or scope.get("vae_necessary") is not None
    or any(scope.get(key) is not False for key in false_scope)
):
    raise SystemExit("fold receipt/checkpoint semantic closure differs")
print(json.dumps({
    "fold": fold, "receipt_sha256": receipt_sha,
    "receipt_digest": receipt_digest, "receipt_size_bytes": receipt_info.st_size,
    "selected_sha256": checkpoint_sha,
    "selected_size_bytes": checkpoint_info.st_size,
    "selected_step": training["selected_step"],
}, sort_keys=True, separators=(",", ":")))
PY
}

run_fold_child() {
  [[ $# -eq 12 ]] || fail "internal fold child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_fold_root=$7 fn_fold=$8
  local fn_job=$9 fn_node=${10} fn_expected_controller_sha=${11} fn_pre_snapshot=${12}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_fold_parent
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${worker_folds[${fn_fold}]}" == "${fn_fold}" \
      && "${worker_jobs[${fn_fold}]}" == "${fn_job}" \
      && "${worker_nodes[${fn_fold}]}" == "${fn_node}" ]] || \
    fail "fold holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "child allocation binding differs"
  [[ "${fn_fold_root}" == /* && ! -e "${fn_fold_root}" && ! -L "${fn_fold_root}" ]] || \
    fail "fold root is not fresh"
  fn_fold_parent="$(dirname -- "${fn_fold_root}")"
  [[ "$(basename -- "${fn_fold_root}")" == "fold${fn_fold}" \
      && -d "${fn_fold_parent}" && ! -L "${fn_fold_parent}" \
      && "${fn_fold_parent}" == "$(readlink -f -- "${fn_fold_parent}")" \
      && "$(stat -c %a "${fn_fold_parent}")" == 700 ]] || \
    fail "fold root parent/name authority differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "fold child pre-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "fold child pre-run authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "v4e-miopen-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}-fold${fn_fold}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" "${fn_fold}"
  mkdir "${fn_fold_root}"
  chmod 0700 "${fn_fold_root}"
  "${fn_python_bin}" -P -B "${fn_runtime}" train-fold \
    --feature-root "${fn_feature_root}" \
    --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
    --v4a-receipt "${fn_v4a_receipt}" \
    --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
    --v4c-frontier-receipt "${fn_v4c_frontier_receipt}" \
    --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
    --v4d-receipt "${fn_v4d_receipt}" \
    --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
    --fold-index "${fn_fold}" --fold-root "${fn_fold_root}" --device cuda:0
  chmod 0555 "${fn_fold_root}"
  verify_fold_exact2 "${fn_python_bin}" "${fn_release_root}" "${fn_feature_root}" \
    "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" "${fn_v4d_receipt}" \
    "${fn_fold_root}" "${fn_fold}"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "fold child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during fold child"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4E_CACHE_CLEANED={"fold_index":%s,"fresh_cache_root":"%s","absent_after_child_cleanup":true}\n' \
    "${fn_fold}" "${fn_cache}"
}

run_exact_test_suite() {
  [[ $# -eq 3 ]] || fail "exact test runner argument count differs"
  local fn_python_bin=$1 fn_optimized=$2 fn_test_module=$3
  local -a fn_flags=(-P -B)
  if [[ "${fn_optimized}" == true ]]; then
    fn_flags=(-O -P -B)
  elif [[ "${fn_optimized}" != false ]]; then
    fail "exact test runner optimization flag differs"
  fi
  "${fn_python_bin}" "${fn_flags[@]}" - "${fn_test_module}" <<'PY'
import importlib, json, sys, unittest
import torch

if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("preflight frozen torch/ROCm runtime differs")
module = importlib.import_module(sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
summary = {
    "errors": len(result.errors), "expected_failures": len(result.expectedFailures),
    "failures": len(result.failures), "optimized": bool(sys.flags.optimize),
    "skipped": len(result.skipped), "tests_run": result.testsRun,
    "unexpected_successes": len(result.unexpectedSuccesses),
}
if summary != {
    "errors": 0, "expected_failures": 0, "failures": 0,
    "optimized": bool(sys.flags.optimize), "skipped": 0, "tests_run": 28,
    "unexpected_successes": 0,
}:
    raise SystemExit("preflight exact28 no-skip test closure differs: " + repr(summary))
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
PY
}

run_preflight_child() {
  [[ $# -eq 10 ]] || fail "internal preflight child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_job=$7 fn_node=$8
  local fn_expected_controller_sha=$9 fn_pre_snapshot=${10}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_gate fn_gate_json
  local fn_train_help fn_aggregate_help fn_token fn_test_module
  local fn_normal_stdout fn_normal_stderr fn_optimized_stdout fn_optimized_stderr
  local fn_normal_pid fn_optimized_pid fn_normal_rc fn_optimized_rc
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${preflight_job}" && "${fn_node}" == "${preflight_node}" ]] || \
    fail "preflight fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-preflight-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "preflight allocation binding differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "preflight child initial authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "preflight child initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "${tag}-preflight-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  fn_gate="$(cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    preflight v4e-vjepa2-exact1-cpu-preflight-gate-v1 "${preflight_cpus}")"
  [[ "${fn_gate}" == V4E_CPU_STEP_GATE=* ]] || fail "preflight CPU gate output differs"
  fn_gate_json=${fn_gate#V4E_CPU_STEP_GATE=}
  fn_test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_multiview_global_codec_v4e_alt
  fn_normal_stdout="${fn_cache}/normal-tests.stdout"
  fn_normal_stderr="${fn_cache}/normal-tests.stderr"
  fn_optimized_stdout="${fn_cache}/optimized-tests.stdout"
  fn_optimized_stderr="${fn_cache}/optimized-tests.stderr"
  (
    set -o noclobber
    cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" false "${fn_test_module}" \
      >"${fn_normal_stdout}" 2>"${fn_normal_stderr}"
  ) &
  fn_normal_pid=$!
  (
    set -o noclobber
    cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" true "${fn_test_module}" \
      >"${fn_optimized_stdout}" 2>"${fn_optimized_stderr}"
  ) &
  fn_optimized_pid=$!
  fn_normal_rc=0
  fn_optimized_rc=0
  wait "${fn_normal_pid}" || fn_normal_rc=$?
  wait "${fn_optimized_pid}" || fn_optimized_rc=$?
  if (( fn_normal_rc != 0 || fn_optimized_rc != 0 )); then
    printf '%s\n' "normal test rc=${fn_normal_rc}" "optimized test rc=${fn_optimized_rc}" >&2
    sed -n '1,240p' "${fn_normal_stdout}" "${fn_normal_stderr}" \
      "${fn_optimized_stdout}" "${fn_optimized_stderr}" >&2
    fail "parallel exact28 no-skip test suites failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_normal_stdout}" "${fn_optimized_stdout}" \
    "${fn_normal_stderr}" "${fn_optimized_stderr}" <<'PY'
from pathlib import Path
import json, sys
expected_common = {
    "errors": 0, "expected_failures": 0, "failures": 0, "skipped": 0,
    "tests_run": 28, "unexpected_successes": 0,
}
for index, optimized in enumerate((False, True), start=1):
    value = json.loads(Path(sys.argv[index]).read_text("ascii"))
    expected = {**expected_common, "optimized": optimized}
    if value != expected:
        raise SystemExit("parallel test result JSON differs")
for index in (3, 4):
    lines = Path(sys.argv[index]).read_text(encoding="utf-8", errors="strict").splitlines()
    if (not any(line.startswith("Ran 28 tests in ") for line in lines)
            or "OK" not in lines
            or any("skipped=" in line for line in lines)):
        raise SystemExit("parallel unittest text summary differs")
PY
  (
    cd "${fn_release_root}"
    "${fn_python_bin}" -P -B -m py_compile "${release_relative_files[@]}"
  )
  fn_train_help="$("${fn_python_bin}" -P -B "${fn_runtime}" train-fold --help)"
  fn_aggregate_help="$("${fn_python_bin}" -P -B "${fn_runtime}" aggregate --help)"
  for fn_token in --feature-root --expected-feature-receipt-sha256 \
    --v4a-receipt --expected-v4a-receipt-sha256 \
    --v4c-frontier-receipt --expected-v4c-frontier-receipt-sha256 \
    --v4d-receipt --expected-v4d-receipt-sha256 --fold-root; do
    grep -F -- "${fn_token}" <<<"${fn_train_help}" >/dev/null || \
      fail "train-fold help misses ${fn_token}"
    grep -F -- "${fn_token}" <<<"${fn_aggregate_help}" >/dev/null || \
      fail "aggregate help misses ${fn_token}"
  done
  for fn_token in --fold-index --device; do
    grep -F -- "${fn_token}" <<<"${fn_train_help}" >/dev/null || \
      fail "train-fold help misses ${fn_token}"
  done
  grep -F -- --output <<<"${fn_aggregate_help}" >/dev/null || \
    fail "aggregate help misses --output"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "preflight child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during CPU preflight"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4E_PREFLIGHT_PASS={"job":%s,"node":"%s","cpus":%s,"memory":"4G","gres":"none","cpu_gate":%s,"normal_tests_passed":true,"normal_tests_run":28,"normal_tests_skipped":0,"optimized_tests_passed":true,"optimized_tests_run":28,"optimized_tests_skipped":0,"tests_ran_in_parallel":true,"omp_threads_per_test_process":1,"compile_passed":true,"cli_help_passed":true,"fresh_cache_cleaned":true}\n' \
    "${fn_job}" "${fn_node}" "${preflight_cpus}" "${fn_gate_json}"
}

run_aggregate_child() {
  [[ $# -eq 11 ]] || fail "internal aggregate child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_job=$8
  local fn_node=$9 fn_expected_controller_sha=${10} fn_pre_snapshot=${11}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_output
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${aggregate_job}" && "${fn_node}" == "${aggregate_node}" ]] || \
    fail "aggregate fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-aggregate-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "aggregate allocation binding differs"
  [[ "${fn_run_root}" == /* && -d "${fn_run_root}" && ! -L "${fn_run_root}" \
      && "${fn_run_root}" == "$(readlink -f -- "${fn_run_root}")" \
      && "$(stat -c %a "${fn_run_root}")" == 700 ]] || \
    fail "aggregate run root differs"
  [[ -d "${fn_run_root}/aggregate" && ! -L "${fn_run_root}/aggregate" \
      && "$(stat -c %a "${fn_run_root}/aggregate")" == 700 \
      && -z "$(find "${fn_run_root}/aggregate" -mindepth 1 -print -quit)" ]] || \
    fail "aggregate output directory is not fresh exact0"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "aggregate child pre-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "aggregate child pre-run authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_output="${fn_run_root}/aggregate/receipt.json"
  fn_cache="$(fresh_cache "${tag}-aggregate-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    aggregate v4e-vjepa2-exact1-cpu-aggregate-gate-v1 "${aggregate_cpus}"
  "${fn_python_bin}" -P -B "${fn_runtime}" aggregate \
    --feature-root "${fn_feature_root}" \
    --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
    --v4a-receipt "${fn_v4a_receipt}" \
    --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
    --v4c-frontier-receipt "${fn_v4c_frontier_receipt}" \
    --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
    --v4d-receipt "${fn_v4d_receipt}" \
    --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
    --fold-root "${fn_run_root}/fold0" --fold-root "${fn_run_root}/fold1" \
    --fold-root "${fn_run_root}/fold2" --fold-root "${fn_run_root}/fold3" \
    --fold-root "${fn_run_root}/fold4" --output "${fn_output}"
  [[ "$(stat -c '%a:%h' "${fn_output}")" == 444:1 && ! -L "${fn_output}" ]] || \
    fail "aggregate receipt output seal differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "aggregate child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during aggregate child"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4E_CACHE_CLEANED={"role":"aggregate","fresh_cache_root":"%s","absent_after_child_cleanup":true}\n' \
    "${fn_cache}"
}

run_seal_child() {
  [[ $# -eq 11 ]] || fail "internal seal child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_job=$8
  local fn_node=$9 fn_expected_controller_sha=${10} fn_pre_snapshot=${11}
  local fn_controller_source fn_cache fn_gate fn_gate_json fn_result fn_post_snapshot
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${aggregate_job}" && "${fn_node}" == "${aggregate_node}" ]] || \
    fail "postflight fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-postflight-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "postflight allocation binding differs"
  [[ "${fn_run_root}" == /* && -d "${fn_run_root}" && ! -L "${fn_run_root}" \
      && "${fn_run_root}" == "$(readlink -f -- "${fn_run_root}")" \
      && "$(stat -c %a "${fn_run_root}")" == 700 \
      && ! -e "${fn_run_root}/seal.json" && ! -L "${fn_run_root}/seal.json" ]] || \
    fail "postflight run root is not fresh for seal"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "postflight initial authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "postflight initial authority differs"
  fn_cache="$(fresh_cache "${tag}-postflight-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  fn_gate="$(cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    postflight v4e-vjepa2-exact1-cpu-postflight-gate-v1 "${aggregate_cpus}")"
  [[ "${fn_gate}" == V4E_CPU_STEP_GATE=* ]] || fail "postflight CPU gate output differs"
  fn_gate_json=${fn_gate#V4E_CPU_STEP_GATE=}
  fn_result="$("${fn_python_bin}" -I -S -B - \
    "${fn_run_root}" "${fn_release_root}" "${fn_python_bin}" "${fn_feature_root}" \
    "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" "${fn_v4d_receipt}" \
    "${fn_controller_source}" "${fn_expected_controller_sha}" "${fn_pre_snapshot}" \
    "${fn_gate_json}" "${expected_release_tree_sha256}" "${expected_runtime_sha256}" \
    "${expected_runtime_test_sha256}" "${expected_python_sha256}" \
    "${expected_feature_receipt_sha256}" "${expected_v4a_receipt_sha256}" \
    "${expected_v4a_receipt_digest}" "${expected_v4c_frontier_receipt_sha256}" \
    "${expected_v4c_frontier_receipt_digest}" "${expected_v4d_receipt_sha256}" \
    "${expected_v4d_receipt_digest}" "${expected_fold_schema}" \
    "${expected_aggregate_schema}" "${expected_aggregate_status}" \
    "${expected_checkpoint_schema}" "${expected_feature_authority_sha256}" \
    "${expected_v2_runtime_sha256}" "${expected_v4a_runtime_sha256}" \
    "${expected_extractor_sha256}" "${expected_v4c_runtime_sha256}" \
    "${expected_v4d_runtime_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, math, os, re, stat, sys

(
    root_text, release_text, python_text, feature_text, v4a_text, v4c_text,
    v4d_text, controller_text, controller_sha, authority_pre, cpu_gate_text,
    release_tree_sha, runtime_sha, runtime_test_sha, python_sha,
    feature_receipt_sha, v4a_receipt_sha, v4a_receipt_digest,
    v4c_receipt_sha, v4c_receipt_digest, v4d_receipt_sha,
    v4d_receipt_digest, fold_schema, aggregate_schema, expected_status,
    checkpoint_schema, feature_authority_sha, v2_runtime_sha, v4a_runtime_sha,
    extractor_sha, v4c_runtime_sha, v4d_runtime_sha,
) = sys.argv[1:]

root = Path(root_text)
def canonical_input(text, label):
    path = Path(text)
    if (not path.is_absolute() or path.is_symlink()
            or str(path) != str(path.resolve(strict=True))):
        raise SystemExit(label + " path is not absolute/plain/canonical")
    return path
release = canonical_input(release_text, "release")
python_bin = canonical_input(python_text, "python")
feature_root = canonical_input(feature_text, "feature root")
v4a_receipt_path = canonical_input(v4a_text, "v4a receipt")
v4c_receipt_path = canonical_input(v4c_text, "v4c receipt")
v4d_receipt_path = canonical_input(v4d_text, "v4d receipt")
controller = canonical_input(controller_text, "controller")
sha_re = re.compile(r"[0-9a-f]{64}")
views = ("original", "monotone_warp", "reverse", "block_shuffle", "phase_swap")
negatives = ("reverse", "block_shuffle", "phase_swap")
oof_counts = (131, 127, 128, 129, 129)
checkpoint_steps = {0, 300, 600, 900, 1200}
workers = [
    {"fold": 0, "job": 143808, "node": "auh7-1b-gpu-315"},
    {"fold": 1, "job": 143812, "node": "auh7-1b-gpu-293"},
    {"fold": 2, "job": 143811, "node": "auh7-1b-gpu-306"},
    {"fold": 3, "job": 143808, "node": "auh7-1b-gpu-233"},
    {"fold": 4, "job": 143808, "node": "auh7-1b-gpu-268"},
]
worker_request = {
    "nodes": 1, "ntasks": 1, "cpus": 8, "memory": "12G",
    "gres": "gpu:mi210:1", "overlap": True, "exact": True,
}
preflight_request = {
    "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
    "ntasks": 1, "cpus": 4, "memory": "4G", "gres": "none",
    "overlap": True, "exact": True,
}
aggregate_request = {
    "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
    "ntasks": 1, "cpus": 8, "memory": "12G", "gres": "none",
    "overlap": True, "exact": True,
}
release_expected = [
    ("methods/bernini_action_editing/semantic_action_cvae_canary_v1.py", feature_authority_sha),
    ("methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py", v2_runtime_sha),
    ("methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py", v4a_runtime_sha),
    ("methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py", extractor_sha),
    ("methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py", v4c_runtime_sha),
    ("methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py", v4d_runtime_sha),
    ("methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py", runtime_sha),
    ("methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py", runtime_test_sha),
]
implementation = {
    "implementation_path": str((release / release_expected[6][0]).resolve(strict=True)),
    "implementation_sha256": runtime_sha,
    "v4c_implementation_path": str((release / release_expected[4][0]).resolve(strict=True)),
    "v4c_implementation_sha256": v4c_runtime_sha,
    "extractor_implementation_path": str((release / release_expected[3][0]).resolve(strict=True)),
    "extractor_implementation_sha256": extractor_sha,
    "v4a_implementation_path": str((release / release_expected[2][0]).resolve(strict=True)),
    "v4a_implementation_sha256": v4a_runtime_sha,
    "v4d_implementation_path": str((release / release_expected[5][0]).resolve(strict=True)),
    "v4d_implementation_sha256": v4d_runtime_sha,
}

def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mode,
            value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)

def read_sealed(path, label, *, mode=0o444, capture=False):
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or not hasattr(os, "O_NOFOLLOW")
            or str(path) != str(path.resolve(strict=True))):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != mode
            or before.st_nlink != 1):
        raise SystemExit(label + " pre-open seal differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    chunks = [] if capture else None
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " single-FD identity differs")
    raw = b"".join(chunks) if chunks is not None else None
    return raw, before, digest.hexdigest()

def reject_pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result

def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: " + value)

def parse_json(raw, label):
    try:
        value = json.loads(raw, object_pairs_hook=reject_pairs,
                           parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(label + " strict JSON parse differs") from error
    if type(value) is not dict:
        raise SystemExit(label + " JSON root differs")
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def self_digest(value, label, key="receipt_digest"):
    unsigned = dict(value)
    digest = unsigned.pop(key, None)
    if sha_re.fullmatch(str(digest)) is None or object_sha(unsigned) != digest:
        raise SystemExit(label + " self-digest differs")
    return digest

def inspect_tree(expected_files, expected_dirs, root_mode):
    if (not root.is_absolute() or root.is_symlink()
            or str(root) != str(root.resolve(strict=True))
            or stat.S_IMODE(root.lstat().st_mode) != root_mode):
        raise SystemExit("run root envelope differs")
    actual_files = set(); actual_dirs = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix(); info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit("run tree symlink: " + relative)
        if stat.S_ISDIR(info.st_mode):
            actual_dirs.add(relative)
        elif stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
            if stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
                raise SystemExit("run file seal differs: " + relative)
        else:
            raise SystemExit("run special member: " + relative)
    if actual_files != expected_files or actual_dirs != set(expected_dirs):
        raise SystemExit("run exact-tree membership differs")
    for relative, mode in expected_dirs.items():
        if stat.S_IMODE((root / relative).lstat().st_mode) != mode:
            raise SystemExit("run directory mode differs: " + relative)

preseal_files = {"launch-plan.json", "aggregate/receipt.json",
                 "logs/aggregate.stdout", "logs/aggregate.stderr"}
for index in range(5):
    preseal_files.update({
        f"fold{index}/fold.json", f"fold{index}/selected.pt",
        f"logs/fold{index}.stdout", f"logs/fold{index}.stderr",
    })
preseal_dirs = {"logs": 0o700, "aggregate": 0o700,
                **{f"fold{index}": 0o555 for index in range(5)}}
if len(preseal_files) != 24:
    raise SystemExit("preseal exact24 declaration differs")
inspect_tree(preseal_files, preseal_dirs, 0o700)

raw_by_path = {}; binding_by_path = {}
for relative in sorted(preseal_files):
    capture = relative.endswith(".json") or relative.endswith(".stdout")
    raw, info, digest = read_sealed(root / relative, relative, capture=capture)
    if raw is not None:
        raw_by_path[relative] = raw
    binding_by_path[relative] = {
        "path": relative, "sha256": digest, "size_bytes": info.st_size,
        "mode_octal": "0444", "nlink": info.st_nlink,
        "physical_identity": {"device": info.st_dev, "inode": info.st_ino,
                              "size_bytes": info.st_size},
    }

launch = parse_json(raw_by_path["launch-plan.json"], "launch plan")
preflight = launch.get("cpu_preflight")
preflight_gate = preflight.get("cpu_gate") if type(preflight) is dict else None
preflight_without_gate = dict(preflight) if type(preflight) is dict else {}
preflight_without_gate.pop("cpu_gate", None)
if (
    launch.get("schema_version") != "v4e-exact5-parallel-launch-plan-v1"
    or launch.get("controller_sha256") != controller_sha
    or launch.get("authority_snapshot") != authority_pre
    or launch.get("fixed_no_fallback") is not True
    or launch.get("workers") != workers
    or launch.get("worker_request") != worker_request
    or launch.get("preflight_request") != preflight_request
    or launch.get("aggregate_request") != aggregate_request
    or launch.get("postflight_request") != aggregate_request
    or launch.get("source_authority") != {
        "release_tree_sha256": release_tree_sha,
        "runtime_sha256": runtime_sha,
        "runtime_test_sha256": runtime_test_sha,
        "python_sha256": python_sha,
    }
    or launch.get("receipt_contract") != {
        "fold_schema": fold_schema, "aggregate_schema": aggregate_schema,
        "status": expected_status, "fold_exact_files": ["fold.json", "selected.pt"],
    }
    or type(preflight) is not dict
    or preflight_without_gate != {
        "job": 143811, "node": "auh7-1b-gpu-306", "cpus": 4,
        "memory": "4G", "gres": "none", "normal_tests_passed": True,
        "normal_tests_run": 28, "normal_tests_skipped": 0,
        "optimized_tests_passed": True, "optimized_tests_run": 28,
        "optimized_tests_skipped": 0, "tests_ran_in_parallel": True,
        "omp_threads_per_test_process": 1, "compile_passed": True,
        "cli_help_passed": True, "fresh_cache_cleaned": True,
    }
    or type(preflight_gate) is not dict
    or preflight_gate.get("schema_version") != "v4e-vjepa2-exact1-cpu-preflight-gate-v1"
    or preflight_gate.get("role") != "preflight"
    or preflight_gate.get("job_id") != "143811"
    or preflight_gate.get("node") != "auh7-1b-gpu-306"
    or preflight_gate.get("ntasks") != 1 or preflight_gate.get("nnodes") != 1
    or preflight_gate.get("cpus_per_task") != 4 or preflight_gate.get("gres") != "none"
    or preflight_gate.get("gpu_visibility_variables_absent") is not True
    or preflight_gate.get("fresh_cache_pre_torch_exact9_empty_directories") is not True
    or re.fullmatch(r"[0-9]+", str(preflight_gate.get("step_id"))) is None
    or Path(str(preflight_gate.get("fresh_cache_root"))).exists()
    or Path(str(preflight_gate.get("fresh_cache_root"))).is_symlink()
):
    raise SystemExit("launch plan semantic closure differs")

def parse_three_line_log(relative, first_prefix, third_prefix):
    try:
        lines = raw_by_path[relative].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise SystemExit(relative + " is not ASCII") from error
    if len(lines) != 3 or not lines[0].startswith(first_prefix) or not lines[2].startswith(third_prefix):
        raise SystemExit(relative + " exact three-line closure differs")
    first = parse_json(lines[0][len(first_prefix):].encode("ascii"), relative + " gate")
    middle = parse_json(lines[1].encode("ascii"), relative + " result")
    third = parse_json(lines[2][len(third_prefix):].encode("ascii"), relative + " cleanup")
    return first, middle, third

def parse_fold_four_line_log(relative):
    try:
        lines = raw_by_path[relative].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise SystemExit(relative + " is not ASCII") from error
    if (len(lines) != 4 or not lines[0].startswith("V4E_GPU_GATE=")
            or not lines[3].startswith("V4E_CACHE_CLEANED=")):
        raise SystemExit(relative + " exact four-line closure differs")
    return (
        parse_json(lines[0][len("V4E_GPU_GATE="):].encode("ascii"), relative + " gate"),
        parse_json(lines[1].encode("ascii"), relative + " runtime result"),
        parse_json(lines[2].encode("ascii"), relative + " controller verifier"),
        parse_json(lines[3][len("V4E_CACHE_CLEANED="):].encode("ascii"),
                   relative + " cleanup"),
    )

false_fold_scope = {
    "latent_metric_qualified", "action_representation_qualified",
    "identity_disentanglement_qualified", "identity_preservation_qualified",
    "prior_qualified", "prior_generation_qualified", "generation_qualified",
    "renderer_qualified", "video_editing_qualified", "inference_authorized",
    "web_evaluation_authorized", "full644_refit_authorized",
}
fold_receipts = []; fold_artifacts = []; fold_receipt_bindings = []
fold_evidence = []; selected_steps = []
for index, worker in enumerate(workers):
    receipt_relative = f"fold{index}/fold.json"
    checkpoint_relative = f"fold{index}/selected.pt"
    receipt = parse_json(raw_by_path[receipt_relative], f"fold{index} receipt")
    digest = self_digest(receipt, f"fold{index} receipt")
    fold = receipt.get("fold"); training = fold.get("training") if type(fold) is dict else None
    artifact = fold.get("selected_checkpoint_artifact") if type(fold) is dict else None
    fit_iids = fold.get("model_fit_ordered_iids") if type(fold) is dict else None
    inner_iids = training.get("inner_validation_ordered_iids") if type(training) is dict else None
    oof_iids = fold.get("oof_ordered_iids") if type(fold) is dict else None
    evidence = receipt.get("oof_evidence")
    scope = receipt.get("qualification_scope")
    checkpoint_binding = binding_by_path[checkpoint_relative]
    if (
        receipt.get("schema_version") != fold_schema
        or receipt.get("status") != expected_status
        or receipt.get("authority") != "burned_exposed_known_transform_development_fold_only"
        or receipt.get("implementation") != implementation
        or receipt.get("fold_root") != str((root / f"fold{index}").resolve(strict=True))
        or receipt.get("feature_authority", {}).get("feature_root") != str(feature_root)
        or receipt.get("feature_authority", {}).get("feature_receipt_sha256") != feature_receipt_sha
        or receipt.get("upstream_authorities", {}).get("v4a_receipt_path") != str(v4a_receipt_path)
        or receipt.get("upstream_authorities", {}).get("v4a_receipt_sha256") != v4a_receipt_sha
        or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_path") != str(v4c_receipt_path)
        or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_sha256") != v4c_receipt_sha
        or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_path") != str(v4d_receipt_path)
        or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_sha256") != v4d_receipt_sha
        or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_digest") != v4d_receipt_digest
        or type(fold) is not dict or fold.get("fold_index") != index
        or type(training) is not dict or training.get("selected_step") not in checkpoint_steps
        or type(fit_iids) is not list or len(set(fit_iids)) != len(fit_iids)
        or len(fit_iids) != fold.get("model_fit_original_count")
        or object_sha(fit_iids) != fold.get("model_fit_iid_digest")
        or type(inner_iids) is not list or len(set(inner_iids)) != len(inner_iids)
        or len(inner_iids) != fold.get("inner_validation_original_count")
        or object_sha(inner_iids) != fold.get("inner_validation_iid_digest")
        or type(oof_iids) is not list or len(set(oof_iids)) != len(oof_iids)
        or len(oof_iids) != oof_counts[index]
        or len(oof_iids) != fold.get("oof_original_count")
        or object_sha(oof_iids) != fold.get("oof_iid_digest")
        or len(set(fit_iids + inner_iids + oof_iids)) != 644
        or type(evidence) is not list or len(evidence) != oof_counts[index]
        or receipt.get("oof_evidence_count") != len(evidence)
        or receipt.get("oof_evidence_sha256") != object_sha(evidence)
        or [row.get("iid") for row in evidence] != oof_iids
        or any(row.get("outer_fold") != index for row in evidence)
        or type(artifact) is not dict
        or artifact.get("path") != str((root / checkpoint_relative).resolve(strict=True))
        or artifact.get("file_sha256") != checkpoint_binding["sha256"]
        or artifact.get("size_bytes") != checkpoint_binding["size_bytes"]
        or artifact.get("mode_octal") != "0444" or artifact.get("nlink") != 1
        or artifact.get("physical_identity") != checkpoint_binding["physical_identity"]
        or artifact.get("outer_fold") != index
        or artifact.get("selected_step") != training.get("selected_step")
        or artifact.get("model_state_sha256") != training.get("selected_state_sha256")
        or artifact.get("implementation_sha256") != runtime_sha
        or artifact.get("single_fd_pre_post_sha256_exact") is not True
        or artifact.get("semantic_metadata_state_replay_verified") is not True
        or artifact.get("basis_metadata_state_hash_join_verified") is not True
        or artifact.get("model_schema_reconstructed_and_strict_loaded") is not True
        or artifact.get("model_forward_executed_by_loader") is not False
        or artifact.get("fresh_reload_strict_state_verified") is not True
        or artifact.get("fresh_reload_output_bit_exact") is not True
        or artifact.get("caller_model_reloaded_from_sealed_artifact_before_oof") is not True
        or sha_re.fullmatch(str(artifact.get("metadata_digest"))) is None
        or type(scope) is not dict
        or scope.get("exposed_five_view_codec_development_gate") is not None
        or scope.get("aggregate_gate_evaluated") is not False
        or scope.get("vae_necessary") is not None
        or any(scope.get(key) is not False for key in false_fold_scope)
    ):
        raise SystemExit(f"fold{index} receipt/checkpoint semantic closure differs")
    gate, result, verifier, cleanup = parse_fold_four_line_log(
        f"logs/fold{index}.stdout"
    )
    if (
        gate.get("schema_version") != "v4e-vjepa2-exact1-fold-gpu-gate-v1"
        or gate.get("fold_index") != index or gate.get("job_id") != str(worker["job"])
        or gate.get("node") != worker["node"] or gate.get("ntasks") != 1
        or gate.get("nnodes") != 1 or gate.get("cpus_per_task") != 8
        or gate.get("logical_device_count") != 1
        or gate.get("device_name") != "AMD Instinct MI210"
        or gate.get("torch") != "2.7.1+rocm6.3"
        or re.fullmatch(r"[0-9]+", str(gate.get("step_id"))) is None
        or re.fullmatch(r"[0-9a-f]{16}", str(gate.get("decoded_logical_unique_id"))) is None
        or gate.get("fresh_cache_pre_torch_exact9_empty_directories") is not True
        or result.get("fold") != index
        or result.get("fold_receipt") != str((root / receipt_relative).resolve(strict=True))
        or result.get("fold_receipt_sha256") != binding_by_path[receipt_relative]["sha256"]
        or result.get("fold_receipt_digest") != digest
        or result.get("selected_checkpoint") != str((root / checkpoint_relative).resolve(strict=True))
        or result.get("selected_step") != training.get("selected_step")
        or verifier.get("fold") != index
        or verifier.get("receipt_sha256") != binding_by_path[receipt_relative]["sha256"]
        or verifier.get("receipt_digest") != digest
        or verifier.get("receipt_size_bytes") != binding_by_path[receipt_relative]["size_bytes"]
        or verifier.get("selected_sha256") != checkpoint_binding["sha256"]
        or verifier.get("selected_size_bytes") != checkpoint_binding["size_bytes"]
        or verifier.get("selected_step") != training.get("selected_step")
        or cleanup.get("fold_index") != index
        or cleanup.get("fresh_cache_root") != gate.get("fresh_cache_root")
        or cleanup.get("absent_after_child_cleanup") is not True
    ):
        raise SystemExit(f"fold{index} stdout/GPU/cache join differs")
    fold_receipts.append(receipt); fold_artifacts.append(artifact)
    fold_evidence.extend(evidence); selected_steps.append(training["selected_step"])
    receipt_info = binding_by_path[receipt_relative]
    fold_receipt_bindings.append({
        "fold_root": str((root / f"fold{index}").resolve(strict=True)),
        "path": str((root / receipt_relative).resolve(strict=True)),
        "file_sha256": receipt_info["sha256"], "receipt_digest": digest,
        "size_bytes": receipt_info["size_bytes"], "mode_octal": "0444",
        "nlink": 1, "physical_identity": receipt_info["physical_identity"],
        "single_fd_pre_post_bytes_and_identity_exact": True,
    })

aggregate_relative = "aggregate/receipt.json"
aggregate = parse_json(raw_by_path[aggregate_relative], "aggregate receipt")
aggregate_digest = self_digest(aggregate, "aggregate receipt")
metrics = aggregate.get("metrics"); scope = aggregate.get("qualification_scope")
runtime = aggregate.get("runtime"); model = aggregate.get("model_contract")
training_contract = aggregate.get("training_contract")
evaluation = aggregate.get("evaluation_contract")
fixed = aggregate.get("fixed_comparator_authority")
oof = aggregate.get("oof_closure")
fold_receipt_manifest = aggregate.get("fold_receipt_artifacts")
checkpoint_manifest = aggregate.get("selected_fold_checkpoint_artifacts")
false_aggregate_scope = {
    "unseen_hostile_transform_gate", "unseen_hostile_transform_gate_evaluated",
    "latent_metric_qualified", "action_representation_qualified",
    "identity_disentanglement_qualified", "identity_preservation_qualified",
    "generation_qualified", "prior_qualified", "prior_generation_qualified",
    "renderer_qualified", "video_editing_qualified", "inference_authorized",
    "web_evaluation_authorized", "full644_refit_authorized",
    "video_model_training_performed",
}
if (
    aggregate.get("schema_version") != aggregate_schema
    or aggregate.get("status") != expected_status
    or aggregate.get("authority") != "burned_exposed_known_transform_development_only"
    or aggregate.get("implementation") != implementation
    or type(runtime) is not dict or runtime.get("torch") != "2.7.1+rocm6.3"
    or runtime.get("aggregate_device") != "cpu"
    or runtime.get("model_schema_reconstructed_and_strict_loaded") is not True
    or runtime.get("model_forward_executed") is not False
    or runtime.get("model_trained_or_recomputed") is not False
    or aggregate.get("feature_authority", {}).get("feature_root") != str(feature_root)
    or aggregate.get("feature_authority", {}).get("feature_receipt_sha256") != feature_receipt_sha
    or aggregate.get("feature_authority", {}).get("unique_original_iids") != 644
    or aggregate.get("feature_authority", {}).get("family_count") != 28
    or aggregate.get("feature_authority", {}).get("stored_views") != list(views)
    or aggregate.get("upstream_authorities", {}).get("v4a_receipt_path") != str(v4a_receipt_path)
    or aggregate.get("upstream_authorities", {}).get("v4a_receipt_file_sha256") != v4a_receipt_sha
    or aggregate.get("upstream_authorities", {}).get("v4a_receipt_self_digest") != v4a_receipt_digest
    or aggregate.get("upstream_authorities", {}).get("v4c_frontier_receipt_path") != str(v4c_receipt_path)
    or aggregate.get("upstream_authorities", {}).get("v4c_frontier_receipt_file_sha256") != v4c_receipt_sha
    or aggregate.get("upstream_authorities", {}).get("v4c_frontier_receipt_self_digest") != v4c_receipt_digest
    or aggregate.get("upstream_authorities", {}).get("v4d_burned_receipt_path") != str(v4d_receipt_path)
    or aggregate.get("upstream_authorities", {}).get("v4d_burned_receipt_file_sha256") != v4d_receipt_sha
    or aggregate.get("upstream_authorities", {}).get("v4d_burned_receipt_self_digest") != v4d_receipt_digest
    or type(fixed) is not dict or fixed.get("fixed_comparator_name") != "clip_pca_b0384_t01_r384"
    or fixed.get("v4c_burned_oof_informed_clip_pca_b384_choice") is not True
    or fixed.get("v4e_oof_used_to_select_comparator") is not False
    or fixed.get("single_v4e_candidate") is not True
    or fixed.get("fold_basis_fit_model_fit_original_only") is not True
    or fixed.get("inner_validation_or_oof_used_for_basis_fit") is not False
    or fixed.get("same_payload_384_scalars_only") is not True
    or fixed.get("called_best_or_winner") is not False
    or type(model) is not dict or model.get("code_shape") != [12, 32]
    or model.get("actual_code_numel") != 384
    or model.get("decoder_input") != "sole [12,32] code"
    or model.get("raw_input_skip_or_side_channel") is not False
    or model.get("exact_trainable_parameter_count") != 79040
    or model.get("trainable_parameter_limit_exclusive") != 150000
    or model.get("latent_scale_or_rotation_gauge_fixed") is not False
    or type(training_contract) is not dict
    or training_contract.get("all_five_known_views_exposed_for_each_model_fit_iid") is not True
    or training_contract.get("all_five_view_reconstruction_terms_equal_weight") is not True
    or training_contract.get("geometry_all_ten_unordered_pairs") is not True
    or training_contract.get("view_axis_permutation_invariant") is not True
    or training_contract.get("view_name_positive_negative_role_family_action_or_strict_labels_enter_loss_or_model") is not False
    or training_contract.get("fixed_full_budget_no_early_stop") is not True
    or training_contract.get("inner_checkpoint_selection_original_raw_mse_only") is not True
    or training_contract.get("oof_selection") is not False
    or training_contract.get("selected_steps_by_fold") != selected_steps
    or type(evaluation) is not dict
    or evaluation.get("known_exposed_transform_families_only") is not True
    or evaluation.get("unseen_hostile_transform_gate_evaluated") is not False
    or evaluation.get("views") != list(views)
    or evaluation.get("fixed_comparator") != "clip_pca_b0384_t01_r384"
    or aggregate.get("folds") != [receipt["fold"] for receipt in fold_receipts]
    or type(fold_receipt_manifest) is not dict
    or fold_receipt_manifest.get("count") != 5
    or fold_receipt_manifest.get("bindings") != fold_receipt_bindings
    or fold_receipt_manifest.get("all_single_fd_mode0444_nlink1") is not True
    or type(checkpoint_manifest) is not dict or checkpoint_manifest.get("count") != 5
    or checkpoint_manifest.get("artifacts") != fold_artifacts
    or checkpoint_manifest.get("artifacts_manifest_sha256") != object_sha(fold_artifacts)
    or checkpoint_manifest.get("all_reverified_by_cpu_aggregate") is not True
    or type(oof) is not dict or oof.get("unique_original_iids") != 644
    or oof.get("each_original_evaluated_exactly_once") is not True
    or oof.get("oof_counts_by_fold") != list(oof_counts)
    or oof.get("embedded_per_iid_evidence_count") != 644
    or oof.get("embedded_per_iid_evidence") != fold_evidence
    or oof.get("embedded_per_iid_evidence_sha256") != object_sha(fold_evidence)
    or len({row.get("iid") for row in fold_evidence}) != 644
    or type(scope) is not dict or type(scope.get("exposed_five_view_codec_development_gate")) is not bool
    or scope.get("vae_necessary") is not None
    or any(scope.get(key) is not False for key in false_aggregate_scope)
):
    raise SystemExit("aggregate receipt scientific/authority/artifact closure differs")

if type(metrics) is not dict:
    raise SystemExit("aggregate metrics root differs")
fidelity = metrics.get("five_view_raw_reconstruction_ratio_vs_fixed_clip_pca_b384")
negative_results = metrics.get("negative_results")
if type(fidelity) is not dict or set(fidelity) != set(views):
    raise SystemExit("five-view fidelity metrics closure differs")
fidelity_gate = True
for view in views:
    row = fidelity[view]
    per_fold = row.get("per_fold_ratio_of_mean_raw_mses") if type(row) is dict else None
    if (type(row) is not dict or type(per_fold) is not dict
            or set(per_fold) != {"0", "1", "2", "3", "4"}
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in per_fold.values())
            or type(row.get("both_ucbs_le_1p05")) is not bool
            or type(row.get("all_five_fold_point_ratios_le_1p05")) is not bool):
        raise SystemExit("fidelity per-view gate structure differs")
    fidelity_gate = fidelity_gate and row["both_ucbs_le_1p05"] \
        and row["all_five_fold_point_ratios_le_1p05"]
if type(negative_results) is not dict or set(negative_results) != set(negatives):
    raise SystemExit("decoded negative metrics closure differs")
all_negative = True
for negative in negatives:
    row = negative_results[negative]
    component_keys = (
        "teacher_margin", "candidate_margin", "candidate_minus_0p8_teacher_margin",
        "candidate_minus_fixed_clip_pca_b384_margin",
    )
    component_gate = True
    for key in component_keys:
        component = row.get(key) if type(row) is dict else None
        if (type(component) is not dict
                or type(component.get("both_lcbs_strictly_gt_zero")) is not bool
                or type(component.get("all_five_fold_point_means_strictly_gt_zero")) is not bool):
            raise SystemExit("negative component gate structure differs")
        component_gate = component_gate and component["both_lcbs_strictly_gt_zero"] \
            and component["all_five_fold_point_means_strictly_gt_zero"]
    if (row.get("all_four_quantities_pass_dual_bootstrap_and_every_fold") is not component_gate
            or row.get("decoded_negative_gate") is not component_gate):
        raise SystemExit("decoded negative derived gate differs")
    all_negative = all_negative and component_gate
development_gate = bool(fidelity_gate and all_negative)
if (
    metrics.get("five_view_fidelity_gate") is not fidelity_gate
    or metrics.get("all_three_decoded_negative_gates") is not all_negative
    or metrics.get("exposed_five_view_codec_development_gate") is not development_gate
    or metrics.get("latent_metric_qualified") is not False
    or metrics.get("latent_gauge_fixed") is not False
    or metrics.get("frozen_oof_counts_by_fold") != list(oof_counts)
    or scope.get("exposed_five_view_codec_development_gate") is not development_gate
):
    raise SystemExit("aggregate derived development gate differs")

aggregate_gate, aggregate_result, aggregate_cleanup = parse_three_line_log(
    "logs/aggregate.stdout", "V4E_CPU_STEP_GATE=", "V4E_CACHE_CLEANED="
)
aggregate_binding = binding_by_path[aggregate_relative]
if (
    aggregate_gate.get("schema_version") != "v4e-vjepa2-exact1-cpu-aggregate-gate-v1"
    or aggregate_gate.get("role") != "aggregate"
    or aggregate_gate.get("job_id") != "143811"
    or aggregate_gate.get("node") != "auh7-1b-gpu-306"
    or aggregate_gate.get("ntasks") != 1 or aggregate_gate.get("nnodes") != 1
    or aggregate_gate.get("cpus_per_task") != 8 or aggregate_gate.get("gres") != "none"
    or re.fullmatch(r"[0-9]+", str(aggregate_gate.get("step_id"))) is None
    or aggregate_gate.get("gpu_visibility_variables_absent") is not True
    or aggregate_gate.get("fresh_cache_pre_torch_exact9_empty_directories") is not True
    or aggregate_result.get("receipt") != str((root / aggregate_relative).resolve(strict=True))
    or aggregate_result.get("receipt_sha256") != aggregate_binding["sha256"]
    or aggregate_result.get("receipt_digest") != aggregate_digest
    or aggregate_result.get("exposed_five_view_codec_development_gate") is not development_gate
    or aggregate_result.get("inference_authorized") is not False
    or aggregate_cleanup.get("role") != "aggregate"
    or aggregate_cleanup.get("fresh_cache_root") != aggregate_gate.get("fresh_cache_root")
    or aggregate_cleanup.get("absent_after_child_cleanup") is not True
    or Path(str(aggregate_gate.get("fresh_cache_root"))).exists()
    or Path(str(aggregate_gate.get("fresh_cache_root"))).is_symlink()
):
    raise SystemExit("aggregate stdout/CPU/cache/receipt join differs")

cpu_gate = parse_json(cpu_gate_text.encode("ascii"), "postflight CPU gate")
if (
    cpu_gate.get("schema_version") != "v4e-vjepa2-exact1-cpu-postflight-gate-v1"
    or cpu_gate.get("role") != "postflight" or cpu_gate.get("job_id") != "143811"
    or cpu_gate.get("node") != "auh7-1b-gpu-306"
    or cpu_gate.get("ntasks") != 1 or cpu_gate.get("nnodes") != 1
    or cpu_gate.get("cpus_per_task") != 8 or cpu_gate.get("gres") != "none"
    or re.fullmatch(r"[0-9]+", str(cpu_gate.get("step_id"))) is None
    or cpu_gate.get("gpu_visibility_variables_absent") is not True
    or cpu_gate.get("fresh_cache_pre_torch_exact9_empty_directories") is not True
):
    raise SystemExit("postflight exact CPU gate differs")

def authority_snapshot():
    rebound = []
    if (release.is_symlink() or stat.S_IMODE(release.lstat().st_mode) != 0o555):
        raise SystemExit("release root seal differs during final authority check")
    members = list(release.rglob("*"))
    expected_paths = {relative for relative, _ in release_expected}
    actual_paths = {path.relative_to(release).as_posix() for path in members if path.is_file()}
    actual_dirs = {path.relative_to(release).as_posix() for path in members if path.is_dir()}
    if (actual_paths != expected_paths
            or actual_dirs != {"methods", "methods/bernini_action_editing",
                               "methods/bernini_action_editing/tests"}
            or any(path.is_symlink() for path in members)
            or any(not (stat.S_ISREG(path.lstat().st_mode)
                        or stat.S_ISDIR(path.lstat().st_mode)) for path in members)
            or any(stat.S_IMODE(path.lstat().st_mode) != 0o555
                   for path in members if path.is_dir())):
        raise SystemExit("release exact8 tree differs during final authority check")
    release_rows = []
    for relative, expected_sha in release_expected:
        _, info, digest = read_sealed(release / relative, "release " + relative)
        if digest != expected_sha:
            raise SystemExit("release SHA differs: " + relative)
        release_rows.append({"path": relative, "sha256": digest,
                             "size_bytes": info.st_size})
        rebound.append((release / relative, identity(info)))
    computed_release = object_sha(sorted(release_rows, key=lambda row: row["path"]))
    if computed_release != release_tree_sha:
        raise SystemExit("release tree digest differs during final authority check")
    _, python_info, actual_python_sha = read_sealed(python_bin, "pinned python", mode=0o755)
    _, controller_info, actual_controller_sha = read_sealed(controller, "detached controller", mode=0o555)
    rebound.extend(((python_bin, identity(python_info)), (controller, identity(controller_info))))
    if actual_python_sha != python_sha or actual_controller_sha != controller_sha:
        raise SystemExit("python/controller authority differs")
    authorities = [
        (feature_root / "feature_extraction_receipt.json", feature_receipt_sha, "feature"),
        (v4a_receipt_path, v4a_receipt_sha, "v4a"),
        (v4c_receipt_path, v4c_receipt_sha, "v4c"),
        (v4d_receipt_path, v4d_receipt_sha, "v4d"),
    ]
    rows = []; feature_receipt = None
    for path, expected_sha, label in authorities:
        raw, info, digest = read_sealed(path, label, capture=label == "feature")
        if digest != expected_sha:
            raise SystemExit(label + " receipt SHA differs")
        rows.append({"label": label, "path": str(path), "sha256": digest,
                     "size_bytes": info.st_size})
        rebound.append((path, identity(info)))
        if label == "feature":
            feature_receipt = parse_json(raw, "feature receipt")
    shards = feature_receipt.get("shards") if type(feature_receipt) is dict else None
    if type(shards) is not list or len(shards) != 6:
        raise SystemExit("feature exact6 shard authority differs")
    for index, shard in enumerate(shards):
        if type(shard) is not dict or shard.get("index") != index:
            raise SystemExit("feature shard order differs")
        path = Path(str(shard.get("path")))
        raw, info, digest = read_sealed(path, "feature-shard-" + str(index))
        if digest != shard.get("sha256"):
            raise SystemExit("feature shard SHA differs")
        rows.append({"label": "feature-shard-" + str(index), "path": str(path),
                     "sha256": digest, "size_bytes": info.st_size})
        rebound.append((path, identity(info)))
    if any(path.is_symlink() or identity(path.lstat()) != expected
           for path, expected in rebound):
        raise SystemExit("final authority cross-file identity rebound differs")
    input_digest = object_sha(rows)
    return f"{computed_release}:{input_digest}:{controller_sha}:{python_sha}"

# This is the last authority operation before create-only seal publication.
# Everything after it touches the run tree only through the seal write and
# read-only exact25 verification.
authority_final = authority_snapshot()
if authority_final != authority_pre:
    raise SystemExit("final pre-seal authority snapshot differs")

artifact_rows = [binding_by_path[path] for path in sorted(preseal_files)]
seal = {
    "schema_version": "v4e-vjepa2-exposed-five-view-exact5-parallel-run-seal-v1",
    "controller_sha256": controller_sha,
    "authority_snapshot_pre_final_preseal": authority_final,
    "release_tree_sha256": release_tree_sha,
    "runtime_sha256": runtime_sha, "runtime_test_sha256": runtime_test_sha,
    "python_sha256": python_sha,
    "fixed_no_fallback_workers": workers,
    "worker_request": worker_request,
    "preflight_request": preflight_request,
    "aggregate_request": aggregate_request,
    "postflight_request": aggregate_request,
    "cpu_preflight": preflight,
    "cpu_aggregate_gate": aggregate_gate,
    "cpu_postflight_gate": cpu_gate,
    "fold_receipt_sha256_by_fold": {
        str(index): binding_by_path[f"fold{index}/fold.json"]["sha256"]
        for index in range(5)
    },
    "selected_checkpoint_sha256_by_fold": {
        str(index): binding_by_path[f"fold{index}/selected.pt"]["sha256"]
        for index in range(5)
    },
    "selected_steps_by_fold": selected_steps,
    "aggregate_receipt_sha256": aggregate_binding["sha256"],
    "aggregate_receipt_digest": aggregate_digest,
    "exposed_five_view_codec_development_gate": development_gate,
    "scientific_scope": {
        "known_exposed_five_view_burned_development_only": True,
        "unseen_hostile_transform_qualified": False,
        "latent_metric_qualified": False,
        "action_representation_qualified": False,
        "identity_disentanglement_qualified": False,
        "identity_preservation_qualified": False,
        "prior_or_generation_qualified": False,
        "renderer_or_video_editing_qualified": False,
        "inference_authorized": False,
        "full644_refit_authorized": False,
        "html_or_video_generated": False,
    },
    "checkpoint_evidence": {
        "semantic_metadata_state_replay_performed_by_cpu_aggregate": True,
        "controller_single_fd_raw_sha_identity_join_verified_on_compute": True,
        "checkpoint_schema": checkpoint_schema,
        "login_node_checkpoint_load_or_hash_performed": False,
    },
    "preseal_artifact_count": 24,
    "preseal_artifacts": artifact_rows,
    "preseal_artifact_manifest_sha256": object_sha(artifact_rows),
    "preseal_exact24_verified_single_fd_each": True,
    "postseal_target_exact25": True,
}
seal["seal_digest"] = object_sha(seal)
seal_raw = json.dumps(seal, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
seal_sha = hashlib.sha256(seal_raw).hexdigest()
seal_path = root / "seal.json"
flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
if not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("seal O_NOFOLLOW is unavailable")
descriptor = os.open(seal_path, flags | os.O_NOFOLLOW, 0o444)
try:
    os.fchmod(descriptor, 0o444)
    os.chmod(root / "logs", 0o555)
    os.chmod(root / "aggregate", 0o555)
    os.chmod(root, 0o555)
    offset = 0
    while offset < len(seal_raw):
        written = os.write(descriptor, seal_raw[offset:])
        if written <= 0:
            raise SystemExit("seal write failed")
        offset += written
    os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    readback = b""
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        readback += chunk
    seal_fd_info = os.fstat(descriptor)
    if (readback != seal_raw or hashlib.sha256(readback).hexdigest() != seal_sha
            or stat.S_IMODE(seal_fd_info.st_mode) != 0o444
            or seal_fd_info.st_nlink != 1 or seal_fd_info.st_size != len(seal_raw)):
        raise SystemExit("seal same-FD write/readback differs")
finally:
    os.close(descriptor)
seal_link = seal_path.lstat()
if (stat.S_ISLNK(seal_link.st_mode)
        or (seal_link.st_dev, seal_link.st_ino) != (seal_fd_info.st_dev, seal_fd_info.st_ino)):
    raise SystemExit("seal path/FD identity differs")

final_files = preseal_files | {"seal.json"}
final_dirs = {relative: 0o555 for relative in preseal_dirs}
inspect_tree(final_files, final_dirs, 0o555)
expected_final = {
    row["path"]: (row["sha256"], row["size_bytes"])
    for row in artifact_rows
}
expected_final["seal.json"] = (seal_sha, len(seal_raw))
if len(expected_final) != 25:
    raise SystemExit("final exact25 manifest differs")
for relative in sorted(final_files):
    _, info, digest = read_sealed(root / relative, "final exact25 " + relative)
    if (digest, info.st_size) != expected_final[relative]:
        raise SystemExit("final exact25 readback differs: " + relative)
print(json.dumps({
    "run_root": str(root), "artifact_count": 25,
    "exact24_to_exact25_verified": True,
    "seal_sha256": seal_sha, "seal_digest": seal["seal_digest"],
    "aggregate_receipt_sha256": aggregate_binding["sha256"],
    "aggregate_receipt_digest": aggregate_digest,
    "exposed_five_view_codec_development_gate": development_gate,
    "all_qualification_claims_false": True,
}, sort_keys=True, separators=(",", ":")))
PY
)"
  [[ "${fn_result}" == \{*\} \
      && "$(printf '%s\n' "${fn_result}" | wc -l | tr -d ' ')" == 1 ]] || \
    fail "postflight seal result differs"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4E_FINAL_SEAL={"postflight":%s,"postflight_cache_cleaned":true,"fresh_cache_root":"%s"}\n' \
    "${fn_result}" "${fn_cache}"
}

if [[ "${1:-}" == __run_fold_child ]]; then
  shift
  run_fold_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_preflight_child ]]; then
  shift
  run_preflight_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_aggregate_child ]]; then
  shift
  run_aggregate_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_seal_child ]]; then
  shift
  run_seal_child "$@"
  exit 0
fi

[[ $# -eq 8 ]] || \
  fail "usage: $0 RELEASE PYTHON FEATURE_ROOT V4A_RECEIPT V4C_FRONTIER_RECEIPT V4D_RECEIPT FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256"
readonly main_release_root=$1
readonly main_python_bin=$2
readonly main_feature_root=$3
readonly main_v4a_receipt=$4
readonly main_v4c_frontier_receipt=$5
readonly main_v4d_receipt=$6
readonly main_run_root=$7
readonly main_expected_controller_sha=$8
readonly main_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
readonly main_runtime="${main_release_root}/${runtime_relative}"
readonly main_test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_multiview_global_codec_v4e_alt

[[ "${main_expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller SHA argument differs"
require_plain_file "${main_controller_source}" "${main_expected_controller_sha}" 555 detached-controller
[[ "${main_controller_source}" != "${main_release_root%/}/"* ]] || \
  fail "controller must remain detached from release"
require_plain_file "${main_python_bin}" "${expected_python_sha256}" 755 python
require_release "${main_release_root}"
require_sealed_json "${main_feature_root}/feature_extraction_receipt.json" \
  "${expected_feature_receipt_sha256}" feature-receipt
require_sealed_json "${main_v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
require_sealed_json "${main_v4c_frontier_receipt}" \
  "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
require_sealed_json "${main_v4d_receipt}" "${expected_v4d_receipt_sha256}" v4d-receipt
[[ "${main_run_root}" == /* && "${main_run_root}" != / \
    && ! -e "${main_run_root}" && ! -L "${main_run_root}" \
    && "${main_run_root}" == "$(readlink -m -- "${main_run_root}")" ]] || \
  fail "run root is not fresh/absolute/canonical/safe"
readonly main_run_parent="$(dirname -- "${main_run_root}")"
[[ -d "${main_run_parent}" && ! -L "${main_run_parent}" && -w "${main_run_parent}" \
    && "${main_run_parent}" == "$(readlink -f -- "${main_run_parent}")" ]] || \
  fail "run parent differs"

if ! main_authority_pre="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "initial authority snapshot failed"
fi
[[ "${main_authority_pre}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}$ ]] || \
  fail "initial authority snapshot format differs"
readonly main_authority_pre

# This exact holder gate is also the deliberate stop point for the required
# real-entry fake-squeue smoke.  It remains before preflight srun and the first
# persistent mkdir.
for main_index in 0 1 2 3 4; do
  main_projection="$(squeue -h -j "${worker_jobs[${main_index}]}" \
    -w "${worker_nodes[${main_index}]}" -o '%T|%u' | tr -d ' ')"
  [[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
    fail "holder fold${main_index}/${worker_jobs[${main_index}]}/${worker_nodes[${main_index}]} differs"
done
main_projection="$(squeue -h -j "${preflight_job}" -w "${preflight_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "preflight holder ${preflight_job}/${preflight_node} differs"

# The expensive normal/-O suite, compilation and CLI audit run on a compute
# node.  No persistent run-root member exists until this exact CPU step passes
# and removes its node-local cache.
main_preflight_output="$(
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${preflight_job}" --nodelist="${preflight_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${preflight_cpus}" --mem="${preflight_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_preflight_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${preflight_job}" "${preflight_node}" "${main_expected_controller_sha}" \
        "${main_authority_pre}"
)"
[[ "${main_preflight_output}" == V4E_PREFLIGHT_PASS=* \
    && "$(printf '%s\n' "${main_preflight_output}" | wc -l | tr -d ' ')" == 1 ]] || \
  fail "CPU exact preflight result differs"
[[ ! -e "${main_run_root}" && ! -L "${main_run_root}" ]] || \
  fail "CPU preflight created the persistent run root"
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-preflight authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "CPU preflight changed authority"
for main_index in 0 1 2 3 4; do
  main_projection="$(squeue -h -j "${worker_jobs[${main_index}]}" \
    -w "${worker_nodes[${main_index}]}" -o '%T|%u' | tr -d ' ')"
  [[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
    fail "post-preflight holder fold${main_index} differs"
done

# First persistent mutation after every CPU/input/holder gate.
mkdir "${main_run_root}"
mkdir "${main_run_root}/logs"
chmod 0700 "${main_run_root}" "${main_run_root}/logs"
"${main_python_bin}" -I -S -B - "${main_run_root}/launch-plan.json" \
  "${main_expected_controller_sha}" "${main_authority_pre}" \
  "${main_preflight_output}" <<'PY'
from pathlib import Path
import json, os, sys
path = Path(sys.argv[1])
payload = {
    "schema_version": "v4e-exact5-parallel-launch-plan-v1",
    "controller_sha256": sys.argv[2], "authority_snapshot": sys.argv[3],
    "cpu_preflight": json.loads(sys.argv[4].split("=", 1)[1]),
    "preflight_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 4, "memory": "4G", "gres": "none",
        "overlap": True, "exact": True,
    },
    "source_authority": {
        "release_tree_sha256": "82b2341b8a14bb7d637b015914ae9507ddc1f205e5f6530ef32d85444c8690d4",
        "runtime_sha256": "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a",
        "runtime_test_sha256": "a2da53be053d6ad5814c1f16c506bb0313d5c1b9360569858fcba86c8a2458a7",
        "python_sha256": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
    },
    "receipt_contract": {
        "fold_schema": "semantic-anchor-vjepa2-multiview-global-codec-fold-receipt-v4e",
        "aggregate_schema": "semantic-anchor-vjepa2-multiview-global-codec-exact5-receipt-v4e",
        "status": "V4E_VJEPA2_EXPOSED_FIVE_VIEW_GLOBAL_CODEC_COMPLETE_BURNED_DEVELOPMENT",
        "fold_exact_files": ["fold.json", "selected.pt"],
    },
    "fixed_no_fallback": True,
    "workers": [
        {"fold": 0, "job": 143808, "node": "auh7-1b-gpu-315"},
        {"fold": 1, "job": 143812, "node": "auh7-1b-gpu-293"},
        {"fold": 2, "job": 143811, "node": "auh7-1b-gpu-306"},
        {"fold": 3, "job": 143808, "node": "auh7-1b-gpu-233"},
        {"fold": 4, "job": 143808, "node": "auh7-1b-gpu-268"},
    ],
    "worker_request": {"nodes": 1, "ntasks": 1, "cpus": 8,
                       "memory": "12G", "gres": "gpu:mi210:1",
                       "overlap": True, "exact": True},
    "aggregate_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 8, "memory": "12G", "gres": "none",
        "overlap": True, "exact": True,
    },
    "postflight_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 8, "memory": "12G", "gres": "none",
        "overlap": True, "exact": True,
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
with os.fdopen(descriptor, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444)
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno())
    handle.seek(0); readback = handle.read(); closed = os.fstat(handle.fileno())
after = path.lstat()
identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode,
                          value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
if (readback != raw or identity(written) != identity(closed) or identity(closed) != identity(after)
        or after.st_nlink != 1 or (after.st_mode & 0o7777) != 0o444 or path.is_symlink()):
    raise SystemExit("launch plan create-only same-FD seal differs")
PY

main_pids=()
main_rcs=()
for main_index in 0 1 2 3 4; do
  main_stdout="${main_run_root}/logs/fold${main_index}.stdout"
  main_stderr="${main_run_root}/logs/fold${main_index}.stderr"
  (
    set -o noclobber
    exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      srun --jobid="${worker_jobs[${main_index}]}" --nodelist="${worker_nodes[${main_index}]}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${worker_cpus}" --mem="${worker_memory}" \
        --gres=gpu:mi210:1 --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_fold_child \
          "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
          "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
          "${main_run_root}/fold${main_index}" "${main_index}" \
          "${worker_jobs[${main_index}]}" "${worker_nodes[${main_index}]}" \
          "${main_expected_controller_sha}" "${main_authority_pre}" \
        >"${main_stdout}" 2>"${main_stderr}"
  ) &
  main_pids[${main_index}]=$!
done

main_any_failure=0
for main_index in 0 1 2 3 4; do
  if wait "${main_pids[${main_index}]}"; then
    main_rcs[${main_index}]=0
  else
    main_rcs[${main_index}]=$?
    main_any_failure=1
  fi
  chmod 0444 "${main_run_root}/logs/fold${main_index}.stdout" \
    "${main_run_root}/logs/fold${main_index}.stderr"
done
(( main_any_failure == 0 )) || \
  fail "one or more v4-E folds failed; aggregate forbidden; rc=${main_rcs[*]}"

for main_index in 0 1 2 3 4; do
  [[ -d "${main_run_root}/fold${main_index}" \
      && ! -L "${main_run_root}/fold${main_index}" \
      && "$(stat -c %a "${main_run_root}/fold${main_index}")" == 555 \
      && "$(find "${main_run_root}/fold${main_index}" -maxdepth 1 -type f -printf '%f\n' \
           | LC_ALL=C sort)" == $'fold.json\nselected.pt' ]] || \
    fail "fold${main_index} did not return a sealed exact2 directory"
done
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-fold authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "fold workers changed authority"

# CPU aggregate is forbidden until all five exact2 fold roots are sealed.  It
# runs in its own exact CPU-only Slurm step so the login/controller process
# never torch-loads the five selected checkpoints.
mkdir "${main_run_root}/aggregate"
chmod 0700 "${main_run_root}/aggregate"
main_projection="$(squeue -h -j "${aggregate_job}" -w "${aggregate_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "aggregate holder ${aggregate_job}/${aggregate_node} differs"
main_aggregate_stdout="${main_run_root}/logs/aggregate.stdout"
main_aggregate_stderr="${main_run_root}/logs/aggregate.stderr"
main_aggregate_rc=0
(
  set -o noclobber
  exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${aggregate_job}" --nodelist="${aggregate_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${aggregate_cpus}" --mem="${aggregate_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_aggregate_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${aggregate_job}" "${aggregate_node}" \
        "${main_expected_controller_sha}" "${main_authority_pre}" \
      >"${main_aggregate_stdout}" 2>"${main_aggregate_stderr}"
) || main_aggregate_rc=$?
chmod 0444 "${main_aggregate_stdout}" "${main_aggregate_stderr}"
(( main_aggregate_rc == 0 )) || \
  fail "v4-E CPU aggregate child failed with status ${main_aggregate_rc}"
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-aggregate authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "CPU aggregate changed authority"

# The final exact24-to-exact25 readback and seal are executed by a second
# exact CPU-only compute step below; no checkpoint is loaded or hashed on the
# login/controller process.
main_projection="$(squeue -h -j "${aggregate_job}" -w "${aggregate_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "postflight holder ${aggregate_job}/${aggregate_node} differs"
main_seal_output="$(
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${aggregate_job}" --nodelist="${aggregate_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${aggregate_cpus}" --mem="${aggregate_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_seal_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${aggregate_job}" "${aggregate_node}" \
        "${main_expected_controller_sha}" "${main_authority_pre}"
)"
[[ "${main_seal_output}" == V4E_FINAL_SEAL=* \
    && "$(printf '%s\n' "${main_seal_output}" | wc -l | tr -d ' ')" == 1 ]] || \
  fail "final compute postflight result differs"
[[ "$(stat -c '%a:%h' "${main_run_root}/seal.json")" == 444:1 \
    && ! -L "${main_run_root}/seal.json" \
    && "$(stat -c %a "${main_run_root}")" == 555 \
    && -z "$(find "${main_run_root}" -type d ! -perm 0555 -print -quit)" ]] || \
  fail "final sealed run envelope differs"
printf '%s\n' "${main_seal_output}"
printf 'V4E_EXACT5_PARALLEL_COMPLETE run_root=%s\n' "${main_run_root}"
