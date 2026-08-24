#!/usr/bin/env bash
# SEALED: six independent exact1 MI210 shards for the v4-C V-JEPA2
# ordered-contextual feature authority, followed by model-free exact6
# postflight.  This controller never submits a new Slurm
# job; it may only create child steps inside the six allowlisted allocations.

set -Eeuo pipefail
umask 077

readonly tag=v4c-vjepa2-exact644-six-shard-v1

# Frozen source and environment pins.  Any byte change produces a new
# controller SHA and requires a fresh output root.
readonly release_sealed=true
readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_extractor_test_sha256=b44108958d2b4df1f7684f16ba949bb92b246cfc396059fca7add87be632e8d9

readonly expected_manifest_sha256=963cc02f9875048120fbea042ecbeac9b59e5e40d23121c52a9d2488556ca4e5
readonly expected_model_config_sha256=3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9
readonly expected_model_weights_sha256=25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b
readonly expected_model_processor_sha256=d2fab4418fc0390b62c4cd72ade56908a7929f80c62288adbe10dd8d23421227
readonly expected_model_config_size=785
readonly expected_model_weights_size=1303947864
readonly expected_model_processor_size=1298

readonly -a release_relative_files=(
  methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
  methods/bernini_action_editing/tests/test_extract_vjepa2_ordered_contextual_features_v4c.py
)
readonly -a release_expected_shas=(
  "${expected_extractor_sha256}"
  "${expected_extractor_test_sha256}"
)
readonly -a shard_jobs=(143808 143808 143808 143808 143812 143811)
readonly -a shard_nodes=(
  auh7-1b-gpu-233
  auh7-1b-gpu-268
  auh7-1b-gpu-292
  auh7-1b-gpu-315
  auh7-1b-gpu-293
  auh7-1b-gpu-306
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
  printf '{"controller":"%s","launch_performed":false,"release_sealed":%s,"source_pins_final":true,"shard_count":6}\n' \
    "${tag}" "${release_sealed}"
  exit 0
fi

[[ "${release_sealed}" == true ]] || \
  fail "UNSEALED CONTROLLER: source pins are placeholders and launch is forbidden"

require_plain_file() {
  local path=$1 expected_sha=$2 expected_mode=$3 label=$4 expected_size=${5:-}
  [[ "${path}" == /* && -f "${path}" && ! -L "${path}" ]] || fail "${label} is not an absolute plain file"
  [[ "$(stat -c '%a:%h' "${path}")" == "${expected_mode}:1" ]] || fail "${label} mode/link differs"
  [[ "$(sha256_file "${path}")" == "${expected_sha}" ]] || fail "${label} SHA differs"
  if [[ -n "${expected_size}" ]]; then
    [[ "$(stat -c %s "${path}")" == "${expected_size}" ]] || fail "${label} size differs"
  fi
}

require_release() {
  local release_root=$1 index path listing expected_listing
  [[ "${release_root}" == /* && -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root differs"
  [[ "$(stat -c %a "${release_root}")" == 555 ]] || fail "release root is not mode0555"
  [[ -z "$(find "${release_root}" -type l -print -quit)" ]] || fail "release contains a symlink"
  listing="$(cd "${release_root}" && find . -type f -printf '%P\n' | LC_ALL=C sort)"
  expected_listing="$(printf '%s\n' "${release_relative_files[@]}" | LC_ALL=C sort)"
  [[ "${listing}" == "${expected_listing}" ]] || fail "release is not exact2 files"
  for index in 0 1; do
    path="${release_root}/${release_relative_files[${index}]}"
    require_plain_file "${path}" "${release_expected_shas[${index}]}" 444 "release file ${release_relative_files[${index}]}"
  done
  [[ -z "$(find "${release_root}" -type d ! -perm 0555 -print -quit)" ]] || fail "release directory mode differs"
}

require_model_exact3() {
  local model_root=$1 listing
  [[ "${model_root}" == /* && -d "${model_root}" && ! -L "${model_root}" ]] || fail "model root differs"
  [[ "$(stat -c %a "${model_root}")" == 555 ]] || fail "model root is not mode0555"
  listing="$(find "${model_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
  [[ "${listing}" == $'config.json\nmodel.safetensors\nvideo_preprocessor_config.json' ]] || \
    fail "model root is not exact3 files"
  require_plain_file "${model_root}/config.json" "${expected_model_config_sha256}" 444 model-config "${expected_model_config_size}"
  require_plain_file "${model_root}/model.safetensors" "${expected_model_weights_sha256}" 444 model-weights "${expected_model_weights_size}"
  require_plain_file "${model_root}/video_preprocessor_config.json" "${expected_model_processor_sha256}" 444 model-processor "${expected_model_processor_size}"
  [[ "$(find "${model_root}" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d ' ')" == 3 ]] || fail "model regular-file count differs"
  [[ -z "$(find "${model_root}" -mindepth 1 -maxdepth 1 ! -type f -print -quit)" ]] || fail "model exact3 member type differs"
}

sealed_tree_digest() {
  "$1" -I -S -B - "$2" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest(), "size_bytes": info.st_size})
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

snapshot_authorities() {
  local python_bin=$1 release_root=$2 model_root=$3 manifest=$4
  local release_digest model_digest manifest_digest
  require_release "${release_root}"
  require_model_exact3 "${model_root}"
  require_plain_file "${manifest}" "${expected_manifest_sha256}" 444 exact644-manifest
  release_digest="$(sealed_tree_digest "${python_bin}" "${release_root}")"
  model_digest="$(sealed_tree_digest "${python_bin}" "${model_root}")"
  manifest_digest="$(sha256_file "${manifest}")"
  printf '%s:%s:%s\n' "${release_digest}" "${model_digest}" "${manifest_digest}"
}

fresh_cache() {
  local prefix=$1 parent=${SLURM_TMPDIR:-/tmp} cache
  [[ "${parent}" == /* && -d "${parent}" && ! -L "${parent}" && -w "${parent}" ]] || fail "scratch parent differs"
  cache="$(mktemp -d "${parent%/}/${prefix}.XXXXXX")"
  [[ -d "${cache}" && ! -L "${cache}" ]] || fail "fresh cache creation failed"
  printf '%s\n' "${cache}"
}

cleanup_cache() {
  local cache=$1 parent=${SLURM_TMPDIR:-/tmp}
  case "${cache}" in "${parent%/}/${tag}-"*.??????) ;; *) fail "unsafe cache cleanup target" ;; esac
  [[ -d "${cache}" && ! -L "${cache}" ]] || fail "cache cleanup identity differs"
  chmod -R u+w -- "${cache}"
  rm -rf -- "${cache}"
  [[ ! -e "${cache}" && ! -L "${cache}" ]] || fail "cache cleanup failed"
}

run_create_only_log() {
  local log=$1 rc=0
  shift
  (
    set -o noclobber
    "$@" >"${log}" 2>&1
  ) || rc=$?
  [[ -f "${log}" && ! -L "${log}" ]] || fail "create-only log was not materialized: ${log}"
  chmod 0444 "${log}"
  (( rc == 0 )) || fail "logged command failed with status ${rc}: ${log}"
}

gpu_gate() {
  local python_bin=$1 expected_job=$2 expected_node=$3 shard_index=$4
  "${python_bin}" -P -B - "${expected_job}" "${expected_node}" "${shard_index}" <<'PY'
import json, os, re, socket, subprocess, sys, time, uuid
import torch
job, node, shard_index = sys.argv[1:]
actual_node = socket.gethostname().split(".", 1)[0]
expected = {
    "SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
    "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
}
if any(os.environ.get(key) != value for key, value in expected.items()):
    raise SystemExit("Slurm exact1 task gate differs")
if actual_node != node or os.environ.get("SLURM_GPUS_ON_NODE") != "1":
    raise SystemExit("Slurm node/GPU count gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
physical_text = os.environ.get("SLURM_STEP_GPUS", "")
if not re.fullmatch(r"[0-9]+", step) or not re.fullmatch(r"[0-9]+", physical_text):
    raise SystemExit("numbered step/physical GPU authority differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.version.hip:
    raise SystemExit("logical exact1 ROCm gate differs")
name = torch.cuda.get_device_name(0)
logical_uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
if name != "AMD Instinct MI210" or not logical_uuid:
    raise SystemExit("logical MI210 name/UUID gate differs")
try:
    decoded_logical_uuid = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (ValueError, UnicodeDecodeError) as error:
    raise SystemExit("logical MI210 UUID does not decode to an HSA unique ID") from error
if re.fullmatch(r"[0-9a-f]{16}", decoded_logical_uuid) is None:
    raise SystemExit("decoded logical MI210 unique ID differs")
inventory = None
last_probe_output = ""
probe_env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX"):
    probe_env.pop(key, None)
for _ in range(3):
    probe = subprocess.run(
        ["rocm-smi", "--showuniqueid", "--json"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=probe_env,
    )
    last_probe_output = probe.stdout + probe.stderr
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
    raise SystemExit("single-device rocm-smi unique-ID probe failed: " + last_probe_output)
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
print("V4C_GPU_GATE=" + json.dumps({
    "schema_version": "v4c-vjepa2-exact1-gpu-gate-v1",
    "shard_index": int(shard_index), "job_id": job, "step_id": step,
    "node": actual_node, "ntasks": 1, "nnodes": 1, "logical_device_count": 1,
    "slurm_step_gpu_token": int(physical_text), "device_name": name,
    "logical_uuid": logical_uuid, "decoded_logical_unique_id": decoded_logical_uuid,
    "rocm_inventory_card": inventory_card, "physical_uuid": unique[0],
    "rocr_visible_devices_from_slurm": os.environ.get("ROCR_VISIBLE_DEVICES"),
    "hip_visible_devices_from_slurm": os.environ.get("HIP_VISIBLE_DEVICES"),
    "cuda_visible_devices_from_slurm": os.environ.get("CUDA_VISIBLE_DEVICES"),
}, sort_keys=True), flush=True)
PY
}

run_shard_child() {
  [[ $# -eq 10 ]] || fail "internal shard child argument count differs"
  local release_root=$1 python_bin=$2 manifest=$3 model_root=$4 shard_index=$5
  local shard_output=$6 expected_job=$7 expected_node=$8 expected_controller_sha=$9
  local pre_snapshot=${10} cache post_snapshot extractor
  [[ "$(sha256_file "${BASH_SOURCE[0]}")" == "${expected_controller_sha}" ]] || fail "child controller bytes differ"
  [[ "${SLURM_JOB_ID:-}" == "${expected_job}" ]] || fail "child job binding differs"
  [[ ! -e "${shard_output}" && ! -L "${shard_output}" ]] || fail "shard output is not fresh"
  require_release "${release_root}"
  require_model_exact3 "${model_root}"
  require_plain_file "${manifest}" "${expected_manifest_sha256}" 444 exact644-manifest
  extractor="${release_root}/${release_relative_files[0]}"
  cache="$(fresh_cache "${tag}-shard${shard_index}-${SLURM_JOB_ID:?}-${SLURM_STEP_ID:?}")"
  trap 'cleanup_cache "${cache}"' EXIT
  export TMPDIR="${cache}/tmp" TMP="${cache}/tmp" TEMP="${cache}/tmp"
  export XDG_CACHE_HOME="${cache}/xdg" HF_HOME="${cache}/hf" HUGGINGFACE_HUB_CACHE="${cache}/hf/hub"
  export PYTHONPYCACHEPREFIX="${cache}/pycache" TORCH_HOME="${cache}/torch"
  export MIOPEN_USER_DB_PATH="${cache}/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="${cache}/miopen-custom"
  export TRITON_CACHE_DIR="${cache}/triton" TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
  mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${PYTHONPYCACHEPREFIX}" \
    "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  export PYTHONPATH="${release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS=8
  gpu_gate "${python_bin}" "${expected_job}" "${expected_node}" "${shard_index}"
  "${python_bin}" -P -B "${extractor}" extract-shard \
    --manifest "${manifest}" --expected-manifest-sha256 "${expected_manifest_sha256}" \
    --model-root "${model_root}" --shard-index "${shard_index}" --num-shards 6 \
    --device cuda:0 --output "${shard_output}"
  post_snapshot="$(snapshot_authorities "${python_bin}" "${release_root}" "${model_root}" "${manifest}")"
  [[ "${post_snapshot}" == "${pre_snapshot}" ]] || fail "child code/model/input authority changed"
  "${python_bin}" -P -B - "${shard_output}" "${shard_index}" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys, torch
path = Path(sys.argv[1]).resolve(strict=True); index = int(sys.argv[2]); info = path.stat()
if stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
    raise SystemExit("shard seal differs")
with path.open("rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest(); handle.seek(0)
    value = torch.load(handle, map_location="cpu", weights_only=True)
count = value.get("record_count")
if value.get("shard_index") != index or value.get("num_shards") != 6 or count not in (107, 108):
    raise SystemExit("shard exact6 postflight differs")
print("V4C_SHARD_COMPLETE=" + json.dumps({
    "shard_index": index, "record_count": count, "sha256": digest,
    "size_bytes": info.st_size, "mode": 444, "nlink": 1,
}, sort_keys=True), flush=True)
PY
  trap - EXIT
  cleanup_cache "${cache}"
}

if [[ "${1:-}" == __extract_shard ]]; then
  shift
  run_shard_child "$@"
  exit 0
fi

[[ $# -eq 6 ]] || fail "usage: $0 RELEASE PYTHON MANIFEST MODEL_ROOT FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256"
release_root=$1
python_bin=$2
manifest=$3
model_root=$4
readonly run_root=$5
readonly expected_controller_sha=$6
controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
readonly controller_source
readonly extractor="${release_root}/${release_relative_files[0]}"
readonly extractor_test=methods.bernini_action_editing.tests.test_extract_vjepa2_ordered_contextual_features_v4c

[[ "${controller_source}" == /* && -f "${controller_source}" && ! -L "${controller_source}" ]] || fail "controller source differs"
[[ "$(stat -c '%a:%h' "${controller_source}")" == 555:1 ]] || fail "controller mode/link differs"
[[ "${expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller SHA argument differs"
[[ "$(sha256_file "${controller_source}")" == "${expected_controller_sha}" ]] || fail "controller SHA differs"
require_plain_file "${python_bin}" "${expected_python_sha256}" 755 python
[[ "${run_root}" == /* && "${run_root}" != / && ! -e "${run_root}" && ! -L "${run_root}" ]] || fail "run root is not fresh/absolute/safe"
run_parent="$(dirname -- "${run_root}")"
readonly run_parent
[[ -d "${run_parent}" && ! -L "${run_parent}" && -w "${run_parent}" ]] || fail "run parent differs"

# Persistent outputs remain impossible until both normal and -O unit suites,
# compile, and every CLI help surface pass in one fresh ephemeral CPU cache.
preflight_cache="$(fresh_cache "${tag}-preflight")"
readonly preflight_cache
trap 'cleanup_cache "${preflight_cache}"' EXIT
export TMPDIR="${preflight_cache}/tmp" TMP="${preflight_cache}/tmp" TEMP="${preflight_cache}/tmp"
export XDG_CACHE_HOME="${preflight_cache}/xdg" PYTHONPYCACHEPREFIX="${preflight_cache}/pycache"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}"
export PYTHONPATH="${release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
authority_pre="$(snapshot_authorities "${python_bin}" "${release_root}" "${model_root}" "${manifest}")"
readonly authority_pre
(
  cd "${release_root}"
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    "${python_bin}" -P -B -m unittest "${extractor_test}"
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    "${python_bin}" -O -P -B -m unittest "${extractor_test}"
  "${python_bin}" -P -B -m py_compile \
    "${release_relative_files[0]}" "${release_relative_files[1]}"
  "${python_bin}" -P -B "${extractor}" --help
  "${python_bin}" -P -B "${extractor}" extract-shard --help
  "${python_bin}" -P -B "${extractor}" aggregate-shards --help
)
[[ "$(snapshot_authorities "${python_bin}" "${release_root}" "${model_root}" "${manifest}")" == "${authority_pre}" ]] || \
  fail "preflight changed code/model/input authority"
trap - EXIT
cleanup_cache "${preflight_cache}"
unset TMPDIR TMP TEMP XDG_CACHE_HOME PYTHONPYCACHEPREFIX

for index in 0 1 2 3 4 5; do
  projection="$(squeue -h -j "${shard_jobs[${index}]}" -w "${shard_nodes[${index}]}" -o '%T|%u' | tr -d ' ')"
  [[ "${projection}" == "RUNNING|guangyi.chen" ]] || fail "holder ${shard_jobs[${index}]}/${shard_nodes[${index}]} differs"
done

# This is the first persistent mutation.  mkdir is create-only by construction.
mkdir "${run_root}"
mkdir "${run_root}/logs" "${run_root}/raw-shards"
chmod 0700 "${run_root}" "${run_root}/logs" "${run_root}/raw-shards"
"${python_bin}" -I -S -B - "${run_root}/launch-plan.json" "${authority_pre}" "${expected_controller_sha}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
path = Path(sys.argv[1]); release_digest, model_digest, manifest_sha = sys.argv[2].split(":")
value = {
    "schema_version": "v4c-vjepa2-exact644-six-shard-launch-plan-v1",
    "status": "SEALED_BEFORE_EXACT6_CHILD_STEPS", "controller_sha256": sys.argv[3],
    "release_tree_sha256": release_digest, "model_tree_sha256": model_digest,
    "manifest_sha256": manifest_sha,
    "jobs_and_nodes": [[143808, "auh7-1b-gpu-233"], [143808, "auh7-1b-gpu-268"],
        [143808, "auh7-1b-gpu-292"], [143808, "auh7-1b-gpu-315"],
        [143812, "auh7-1b-gpu-293"], [143811, "auh7-1b-gpu-306"]],
    "num_shards": 6, "ordinary_exact1_mi210_per_shard": True,
    "manual_gpu_visibility_binding": False,
    "downstream_frontier_included": False,
    "downstream_frontier_authorized_by_this_seal": False,
}
value["receipt_digest"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(raw); handle.flush(); os.fchmod(handle.fileno(), 0o444); os.fsync(handle.fileno())
info = path.lstat()
if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
    raise SystemExit("launch-plan seal differs")
PY

pids=()
for index in 0 1 2 3 4 5; do
  log="${run_root}/logs/shard-${index}.log"
  shard="${run_root}/raw-shards/shard-${index}.pt"
  (
    set -o noclobber
    exec >"${log}" 2>&1
    exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      srun --jobid="${shard_jobs[${index}]}" --nodelist="${shard_nodes[${index}]}" \
        --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=24G --gres=gpu:mi210:1 \
        --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${controller_source}" __extract_shard "${release_root}" "${python_bin}" "${manifest}" \
          "${model_root}" "${index}" "${shard}" "${shard_jobs[${index}]}" \
          "${shard_nodes[${index}]}" "${expected_controller_sha}" "${authority_pre}"
  ) &
  pids+=("$!")
done

shard_failure=0
for index in 0 1 2 3 4 5; do
  wait "${pids[${index}]}" || shard_failure=1
  [[ -f "${run_root}/logs/shard-${index}.log" && ! -L "${run_root}/logs/shard-${index}.log" ]] || fail "shard log absent"
  chmod 0444 "${run_root}/logs/shard-${index}.log"
done
(( shard_failure == 0 )) || fail "one or more exact6 shard steps failed"

shard_sha_text="$("${python_bin}" -P -B - "${run_root}" <<'PY'
from pathlib import Path
import hashlib, json, re, stat, sys, torch
root = Path(sys.argv[1]); all_ordinals = []; uuids = []
for index in range(6):
    log = root / "logs" / f"shard-{index}.log"; shard = root / "raw-shards" / f"shard-{index}.pt"
    lines = log.read_text(encoding="utf-8").splitlines()
    gates = [json.loads(line.split("=", 1)[1]) for line in lines if line.startswith("V4C_GPU_GATE=")]
    completions = [json.loads(line.split("=", 1)[1]) for line in lines if line.startswith("V4C_SHARD_COMPLETE=")]
    if len(gates) != 1 or len(completions) != 1 or gates[0]["shard_index"] != index:
        raise SystemExit("inline shard log gate differs")
    if gates[0]["logical_device_count"] != 1 or gates[0]["device_name"] != "AMD Instinct MI210":
        raise SystemExit("inline exact1 MI210 gate differs")
    uuids.append(gates[0]["logical_uuid"])
    info = shard.stat()
    if shard.is_symlink() or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
        raise SystemExit("raw shard seal differs")
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    if completions[0]["sha256"] != digest:
        raise SystemExit("raw shard/log SHA join differs")
    value = torch.load(shard, map_location="cpu", weights_only=True)
    if value.get("shard_index") != index or value.get("num_shards") != 6:
        raise SystemExit("raw shard placement differs")
    all_ordinals.extend(value.get("global_anchor_ordinals", []))
    print(digest)
if sorted(all_ordinals) != list(range(644)) or len(set(all_ordinals)) != 644:
    raise SystemExit("exact6 population is not exhaustive/disjoint")
if len(set(uuids)) != 6:
    raise SystemExit("six scientific processes did not bind six distinct MI210 UUIDs")
PY
)"
mapfile -t shard_shas <<<"${shard_sha_text}"
[[ "${#shard_shas[@]}" == 6 ]] || fail "postflight shard SHA count differs"

aggregate_args=("${python_bin}" -P -B "${extractor}" aggregate-shards
  --manifest "${manifest}" --expected-manifest-sha256 "${expected_manifest_sha256}"
  --output "${run_root}/features/feature_extraction_receipt.json")
for index in 0 1 2 3 4 5; do
  aggregate_args+=(--shard "${run_root}/raw-shards/shard-${index}.pt" --expected-shard-sha256 "${shard_shas[${index}]}")
done
mkdir "${run_root}/features"
chmod 0700 "${run_root}/features"
run_create_only_log "${run_root}/logs/postflight.log" \
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    "${aggregate_args[@]}"
readonly feature_receipt="${run_root}/features/feature_extraction_receipt.json"
[[ -f "${feature_receipt}" && ! -L "${feature_receipt}" && "$(stat -c '%a:%h' "${feature_receipt}")" == 444:1 ]] || fail "feature postflight receipt differs"
feature_receipt_sha="$(sha256_file "${feature_receipt}")"
readonly feature_receipt_sha
authority_postflight="$(snapshot_authorities "${python_bin}" "${release_root}" "${model_root}" "${manifest}")"
readonly authority_postflight
[[ "${authority_postflight}" == "${authority_pre}" ]] || fail "postflight code/model/input authority changed"

"${python_bin}" -I -S -B - "${run_root}" "${authority_pre}" "${feature_receipt_sha}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
root = Path(sys.argv[1]).resolve(strict=True); seal = root / "seal.json"
release_digest, model_digest, manifest_sha = sys.argv[2].split(":")
rows = []
for path in sorted(root.rglob("*")):
    if path == seal or path.is_dir():
        continue
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
        raise SystemExit("final artifact envelope differs: " + str(path))
    rows.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": info.st_size})
value = {
    "schema_version": "v4c-vjepa2-exact644-six-shard-seal-v1",
    "status": "V4C_EXACT6_FEATURES_COMPLETE_BURNED_DEVELOPMENT",
    "artifacts": rows, "release_tree_sha256_pre_post": release_digest,
    "model_tree_sha256_pre_post": model_digest, "manifest_sha256_pre_post": manifest_sha,
    "feature_receipt_sha256": sys.argv[3],
    "downstream_frontier_run": False,
    "downstream_frontier_authorized_by_this_seal": False,
    "video_generation_authorized": False, "renderer_authorized": False,
}
value["receipt_digest"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
fd = os.open(seal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(raw); handle.flush(); os.fchmod(handle.fileno(), 0o444); os.fsync(handle.fileno())
seal_info = seal.lstat()
if seal.is_symlink() or not stat.S_ISREG(seal_info.st_mode) or stat.S_IMODE(seal_info.st_mode) != 0o444 or seal_info.st_nlink != 1 or hashlib.sha256(seal.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
    raise SystemExit("final seal write/readback differs")
for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
    os.chmod(directory, 0o555)
os.chmod(root, 0o555)
print(hashlib.sha256(raw).hexdigest())
PY
printf 'V4C_EXACT6_FEATURES_PASS run_root=%s\n' "${run_root}"
