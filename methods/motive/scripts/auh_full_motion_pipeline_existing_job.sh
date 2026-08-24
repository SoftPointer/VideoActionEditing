#!/usr/bin/env bash
# Orchestrate the post-smoke full-motion pipeline inside one existing 8-node
# allocation.  This controller owns no model logic: it freshly revalidates the
# raw smoke gate, then invokes the frozen Qwen, Wan, postcheck, and selector
# controllers in order and validates every terminal receipt before advancing.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
nodes_csv="${MOTIVE_FULL_MOTION_PIPELINE_NODES:?set eight ordered nodes}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set source snapshot}"
source_tree_sha="${MOTIVE_FULL_MOTION_SOURCE_TREE_SHA256:?set source tree SHA-256}"
snapshot_tool="${MOTIVE_FULL_MOTION_SNAPSHOT_TOOL:?set snapshot verifier}"
code_root="${snapshot}/methods/motive"

smoke_input="${MOTIVE_FULL_MOTION_SMOKE_INPUT:?set smoke input}"
smoke_input_sha="${MOTIVE_FULL_MOTION_SMOKE_INPUT_SHA256:?set smoke input SHA-256}"
smoke_root="${MOTIVE_FULL_MOTION_SMOKE_QWEN_ROOT:?set smoke Qwen root}"
raw_gate="${MOTIVE_FULL_MOTION_SMOKE_GATE:?set raw smoke gate}"
raw_gate_sha="${MOTIVE_FULL_MOTION_SMOKE_GATE_SHA256:?set raw gate SHA-256}"
verified_gate="${MOTIVE_FULL_MOTION_VERIFIED_GATE:?set verified gate output}"
canary_iid="${MOTIVE_FULL_MOTION_CANARY_IID:?set smoke canary IID}"
minimum_hard_passes="${MOTIVE_FULL_MOTION_MINIMUM_HARD_PASSES:?set smoke hard-pass threshold}"
minimum_canary_dynamic_units="${MOTIVE_FULL_MOTION_MINIMUM_CANARY_DYNAMIC_UNITS:?set canary dynamic-unit threshold}"

full_input="${MOTIVE_FULL_MOTION_FULL_INPUT:?set full Qwen input}"
full_input_sha="${MOTIVE_FULL_MOTION_FULL_INPUT_SHA256:?set full Qwen input SHA-256}"
qwen_root="${MOTIVE_FULL_MOTION_FULL_QWEN_ROOT:?set full Qwen output root}"
qwen_done="${MOTIVE_FULL_MOTION_FULL_QWEN_DONE:?set full Qwen receipt}"
qwen_model="${MOTIVE_FULL_MOTION_QWEN_MODEL:?set Qwen model}"
qwen_model_metadata_sha="${MOTIVE_FULL_MOTION_QWEN_MODEL_METADATA_SHA256:?set Qwen model metadata SHA-256}"
qwen_python="${MOTIVE_FULL_MOTION_QWEN_PYTHON:?set Qwen Python}"
qwen_controller="${MOTIVE_FULL_MOTION_QWEN_DISTRIBUTED_CONTROLLER:?set distributed Qwen controller}"
qwen_gate_wait="${MOTIVE_FULL_MOTION_GATE_WAIT_SECONDS:?set Qwen gate wait}"

finalize_release_watcher="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER:?set finalizer/release-request watcher}"
finalize_node="${MOTIVE_FULL_MOTION_FINALIZE_NODE:?set finalizer node}"
watcher_receipt="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT:?set watcher receipt}"
watcher_ready="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY:?set watcher readiness receipt}"
watcher_log="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_LOG:?set watcher log}"
watcher_startup_wait="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_STARTUP_WAIT_SECONDS:?set watcher startup timeout}"
watcher_wait="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WAIT_SECONDS:?set watcher wait timeout}"
watcher_poll="${MOTIVE_FULL_MOTION_FINALIZE_RELEASE_POLL_SECONDS:?set watcher poll interval}"
watcher_cpus="${MOTIVE_FULL_MOTION_FINALIZE_CPUS:?set finalizer CPUs}"
final_pool="${MOTIVE_FULL_MOTION_FINAL_POOL:?set final-pool output root}"
generation_primary="${MOTIVE_FULL_MOTION_GENERATION_PRIMARY:?set primary-256 manifest}"
generation_done="${MOTIVE_FULL_MOTION_GENERATION_DONE:?set finalizer done receipt}"
production_root="${MOTIVE_FULL_MOTION_PRODUCTION_ROOT:?set production root}"
shard_manifest_root="${MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR:?set shard-manifest root}"
generation_shard_dir="${MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR:?set 32-shard leaf directory}"
release_dir="${MOTIVE_FULL_MOTION_RELEASE_DIR:?set release directory}"
release_request="${MOTIVE_FULL_MOTION_RELEASE_REQUEST:?set release request}"
release_id="${MOTIVE_FULL_MOTION_RELEASE_ID:?set release ID}"
release_challenge="${MOTIVE_FULL_MOTION_RELEASE_CHALLENGE:?set release challenge}"
root_release="${MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE:?set signed root release}"
release_python="${MOTIVE_FULL_MOTION_RELEASE_PYTHON:?set release verifier Python}"
release_wait="${MOTIVE_FULL_MOTION_RELEASE_WAIT_SECONDS:?set release wait timeout}"
release_poll="${MOTIVE_FULL_MOTION_RELEASE_POLL_SECONDS:?set release poll interval}"

wan_controller="${MOTIVE_FULL_MOTION_WAN_DISPATCHER:?set Wan dispatcher}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set Wan code root}"
wan_checkpoint="${MOTIVE_WAN22_CKPT_DIR:?set Wan checkpoint}"
wan_python="${MOTIVE_WAN22_PYTHON_BIN:?set Wan Python}"
wan_ffprobe="${MOTIVE_WAN22_FFPROBE_BIN:?set Wan ffprobe}"
wan_output="${MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT:?set Wan output root}"
wan_shards_root="${MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT:?set Wan shards root}"
wan_receipt="${MOTIVE_FULL_MOTION_WAN_DISPATCH_RECEIPT:?set Wan dispatcher receipt}"
wan_step_cpus="${MOTIVE_FULL_MOTION_WAN_STEP_CPUS:?set Wan step CPUs}"
wan_idle_interval="${MOTIVE_FULL_MOTION_WAN_IDLE_PROBE_INTERVAL_SECONDS:?set Wan idle probe interval}"
wan_frame_num="${MOTIVE_WAN22_FRAME_NUM:?set Wan frame count}"
wan_sample_steps="${MOTIVE_WAN22_SAMPLE_STEPS:?set Wan sample steps}"
wan_sample_shift="${MOTIVE_WAN22_SAMPLE_SHIFT:?set Wan sample shift}"
wan_size="${MOTIVE_WAN22_SIZE:?set Wan size}"
wan_base_seed="${MOTIVE_WAN22_BASE_SEED:?set Wan base seed}"

postcheck_controller="${MOTIVE_FULL_MOTION_POSTCHECK_DISPATCHER:?set postcheck dispatcher}"
postcheck_model="${MOTIVE_FULL_MOTION_POSTCHECK_MODEL:?set postcheck model}"
postcheck_python="${MOTIVE_FULL_MOTION_POSTCHECK_PYTHON:?set postcheck Python}"
postcheck_ffprobe="${MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE:?set postcheck ffprobe}"
postcheck_ffmpeg="${MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG:?set postcheck ffmpeg}"
postcheck_output="${MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT:?set postcheck output root}"
postcheck_receipt="${MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT:?set postcheck dispatcher receipt}"
postcheck_cpus="${MOTIVE_FULL_MOTION_POSTCHECK_CPUS:?set postcheck worker CPUs}"
postcheck_idle_interval="${MOTIVE_FULL_MOTION_POSTCHECK_IDLE_RECHECK_SECONDS:?set postcheck idle interval}"

select_controller="${MOTIVE_FULL_MOTION_SELECT128_CONTROLLER:?set select128 controller}"
select_python="${MOTIVE_FULL_MOTION_SELECT_PYTHON:?set selector Python}"
select_ffprobe="${MOTIVE_FULL_MOTION_FFPROBE:?set selector ffprobe}"
select_ffmpeg="${MOTIVE_FULL_MOTION_FFMPEG:?set selector ffmpeg}"
exact128_output="${MOTIVE_FULL_MOTION_EXACT128_OUTPUT:?set exact128 output}"
exact128_receipt="${MOTIVE_FULL_MOTION_EXACT128_RECEIPT:?set exact128 receipt}"
select_wait="${MOTIVE_FULL_MOTION_EXACT128_WAIT_SECONDS:?set selector wait timeout}"
select_poll="${MOTIVE_FULL_MOTION_EXACT128_POLL_SECONDS:?set selector poll interval}"

pipeline_python="${MOTIVE_FULL_MOTION_PIPELINE_PYTHON:?set pipeline validation Python}"
pipeline_controller="${MOTIVE_FULL_MOTION_PIPELINE_CONTROLLER:?set frozen pipeline controller path}"
pipeline_receipt="${MOTIVE_FULL_MOTION_PIPELINE_RECEIPT:?set pipeline receipt}"

fail() {
  printf '[full-motion-pipeline] %s\n' "$*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local path="$2"
  [[ "${path}" == /* && "${path}" != "/" ]] \
    || fail "${label} must be a non-root absolute path: ${path}"
}

require_plain_file() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -f "${path}" ]] \
    || fail "${label} must be a regular non-symlink file: ${path}"
}

require_plain_directory() {
  local label="$1"
  local path="$2"
  [[ ! -L "${path}" && -d "${path}" ]] \
    || fail "${label} must be a non-symlink directory: ${path}"
}

require_new() {
  local label="$1"
  local path="$2"
  [[ ! -e "${path}" && ! -L "${path}" ]] \
    || fail "create-only ${label} already exists: ${path}"
}

for digest in \
  "${source_tree_sha}" "${smoke_input_sha}" "${raw_gate_sha}" \
  "${full_input_sha}" "${qwen_model_metadata_sha}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid SHA-256 input"
done
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid Slurm job ID"
for value in \
  "${qwen_gate_wait}" "${release_wait}" "${release_poll}" \
  "${watcher_wait}" "${watcher_poll}" "${watcher_cpus}" \
  "${watcher_startup_wait}" \
  "${minimum_hard_passes}" "${minimum_canary_dynamic_units}" \
  "${wan_step_cpus}" "${wan_frame_num}" "${wan_sample_steps}" \
  "${postcheck_cpus}" "${postcheck_idle_interval}" \
  "${select_wait}" "${select_poll}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "invalid positive integer input"
done
[[ "${canary_iid}" == 1dbe39537c984690 ]] \
  || fail "pipeline requires the frozen two-person smoke canary"
[[ "${minimum_hard_passes}" == 3 \
  && "${minimum_canary_dynamic_units}" == 2 ]] \
  || fail "pipeline requires the frozen 3-pass/two-actor smoke policy"
[[ "${release_challenge}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "invalid release challenge"
[[ "${release_id}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid release ID"
[[ "${wan_idle_interval}" =~ ^[0-9]+$ ]] || fail "invalid Wan idle interval"
[[ "${wan_sample_shift}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || fail "invalid Wan sample shift"
[[ "${wan_base_seed}" =~ ^[0-9]+$ ]] || fail "invalid Wan base seed"
[[ "${wan_frame_num}" == 81 ]] || fail "pipeline requires exactly 81 Wan frames"
case "${wan_size}" in
  1280\*720|720\*1280|832\*480|480\*832) ;;
  *) fail "unsupported Wan size: ${wan_size}" ;;
esac

IFS=, read -r -a nodes <<<"${nodes_csv}"
(( ${#nodes[@]} == 8 )) || fail "exactly eight ordered nodes are required"
declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  [[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node: ${node}"
  [[ -z "${seen_nodes[${node}]:-}" ]] || fail "duplicate node: ${node}"
  seen_nodes["${node}"]=1
done
first_four="${nodes[0]},${nodes[1]},${nodes[2]},${nodes[3]}"
[[ "${finalize_node}" == "${nodes[0]}" ]] \
  || fail "finalizer must use the first ordered node"

for binding in \
  "snapshot:${snapshot}" "snapshot_tool:${snapshot_tool}" \
  "smoke_input:${smoke_input}" "smoke_root:${smoke_root}" \
  "raw_gate:${raw_gate}" "verified_gate:${verified_gate}" \
  "full_input:${full_input}" "qwen_root:${qwen_root}" \
  "qwen_done:${qwen_done}" "qwen_model:${qwen_model}" \
  "qwen_python:${qwen_python}" "qwen_controller:${qwen_controller}" \
  "finalize_release_watcher:${finalize_release_watcher}" \
  "watcher_receipt:${watcher_receipt}" "watcher_ready:${watcher_ready}" \
  "watcher_log:${watcher_log}" \
  "final_pool:${final_pool}" \
  "generation_primary:${generation_primary}" \
  "generation_done:${generation_done}" \
  "production_root:${production_root}" \
  "shard_manifest_root:${shard_manifest_root}" \
  "generation_shard_dir:${generation_shard_dir}" \
  "release_dir:${release_dir}" "release_request:${release_request}" \
  "root_release:${root_release}" "release_python:${release_python}" \
  "wan_controller:${wan_controller}" "wan_code_root:${wan_code_root}" \
  "wan_checkpoint:${wan_checkpoint}" "wan_python:${wan_python}" \
  "wan_ffprobe:${wan_ffprobe}" "wan_output:${wan_output}" \
  "wan_shards_root:${wan_shards_root}" "wan_receipt:${wan_receipt}" \
  "postcheck_controller:${postcheck_controller}" \
  "postcheck_model:${postcheck_model}" \
  "postcheck_python:${postcheck_python}" \
  "postcheck_ffprobe:${postcheck_ffprobe}" \
  "postcheck_ffmpeg:${postcheck_ffmpeg}" \
  "postcheck_output:${postcheck_output}" \
  "postcheck_receipt:${postcheck_receipt}" \
  "select_controller:${select_controller}" \
  "select_python:${select_python}" "select_ffprobe:${select_ffprobe}" \
  "select_ffmpeg:${select_ffmpeg}" "exact128_output:${exact128_output}" \
  "exact128_receipt:${exact128_receipt}" \
  "pipeline_python:${pipeline_python}" \
  "pipeline_controller:${pipeline_controller}" \
  "pipeline_receipt:${pipeline_receipt}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done

require_plain_directory "source snapshot" "${snapshot}"
require_plain_file "source snapshot closure" "${snapshot}/SOURCE_FILES.jsonl"
require_plain_directory "frozen Motive code root" "${code_root}"
require_plain_file "frozen smoke-gate implementation" \
  "${code_root}/motive/goku_full_motion_smoke_gate.py"
require_plain_file "frozen Qwen implementation" \
  "${code_root}/motive/goku_full_motion_qwen.py"
require_plain_file "snapshot verifier" "${snapshot_tool}"
require_plain_file "smoke input" "${smoke_input}"
require_plain_directory "smoke Qwen root" "${smoke_root}"
require_plain_file "raw smoke gate" "${raw_gate}"
require_plain_file "full Qwen input" "${full_input}"
require_plain_directory "Qwen model" "${qwen_model}"
require_plain_file "Qwen model config" "${qwen_model}/config.json"
require_plain_directory "postcheck model" "${postcheck_model}"
require_plain_file "postcheck model config" "${postcheck_model}/config.json"
require_plain_directory "Wan code root" "${wan_code_root}"
require_plain_directory "Wan checkpoint" "${wan_checkpoint}"
require_plain_directory "production root" "${production_root}"
require_plain_directory "release directory" "${release_dir}"
for path in \
  "${qwen_python}" "${release_python}" "${wan_python}" \
  "${postcheck_python}" "${select_python}" "${pipeline_python}" \
  "${wan_ffprobe}" "${postcheck_ffprobe}" "${postcheck_ffmpeg}" \
  "${select_ffprobe}" "${select_ffmpeg}"; do
  require_plain_file "runtime executable" "${path}"
  [[ -x "${path}" ]] || fail "runtime path is not executable: ${path}"
done
for path in \
  "${pipeline_controller}" "${qwen_controller}" \
  "${finalize_release_watcher}" \
  "${wan_controller}" \
  "${postcheck_controller}" "${select_controller}"; do
  require_plain_file "frozen child controller" "${path}"
  [[ -r "${path}" ]] || fail "frozen child controller is not readable: ${path}"
done
[[ "$(wc -l <"${smoke_input}" | tr -d '[:space:]')" == 8 ]] \
  || fail "smoke input must contain eight rows"
[[ "$(wc -l <"${full_input}" | tr -d '[:space:]')" == 768 ]] \
  || fail "full input must contain 768 rows"

[[ "${snapshot_tool}" == "${snapshot}/"* ]] \
  || fail "snapshot verifier must belong to the frozen snapshot"
[[ "${pipeline_controller}" == "${snapshot}/"* \
  && "${qwen_controller}" == "${snapshot}/"* \
  && "${finalize_release_watcher}" == "${snapshot}/"* \
  && "${wan_controller}" == "${snapshot}/"* \
  && "${postcheck_controller}" == "${snapshot}/"* \
  && "${select_controller}" == "${snapshot}/"* ]] \
  || fail "all child controllers must belong to the frozen snapshot"
[[ "${BASH_SOURCE[0]}" == "${pipeline_controller}" ]] \
  || fail "running pipeline path differs from the explicit frozen controller"
[[ "${generation_shard_dir}" == "${shard_manifest_root}/shards" ]] \
  || fail "generation shard leaf differs from shard-manifest root"
[[ "${shard_manifest_root}" == "${production_root}/generation_shards" ]] \
  || fail "generation shard root differs from production layout"
[[ "${release_dir}" == "${production_root}/release" \
  && "${release_request}" == "${release_dir}/release_request.json" \
  && "${root_release}" == "${release_dir}/root_signed_release.json" ]] \
  || fail "release products differ from the fixed production layout"
[[ "${generation_primary}" == "${final_pool}/primary_256.jsonl" \
  && "${generation_done}" == "${final_pool}/done.json" ]] \
  || fail "generation products differ from the fixed final-pool layout"
[[ "${wan_shards_root}" == "${wan_output}/wan_shards" ]] \
  || fail "Wan shards root differs from Wan output root"
[[ "${wan_receipt}" == "${wan_output}/dispatch_complete.json" ]] \
  || fail "Wan receipt differs from fixed dispatcher output"
[[ "${postcheck_receipt}" == "${postcheck_output}/dispatcher_receipt.json" ]] \
  || fail "postcheck receipt differs from fixed dispatcher output"
for auxiliary in \
  "${watcher_ready}" "${watcher_receipt}" "${watcher_log}" \
  "${pipeline_receipt}"; do
  case "${auxiliary}/" in
    "${final_pool}/"*|"${shard_manifest_root}/"*|"${release_dir}/"*)
      fail "auxiliary output overlaps a closed data product: ${auxiliary}"
      ;;
  esac
done
case "${final_pool}/" in
  "${production_root}/"*) fail "final pool must be outside production root" ;;
esac

for parent in \
  "${verified_gate%/*}" "${qwen_root%/*}" "${qwen_done%/*}" \
  "${watcher_receipt%/*}" "${watcher_ready%/*}" "${watcher_log%/*}" \
  "${final_pool%/*}" \
  "${shard_manifest_root%/*}" "${root_release%/*}" \
  "${wan_output%/*}" "${postcheck_output%/*}" \
  "${exact128_output%/*}" "${exact128_receipt%/*}" \
  "${pipeline_receipt%/*}"; do
  require_plain_directory "output parent" "${parent}"
  [[ -w "${parent}" ]] || fail "output parent is not writable: ${parent}"
done

require_new "verified smoke gate" "${verified_gate}"
require_new "full Qwen receipt" "${qwen_done}"
require_new "finalizer/release watcher receipt" "${watcher_receipt}"
require_new "finalizer/release watcher readiness" "${watcher_ready}"
require_new "finalizer/release watcher log" "${watcher_log}"
if [[ -e "${qwen_root}" || -L "${qwen_root}" ]]; then
  require_plain_directory "reserved full Qwen root" "${qwen_root}"
  [[ -z "$(find "${qwen_root}" -mindepth 1 -print -quit)" ]] \
    || fail "reserved full Qwen root is not empty"
fi
for binding in \
  "final pool:${final_pool}" \
  "generation primary:${generation_primary}" \
  "generation done:${generation_done}" \
  "generation shard root:${shard_manifest_root}" \
  "release request:${release_request}" \
  "root signed release:${root_release}" \
  "Wan output:${wan_output}" "postcheck output:${postcheck_output}" \
  "exact128 output:${exact128_output}" \
  "exact128 receipt:${exact128_receipt}" \
  "pipeline receipt:${pipeline_receipt}"; do
  require_new "${binding%%:*}" "${binding#*:}"
done

for command in scontrol squeue cmp mktemp; do
  command -v "${command}" >/dev/null || fail "required command is unavailable: ${command}"
done
job_record="$(scontrol show job -o "${job_id}")"
for required in \
  "JobId=${job_id}" "UserId=$(id -un)(" "JobState=RUNNING" \
  "NumNodes=8" "gres/gpu:mi210=64"; do
  [[ "${job_record}" == *"${required}"* ]] \
    || fail "allocation identity differs: ${required}"
done
allocation_expression="$(squeue -j "${job_id}" -h -o '%N')"
[[ -n "${allocation_expression}" ]] || fail "allocation node list is unavailable"
mapfile -t allocated_nodes < <(scontrol show hostnames "${allocation_expression}")
(( ${#allocated_nodes[@]} == 8 )) || fail "allocation does not expand to eight nodes"
for index in 0 1 2 3 4 5 6 7; do
  [[ "${allocated_nodes[${index}]}" == "${nodes[${index}]}" ]] \
    || fail "ordered allocation nodes differ at index ${index}"
done

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE || true
export PYTHONPATH="${snapshot}/methods/motive:${snapshot}"

verify_snapshot() {
  "${qwen_python}" "${snapshot_tool}" verify \
    --snapshot "${snapshot}" --expected-tree-sha256 "${source_tree_sha}" \
    >/dev/null || fail "frozen source snapshot verification failed"
}
verify_snapshot
cd "${code_root}"

gate_tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/motive-pipeline-gate.XXXXXX")"
rebuilt_gate="${gate_tmp_root}/rebuilt_gate.json"
cleanup_gate() {
  rm -f -- "${rebuilt_gate}"
  rmdir "${gate_tmp_root}" 2>/dev/null || true
}
trap cleanup_gate EXIT
if ! "${qwen_python}" -m motive.goku_full_motion_smoke_gate \
  --input "${smoke_input}" --qwen-root "${smoke_root}" \
  --output "${rebuilt_gate}" --canary-iid "${canary_iid}" \
  --minimum-hard-passes "${minimum_hard_passes}" \
  --minimum-canary-dynamic-units "${minimum_canary_dynamic_units}"; then
  fail "fresh smoke-gate reconstruction failed"
fi
cmp -s "${raw_gate}" "${rebuilt_gate}" \
  || fail "fresh smoke gate bytes differ from the raw gate"

"${qwen_python}" - \
  "${raw_gate}" "${verified_gate}" "${raw_gate_sha}" \
  "${smoke_input}" "${smoke_input_sha}" "${qwen_model}" \
  "${qwen_model_metadata_sha}" "${full_input}" \
  "${full_input_sha}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from motive.goku_full_motion_contract import object_sha256
from motive import goku_full_motion_qwen as qwen


def fail(message):
    raise RuntimeError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def closed_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail("duplicate gate key: " + key)
        value[key] = item
    return value


def model_metadata_digest(root):
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            fail("invalid model closure entry: " + str(path))
        if path.is_dir():
            continue
        stat = path.stat()
        row = {
            "path": path.relative_to(root).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 64 * 1024 * 1024:
            row["sha256"] = file_sha(path)
        rows.append(row)
    return object_sha256(rows)


gate_path = Path(sys.argv[1]).resolve(strict=True)
verified_path = Path(sys.argv[2])
expected_gate_sha = sys.argv[3]
input_path = Path(sys.argv[4]).resolve(strict=True)
expected_input_sha = sys.argv[5]
model_path = Path(sys.argv[6]).resolve(strict=True)
expected_model_metadata = sys.argv[7]
full_input_path = Path(sys.argv[8]).resolve(strict=True)
expected_full_input_sha = sys.argv[9]
if gate_path.is_symlink() or file_sha(gate_path) != expected_gate_sha:
    fail("raw gate path or SHA-256 differs")
if file_sha(input_path) != expected_input_sha:
    fail("smoke input SHA-256 differs")
if file_sha(full_input_path) != expected_full_input_sha:
    fail("full Qwen input SHA-256 differs")
raw = gate_path.read_bytes()
try:
    gate = json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda value: fail("non-finite gate JSON: " + value),
        object_pairs_hook=closed_pairs,
    )
except (UnicodeError, json.JSONDecodeError) as error:
    fail("invalid raw gate JSON: " + str(error))
if not isinstance(gate, dict):
    fail("raw gate is not an object")
payload = dict(gate)
gate_digest = payload.pop("gate_digest", None)
if (
    gate.get("schema_version") != "motive-goku-full-motion-qwen-smoke-gate-v6"
    or gate.get("status") != "pass"
    or gate_digest != object_sha256(payload)
):
    fail("raw gate identity or digest differs")
required_lineage = {
    "record": "goku-full-motion-qwen-record-v6",
    "hard_gate": "goku-full-motion-hard-gate-v6",
    "provenance": "goku-full-motion-qwen-provenance-v6",
    "source_inventory_alignment": (
        "motive-goku-full-motion-source-inventory-alignment-v4"
    ),
    "change_region_proposals": qwen.CHANGE_REGION_PROPOSALS_SCHEMA,
    "coverage_authority": qwen.COVERAGE_AUTHORITY_SCHEMA,
    "coverage_authority_inventory": (
        qwen.COVERAGE_AUTHORITY_INVENTORY_SCHEMA
    ),
    "coverage_authority_assignments": (
        qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA
    ),
    "coverage_authority_allowed_owner_map": (
        qwen.COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA
    ),
    "coverage_authority_alignment": (
        qwen.COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA
    ),
}
qwen_lineage = {
    "record": qwen.RECORD_SCHEMA,
    "hard_gate": qwen.HARD_GATE_SCHEMA,
    "provenance": qwen.PROVENANCE_SCHEMA,
    "source_inventory_alignment": qwen.SOURCE_INVENTORY_ALIGNMENT_SCHEMA,
    "change_region_proposals": qwen.CHANGE_REGION_PROPOSALS_SCHEMA,
    "coverage_authority": qwen.COVERAGE_AUTHORITY_SCHEMA,
    "coverage_authority_inventory": qwen.COVERAGE_AUTHORITY_INVENTORY_SCHEMA,
    "coverage_authority_assignments": (
        qwen.COVERAGE_AUTHORITY_ASSIGNMENTS_SCHEMA
    ),
    "coverage_authority_allowed_owner_map": (
        qwen.COVERAGE_AUTHORITY_ALLOWED_OWNER_MAP_SCHEMA
    ),
    "coverage_authority_alignment": qwen.COVERAGE_AUTHORITY_ALIGNMENT_SCHEMA,
}
if qwen_lineage != required_lineage or gate.get("qwen_lineage") != required_lineage:
    fail("raw gate is not the frozen two-stage Qwen v6/source-alignment-v4 lineage")
bound_input = gate.get("input")
if (
    not isinstance(bound_input, dict)
    or Path(str(bound_input.get("path", ""))).resolve(strict=True) != input_path
    or bound_input.get("sha256") != expected_input_sha
    or bound_input.get("rows") != 8
):
    fail("raw gate input binding differs")
runtime = gate.get("qwen_runtime")
if not isinstance(runtime, dict):
    fail("raw gate runtime is absent")
run_config = runtime.get("run_config")
schemas = run_config.get("schemas") if isinstance(run_config, dict) else None
if not isinstance(schemas, dict) or any(
    schemas.get(key) != value for key, value in required_lineage.items()
):
    fail("raw gate runtime schema lineage differs")
implementation_digest = object_sha256(qwen._implementation_bundle())
if runtime.get("implementation_digest") != implementation_digest:
    fail("raw gate implementation differs from frozen snapshot")
if Path(str(runtime.get("model_path", ""))).resolve(strict=True) != model_path:
    fail("raw gate model path differs")
hard_pass_iids = gate.get("hard_pass_iids")
hard_pass_bindings = gate.get("hard_pass_bindings")
if (
    type(gate.get("hard_passes")) is not int
    or gate["hard_passes"] < 3
    or not isinstance(hard_pass_iids, list)
    or len(hard_pass_iids) != gate["hard_passes"]
    or hard_pass_iids != sorted(set(hard_pass_iids))
    or not isinstance(hard_pass_bindings, list)
    or len(hard_pass_bindings) != gate["hard_passes"]
):
    fail("raw gate hard-pass closure differs")
binding_keys = {
    "iid",
    "record_schema_version",
    "provenance_schema_version",
    "hard_gate_schema_version",
    "change_region_proposals_schema_version",
    "coverage_authority_schema_version",
    "coverage_authority_inventory_schema_version",
    "coverage_authority_assignments_schema_version",
    "source_inventory_alignment_schema_version",
    "coverage_authority_alignment_schema_version",
    "media_verification_sha256",
    "change_region_proposals_sha256",
    "coverage_authority_inventory_prompt_sha256",
    "coverage_authority_inventory_visual_input_sha256",
    "coverage_authority_inventory_sha256",
    "coverage_authority_assignments_prompt_sha256",
    "coverage_authority_assignments_visual_input_sha256",
    "coverage_authority_assignments_sha256",
    "coverage_authority_sha256",
    "i0_grounding_sha256",
    "primary_source_census_sha256",
    "secondary_source_census_sha256",
    "source_inventory_alignment_sha256",
    "coverage_authority_alignment_sha256",
    "hard_gate_sha256",
    "result_sha256",
    "provenance_sha256",
}
binding_schema_fields = {
    "record_schema_version": required_lineage["record"],
    "provenance_schema_version": required_lineage["provenance"],
    "hard_gate_schema_version": required_lineage["hard_gate"],
    "change_region_proposals_schema_version": required_lineage[
        "change_region_proposals"
    ],
    "coverage_authority_schema_version": required_lineage[
        "coverage_authority"
    ],
    "coverage_authority_inventory_schema_version": required_lineage[
        "coverage_authority_inventory"
    ],
    "coverage_authority_assignments_schema_version": required_lineage[
        "coverage_authority_assignments"
    ],
    "source_inventory_alignment_schema_version": required_lineage[
        "source_inventory_alignment"
    ],
    "coverage_authority_alignment_schema_version": required_lineage[
        "coverage_authority_alignment"
    ],
}
digest_fields = binding_keys - set(binding_schema_fields) - {"iid"}
for iid, binding in zip(hard_pass_iids, hard_pass_bindings):
    if (
        not isinstance(binding, dict)
        or set(binding) != binding_keys
        or binding.get("iid") != iid
        or any(binding.get(key) != value for key, value in binding_schema_fields.items())
        or any(
            not isinstance(binding.get(key), str)
            or len(binding[key]) != 64
            or any(character not in "0123456789abcdef" for character in binding[key])
            for key in digest_fields
        )
    ):
        fail("raw gate hard-pass semantic binding differs")
canary = gate.get("canary")
if (
    not isinstance(canary, dict)
    or canary.get("iid") not in hard_pass_iids
    or canary.get("qwen_record_schema_version") != required_lineage["record"]
    or canary.get("qwen_hard_gate_schema_version") != required_lineage["hard_gate"]
    or canary.get("qwen_provenance_schema_version") != required_lineage["provenance"]
    or canary.get("coverage_authority_inventory_schema_version")
    != required_lineage["coverage_authority_inventory"]
    or canary.get("coverage_authority_assignments_schema_version")
    != required_lineage["coverage_authority_assignments"]
    or canary.get("source_inventory_alignment_schema_version")
    != required_lineage["source_inventory_alignment"]
    or canary.get("coverage_authority_alignment_schema_version")
    != required_lineage["coverage_authority_alignment"]
):
    fail("raw gate v6 canary lineage differs")
if model_metadata_digest(model_path) != expected_model_metadata:
    fail("Qwen model closure metadata differs")
if verified_path.exists() or verified_path.is_symlink():
    fail("verified gate already exists")
descriptor = os.open(
    verified_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o400,
)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
except BaseException:
    verified_path.unlink(missing_ok=True)
    raise
directory = os.open(verified_path.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
if verified_path.read_bytes() != raw or verified_path.stat().st_mode & 0o777 != 0o400:
    fail("published verified gate differs")
PY
require_plain_file "verified smoke gate" "${verified_gate}"
cleanup_gate
trap - EXIT
echo "[full-motion-pipeline] verified smoke gate published: ${verified_gate}"

validate_stage() {
  "${pipeline_python}" - "$@" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


def fail(message):
    raise RuntimeError(message)


def read(path, label):
    if path.is_symlink() or not path.is_file():
        fail(label + " is not a plain file: " + str(path))
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        fail(label + " changed while read")
    return raw


def closed_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail("duplicate JSON key: " + key)
        value[key] = item
    return value


def load(path, label):
    raw = read(path, label)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda item: fail("non-finite JSON: " + item),
            object_pairs_hook=closed_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(label + " is not strict JSON: " + str(error))
    if not isinstance(value, dict):
        fail(label + " is not an object")
    return value, raw


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_digest(value, field, label):
    payload = dict(value)
    stored = payload.pop(field, None)
    if not isinstance(stored, str) or stored != sha(canonical(payload)):
        fail(label + " digest differs")


def resolved_plain(raw_path, label, kind):
    original = Path(raw_path)
    if original.is_symlink():
        fail(label + " is symlinked")
    resolved = original.resolve(strict=True)
    if kind == "file" and not resolved.is_file():
        fail(label + " is not a file")
    if kind == "directory" and not resolved.is_dir():
        fail(label + " is not a directory")
    return resolved


mode = sys.argv[1]
if mode == "qwen":
    done = resolved_plain(sys.argv[2], "Qwen controller receipt", "file")
    if done.stat().st_mode & 0o777 != 0o400:
        fail("Qwen controller receipt mode differs")
    input_path = resolved_plain(sys.argv[3], "full Qwen input", "file")
    root = resolved_plain(sys.argv[4], "Qwen output root", "directory")
    job, nodes, expected_input_sha = sys.argv[5:8]
    observed_input_sha = sha(read(input_path, "full Qwen input"))
    if observed_input_sha != expected_input_sha:
        fail("full Qwen input SHA-256 differs")
    lines = read(done, "Qwen controller receipt").decode("utf-8").splitlines()
    expected = [
        "schema=motive-goku-full-motion-qwen-controller-v1",
        "status=complete",
        "input=" + str(input_path),
        "input_sha256=" + observed_input_sha,
        "output_root=" + str(root),
        "slurm_job_id=" + job,
        "nodes=" + nodes,
    ]
    if lines[:7] != expected or len(lines) != 8:
        fail("Qwen controller receipt binding differs")
    if not re.fullmatch(r"completed_at_utc=20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", lines[7]):
        fail("Qwen completion timestamp differs")
    expected_names = {
        *(f"qwen_shard_{index:03d}.jsonl" for index in range(8)),
        *(f"qwen_shard_{index:03d}.receipt.json" for index in range(8)),
    }
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != expected_names:
        fail("Qwen output closure differs")
    for name in expected_names:
        read(root / name, "Qwen artifact " + name)
elif mode == "watcher_ready":
    receipt = resolved_plain(
        sys.argv[2], "finalizer/release watcher readiness", "file"
    )
    if receipt.stat().st_mode & 0o777 != 0o400:
        fail("finalizer/release watcher readiness mode differs")
    job, nodes, qwen_nodes, finalize_node = sys.argv[3:7]
    value, _ = load(receipt, "finalizer/release watcher readiness")
    if (
        value.get("schema_version")
        != "motive-goku-full-motion-finalize-release-watcher-ready-v1"
        or value.get("status") != "ready"
    ):
        fail("finalizer/release watcher readiness identity differs")
    check_digest(value, "receipt_digest", "finalizer/release watcher readiness")
    if (
        str(value.get("slurm_job_id")) != job
        or value.get("nodes") != nodes.split(",")
        or value.get("qwen_nodes") != qwen_nodes.split(",")
        or value.get("finalize_node") != finalize_node
    ):
        fail("finalizer/release watcher readiness binding differs")
elif mode == "finalize_watcher":
    receipt = resolved_plain(
        sys.argv[2], "finalizer/release watcher receipt", "file"
    )
    if receipt.stat().st_mode & 0o777 != 0o400:
        fail("finalizer/release watcher receipt mode differs")
    job, nodes, finalize_node = sys.argv[3:6]
    snapshot = resolved_plain(sys.argv[6], "source snapshot", "directory")
    full_input = resolved_plain(sys.argv[7], "full Qwen input", "file")
    expected_full_sha = sys.argv[8]
    if sha(read(full_input, "full Qwen input")) != expected_full_sha:
        fail("full Qwen input changed before watcher completion")
    qwen_done = resolved_plain(sys.argv[9], "full Qwen done", "file")
    final_pool = resolved_plain(sys.argv[10], "final pool", "directory")
    shard_root = resolved_plain(
        sys.argv[11], "shard-manifest root", "directory"
    )
    release_request = resolved_plain(
        sys.argv[12], "release request", "file"
    )
    value, _ = load(receipt, "finalizer/release watcher receipt")
    if (
        value.get("schema_version")
        != "motive-goku-full-motion-finalize-release-watcher-v1"
        or value.get("status") != "complete"
    ):
        fail("finalizer/release watcher receipt identity differs")
    check_digest(value, "receipt_digest", "finalizer/release watcher receipt")
    if (
        str(value.get("slurm_job_id")) != job
        or value.get("nodes") != nodes.split(",")
        or value.get("finalize_node") != finalize_node
        or Path(str(value.get("source_snapshot", ""))).resolve(strict=True)
        != snapshot
    ):
        fail("finalizer/release watcher allocation binding differs")
    expected_records = (
        ("full_input", full_input, expected_full_sha),
        ("qwen_done", qwen_done, sha(read(qwen_done, "full Qwen done"))),
        (
            "release_request",
            release_request,
            sha(read(release_request, "release request")),
        ),
    )
    for field, path, expected_sha in expected_records:
        record = value.get(field)
        if (
            not isinstance(record, dict)
            or Path(str(record.get("path", ""))).resolve(strict=True) != path
            or record.get("sha256") != expected_sha
        ):
            fail("finalizer/release watcher file binding differs: " + field)
    if Path(str(value.get("final_pool", ""))).resolve(strict=True) != final_pool:
        fail("finalizer/release watcher final-pool binding differs")
    if (
        Path(str(value.get("shard_manifest_root", ""))).resolve(strict=True)
        != shard_root
    ):
        fail("finalizer/release watcher shard binding differs")
    if not re.fullmatch(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        str(value.get("completed_at_utc", "")),
    ):
        fail("finalizer/release watcher timestamp differs")
elif mode == "watchers":
    primary = resolved_plain(sys.argv[2], "generation primary", "file")
    final_done = resolved_plain(sys.argv[3], "finalizer done", "file")
    shard_root = resolved_plain(
        sys.argv[4], "shard-manifest root", "directory"
    )
    release = resolved_plain(sys.argv[5], "root signed release", "file")
    primary_raw = read(primary, "generation primary")
    if len(primary_raw.splitlines()) != 256 or any(not line.strip() for line in primary_raw.splitlines()):
        fail("generation primary is not exact-256 JSONL")
    final, _ = load(final_done, "finalizer done")
    if final.get("schema_version") != "motive-goku-full-motion-finalize-done-v1" or final.get("status") != "complete":
        fail("finalizer done identity differs")
    check_digest(final, "done_digest", "finalizer done")
    if shard_root.is_symlink() or not shard_root.is_dir():
        fail("shard-manifest root differs")
    shard_done, _ = load(shard_root / "done.json", "shard-manifest done")
    if shard_done.get("schema_version") != "motive-goku-full-motion-shard-manifest-done-v1" or shard_done.get("status") != "complete":
        fail("shard-manifest done identity differs")
    check_digest(shard_done, "done_digest", "shard-manifest done")
    leaf = shard_root / "shards"
    if leaf.is_symlink() or not leaf.is_dir():
        fail("shard-manifest leaf differs")
    expected = {f"shard_{index:03d}.jsonl" for index in range(32)}
    if {path.name for path in leaf.iterdir()} != expected:
        fail("generation shard file set differs")
    for name in expected:
        if len(read(leaf / name, "generation shard " + name).splitlines()) != 8:
            fail("generation shard row count differs: " + name)
    read(release, "root signed release")
elif mode == "wan":
    receipt = resolved_plain(sys.argv[2], "Wan dispatcher receipt", "file")
    job, nodes = sys.argv[3:5]
    shard_root = resolved_plain(
        sys.argv[5], "shard-manifest root", "directory"
    )
    release = resolved_plain(sys.argv[6], "root signed release", "file")
    output = resolved_plain(sys.argv[7], "Wan output root", "directory")
    value, _ = load(receipt, "Wan dispatcher receipt")
    if value.get("schema_version") != "motive-full-motion-wan-existing-allocation-dispatch-v1" or value.get("status") != "complete":
        fail("Wan dispatcher receipt identity differs")
    check_digest(value, "complete_digest", "Wan dispatcher receipt")
    if str(value.get("slurm_job_id")) != job or value.get("nodes") != nodes.split(","):
        fail("Wan dispatcher allocation binding differs")
    if Path(str(value.get("shard_manifest_dir", ""))).resolve(strict=True) != shard_root:
        fail("Wan dispatcher shard root differs")
    if Path(str(value.get("root_signed_release", ""))).resolve(strict=True) != release:
        fail("Wan dispatcher signed release differs")
    if Path(str(value.get("output_root", ""))).resolve(strict=True) != output:
        fail("Wan dispatcher output root differs")
    completed = value.get("completed_shards")
    if not isinstance(completed, list) or len(completed) != 32 or [item.get("shard_index") for item in completed] != list(range(32)):
        fail("Wan dispatcher completed-shard closure differs")
elif mode == "postcheck":
    receipt = resolved_plain(
        sys.argv[2], "postcheck dispatcher receipt", "file"
    )
    job, nodes = sys.argv[3:5]
    snapshot = resolved_plain(sys.argv[5], "source snapshot", "directory")
    shard_leaf = resolved_plain(
        sys.argv[6], "generation shard leaf", "directory"
    )
    wan_root = resolved_plain(sys.argv[7], "Wan shards root", "directory")
    model = resolved_plain(sys.argv[8], "postcheck model", "directory")
    output = resolved_plain(
        sys.argv[9], "postcheck output root", "directory"
    )
    value, _ = load(receipt, "postcheck dispatcher receipt")
    if value.get("schema_version") != "motive-goku-full-motion-postcheck-dispatch-receipt-v2" or value.get("status") != "complete":
        fail("postcheck dispatcher receipt identity differs")
    check_digest(value, "receipt_digest", "postcheck dispatcher receipt")
    if str(value.get("slurm_job_id")) != job or value.get("nodes") != nodes.split(","):
        fail("postcheck allocation binding differs")
    bindings = (
        ("source_snapshot", snapshot),
        ("generation_shard_dir", shard_leaf),
        ("wan_shards_root", wan_root),
        ("model", model),
    )
    for field, expected in bindings:
        if Path(str(value.get(field, ""))).resolve(strict=True) != expected:
            fail("postcheck binding differs: " + field)
    if value.get("completed_shards") != 32 or value.get("failed_shards") != []:
        fail("postcheck did not close all 32 shards")
    shards = value.get("shards")
    if not isinstance(shards, list) or len(shards) != 32:
        fail("postcheck shard receipt count differs")
    status = Path(str(value.get("status_tsv", ""))).resolve(strict=True)
    if status.parent != output or sha(read(status, "postcheck status")) != value.get("status_tsv_sha256"):
        fail("postcheck status binding differs")
elif mode == "select":
    receipt = resolved_plain(sys.argv[2], "select128 receipt", "file")
    snapshot = resolved_plain(sys.argv[3], "source snapshot", "directory")
    primary = resolved_plain(sys.argv[4], "generation primary", "file")
    generation_done = resolved_plain(sys.argv[5], "generation done", "file")
    shard_root = resolved_plain(
        sys.argv[6], "shard-manifest root", "directory"
    )
    wan_root = resolved_plain(sys.argv[7], "Wan shards root", "directory")
    post_receipt = resolved_plain(
        sys.argv[8], "postcheck dispatcher receipt", "file"
    )
    exact_root = resolved_plain(sys.argv[9], "exact128 output root", "directory")
    value, _ = load(receipt, "select128 controller receipt")
    if value.get("schema_version") != "motive-goku-full-motion-select128-controller-receipt-v1" or value.get("status") != "complete":
        fail("select128 controller receipt identity differs")
    check_digest(value, "receipt_digest", "select128 controller receipt")
    if value.get("config") != {"exact_size": 128, "min_multi_unit": 32}:
        fail("select128 policy differs")
    if Path(str(value.get("source_snapshot", ""))).resolve(strict=True) != snapshot:
        fail("select128 snapshot binding differs")
    generation = value.get("generation")
    if not isinstance(generation, dict):
        fail("select128 generation binding is absent")
    if Path(str(generation.get("primary", {}).get("path", ""))).resolve(strict=True) != primary or Path(str(generation.get("done", {}).get("path", ""))).resolve(strict=True) != generation_done:
        fail("select128 generation input differs")
    if Path(str(value.get("shard_manifest_root", ""))).resolve(strict=True) != shard_root or Path(str(value.get("wan_shards_root", ""))).resolve(strict=True) != wan_root:
        fail("select128 shard binding differs")
    dispatch = value.get("postcheck_dispatch")
    if not isinstance(dispatch, dict) or Path(str(dispatch.get("receipt", {}).get("path", ""))).resolve(strict=True) != post_receipt:
        fail("select128 postcheck receipt differs")
    output = value.get("output")
    if not isinstance(output, dict) or Path(str(output.get("root", ""))).resolve(strict=True) != exact_root:
        fail("select128 output binding differs")
    dataset = read(exact_root / "dataset_manifest.jsonl", "exact128 manifest")
    if len(dataset.splitlines()) != 128:
        fail("selected dataset is not exact-128")
    done, _ = load(exact_root / "done.json", "exact128 done")
    if done.get("status") != "complete":
        fail("exact128 done is not complete")
else:
    fail("unknown validation mode: " + mode)
PY
}

watcher_pid=""
stop_watcher() {
  if [[ -n "${watcher_pid}" ]] && kill -0 "${watcher_pid}" 2>/dev/null; then
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
  fi
}
trap stop_watcher EXIT

echo "[full-motion-pipeline] starting finalizer/release-request watcher"
verify_snapshot
(
  set -o noclobber
  exec >"${watcher_log}" 2>&1
  exec env \
    PYTHONOPTIMIZE= \
    MOTIVE_EXISTING_SLURM_JOB_ID="${job_id}" \
    MOTIVE_FULL_MOTION_PIPELINE_NODES="${nodes_csv}" \
    MOTIVE_FULL_MOTION_QWEN_NODES="${first_four}" \
    MOTIVE_FULL_MOTION_FINALIZE_NODE="${finalize_node}" \
    MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT="${snapshot}" \
    MOTIVE_FULL_MOTION_SOURCE_TREE_SHA256="${source_tree_sha}" \
    MOTIVE_FULL_MOTION_SNAPSHOT_TOOL="${snapshot_tool}" \
    MOTIVE_FULL_MOTION_FULL_INPUT="${full_input}" \
    MOTIVE_FULL_MOTION_FULL_INPUT_SHA256="${full_input_sha}" \
    MOTIVE_FULL_MOTION_FULL_QWEN_ROOT="${qwen_root}" \
    MOTIVE_FULL_MOTION_FULL_QWEN_DONE="${qwen_done}" \
    MOTIVE_FULL_MOTION_FINAL_POOL="${final_pool}" \
    MOTIVE_FULL_MOTION_PRODUCTION_ROOT="${production_root}" \
    MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR="${shard_manifest_root}" \
    MOTIVE_FULL_MOTION_RELEASE_DIR="${release_dir}" \
    MOTIVE_FULL_MOTION_RELEASE_REQUEST="${release_request}" \
    MOTIVE_FULL_MOTION_RELEASE_ID="${release_id}" \
    MOTIVE_FULL_MOTION_RELEASE_CHALLENGE="${release_challenge}" \
    MOTIVE_FULL_MOTION_QWEN_PYTHON="${qwen_python}" \
    MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WAIT_SECONDS="${watcher_wait}" \
    MOTIVE_FULL_MOTION_FINALIZE_RELEASE_POLL_SECONDS="${watcher_poll}" \
    MOTIVE_FULL_MOTION_FINALIZE_CPUS="${watcher_cpus}" \
    MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_READY="${watcher_ready}" \
    MOTIVE_FULL_MOTION_FINALIZE_RELEASE_WATCHER_RECEIPT="${watcher_receipt}" \
    /bin/bash "${finalize_release_watcher}"
) &
watcher_pid=$!
watcher_startup_deadline=$(( $(date +%s) + watcher_startup_wait ))
while [[ ! -s "${watcher_ready}" ]]; do
  [[ ! -L "${watcher_ready}" ]] \
    || fail "finalizer/release watcher readiness is symlinked"
  if ! kill -0 "${watcher_pid}" 2>/dev/null; then
    watcher_status=0
    wait "${watcher_pid}" || watcher_status=$?
    watcher_pid=""
    fail "finalizer/release-request watcher failed during startup: ${watcher_status}"
  fi
  (( $(date +%s) < watcher_startup_deadline )) \
    || fail "timed out waiting for finalizer/release watcher readiness"
  sleep 1
done
validate_stage watcher_ready \
  "${watcher_ready}" "${job_id}" "${nodes_csv}" "${first_four}" \
  "${finalize_node}" \
  || fail "finalizer/release watcher readiness receipt is invalid"

echo "[full-motion-pipeline] starting distributed full Qwen on ${first_four}"
verify_snapshot
if ! env \
  MOTIVE_EXISTING_SLURM_JOB_ID="${job_id}" \
  MOTIVE_FULL_MOTION_NODES="${first_four}" \
  MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT="${snapshot}" \
  MOTIVE_FULL_MOTION_SMOKE_GATE="${verified_gate}" \
  MOTIVE_FULL_MOTION_FULL_INPUT="${full_input}" \
  MOTIVE_FULL_MOTION_FULL_QWEN_ROOT="${qwen_root}" \
  MOTIVE_FULL_MOTION_FULL_QWEN_DONE="${qwen_done}" \
  MOTIVE_FULL_MOTION_QWEN_MODEL="${qwen_model}" \
  MOTIVE_FULL_MOTION_QWEN_PYTHON="${qwen_python}" \
  MOTIVE_FULL_MOTION_GATE_WAIT_SECONDS="${qwen_gate_wait}" \
  /bin/bash "${qwen_controller}"; then
  fail "distributed full-Qwen controller failed"
fi
validate_stage qwen \
  "${qwen_done}" "${full_input}" "${qwen_root}" "${job_id}" \
  "${first_four}" "${full_input_sha}" \
  || fail "distributed full-Qwen terminal closure is invalid"
echo "[full-motion-pipeline] distributed full Qwen complete"

watcher_status=0
wait "${watcher_pid}" || watcher_status=$?
watcher_pid=""
(( watcher_status == 0 )) \
  || fail "finalizer/release-request watcher failed: ${watcher_status}"
validate_stage finalize_watcher \
  "${watcher_receipt}" "${job_id}" "${nodes_csv}" "${finalize_node}" \
  "${snapshot}" "${full_input}" "${full_input_sha}" "${qwen_done}" \
  "${final_pool}" "${shard_manifest_root}" "${release_request}" \
  || fail "finalizer/release-request watcher receipt is invalid"
echo "[full-motion-pipeline] final pool, shards, and release request complete"

release_deadline=$(( $(date +%s) + release_wait ))
while true; do
  for path in \
    "${generation_primary}" "${generation_done}" \
    "${shard_manifest_root}/done.json" "${root_release}"; do
    [[ ! -L "${path}" ]] || fail "watcher publication is symlinked: ${path}"
  done
  if [[ -s "${generation_primary}" && -s "${generation_done}" \
    && -s "${shard_manifest_root}/done.json" && -s "${root_release}" ]]; then
    break
  fi
  (( $(date +%s) < release_deadline )) \
    || fail "timed out waiting for finalizer/release/signing watchers"
  sleep "${release_poll}"
done
validate_stage watchers \
  "${generation_primary}" "${generation_done}" \
  "${shard_manifest_root}" "${root_release}" \
  || fail "finalizer/shard/release watcher closure is invalid"

# The release verifier authorizes exactly one contiguous eight-row shard; the
# 256-row primary is deliberately not a valid argument to that interface.
# Verify every member of the exact 32x8 partition and independently close the
# shard bytes back over primary_256 before allowing Wan to reserve a GPU.
if ! PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH="${snapshot}/methods/motive:${snapshot}" \
  "${pipeline_python}" - \
    "${release_python}" "${root_release}" "${generation_primary}" \
    "${generation_shard_dir}" "${release_id}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def fail(message):
    raise RuntimeError(message)


def closed_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            fail("duplicate release-verifier key: " + key)
        value[key] = item
    return value


def stable_read(path, label):
    if path.is_symlink() or not path.is_file():
        fail(label + " is not a plain file: " + str(path))
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after):
        fail(label + " changed while read")
    return raw


python = Path(sys.argv[1])
release_path = Path(sys.argv[2]).resolve(strict=True)
primary_path = Path(sys.argv[3]).resolve(strict=True)
shard_dir = Path(sys.argv[4]).resolve(strict=True)
expected_release_id = sys.argv[5]
if python.is_symlink() or not python.is_file() or not os.access(python, os.X_OK):
    fail("release verifier Python is not a plain executable")
if shard_dir.is_symlink() or not shard_dir.is_dir():
    fail("generation shard leaf is not a plain directory")

expected_names = {f"shard_{index:03d}.jsonl" for index in range(32)}
if {entry.name for entry in shard_dir.iterdir()} != expected_names:
    fail("generation shard leaf is not the exact 32-file partition")

primary_raw = stable_read(primary_path, "primary-256 manifest")
primary_lines = primary_raw.splitlines(keepends=True)
if (
    len(primary_lines) != 256
    or any(not line.strip() or not line.endswith(b"\n") for line in primary_lines)
):
    fail("primary manifest is not canonical exact-256 line text")
primary_sha = hashlib.sha256(primary_raw).hexdigest()
release_raw = stable_read(release_path, "root signed release")
expected_binding_keys = {
    "path",
    "release_id",
    "payload_sha256",
    "signer_key_fingerprint",
    "root_manifest_sha256",
    "root_manifest_rows",
    "root_row_start_zero_based",
    "root_row_stop_exclusive",
}
expected_fingerprint = "SHA256:A6zKKVBr6MSG29PO5J7A91aJYKcORNOkidofuI+jf6Y"
payload_sha = None
ordered_shard_raw = []

for shard_index in range(32):
    manifest = shard_dir / f"shard_{shard_index:03d}.jsonl"
    manifest_raw = stable_read(manifest, f"generation shard {shard_index}")
    shard_lines = manifest_raw.splitlines(keepends=True)
    if (
        len(shard_lines) != 8
        or any(not line.strip() or not line.endswith(b"\n") for line in shard_lines)
        or shard_lines != primary_lines[shard_index * 8 : (shard_index + 1) * 8]
    ):
        fail(f"generation shard {shard_index} is not its exact primary slice")
    completed = subprocess.run(
        [
            str(python),
            "-m",
            "motive.wan22_full_motion_signed_release",
            "verify",
            "--release",
            str(release_path),
            "--manifest",
            str(manifest),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        fail(
            f"signed release rejected generation shard {shard_index}: "
            + detail[:1000]
        )
    try:
        binding = json.loads(
            completed.stdout.decode("utf-8"),
            parse_constant=lambda item: fail("non-finite verifier JSON: " + item),
            object_pairs_hook=closed_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(f"release verifier output is not strict JSON for shard {shard_index}: {error}")
    if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
        fail(f"release verifier binding schema differs for shard {shard_index}")
    observed_payload_sha = binding.get("payload_sha256")
    if (
        Path(str(binding.get("path", ""))).resolve(strict=True) != release_path
        or binding.get("release_id") != expected_release_id
        or not isinstance(observed_payload_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", observed_payload_sha) is None
        or binding.get("signer_key_fingerprint") != expected_fingerprint
        or binding.get("root_manifest_sha256") != primary_sha
        or binding.get("root_manifest_rows") != 256
        or binding.get("root_row_start_zero_based") != shard_index * 8
        or binding.get("root_row_stop_exclusive") != (shard_index + 1) * 8
    ):
        fail(f"release verifier root/partition binding differs for shard {shard_index}")
    if payload_sha is None:
        payload_sha = observed_payload_sha
    elif observed_payload_sha != payload_sha:
        fail("release payload SHA differs across generation shards")
    if stable_read(manifest, f"generation shard {shard_index} closure") != manifest_raw:
        fail(f"generation shard {shard_index} changed after signed verification")
    ordered_shard_raw.append(manifest_raw)

if b"".join(ordered_shard_raw) != primary_raw:
    fail("verified 32x8 generation shards do not reconstruct primary-256 bytes")
if stable_read(primary_path, "primary-256 closure") != primary_raw:
    fail("primary-256 changed during signed-partition verification")
if stable_read(release_path, "root signed release closure") != release_raw:
    fail("root signed release changed during partition verification")
PY
then
  fail "root signed release exact-32x8 partition verification failed"
fi
validate_stage watchers \
  "${generation_primary}" "${generation_done}" \
  "${shard_manifest_root}" "${root_release}" \
  || fail "finalizer/shard/release closure changed after signed verification"
echo "[full-motion-pipeline] root signed release verified over exact 32x8 partition"

echo "[full-motion-pipeline] starting Wan dispatch on ${nodes_csv}"
verify_snapshot
if ! env \
  MOTIVE_EXISTING_SLURM_JOB_ID="${job_id}" \
  MOTIVE_FULL_MOTION_WAN_NODES="${nodes_csv}" \
  MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT="${snapshot}" \
  MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR="${shard_manifest_root}" \
  MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE="${root_release}" \
  MOTIVE_WAN22_CODE_ROOT="${wan_code_root}" \
  MOTIVE_WAN22_CKPT_DIR="${wan_checkpoint}" \
  MOTIVE_WAN22_PYTHON_BIN="${wan_python}" \
  MOTIVE_WAN22_FFPROBE_BIN="${wan_ffprobe}" \
  MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT="${wan_output}" \
  MOTIVE_FULL_MOTION_WAN_STEP_CPUS="${wan_step_cpus}" \
  MOTIVE_FULL_MOTION_WAN_IDLE_PROBE_INTERVAL_SECONDS="${wan_idle_interval}" \
  MOTIVE_WAN22_FRAME_NUM="${wan_frame_num}" \
  MOTIVE_WAN22_SAMPLE_STEPS="${wan_sample_steps}" \
  MOTIVE_WAN22_SAMPLE_SHIFT="${wan_sample_shift}" \
  MOTIVE_WAN22_SIZE="${wan_size}" \
  MOTIVE_WAN22_BASE_SEED="${wan_base_seed}" \
  /bin/bash "${wan_controller}"; then
  fail "Wan dispatcher failed"
fi
validate_stage wan \
  "${wan_receipt}" "${job_id}" "${nodes_csv}" \
  "${shard_manifest_root}" "${root_release}" "${wan_output}" \
  || fail "Wan dispatcher terminal receipt is invalid"
echo "[full-motion-pipeline] Wan dispatch complete"

echo "[full-motion-pipeline] starting postcheck on ${first_four}"
verify_snapshot
if ! env \
  MOTIVE_EXISTING_SLURM_JOB_ID="${job_id}" \
  MOTIVE_FULL_MOTION_POSTCHECK_NODES="${first_four}" \
  MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT="${snapshot}" \
  MOTIVE_FULL_MOTION_GENERATION_SHARD_DIR="${generation_shard_dir}" \
  MOTIVE_FULL_MOTION_WAN_SHARDS_ROOT="${wan_shards_root}" \
  MOTIVE_FULL_MOTION_POSTCHECK_MODEL="${postcheck_model}" \
  MOTIVE_FULL_MOTION_POSTCHECK_PYTHON="${postcheck_python}" \
  MOTIVE_FULL_MOTION_POSTCHECK_FFPROBE="${postcheck_ffprobe}" \
  MOTIVE_FULL_MOTION_POSTCHECK_FFMPEG="${postcheck_ffmpeg}" \
  MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT="${postcheck_output}" \
  MOTIVE_FULL_MOTION_POSTCHECK_CPUS="${postcheck_cpus}" \
  MOTIVE_FULL_MOTION_POSTCHECK_IDLE_RECHECK_SECONDS="${postcheck_idle_interval}" \
  /bin/bash "${postcheck_controller}"; then
  fail "postcheck dispatcher failed"
fi
validate_stage postcheck \
  "${postcheck_receipt}" "${job_id}" "${first_four}" "${snapshot}" \
  "${generation_shard_dir}" "${wan_shards_root}" "${postcheck_model}" \
  "${postcheck_output}" \
  || fail "postcheck dispatcher terminal receipt is invalid"
echo "[full-motion-pipeline] postcheck complete"

echo "[full-motion-pipeline] starting exact128 selection"
verify_snapshot
if ! env \
  MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT="${snapshot}" \
  MOTIVE_FULL_MOTION_SELECT_PYTHON="${select_python}" \
  MOTIVE_FULL_MOTION_GENERATION_PRIMARY="${generation_primary}" \
  MOTIVE_FULL_MOTION_GENERATION_DONE="${generation_done}" \
  MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR="${shard_manifest_root}" \
  MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT="${wan_output}" \
  MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT="${postcheck_output}" \
  MOTIVE_FULL_MOTION_POSTCHECK_DISPATCH_RECEIPT="${postcheck_receipt}" \
  MOTIVE_FULL_MOTION_EXACT128_OUTPUT="${exact128_output}" \
  MOTIVE_FULL_MOTION_EXACT128_RECEIPT="${exact128_receipt}" \
  MOTIVE_FULL_MOTION_EXACT128_WAIT_SECONDS="${select_wait}" \
  MOTIVE_FULL_MOTION_EXACT128_POLL_SECONDS="${select_poll}" \
  MOTIVE_FULL_MOTION_FFPROBE="${select_ffprobe}" \
  MOTIVE_FULL_MOTION_FFMPEG="${select_ffmpeg}" \
  /bin/bash "${select_controller}"; then
  fail "select128 controller failed"
fi
validate_stage select \
  "${exact128_receipt}" "${snapshot}" "${generation_primary}" \
  "${generation_done}" "${shard_manifest_root}" "${wan_shards_root}" \
  "${postcheck_receipt}" "${exact128_output}" \
  || fail "select128 terminal receipt is invalid"

verify_snapshot

"${pipeline_python}" - \
  "${pipeline_receipt}" "${job_id}" "${nodes_csv}" "${first_four}" \
  "${snapshot}" "${verified_gate}" "${qwen_done}" \
  "${watcher_ready}" "${watcher_receipt}" \
  "${generation_primary}" "${generation_done}" \
  "${shard_manifest_root}/done.json" "${root_release}" \
  "${wan_receipt}" "${postcheck_receipt}" "${exact128_receipt}" \
  "${exact128_output}" "${pipeline_controller}" "${qwen_controller}" \
  "${finalize_release_watcher}" "${wan_controller}" \
  "${postcheck_controller}" "${select_controller}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def fail(message):
    raise RuntimeError(message)


def file_record(path, label):
    if path.is_symlink() or not path.is_file():
        fail(label + " is not a plain file")
    raw = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


target = Path(sys.argv[1])
job, nodes, first_four = sys.argv[2:5]
snapshot = Path(sys.argv[5]).resolve(strict=True)
paths = [Path(value).resolve(strict=True) for value in sys.argv[6:17]]
(
    gate,
    qwen_done,
    watcher_ready,
    watcher_receipt,
    primary,
    generation_done,
    shard_done,
    release,
    wan_receipt,
    post_receipt,
    select_receipt,
) = paths
exact_root = Path(sys.argv[17]).resolve(strict=True)
controllers = [Path(value).resolve(strict=True) for value in sys.argv[18:24]]
if target.exists() or target.is_symlink():
    fail("pipeline receipt already exists")
payload = {
    "schema_version": "motive-full-motion-existing-allocation-pipeline-v1",
    "status": "complete",
    "slurm_job_id": int(job),
    "nodes": nodes.split(","),
    "first_four_nodes": first_four.split(","),
    "source_snapshot": str(snapshot),
    "controllers": {
        name: file_record(path, name + " controller")
        for name, path in zip(
            (
                "pipeline",
                "qwen",
                "finalize_release_watcher",
                "wan",
                "postcheck",
                "select128",
            ),
            controllers,
        )
    },
    "stages": {
        "verified_smoke_gate": file_record(gate, "verified smoke gate"),
        "full_qwen": file_record(qwen_done, "full Qwen receipt"),
        "finalize_release_watcher": {
            "ready": file_record(
                watcher_ready, "finalizer/release watcher readiness"
            ),
            "complete": file_record(
                watcher_receipt, "finalizer/release watcher receipt"
            ),
        },
        "release": {
            "primary": file_record(primary, "generation primary"),
            "finalizer_done": file_record(generation_done, "generation done"),
            "shard_manifest_done": file_record(shard_done, "shard done"),
            "root_signed_release": file_record(release, "root release"),
        },
        "wan": file_record(wan_receipt, "Wan receipt"),
        "postcheck": file_record(post_receipt, "postcheck receipt"),
        "select128": {
            "receipt": file_record(select_receipt, "select128 receipt"),
            "output_root": str(exact_root),
        },
    },
    "completed_at_utc": datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    ),
}
receipt = dict(payload)
receipt["receipt_digest"] = hashlib.sha256(canonical(payload)).hexdigest()
raw = canonical(receipt) + b"\n"
parent = target.parent.resolve(strict=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix="." + target.name + ".", suffix=".tmp", dir=parent
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o400)
    os.link(temporary, target)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    temporary.unlink(missing_ok=True)
published = json.loads(target.read_text(encoding="utf-8"))
published_payload = dict(published)
published_digest = published_payload.pop("receipt_digest", None)
if target.read_bytes() != raw or published_digest != hashlib.sha256(canonical(published_payload)).hexdigest():
    fail("published pipeline receipt differs")
PY
require_plain_file "terminal pipeline receipt" "${pipeline_receipt}"
echo "[full-motion-pipeline] complete: ${pipeline_receipt}"
