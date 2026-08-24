#!/usr/bin/env bash
#SBATCH --job-name=saic-r6-rdzv-cny2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=00:30:00

set -Eeuo pipefail
umask 077

archive="${SAIC_R6_CANARY_V2_SOURCE_ARCHIVE:?}"
archive_sha="${SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256:?}"
guard="${SAIC_R6_CANARY_V2_GUARD:?}"
guard_sha="${SAIC_R6_CANARY_V2_GUARD_SHA256:?}"
python_bin="${SAIC_R6_CANARY_V2_PYTHON:?}"
python_sha="${SAIC_R6_CANARY_V2_PYTHON_SHA256:?}"
output_parent="${SAIC_R6_CANARY_V2_OUTPUT_PARENT:?}"
launcher_path="${SAIC_R6_CANARY_V2_LAUNCHER:?}"
launcher_sha="${SAIC_R6_CANARY_V2_LAUNCHER_SHA256:?}"
submission_receipt="${SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT:?}"
executed_launcher=""
readonly runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly maximum_launches=16

fail() { echo "[saic-r6-rdzv-canary-v2] ERROR: $*" >&2; exit 2; }
for value in archive guard python_bin output_parent launcher_path submission_receipt; do
  [[ "${!value}" == /* ]] || fail "${value} must be absolute"
done
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${guard_sha}" =~ ^[0-9a-f]{64}$ && \
   "${python_sha}" =~ ^[0-9a-f]{64}$ && "${launcher_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "input SHA differs"
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
archive="$(realpath -e -- "${archive}")"
guard="$(realpath -e -- "${guard}")"
python_bin="$(realpath -e -- "${python_bin}")"
output_parent="$(realpath -e -- "${output_parent}")"
launcher_path="$(realpath -e -- "${launcher_path}")"
executed_launcher="$(realpath -e -- "$0")"
[[ -f "${archive}" && ! -L "${archive}" && "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "archive differs"
[[ -f "${guard}" && ! -L "${guard}" && "$(sha256sum "${guard}" | awk '{print $1}')" == "${guard_sha}" ]] || fail "guard differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" && \
   "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "Python differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ -f "${launcher_path}" && ! -L "${launcher_path}" && \
   "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_sha}" ]] || fail "launcher differs"
[[ -r "${executed_launcher}" && -f "${executed_launcher}" && ! -L "${executed_launcher}" && \
   "$(stat -c '%h' -- "${executed_launcher}")" == 1 && \
   "$(sha256sum "${executed_launcher}" | awk '{print $1}')" == "${launcher_sha}" ]] || \
  fail "executed Slurm launcher differs"
[[ "${submission_receipt}" == "${output_parent}/submission-receipt.json" ]] || fail "submission receipt path differs"

"${python_bin}" -B - "${guard}" "${submission_receipt}" "${SLURM_JOB_ID}" \
  "${archive}" "${archive_sha}" "${guard}" "${guard_sha}" \
  "${python_bin}" "${python_sha}" "${output_parent}" "${launcher_path}" \
  "${launcher_sha}" <<'PY'
import importlib.util
from pathlib import Path
import sys
(
    guard_path, receipt_path,
) = map(Path, sys.argv[1:3])
(
    job, archive_value, archive_sha, guard_value, guard_sha,
    python_value, python_sha, output_parent_value, launcher_value, launcher_sha,
) = sys.argv[3:13]
output_parent = Path(output_parent_value)
spec = importlib.util.spec_from_file_location("canary_submission_gate", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
raw = module.wait_ready_bytes(receipt_path, label="own canary submission receipt")
value = module._decode_sealed(
    raw,
    schema_version="saic-r6-rendezvous-canary-submission-v2",
    exact_fields={
        "schema_version", "status", "submission_success", "job_success",
        "submitted_job", "request", "submission_boundary", "inputs",
        "outputs", "authority", "receipt_digest",
    },
)
submitted = value.get("submitted_job")
request = value.get("request")
boundary = value.get("submission_boundary")
inputs = value.get("inputs")
outputs = value.get("outputs")
authority = value.get("authority")
expected_exports = [
    "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE",
    "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256",
    "SAIC_R6_CANARY_V2_GUARD",
    "SAIC_R6_CANARY_V2_GUARD_SHA256",
    "SAIC_R6_CANARY_V2_PYTHON",
    "SAIC_R6_CANARY_V2_PYTHON_SHA256",
    "SAIC_R6_CANARY_V2_OUTPUT_PARENT",
    "SAIC_R6_CANARY_V2_LAUNCHER",
    "SAIC_R6_CANARY_V2_LAUNCHER_SHA256",
    "SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT",
]
if (
    value.get("status") != "submitted"
    or value.get("submission_success") is not True
    or value.get("job_success") is not None
    or not isinstance(submitted, dict)
    or set(submitted) != {"job_id", "cluster", "stdout_sha256", "stderr_sha256"}
    or submitted.get("job_id") != job
    or submitted.get("cluster") is not None and not isinstance(submitted.get("cluster"), str)
    or module.SHA256.fullmatch(str(submitted.get("stdout_sha256", ""))) is None
    or module.SHA256.fullmatch(str(submitted.get("stderr_sha256", ""))) is None
    or request != {
        "job_name": "saic-r6-rdzv-cny2", "partition": "faculty",
        "qos": "bgqos", "nodes": 1, "ntasks": 1, "cpus_per_task": 32,
        "memory": "64G", "walltime": "00:30:00",
        "gpu_resource_requested": "gpu:mi210:8",
        "world_topology": "two_concurrent_world4", "candidate_count": 60,
        "hold": False, "dependency": None, "scientific_generation": False,
    }
    or not isinstance(boundary, dict)
    or set(boundary) != {
        "environment_replaced", "exact_job_export_names", "export_all",
        "reservation_created_before_sbatch", "same_inode_retained",
        "launcher_submitted_from_retained_fd",
        "reservation_device", "reservation_inode", "success_mode",
    }
    or boundary.get("environment_replaced") is not True
    or boundary.get("exact_job_export_names") != expected_exports
    or boundary.get("export_all") is not False
    or boundary.get("reservation_created_before_sbatch") is not True
    or boundary.get("same_inode_retained") is not True
    or boundary.get("launcher_submitted_from_retained_fd") is not True
    or not isinstance(boundary.get("reservation_device"), int)
    or not isinstance(boundary.get("reservation_inode"), int)
    or boundary.get("success_mode") != "0444"
    or inputs != {
        "launcher": launcher_value,
        "launcher_sha256": launcher_sha,
        "guard": guard_value,
        "guard_sha256": guard_sha,
        "source_archive": archive_value,
        "source_archive_sha256": archive_sha,
        "runtime_sha256": module.EXPECTED_RUNTIME_SHA256,
        "python": python_value,
        "python_sha256": python_sha,
    }
    or outputs != {
        "output_parent": str(output_parent),
        "job_output_root": str(output_parent / f"job-{job}"),
        "submission_receipt": str(receipt_path),
        "fresh_before_submission": True,
    }
    or authority != {
        "scientific": False, "generation": False, "training": False,
        "publication": False, "formal_job_authorized": False,
    }
):
    raise SystemExit("own canary submission receipt differs")
PY

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || fail "scratch parent differs"
output_root="${output_parent}/job-${SLURM_JOB_ID}"
task_scratch=""
listener_pid=""
listener_recorded_pid=""
dog_pid=""
human_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  if [[ "${listener_pid:-}" =~ ^[0-9]+$ ]]; then kill "${listener_pid}" 2>/dev/null || true; wait "${listener_pid}" 2>/dev/null || true; fi
  for child in "${dog_pid:-}" "${human_pid:-}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; wait "${child}" 2>/dev/null || true; fi
  done
  if [[ -n "${task_scratch:-}" ]]; then
    case "${task_scratch}" in "${scratch_parent%/}/saic-r6-rdzv-canary-v2-${SLURM_JOB_ID}."*) rm -rf -- "${task_scratch}" || true;; esac
  fi
  if [[ "${status}" != 0 && -d "${output_root}" && ! -L "${output_root}" && \
        ! -e "${output_root}/canary-receipt.json" && ! -L "${output_root}/canary-receipt.json" && \
        ! -e "${output_root}/canary-failure-receipt.json" && ! -L "${output_root}/canary-failure-receipt.json" ]]; then
    "${python_bin}" -B - "${guard}" "${output_root}/canary-failure-receipt.json" \
      "${SLURM_JOB_ID}" "${status}" "${guard_sha}" "${archive_sha}" <<'PY' || true
import importlib.util
import os
from pathlib import Path
import sys
guard_path, receipt = map(Path, sys.argv[1:3])
job_id, status, guard_sha, archive_sha = sys.argv[3:7]
spec = importlib.util.spec_from_file_location("canary_guard_failure", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
core = {
    "schema_version": "saic-r6-dynamic-rendezvous-all8-canary-failure-v2",
    "status": "failed_before_operational_dynamic_rendezvous_exact60_pass",
    "slurm_job_id": job_id,
    "exit_status": int(status),
    "guard_sha256": guard_sha,
    "source_archive_sha256": archive_sha,
    "runtime_sha256": module.EXPECTED_RUNTIME_SHA256,
    "scientific_generation_entered": False,
    "scientific_output_created": False,
    "success_claimed": False,
    "authority": module.AUTHORITY,
}
module.write_create_only(receipt, module.seal(core))
os._exit(0)
PY
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "canary output must be fresh"
mkdir -m 0700 -- "${output_root}"
mkdir -m 0700 -- "${output_root}/logs" "${output_root}/rendezvous" \
  "${output_root}/rendezvous/port-claims" "${output_root}/forbidden-attempts"
claim_root="${output_root}/rendezvous/port-claims"
claim_identity="$(stat -Lc '%d:%i' -- "${claim_root}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-r6-rdzv-canary-v2-${SLURM_JOB_ID}.XXXXXX")"

mkdir -m 0700 -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
runtime="${method_root}/generate_saic_pure_t2v_event_bank_topup_v2.py"
find "${method_root}" -type f -exec chmod a-w -- {} +
[[ -f "${runtime}" && ! -L "${runtime}" && "$(sha256sum "${runtime}" | awk '{print $1}')" == "${runtime_sha}" ]] || fail "scientific runtime differs"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="${method_root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0

listener_ready="${output_root}/logs/legacy-arithmetic-port-listener.json"
listener_stop="${task_scratch}/stop-legacy-listener"
"${python_bin}" -B - "${guard}" "${listener_ready}" "${listener_stop}" "${SLURM_JOB_ID}" <<'PY' \
  >"${output_root}/logs/legacy-listener.stdout" 2>"${output_root}/logs/legacy-listener.stderr" &
import importlib.util, os
from pathlib import Path
import socket, sys, time
guard_path, ready, stop = map(Path, sys.argv[1:4])
job_id = sys.argv[4]
spec = importlib.util.spec_from_file_location("canary_guard_listener", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sockets = []
for port in range(48730, 48790):
    item = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    item.bind(("127.0.0.1", port))
    item.listen(8)
    sockets.append(item)
rows = []
for item in sockets:
    link = os.readlink(f"/proc/{os.getpid()}/fd/{item.fileno()}")
    rows.append({"port": item.getsockname()[1], "socket_inode": int(link[8:-1])})
core = {
    "schema_version": "saic-r6-legacy-arithmetic-port-listener-v2",
    "status": "all_legacy_ports_listening_before_dynamic_canary",
    "slurm_job_id": job_id,
    "node": socket.gethostname(),
    "pid": os.getpid(),
    "address": "127.0.0.1",
    "port_range_inclusive": [48730, 48789],
    "listener_count": len(rows),
    "listeners": rows,
    "receipt_publication_protocol": "open_first_inode_pinned_terminal_fchmod_v2",
    "authority": {"scientific": False, "generation": False, "training": False},
}
module.write_create_only(ready, module.seal(core))
while not stop.exists(): time.sleep(0.02)
for item in sockets: item.close()
PY
listener_pid=$!
listener_recorded_pid="${listener_pid}"
for _ in {1..500}; do
  if [[ -f "${listener_ready}" && "$(stat -Lc '%a' -- "${listener_ready}")" == 444 ]]; then break; fi
  kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener exited before ready"
  sleep 0.02
done
[[ -f "${listener_ready}" && "$(stat -Lc '%a' -- "${listener_ready}")" == 444 ]] || fail "legacy listener not ready"
"${python_bin}" -B - "${guard}" "${listener_ready}" "${SLURM_JOB_ID}" "${listener_recorded_pid}" <<'PY'
import importlib.util
import socket
from pathlib import Path
import sys
guard_path, listener = map(Path, sys.argv[1:3])
job, expected_pid = sys.argv[3:5]
spec = importlib.util.spec_from_file_location("canary_listener_preflight", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
fields = {
    "schema_version", "status", "slurm_job_id", "node", "pid", "address",
    "port_range_inclusive", "listener_count", "listeners",
    "receipt_publication_protocol", "authority", "receipt_digest",
}
raw = module.wait_ready_bytes(listener, label="legacy listener preflight")
value = module._decode_sealed(
    raw,
    schema_version="saic-r6-legacy-arithmetic-port-listener-v2",
    exact_fields=fields,
)
rows = value.get("listeners")
if (
    value.get("status") != "all_legacy_ports_listening_before_dynamic_canary"
    or value.get("slurm_job_id") != job
    or value.get("node") != socket.gethostname()
    or not expected_pid.isdigit()
    or value.get("pid") != int(expected_pid)
    or value.get("address") != "127.0.0.1"
    or value.get("port_range_inclusive") != [48730, 48789]
    or value.get("listener_count") != 60
    or value.get("receipt_publication_protocol")
    != "open_first_inode_pinned_terminal_fchmod_v2"
    or value.get("authority")
    != {"scientific": False, "generation": False, "training": False}
    or not isinstance(rows, list)
    or len(rows) != 60
    or {row.get("port") for row in rows if isinstance(row, dict)}
    != set(range(48730, 48790))
    or len({row.get("socket_inode") for row in rows if isinstance(row, dict)}) != 60
    or any(
        not isinstance(row, dict)
        or set(row) != {"port", "socket_inode"}
        or not isinstance(row.get("socket_inode"), int)
        or row["socket_inode"] <= 0
        for row in rows
    )
):
    raise SystemExit("legacy listener preflight differs")
PY
kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener exited after deep preflight"

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible="$2"
  local group_root="${output_root}/rendezvous/${group_id}"
  local group_scratch="${task_scratch}/${group_id}"
  mkdir -m 0700 -- "${group_root}" "${group_scratch}"
  mkdir -m 0700 -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" \
    "${group_scratch}/xdg" "${group_scratch}/triton" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export TMPDIR="${group_scratch}/tmp"
  local index candidate_id candidate_digest candidate_root forbidden_output
  local ordinal launch_dir life_identity rdzv_id status completed
  for ((index=0; index<30; index++)); do
    kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener died during canary"
    candidate_id="canary-${group_id}-candidate-$(printf '%02d' "${index}")"
    candidate_digest="$(printf '%s' "${candidate_id}" | sha256sum | awk '{print $1}')"
    candidate_root="${group_root}/candidate-$(printf '%02d' "${index}")-${candidate_digest:0:16}"
    forbidden_output="${output_root}/forbidden-attempts/${candidate_id}"
    mkdir -m 0700 -- "${candidate_root}"
    completed=false
    for ((ordinal=1; ordinal<=maximum_launches; ordinal++)); do
      [[ ! -e "${forbidden_output}" && ! -L "${forbidden_output}" ]] || fail "canary generated scientific output"
      launch_dir="${candidate_root}/launch-$(printf '%02d' "${ordinal}")"
      mkdir -m 0700 -- "${launch_dir}"
      life_identity="$(stat -Lc '%d:%i' -- "${launch_dir}")"
      rdzv_id="saic-${SLURM_JOB_ID}-${group_id}-c$(printf '%02d' "${index}")-${candidate_digest:0:16}-l$(printf '%02d' "${ordinal}")"
      if "${python_bin}" -B -m torch.distributed.run \
        --nproc-per-node=4 --max-restarts=0 \
        --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
        --rdzv-id="${rdzv_id}" --local-addr=127.0.0.1 \
        "${guard}" worker --runtime "${runtime}" \
        --expected-runtime-sha256 "${runtime_sha}" --slurm-job-id "${SLURM_JOB_ID}" \
        --group-id "${group_id}" --candidate-index "${index}" --candidate-id "${candidate_id}" \
        --launch-ordinal "${ordinal}" --expected-rdzv-id "${rdzv_id}" \
        --claim-root "${claim_root}" --claim-root-identity "${claim_identity}" \
        --lifecycle-dir "${launch_dir}" --lifecycle-dir-identity "${life_identity}" \
        -- --help >"${launch_dir}/torchrun.log" 2>&1; then
        status=0
      else
        status=$?
      fi
      chmod 0444 -- "${launch_dir}/torchrun.log"
      if [[ "${status}" == 0 ]]; then
        "${python_bin}" -B - "${guard}" "${launch_dir}" "${claim_root}" "${SLURM_JOB_ID}" \
          "${group_id}" "${index}" "${candidate_id}" "${ordinal}" "${rdzv_id}" "${forbidden_output}" <<'PY'
import importlib.util, os
from pathlib import Path
import sys
guard_path, life_value, claim_value = map(Path, sys.argv[1:4])
job, group, index, candidate, ordinal, rdzv, forbidden = sys.argv[4:11]
spec = importlib.util.spec_from_file_location("canary_guard", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
life = module.exact_directory(life_value, label="canary lifecycle")
if Path(forbidden).exists() or Path(forbidden).is_symlink(): raise SystemExit("scientific output appeared")
decision = module.wait_load_sealed(life / "admission.json", schema_version=module.DECISION_SCHEMA_VERSION, exact_fields=module.DECISION_FIELDS, label="completed canary admission")
port = decision["actual_master_port"]
claim_path = claim_value / f"port-{port}.json"
claim = module.wait_load_sealed(claim_path, schema_version=module.CLAIM_SCHEMA_VERSION, exact_fields=module.CLAIM_FIELDS, label="completed canary claim")
module._validate_generic_admission(claim, require_rank_packets=True)
if any(decision.get(k) != v for k, v in {"slurm_job_id":job,"group_id":group,"candidate_index":int(index),"candidate_id":candidate,"launch_ordinal":int(ordinal),"rdzv_id":rdzv}.items()): raise SystemExit("canary admission identity differs")
if os.environ.get("TORCH_DISABLE_SHARE_RDZV_TCP_STORE") != "0": raise SystemExit("shared TCPStore environment differs")
core = {"schema_version":"saic-r6-dynamic-rendezvous-operational-completion-v2","status":"unchanged_runtime_import_and_help_completed_after_exact_world4_admission","slurm_job_id":job,"group_id":group,"candidate_index":int(index),"candidate_id":candidate,"launch_ordinal":int(ordinal),"rdzv_id":rdzv,"actual_master_port":port,"claim_path":str(claim_path),"claim_receipt_digest":claim["receipt_digest"],"rank_packet_digests":decision["rank_packet_digests"],"admission_receipt_digest":decision["receipt_digest"],"runtime_sha256":module.EXPECTED_RUNTIME_SHA256,"torch_disable_share_rdzv_tcp_store":"0","shared_tcp_store_bootstrap":True,"receipt_publication_protocol":"open_first_inode_pinned_terminal_fchmod_v2","scientific_generation_entered":False,"scientific_output_created":False,"authority":module.AUTHORITY}
module.write_create_only(life / "operational-completion.json", module.seal(core))
PY
        completed=true
        break
      fi
      if "${python_bin}" -B "${guard}" admit-collision \
        --expected-runtime-sha256 "${runtime_sha}" --slurm-job-id "${SLURM_JOB_ID}" \
        --group-id "${group_id}" --candidate-index "${index}" --candidate-id "${candidate_id}" \
        --launch-ordinal "${ordinal}" --expected-rdzv-id "${rdzv_id}" \
        --claim-root "${claim_root}" --claim-root-identity "${claim_identity}" \
        --lifecycle-dir "${launch_dir}" --lifecycle-dir-identity "${life_identity}" \
        --candidate-output "${forbidden_output}" >/dev/null; then
        [[ "${ordinal}" -lt "${maximum_launches}" ]] || fail "canary collision retry budget exhausted"
        continue
      fi
      fail "canary torchrun failed without valid collision status=${status}"
    done
    [[ "${completed}" == true ]] || fail "canary candidate incomplete"
  done
)

run_group sp4-a "0,1,2,3" >"${output_root}/logs/sp4-a.log" 2>&1 & dog_pid=$!
run_group sp4-b "4,5,6,7" >"${output_root}/logs/sp4-b.log" 2>&1 & human_pid=$!
dog_status=0; human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
dog_pid=""
human_pid=""
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "canary group failed dog=${dog_status} human=${human_status}"
kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener not alive through all launches"
touch "${listener_stop}"
wait "${listener_pid}" || fail "legacy listener shutdown failed"
listener_pid=""

chmod 0444 -- "${output_root}/logs/legacy-listener.stdout" \
  "${output_root}/logs/legacy-listener.stderr" \
  "${output_root}/logs/sp4-a.log" "${output_root}/logs/sp4-b.log"
rm -rf -- "${task_scratch}"
task_scratch=""

trap - EXIT TERM INT
exec "${python_bin}" -B - "${guard}" "${output_root}" "${claim_root}" "${listener_ready}" \
  "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-$(hostname)}" "${guard_sha}" \
  "${archive}" "${archive_sha}" "${python_bin}" "${python_sha}" \
  "${launcher_path}" "${launcher_sha}" "${executed_launcher}" \
  "${submission_receipt}" "${listener_recorded_pid}" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

guard_path, root_value, claims_value, listener = map(Path, sys.argv[1:5])
(
    job, node, guard_sha, archive_value, archive_sha, python_value, python_sha,
    launcher_value, launcher_sha, executed_launcher_value,
    submission_receipt_value, expected_listener_pid,
) = sys.argv[5:17]
archive_path = Path(archive_value)
python_path = Path(python_value)
launcher_path = Path(launcher_value)
executed_launcher_path = Path(executed_launcher_value)
submission_receipt_path = Path(submission_receipt_value)
spec = importlib.util.spec_from_file_location("canary_guard_terminal", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
root = module.exact_directory(root_value, label="canary output root")
claims = module.exact_directory(claims_value, label="canary claim root")
success_path = root / "canary-receipt.json"
failure_path = root / "canary-failure-receipt.json"

LISTENER_FIELDS = {
    "schema_version", "status", "slurm_job_id", "node", "pid", "address",
    "port_range_inclusive", "listener_count", "listeners",
    "receipt_publication_protocol", "authority", "receipt_digest",
}
COMPLETION_FIELDS = {
    "schema_version", "status", "slurm_job_id", "group_id",
    "candidate_index", "candidate_id", "launch_ordinal", "rdzv_id",
    "actual_master_port", "claim_path", "claim_receipt_digest",
    "rank_packet_digests", "admission_receipt_digest", "runtime_sha256",
    "torch_disable_share_rdzv_tcp_store", "shared_tcp_store_bootstrap",
    "receipt_publication_protocol", "scientific_generation_entered",
    "scientific_output_created", "authority", "receipt_digest",
}
SUBMISSION_FIELDS = {
    "schema_version", "status", "submission_success", "job_success",
    "submitted_job", "request", "submission_boundary", "inputs",
    "outputs", "authority", "receipt_digest",
}
EXPECTED_EXPORTS = [
    "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE",
    "SAIC_R6_CANARY_V2_SOURCE_ARCHIVE_SHA256",
    "SAIC_R6_CANARY_V2_GUARD",
    "SAIC_R6_CANARY_V2_GUARD_SHA256",
    "SAIC_R6_CANARY_V2_PYTHON",
    "SAIC_R6_CANARY_V2_PYTHON_SHA256",
    "SAIC_R6_CANARY_V2_OUTPUT_PARENT",
    "SAIC_R6_CANARY_V2_LAUNCHER",
    "SAIC_R6_CANARY_V2_LAUNCHER_SHA256",
    "SAIC_R6_CANARY_V2_SUBMISSION_RECEIPT",
]

try:
    if (
        not job.isdigit()
        or not node
        or guard_sha != module.file_sha256(guard_path)
        or archive_path.resolve(strict=True) != archive_path
        or archive_sha != module.file_sha256(archive_path)
        or python_path.resolve(strict=True) != python_path
        or python_sha != module.file_sha256(python_path)
        or launcher_path.resolve(strict=True) != launcher_path
        or launcher_sha != module.file_sha256(launcher_path)
        or executed_launcher_path.resolve(strict=True) != executed_launcher_path
        or launcher_sha != module.file_sha256(executed_launcher_path)
        or module.SHA256.fullmatch(archive_sha) is None
        or os.environ.get("TORCH_DISABLE_SHARE_RDZV_TCP_STORE") != "0"
        or claims != root / "rendezvous/port-claims"
        or submission_receipt_path != root.parent / "submission-receipt.json"
        or sorted(item.name for item in root.iterdir())
        != ["forbidden-attempts", "logs", "rendezvous"]
    ):
        raise RuntimeError("canary terminal identity differs")

    submission_raw = module.wait_ready_bytes(
        submission_receipt_path, label="terminal own submission receipt"
    )
    submission_value = module._decode_sealed(
        submission_raw,
        schema_version="saic-r6-rendezvous-canary-submission-v2",
        exact_fields=SUBMISSION_FIELDS,
    )
    submitted = submission_value.get("submitted_job")
    boundary = submission_value.get("submission_boundary")
    if (
        submission_value.get("status") != "submitted"
        or submission_value.get("submission_success") is not True
        or submission_value.get("job_success") is not None
        or not isinstance(submitted, dict)
        or set(submitted) != {
            "job_id", "cluster", "stdout_sha256", "stderr_sha256"
        }
        or submitted.get("job_id") != job
        or (
            submitted.get("cluster") is not None
            and not isinstance(submitted.get("cluster"), str)
        )
        or module.SHA256.fullmatch(
            str(submitted.get("stdout_sha256", ""))
        ) is None
        or module.SHA256.fullmatch(
            str(submitted.get("stderr_sha256", ""))
        ) is None
        or submission_value.get("request") != {
            "job_name": "saic-r6-rdzv-cny2", "partition": "faculty",
            "qos": "bgqos", "nodes": 1, "ntasks": 1,
            "cpus_per_task": 32, "memory": "64G",
            "walltime": "00:30:00",
            "gpu_resource_requested": "gpu:mi210:8",
            "world_topology": "two_concurrent_world4",
            "candidate_count": 60, "hold": False, "dependency": None,
            "scientific_generation": False,
        }
        or not isinstance(boundary, dict)
        or set(boundary) != {
            "environment_replaced", "exact_job_export_names", "export_all",
            "reservation_created_before_sbatch", "same_inode_retained",
            "launcher_submitted_from_retained_fd",
            "reservation_device", "reservation_inode", "success_mode",
        }
        or boundary.get("environment_replaced") is not True
        or boundary.get("exact_job_export_names") != EXPECTED_EXPORTS
        or boundary.get("export_all") is not False
        or boundary.get("reservation_created_before_sbatch") is not True
        or boundary.get("same_inode_retained") is not True
        or boundary.get("launcher_submitted_from_retained_fd") is not True
        or not isinstance(boundary.get("reservation_device"), int)
        or not isinstance(boundary.get("reservation_inode"), int)
        or boundary.get("success_mode") != "0444"
        or submission_value.get("inputs") != {
            "launcher": str(launcher_path),
            "launcher_sha256": launcher_sha,
            "guard": str(guard_path),
            "guard_sha256": guard_sha,
            "source_archive": str(archive_path),
            "source_archive_sha256": archive_sha,
            "runtime_sha256": module.EXPECTED_RUNTIME_SHA256,
            "python": str(python_path),
            "python_sha256": python_sha,
        }
        or submission_value.get("outputs") != {
            "output_parent": str(root.parent),
            "job_output_root": str(root),
            "submission_receipt": str(submission_receipt_path),
            "fresh_before_submission": True,
        }
        or submission_value.get("authority") != {
            "scientific": False, "generation": False, "training": False,
            "publication": False, "formal_job_authorized": False,
        }
    ):
        raise RuntimeError("terminal own submission receipt differs")
    logs = module.exact_directory(root / "logs", label="canary logs")
    rendezvous = module.exact_directory(root / "rendezvous", label="canary rendezvous")
    forbidden = module.exact_directory(
        root / "forbidden-attempts", label="forbidden scientific output"
    )
    if (
        sorted(item.name for item in logs.iterdir())
        != [
            "legacy-arithmetic-port-listener.json", "legacy-listener.stderr",
            "legacy-listener.stdout", "sp4-a.log", "sp4-b.log",
        ]
        or sorted(item.name for item in rendezvous.iterdir())
        != ["port-claims", "sp4-a", "sp4-b"]
        or any(forbidden.iterdir())
        or listener != logs / "legacy-arithmetic-port-listener.json"
    ):
        raise RuntimeError("canary output closure differs")
    for log_name in (
        "legacy-listener.stderr", "legacy-listener.stdout", "sp4-a.log", "sp4-b.log"
    ):
        module.exact_plain_file(logs / log_name, label=log_name, mode=0o444)

    listener_raw = module.wait_ready_bytes(listener, label="legacy listener receipt")
    listener_value = module._decode_sealed(
        listener_raw,
        schema_version="saic-r6-legacy-arithmetic-port-listener-v2",
        exact_fields=LISTENER_FIELDS,
    )
    listener_rows = listener_value.get("listeners")
    if (
        listener_value.get("status")
        != "all_legacy_ports_listening_before_dynamic_canary"
        or listener_value.get("slurm_job_id") != job
        or listener_value.get("node") != socket.gethostname()
        or not expected_listener_pid.isdigit()
        or listener_value.get("pid") != int(expected_listener_pid)
        or listener_value.get("address") != "127.0.0.1"
        or listener_value.get("port_range_inclusive") != [48730, 48789]
        or listener_value.get("listener_count") != 60
        or listener_value.get("receipt_publication_protocol")
        != "open_first_inode_pinned_terminal_fchmod_v2"
        or listener_value.get("authority")
        != {"scientific": False, "generation": False, "training": False}
        or not isinstance(listener_rows, list)
        or len(listener_rows) != 60
        or any(
            not isinstance(row, dict)
            or set(row) != {"port", "socket_inode"}
            or not isinstance(row.get("port"), int)
            or not isinstance(row.get("socket_inode"), int)
            or row["socket_inode"] <= 0
            for row in listener_rows
        )
    ):
        raise RuntimeError("legacy listener receipt differs")
    ports_legacy = {row["port"] for row in listener_rows}
    if (
        ports_legacy != set(range(48730, 48790))
        or len({row["socket_inode"] for row in listener_rows}) != 60
    ):
        raise RuntimeError("legacy listener coverage differs")

    rows = []
    all_run_ids = []
    referenced_claims = set()
    collision_prior_claims = set()
    collision_count = 0
    for group in ("sp4-a", "sp4-b"):
        group_root = module.exact_directory(
            rendezvous / group, label=f"{group} canary group"
        )
        candidate_dirs = sorted(group_root.iterdir(), key=lambda item: item.name)
        if len(candidate_dirs) != 30:
            raise RuntimeError("canary candidate count differs")
        for index, candidate_root in enumerate(candidate_dirs):
            candidate_id = f"canary-{group}-candidate-{index:02d}"
            candidate_digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()
            if candidate_root.name != f"candidate-{index:02d}-{candidate_digest[:16]}":
                raise RuntimeError("canary candidate directory identity differs")
            module.exact_directory(candidate_root, label="canary candidate")
            launches = sorted(candidate_root.iterdir(), key=lambda item: item.name)
            if not launches or len(launches) > module.MAX_LAUNCH_ORDINAL:
                raise RuntimeError("canary launch count differs")
            completion = None
            completion_life = None
            for ordinal, life in enumerate(launches, 1):
                if life.name != f"launch-{ordinal:02d}":
                    raise RuntimeError("canary launch order differs")
                module.exact_directory(life, label="canary launch")
                log_path = module.exact_plain_file(
                    life / "torchrun.log", label="canary torchrun log", mode=0o444
                )
                log = module.read_ready_bytes(log_path, label="canary torchrun log")
                if b"EADDRINUSE" in log or b"Address already in use" in log:
                    raise RuntimeError("canary saw address collision")
                expected_rdzv = (
                    f"saic-{job}-{group}-c{index:02d}-{candidate_digest[:16]}-l{ordinal:02d}"
                )
                entries = sorted(item.name for item in life.iterdir())
                if "operational-completion.json" in entries:
                    if entries != [
                        "admission.json", "operational-completion.json",
                        "rank-0.json", "rank-1.json", "rank-2.json",
                        "rank-3.json", "torchrun.log",
                    ] or ordinal != len(launches) or completion is not None:
                        raise RuntimeError("canary success evidence closure differs")
                    completion = module.wait_load_sealed(
                        life / "operational-completion.json",
                        schema_version=(
                            "saic-r6-dynamic-rendezvous-operational-completion-v2"
                        ),
                        exact_fields=COMPLETION_FIELDS,
                        label="canary terminal completion",
                    )
                    completion_life = life
                elif "collision.json" in entries:
                    if entries != ["collision.json", "torchrun.log"] or ordinal == len(launches):
                        raise RuntimeError("canary collision evidence closure differs")
                    collision = module._validate_collision_for_job_audit(
                        life / "collision.json", claim_root=claims, slurm_job_id=job
                    )
                    prior_claim_path = (
                        claims / f"port-{collision['actual_master_port']}.json"
                    )
                    prior_claim = module.wait_load_sealed(
                        prior_claim_path,
                        schema_version=module.CLAIM_SCHEMA_VERSION,
                        exact_fields=module.CLAIM_FIELDS,
                        label="canary collision prior claim",
                    )
                    if (
                        collision.get("status")
                        != "kernel_port_already_claimed_in_this_job_before_runtime"
                        or collision.get("slurm_job_id") != job
                        or collision.get("group_id") != group
                        or collision.get("candidate_index") != index
                        or collision.get("candidate_id") != candidate_id
                        or collision.get("launch_ordinal") != ordinal
                        or collision.get("rdzv_id") != expected_rdzv
                        or collision.get("authority") != module.AUTHORITY
                        or all(
                            prior_claim.get(key) == collision.get(key)
                            for key in (
                                "group_id", "candidate_index", "candidate_id"
                            )
                        )
                    ):
                        raise RuntimeError("canary collision identity differs")
                    collision_prior_claims.add(prior_claim_path)
                    all_run_ids.append(collision["rdzv_id"])
                    collision_count += 1
                else:
                    raise RuntimeError("canary launch terminal evidence differs")
            if completion is None or completion_life != launches[-1]:
                raise RuntimeError("canary completion placement differs")
            port = completion.get("actual_master_port")
            expected_rdzv = (
                f"saic-{job}-{group}-c{index:02d}-{candidate_digest[:16]}-l{len(launches):02d}"
            )
            claim_path = claims / f"port-{port}.json"
            if (
                completion.get("status")
                != "unchanged_runtime_import_and_help_completed_after_exact_world4_admission"
                or completion.get("slurm_job_id") != job
                or completion.get("group_id") != group
                or completion.get("candidate_index") != index
                or completion.get("candidate_id") != candidate_id
                or completion.get("launch_ordinal") != len(launches)
                or completion.get("rdzv_id") != expected_rdzv
                or not isinstance(port, int)
                or not 1024 <= port <= 65535
                or port in ports_legacy
                or completion.get("claim_path") != str(claim_path)
                or not isinstance(completion.get("rank_packet_digests"), list)
                or len(completion["rank_packet_digests"]) != module.WORLD_SIZE
                or any(module.SHA256.fullmatch(item) is None for item in completion["rank_packet_digests"])
                or completion.get("runtime_sha256") != module.EXPECTED_RUNTIME_SHA256
                or completion.get("torch_disable_share_rdzv_tcp_store") != "0"
                or completion.get("shared_tcp_store_bootstrap") is not True
                or completion.get("receipt_publication_protocol")
                != "open_first_inode_pinned_terminal_fchmod_v2"
                or completion.get("scientific_generation_entered") is not False
                or completion.get("scientific_output_created") is not False
                or completion.get("authority") != module.AUTHORITY
            ):
                raise RuntimeError("canary completion content differs")
            claim = module.wait_load_sealed(
                claim_path,
                schema_version=module.CLAIM_SCHEMA_VERSION,
                exact_fields=module.CLAIM_FIELDS,
                label="canary terminal claim",
            )
            module._validate_claim(claim, port=port)
            decision = module._validate_generic_admission(
                claim, require_rank_packets=True
            )
            identity_fields = (
                "slurm_job_id", "group_id", "candidate_index", "candidate_id",
                "launch_ordinal", "rdzv_id", "actual_master_port",
            )
            if (
                any(completion.get(key) != claim.get(key) for key in identity_fields)
                or any(completion.get(key) != decision.get(key) for key in identity_fields)
                or claim.get("lifecycle_dir") != str(completion_life)
                or claim.get("admission_receipt_path")
                != str(completion_life / "admission.json")
                or completion.get("claim_receipt_digest") != claim["receipt_digest"]
                or completion.get("rank_packet_digests")
                != decision["rank_packet_digests"]
                or completion.get("admission_receipt_digest")
                != decision["receipt_digest"]
            ):
                raise RuntimeError("canary completion linkage differs")
            referenced_claims.add(claim_path)
            rows.append(
                {
                    "group_id": group,
                    "candidate_index": index,
                    "candidate_id": candidate_id,
                    "successful_launch_ordinal": len(launches),
                    "rdzv_id": expected_rdzv,
                    "actual_master_port": port,
                    "claim_receipt_digest": claim["receipt_digest"],
                    "admission_receipt_digest": decision["receipt_digest"],
                    "completion_receipt_digest": completion["receipt_digest"],
                }
            )
            all_run_ids.append(expected_rdzv)

    claim_paths = set(claims.iterdir())
    expected_claim_paths = {
        claims / f"port-{row['actual_master_port']}.json" for row in rows
    }
    if (
        len(rows) != 60
        or {row["candidate_index"] for row in rows if row["group_id"] == "sp4-a"}
        != set(range(30))
        or {row["candidate_index"] for row in rows if row["group_id"] == "sp4-b"}
        != set(range(30))
        or len({row["candidate_id"] for row in rows}) != 60
        or len({row["rdzv_id"] for row in rows}) != 60
        or len({row["actual_master_port"] for row in rows}) != 60
        or len(claim_paths) != 60
        or claim_paths != referenced_claims
        or claim_paths != expected_claim_paths
        or not collision_prior_claims.issubset(referenced_claims)
        or len(all_run_ids) != len(set(all_run_ids))
    ):
        raise RuntimeError("canary exact60 uniqueness/claim closure differs")

    rows.sort(key=lambda row: (row["group_id"], row["candidate_index"]))
    core = {
        "schema_version": "saic-r6-dynamic-rendezvous-all8-canary-v2",
        "status": "operational_dynamic_rendezvous_exact60_pass",
        "slurm_job_id": job,
        "node": node,
        "listener_node": listener_value["node"],
        "topology": "two_concurrent_world4_groups",
        "requested_gpu_count": 8,
        "candidate_count": 60,
        "completion_receipt_count": 60,
        "rank_packet_count": 240,
        "group_candidate_counts": {"sp4-a": 30, "sp4-b": 30},
        "unique_actual_master_port_count": 60,
        "permanent_claim_count": 60,
        "collision_receipt_count": collision_count,
        "all_launch_rdzv_id_count": len(all_run_ids),
        "all_launch_rdzv_ids_unique": True,
        "torch_disable_share_rdzv_tcp_store": "0",
        "shared_tcp_store_bootstrap": True,
        "receipt_publication_protocol": (
            "open_first_inode_pinned_terminal_fchmod_v2"
        ),
        "legacy_listener_receipt_sha256": hashlib.sha256(listener_raw).hexdigest(),
        "legacy_listener_receipt_digest": listener_value["receipt_digest"],
        "legacy_listener_count": 60,
        "legacy_port_range_inclusive": [48730, 48789],
        "legacy_listener_held_through_all_torchruns": True,
        "legacy_listener_clean_shutdown_before_receipt": True,
        "actual_ports_disjoint_from_legacy_ports": True,
        "eaddrinuse_observed": False,
        "guard_sha256": guard_sha,
        "source_archive_sha256": archive_sha,
        "python_sha256": python_sha,
        "launcher_sha256": launcher_sha,
        "executed_launcher_sha256": launcher_sha,
        "submission_receipt_path": str(submission_receipt_path),
        "submission_receipt_sha256": hashlib.sha256(
            submission_raw
        ).hexdigest(),
        "submission_receipt_digest": submission_value["receipt_digest"],
        "submission_job_id": submitted["job_id"],
        "runtime_sha256": module.EXPECTED_RUNTIME_SHA256,
        "runtime_action": "import_and_argparse_help_only",
        "scientific_generation_entered": False,
        "scientific_output_created": False,
        "candidate_rows": rows,
        "authority": module.AUTHORITY,
    }
    module.write_create_only(success_path, module.seal(core))
    os._exit(0)
except BaseException as error:
    try:
        if not success_path.exists() and not failure_path.exists():
            failure_core = {
                "schema_version": (
                    "saic-r6-dynamic-rendezvous-all8-canary-failure-v2"
                ),
                "status": "terminal_deep_audit_failed_without_success",
                "slurm_job_id": job,
                "exit_status": 2,
                "error_type": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8", "backslashreplace")
                ).hexdigest(),
                "guard_sha256": guard_sha,
                "source_archive_sha256": archive_sha,
                "python_sha256": python_sha,
                "launcher_sha256": launcher_sha,
                "executed_launcher_sha256": launcher_sha,
                "submission_receipt_path": str(submission_receipt_path),
                "runtime_sha256": module.EXPECTED_RUNTIME_SHA256,
                "scientific_generation_entered": False,
                "scientific_output_created": False,
                "success_claimed": False,
                "authority": module.AUTHORITY,
            }
            module.write_create_only(failure_path, module.seal(failure_core))
    except BaseException:
        pass
    os._exit(2)
PY
