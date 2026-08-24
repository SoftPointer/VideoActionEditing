#!/usr/bin/env bash
#SBATCH --job-name=saic-r6-rdzv-cny
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=00:30:00

set -Eeuo pipefail

archive="${SAIC_R6_CANARY_SOURCE_ARCHIVE:?}"
archive_sha="${SAIC_R6_CANARY_SOURCE_ARCHIVE_SHA256:?}"
guard="${SAIC_R6_CANARY_GUARD:?}"
guard_sha="${SAIC_R6_CANARY_GUARD_SHA256:?}"
python_bin="${SAIC_R6_CANARY_PYTHON:?}"
output_parent="${SAIC_R6_CANARY_OUTPUT_PARENT:?}"
readonly runtime_sha=3372f1f48b9cb235d269ee6352ad4f289a6ee4a4a781c69ce0f7b1862ce77d36
readonly maximum_launches=16

fail() { echo "[saic-r6-rdzv-canary] ERROR: $*" >&2; exit 2; }
for value in archive guard python_bin output_parent; do
  [[ "${!value}" == /* ]] || fail "${value} must be absolute"
done
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${guard_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "input SHA differs"
[[ "${SLURM_JOB_ID:?}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
archive="$(realpath -e -- "${archive}")"
guard="$(realpath -e -- "${guard}")"
python_bin="$(realpath -e -- "${python_bin}")"
output_parent="$(realpath -e -- "${output_parent}")"
[[ -f "${archive}" && ! -L "${archive}" && "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "archive differs"
[[ -f "${guard}" && ! -L "${guard}" && "$(sha256sum "${guard}" | awk '{print $1}')" == "${guard_sha}" ]] || fail "guard differs"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

output_root="${output_parent}/job-${SLURM_JOB_ID}"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "canary output must be fresh"
mkdir -m 0700 -- "${output_root}"
mkdir -m 0700 -- "${output_root}/logs" "${output_root}/rendezvous" \
  "${output_root}/rendezvous/port-claims" "${output_root}/forbidden-attempts"
claim_root="${output_root}/rendezvous/port-claims"
claim_identity="$(stat -Lc '%d:%i' -- "${claim_root}")"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-r6-rdzv-canary-${SLURM_JOB_ID}.XXXXXX")"
listener_pid=""
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
    case "${task_scratch}" in "${scratch_parent%/}/saic-r6-rdzv-canary-${SLURM_JOB_ID}."*) rm -rf -- "${task_scratch}" || true;; esac
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

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
"${python_bin}" -B - "${listener_ready}" "${listener_stop}" "${SLURM_JOB_ID}" <<'PY' \
  >"${output_root}/logs/legacy-listener.stdout" 2>"${output_root}/logs/legacy-listener.stderr" &
import hashlib, json, os
from pathlib import Path
import socket, sys, time
ready, stop, job_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
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
    "schema_version": "saic-r6-legacy-arithmetic-port-listener-v1",
    "status": "all_legacy_ports_listening_before_dynamic_canary",
    "slurm_job_id": job_id,
    "node": socket.gethostname(),
    "pid": os.getpid(),
    "address": "127.0.0.1",
    "port_range_inclusive": [48730, 48789],
    "listener_count": len(rows),
    "listeners": rows,
    "authority": {"scientific": False, "generation": False, "training": False},
}
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
value = {**core, "receipt_digest": hashlib.sha256(canonical(core)).hexdigest()}
payload = canonical(value) + b"\n"
fd = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
offset = 0
while offset < len(payload):
    wrote = os.write(fd, payload[offset:])
    if wrote <= 0: raise SystemExit("listener receipt write stalled")
    offset += wrote
os.fsync(fd); os.fchmod(fd, 0o444); os.fsync(fd); os.close(fd)
while not stop.exists(): time.sleep(0.02)
for item in sockets: item.close()
PY
listener_pid=$!
for _ in {1..500}; do
  if [[ -f "${listener_ready}" && "$(stat -Lc '%a' -- "${listener_ready}")" == 444 ]]; then break; fi
  kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener exited before ready"
  sleep 0.02
done
[[ -f "${listener_ready}" && "$(stat -Lc '%a' -- "${listener_ready}")" == 444 ]] || fail "legacy listener not ready"

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
decision = module.load_sealed(life / "admission.json", schema_version=module.DECISION_SCHEMA_VERSION, exact_fields=module.DECISION_FIELDS)
port = decision["actual_master_port"]
claim_path = claim_value / f"port-{port}.json"
claim = module.load_sealed(claim_path, schema_version=module.CLAIM_SCHEMA_VERSION, exact_fields=module.CLAIM_FIELDS)
module._validate_generic_admission(claim, require_rank_packets=True)
if any(decision.get(k) != v for k, v in {"slurm_job_id":job,"group_id":group,"candidate_index":int(index),"candidate_id":candidate,"launch_ordinal":int(ordinal),"rdzv_id":rdzv}.items()): raise SystemExit("canary admission identity differs")
if os.environ.get("TORCH_DISABLE_SHARE_RDZV_TCP_STORE") != "0": raise SystemExit("shared TCPStore environment differs")
core = {"schema_version":"saic-r6-dynamic-rendezvous-operational-completion-v1","status":"unchanged_runtime_import_and_help_completed_after_exact_world4_admission","slurm_job_id":job,"group_id":group,"candidate_index":int(index),"candidate_id":candidate,"launch_ordinal":int(ordinal),"rdzv_id":rdzv,"actual_master_port":port,"claim_path":str(claim_path),"claim_receipt_digest":claim["receipt_digest"],"rank_packet_digests":decision["rank_packet_digests"],"admission_receipt_digest":decision["receipt_digest"],"runtime_sha256":module.EXPECTED_RUNTIME_SHA256,"torch_disable_share_rdzv_tcp_store":"0","shared_tcp_store_bootstrap":True,"scientific_generation_entered":False,"scientific_output_created":False,"authority":module.AUTHORITY}
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
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "canary group failed dog=${dog_status} human=${human_status}"
kill -0 "${listener_pid}" 2>/dev/null || fail "legacy listener not alive through all launches"
touch "${listener_stop}"
wait "${listener_pid}" || fail "legacy listener shutdown failed"
listener_pid=""

rm -rf -- "${task_scratch}"
task_scratch=""

"${python_bin}" -B - "${guard}" "${output_root}" "${claim_root}" "${listener_ready}" \
  "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-$(hostname)}" "${guard_sha}" "${archive_sha}" <<'PY'
import hashlib, importlib.util, json, os
from pathlib import Path
import sys
guard_path, root, claims, listener = map(Path, sys.argv[1:5])
job, node, guard_sha, archive_sha = sys.argv[5:9]
spec = importlib.util.spec_from_file_location("canary_guard", guard_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
raw = module.read_ready_bytes(listener, label="legacy listener receipt")
listener_value = json.loads(raw.decode("ascii")); unsigned = dict(listener_value); digest = unsigned.pop("receipt_digest")
if digest != module.object_sha256(unsigned) or raw != module.canonical_json_bytes(listener_value)+b"\n": raise SystemExit("listener receipt differs")
ports_legacy = {row["port"] for row in listener_value["listeners"]}
if ports_legacy != set(range(48730,48790)) or len({row["socket_inode"] for row in listener_value["listeners"]}) != 60: raise SystemExit("legacy listeners differ")
rows=[]; all_run_ids=[]
for group in ("sp4-a","sp4-b"):
    group_root = module.exact_directory(root / "rendezvous" / group, label="canary group")
    candidates = sorted(group_root.iterdir())
    if len(candidates) != 30: raise SystemExit("canary candidate count differs")
    for index, candidate_root in enumerate(candidates):
        module.exact_directory(candidate_root, label="canary candidate")
        launches=sorted(candidate_root.iterdir())
        completions=[]
        for ordinal, life in enumerate(launches,1):
            if life.name != f"launch-{ordinal:02d}": raise SystemExit("canary launch order differs")
            module.exact_directory(life,label="canary launch")
            log=(life/"torchrun.log").read_bytes()
            if b"EADDRINUSE" in log or b"Address already in use" in log: raise SystemExit("canary saw address collision")
            expected=f"saic-{job}-{group}-c{index:02d}-{hashlib.sha256(f'canary-{group}-candidate-{index:02d}'.encode('ascii')).hexdigest()[:16]}-l{ordinal:02d}"
            if (life/"operational-completion.json").exists():
                if sorted(item.name for item in life.iterdir()) != ["admission.json","operational-completion.json","rank-0.json","rank-1.json","rank-2.json","rank-3.json","torchrun.log"]: raise SystemExit("canary success evidence closure differs")
                completions.append(life/"operational-completion.json")
            elif (life/"collision.json").exists():
                if sorted(item.name for item in life.iterdir()) != ["collision.json","torchrun.log"]: raise SystemExit("canary collision evidence closure differs")
                collision=module.load_sealed(life/"collision.json",schema_version=module.COLLISION_SCHEMA_VERSION,exact_fields=module.COLLISION_FIELDS)
                if collision["rdzv_id"] != expected or collision["launch_ordinal"] != ordinal: raise SystemExit("canary collision run ID differs")
                all_run_ids.append(collision["rdzv_id"])
            else: raise SystemExit("canary launch terminal evidence differs")
        if len(completions)!=1 or completions[0].parent != launches[-1]: raise SystemExit("canary completion differs")
        value=module.load_sealed(completions[0],schema_version="saic-r6-dynamic-rendezvous-operational-completion-v1")
        expected_success=f"saic-{job}-{group}-c{index:02d}-{hashlib.sha256(value['candidate_id'].encode('ascii')).hexdigest()[:16]}-l{len(launches):02d}"
        if value["group_id"]!=group or value["candidate_index"]!=index or value["launch_ordinal"]!=len(launches) or value["rdzv_id"]!=expected_success or value["actual_master_port"] in ports_legacy or value["torch_disable_share_rdzv_tcp_store"]!="0" or value["shared_tcp_store_bootstrap"] is not True or value["scientific_generation_entered"] is not False or value["scientific_output_created"] is not False: raise SystemExit("canary completion content differs")
        claim=module.load_sealed(Path(value["claim_path"]),schema_version=module.CLAIM_SCHEMA_VERSION,exact_fields=module.CLAIM_FIELDS)
        decision=module._validate_generic_admission(claim,require_rank_packets=True)
        if value["claim_receipt_digest"]!=claim["receipt_digest"] or value["rank_packet_digests"]!=decision["rank_packet_digests"] or value["admission_receipt_digest"]!=decision["receipt_digest"]: raise SystemExit("canary linkage differs")
        rows.append({"group_id":group,"candidate_index":index,"candidate_id":value["candidate_id"],"successful_launch_ordinal":value["launch_ordinal"],"rdzv_id":value["rdzv_id"],"actual_master_port":value["actual_master_port"],"completion_receipt_digest":value["receipt_digest"]})
        all_run_ids.append(value["rdzv_id"])
claims_set=set(claims.glob("port-*.json"))
if len(rows)!=60 or len({r["actual_master_port"] for r in rows})!=60 or len({r["rdzv_id"] for r in rows})!=60 or len(claims_set)!=60 or len(all_run_ids)!=len(set(all_run_ids)): raise SystemExit("canary exact60 uniqueness differs")
if any((root/"forbidden-attempts").iterdir()): raise SystemExit("canary created scientific output")
if os.environ.get("TORCH_DISABLE_SHARE_RDZV_TCP_STORE") != "0": raise SystemExit("shared TCPStore environment differs")
core={"schema_version":"saic-r6-dynamic-rendezvous-all8-canary-v1","status":"operational_dynamic_rendezvous_exact60_pass","slurm_job_id":job,"node":node,"topology":"two_concurrent_world4_groups","requested_gpu_count":8,"candidate_count":60,"rank_packet_count":240,"unique_actual_master_port_count":60,"permanent_claim_count":60,"all_launch_rdzv_id_count":len(all_run_ids),"all_launch_rdzv_ids_unique":True,"torch_disable_share_rdzv_tcp_store":"0","shared_tcp_store_bootstrap":True,"legacy_listener_receipt_sha256":hashlib.sha256(raw).hexdigest(),"legacy_listener_count":60,"legacy_port_range_inclusive":[48730,48789],"legacy_listener_held_through_all_torchruns":True,"legacy_listener_clean_shutdown_before_receipt":True,"actual_ports_disjoint_from_legacy_ports":True,"eaddrinuse_observed":False,"guard_sha256":guard_sha,"source_archive_sha256":archive_sha,"runtime_sha256":module.EXPECTED_RUNTIME_SHA256,"runtime_action":"import_and_argparse_help_only","scientific_generation_entered":False,"scientific_output_created":False,"candidate_rows":sorted(rows,key=lambda r:(r["group_id"],r["candidate_index"])),"authority":module.AUTHORITY}
module.write_create_only(root/"canary-receipt.json",module.seal(core))
PY

echo "SAIC_R6_DYNAMIC_RENDEZVOUS_ALL8_CANARY_OK output=${output_root}"
trap - EXIT TERM INT
exit 0
