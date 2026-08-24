#!/usr/bin/env bash
# Dispatch all 32 contiguous full-motion Wan shards across eight verified-idle
# nodes of one existing Slurm allocation.  Run this controller under external
# nohup; it deliberately remains foreground and returns nonzero if any shard
# fails, after allowing every other node worker to finish its assigned shards.
# Existing allocation-holder steps are preserved and overlapped only after two
# physical zero-VRAM audits on every requested node.

set -Eeuo pipefail
umask 077

job_id="${MOTIVE_EXISTING_SLURM_JOB_ID:?set MOTIVE_EXISTING_SLURM_JOB_ID}"
nodes_csv="${MOTIVE_FULL_MOTION_WAN_NODES:?set eight comma-separated nodes}"
snapshot="${MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT:?set MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT}"
shard_manifest_dir="${MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR:?set MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR}"
root_release="${MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE:?set MOTIVE_FULL_MOTION_ROOT_SIGNED_RELEASE}"
wan_code_root="${MOTIVE_WAN22_CODE_ROOT:?set MOTIVE_WAN22_CODE_ROOT}"
checkpoint_dir="${MOTIVE_WAN22_CKPT_DIR:?set MOTIVE_WAN22_CKPT_DIR}"
python_bin="${MOTIVE_WAN22_PYTHON_BIN:?set MOTIVE_WAN22_PYTHON_BIN}"
ffprobe_bin="${MOTIVE_WAN22_FFPROBE_BIN:?set MOTIVE_WAN22_FFPROBE_BIN}"
output_root="${MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT:?set MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT}"
step_cpus="${MOTIVE_FULL_MOTION_WAN_STEP_CPUS:-120}"
idle_probe_interval="${MOTIVE_FULL_MOTION_WAN_IDLE_PROBE_INTERVAL_SECONDS:-5}"
frame_num="${MOTIVE_WAN22_FRAME_NUM:-81}"
sample_steps="${MOTIVE_WAN22_SAMPLE_STEPS:-40}"
sample_shift="${MOTIVE_WAN22_SAMPLE_SHIFT:-5.0}"
size="${MOTIVE_WAN22_SIZE:-1280*720}"
base_seed="${MOTIVE_WAN22_BASE_SEED:-260730}"
code_root="${snapshot}/methods/motive"
runner_script="${snapshot}/methods/motive/scripts/auh_wan22_i2v_full.sbatch"
wan_shards_root="${output_root}/wan_shards"
logs_root="${output_root}/logs"
state_root="${output_root}/state"
controller_done="${output_root}/dispatch_complete.json"

fail() {
  echo "[full-motion-wan-dispatch] $*" >&2
  exit 2
}

require_absolute() {
  local label="$1"
  local value="$2"
  [[ "${value}" == /* && "${value}" != "/" ]] \
    || fail "${label} must be a non-root absolute path: ${value}"
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

[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid existing job ID"
[[ "${step_cpus}" =~ ^[1-9][0-9]*$ ]] || fail "invalid step CPU count"
[[ "${idle_probe_interval}" =~ ^[0-9]+$ ]] || fail "invalid idle probe interval"
[[ "${frame_num}" == "81" ]] || fail "full-motion generation requires exactly 81 frames"
[[ "${sample_steps}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sample steps"
[[ "${sample_shift}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "invalid sample shift"
[[ "${base_seed}" =~ ^[0-9]+$ ]] || fail "invalid base seed"
case "${size}" in
  1280\*720|720\*1280|832\*480|480\*832) ;;
  *) fail "unsupported Wan size: ${size}" ;;
esac

for binding in \
  "snapshot:${snapshot}" \
  "shard_manifest_dir:${shard_manifest_dir}" \
  "root_release:${root_release}" \
  "wan_code_root:${wan_code_root}" \
  "checkpoint_dir:${checkpoint_dir}" \
  "python_bin:${python_bin}" \
  "ffprobe_bin:${ffprobe_bin}" \
  "output_root:${output_root}"; do
  require_absolute "${binding%%:*}" "${binding#*:}"
done

IFS=, read -r -a nodes <<<"${nodes_csv}"
(( ${#nodes[@]} == 8 )) || fail "exactly eight nodes are required"
declare -A seen_nodes=()
for node in "${nodes[@]}"; do
  [[ "${node}" =~ ^auh[0-9A-Za-z-]+$ ]] || fail "invalid node: ${node}"
  [[ -z "${seen_nodes[${node}]:-}" ]] || fail "duplicate node: ${node}"
  seen_nodes["${node}"]=1
done

require_plain_directory "source snapshot" "${snapshot}"
require_plain_directory "shard manifest directory" "${shard_manifest_dir}"
require_plain_directory "Wan code root" "${wan_code_root}"
require_plain_directory "Wan checkpoint" "${checkpoint_dir}"
for path in \
  "${snapshot}/SOURCE_FILES.jsonl" \
  "${runner_script}" \
  "${code_root}/motive/wan22_i2v_batch.py" \
  "${code_root}/motive/wan22_signed_release.py" \
  "${code_root}/motive/wan22_full_motion_signed_release.py" \
  "${code_root}/motive/goku_full_motion_finalize.py" \
  "${shard_manifest_dir}/summary.json" \
  "${shard_manifest_dir}/done.json" \
  "${shard_manifest_dir}/jobs.tsv" \
  "${root_release}" \
  "${wan_code_root}/generate.py" \
  "${wan_code_root}/wan/image2video.py" \
  "${checkpoint_dir}/Wan2.1_VAE.pth" \
  "${checkpoint_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
  "${checkpoint_dir}/high_noise_model/config.json" \
  "${checkpoint_dir}/high_noise_model/diffusion_pytorch_model.safetensors.index.json" \
  "${checkpoint_dir}/low_noise_model/config.json" \
  "${checkpoint_dir}/low_noise_model/diffusion_pytorch_model.safetensors.index.json"; do
  require_plain_file "required runtime input" "${path}"
done
[[ -x "${python_bin}" ]] || fail "Wan Python is not executable"
require_plain_file "ffprobe" "${ffprobe_bin}"
[[ -x "${ffprobe_bin}" ]] || fail "ffprobe is not executable"
[[ ! -L "${wan_code_root}/.git" && -d "${wan_code_root}/.git" ]] \
  || fail "Wan checkout lacks a plain .git directory"
[[ ! -L "${checkpoint_dir}/.cache/huggingface/download" \
  && -d "${checkpoint_dir}/.cache/huggingface/download" ]] \
  || fail "checkpoint download closure is missing"
for command in scontrol squeue srun git flock awk grep seq; do
  command -v "${command}" >/dev/null || fail "required command is unavailable: ${command}"
done
actual_wan_commit="$(git -C "${wan_code_root}" rev-parse HEAD)"
[[ "${actual_wan_commit}" == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc" ]] \
  || fail "official Wan checkout commit differs: ${actual_wan_commit}"

shard_closure_check_py='import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
def die(x): raise SystemExit(x)
def read(p):
    if p.is_symlink() or not p.is_file(): die("not a plain file: "+str(p))
    return p.read_bytes()
def obj(raw):
    return json.loads(raw.decode("utf-8"), parse_constant=lambda x: die("nonfinite"))
def canon(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
if root.is_symlink() or not root.is_dir(): die("bad shard root")
if {p.name for p in root.iterdir()} != {"shards","jobs.tsv","summary.json","done.json"}: die("top closure")
if (root/"shards").is_symlink() or not (root/"shards").is_dir(): die("bad shards dir")
summary_raw=read(root/"summary.json"); done_raw=read(root/"done.json")
summary=obj(summary_raw); done=obj(done_raw)
if summary.get("schema_version") != "motive-goku-full-motion-shard-manifest-v1" or summary.get("status") != "complete": die("summary identity")
if done.get("schema_version") != "motive-goku-full-motion-shard-manifest-done-v1" or done.get("status") != "complete": die("done identity")
done_payload=dict(done); done_digest=done_payload.pop("done_digest",None)
if done_digest != sha(canon(done_payload)): die("done digest")
if done.get("artifact_digest") != sha(canon(done.get("artifacts"))): die("artifact digest")
if done.get("source") != summary.get("source") or done.get("input_digest") != summary.get("input_digest"): die("source binding")
if done.get("implementation") != summary.get("implementation") or done.get("implementation_digest") != summary.get("implementation_digest"): die("implementation binding")
layout=summary.get("layout",{})
if layout.get("root_rows") != 256 or layout.get("rows_per_shard") != 8 or layout.get("shard_count") != 32 or layout.get("complete_nonoverlapping_coverage") is not True: die("layout")
descriptors=summary.get("shards")
if not isinstance(descriptors,list) or len(descriptors) != 32: die("descriptors")
if summary.get("shards_digest") != sha(canon(descriptors)): die("shards digest")
artifacts=done.get("artifacts")
if not isinstance(artifacts,dict) or set(artifacts) != {"jobs.tsv","summary.json",*[f"shards/shard_{i:03d}.jsonl" for i in range(32)]}: die("artifacts")
all_raw=[]
for i,d in enumerate(descriptors):
    rel=f"shards/shard_{i:03d}.jsonl"; raw=read(root/rel); lines=raw.splitlines()
    if d.get("shard_index") != i or d.get("path") != rel or d.get("root_row_start_zero_based") != i*8 or d.get("root_row_end_exclusive") != i*8+8 or d.get("root_row_indices_zero_based") != list(range(i*8,i*8+8)): die("indices")
    if d.get("rows") != 8 or len(lines) != 8 or d.get("bytes") != len(raw) or d.get("sha256") != sha(raw): die("shard bytes")
    rows=[obj(line) for line in lines]; iids=[x.get("iid") for x in rows]
    if iids != d.get("ordered_iids"): die("iid order")
    ordered=lambda xs: sha(b"".join(str(x).encode()+b"\n" for x in xs))
    if d.get("ordered_iids_sha256") != ordered(iids) or d.get("ordered_row_sha256") != ordered([sha(canon(x)) for x in rows]): die("ordered digest")
    meta=artifacts.get(rel,{})
    if meta != {"sha256":sha(raw),"bytes":len(raw),"rows":8}: die("shard artifact")
    all_raw.append(raw)
jobs_raw=read(root/"jobs.tsv")
if len(jobs_raw.splitlines()) != 33 or summary.get("jobs",{}).get("sha256") != sha(jobs_raw): die("jobs")
if artifacts.get("jobs.tsv") != {"sha256":sha(jobs_raw),"bytes":len(jobs_raw),"rows":32}: die("jobs artifact")
if artifacts.get("summary.json") != {"sha256":sha(summary_raw),"bytes":len(summary_raw),"rows":1}: die("summary artifact")
source=summary.get("source",{}); primary=Path(source.get("primary_path","")); primary_raw=read(primary)
if b"".join(all_raw) != primary_raw or source.get("primary_sha256") != sha(primary_raw) or source.get("primary_bytes") != len(primary_raw) or source.get("primary_rows") != 256: die("root reconstruction")'

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${code_root}" \
  "${python_bin}" -c "${shard_closure_check_py}" "${shard_manifest_dir}" \
  || fail "shard-manifest closure validation failed"

# Verify all 32 shard authorizations before reserving a GPU step.  This is only
# preflight evidence; wan22_i2v_batch repeats the signed authorization check.
for shard_index in $(seq 0 31); do
  shard="${shard_manifest_dir}/shards/shard_$(printf '%03d' "${shard_index}").jsonl"
  require_plain_file "generation shard" "${shard}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="${code_root}" \
    "${python_bin}" -m motive.wan22_full_motion_signed_release verify \
      --release "${root_release}" --manifest "${shard}" >/dev/null \
    || fail "root release does not authorize shard ${shard_index}"
done

job_record="$(scontrol show job -o "${job_id}")"
[[ "${job_record}" == *"JobState=RUNNING"* ]] || fail "allocation is not running"
[[ "${job_record}" == *"gres/gpu:mi210=64"* ]] \
  || fail "allocation does not contain exactly 64 MI210 GPUs"
allocation_nodelist="$(squeue -j "${job_id}" -h -o '%N')"
[[ -n "${allocation_nodelist}" ]] || fail "allocation node list is unavailable"
mapfile -t allocated_nodes < <(scontrol show hostnames "${allocation_nodelist}")
(( ${#allocated_nodes[@]} == 8 )) || fail "allocation does not contain exactly eight nodes"
for node in "${nodes[@]}"; do
  printf '%s\n' "${allocated_nodes[@]}" | grep -Fxq "${node}" \
    || fail "node is outside allocation: ${node}"
done
for node in "${allocated_nodes[@]}"; do
  [[ -n "${seen_nodes[${node}]:-}" ]] || fail "allocation contains unrequested node: ${node}"
done
mapfile -t existing_steps < <(squeue -s -j "${job_id}" -h -o '%i')
if (( ${#existing_steps[@]} > 0 )); then
  printf '[full-motion-wan-dispatch] preserving existing steps:'
  printf ' %s' "${existing_steps[@]}"
  printf '\n'
fi

check_idle_node() {
  local node="$1"
  # Holder steps reserve the allocation's complete memory TRES.  mem=0 lets
  # this overlapping probe share that reservation; GPU/PID checks remain the
  # fail-closed admission authority.
  srun --overlap --jobid="${job_id}" \
    --nodelist="${node}" --nodes=1 \
    --ntasks=1 --cpus-per-task=1 --mem=0 \
    bash -lc '
      set -Eeuo pipefail
      metrics=$(mktemp)
      processes=$(mktemp)
      trap '\''rm -f -- "$metrics" "$processes"'\'' EXIT
      rocm-smi --showuse --showmemuse --showmeminfo vram --csv \
        >"$metrics" 2>/dev/null
      awk -F, '\''
        NR == 1 {
          for (column_number=1; column_number<=NF; column_number++) {
            label=$column_number
            gsub(/^[ "\t]+|[ "\t\r]+$/, "", label)
            if (label == "GPU use (%)") use_index=column_number
            if (label == "GPU Memory Allocated (VRAM%)") percent_index=column_number
            if (label == "VRAM Total Used Memory (B)") used_index=column_number
          }
          if (!use_index || !percent_index || !used_index) bad=1
          next
        }
        /^card[0-7],/ {
          card=$1
          if (seen_card[card]++) bad=1
          seen++
          use_value=$(use_index)
          percent_value=$(percent_index)
          used_value=$(used_index)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", use_value)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", percent_value)
          gsub(/^[ "\t]+|[ "\t\r]+$/, "", used_value)
          if (use_value !~ /^[0-9]+$/ || percent_value !~ /^[0-9]+$/ || used_value !~ /^[0-9]+$/) bad=1
          if ((use_value+0) != 0 || (percent_value+0) != 0 || (used_value+0) > 1073741824) bad=1
        }
        END { exit !(seen == 8 && !bad) }
      '\'' "$metrics"
      rocm-smi --showpids --csv >"$processes" 2>/dev/null
      awk -F, '\''
        NR == 1 { next }
        /^[[:space:]]*$/ { next }
        {
          gpu_flag=$3
          vram=$4
          gsub(/[ "\r]/, "", gpu_flag)
          gsub(/[ "\r]/, "", vram)
          if (gpu_flag !~ /^[0-9]+$/ || vram !~ /^[0-9]+$/) bad=1
          if ((gpu_flag+0) != 0 || (vram+0) != 0) bad=1
        }
        END { exit bad ? 1 : 0 }
      '\'' "$processes"
    '
}
for audit in 1 2; do
  for node in "${nodes[@]}"; do
    check_idle_node "${node}" \
      || fail "node ${node} failed idle GPU audit ${audit}/2"
  done
  (( audit == 2 || idle_probe_interval == 0 )) || sleep "${idle_probe_interval}"
done

if [[ -e "${output_root}" || -L "${output_root}" ]]; then
  require_plain_directory "resume output root" "${output_root}"
else
  [[ ! -L "${output_root%/*}" && -d "${output_root%/*}" \
    && -w "${output_root%/*}" ]] || fail "output parent is unavailable"
  mkdir "${output_root}"
fi
for directory in "${wan_shards_root}" "${logs_root}" "${state_root}"; do
  if [[ -e "${directory}" || -L "${directory}" ]]; then
    require_plain_directory "dispatcher subdirectory" "${directory}"
  else
    mkdir "${directory}"
  fi
done
exec 9>"${output_root}/dispatcher.lock"
flock -n 9 || fail "another dispatcher holds the output lock"

completion_check_py='import hashlib,json,sys
from pathlib import Path
manifest=Path(sys.argv[1]); root=Path(sys.argv[2])
def die(x): raise SystemExit(x)
def read(p):
    if p.is_symlink() or not p.is_file(): die("not plain: "+str(p))
    return p.read_bytes()
def load(p): return json.loads(read(p).decode("utf-8"),parse_constant=lambda x:die("nonfinite"))
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
manifest_raw=read(manifest)
if not manifest_raw.endswith(b"\n") or len(manifest_raw.splitlines()) != 8: die("manifest rows")
manifest_rows=[json.loads(x) for x in manifest_raw.splitlines()]; manifest_iids=[x.get("iid") for x in manifest_rows]
contract=load(root/"run_contract.json"); contract_payload=dict(contract); contract_digest=contract_payload.pop("contract_digest",None)
if contract.get("schema_version") != "motive-wan22-i2v-batch-run-v1" or contract_digest != sha(canon(contract_payload)): die("contract digest")
bound=contract.get("manifest",{})
if bound.get("sha256") != sha(manifest_raw) or bound.get("bytes") != len(manifest_raw) or bound.get("row_count") != 8 or bound.get("selected_row_count") != 8 or bound.get("max_samples") != 8: die("manifest binding")
selected=contract.get("selected_inputs")
if not isinstance(selected,list) or [x.get("iid") for x in selected] != manifest_iids: die("selected order")
if contract.get("distributed_execution",{}).get("world_size") != 8 or contract.get("distributed_execution",{}).get("max_new_samples_per_allocation") != 8: die("distributed contract")
if contract.get("generation_parameters",{}).get("frame_num") != 81: die("frame contract")
if contract.get("authorization",{}).get("mode") != "sshsig_full_motion_root_contiguous8_release_v3": die("authorization mode")
complete=load(root/"run_complete.json"); payload=dict(complete); complete_digest=payload.pop("complete_digest",None)
if complete.get("schema_version") != "motive-wan22-i2v-batch-complete-v1" or complete_digest != sha(canon(payload)): die("complete digest")
if complete.get("contract_digest") != contract_digest or complete.get("manifest_sha256") != sha(manifest_raw) or complete.get("selected_sample_count") != 8 or complete.get("completed_sample_count") != 8: die("complete binding")
if complete.get("generated_manifest") != "generated_manifest.jsonl" or complete.get("temporal_policy") != contract.get("temporal_policy"): die("complete policy")
generated_raw=read(root/"generated_manifest.jsonl")
if complete.get("generated_manifest_sha256") != sha(generated_raw): die("generated hash")
generated=[json.loads(x) for x in generated_raw.splitlines()]
if len(generated) != 8 or [x.get("iid") for x in generated] != manifest_iids: die("generated order")
result_digests=[]
for iid,item in zip(manifest_iids,generated):
    if item.get("schema_version") != "motive-wan22-i2v-generated-target-v1": die("generated schema")
    result_path=Path(item.get("result_json","")); expected=root/"samples"/iid/"result.json"
    if result_path != expected: die("result path")
    result=load(result_path); result_payload=dict(result); result_digest=result_payload.pop("result_digest",None)
    if result_digest != sha(canon(result_payload)) or result_digest != item.get("result_digest"): die("result digest")
    if result.get("iid") != iid or result.get("contract_digest") != contract_digest or result.get("manifest_sha256") != sha(manifest_raw): die("result binding")
    outputs=result.get("outputs",{})
    for name,hash_name in (("preview_mp4","preview_mp4_sha256"),("conditioning_anchor_original","conditioning_anchor_original_sha256"),("conditioning_frame0_float32","conditioning_frame0_float32_sha256"),("conditioning_frame0_png","conditioning_frame0_png_sha256")):
        artifact=root/"samples"/iid/str(outputs.get(name,"")); raw=read(artifact)
        if outputs.get(hash_name) != sha(raw): die("sample artifact hash")
    preview=Path(item.get("target_preview_mp4",""))
    if preview != root/"samples"/iid/str(outputs.get("preview_mp4")) or item.get("target_preview_mp4_sha256") != outputs.get("preview_mp4_sha256"): die("preview binding")
    result_digests.append(result_digest)
if complete.get("sample_result_digests") != result_digests: die("result order")'

verify_completed_shard() {
  local shard_index="$1"
  local manifest="${shard_manifest_dir}/shards/shard_$(printf '%03d' "${shard_index}").jsonl"
  local shard_output="${wan_shards_root}/shard_$(printf '%03d' "${shard_index}")"
  [[ ! -L "${shard_output}" && -d "${shard_output}" ]] || return 1
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${python_bin}" -c "${completion_check_py}" "${manifest}" "${shard_output}"
}

verify_all_completed() {
  local shard_index
  for shard_index in $(seq 0 31); do
    verify_completed_shard "${shard_index}" || return 1
  done
}

dispatch_done_check_py='import hashlib,json,sys
from pathlib import Path
target=Path(sys.argv[1]); output=Path(sys.argv[2]); shard_root=Path(sys.argv[3]); release=Path(sys.argv[4]); job=sys.argv[5]; nodes=sys.argv[6].split(",")
if target.is_symlink() or not target.is_file(): raise SystemExit("bad dispatcher completion")
value=json.loads(target.read_text(encoding="utf-8"))
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload=dict(value); digest=payload.pop("complete_digest",None)
if value.get("schema_version") != "motive-full-motion-wan-existing-allocation-dispatch-v1" or value.get("status") != "complete" or digest != hashlib.sha256(canon(payload)).hexdigest(): raise SystemExit("dispatcher completion digest")
if value.get("slurm_job_id") != job or value.get("nodes") != nodes or value.get("shard_manifest_dir") != str(shard_root) or value.get("root_signed_release") != str(release) or value.get("output_root") != str(output): raise SystemExit("dispatcher completion binding")
if value.get("shard_manifest_done_sha256") != sha(shard_root/"done.json") or value.get("root_signed_release_sha256") != sha(release): raise SystemExit("dispatcher input hash")
completed=value.get("completed_shards")
if not isinstance(completed,list) or len(completed) != 32: raise SystemExit("dispatcher completion count")
for i,item in enumerate(completed):
    root=output/"wan_shards"/f"shard_{i:03d}"
    if item != {"shard_index":i,"run_complete_sha256":sha(root/"run_complete.json"),"generated_manifest_sha256":sha(root/"generated_manifest.jsonl")}: raise SystemExit("dispatcher shard hash")'

if [[ -e "${controller_done}" || -L "${controller_done}" ]]; then
  require_plain_file "dispatcher completion" "${controller_done}"
  "${python_bin}" -c "${dispatch_done_check_py}" \
    "${controller_done}" "${output_root}" "${shard_manifest_dir}" \
    "${root_release}" "${job_id}" "${nodes_csv}" \
    || fail "dispatcher completion receipt is invalid"
  verify_all_completed || fail "dispatcher completion exists but a shard is invalid"
  echo "[full-motion-wan-dispatch] already complete: ${controller_done}"
  exit 0
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code_root}:${snapshot}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SLURM_EXPORT_ENV=ALL

run_node_worker() {
  local node_rank="$1"
  local node="${nodes[${node_rank}]}"
  local worker_state="${state_root}/worker_$(printf '%02d' "${node_rank}").tsv"
  local worker_temporary="${worker_state}.tmp.$$"
  local worker_failed=0
  printf 'shard_index\tnode\tstatus\tlog\n' >"${worker_temporary}"
  local shard_index
  for (( shard_index=node_rank; shard_index<32; shard_index+=8 )); do
    local stem
    stem="shard_$(printf '%03d' "${shard_index}")"
    local manifest="${shard_manifest_dir}/shards/${stem}.jsonl"
    local shard_output="${wan_shards_root}/${stem}"
    local shard_log="${logs_root}/${stem}.log"
    if [[ -e "${shard_output}/run_complete.json" || -L "${shard_output}/run_complete.json" ]]; then
      if verify_completed_shard "${shard_index}" >>"${shard_log}" 2>&1; then
        printf '%s\t%s\tskipped_verified_complete\t%s\n' \
          "${shard_index}" "${node}" "${shard_log}" >>"${worker_temporary}"
        continue
      fi
      printf '%s\t%s\tinvalid_existing_completion\t%s\n' \
        "${shard_index}" "${node}" "${shard_log}" >>"${worker_temporary}"
      worker_failed=1
      continue
    fi
    if [[ -e "${shard_output}" || -L "${shard_output}" ]]; then
      if [[ -L "${shard_output}" || ! -d "${shard_output}" ]]; then
        printf '%s\t%s\tinvalid_resume_output\t%s\n' \
          "${shard_index}" "${node}" "${shard_log}" >>"${worker_temporary}"
        worker_failed=1
        continue
      fi
    fi
    {
      printf '[full-motion-wan-dispatch] start shard=%s node=%s utc=%s\n' \
        "${shard_index}" "${node}" "$(date -u +%FT%TZ)"
      # Exactly one Wan worker is admitted per 2-TiB node.  mem=0 is necessary
      # to overlap the existing all-memory holder step.
      if srun --overlap \
        --jobid="${job_id}" --nodelist="${node}" --nodes=1 \
        --exact \
        --ntasks=1 --cpus-per-task="${step_cpus}" --mem=0 \
        --gpus-per-task=8 --gpu-bind=none \
        /usr/bin/env \
          ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
          HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
          CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
          SLURM_EXPORT_ENV=ALL \
          MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT="${snapshot}" \
          MOTIVE_GOKU_ACTION_GENERATION_MANIFEST="${manifest}" \
          MOTIVE_WAN22_SIGNED_RELEASE="${root_release}" \
          MOTIVE_WAN22_CODE_ROOT="${wan_code_root}" \
          MOTIVE_WAN22_CKPT_DIR="${checkpoint_dir}" \
          MOTIVE_WAN22_PYTHON_BIN="${python_bin}" \
          MOTIVE_WAN22_FFPROBE_BIN="${ffprobe_bin}" \
          MOTIVE_WAN22_OUTPUT_ROOT="${shard_output}" \
          MOTIVE_WAN22_ALLOW_PENDING_REVIEW=0 \
          MOTIVE_WAN22_MAX_NEW_SAMPLES=8 \
          MOTIVE_WAN22_FRAME_NUM=81 \
          MOTIVE_WAN22_SAMPLE_STEPS="${sample_steps}" \
          MOTIVE_WAN22_SAMPLE_SHIFT="${sample_shift}" \
          MOTIVE_WAN22_SIZE="${size}" \
          MOTIVE_WAN22_BASE_SEED="${base_seed}" \
          bash "${runner_script}"; then
        if verify_completed_shard "${shard_index}"; then
          printf '[full-motion-wan-dispatch] complete shard=%s node=%s\n' \
            "${shard_index}" "${node}"
          printf '%s\t%s\tcompleted\t%s\n' \
            "${shard_index}" "${node}" "${shard_log}" >>"${worker_temporary}"
        else
          printf '[full-motion-wan-dispatch] invalid terminal artifacts shard=%s\n' \
            "${shard_index}" >&2
          printf '%s\t%s\tinvalid_terminal_artifacts\t%s\n' \
            "${shard_index}" "${node}" "${shard_log}" >>"${worker_temporary}"
          worker_failed=1
        fi
      else
        step_status=$?
        printf '[full-motion-wan-dispatch] srun failed shard=%s status=%s\n' \
          "${shard_index}" "${step_status}" >&2
        printf '%s\t%s\tsrun_failed_%s\t%s\n' \
          "${shard_index}" "${node}" "${step_status}" "${shard_log}" \
          >>"${worker_temporary}"
        worker_failed=1
      fi
    } >>"${shard_log}" 2>&1
  done
  mv "${worker_temporary}" "${worker_state}"
  return "${worker_failed}"
}

echo "[full-motion-wan-dispatch] start job=${job_id} nodes=${nodes_csv} shards=32"
worker_pids=()
for node_rank in 0 1 2 3 4 5 6 7; do
  run_node_worker "${node_rank}" &
  worker_pids+=("$!")
done
overall_status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    overall_status=1
  fi
done
if (( overall_status != 0 )); then
  echo "[full-motion-wan-dispatch] one or more workers failed; valid peers completed" >&2
  exit 1
fi
verify_all_completed || fail "workers returned success but terminal closure is incomplete"

dispatch_done_py='import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]); shard_root=Path(sys.argv[2]); release=Path(sys.argv[3]); job=sys.argv[4]; nodes=sys.argv[5]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
value={"schema_version":"motive-full-motion-wan-existing-allocation-dispatch-v1","status":"complete","completed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"slurm_job_id":job,"nodes":nodes.split(","),"shard_manifest_dir":str(shard_root),"shard_manifest_done_sha256":sha(shard_root/"done.json"),"root_signed_release":str(release),"root_signed_release_sha256":sha(release),"output_root":str(root),"assignments":[{"node_rank":r,"node":nodes.split(",")[r],"shard_indices":[r,r+8,r+16,r+24]} for r in range(8)],"completed_shards":[{"shard_index":i,"run_complete_sha256":sha(root/"wan_shards"/f"shard_{i:03d}"/"run_complete.json"),"generated_manifest_sha256":sha(root/"wan_shards"/f"shard_{i:03d}"/"generated_manifest.jsonl")} for i in range(32)]}
canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
value["complete_digest"]=hashlib.sha256(canon(value)).hexdigest()
raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
target=root/"dispatch_complete.json"
fd,name=tempfile.mkstemp(prefix=".dispatch_complete.",suffix=".tmp",dir=root)
try:
    with os.fdopen(fd,"wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.link(name,target)
finally:
    try: os.unlink(name)
    except FileNotFoundError: pass'
"${python_bin}" -c "${dispatch_done_py}" \
  "${output_root}" "${shard_manifest_dir}" "${root_release}" \
  "${job_id}" "${nodes_csv}"
echo "[full-motion-wan-dispatch] complete: ${controller_done}"
