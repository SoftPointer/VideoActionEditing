#!/bin/bash
# Sole node279 Slurm-step payload for one PRE_D0 Level-B P2 full render.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly capacity_python=/usr/bin/python3.10
readonly capacity_python_sha=11dde438e1a636073e79c81d4c2543708cc0a2922e7c42c38b1b588e17545f96
readonly env_bin=/usr/bin/env
readonly env_sha=85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0
readonly base64_bin=/usr/bin/base64
readonly base64_sha=b10f8c059f50c0681c6497e7b09ebdba168e341498ae1733de9089dc8efa0898
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v4
readonly release_root="${experiment_root}/releases/fresh-world8-level-b-p2-00435-v3"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly step_self="${launch_root}/auh_action_edit_level_b_p2_00435_step_v4.sh"
readonly rank_exec="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v4.sh"
readonly rank_exec_sha=64fc0df647ab28d950d81b6735aead559d1e91216416a8e44e8ac0c3707620c8
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v4.py"
readonly bootstrap_sha=1d72a1594ab52e258f0fbac5410ea1d27e5c557a12d76e88b806b3ac99794391
readonly capacity_member="${launch_root}/action_edit_level_b_p2_00435_capacity_0817_v4.py"
readonly capacity_member_sha=87fc10c580070eef660fdfeaecf18ddd997d031a009508edbcc34a263cd6c4dc
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=380b433d4be8c349bb79c8eb3914442136e153c2dccd4cb57ff25db9f7688a16
readonly renderer="${release_root}/infer_action_edit_level_b_renderer_0817_v1.py"
readonly renderer_sha=8e34d976481ed81e3b8b285253878f0c02bbfbe177ea608aa51b0f4b594bf1c6
readonly output_mp4="${run_root}/00435ad621c44fac_p2_seed2026080821_v4.mp4"
readonly source_mp4=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4
readonly source_sha=b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1

fail() {
  printf 'Level-B P2 node279 step refused: %s\n' "$*" >&2
  exit 96
}

[[ "${BASH_ENV:-}" == /dev/null ]] || fail "pre-script BASH_ENV boundary differs"
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C && "${LANG:-}" == C ]] || \
  fail "pre-script path or locale boundary differs"
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${ROCM_SMI_LIB_PATH:-}" ]] || \
  fail "pre-script inherited loader or Python environment differs"

require_stat_value() {
  local stat_path="$1"
  local stat_format="$2"
  local stat_expected="$3"
  local stat_label="$4"
  local stat_observed
  if ! stat_observed="$(/usr/bin/stat -c "${stat_format}" "${stat_path}")"; then
    fail "${stat_label}: stat query failed"
  fi
  [[ "${stat_observed}" == "${stat_expected}" ]] || fail "${stat_label} differs"
}

require_sha256() {
  local sha_path="$1"
  local sha_expected="$2"
  local sha_label="$3"
  local sha_observed
  if ! sha_observed="$(/usr/bin/sha256sum "${sha_path}" | /usr/bin/awk '{print $1}')"; then
    fail "${sha_label}: SHA query failed"
  fi
  [[ "${sha_observed}" =~ ^[0-9a-f]{64}$ ]] || fail "${sha_label}: SHA format differs"
  [[ "${sha_observed}" == "${sha_expected}" ]] || fail "${sha_label}: SHA differs"
}

capture_capacity_output_base64() {
  local transport_output_name="$1"
  local transport_label="$2"
  shift 2
  local transport_frame
  local transport_status
  local transport_suffix
  local transport_child_status
  local transport_encoder_status
  local transport_payload
  local transport_sentinel=__LEVEL_B_P2_00435_V4_CAPACITY_PIPESTATUS_
  if ! transport_frame="$({
    set +e
    "${env_bin}" -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
      HOME=/nonexistent/bernini-level-b-p2-00435-v4-capacity-command \
      "${capacity_python}" -I -S -B "${capacity_member}" "$@" 2>&1 | \
      "${base64_bin}" -w0
    transport_status=("${PIPESTATUS[@]}")
    printf '%s%03d_%03d__' "${transport_sentinel}" \
      "${transport_status[0]}" "${transport_status[1]}"
    exit 0
  })"; then
    fail "${transport_label}: capacity framing failed"
  fi
  [[ "${transport_frame}" =~ ${transport_sentinel}([0-9]{3})_([0-9]{3})__$ ]] || \
    fail "${transport_label}: capacity frame suffix differs"
  transport_suffix="${BASH_REMATCH[0]}"
  transport_child_status="${BASH_REMATCH[1]}"
  transport_encoder_status="${BASH_REMATCH[2]}"
  transport_payload="${transport_frame%"${transport_suffix}"}"
  [[ "${transport_child_status}" == 000 ]] || \
    fail "${transport_label}: capacity command failed rc=${transport_child_status}"
  [[ "${transport_encoder_status}" == 000 ]] || \
    fail "${transport_label}: capacity encoder failed rc=${transport_encoder_status}"
  [[ -n "${transport_payload}" && "${transport_payload}" != *$'\n'* \
    && "${transport_payload}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || \
    fail "${transport_label}: capacity transport base64 differs"
  printf -v "${transport_output_name}" '%s' "${transport_payload}"
}

decode_capacity_transport() {
  local decode_output_name="$1"
  local decode_payload="$2"
  local decode_label="$3"
  local decode_raw
  local decode_roundtrip
  if ! decode_raw="$(printf '%s' "${decode_payload}" | "${base64_bin}" -d)"; then
    fail "${decode_label}: capacity transport decode failed"
  fi
  if ! decode_roundtrip="$(printf '%s' "${decode_raw}" | "${base64_bin}" -w0)"; then
    fail "${decode_label}: capacity transport re-encode failed"
  fi
  [[ "${decode_roundtrip}" == "${decode_payload}" ]] || \
    fail "${decode_label}: capacity transport contains trailing bytes"
  printf -v "${decode_output_name}" '%s' "${decode_raw}"
}

for pending_sha in "${rank_exec_sha}" "${bootstrap_sha}" "${capacity_member_sha}" \
  "${release_manifest_sha}" "${renderer_sha}" "${capacity_python_sha}" \
  "${base64_sha}" "${env_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ $# == 8 ]] || fail "exact step authority argv differs"
readonly launch_authority_core="$1"
readonly launch_authority_core_sha="$2"
readonly attempt_intent="$3"
readonly attempt_intent_sha="$4"
readonly controller_capacity_receipt="$5"
readonly controller_capacity_receipt_sha="$6"
readonly controller_capacity_challenge="$7"
readonly foreground_capacity_challenge="$8"
[[ "${launch_authority_core}" == "${launch_root}/LAUNCH_AUTHORITY_CORE.json" ]] || fail "launch authority path differs"
[[ "${launch_authority_core_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launch authority SHA differs"
[[ "${attempt_intent_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "attempt intent SHA differs"
[[ "${attempt_intent}" == "${experiment_root}/attempts/${tag}/STARTED/intent.json" ]] || fail "attempt intent path differs"
readonly started_root="${experiment_root}/attempts/${tag}/STARTED"
[[ "${controller_capacity_receipt}" == "${started_root}/controller-capacity-receipt.json" ]] || fail "controller capacity receipt path differs"
[[ "${controller_capacity_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller capacity receipt SHA differs"
[[ "${controller_capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || fail "controller capacity challenge differs"
[[ "${foreground_capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || fail "foreground capacity challenge differs"
[[ "${foreground_capacity_challenge}" != "${controller_capacity_challenge}" ]] || \
  fail "foreground/controller capacity challenge collision"

[[ "$0" == "${step_self}" ]] || fail "step payload absolute path differs"
[[ -x "${step_self}" && ! -L "${step_self}" ]] || fail "step payload file differs"
require_stat_value "${step_self}" %a 555 "step payload mode"
require_stat_value "${step_self}" %h 1 "step payload link count"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
observed_hostname=
if ! observed_hostname="$(/bin/hostname -s)"; then
  fail "physical hostname query failed"
fi
readonly observed_hostname
[[ "${observed_hostname}" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm child is absent"
[[ "${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail "step node count differs"
[[ "${SLURM_STEP_NUM_TASKS:-${SLURM_NTASKS:-}}" == 1 ]] || fail "outer step task count differs"
[[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || fail "step CPU allocation differs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || fail "step must receive the full parent 64GiB host-memory allocation"
readonly current_step="${job_id}.${SLURM_STEP_ID}"
sibling_steps=
if ! sibling_steps="$(
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    /usr/bin/awk -v prefix="${job_id}." \
      'index($0,prefix)==1 && substr($0,length(prefix)+1) ~ /^[0-9]+$/ {print $0}' |
    LC_ALL=C /usr/bin/sort
)"; then
  fail "parent numeric-child closure query failed"
fi
readonly sibling_steps
[[ "${sibling_steps}" == "${current_step}" ]] || fail "parent numeric-child closure differs: ${sibling_steps}"

for authority_file in "${launch_authority_core}" "${attempt_intent}"; do
  [[ -f "${authority_file}" && ! -L "${authority_file}" ]] || fail "authority file differs"
  require_stat_value "${authority_file}" %a 444 "authority mode"
  require_stat_value "${authority_file}" %h 1 "authority link count"
done
require_sha256 "${launch_authority_core}" "${launch_authority_core_sha}" "launch authority"
require_sha256 "${attempt_intent}" "${attempt_intent_sha}" "attempt intent"

[[ -d "${release_root}" && ! -L "${release_root}" ]] || fail "Level-B release root differs"
require_stat_value "${release_root}" %a 555 "Level-B release root mode"
readonly expected_release_entries=$'RELEASE_MANIFEST.json\naction_preservation_decoded_eval_model_authority_v2.py\ninfer_action_edit_level_b_renderer_0817_v1.py\ninfer_lora.py\ntools'
observed_release_entries=
if ! observed_release_entries="$(/usr/bin/find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "Level-B release closure query failed"
fi
readonly observed_release_entries
[[ "${observed_release_entries}" == "${expected_release_entries}" ]] || fail "Level-B exact-five root closure differs"
[[ -d "${release_root}/tools" && ! -L "${release_root}/tools" ]] || fail "Level-B tools root differs"
require_stat_value "${release_root}/tools" %a 555 "Level-B tools root mode"
readonly expected_tool_entries=$'build_renderer_dataset.py\nmaterialize_vae.py'
observed_tool_entries=
if ! observed_tool_entries="$(/usr/bin/find "${release_root}/tools" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C /usr/bin/sort)"; then
  fail "Level-B tools closure query failed"
fi
readonly observed_tool_entries
[[ "${observed_tool_entries}" == "${expected_tool_entries}" ]] || fail "Level-B tools closure differs"
require_sha256 "${release_manifest}" "${release_manifest_sha}" "Level-B manifest"
require_sha256 "${renderer}" "${renderer_sha}" "Level-B renderer"
require_sha256 "${rank_exec}" "${rank_exec_sha}" "rank wrapper"
require_sha256 "${bootstrap}" "${bootstrap_sha}" "bootstrap"
[[ -f "${capacity_member}" && ! -L "${capacity_member}" ]] || fail "capacity member differs"
require_stat_value "${capacity_member}" %a 444 "capacity member mode"
require_stat_value "${capacity_member}" %h 1 "capacity member link count"
require_sha256 "${capacity_member}" "${capacity_member_sha}" "capacity member"
[[ -x "${capacity_python}" && ! -L "${capacity_python}" ]] || fail "capacity Python differs"
require_stat_value "${capacity_python}" %a 755 "capacity Python mode"
require_stat_value "${capacity_python}" %h 1 "capacity Python link count"
require_stat_value "${capacity_python}" %s 5937800 "capacity Python size"
require_sha256 "${capacity_python}" "${capacity_python_sha}" "capacity Python"
[[ -x "${env_bin}" && ! -L "${env_bin}" ]] || fail "env tool differs"
require_stat_value "${env_bin}" %a 755 "env tool mode"
require_stat_value "${env_bin}" %h 1 "env tool link count"
require_stat_value "${env_bin}" %s 43976 "env tool size"
require_sha256 "${env_bin}" "${env_sha}" "env tool"
[[ -x "${base64_bin}" && ! -L "${base64_bin}" ]] || fail "base64 tool differs"
require_stat_value "${base64_bin}" %a 755 "base64 tool mode"
require_stat_value "${base64_bin}" %h 1 "base64 tool link count"
require_stat_value "${base64_bin}" %s 35336 "base64 tool size"
require_sha256 "${base64_bin}" "${base64_sha}" "base64 tool"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python topology differs"
require_stat_value "${python_bin}" %a 755 "Python mode"
require_stat_value "${python_bin}" %h 1 "Python link count"
require_stat_value "${python_bin}" %s 31490256 "Python size"
require_sha256 "${python_bin}" "${python_sha}" "Python"
require_sha256 "${source_mp4}" "${source_sha}" "source MP4"
[[ -d "${run_root}" && ! -L "${run_root}" ]] || fail "run root differs"
require_stat_value "${run_root}" %a 700 "run root mode"
run_initial_entry=
if ! run_initial_entry="$(/usr/bin/find "${run_root}" -mindepth 1 -print -quit)"; then
  fail "run freshness query failed"
fi
readonly run_initial_entry
[[ -z "${run_initial_entry}" ]] || fail "run root is not fresh"
[[ ! -e "${output_mp4}" && ! -L "${output_mp4}" ]] || fail "output MP4 already exists"

# Bind the controller's independent direct-node sample, then take a third,
# fresh local management sample before importing torch or touching HIP.
validated_controller_capacity=
controller_validation_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before controller receipt validation"
capture_capacity_output_base64 controller_validation_transport \
  "controller capacity receipt validation" validate-file \
  "${controller_capacity_receipt}" "${controller_capacity_receipt_sha}" \
  controller "${controller_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after controller receipt validation"
decode_capacity_transport validated_controller_capacity \
  "${controller_validation_transport}" "controller capacity receipt validation"
readonly validated_controller_capacity
validated_controller_capacity_sha=
if ! validated_controller_capacity_sha="$(printf '%s' "${validated_controller_capacity}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "controller capacity receipt SHA query failed"
fi
readonly validated_controller_capacity_sha
[[ "${validated_controller_capacity_sha}" == "${controller_capacity_receipt_sha}" ]] || fail "controller capacity receipt bytes differ"

step_capacity_challenge=
step_challenge_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before step challenge generation"
capture_capacity_output_base64 step_challenge_transport \
  "step capacity challenge" challenge
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after step challenge generation"
decode_capacity_transport step_capacity_challenge \
  "${step_challenge_transport}" "step capacity challenge"
readonly step_capacity_challenge
[[ "${step_capacity_challenge}" =~ ^[0-9a-f]{64}$ ]] || fail "step capacity challenge differs"
[[ "${step_capacity_challenge}" != "${controller_capacity_challenge}" \
  && "${step_capacity_challenge}" != "${foreground_capacity_challenge}" ]] || \
  fail "step capacity challenge reused an earlier challenge"
step_capacity_base64=
step_probe_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before step local sample"
capture_capacity_output_base64 step_probe_transport \
  "step local capacity probe" probe-base64 step "${step_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after step local sample"
decode_capacity_transport step_capacity_base64 \
  "${step_probe_transport}" "step local capacity probe"
readonly step_capacity_base64
[[ -n "${step_capacity_base64}" && "${step_capacity_base64}" != *$'\n'* \
  && "${step_capacity_base64}" =~ ^[A-Za-z0-9+/]+={0,2}$ ]] || fail "step capacity base64 framing differs"
step_capacity_raw=
if ! step_capacity_raw="$(printf '%s' "${step_capacity_base64}" | "${base64_bin}" -d)"; then
  fail "step capacity receipt decode failed"
fi
readonly step_capacity_raw
step_capacity_sha=
if ! step_capacity_sha="$(printf '%s' "${step_capacity_raw}" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"; then
  fail "step capacity receipt SHA query failed"
fi
readonly step_capacity_sha
[[ "${step_capacity_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "step capacity receipt SHA differs"
validated_step_capacity=
step_validation_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before step receipt validation"
capture_capacity_output_base64 step_validation_transport \
  "step capacity receipt validation" validate-base64 \
  "${step_capacity_base64}" "${step_capacity_sha}" step \
  "${step_capacity_challenge}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after step receipt validation"
decode_capacity_transport validated_step_capacity \
  "${step_validation_transport}" "step capacity receipt validation"
readonly validated_step_capacity
[[ "${validated_step_capacity}" == "${step_capacity_raw}" ]] || fail "step capacity receipt canonical bytes differ"
readonly step_capacity_receipt="${started_root}/step-capacity-receipt.json"
[[ ! -e "${step_capacity_receipt}" && ! -L "${step_capacity_receipt}" ]] || fail "step capacity receipt already exists"
published_step_capacity_sha=
step_publish_transport=
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member before step receipt publication"
capture_capacity_output_base64 step_publish_transport \
  "step capacity receipt publication" publish-base64 \
  "${step_capacity_base64}" "${step_capacity_sha}" step \
  "${step_capacity_challenge}" "${step_capacity_receipt}"
require_sha256 "${capacity_member}" "${capacity_member_sha}" \
  "capacity member after step receipt publication"
decode_capacity_transport published_step_capacity_sha \
  "${step_publish_transport}" "step capacity receipt publication"
readonly published_step_capacity_sha
[[ "${published_step_capacity_sha}" == "${step_capacity_sha}" ]] || fail "published step capacity receipt SHA differs"
require_stat_value "${step_capacity_receipt}" %a 444 "step capacity receipt mode"
require_stat_value "${step_capacity_receipt}" %h 1 "step capacity receipt link count"
require_sha256 "${step_capacity_receipt}" "${step_capacity_sha}" "step capacity receipt"

# Retained independent physical admission: exact WORLD8 MI210 visibility,
# per-card >=95% free VRAM, and the real 64GiB Slurm host-memory grant.  The
# management probe above is the no-context low-use gate; this allocates no
# dummy tensors and remains immediately before torchrun.
"${python_bin}" -I -B -c '
import os, pathlib, torch
assert torch.__version__ == "2.7.1+rocm6.3", torch.__version__
assert torch.version.hip == "6.3.42131-fa1d09cbd", torch.version.hip
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
rows=[]
for index in range(8):
    name=torch.cuda.get_device_name(index)
    free,total=torch.cuda.mem_get_info(index)
    assert "MI210" in name, (index,name)
    assert total >= 63 * 1024**3, (index,total)
    assert free * 100 >= total * 95, (index,free,total)
    rows.append((index,name,free,total))
meminfo={}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
    if ":" in line:
        key,value=line.split(":",1); meminfo[key]=int(value.strip().split()[0])*1024
assert meminfo["MemTotal"] >= 120 * 1024**3, meminfo["MemTotal"]
assert meminfo["MemAvailable"] >= 8 * 1024**3, meminfo["MemAvailable"]
assert os.environ.get("SLURM_MEM_PER_NODE") == "65536"
parts=pathlib.Path("/proc/self/cgroup").read_text().strip().splitlines()
unified=[line.split(":",2)[2] for line in parts if line.startswith("0::")]
if unified:
    maximum=pathlib.Path("/sys/fs/cgroup") / unified[0].lstrip("/") / "memory.max"
    if maximum.is_file():
        value=maximum.read_text().strip()
        assert value == "max" or int(value) >= 64 * 1024**3, value
print("PASS physical Level-B gate", rows, "host", (meminfo["MemTotal"],meminfo["MemAvailable"]), flush=True)
'

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export BASH_ENV=/dev/null
unset ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
export -n SHELLOPTS BASHOPTS 2>/dev/null || true
export LEVEL_B_STEP_CAPACITY_RECEIPT="${step_capacity_receipt}"
export LEVEL_B_STEP_CAPACITY_RECEIPT_SHA256="${step_capacity_sha}"
export LEVEL_B_STEP_CAPACITY_CHALLENGE="${step_capacity_challenge}"

exec "${python_bin}" -I -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --no_python "${rank_exec}" "${bootstrap}" run
