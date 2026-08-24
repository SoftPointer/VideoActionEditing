#!/usr/bin/env bash
# Sole node279 Slurm-step payload for one PRE_D0 Level-B P2 full render.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-b-p2-00435-v1
readonly release_root="${experiment_root}/releases/${tag}"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly step_self="${launch_root}/auh_action_edit_level_b_p2_00435_step_v1.sh"
readonly rank_exec="${launch_root}/auh_action_edit_level_b_p2_00435_rank_exec_v1.sh"
readonly rank_exec_sha=6ec31d8bbc35f960d52a1aae5ac9160f1002cf86f0893094d51ae2cea1981e26
readonly bootstrap="${launch_root}/action_edit_level_b_p2_00435_bootstrap_0817_v1.py"
readonly bootstrap_sha=e00e3ece252ca6b37e00bb717419199f2371cf59b6e4123c59eb19f69a98a5d1
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=8cf24d45f64eed4d6bc3e02b60d68ab997ab8f56ccdfe33849e04dc7c6f684bf
readonly renderer="${release_root}/infer_action_edit_level_b_renderer_0817_v1.py"
readonly renderer_sha=2b807aec19c17953a890ef76b9164b786af2b7f1912c32b2c33194c15ca29eed
readonly output_mp4="${run_root}/00435ad621c44fac_p2_seed2026080821.mp4"
readonly source_mp4=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/00435ad621c44fac/samples/00435ad621c44fac/source_video.mp4
readonly source_sha=b9218921597e43e2a3a6b223899ab84fb1b8d1a51692766bb2167e5941efbba1

fail() {
  printf 'Level-B P2 node279 step refused: %s\n' "$*" >&2
  exit 96
}

for pending_sha in "${rank_exec_sha}" "${bootstrap_sha}" "${release_manifest_sha}" "${renderer_sha}"; do
  [[ "${pending_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launcher authority SHA differs"
done
[[ $# == 4 ]] || fail "exact step authority argv differs"
readonly launch_authority_core="$1"
readonly launch_authority_core_sha="$2"
readonly attempt_intent="$3"
readonly attempt_intent_sha="$4"
[[ "${launch_authority_core}" == "${launch_root}/LAUNCH_AUTHORITY_CORE.json" ]] || fail "launch authority path differs"
[[ "${launch_authority_core_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launch authority SHA differs"
[[ "${attempt_intent_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "attempt intent SHA differs"
[[ "${attempt_intent}" == "${experiment_root}/attempts/${tag}/STARTED/intent.json" ]] || fail "attempt intent path differs"

[[ "$0" == "${step_self}" ]] || fail "step payload absolute path differs"
[[ -x "${step_self}" && ! -L "${step_self}" ]] || fail "step payload file differs"
[[ "$(stat -c %a "${step_self}")" == 555 && "$(stat -c %h "${step_self}")" == 1 ]] || fail "step payload topology differs"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
[[ "$(hostname -s)" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm child is absent"
[[ "${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail "step node count differs"
[[ "${SLURM_STEP_NUM_TASKS:-${SLURM_NTASKS:-}}" == 1 ]] || fail "outer step task count differs"
[[ "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || fail "step CPU allocation differs"
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || fail "step must receive the full parent 64GiB host-memory allocation"
readonly current_step="${job_id}.${SLURM_STEP_ID}"
readonly sibling_steps="$(
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    awk -v prefix="${job_id}." \
      'index($0,prefix)==1 && substr($0,length(prefix)+1) ~ /^[0-9]+$/ {print $0}' |
    LC_ALL=C sort
)"
[[ "${sibling_steps}" == "${current_step}" ]] || fail "parent numeric-child closure differs: ${sibling_steps}"

for authority_file in "${launch_authority_core}" "${attempt_intent}"; do
  [[ -f "${authority_file}" && ! -L "${authority_file}" ]] || fail "authority file differs"
  [[ "$(stat -c %a "${authority_file}")" == 444 && "$(stat -c %h "${authority_file}")" == 1 ]] || fail "authority topology differs"
done
[[ "$(sha256sum "${launch_authority_core}" | awk '{print $1}')" == "${launch_authority_core_sha}" ]] || fail "launch authority bytes differ"
[[ "$(sha256sum "${attempt_intent}" | awk '{print $1}')" == "${attempt_intent_sha}" ]] || fail "attempt intent bytes differ"

[[ -d "${release_root}" && ! -L "${release_root}" && "$(stat -c %a "${release_root}")" == 555 ]] || fail "Level-B release root differs"
readonly expected_release_entries=$'RELEASE_MANIFEST.json\naction_preservation_decoded_eval_model_authority_v2.py\ninfer_action_edit_level_b_renderer_0817_v1.py\ninfer_lora.py\ntools'
readonly observed_release_entries="$(find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_release_entries}" == "${expected_release_entries}" ]] || fail "Level-B exact-five root closure differs"
[[ -d "${release_root}/tools" && ! -L "${release_root}/tools" && "$(stat -c %a "${release_root}/tools")" == 555 ]] || fail "Level-B tools root differs"
readonly expected_tool_entries=$'build_renderer_dataset.py\nmaterialize_vae.py'
readonly observed_tool_entries="$(find "${release_root}/tools" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_tool_entries}" == "${expected_tool_entries}" ]] || fail "Level-B tools closure differs"
[[ "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || fail "Level-B manifest SHA differs"
[[ "$(sha256sum "${renderer}" | awk '{print $1}')" == "${renderer_sha}" ]] || fail "Level-B renderer SHA differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ "$(sha256sum "${bootstrap}" | awk '{print $1}')" == "${bootstrap_sha}" ]] || fail "bootstrap SHA differs"
[[ "$(sha256sum "${python_bin}" | awk '{print $1}')" == "${python_sha}" ]] || fail "Python SHA differs"
[[ "$(sha256sum "${source_mp4}" | awk '{print $1}')" == "${source_sha}" ]] || fail "source MP4 SHA differs"
[[ -d "${run_root}" && ! -L "${run_root}" && "$(stat -c %a "${run_root}")" == 700 ]] || fail "run root differs"
[[ -z "$(find "${run_root}" -mindepth 1 -print -quit)" ]] || fail "run root is not fresh"
[[ ! -e "${output_mp4}" && ! -L "${output_mp4}" ]] || fail "output MP4 already exists"

# Physical admission: eight visible MI210s, no pre-existing VRAM occupants, and
# a real 64GiB Slurm host-memory grant.  This allocates no dummy tensors.
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

exec "${python_bin}" -I -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --no_python "${rank_exec}" "${bootstrap}" run
