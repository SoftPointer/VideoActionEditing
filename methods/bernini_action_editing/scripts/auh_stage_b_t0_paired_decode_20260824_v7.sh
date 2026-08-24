#!/usr/bin/env bash
set -euo pipefail

experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_middle_g1_20260824_v2
source_root="$experiment_root/source_stage_b_t0_decode_v7_paired"
method_root="$source_root/methods/bernini_action_editing"
runner="$method_root/infer_action_repr_target_t0_paired_decode_v1.py"
manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_repr_target_selfgen_20260824_v1/source/experiment_manifest.json
g2a_receipt="$experiment_root/stage1_v2_7/g2a/production_world4/0be6494dfac3/receipt.json"
flow_cohort="$experiment_root/stage1_v2_7/g1_cohorts/target/0be6494dfac3/flow/cohort_receipt.json"
middle_cohort="$experiment_root/stage1_v2_7/g1_cohorts/target/0be6494dfac3/middle/cohort_receipt.json"
t0_output="$experiment_root/stage_b_t0_retry8/target_t0/0be6494dfac3/single_update"
decode_root="$experiment_root/stage_b_t0_retry8/matched_decode_v7_paired/0be6494dfac3"
log_root="$experiment_root/logs/stage_b_t0_retry8/matched_decode_v7_paired"
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
source_video=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1/preprocessed_sources/0be6494dfac3/source-exact81.mp4
instruction='Edit the action so that the cockatoo turns its head towards frame left, raising its curved yellow crest.'
seed=2026081908

expected_runner_sha=edd5d6e9d37116c81218525061ce89f31f1dcc87ef801c381e8e66c0e8ecd2c4
expected_test_sha=52ba3e013925b011e5fd0165e099eeaed23525b724ca308871a41a3b75c62d0b
expected_single_runner_sha=c583ef9a74338aa62b3fc74d4fdbedcf6f8cdeb5b9612d626fe9c9addc6d44da
expected_single_test_sha=908e12db0476987ed12b4e8e2328000a278bb65790f5ce710ba6e46318f37e4f
expected_t0_receipt_sha=f94c6ae79e2e377f875c137b5be45a6040ddf862366736abb092896910167a83
expected_g2a_receipt_sha=7ea0ab20709d942ca51a3062f2306407be8f9d0f4445926dca57af9b83fc3f09
expected_manifest_sha=c78e42f0661e5905407505037ce322d32d67ffec0b70b1cab466f895dc8d0632
expected_flow_cohort_sha=f72a3ee8002002c7d3d975b2c378cb32a3afc7cc04832d8cd144b4306d36910e
expected_middle_cohort_sha=d365a3797b8ee35c6a5570be37188dcd1453ae8dfc88a1ff8193431137e69c5f
expected_step0_state_sha=2e10edd015abdc0ce077a59ba1e6ce45f79df8f2c2805ad64971a4de055ddee8
expected_step1_state_sha=91f06e92837dadf8229ca1f2e5a26e512e8bbc26ddac8e1057bc832fa93ea44c
expected_source_sha=8386303e5a32d4d70292d86ba46b7a701435eab886550bc95bbe9d541098d4dc
expected_g2a_core_sha=3e3542042d71cd567429e764637e0be561f3d08827058e251dcdecfd017bb758
expected_g2a_world4_sha=8f6f13e76bcba0defd9af7576912eb55eddd76c1cc74ba334e1b11d5e0dd359d
expected_retry8_sha=2e8c6dcef64dbbb854d51ae6bbe9164a96969f2c6566767f49b2db61c767ae2d
expected_infer_lora_sha=5a7a790682ba058b501582a2a9c70327467be1e4f05dc1d1b39a0c54b6b9b83e
expected_model_authority_sha=b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf
expected_exact_video_sha=83c0410ebbba177af73eff658f70146b1e0a7a9b711bfc3e4a94c4c29ba2409a

fail() {
  printf '[stage-b-t0-paired-v7] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'usage: %s {preflight|decode JOB_ID|worker JOB_ID|status}\n' "$0" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_sha() {
  local path="$1" expected="$2" label="$3"
  [ -f "$path" ] && [ ! -L "$path" ] || fail "$label is unavailable"
  [ "$(sha256_file "$path")" = "$expected" ] || fail "$label SHA-256 differs"
}

common_preflight() {
  [ -x "$python_bin" ] || fail 'vace Python is unavailable'
  "$python_bin" -B -c 'import av,torch; assert av.__version__ and torch.version.hip' || fail 'AUH ROCm/PyAV runtime is unavailable'
  require_sha "$runner" "$expected_runner_sha" 'paired decode runner'
  require_sha "$method_root/tests/test_infer_action_repr_target_t0_paired_decode_v1.py" "$expected_test_sha" 'paired decode test'
  require_sha "$method_root/infer_action_repr_target_t0_matched_decode_v1.py" "$expected_single_runner_sha" 'single decode dependency'
  require_sha "$method_root/tests/test_infer_action_repr_target_t0_matched_decode_v1.py" "$expected_single_test_sha" 'single decode dependency test'
  require_sha "$method_root/action_repr_g2a_adapter_v1.py" "$expected_g2a_core_sha" 'G2a adapter core'
  require_sha "$method_root/audit_action_repr_g2a_world4_v1.py" "$expected_g2a_world4_sha" 'G2a WORLD4 core'
  require_sha "$method_root/train_action_repr_target_t0_canary_retry8_v1.py" "$expected_retry8_sha" 'retry8 T0 runner'
  require_sha "$method_root/infer_lora.py" "$expected_infer_lora_sha" 'native inference transport'
  require_sha "$method_root/action_preservation_decoded_eval_model_authority_v2.py" "$expected_model_authority_sha" 'native model-consumption authority'
  require_sha "$method_root/exact_local_video_materializer_v1.py" "$expected_exact_video_sha" 'exact source materializer'
  require_sha "$t0_output/receipt.json" "$expected_t0_receipt_sha" 'T0 receipt'
  require_sha "$g2a_receipt" "$expected_g2a_receipt_sha" 'production G2a receipt'
  require_sha "$manifest" "$expected_manifest_sha" 'experiment manifest'
  require_sha "$flow_cohort" "$expected_flow_cohort_sha" 'target flow cohort'
  require_sha "$middle_cohort" "$expected_middle_cohort_sha" 'target middle cohort'
  require_sha "$t0_output/step0000/adapter_model.safetensors" "$expected_step0_state_sha" 'step-0 adapter state'
  require_sha "$t0_output/step0001/adapter_model.safetensors" "$expected_step1_state_sha" 'step-1 adapter state'
  require_sha "$source_video" "$expected_source_sha" 'source video'
  [ -d "$bernini_root" ] && [ -d "$veomni_root" ] && [ -d "$checkpoint" ] || fail 'pinned Bernini runtime is unavailable'
  PYTHONPATH="$source_root:$method_root" "$python_bin" -B -c 'import sys; from train_action_repr_target_t0_canary_retry8_v1 import validate_published_t0_output; validate_published_t0_output(sys.argv[1])' "$t0_output"
}

job_row() {
  scontrol show job "$1" -o
}

job_node() {
  local row node
  row="$(job_row "$1")"
  node="$(sed -n 's/.* NodeList=\([^ ]*\).*/\1/p' <<<"$row")"
  [[ "$node" =~ ^auh7-1b-gpu-[0-9]+$ ]] || fail "job node differs: $node"
  printf '%s\n' "$node"
}

validate_job() {
  local job="$1" row node alloc
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'job id must be numeric'
  row="$(job_row "$job")"
  node="$(job_node "$job")"
  [[ " $row " == *" UserId=guangyi.chen(2012) "* ]] || fail 'job owner differs'
  [[ " $row " == *" Account=faculty-acc "* ]] || fail 'job account differs'
  [[ " $row " == *" QOS=bgqos "* ]] || fail 'job QOS differs'
  [[ " $row " == *" Partition=faculty "* ]] || fail 'job partition differs'
  [[ " $row " == *" JobState=RUNNING "* ]] || fail 'job is not RUNNING'
  [[ " $row " == *" NumNodes=1 "* ]] || fail 'paired decode job must have one node'
  alloc="$(sed -n 's/.* AllocTRES=\([^ ]*\).*/\1/p' <<<"$row")"
  [[ ",$alloc," == *",gres/gpu:mi210=4,"* || ",$alloc," == *",gres/gpu:mi210=8,"* ]] || fail 'job must allocate 4 or 8 MI210 GPUs'
  if [[ " $row " != *" MinMemoryNode=64G "* && " $row " != *" MinMemoryNode=65536M "* ]]; then
    fail 'job must advertise 64G host memory'
  fi
  printf '%s\n' "$node"
}

validate_worker() {
  local expected_job="$1" raw count token seen=,
  local -a devices
  validate_job "$expected_job" >/dev/null
  [ "${SLURM_JOB_ID-}" = "$expected_job" ] || fail 'worker job id differs'
  [ "$(hostname -s)" = "$(job_node "$expected_job")" ] || fail 'worker node differs'
  [ -z "${HIP_VISIBLE_DEVICES-}" ] && [ -z "${CUDA_VISIBLE_DEVICES-}" ] || fail 'HIP/CUDA visibility overrides are forbidden'
  raw="${ROCR_VISIBLE_DEVICES-}"
  IFS=',' read -r -a devices <<<"$raw"
  [ "${#devices[@]}" -eq 4 ] || fail 'worker requires exactly four visible GPUs'
  for token in "${devices[@]}"; do
    [[ "$token" =~ ^[0-7]$ ]] || fail 'ROCR device token differs'
    [[ "$seen" != *",$token,"* ]] || fail 'ROCR device token is duplicated'
    seen+="$token,"
  done
  count="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
  [ "$count" = 4 ] || fail 'PyTorch does not see exactly four MI210 GPUs'
}

validate_output() {
  [ -s "$decode_root/paired_receipt.json" ] || fail 'paired receipt is absent'
  PYTHONPATH="$source_root:$method_root" "$python_bin" -B - "$decode_root" <<'PY'
import json,sys
from pathlib import Path
from infer_action_repr_target_t0_paired_decode_v1 import validate_paired_receipt
from infer_action_repr_target_t0_matched_decode_v1 import validate_video_artifact
root=Path(sys.argv[1])
receipt=validate_paired_receipt(json.loads((root/'paired_receipt.json').read_text(encoding='ascii')))
for cell in receipt['cells']:
    validate_video_artifact(cell['output'])
print('[stage-b-t0-paired-v7] OUTPUT_REPLAY_PASS cells=%d baseline_gate=%s route_effect=%s' % (
    len(receipt['cells']), receipt['paired_gate']['baseline_gate_passed'], receipt['paired_gate']['route_effect_detected']))
PY
}

run_worker() {
  local job="$1" log scratch step_token rc
  common_preflight
  validate_worker "$job"
  [ ! -e "$decode_root" ] && [ ! -L "$decode_root" ] || fail 'paired decode coordinate is permanently consumed; use a new revision'
  mkdir -p "$log_root"
  log="$log_root/run-job${job}.log"
  (set -o noclobber; umask 077; : >"$log") 2>/dev/null || fail 'create-only paired decode log already exists'
  step_token="${SLURM_STEP_ID//[^A-Za-z0-9_.-]/_}"
  scratch="/tmp/action-repr-t0-paired-v7-${job}-${step_token}"
  [ ! -e "$scratch" ] || fail 'paired decode scratch path already exists'
  mkdir -p "$scratch/xdg" "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export XDG_CACHE_HOME="$scratch/xdg"
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
  export TRITON_CACHE_DIR="$scratch/triton"
  set +e
  /usr/bin/timeout --signal=TERM --kill-after=60s 120m "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 "$runner" --t0-output "$t0_output" --g2a-receipt "$g2a_receipt" --experiment-manifest "$manifest" --flow-cohort-receipt "$flow_cohort" --middle-cohort-receipt "$middle_cohort" --paired-output-root "$decode_root" --bernini-root "$bernini_root" --veomni-root "$veomni_root" --checkpoint "$checkpoint" --source-video "$source_video" --instruction "$instruction" --output "$decode_root/cells/s0_route_off_a/output.mp4" --num-inference-steps 40 --seed "$seed" --source-onset-policy hard1_every_step --method-source-revision e1e67ca9974d28b6691ac77e00de19a485b99ee9 --method-source-archive-sha256 5127d99310cf5b0cce7128f2cb14e02cf58e561ab3529d0e66b1e047e1d29d7c >>"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 4 ]; then
    validate_output
    fail 'paired latent determinism gate failed; preserve v7 evidence and do not compare step1'
  fi
  [ "$rc" -eq 0 ] || fail "WORLD4 paired decode failed rc=$rc; preserve v7 evidence"
  validate_output
  printf '[stage-b-t0-paired-v7] PAIRED_DECODE_PASS output=%s\n' "$decode_root"
}

launch_decode() {
  local job="$1" node outer
  common_preflight
  node="$(validate_job "$job")"
  mkdir -p "$log_root"
  outer="$log_root/job${job}-outer.log"
  (set -o noclobber; umask 077; : >"$outer") 2>/dev/null || fail 'create-only paired outer log already exists'
  srun --jobid="$job" --exclusive --exact --nodelist="$node" --nodes=1 --ntasks=1 --gres=gpu:mi210:4 --cpus-per-task=16 --mem=0 "$0" worker "$job" >>"$outer" 2>&1 || fail 'paired decode srun failed; preserve output/logs and use a new revision'
  validate_output
  printf '[stage-b-t0-paired-v7] srun complete job=%s\n' "$job"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export PYTHONPATH="$source_root:$method_root"

[ "$#" -ge 1 ] || usage
command="$1"
shift
case "$command" in
  preflight)
    [ "$#" -eq 0 ] || usage
    common_preflight
    echo '[stage-b-t0-paired-v7] PREFLIGHT_PASS WORLD4_SP4=true cells=11 one_model_load=true per_cell_real_clamp=true latent_rank_hash=true baseline_exact_gate=true videos=true no_cpu_fallback=true'
    ;;
  decode)
    [ "$#" -eq 1 ] || usage
    launch_decode "$1"
    ;;
  worker)
    [ "$#" -eq 1 ] || usage
    run_worker "$1"
    ;;
  status)
    [ "$#" -eq 0 ] || usage
    if [ ! -d "$decode_root" ]; then
      echo '[stage-b-t0-paired-v7] NOT_STARTED'
      exit 0
    fi
    find "$decode_root" -maxdepth 4 -type f -print | sort
    ;;
  *) usage ;;
esac
