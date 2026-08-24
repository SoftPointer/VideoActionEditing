#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ABSENT_CANDIDATE CACHE_DIR ABSENT_STEP_META ABSENT_RANK_ARGV IMMUTABLE_SNAPSHOT_ROOT MIOPEN_USER_DIR MIOPEN_CUSTOM_CACHE_DIR" >&2
  exit 64
fi

candidate=$1
cache_dir=$2
step_meta=$3
rank_argv=$4
snapshot_root=$5
miopen_user_dir=$6
miopen_custom_cache_dir=$7
controller="${snapshot_root}/actual_target_foundation_controller_v3.py"
runtime="${snapshot_root}/actual_target_foundation_runtime_v3.py"
expected_wrapper="${snapshot_root}/scripts/auh_actual_target_foundation_canary_rank_wrapper_v3.sh"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
expected_run_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/actual_target_foundation_canary_v3_20260823/run_7f3c21a9_v3r4
expected_miopen_user_dir="${expected_run_root}/miopen-user"
expected_miopen_custom_cache_dir="${expected_run_root}/miopen-custom"

if [[ "$0" != "$expected_wrapper" || ! -f "$controller" || -L "$controller" || ! -f "$runtime" || -L "$runtime" ]]; then
  echo "rank wrapper/source snapshot binding differs" >&2
  exit 65
fi
if [[ ${LOCAL_RANK:-} != 0 || ${WORLD_SIZE:-} != 1 ]]; then
  echo "rank wrapper requires LOCAL_RANK=0 WORLD_SIZE=1" >&2
  exit 66
fi
if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_STEP_ID:-} || -z ${ROCR_VISIBLE_DEVICES:-} || ${ROCR_VISIBLE_DEVICES} == *,* ]]; then
  echo "rank wrapper requires one real Slurm step and one ROCR device token" >&2
  exit 67
fi
if [[ -e "$candidate" || -L "$candidate" || -e "$step_meta" || -L "$step_meta" || -e "$rank_argv" || -L "$rank_argv" ]]; then
  echo "candidate/step metadata/rank argv must be absent" >&2
  exit 68
fi
if [[ ! -d "$cache_dir" || -L "$cache_dir" || "$miopen_user_dir" != "$expected_miopen_user_dir" || "$miopen_custom_cache_dir" != "$expected_miopen_custom_cache_dir" ]]; then
  echo "cache/MIOpen directory binding differs" >&2
  exit 69
fi
for scratch_dir in "$miopen_user_dir" "$miopen_custom_cache_dir"; do
  if [[ ! -d "$scratch_dir" || -L "$scratch_dir" || $(stat -c '%a' "$scratch_dir") != 700 ]]; then
    echo "MIOpen scratch must be a plain mode-0700 directory" >&2
    exit 70
  fi
  if [[ -n $(find "$scratch_dir" -mindepth 1 -print -quit) ]]; then
    echo "MIOpen scratch must be initially empty" >&2
    exit 71
  fi
done

# These exact fresh paths are established before the pinned Python process can
# import torch or initialize MIOpen.  Never fall back to HOME and never disable
# MIOpen's cache as a substitute for a writable, auditable database/cache.
export MIOPEN_USER_DB_PATH="$miopen_user_dir"
export MIOPEN_CUSTOM_CACHE_DIR="$miopen_custom_cache_dir"
unset MIOPEN_DISABLE_CACHE

export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -B "$controller" write-nul "$rank_argv" -- "$0" "$@"
"$python_bin" -B "$controller" write-step-meta \
  --path "$step_meta" \
  --candidate "$candidate" \
  --cache-dir "$cache_dir" \
  --rank-argv "$rank_argv" \
  --snapshot-root "$snapshot_root" \
  --miopen-user-dir "$miopen_user_dir" \
  --miopen-custom-cache-dir "$miopen_custom_cache_dir"
exec "$python_bin" -B "$runtime" --run-real --output "$candidate" --cache-dir "$cache_dir"
