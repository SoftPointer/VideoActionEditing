#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ABSENT_OUTPUT_JSON PYTHON_ENTRYPOINT" >&2
  exit 64
fi
output_json="$1"
entrypoint="$2"
case "${LOCAL_RANK:-}" in
  0|1|2|3) ;;
  *) echo "LOCAL_RANK must be one of 0,1,2,3" >&2; exit 65 ;;
esac
cache_root="${MIOPEN_CACHE_ROOT:?MIOPEN_CACHE_ROOT is required}"
if [[ "$cache_root" != /* || -L "$cache_root" || ! -d "$cache_root" ]]; then
  echo "MIOPEN_CACHE_ROOT must be an absolute existing non-symlink directory" >&2
  exit 66
fi
if [[ "$output_json" != /* || -e "$output_json" || -L "$output_json" ]]; then
  echo "output JSON must be an absolute absent non-symlink path" >&2
  exit 67
fi
if [[ "$entrypoint" != /* || -L "$entrypoint" || ! -f "$entrypoint" ]]; then
  echo "entrypoint must be an absolute existing non-symlink file" >&2
  exit 68
fi

rank_root="${cache_root}/rank-${LOCAL_RANK}"
user_db="${rank_root}/user-db"
kernel_cache="${rank_root}/kernel-cache"
local_tmp_root="/tmp/bernini-branch-graph-u${UID}-j${SLURM_JOB_ID:?SLURM_JOB_ID is required}-s${SLURM_STEP_ID:?SLURM_STEP_ID is required}"
rank_tmp="${local_tmp_root}/rank-${LOCAL_RANK}"
if [[ -L "$local_tmp_root" || -L "$rank_tmp" ]]; then
  echo "rank-local TMPDIR authority is a symlink" >&2
  exit 69
fi
mkdir -p "$user_db" "$kernel_cache" "$rank_tmp"
chmod 0700 "$rank_root" "$user_db" "$kernel_cache" "$local_tmp_root" "$rank_tmp"
export MIOPEN_USER_DB_PATH="$user_db"
export MIOPEN_CUSTOM_CACHE_DIR="$kernel_cache"
export TMPDIR="$rank_tmp"

exec /vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -B \
  "$entrypoint" --run --output "$output_json"
