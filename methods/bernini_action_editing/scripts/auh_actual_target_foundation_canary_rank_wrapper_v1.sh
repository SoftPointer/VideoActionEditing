#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ABSENT_OUTPUT_JSON OPTIONAL_CACHE_DIR_OR_NONE" >&2
  exit 64
fi
if [[ "${LOCAL_RANK:-0}" != "0" || "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "actual-target canary requires exactly one rank" >&2
  exit 65
fi
output_json="$1"
cache_dir="$2"
if [[ "$output_json" != /* || -e "$output_json" || -L "$output_json" ]]; then
  echo "output must be an absolute absent non-symlink path" >&2
  exit 66
fi
if [[ "$cache_dir" != "NONE" && ( "$cache_dir" != /* || -L "$cache_dir" || ! -d "$cache_dir" ) ]]; then
  echo "cache must be NONE or an absolute existing non-symlink directory" >&2
  exit 67
fi
if [[ "${ROCR_VISIBLE_DEVICES:-}" == *","* || -z "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  echo "ROCR_VISIBLE_DEVICES must expose exactly one externally isolated GPU" >&2
  exit 68
fi
runner="$(cd "$(dirname "$0")/.." && pwd -P)/actual_target_foundation_runtime_v1.py"
arguments=(--run-real --output "$output_json")
if [[ "$cache_dir" != "NONE" ]]; then arguments+=(--cache-dir "$cache_dir"); fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec /vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -B "$runner" "${arguments[@]}"
