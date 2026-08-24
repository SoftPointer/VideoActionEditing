#!/usr/bin/env bash
set -euo pipefail

# Extract RAFT flow for one audited self-generated target.  This is data
# preparation, not a training job; the later SP4 training memory gate remains
# strictly based on real model allocations.
if [ "$#" -ne 2 ]; then
  echo "usage: $0 IID GPU_INDEX" >&2
  exit 2
fi
iid="$1"
gpu="$2"
case "$(hostname -s)" in
  auh7-1b-gpu-246|auh7-1b-gpu-247|auh7-1b-gpu-248|auh7-1b-gpu-279) ;;
  *) echo "forbidden node: extraction is restricted to Job 140846 nodes" >&2; exit 3 ;;
esac
case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) echo "GPU_INDEX must lie in [0,7]" >&2; exit 2 ;; esac
case "$iid" in
  311c82f83eca4a7f) filename=311c82f83eca4a7f__query-forward-s2026082221.mp4; expected=4abba7c19d65ffe1e2dc31af88bb4c9f249760ac092f0eaa8a579338cbd8d93a ;;
  31c34509415745ca) filename=31c34509415745ca__query-forward-s2026082131.mp4; expected=297ee47fa24d0d3ac37f9daf75670c67e2ba5d92153a2a938bdc67633f8d3265 ;;
  6d346c38cf504493) filename=6d346c38cf504493__query-forward-s2026082231.mp4; expected=77dbe10b5104e7b1dea36c5b14663b1825632a25a6ad19c47e3d981107511219 ;;
  6ea45d35943742bb) filename=6ea45d35943742bb__query-forward-s2026082211.mp4; expected=aa4ec85b20914011683042a5d612a6a21f1ce8010ad26fcdeff4417f95c6b3a8 ;;
  99cde432839f4240) filename=99cde432839f4240__query-forward-s2026082201.mp4; expected=158850ed7a0b840a19b131a9b7a300f8eae84b81cf960ac0858bb9e0a9ae09d3 ;;
  *) echo "unknown expanded-bank IID: $iid" >&2; exit 2 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
flow_tree="$root/stage1/source-flowfix-v2"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
video="$root/stage1/expanded_bank_v1/source_videos/$filename"
flow_output_root="${FLOW_OUTPUT_ROOT:-$root/stage1/expanded_bank_v1/flows}"
output="$flow_output_root/$iid.safetensors"
test -f "$video"
test "$(sha256sum "$video" | awk '{print $1}')" = "$expected"
test ! -e "$output"
test ! -e "${output%.safetensors}.json"
mkdir -p "${output%/*}"

scratch="/tmp/expanded-bank-flow-${iid}-g${gpu}-140846"
mkdir -p "$scratch/cache" "$scratch/miopen-user" "$scratch/miopen-custom"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/vast/users/guangyi.chen/.cache/torch
export XDG_CACHE_HOME="$scratch/cache"
export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export ROCR_VISIBLE_DEVICES="$gpu"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
exec "$python_bin" -B "$flow_tree/methods/bernini_action_editing/extract_anchor_raft_flow_v1.py" \
  --source "$video" --anchor "$video" --output "$output"
