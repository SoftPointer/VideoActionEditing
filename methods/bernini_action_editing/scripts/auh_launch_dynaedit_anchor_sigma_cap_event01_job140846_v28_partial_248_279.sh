#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "usage: $0" >&2
  exit 2
fi

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
payload="$stage/launch_anchor_sigma_cap_v28_860b5139.sh"
output_root="$stage/dynaedit_anchor_sigma_cap_event01_v28_9648a402_b97faa62"

test "$(sha256sum "$payload" | awk '{print $1}')" = 860b5139d1d6fc3389b20cfaf2151df3d78f61125ba23a6da2e784122e2c3dac
for label in \
  DYNSTATIC_VEL_A010_EARLY8_CAP06 \
  DYNSTATIC_SELFATT_A003_EARLY8_B8_13_CAP08
do
  test ! -e "$output_root/$label.mp4"
  test ! -e "$output_root/$label.mp4.receipt.json"
done
mkdir -p "$output_root"

srun --overlap --exact --jobid=140846 \
  -w auh7-1b-gpu-248 -N1 -n1 -c64 --mem=64G \
  --gres=gpu:mi210:8 \
  bash "$payload" >"$output_root/launch_r3_full64g_auh7-1b-gpu-248.log" 2>&1 &
pid_248=$!

srun --overlap --exact --jobid=140846 \
  -w auh7-1b-gpu-279 -N1 -n1 -c64 --mem=64G \
  --gres=gpu:mi210:8 \
  bash "$payload" >"$output_root/launch_r3_full64g_auh7-1b-gpu-279.log" 2>&1 &
pid_279=$!

status=0
wait "$pid_248" || status=$?
wait "$pid_279" || status=$?
exit "$status"
