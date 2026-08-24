#!/usr/bin/env bash
# Wait for the nine-arm rollout to release all GPU groups, canary all eight
# anti-hacking arms, then launch the formal 320-update screen.

set -Eeuo pipefail

eval_complete=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_objective_ninearm_eval_20260815_v4/eval/EVALUATION_COMPLETE
root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_antihack_eightarm_20260815_v5
source_archive="${root}/release/reward-antihack-v5-9b1f54d.tar"
source_sha=b5aba0f22343e45fa45d4f5d9d4f5974c597b0c8870f033114682fc30c4a974d
source_revision=9b1f54dd41f880538c7166abbf6a2727de0139d2
node_launcher="${root}/release/auh_train_reward_objective_v2.sh"
eight_launcher="${root}/release/auh_launch_reward_antihack_eightarm_job135096_v5.sh"

while [[ ! -f "${eval_complete}" ]]; do
  echo "WAITING_FOR_EVAL $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 15
done
echo "EVAL_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

common_env=(
  BERNINI_ACTION_SOURCE_ARCHIVE="${source_archive}"
  BERNINI_ACTION_SOURCE_ARCHIVE_SHA256="${source_sha}"
  BERNINI_ACTION_SOURCE_REVISION="${source_revision}"
  BERNINI_ACTION_NODE_LAUNCHER="${node_launcher}"
  BERNINI_ACTION_EXPERIMENT_ROOT="${root}"
)

env "${common_env[@]}" \
  ANTIHACK_RUN_TAG=canary_u1 ANTIHACK_MAX_STEPS=1 ANTIHACK_SAVE_EVERY=1 \
  "${eight_launcher}" |& tee "${root}/logs/controller-canary_u1.log"
echo "CANARY_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"

env "${common_env[@]}" \
  ANTIHACK_RUN_TAG=formal_u320 ANTIHACK_MAX_STEPS=320 ANTIHACK_SAVE_EVERY=80 \
  "${eight_launcher}" |& tee "${root}/logs/controller-formal_u320.log"
echo "FORMAL_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
