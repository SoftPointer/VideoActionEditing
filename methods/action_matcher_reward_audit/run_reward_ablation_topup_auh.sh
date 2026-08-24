#!/usr/bin/env bash
# Launch four independent Bernini Best-of-N top-ups inside Job 135096.
# The script never submits, cancels, releases, or signals the parent allocation.

set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <job-id> <experiment-root> <payload-launcher>" >&2
  exit 64
fi

job_id=$1
experiment_root=$2
payload=$3
[[ "$job_id" =~ ^[1-9][0-9]*$ ]] || { echo "invalid job ID" >&2; exit 65; }
[[ "$experiment_root" == /* && "$payload" == /* ]] || { echo "paths must be absolute" >&2; exit 65; }
[[ -d "$experiment_root/specs" && -f "$payload" && ! -L "$payload" ]] || { echo "inputs absent" >&2; exit 66; }

readonly source_archive=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_native_core4_20260808_0bb0f20/inputs/bernini_pair_v5_method_0bb0f20.tar
readonly source_archive_sha256=1555f3b24b6eb3654c3be6318abe36eb99330a71cf9aacdcfa563fe980a9abe5
readonly source_revision=0bb0f205ed6c9f113906b5a3b121e330cdf01d3f
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_counterfactual_identity_orbit_v5_20260808_c099c6f/runtime/source_ea900d5/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12

for path in "$source_archive" "$checkpoint_manifest" "$payload"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "plain file absent: $path" >&2; exit 66; }
done
for path in "$bernini_root" "$veomni_root" "$checkpoint"; do
  [[ -d "$path" && ! -L "$path" ]] || { echo "directory absent: $path" >&2; exit 66; }
done
[[ -x "$python_bin" && "$(sha256sum "$source_archive" | awk '{print $1}')" == "$source_archive_sha256" ]] \
  || { echo "Python/source archive identity differs" >&2; exit 66; }

mkdir -p "$experiment_root/logs/topup" "$experiment_root/rollouts"

nodes=(auh7-1b-gpu-245 auh7-1b-gpu-246 auh7-1b-gpu-247 auh7-1b-gpu-248)
iids=(7b88a1ca1f804f41 841b5e0080a1441d a35b590961d24694 a66e6818e4144928)
pids=()
for index in 0 1 2 3; do
  node=${nodes[$index]}
  iid=${iids[$index]}
  spec="$experiment_root/specs/rollout-${iid}.json"
  spec_sha=$(sha256sum "$spec" | awk '{print $1}')
  output="$experiment_root/rollouts/$iid"
  [[ ! -e "$output" && ! -L "$output" ]] || { echo "refusing output reuse: $output" >&2; exit 67; }
  srun --overlap --jobid="$job_id" --nodes=1 --ntasks=1 --cpus-per-task=64 \
    --gres=gpu:mi210:8 --mem=512G --nodelist="$node" \
    env \
      PAIR_V5_SOURCE_ARCHIVE="$source_archive" \
      PAIR_V5_SOURCE_ARCHIVE_SHA256="$source_archive_sha256" \
      PAIR_V5_SOURCE_REVISION="$source_revision" \
      PAIR_V5_ROLLOUT_SPEC="$spec" \
      PAIR_V5_ROLLOUT_SPEC_SHA256="$spec_sha" \
      BERNINI_OFFICIAL_ROOT="$bernini_root" \
      BERNINI_VEOMNI_ROOT="$veomni_root" \
      BERNINI_ACTION_CHECKPOINT="$checkpoint" \
      BERNINI_CHECKPOINT_CONTENT_MANIFEST="$checkpoint_manifest" \
      PAIR_V5_ROLLOUT_OUTPUT_DIR="$output" \
      PAIR_V5_PYTHON_BIN="$python_bin" \
      bash "$payload" \
    >"$experiment_root/logs/topup/${iid}.log" 2>&1 &
  pids+=("$!")
  printf 'launched iid=%s node=%s controller_pid=%s\n' "$iid" "$node" "$!"
done

status=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    status=1
    printf 'FAILED iid=%s node=%s\n' "${iids[$index]}" "${nodes[$index]}" >&2
  else
    printf 'COMPLETED iid=%s node=%s\n' "${iids[$index]}" "${nodes[$index]}"
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "one or more top-up tasks failed; Job 135096 was not cancelled or released" >&2
  exit "$status"
fi
sha256sum "$experiment_root"/rollouts/*/*/rv2v.mp4 >"$experiment_root/topup-mp4-SHA256SUMS"
printf 'all top-ups completed; parent_allocation_cancelled=false parent_allocation_released=false\n'
