#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh7-1b-gpu-246) mode=matched ;;
  auh7-1b-gpu-247) mode=cross ;;
  *) echo "inference suite is restricted to Job 140846 nodes 246/247" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
stage="$root/stage1"
release="$stage/complex8_crossappearance_motion_v1"
runner="$stage/source-crossappearance-motion-v1/methods/bernini_action_editing/scripts/auh_infer_complex8_crossappearance_motion_job140846_v1.sh"

run_one() {
  local step="$1"
  local case_id="$2"
  local output="$release/infer/$mode/s$(printf '%03d' "$step")/$case_id/output.mp4"
  if [ -f "$output" ]; then
    echo "SKIP existing $mode step=$step case=$case_id"
    return
  fi
  echo "START $mode step=$step case=$case_id"
  bash "$runner" "$step" "$case_id"
  test -s "$output"
  echo "DONE $mode step=$step case=$case_id"
}

# Real held-out editing is the primary decision surface.  Synthetic fitted
# decoding is limited to the best fixed-probe checkpoint and the final step.
for step in 1 10 40 64; do
  run_one "$step" real
done
for step in 10 64; do
  run_one "$step" synthetic
done

printf 'complete\n' > "$release/infer/$mode/SUITE_COMPLETE"
