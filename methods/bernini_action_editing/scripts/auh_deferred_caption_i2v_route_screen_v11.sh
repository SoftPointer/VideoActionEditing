#!/usr/bin/env bash
set -euo pipefail

case "$(hostname -s)" in
  auh-1b-cpu-login-002|auh7-ib|auh7-ib.rcs.ku.ac.ae|auh) ;;
  *) echo "this deferred controller must run on the AUH login host" >&2; exit 2 ;;
esac

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
release="$stage/online_anchor_attention_training_v1"
source_tree="$stage/source-online-anchor-attention-training-v1"
fallback_tree="$stage/source-sga-anc-training-v1"
runtime_tree="$stage/source-be31323"
archive="$release/online-anchor-caption-i2v-route-screen-v11.tar"
revision="$release/online-anchor-caption-i2v-route-screen-v11.revision"
expected_sha=1c3c13d5d58ad7b76a8378218df5efd6590a896475f031faa4086b7abc114e54
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
logs="$release/logs/caption_i2v_route_screen_v11"
mkdir -p "$logs"

test -f "$archive"
test -f "$revision"
test "$(sha256sum "$archive" | awk '{print $1}')" = "$expected_sha"
test "$(tr -d '\n' < "$revision")" = online-anchor-caption-i2v-route-screen-v11-20260820

wait_file() {
  local path="$1"
  while [ ! -f "$path" ]; do sleep 20; done
}

test_root="$(mktemp -d /tmp/caption-i2v-route-screen-v11-test.XXXXXX)"
tar -xf "$archive" -C "$test_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$test_root/methods/bernini_action_editing:$source_tree/methods/bernini_action_editing:$fallback_tree/methods/bernini_action_editing:$runtime_tree/methods/bernini_action_editing"
"$python_bin" "$test_root/methods/bernini_action_editing/tests/test_train_online_anchor_attention_v1.py"

# The archived attention/trainer files are byte-identical to the already
# audited v10r2 source tree and decoder bridge.  Run launch scripts from this
# private extraction, so no shared source file changes while v10r2 decodes.
test "$(sha256sum "$source_tree/methods/bernini_action_editing/anchor_qk_transport.py" | awk '{print $1}')" = 603046b6bd419245a24bdd4e45696d6dfea4f68a1bca2db313405b07fec26968
test "$(sha256sum "$source_tree/methods/bernini_action_editing/anchor_cross_attention_transport.py" | awk '{print $1}')" = cbbbc490711f4f1131a701d042309b024fd5b3922f2d88e39e35bff7d39396f6
test "$(sha256sum "$source_tree/methods/bernini_action_editing/train_online_anchor_attention_v1.py" | awk '{print $1}')" = fe42dcba50f48581316aece9e5e11a5d80dea9ec51b3f5df5af35432aa0cc866
launcher="$test_root/methods/bernini_action_editing/scripts/auh_launch_online_anchor_caption_i2v_route_screen_v11.sh"
watcher="$test_root/methods/bernini_action_editing/scripts/auh_watch_decode_caption_i2v_route_screen_v11.sh"

# These three nodes are free as soon as their own v10r2 training ends; their
# Round102 decodes run on other nodes.  Smoke first, then occupy all three.
wait_file "$release/train_corr25_captioni2v_delta_tf025_replay025_s64_v10r2/TRAINING_COMPLETE"
wait_file "$release/train_noanchor_captioni2v_delta_tf025_replay025_s64_v10r2/TRAINING_COMPLETE"
wait_file "$release/train_corr25_captioni2v_delta_tf025_replay010_s64_v10r2/TRAINING_COMPLETE"
if [ ! -f "$release/train_cross_captioni2v_delta_tf025_replay025_s1_smoke_v11/checkpoint-00000001/receipt.json" ]; then
  bash "$launcher" _one 143808 auh7-1b-gpu-233 \
    cross_captioni2v_delta_tf025_replay025_s1_smoke_v11 cross_sparse 0.25 1
fi
jq -e '
  .complete == true and
  .training_contract.training_interface == "first_phase_caption_i2v" and
  .training_contract.route_operator == "cross_sparse" and
  .training_contract.true_training_memory_fraction_strictly_above_half == true and
  .training_contract.dummy_or_padding_allocations == false and
  .gradient_coverage.nonzero_tensor_count > 0
' "$release/train_cross_captioni2v_delta_tf025_replay025_s1_smoke_v11/checkpoint-00000001/receipt.json" >/dev/null

launch_now() {
  local job="$1" node="$2" experiment="$3" operator="$4" target_weight="$5"
  if [ -e "$release/train_${experiment}" ]; then
    echo "existing output; not launching duplicate: $experiment"
    return 0
  fi
  nohup bash "$launcher" _one "$job" "$node" "$experiment" "$operator" "$target_weight" 32 \
    >"$logs/controller_${experiment}_${node}.log" 2>&1 &
  echo "$! $experiment"
}
launch_now 143808 auh7-1b-gpu-233 cross_captioni2v_delta_tf025_replay025_s32_v11 cross_sparse 0.25
launch_now 143808 auh7-1b-gpu-268 cross_captioni2v_delta_pure_replay025_s32_v11 cross_sparse 0.0
launch_now 141620 auh7-1b-gpu-226 targetgate_captioni2v_delta_tf025_replay025_s32_v11 self_target_gated_kernel25 0.25

# Node315 is also the Round102 replay-.10 decode node.  Wait for its last E04
# output before starting the temporal-kernel training there.
(
  wait_file "$release/dynaedit_fullgrid_v2/corr25_captioni2v_delta_tf025_replay010_early3_v10r2/step_00000064/e04/E04_close-door-then-drawer_corr25_captioni2v_delta_tf025_replay010_early3_v10r2_S64_ONLINE_ANCHOR_REAL_SGA_ANC.mp4"
  launch_now 143808 auh7-1b-gpu-315 temporalkernel_captioni2v_delta_tf025_replay025_s32_v11 self_temporal_kernel 0.25
) >"$logs/temporal_kernel_deferred_controller.log" 2>&1 &

bash "$watcher"
printf '%s\n' ROUTE_SCREEN_V11_LAUNCHED
