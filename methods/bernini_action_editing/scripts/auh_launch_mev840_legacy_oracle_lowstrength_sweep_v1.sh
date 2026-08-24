#!/usr/bin/env bash
set -euo pipefail

# Fresh zero-update calibration of the legacy MEV840 activity25 Q/K operator.
# The normalized real target is an oracle-only anchor.  This is not a v15 graph run.

usage() {
  echo "usage: $0 launch | worker s005_k05|s010_k05|s025_k05|s005_k10|s010_k10|s025_k10" >&2
  exit 2
}

mode="${1:-}"
arm="${2:-}"
if [ "$mode" = launch ]; then
  [ "$#" -eq 1 ] || usage
elif [ "$mode" = worker ]; then
  [ "$#" -eq 2 ] || usage
else
  usage
fi

stage=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1
mev_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev_action_anchor_target_gap16_20260819_v1
dev="$stage/source-online-anchor-targetowned-qk-decode-v14r3-gradgeom-dfix2/methods/bernini_action_editing"
runtime="$stage/source-be31323/methods/bernini_action_editing"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
source_video="$mev_root/preprocessed_sources/840b214afead/source-exact81.mp4"
oracle_anchor="$mev_root/review_html_v1/media/840b214afead/real_target.mp4"
output_root="$stage/mev840_legacy_oracle_activity25_lowstrength_sweep_v1_20260821"
control_root="$stage/mev840_legacy_oracle_activity25_lowstrength_sweep_v1_20260821_control"

source_sha=a6e42a447d7fef26b073878c551a631afd6371987910a332e51e4a9e4dfb4646
oracle_anchor_sha=355535f4f5ff83581c2286dfb70a64c7f5131f5ae81d76fbb6351b2aa972baf0
python_sha=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
runner_sha=4ed2f22df876613ecfc720a662a48f8e028eb89fe9778e491bc962a4f8f68ab1
infer_sha=dd3558a4c38c5541ba6b7ad455ac599f43eb48b1b56f207a07776c9e1819145f
controller_sha=1427a4908e0a4239e95a353d3406c41cb77fdb7f0be81727126a2cfd23f1f3ad
qk_sha=37941e30853b16fa242a7c91940620069f87a1a975d2ecf610f3cde800557a99
cross_sha=cbbbc490711f4f1131a701d042309b024fd5b3922f2d88e39e35bff7d39396f6
guided_sha=3e7f8e449447c8cc0f2678da82b9e298d84d0b5b9f729281a5b19369cba7ddc6
lora_sha=0c79faa8417a40a5735571db3a5ba828d6aa977d7d0507a5bfcb63368c07728d
native_sha=bf402cd65257121d1ebedcc83c2c59965b37305a36b0b5a6327241e74d7b4f42
source_audit_sha=9ae3a41e52f520f66ebcddba331b26837a5c8291426d13379eaa4c8a01a80e02

source_caption='The woman with reddish-blonde hair stands on a treadmill, holding a water bottle while looking forward.'
target_caption='A young woman with long reddish-blonde hair, wearing a mint green athletic outfit, stands on a treadmill in a modern gym with blue and white striped pillars, holding a water bottle. In this continuous shot, The woman turns her head to the left and places her water bottle onto the treadmill. The same subject identity, scene, lighting, framing, and camera remain stable.'
instruction='Use the source video as the sole authority for identity, appearance, clothing, object instances, background, lighting, framing, camera and initial state. Frame 0 must retain the original source state; do not pre-apply the requested endpoint. Perform only this temporal edit: The woman turns her head to the left and places her water bottle onto the treadmill. The edit must be one continuous 81-frame video at 25 fps and must not introduce appearance changes as a substitute for the requested action.'
anchor_caption="$target_caption"
anchor_noop='The same woman remains looking forward while holding the same bottle and does not turn her head left or place the bottle onto the treadmill.'

verify_sha() {
  local path="$1" expected="$2" actual
  [ -f "$path" ] && [ ! -L "$path" ]
  actual="$(sha256sum -- "$path")"
  actual="${actual%% *}"
  [ "$actual" = "$expected" ]
}

verify_authority() {
  verify_sha "$python_bin" "$python_sha"
  verify_sha "$source_video" "$source_sha"
  verify_sha "$oracle_anchor" "$oracle_anchor_sha"
  verify_sha "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py" "$runner_sha"
  verify_sha "$dev/infer_anchor_sga_anc_event_v1.py" "$infer_sha"
  verify_sha "$dev/anchor_sga_anc_controller.py" "$controller_sha"
  verify_sha "$dev/anchor_qk_transport.py" "$qk_sha"
  verify_sha "$dev/anchor_cross_attention_transport.py" "$cross_sha"
  verify_sha "$dev/guided_source_aligned_controller.py" "$guided_sha"
  verify_sha "$dev/infer_lora.py" "$lora_sha"
  verify_sha "$dev/infer_native_identity_generation_canary.py" "$native_sha"
  verify_sha "$dev/infer_source_aligned_controller_oracle.py" "$source_audit_sha"
  [ -d "$bernini_root" ] && [ ! -L "$bernini_root" ]
  [ -d "$veomni_root" ] && [ ! -L "$veomni_root" ]
  [ -d "$checkpoint" ] && [ ! -L "$checkpoint" ]
  [ -f "$checkpoint_manifest" ] && [ ! -L "$checkpoint_manifest" ]
}

set_arm() {
  case "$1" in
    s005_k05)
      expected_job=147881; expected_node=auh7-1b-gpu-213
      transport_strength=0.05; transport_steps=5
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K05
      ;;
    s010_k05)
      expected_job=147873; expected_node=auh7-1b-gpu-284
      transport_strength=0.10; transport_steps=5
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K05
      ;;
    s025_k05)
      expected_job=147871; expected_node=auh7-1b-gpu-232
      transport_strength=0.25; transport_steps=5
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K05
      ;;
    s005_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-268
      transport_strength=0.05; transport_steps=10
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S005_K10
      ;;
    s010_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-315
      transport_strength=0.10; transport_steps=10
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S010_K10
      ;;
    s025_k10)
      expected_job=143808; expected_node=auh7-1b-gpu-233
      transport_strength=0.25; transport_steps=10
      label=MEV840_LEGACY_QK_ORACLE_ACTIVITY25_S025_K10
      ;;
    *) usage ;;
  esac
}

arms=(s005_k05 s010_k05 s025_k05 s005_k10 s010_k10 s025_k10)

if [ "$mode" = launch ]; then
  [ -d "$control_root" ] && [ ! -L "$control_root" ]
  [ ! -e "$output_root" ] && [ ! -L "$output_root" ]
  verify_authority

  # All allocations and all step slots are checked before any output root or
  # work step is created.  Only allocation-owned batch/extern steps may exist.
  for launch_arm in "${arms[@]}"; do
    set_arm "$launch_arm"
    state="$(squeue -h -j "$expected_job" -o '%T|%N')"
    case "$expected_job|$expected_node|$state" in
      "143808|auh7-1b-gpu-268|RUNNING|auh7-1b-gpu-[233,268,292,315]"|\
      "143808|auh7-1b-gpu-315|RUNNING|auh7-1b-gpu-[233,268,292,315]"|\
      "143808|auh7-1b-gpu-233|RUNNING|auh7-1b-gpu-[233,268,292,315]"|\
      "147881|auh7-1b-gpu-213|RUNNING|auh7-1b-gpu-213"|\
      "147873|auh7-1b-gpu-284|RUNNING|auh7-1b-gpu-284"|\
      "147871|auh7-1b-gpu-232|RUNNING|auh7-1b-gpu-232") ;;
      *) echo "allocation contract differs for $launch_arm: $state" >&2; exit 1 ;;
    esac
  done
  for launch_job in 143808 147881 147873 147871; do
    non_system="$(squeue --steps -h -j "$launch_job" -o '%i' | awk '$0 !~ /\.(batch|extern)$/ {print}')"
    [ -z "$non_system" ] || { echo "active work step exists in $launch_job: $non_system" >&2; exit 1; }
  done

  mkdir "$output_root"
  for launch_arm in "${arms[@]}"; do
    set_arm "$launch_arm"
    log="$control_root/$launch_arm.log"
    pid_file="$control_root/$launch_arm.pid"
    [ ! -e "$log" ] && [ ! -e "$pid_file" ]
    nohup srun --jobid="$expected_job" --exclusive --nodes=1 --ntasks=1 \
      --cpus-per-task=64 --gres=gpu:4 --mem=0 --nodelist="$expected_node" \
      bash "$control_root/auh_launch_mev840_legacy_oracle_lowstrength_sweep_v1.sh" worker "$launch_arm" \
      >"$log" 2>&1 &
    printf '%s\n' "$!" >"$pid_file"
  done
  echo "MEV840_LEGACY_ORACLE_LOWSTRENGTH_SWEEP_LAUNCHED $output_root"
  exit 0
fi

set_arm "$arm"
[ "${SLURM_JOB_ID:-}" = "$expected_job" ]
[ "$(hostname -s)" = "$expected_node" ]
verify_authority
verify_sha "$oracle_anchor" "$oracle_anchor_sha"

output="$output_root/$label.mp4"
native_receipt="$output.receipt.json"
worker_receipt="$output.worker.json"
audit="$output.legacy-mev840-lowstrength-audit.json"
complete="$output.complete.json"
for candidate in "$output" "$native_receipt" "$worker_receipt" "$audit" "$complete"; do
  [ ! -e "$candidate" ] && [ ! -L "$candidate" ]
done

scratch="/tmp/mev840-legacy-oracle-lowstrength-${arm}-${SLURM_STEP_ID:-nostep}"
mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MIOPEN_USER_DB_PATH="$scratch/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions"
export TRITON_CACHE_DIR="$scratch/triton"
export PYTHONPATH="$dev:$runtime"

"$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$dev/infer_anchor_sga_anc_trained_editor_decode_v1.py" \
  --bernini-root "$bernini_root" \
  --veomni-root "$veomni_root" \
  --checkpoint "$checkpoint" \
  --checkpoint-content-manifest "$checkpoint_manifest" \
  --source-video "$source_video" \
  --expected-source-sha256 "$source_sha" \
  --anchor-video "$oracle_anchor" \
  --expected-anchor-sha256 "$oracle_anchor_sha" \
  --instruction "$instruction" \
  --source-caption "$source_caption" \
  --target-caption "$target_caption" \
  --anchor-caption "$anchor_caption" \
  --anchor-noop-caption "$anchor_noop" \
  --arm AQK_SGA5 \
  --transport self_target_owned_activity_kernel25_attn_output_v14r2 \
  --transport-strength "$transport_strength" \
  --transport-steps "$transport_steps" \
  --blocks 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29 \
  --field-guidance raw_cfg \
  --field-model first_phase_caption_i2v \
  --source-cfg-scale 4.5 \
  --target-cfg-scale 4.5 \
  --sga-temperature 0.01 \
  --early-candidate-count 5 \
  --initial-noise-proposal-mode keyed_only \
  --anchor-state-mode clean_noised \
  --anchor-cfg-scope shared \
  --anchor-contrast-mode dynamic_static_same_caption \
  --anchor-sigma-cap 1.0 \
  --preservation-mode none \
  --preservation-keep-fraction 0.20 \
  --preservation-outside-scale 0.0 \
  --preservation-dilation 1 \
  --preservation-residual-fraction 0.0 \
  --preservation-object-identity-strength 0.0 \
  --preservation-start-step 0 \
  --preservation-ramp-steps 1 \
  --sga-score-mode global_source_cosine \
  --anchor-candidate-mode single_shared \
  --anchor-spatial-alignment none \
  --event01-forced-role-proposal-index -1 \
  --output "$output"

# Compute nodes deliberately stop before ffprobe.  The immutable native receipt
# and video are closed by the separately pinned login-node finalizer.
[ -f "$output" ] && [ ! -L "$output" ]
[ -f "$native_receipt" ] && [ ! -L "$native_receipt" ]
jq -e \
  --arg source_sha "$source_sha" --arg anchor_sha "$oracle_anchor_sha" \
  --argjson strength "$transport_strength" --argjson steps "$transport_steps" '
    .complete == true and .training_performed == false and .optimization_steps == 0 and
    .source.sha256 == $source_sha and .pure_t2v_anchor.sha256 == $anchor_sha and
    .mechanism.transport_strength == $strength and
    .mechanism.decode_audit_contract.transport_steps == $steps
  ' "$native_receipt" >/dev/null
video_sha="$(sha256sum -- "$output")"; video_sha="${video_sha%% *}"
receipt_sha="$(sha256sum -- "$native_receipt")"; receipt_sha="${receipt_sha%% *}"
worker_tmp="$worker_receipt.tmp.${SLURM_STEP_ID}"
jq -n \
  --arg arm "$arm" --arg job "$expected_job" --arg step "${SLURM_STEP_ID}" \
  --arg node "$expected_node" --arg output "$output" --arg output_sha "$video_sha" \
  --arg receipt "$native_receipt" --arg receipt_sha "$receipt_sha" \
  --argjson strength "$transport_strength" --argjson transport_steps "$transport_steps" \
  '{schema:"mev840-legacy-oracle-lowstrength-worker-v1",complete:true,
    arm:$arm,slurm:{job_id:$job,step_id:$step,node:$node},
    zero_update:true,training_performed:false,optimization_steps:0,
    mechanism:{strength:$strength,transport_steps:$transport_steps},
    output:{path:$output,sha256:$output_sha},native_receipt:{path:$receipt,sha256:$receipt_sha}}' \
  >"$worker_tmp"
jq -e '.complete and .zero_update and (.optimization_steps == 0)' "$worker_tmp" >/dev/null
mv "$worker_tmp" "$worker_receipt"
echo "MEV840_LEGACY_ORACLE_LOWSTRENGTH_INFERENCE_COMPLETE $arm $output"
