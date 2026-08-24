#!/usr/bin/env bash
# Prospective reward audit on four sealed confirmation IIDs and three fixed
# inference seeds. Each of eight SP4 workers owns exactly one model condition.

set -Eeuo pipefail

job_id=135096
output_root="${REWARD_UNSEEN_EVAL_ROOT:?set REWARD_UNSEEN_EVAL_ROOT}"
manifest="${output_root}/release/unseen_multiseed_manifest_v1.json"
one="${output_root}/release/auh_infer_reward_training_one_v1.sh"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
capacity_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_capacity_fivearm_20260815_v3/formal_u320/action_only_sft-formal_u320
reward_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/reward_antihack_eightarm_20260815_v5/formal_u320

condition_names=(
  frozen_base
  action_sft_u80
  action_sft_u160
  action_sft_u320
  detached_rotate_u80
  detached_rotate_u160
  detached_incomplete_u80
  detached_incomplete_u160
)
condition_modes=(
  frozen_base
  trained_adapter
  trained_adapter
  trained_adapter
  trained_adapter
  trained_adapter
  trained_adapter
  trained_adapter
)
condition_steps=(0 80 160 320 80 160 80 160)
condition_checkpoints=(
  ""
  "${capacity_root}/checkpoint-00000080"
  "${capacity_root}/checkpoint-00000160"
  "${capacity_root}/checkpoint-00000320"
  "${reward_root}/detached_rotate_w1-formal_u320/checkpoint-00000080"
  "${reward_root}/detached_rotate_w1-formal_u320/checkpoint-00000160"
  "${reward_root}/detached_incomplete_w1-formal_u320/checkpoint-00000080"
  "${reward_root}/detached_incomplete_w1-formal_u320/checkpoint-00000160"
)
nodes=(
  auh7-1b-gpu-245 auh7-1b-gpu-245
  auh7-1b-gpu-246 auh7-1b-gpu-246
  auh7-1b-gpu-247 auh7-1b-gpu-247
  auh7-1b-gpu-248 auh7-1b-gpu-248
)
groups=(0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7 0,1,2,3 4,5,6,7)

state="$(squeue -j "${job_id}" -h -o '%T|%N|%b')"
[[ "${state}" == RUNNING\|auh7-1b-gpu-\[245-248\]\|gres/gpu:mi210:8 ]]
[[ -f "${manifest}" && ! -L "${manifest}" && -x "${one}" && ! -L "${one}" ]]
[[ ! -e "${output_root}/eval" && ! -L "${output_root}/eval" ]]
[[ "${#condition_names[@]}" == 8 && "${#condition_modes[@]}" == 8 ]]
[[ "${#condition_steps[@]}" == 8 && "${#condition_checkpoints[@]}" == 8 ]]

"${python_bin}" - "${manifest}" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
assert value["schema_version"] == "bernini-reward-unseen-multiseed-manifest-v1"
assert value["inference_seeds"] == [2026081701, 2026081702, 2026081703]
assert value["seed_selection"] == {
    "registered_before_inference": True,
    "posthoc_seed_filtering": False,
    "best_of_k_selection": False,
    "same_seed_across_all_conditions": True,
}
rows = value["rows"]
assert len(rows) == 4
assert len({row["iid"] for row in rows}) == 4
assert all(row["analysis_split"] == "confirmation" for row in rows)
assert not ({row["iid"] for row in rows} & set(value["training_iids_excluded"]))
authority = pathlib.Path(value["source_authority"]["path"])
assert authority.is_file() and not authority.is_symlink()
assert hashlib.sha256(authority.read_bytes()).hexdigest() == value["source_authority"]["sha256"]
print("MANIFEST_PASS rows=4 seeds=3 posthoc_filtering=false")
PY

for ((index=1; index<8; index++)); do
  checkpoint="${condition_checkpoints[$index]}"
  receipt="${checkpoint}/receipt.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]]
  actual_step="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["global_step"])' "${receipt}")"
  [[ "${actual_step}" == "${condition_steps[$index]}" ]]
done

for ((row=0; row<4; row++)); do
  source_video="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])]["source_video_path"])' "${manifest}" "${row}")"
  expected_sha="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])]["source_video_sha256"])' "${manifest}" "${row}")"
  [[ -f "${source_video}" && ! -L "${source_video}" ]]
  [[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${expected_sha}" ]]
  [[ "$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "${source_video}")" == 81 ]]
  [[ "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${source_video}")" == 3.240000 ]]
done

mkdir -p "${output_root}/logs" "${output_root}/eval"

payload='set -Eeuo pipefail
output_root="$1"; worker="$2"; group="$3"; one="$4"; manifest="$5"; python_bin="$6"
condition="$7"; mode="$8"; checkpoint="$9"
field() { "$python_bin" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"rows\"][int(sys.argv[2])][sys.argv[3]])" "$manifest" "$1" "$2"; }
seed_value() { "$python_bin" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[\"inference_seeds\"][int(sys.argv[2])])" "$manifest" "$1"; }
for ((cell=0; cell<12; cell++)); do
  row_index=$((cell / 3)); seed_index=$((cell % 3))
  iid="$(field "$row_index" iid)"
  source_video="$(field "$row_index" source_video_path)"
  instruction="$(field "$row_index" instruction)"
  seed="$(seed_value "$seed_index")"
  output="$output_root/eval/$iid/seed-$seed/$condition.mp4"
  log="$output_root/logs/eval-$iid-seed-$seed-$condition.log"
  [[ ! -e "$output" && ! -L "$output" ]]
  ROCR_VISIBLE_DEVICES="$group" env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    REWARD_EVAL_MODE="$mode" REWARD_EVAL_IID="$iid" REWARD_EVAL_SEED="$seed" \
    REWARD_EVAL_SOURCE_VIDEO="$source_video" REWARD_EVAL_INSTRUCTION="$instruction" \
    REWARD_EVAL_OUTPUT_VIDEO="$output" REWARD_EVAL_ADAPTER_CHECKPOINT="$checkpoint" \
    "$one" >"$log" 2>&1
  echo "OUTPUT worker=$worker condition=$condition iid=$iid seed=$seed physical_gpus=$group"
done'

pids=()
for worker in 0 1 2 3 4 5 6 7; do
  node="${nodes[$worker]}"
  group="${groups[$worker]}"
  condition="${condition_names[$worker]}"
  mode="${condition_modes[$worker]}"
  checkpoint="${condition_checkpoints[$worker]}"
  srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    --nodelist="${node}" --cpus-per-task=32 --mem=240G --gres=gpu:mi210:8 \
    bash -c "${payload}" _ "${output_root}" "${worker}" "${group}" \
      "${one}" "${manifest}" "${python_bin}" "${condition}" "${mode}" "${checkpoint}" \
      >"${output_root}/logs/worker-${worker}.log" 2>&1 &
  pids+=("$!")
  echo "LAUNCHED worker=${worker} condition=${condition} node=${node} physical_gpus=${group} pid=${pids[-1]}"
done

overall=0
for worker in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$worker]}"; then
    echo "COMPLETED worker=${worker} condition=${condition_names[$worker]} node=${nodes[$worker]}"
  else
    overall=1
    echo "FAILED worker=${worker} condition=${condition_names[$worker]} node=${nodes[$worker]}" >&2
  fi
done
(( overall == 0 ))
[[ "$(find "${output_root}/eval" -type f -name '*.mp4' | wc -l)" == 96 ]]
[[ "$(find "${output_root}/eval" -type f -name '*.receipt.json' | wc -l)" == 96 ]]
echo COMPLETE >"${output_root}/eval/EVALUATION_COMPLETE"
echo "ALL_COMPLETE outputs=96 parent_allocation_cancelled=false posthoc_seed_filtering=false"
