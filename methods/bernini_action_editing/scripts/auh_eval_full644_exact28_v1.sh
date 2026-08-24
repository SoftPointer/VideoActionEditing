#!/usr/bin/env bash
# Matched exact28 inference: base plus three independently trained exact644 adapters.

set -Eeuo pipefail

selection="${FULL644_EVAL_SELECTION:?}"
eval_root="${FULL644_EVAL_ROOT:?}"
runner="${FULL644_EVAL_RUNNER:?}"
archive="${FULL644_EVAL_INFER_ARCHIVE:?}"
archive_sha="${FULL644_EVAL_INFER_ARCHIVE_SHA256:?}"
revision="${FULL644_EVAL_INFER_REVISION:?}"
variants="${FULL644_EVAL_VARIANTS:-all}"
slot_count="${FULL644_EVAL_SLOT_COUNT:-3}"
node_offset="${FULL644_EVAL_NODE_OFFSET:-0}"

nodes=(auh7-1b-gpu-226 auh7-1b-gpu-249 auh7-1b-gpu-257)
jobs=(141620 141618 141619)
train_roots=(
  /vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/runs/exact644_pcgrad_seed20260820_v1
  /vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/runs/exact644_pcgrad_seed20260821_replica2
  /vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_self_generated_anchor_v1_20260818/runs/exact644_pcgrad_seed20260822_replica3
)
labels=(seed20260820 seed20260821 seed20260822)

[[ -f "${selection}" && -x "${runner}" && -f "${archive}" ]]
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]]
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]]
[[ "${slot_count}" =~ ^[123]$ ]]
[[ "${node_offset}" =~ ^[01]$ ]]
(( node_offset + slot_count <= 3 ))
case "${variants}" in
  all|frozen,seed20260820|seed20260821,seed20260822) ;;
  *) echo "unsupported variant projection: ${variants}" >&2; exit 2 ;;
esac

python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
roots_to_check=("${train_roots[@]}")
if [[ "${variants}" == frozen,seed20260820 ]]; then roots_to_check=("${train_roots[0]}"); fi
if [[ "${variants}" == seed20260821,seed20260822 ]]; then roots_to_check=("${train_roots[1]}" "${train_roots[2]}"); fi
for root in "${roots_to_check[@]}"; do
  "${python_bin}" - "${root}/checkpoint-00000644/receipt.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["global_step"] == x["max_steps"] == 644
assert x["dataset_row_count"] == x["consumed_unique_row_count"] == 644
assert x["all_manifest_rows_consumed_exactly_once"] is True
assert x["gradient_coverage"]["active_tensor_count"] == 480
assert x["gradient_coverage"]["trainable_tensor_count"] == 480
assert x["gradient_coverage"]["active_blocks"] == list(range(30))
assert x["scientific_claim_authorized"] is False
PY
done

mkdir -m 0755 -p "${eval_root}/logs" "${eval_root}/media" "${eval_root}/control"
if [[ -e "${eval_root}/control/selection.json" ]]; then
  cmp "${selection}" "${eval_root}/control/selection.json"
else
  cp "${selection}" "${eval_root}/control/selection.json"
  chmod 0444 "${eval_root}/control/selection.json"
fi
tasks="${eval_root}/control/tasks.tsv"
if [[ ! -e "${tasks}" ]]; then
  "${python_bin}" - "${selection}" "${tasks}" "${train_roots[@]}" <<'PY'
import base64, json, sys
selection=json.load(open(sys.argv[1], encoding="utf-8"))
roots=sys.argv[3:]
assert selection["evaluation_row_count"] == len(selection["rows"]) == 28
with open(sys.argv[2], "x", encoding="utf-8") as out:
    for row in selection["rows"]:
        ins=base64.b64encode(row["instruction"].encode()).decode()
        out.write("\t".join((row["iid"], row["family"], row["source_video_path"], ins,
                             "frozen", "-")) + "\n")
        for label,root in zip(("seed20260820","seed20260821","seed20260822"),roots):
            out.write("\t".join((row["iid"], row["family"], row["source_video_path"], ins,
                                 label, root+"/checkpoint-00000644")) + "\n")
PY
  chmod 0444 "${tasks}"
fi
[[ "$(wc -l <"${tasks}" | tr -d ' ')" == 112 ]]

run_one() {
  local holder="$1" node="$2" iid="$3" family="$4" source="$5" encoded="$6" label="$7" checkpoint="$8"
  local instruction output receipt log mode attempt
  instruction="$(printf '%s' "${encoded}" | base64 --decode)"
  mkdir -p "${eval_root}/media/${iid}"
  output="${eval_root}/media/${iid}/${label}.mp4"
  receipt="${output}.receipt.json"
  log="${eval_root}/logs/${iid}__${label}.log"
  if [[ -f "${output}" && -f "${receipt}" ]]; then return 0; fi
  [[ ! -e "${output}" && ! -e "${receipt}" ]]
  if [[ "${label}" == frozen ]]; then mode=base; checkpoint=""; else mode=adapter; fi
  : >"${log}"
  for attempt in 1 2; do
    printf 'attempt=%s iid=%s family=%s label=%s\n' "${attempt}" "${iid}" "${family}" "${label}" >>"${log}"
    if srun --jobid="${holder}" --overlap --exact --nodes=1 --ntasks=1 \
      --nodelist="${node}" --cpus-per-task=32 --mem=64G --gres=gpu:mi210:8 \
      env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
      ROCR_VISIBLE_DEVICES=0,1,2,3 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 \
      FULL644_EVAL_MODE="${mode}" FULL644_EVAL_SOURCE_VIDEO="${source}" \
      FULL644_EVAL_INSTRUCTION="${instruction}" FULL644_EVAL_OUTPUT="${output}" \
      FULL644_EVAL_ADAPTER="${checkpoint}" FULL644_EVAL_INFER_ARCHIVE="${archive}" \
      FULL644_EVAL_INFER_ARCHIVE_SHA256="${archive_sha}" FULL644_EVAL_INFER_REVISION="${revision}" \
      "${runner}" </dev/null >>"${log}" 2>&1; then
      [[ -f "${output}" && -f "${receipt}" ]]
      return 0
    fi
    [[ ! -e "${output}" && ! -e "${receipt}" ]] || return 1
    sleep 5
  done
  return 1
}

worker() {
  local slot="$1" index=0 status=0
  while IFS=$'\t' read -r iid family source encoded label checkpoint; do
    selected=true
    if [[ "${variants}" == frozen,seed20260820 && "${label}" != frozen && "${label}" != seed20260820 ]]; then selected=false; fi
    if [[ "${variants}" == seed20260821,seed20260822 && "${label}" != seed20260821 && "${label}" != seed20260822 ]]; then selected=false; fi
    if (( index % slot_count == slot )) && [[ "${selected}" == true ]]; then
      physical_slot=$((node_offset + slot))
      run_one "${jobs[$physical_slot]}" "${nodes[$physical_slot]}" "${iid}" "${family}" \
        "${source}" "${encoded}" "${label}" "${checkpoint}" || status=1
      (( status == 0 )) || return 1
    fi
    index=$((index + 1))
  done <"${tasks}"
}

if (( slot_count == 1 )); then
  worker 0
else
  pids=()
  slot=0
  while (( slot < slot_count )); do
    worker "${slot}" & pids+=("$!")
    slot=$((slot + 1))
  done
  status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  (( status == 0 ))
fi

observed="$(find "${eval_root}/media" -type f -name '*.mp4' | wc -l | tr -d ' ')"
receipts="$(find "${eval_root}/media" -type f -name '*.mp4.receipt.json' | wc -l | tr -d ' ')"
if [[ "${variants}" == all ]]; then
  [[ "${observed}" == 112 && "${receipts}" == 112 ]]
  printf 'training_coverage=644\nevaluation_rows=28\naction_families=28\nvariants_per_row=4\nvideo_count=112\nsource_onset_policy=hard1_every_step\n' \
    >"${eval_root}/EVALUATION_COMPLETE"
  chmod 0444 "${eval_root}/EVALUATION_COMPLETE"
elif [[ "${variants}" == frozen,seed20260820 ]]; then
  [[ "${observed}" == 56 && "${receipts}" == 56 ]]
  printf 'variants=frozen,seed20260820\nvideo_count=56\n' >"${eval_root}/PARTIAL_FROZEN_SEED20260820_COMPLETE"
  chmod 0444 "${eval_root}/PARTIAL_FROZEN_SEED20260820_COMPLETE"
else
  projected="$(find "${eval_root}/media" -type f \( -name 'seed20260821.mp4' -o -name 'seed20260822.mp4' \) | wc -l | tr -d ' ')"
  [[ "${projected}" == 56 ]]
  printf 'variants=seed20260821,seed20260822\nvideo_count=56\n' >"${eval_root}/PARTIAL_SEED20260821_SEED20260822_COMPLETE"
  chmod 0444 "${eval_root}/PARTIAL_SEED20260821_SEED20260822_COMPLETE"
fi
