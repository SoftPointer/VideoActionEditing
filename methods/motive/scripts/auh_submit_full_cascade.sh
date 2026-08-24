#!/usr/bin/env bash
set -Eeuo pipefail

action="${1:-plan}"
feature_mode="${2:-array}"
ssh_host="${MOTIVE_SSH_HOST:-auh}"
remote_repo="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
remote_data="${MOTIVE_DATA_ROOT:-${remote_repo}/data/goku/subject_movement/extracted}"
remote_output="${MOTIVE_OUTPUT_ROOT:-${remote_repo}/methods/motive/outputs/goku_subject_movement_full}"
cpu_python="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
qwen_python="${QWEN_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
qwen_model="${QWEN_MODEL_PATH:-/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
feature_tasks="${MOTIVE_FEATURE_ARRAY_COUNT:-8}"

if [[ "${action}" != "plan" && "${action}" != "submit" ]]; then
  echo "usage: $0 [plan|submit] [array|single]" >&2
  exit 2
fi
if [[ "${feature_mode}" != "array" && "${feature_mode}" != "single" ]]; then
  echo "feature mode must be array or single" >&2
  exit 2
fi
if ! [[ "${feature_tasks}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid MOTIVE_FEATURE_ARRAY_COUNT=${feature_tasks}" >&2
  exit 2
fi

if [[ "${action}" == "plan" ]]; then
  cat <<EOF
No jobs submitted. Planned AUH dependency chain:
  1. full rules CPU
  2. full features CPU (${feature_mode}, tasks=${feature_tasks})
  3. feature-array merge (array mode only)
  4. Qwen2.5-VL visual judge: 8 one-GPU array tasks
  5. Qwen merge + rule/feature/Qwen fusion CPU
  6. provenance-bound human-review template CPU

Remote repo:   ${remote_repo}
Data root:     ${remote_data}
Output root:   ${remote_output}
Submit with:   $0 submit ${feature_mode}
EOF
  exit 0
fi

ssh "${ssh_host}" bash -s -- \
  "${remote_repo}" "${remote_data}" "${remote_output}" "${feature_mode}" \
  "${feature_tasks}" "${cpu_python}" "${qwen_python}" "${qwen_model}" <<'REMOTE'
set -Eeuo pipefail

repo_root="$1"
data_root="$2"
output_root="$3"
feature_mode="$4"
feature_tasks="$5"
cpu_python="$6"
qwen_python="$7"
qwen_model="$8"
source_script_root="${repo_root}/methods/motive/scripts"
mkdir -p "${repo_root}/logs"
cd "${repo_root}"

snapshot_root="${output_root}/source_snapshot"
snapshot_temporary="${snapshot_root}.tmp.$$"
if [[ -e "${snapshot_root}" || -e "${snapshot_temporary}" ]]; then
  echo "source snapshot already exists; use a fresh MOTIVE_OUTPUT_ROOT: ${snapshot_root}" >&2
  exit 2
fi
mkdir -p "${snapshot_temporary}/scripts"
cp -a "${repo_root}/methods/motive/motive" "${snapshot_temporary}/motive"
for script in \
  auh_rules_full_cpu.sbatch \
  auh_features_full_cpu.sbatch \
  auh_features_finalize_cpu.sbatch \
  auh_qwen_visual_array.sbatch \
  auh_qwen_finalize_cpu.sbatch \
  auh_human_review_prepare_cpu.sbatch; do
  cp "${source_script_root}/${script}" "${snapshot_temporary}/scripts/${script}"
done
(
  cd "${snapshot_temporary}"
  find motive scripts -type f \( -name '*.py' -o -name '*.sbatch' \) -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)
chmod -R a-w "${snapshot_temporary}"
mv "${snapshot_temporary}" "${snapshot_root}"
code_root="${snapshot_root}"
script_root="${snapshot_root}/scripts"

rules_job="$(
  sbatch --parsable \
    --chdir="${repo_root}" \
    --output="${repo_root}/logs/motive_rules_%j.log" \
    --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_DATA_ROOT=${data_root},MOTIVE_OUTPUT_ROOT=${output_root},PYTHON_BIN=${cpu_python}" \
    "${script_root}/auh_rules_full_cpu.sbatch"
)"

if [[ "${feature_mode}" == "array" ]]; then
  feature_job="$(
    sbatch --parsable \
      --dependency="afterok:${rules_job}" \
      --array="0-$((feature_tasks - 1))%${feature_tasks}" \
      --chdir="${repo_root}" \
      --output="${repo_root}/logs/motive_features_%A_%a.log" \
      --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_DATA_ROOT=${data_root},MOTIVE_OUTPUT_ROOT=${output_root},PYTHON_BIN=${cpu_python},MOTIVE_FEATURE_ARRAY_COUNT=${feature_tasks}" \
      "${script_root}/auh_features_full_cpu.sbatch"
  )"
  feature_merge_job="$(
    sbatch --parsable \
      --dependency="afterok:${feature_job}" \
      --chdir="${repo_root}" \
      --output="${repo_root}/logs/motive_features_merge_%j.log" \
      --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_OUTPUT_ROOT=${output_root},PYTHON_BIN=${cpu_python},MOTIVE_FEATURE_ARRAY_COUNT=${feature_tasks}" \
      "${script_root}/auh_features_finalize_cpu.sbatch"
  )"
  qwen_dependency="${feature_merge_job}"
  qwen_input="${output_root}/feature_candidates/qwen_queue.jsonl"
  fuse_feature_dir="${output_root}/features_merged/features"
else
  feature_job="$(
    sbatch --parsable \
      --dependency="afterok:${rules_job}" \
      --chdir="${repo_root}" \
      --output="${repo_root}/logs/motive_features_%j.log" \
      --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_DATA_ROOT=${data_root},MOTIVE_OUTPUT_ROOT=${output_root},PYTHON_BIN=${cpu_python}" \
      "${script_root}/auh_features_full_cpu.sbatch"
  )"
  feature_merge_job="not-needed"
  qwen_dependency="${feature_job}"
  qwen_input="${output_root}/features_single/export/qwen_queue.jsonl"
  fuse_feature_dir="${output_root}/features_single/stage/features"
fi

qwen_job="$(
  sbatch --parsable \
    --dependency="afterok:${qwen_dependency}" \
    --array="0-7%8" \
    --chdir="${repo_root}" \
    --output="${repo_root}/logs/motive_qwen_%A_%a.log" \
    --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_DATA_ROOT=${data_root},MOTIVE_OUTPUT_ROOT=${output_root},MOTIVE_QWEN_INPUT=${qwen_input},MOTIVE_QWEN_NUM_SHARDS=8,QWEN_PYTHON_BIN=${qwen_python},QWEN_MODEL_PATH=${qwen_model}" \
    "${script_root}/auh_qwen_visual_array.sbatch"
)"
finalize_job="$(
  sbatch --parsable \
    --dependency="afterok:${qwen_job}" \
    --chdir="${repo_root}" \
    --output="${repo_root}/logs/motive_qwen_merge_%j.log" \
    --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_OUTPUT_ROOT=${output_root},MOTIVE_QWEN_INPUT=${qwen_input},MOTIVE_FUSE_FEATURE_DIR=${fuse_feature_dir},MOTIVE_QWEN_NUM_SHARDS=8,PYTHON_BIN=${cpu_python}" \
    "${script_root}/auh_qwen_finalize_cpu.sbatch"
)"
review_pack_job="$(
  sbatch --parsable \
    --dependency="afterok:${finalize_job}" \
    --chdir="${repo_root}" \
    --output="${repo_root}/logs/motive_review_pack_%j.log" \
    --export="ALL,MOTIVE_REPO_ROOT=${repo_root},MOTIVE_CODE_ROOT=${code_root},MOTIVE_OUTPUT_ROOT=${output_root},PYTHON_BIN=${cpu_python}" \
    "${script_root}/auh_human_review_prepare_cpu.sbatch"
)"

printf 'rules_job=%s\n' "${rules_job}"
printf 'feature_job=%s\n' "${feature_job}"
printf 'feature_merge_job=%s\n' "${feature_merge_job}"
printf 'qwen_array_job=%s\n' "${qwen_job}"
printf 'finalize_job=%s\n' "${finalize_job}"
printf 'review_pack_job=%s\n' "${review_pack_job}"
printf 'source_snapshot=%s\n' "${snapshot_root}"
REMOTE
