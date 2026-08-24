#!/usr/bin/env bash
set -Eeuo pipefail

experiment_root="${1:?experiment root}"
source_tree_sha256="${2:?source tree SHA-256}"
parent_run="${3:?parent R7 run}"
model_workspace="${4:?model workspace}"
python_bin="${5:?Python executable}"

allowed_prefix=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto
case "${experiment_root}" in
  "${allowed_prefix}"/goku_repr_auto_r9_*) ;;
  *)
    echo "[r9-submit] unsafe experiment root: ${experiment_root}" >&2
    exit 2
    ;;
esac
if [[ ! "${source_tree_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[r9-submit] invalid source tree SHA-256" >&2
  exit 2
fi

source_snapshot="${experiment_root}/source_snapshot"
controller="${source_snapshot}/methods/motive/scripts/auh_r9_representation_controller.sh"
registry="${source_snapshot}/methods/motive/configs/instruction_video_editor_registry_v1.json"
receipt="${experiment_root}/submission.json"
mkdir -p "${experiment_root}/logs" "${experiment_root}/provenance"
exec 9>"${experiment_root}/.submission.lock"
if ! flock -n 9; then
  echo "[r9-submit] another submitter owns this experiment" >&2
  exit 4
fi
if [[ -s "${receipt}" ]]; then
  echo "[r9-submit] existing receipt"
  sed -n "1,160p" "${receipt}"
  exit 0
fi
for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/SOURCE_PROVENANCE.json" \
  "${controller}" \
  "${registry}"; do
  if [[ ! -s "${required}" ]] || [[ -L "${required}" ]]; then
    echo "[r9-submit] source snapshot is incomplete: ${required}" >&2
    exit 2
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  echo "[r9-submit] Python is not executable: ${python_bin}" >&2
  exit 2
fi

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

candidate_done_sha256=200bc59872547fb7029aa53ea422a86986f9df8fa614e487b81ae7dce1644c07
track_cache_done_sha256=3d64b89bd17b6880239d58b78b5718c50ddfa0b67ed8111b73cf147a0399ede2
visual_features_done_sha256=f92082f58e25a40d636633abb1669c950044b37dd3b8abca207ee07356125b71
visual_candidates_sha256=2026b87cc01107c77b6a1cc993bfb6eb64e0381a9b52f8a87ac59ac661db628d
screen_inputs_receipt_sha256=fad2f1d5c6ea30b6f0a33f150a256c22fbd57938d44ae88a5014f9ddf926c6fb

candidate_manifest_dir="${parent_run}/expansion/candidate_temporal_screen_v1"
track_cache_final="${parent_run}/expansion/candidate_track_cache_v1/final"
screen_input_bundle="${parent_run}/expansion/screen_inputs_v14"
visual_features_final="${screen_input_bundle}/visual_features_v1/final"
visual_candidates_manifest="${screen_input_bundle}/visual_candidates_v1/candidates.jsonl"
screen_inputs_receipt="${parent_run}/provenance/screen_inputs_v14.receipt.txt"
for required_dir in \
  "${candidate_manifest_dir}" \
  "${track_cache_final}" \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")" \
  "${model_workspace}"; do
  if [[ ! -d "${required_dir}" ]] || [[ -L "${required_dir}" ]]; then
    echo "[r9-submit] preflight directory differs: ${required_dir}" >&2
    exit 2
  fi
done
for anchored in \
  "${candidate_manifest_dir}/done.json|${candidate_done_sha256}" \
  "${track_cache_final}/done.json|${track_cache_done_sha256}" \
  "${visual_features_final}/done.json|${visual_features_done_sha256}" \
  "${visual_candidates_manifest}|${visual_candidates_sha256}" \
  "${screen_inputs_receipt}|${screen_inputs_receipt_sha256}"; do
  anchored_path="${anchored%%|*}"
  anchored_sha256="${anchored#*|}"
  if [[ ! -f "${anchored_path}" ]] \
    || [[ -L "${anchored_path}" ]] \
    || [[ "$(sha256sum "${anchored_path}" | awk '{print $1}')" \
      != "${anchored_sha256}" ]]; then
    echo "[r9-submit] preflight input differs: ${anchored_path}" >&2
    exit 2
  fi
done
if [[ "$(stat -c '%h' "${screen_inputs_receipt}")" != "1" ]] \
  || [[ "$(stat -c '%a' "${screen_inputs_receipt}")" != "444" ]]; then
  echo "[r9-submit] sealed screen-input receipt mode differs" >&2
  exit 2
fi
PYTHONPATH="${source_snapshot}/methods/motive" \
PYTHONDONTWRITEBYTECODE=1 \
"${python_bin}" -c '
from pathlib import Path
from motive.r7_artifact_permissions import assert_sealed_tree
import sys
for raw in sys.argv[1:]:
    assert_sealed_tree(Path(raw))
' \
  "${screen_input_bundle}" \
  "${visual_features_final}" \
  "$(dirname "${visual_candidates_manifest}")"
for test_file in \
  test_r9_automated_representation_search.py \
  test_r9_representation_orchestration.py \
  test_r9_submission_scripts.py \
  test_instruction_model_registry.py; do
  PYTHONPATH="${source_snapshot}/methods/motive" \
  PYTHONDONTWRITEBYTECODE=1 \
  "${python_bin}" \
    "${source_snapshot}/methods/motive/tests/${test_file}"
done

job_ids=()
seeds=(260108835 260108836)
for seed in "${seeds[@]}"; do
  run_root="${experiment_root}/seed_${seed}"
  mkdir -p "${run_root}" "${run_root}/logs"
  job_receipt="${experiment_root}/provenance/seed_${seed}.job_id"
  if [[ -s "${job_receipt}" ]]; then
    job_id="$(tr -d '[:space:]' < "${job_receipt}")"
  else
    job_id="$(
      sbatch \
        --parsable \
        --job-name="m9-r${seed: -4}" \
        --partition=faculty \
        --account=test-acc \
        --qos=stqos \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --gres=gpu:mi210:1 \
        --time=04:00:00 \
        --output="${experiment_root}/logs/r9_seed_${seed}_%j.out" \
        --error="${experiment_root}/logs/r9_seed_${seed}_%j.err" \
        --export="ALL,MOTIVE_SOURCE_SNAPSHOT=${source_snapshot},MOTIVE_SOURCE_TREE_SHA256=${source_tree_sha256},MOTIVE_R9_RUN_ROOT=${run_root},MOTIVE_R7_PARENT_RUN_ROOT=${parent_run},MOTIVE_MODEL_WORKSPACE=${model_workspace},MOTIVE_R9_SEED=${seed},MOTIVE_R7_CANDIDATE_TEMPORAL_DONE_SHA256=${candidate_done_sha256},MOTIVE_R7_CANDIDATE_TRACK_CACHE_DONE_SHA256=${track_cache_done_sha256},MOTIVE_R7_VISUAL_FEATURES_DONE_SHA256=${visual_features_done_sha256},MOTIVE_R7_VISUAL_CANDIDATES_SHA256=${visual_candidates_sha256},MOTIVE_R7_SCREEN_INPUTS_RECEIPT_SHA256=${screen_inputs_receipt_sha256},PYTHON_BIN=${python_bin}" \
        "${controller}"
    )"
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
      echo "[r9-submit] invalid sbatch response: ${job_id}" >&2
      exit 3
    fi
    temporary_job_receipt="${job_receipt}.tmp.$$"
    printf "%s\n" "${job_id}" > "${temporary_job_receipt}"
    mv "${temporary_job_receipt}" "${job_receipt}"
  fi
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "[r9-submit] invalid sbatch response: ${job_id}" >&2
    exit 3
  fi
  job_ids+=("${job_id}")
done

temporary="${receipt}.tmp.$$"
"${python_bin}" -c '
import json
import os
import sys
from datetime import datetime, timezone
payload = {
    "schema_version": "motive-r9-two-seed-submission-v1",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_root": sys.argv[1],
    "source_tree_sha256": sys.argv[2],
    "jobs": [
        {"seed": 260108835, "job_id": int(sys.argv[3])},
        {"seed": 260108836, "job_id": int(sys.argv[4])},
    ],
    "maximum_concurrent_nodes": 2,
    "cpus_per_job": 16,
    "memory_gib_per_job": 128,
    "gpus_per_job": 1,
    "gpu_compute_expected": False,
    "gpu_allocation_reason": (
        "AUH exposes only the faculty GPU partition and stqos enforces "
        "MinTRES=gres/gpu=1. R9 retrieval itself consumes sealed "
        "NumPy/track/DINO features and remains CPU-heavy; no renderer or "
        "editor training is executed."
    ),
    "reason_for_single_gpu": (
        "R9 retrieval consumes sealed NumPy/track/DINO features and does not "
        "justify an eight-GPU allocation; request only the cluster-enforced "
        "minimum and reserve multi-GPU nodes for a representation-gated "
        "renderer probe"
    ),
    "renderer_training_submitted": False,
}
path = sys.argv[5]
with open(path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
' \
  "${experiment_root}" \
  "${source_tree_sha256}" \
  "${job_ids[0]}" \
  "${job_ids[1]}" \
  "${temporary}"
mv "${temporary}" "${receipt}"
echo "[r9-submit] jobs=${job_ids[*]} receipt=${receipt}"
