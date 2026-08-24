#!/usr/bin/env bash
# Run exact47 all-three same-actor raw diagnostics in one owned 8-GPU allocation.

set -euo pipefail
umask 077

if [[ "$#" -ne 20 ]]; then
  echo "usage: $0 <preflight|full> <allocation-job-id> <python-bin> <diagnostic.py> <diagnostic-sha256> <attempts-root> <root-spec-sha256> <source-manifest> <source-manifest-sha256> <visual-checkpoint> <checkpoint-manifest> <evaluator-spec> <evaluator-spec-sha256> <visual-scorer.py> <visual-scorer-sha256> <visual-contract.py> <visual-contract-sha256> <legacy-cyclic-aggregate> <legacy-cyclic-aggregate-sha256> <fresh-output-root>" >&2
  exit 64
fi

mode="$1"
allocation_job_id="$2"
python_bin="$3"
diagnostic_source="$4"
diagnostic_sha256="$5"
attempts_root="$6"
root_spec_sha256="$7"
source_manifest="$8"
source_manifest_sha256="$9"
visual_checkpoint="${10}"
checkpoint_manifest="${11}"
evaluator_spec="${12}"
evaluator_spec_sha256="${13}"
visual_scorer_source="${14}"
visual_scorer_sha256="${15}"
visual_contract_source="${16}"
visual_contract_sha256="${17}"
legacy_cyclic_aggregate="${18}"
legacy_cyclic_aggregate_sha256="${19}"
output_root="${20}"

if [[ "$mode" != "preflight" && "$mode" != "full" ]]; then
  echo "mode must be preflight or full" >&2
  exit 64
fi
if [[ ! "$allocation_job_id" =~ ^[1-9][0-9]*$ || "${SLURM_JOB_ID:-}" != "$allocation_job_id" ]]; then
  echo "launcher must run inside the named live allocation" >&2
  exit 65
fi
allocated_node_count="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
if [[ "$allocated_node_count" != "1" ]]; then
  echo "launcher requires exactly one allocated node" >&2
  exit 65
fi
if [[ ! -x "$python_bin" || ! -f "$diagnostic_source" || -L "$diagnostic_source" ]]; then
  echo "python or diagnostic source is not a plain executable input" >&2
  exit 66
fi
if [[ ! -d "$attempts_root" || ! -f "$source_manifest" || -L "$source_manifest" || ! -d "$visual_checkpoint" || ! -f "$checkpoint_manifest" || ! -f "$evaluator_spec" || ! -f "$visual_scorer_source" || ! -f "$visual_contract_source" || ! -f "$legacy_cyclic_aggregate" || -L "$legacy_cyclic_aggregate" ]]; then
  echo "one sealed input path is absent" >&2
  exit 66
fi
if [[ "$output_root" != /* || "$output_root" == "/" || -e "$output_root" || -L "$output_root" ]]; then
  echo "output root must be fresh, absolute, and non-root" >&2
  exit 67
fi

"$python_bin" "$diagnostic_source" build-manifest \
  --attempts-root "$attempts_root" \
  --expected-root-spec-sha256 "$root_spec_sha256" \
  --source-manifest "$source_manifest" \
  --expected-source-manifest-sha256 "$source_manifest_sha256" \
  --expected-source-sha256 "$diagnostic_sha256" \
  --legacy-cyclic-aggregate "$legacy_cyclic_aggregate" \
  --expected-legacy-cyclic-aggregate-sha256 "$legacy_cyclic_aggregate_sha256" \
  --output-root "$output_root"

input_manifest="$output_root/input-manifest.json"
input_manifest_sha256="$(sha256sum "$input_manifest" | awk '{print $1}')"
common=(
  --input-manifest "$input_manifest"
  --expected-input-manifest-sha256 "$input_manifest_sha256"
  --expected-source-sha256 "$diagnostic_sha256"
  --visual-checkpoint "$visual_checkpoint"
  --visual-checkpoint-manifest "$checkpoint_manifest"
  --evaluator-spec "$evaluator_spec"
  --expected-evaluator-spec-sha256 "$evaluator_spec_sha256"
  --visual-scorer-source "$visual_scorer_source"
  --expected-visual-scorer-sha256 "$visual_scorer_sha256"
  --visual-contract-source "$visual_contract_source"
  --expected-visual-contract-sha256 "$visual_contract_sha256"
  --output-root "$output_root"
)

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  log_prefix="$output_root/${mode}-rank-$(printf '%02d' "$rank")"
  if [[ "$mode" == "preflight" ]]; then
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 \
      ROCR_VISIBLE_DEVICES="$rank" \
      "$python_bin" "$diagnostic_source" preflight "${common[@]}" --rank "$rank" \
      >"$log_prefix.out" 2>"$log_prefix.err" &
  else
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 \
      ROCR_VISIBLE_DEVICES="$rank" \
      "$python_bin" "$diagnostic_source" worker "${common[@]}" --rank "$rank" --world-size 8 \
      >"$log_prefix.out" 2>"$log_prefix.err" &
  fi
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more GPU ranks failed; aggregate is forbidden" >&2
  exit 70
fi

if [[ "$mode" == "full" ]]; then
  "$python_bin" "$diagnostic_source" aggregate \
    --input-manifest "$input_manifest" \
    --expected-input-manifest-sha256 "$input_manifest_sha256" \
    --expected-source-sha256 "$diagnostic_sha256" \
    --legacy-cyclic-aggregate "$legacy_cyclic_aggregate" \
    --expected-legacy-cyclic-aggregate-sha256 "$legacy_cyclic_aggregate_sha256" \
    --output-root "$output_root"
fi

for artifact in "$output_root"/*.json "$output_root"/*.out "$output_root"/*.err; do
  [[ -e "$artifact" ]] || continue
  chmod 0400 "$artifact"
done
chmod 0500 "$output_root"
echo "partial47 all-three same-actor DINO raw ${mode} complete: $output_root"
