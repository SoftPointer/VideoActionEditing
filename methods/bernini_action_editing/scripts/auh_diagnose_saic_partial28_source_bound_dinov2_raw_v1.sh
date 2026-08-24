#!/usr/bin/env bash
# Run the frozen r6 exact28 source-bound raw diagnostic in an owned 8-GPU allocation.

set -euo pipefail
umask 077

if [[ "$#" -ne 18 ]]; then
  echo "usage: $0 <preflight|full> <allocation-job-id> <python-bin> <diagnostic.py> <diagnostic-sha256> <attempts-root> <root-spec-sha256> <source-manifest> <source-manifest-sha256> <visual-checkpoint> <checkpoint-manifest> <evaluator-spec> <evaluator-spec-sha256> <visual-scorer.py> <visual-scorer-sha256> <visual-contract.py> <visual-contract-sha256> <fresh-output-root>" >&2
  exit 64
fi

mode="$1"; allocation_job_id="$2"; python_bin="$3"
diagnostic_source="$4"; diagnostic_sha256="$5"; attempts_root="$6"
root_spec_sha256="$7"; source_manifest="$8"; source_manifest_sha256="$9"
visual_checkpoint="${10}"; checkpoint_manifest="${11}"
evaluator_spec="${12}"; evaluator_spec_sha256="${13}"
visual_scorer_source="${14}"; visual_scorer_sha256="${15}"
visual_contract_source="${16}"; visual_contract_sha256="${17}"
output_root="${18}"

readonly expected_python=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/runtime/venv-transformers-4.53.2/bin/python
readonly expected_attempts=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-umaskfix-72f3a40-r1/attempts
readonly expected_root_sha=d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145
readonly expected_source_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-umaskfix-72f3a40-r1/sealed-saic-source-manifest.json
readonly expected_source_manifest_sha=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly portable_ffprobe_dir=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime
readonly portable_ffprobe="$portable_ffprobe_dir/ffprobe"
readonly portable_ffprobe_sha=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5

if [[ "$mode" != "preflight" && "$mode" != "full" ]]; then echo "mode must be preflight or full" >&2; exit 64; fi
if [[ ! "$allocation_job_id" =~ ^[1-9][0-9]*$ || "${SLURM_JOB_ID:-}" != "$allocation_job_id" ]]; then echo "launcher must run inside the named live allocation" >&2; exit 65; fi
allocated_node_count="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
if [[ "$allocated_node_count" != "1" ]]; then echo "launcher requires exactly one allocated node" >&2; exit 65; fi
if [[ "$python_bin" != "$expected_python" || ! -x "$python_bin" ]]; then echo "python must be the lexical frozen Transformers-4.53.2 PAIR venv path" >&2; exit 66; fi
if [[ "$attempts_root" != "$expected_attempts" || "$root_spec_sha256" != "$expected_root_sha" || "$source_manifest" != "$expected_source_manifest" || "$source_manifest_sha256" != "$expected_source_manifest_sha" ]]; then echo "r6 exact28 root/spec/source arguments differ" >&2; exit 66; fi
if [[ ! -f "$diagnostic_source" || -L "$diagnostic_source" || ! -d "$attempts_root" || ! -f "$source_manifest" || -L "$source_manifest" || ! -d "$visual_checkpoint" || ! -f "$checkpoint_manifest" || ! -f "$evaluator_spec" || ! -f "$visual_scorer_source" || ! -f "$visual_contract_source" ]]; then echo "one sealed input path is absent" >&2; exit 66; fi
if [[ ! -x "$portable_ffprobe" || -L "$portable_ffprobe" || "$(sha256sum "$portable_ffprobe" | awk '{print $1}')" != "$portable_ffprobe_sha" ]]; then echo "portable ffprobe identity differs" >&2; exit 66; fi
export PATH="$portable_ffprobe_dir:${PATH:-/usr/bin:/bin}"
if [[ "$(command -v ffprobe)" != "$portable_ffprobe" ]]; then echo "portable ffprobe is not PATH-first" >&2; exit 66; fi
if [[ "$output_root" != /* || "$output_root" == "/" || -e "$output_root" || -L "$output_root" ]]; then echo "output root must be fresh, absolute, and non-root" >&2; exit 67; fi

"$python_bin" "$diagnostic_source" build-manifest \
  --attempts-root "$attempts_root" --expected-root-spec-sha256 "$root_spec_sha256" \
  --source-manifest "$source_manifest" --expected-source-manifest-sha256 "$source_manifest_sha256" \
  --expected-source-sha256 "$diagnostic_sha256" --output-root "$output_root"

input_manifest="$output_root/input-manifest.json"
input_manifest_sha256="$(sha256sum "$input_manifest" | awk '{print $1}')"
common=(
  --input-manifest "$input_manifest" --expected-input-manifest-sha256 "$input_manifest_sha256"
  --expected-source-sha256 "$diagnostic_sha256" --visual-checkpoint "$visual_checkpoint"
  --visual-checkpoint-manifest "$checkpoint_manifest" --evaluator-spec "$evaluator_spec"
  --expected-evaluator-spec-sha256 "$evaluator_spec_sha256" --visual-scorer-source "$visual_scorer_source"
  --expected-visual-scorer-sha256 "$visual_scorer_sha256" --visual-contract-source "$visual_contract_source"
  --expected-visual-contract-sha256 "$visual_contract_sha256" --output-root "$output_root"
)

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  log_prefix="$output_root/${mode}-rank-$(printf '%02d' "$rank")"
  if [[ "$mode" == "preflight" ]]; then
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 ROCR_VISIBLE_DEVICES="$rank" \
      "$python_bin" "$diagnostic_source" preflight "${common[@]}" --rank "$rank" >"$log_prefix.out" 2>"$log_prefix.err" &
  else
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 ROCR_VISIBLE_DEVICES="$rank" \
      "$python_bin" "$diagnostic_source" worker "${common[@]}" --rank "$rank" --world-size 8 >"$log_prefix.out" 2>"$log_prefix.err" &
  fi
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then echo "one or more GPU ranks failed; aggregate is forbidden" >&2; exit 70; fi
if [[ "$mode" == "full" ]]; then
  "$python_bin" "$diagnostic_source" aggregate --input-manifest "$input_manifest" \
    --expected-input-manifest-sha256 "$input_manifest_sha256" --expected-source-sha256 "$diagnostic_sha256" --output-root "$output_root"
fi
for artifact in "$output_root"/*.json "$output_root"/*.out "$output_root"/*.err; do [[ -e "$artifact" ]] || continue; chmod 0400 "$artifact"; done
chmod 0500 "$output_root"
echo "r6 partial28 source-bound DINO raw ${mode} complete: $output_root"
