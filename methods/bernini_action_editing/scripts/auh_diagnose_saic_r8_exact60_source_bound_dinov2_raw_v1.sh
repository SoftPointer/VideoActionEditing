#!/usr/bin/env bash
# Run the frozen r8 exact60 cyclic source-bound raw diagnostic in one owned allocation.

set -euo pipefail
umask 077

if [[ "$#" -ne 19 ]]; then
  echo "usage: $0 <preflight|full> <allocation-job-id> <python-bin> <diagnostic.py> <diagnostic-sha256> <attempts-root> <root-spec-sha256> <source-manifest> <source-manifest-sha256> <terminal-evidence> <visual-checkpoint> <checkpoint-manifest> <evaluator-spec> <evaluator-spec-sha256> <visual-scorer.py> <visual-scorer-sha256> <visual-contract.py> <visual-contract-sha256> <fresh-output-root>" >&2
  exit 64
fi

mode="$1"; allocation_job_id="$2"; python_bin="$3"
diagnostic_source="$4"; diagnostic_sha256="$5"; attempts_root="$6"
root_spec_sha256="$7"; source_manifest="$8"; source_manifest_sha256="$9"
terminal_evidence="${10}"; visual_checkpoint="${11}"; checkpoint_manifest="${12}"
evaluator_spec="${13}"; evaluator_spec_sha256="${14}"
visual_scorer_source="${15}"; visual_scorer_sha256="${16}"
visual_contract_source="${17}"; visual_contract_sha256="${18}"
output_root="${19}"

readonly expected_python=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/runtime/venv-transformers-4.53.2/bin/python
readonly expected_diagnostic_sha=2839641766b4605311f0c4e7a3ff41ea9a322ad44cf1d12d21fb7f0cca5ab24e
readonly expected_checkpoint_manifest_sha=b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea
readonly expected_attempts=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r8-ddc8a79-r1/attempts
readonly expected_root_sha=d693d0784530f007888e2825d15db3db808fdf4f1d111b5d080d968c894ff145
readonly expected_source_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r8-ddc8a79-r1/sealed-saic-source-manifest.json
readonly expected_source_manifest_sha=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly expected_terminal_evidence=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/releases/saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/saic-exact60-terminal-evidence-135056.json
readonly expected_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
readonly expected_checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/dinov2-base-f9e44c8.sha256
readonly expected_evaluator_spec=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json
readonly expected_evaluator_spec_sha=6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736
readonly expected_visual_scorer=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py
readonly expected_visual_scorer_sha=9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39
readonly expected_visual_contract=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py
readonly expected_visual_contract_sha=183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a
readonly portable_ffprobe_dir=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime
readonly portable_ffprobe="$portable_ffprobe_dir/ffprobe"
readonly portable_ffprobe_sha=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5

if [[ "$mode" != "preflight" && "$mode" != "full" ]]; then echo "mode must be preflight or full" >&2; exit 64; fi
if [[ ! "$allocation_job_id" =~ ^[1-9][0-9]*$ || "${SLURM_JOB_ID:-}" != "$allocation_job_id" ]]; then echo "launcher must run inside the named live allocation" >&2; exit 65; fi
allocated_node_count="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
if [[ "$allocated_node_count" != "1" ]]; then echo "launcher requires exactly one allocated node" >&2; exit 65; fi
if [[ "$python_bin" != "$expected_python" || ! -x "$python_bin" ]]; then echo "python must be the lexical frozen Transformers-4.53.2 PAIR venv path" >&2; exit 66; fi
if [[ "$diagnostic_sha256" != "$expected_diagnostic_sha" ]]; then echo "r8 diagnostic SHA argument differs from the hard pin" >&2; exit 66; fi
if [[ "$attempts_root" != "$expected_attempts" || "$root_spec_sha256" != "$expected_root_sha" || "$source_manifest" != "$expected_source_manifest" || "$source_manifest_sha256" != "$expected_source_manifest_sha" || "$terminal_evidence" != "$expected_terminal_evidence" ]]; then echo "r8 exact60 root/spec/source/terminal arguments differ" >&2; exit 66; fi
if [[ "$visual_checkpoint" != "$expected_checkpoint" || "$checkpoint_manifest" != "$expected_checkpoint_manifest" || "$evaluator_spec" != "$expected_evaluator_spec" || "$evaluator_spec_sha256" != "$expected_evaluator_spec_sha" || "$visual_scorer_source" != "$expected_visual_scorer" || "$visual_scorer_sha256" != "$expected_visual_scorer_sha" || "$visual_contract_source" != "$expected_visual_contract" || "$visual_contract_sha256" != "$expected_visual_contract_sha" ]]; then echo "frozen evaluator arguments differ" >&2; exit 66; fi
if [[ ! -f "$diagnostic_source" || -L "$diagnostic_source" || ! -d "$attempts_root" || ! -f "$source_manifest" || -L "$source_manifest" || ! -f "$terminal_evidence" || -L "$terminal_evidence" || ! -d "$visual_checkpoint" || ! -f "$checkpoint_manifest" || -L "$checkpoint_manifest" || ! -f "$evaluator_spec" || -L "$evaluator_spec" || ! -f "$visual_scorer_source" || -L "$visual_scorer_source" || ! -f "$visual_contract_source" || -L "$visual_contract_source" ]]; then echo "one sealed input path is absent or linked" >&2; exit 66; fi
if [[ "$(sha256sum "$diagnostic_source" | awk '{print $1}')" != "$diagnostic_sha256" || "$(sha256sum "$source_manifest" | awk '{print $1}')" != "$expected_source_manifest_sha" || "$(sha256sum "$checkpoint_manifest" | awk '{print $1}')" != "$expected_checkpoint_manifest_sha" || "$(sha256sum "$evaluator_spec" | awk '{print $1}')" != "$expected_evaluator_spec_sha" || "$(sha256sum "$visual_scorer_source" | awk '{print $1}')" != "$expected_visual_scorer_sha" || "$(sha256sum "$visual_contract_source" | awk '{print $1}')" != "$expected_visual_contract_sha" ]]; then echo "one frozen source/spec SHA-256 differs" >&2; exit 66; fi
if [[ ! -x "$portable_ffprobe" || -L "$portable_ffprobe" || "$(sha256sum "$portable_ffprobe" | awk '{print $1}')" != "$portable_ffprobe_sha" ]]; then echo "portable ffprobe identity differs" >&2; exit 66; fi
export PATH="$portable_ffprobe_dir:${PATH:-/usr/bin:/bin}"
if [[ "$(command -v ffprobe)" != "$portable_ffprobe" ]]; then echo "portable ffprobe is not PATH-first" >&2; exit 66; fi
if [[ "$output_root" != /* || "$output_root" == "/" || -e "$output_root" || -L "$output_root" ]]; then echo "output root must be fresh, absolute, and non-root" >&2; exit 67; fi

scratch_parent="${SLURM_TMPDIR:-/tmp}"
if [[ "$scratch_parent" != /* || ! -d "$scratch_parent" || -L "$scratch_parent" || ! -w "$scratch_parent" ]]; then echo "allocation-local scratch parent differs" >&2; exit 67; fi
scratch_parent="$(readlink -f -- "$scratch_parent")"
if [[ -z "$scratch_parent" || "$scratch_parent" == "/" ]]; then echo "allocation-local scratch parent resolution differs" >&2; exit 67; fi
readonly scratch_parent
declare -a pids=()
declare -a cache_paths=()
declare -a cache_identities=()
runtime_scratch=""
runtime_scratch_identity=""
current_uid="$(id -u)"
readonly current_uid

validate_cache_roots() {
  local index path identity owner mode_value resolved
  [[ -n "$runtime_scratch" && -d "$runtime_scratch" && ! -L "$runtime_scratch" ]] || return 1
  resolved="$(readlink -f -- "$runtime_scratch")" || return 1
  identity="$(stat -c '%d:%i' -- "$runtime_scratch")" || return 1
  owner="$(stat -c '%u' -- "$runtime_scratch")" || return 1
  mode_value="$(stat -c '%a' -- "$runtime_scratch")" || return 1
  [[ "$resolved" == "$runtime_scratch" && "$identity" == "$runtime_scratch_identity" && "$owner" == "$current_uid" && "$mode_value" == 700 ]] || return 1
  for index in "${!cache_paths[@]}"; do
    path="${cache_paths[index]}"
    [[ "$path" == "$runtime_scratch/"* && -d "$path" && ! -L "$path" ]] || return 1
    resolved="$(readlink -f -- "$path")" || return 1
    identity="$(stat -c '%d:%i' -- "$path")" || return 1
    owner="$(stat -c '%u' -- "$path")" || return 1
    mode_value="$(stat -c '%a' -- "$path")" || return 1
    [[ "$resolved" == "$path" && "$identity" == "${cache_identities[index]}" && "$owner" == "$current_uid" && "$mode_value" == 700 ]] || return 1
  done
}

remove_runtime_scratch_exact() {
  [[ -z "$runtime_scratch" ]] && return 0
  validate_cache_roots || return 1
  find "$runtime_scratch" -xdev -depth -mindepth 1 -delete || return 1
  rmdir -- "$runtime_scratch" || return 1
  runtime_scratch=""
  runtime_scratch_identity=""
  cache_paths=()
  cache_identities=()
}

cleanup_on_exit() {
  local status="$?" pid cleanup_failed=0
  trap - EXIT INT TERM HUP
  for pid in "${pids[@]}"; do
    if kill "$pid" 2>/dev/null; then wait "$pid" 2>/dev/null || true; fi
  done
  pids=()
  if [[ -n "$runtime_scratch" ]] && ! remove_runtime_scratch_exact; then
    echo "unsafe rank-local cache cleanup refused; retained $runtime_scratch" >&2
    cleanup_failed=1
  fi
  if [[ "$cleanup_failed" -ne 0 ]]; then exit 73; fi
  exit "$status"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

runtime_prefix="${scratch_parent%/}/saic-r8-exact60-cyclic-${allocation_job_id}-${mode}."
runtime_scratch="$(mktemp -d -- "${runtime_prefix}XXXXXXXX")"
if [[ "$runtime_scratch" != "$runtime_prefix"???????? || -L "$runtime_scratch" ]]; then echo "fresh runtime scratch framing differs" >&2; exit 67; fi
chmod 0700 -- "$runtime_scratch"
runtime_scratch_identity="$(stat -c '%d:%i' -- "$runtime_scratch")"
if [[ "$(readlink -f -- "$runtime_scratch")" != "$runtime_scratch" || "$(stat -c '%u:%a' -- "$runtime_scratch")" != "$current_uid:700" ]]; then echo "fresh runtime scratch identity/mode differs" >&2; exit 67; fi

register_cache_directory() {
  local path="$1"
  mkdir -m 0700 -- "$path"
  if [[ -L "$path" || "$(readlink -f -- "$path")" != "$path" || "$(stat -c '%u:%a' -- "$path")" != "$current_uid:700" ]]; then echo "rank-local cache identity/mode differs" >&2; exit 67; fi
  cache_paths+=("$path")
  cache_identities+=("$(stat -c '%d:%i' -- "$path")")
}

prepare_cache_tree() {
  local cache="$1" child
  register_cache_directory "$cache"
  for child in home xdg-config xdg-cache miopen-user miopen-custom tmp torch-extensions triton torchinductor pycache; do
    register_cache_directory "$cache/$child"
  done
}

control_cache="$runtime_scratch/control"
prepare_cache_tree "$control_cache"
declare -a rank_caches=()
for rank in 0 1 2 3 4 5 6 7; do
  printf -v rank_label '%02d' "$rank"
  rank_cache="$runtime_scratch/rank-$rank_label"
  prepare_cache_tree "$rank_cache"
  rank_caches+=("$rank_cache")
done
validate_cache_roots || { echo "initial rank-local cache closure differs" >&2; exit 67; }

run_isolated() {
  local cache="$1" rocr_device="$2"
  shift 2
  validate_cache_roots || { echo "rank-local cache closure changed" >&2; exit 67; }
  env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    PATH="$PATH" HOME="$cache/home" TMPDIR="$cache/tmp" \
    XDG_CONFIG_HOME="$cache/xdg-config" XDG_CACHE_HOME="$cache/xdg-cache" \
    MIOPEN_USER_DB_PATH="$cache/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="$cache/miopen-custom" \
    TORCH_EXTENSIONS_DIR="$cache/torch-extensions" \
    TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    PYTHONPYCACHEPREFIX="$cache/pycache" PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    HF_HOME="$cache/xdg-cache/huggingface" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    ROCR_VISIBLE_DEVICES="$rocr_device" "$@"
}

run_isolated "$control_cache" "" "$python_bin" "$diagnostic_source" build-manifest \
  --attempts-root "$attempts_root" --expected-root-spec-sha256 "$root_spec_sha256" \
  --source-manifest "$source_manifest" --expected-source-manifest-sha256 "$source_manifest_sha256" \
  --terminal-evidence "$terminal_evidence" \
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

for rank in 0 1 2 3 4 5 6 7; do
  log_prefix="$output_root/${mode}-rank-$(printf '%02d' "$rank")"
  rank_cache="${rank_caches[rank]}"
  if [[ "$mode" == "preflight" ]]; then
    run_isolated "$rank_cache" "$rank" \
      "$python_bin" "$diagnostic_source" preflight "${common[@]}" --rank "$rank" >"$log_prefix.out" 2>"$log_prefix.err" &
  else
    run_isolated "$rank_cache" "$rank" \
      "$python_bin" "$diagnostic_source" worker "${common[@]}" --rank "$rank" --world-size 8 >"$log_prefix.out" 2>"$log_prefix.err" &
  fi
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then echo "one or more GPU ranks failed; aggregate is forbidden" >&2; exit 70; fi
if [[ "$mode" == "full" ]]; then
  run_isolated "$control_cache" "" "$python_bin" "$diagnostic_source" aggregate --input-manifest "$input_manifest" \
    --expected-input-manifest-sha256 "$input_manifest_sha256" --expected-source-sha256 "$diagnostic_sha256" --output-root "$output_root"
fi
for artifact in "$output_root"/*.json "$output_root"/*.out "$output_root"/*.err; do [[ -e "$artifact" ]] || continue; chmod 0400 "$artifact"; done
chmod 0500 "$output_root"
echo "r8 exact60 cyclic source-bound DINO raw ${mode} complete: $output_root authority=zero"
