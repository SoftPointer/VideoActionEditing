#!/bin/bash -p
# Run the exact8 registered-source DINO pair matrix inside one named owned allocation.

case "$-" in
  *p*) ;;
  *) printf '%s\n' 'privileged bash mode is required' >&2; exit 64 ;;
esac
set -Eeuo pipefail
set -C
umask 077

PATH=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime:/usr/bin:/bin
export PATH
readonly PATH
readonly sha256_bin=/usr/bin/sha256sum
readonly readlink_bin=/usr/bin/readlink
readonly stat_bin=/usr/bin/stat
readonly id_bin=/usr/bin/id
readonly env_bin=/usr/bin/env
readonly mkdir_bin=/usr/bin/mkdir
readonly mktemp_bin=/usr/bin/mktemp
readonly find_bin=/usr/bin/find
readonly rmdir_bin=/usr/bin/rmdir
readonly chmod_bin=/usr/bin/chmod

fail() {
  printf 'source8-matrix launcher: %s\n' "$1" >&2
  exit "${2:-68}"
}

[[ -x "$sha256_bin" && -x "$readlink_bin" && -x "$stat_bin" \
  && -x "$id_bin" && -x "$env_bin" && -x "$mkdir_bin" \
  && -x "$mktemp_bin" && -x "$find_bin" && -x "$rmdir_bin" \
  && -x "$chmod_bin" ]] || fail 'one absolute runtime tool is unavailable' 66

if [[ "$#" -ne 20 ]]; then
  printf '%s\n' \
    "usage: $0 <preflight|full> <allocation-job-id> <python-bin> <diagnostic.py> <diagnostic-sha256> <source-manifest> <source-manifest-raw-sha256> <source-manifest-content-sha256> <source-validator-summary-sha256> <visual-checkpoint> <checkpoint-manifest> <evaluator-spec> <evaluator-spec-sha256> <visual-scorer.py> <visual-scorer-sha256> <visual-contract.py> <visual-contract-sha256> <r4-all3-aggregate> <r4-all3-aggregate-sha256> <fresh-output-root>" >&2
  exit 64
fi

mode="$1"
allocation_job_id="$2"
python_bin="$3"
diagnostic_source="$4"
diagnostic_sha256="$5"
source_manifest="$6"
source_manifest_sha256="$7"
source_manifest_content_sha256="$8"
source_validator_summary_sha256="$9"
visual_checkpoint="${10}"
checkpoint_manifest="${11}"
evaluator_spec="${12}"
evaluator_spec_sha256="${13}"
visual_scorer_source="${14}"
visual_scorer_sha256="${15}"
visual_contract_source="${16}"
visual_contract_sha256="${17}"
legacy_all3_aggregate="${18}"
legacy_all3_aggregate_sha256="${19}"
output_root="${20}"

readonly staging_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/staging/saic-formal-v2-retained-fd-world8-canary-5a8dd488-865f3713-r4-upload-20260812
readonly expected_diagnostic_source="$staging_root/diagnose_saic_registered_source8_dinov2_pair_matrix_raw_v1.py"
readonly expected_diagnostic_sha256=82c472e3a54b988ffa737e84707decf844a08df4dbbc6d19f75e4c57a43dc5a7
readonly expected_source_validator="$staging_root/build_saic_reversible_source_set_v1.py"
readonly expected_source_validator_sha256=0cf012adf25dd1afffb33d1e0c918630a574c9075e9aa293914e04890c71cf5b
readonly expected_python=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/runtime/venv-transformers-4.53.2/bin/python
readonly expected_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
readonly expected_checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/dinov2-base-f9e44c8.sha256
readonly expected_checkpoint_manifest_sha256=b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea
readonly expected_evaluator_spec=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json
readonly expected_evaluator_spec_sha256=6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736
readonly expected_visual_scorer=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py
readonly expected_visual_scorer_sha256=9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39
readonly expected_visual_contract=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py
readonly expected_visual_contract_sha256=183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a
readonly expected_source_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/t2v-events-topup-r6-umaskfix-72f3a40-r1/sealed-saic-source-manifest.json
readonly expected_source_manifest_sha256=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly expected_source_manifest_content_sha256=9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f
readonly expected_source_validator_summary_sha256=257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218
readonly expected_legacy_all3=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/allocation-134936-sourcebound47-all3-dinov2-full-847f3cf3-r1/aggregate-receipt.json
readonly expected_legacy_all3_sha256=3f5169d45b603ac6c10da12d6736b7878c30a6654b8aeedd335a8548865b7beb
readonly diagnostics_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics
readonly portable_ffprobe_dir=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime
readonly portable_ffprobe="$portable_ffprobe_dir/ffprobe"
readonly portable_ffprobe_sha256=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5
readonly runtime_ld_library_path="$portable_ffprobe_dir:$portable_ffprobe_dir/lib:/opt/rocm/lib:/opt/rocm/lib64:/usr/lib64:/usr/lib/x86_64-linux-gnu"

hash_file() {
  local path="$1" output digest suffix
  output="$("$sha256_bin" -- "$path")" || fail "cannot hash $path" 66
  digest="${output:0:64}"
  suffix="${output:64}"
  [[ "$digest" =~ ^[0-9a-f]{64}$ \
    && ( "$suffix" == "  $path" || "$suffix" == " *$path" ) ]] \
    || fail "hash output framing differs for $path" 66
  printf '%s' "$digest"
}

require_plain_file() {
  local path="$1" label="$2" resolved nlink
  [[ "$path" == /* && -f "$path" && ! -L "$path" ]] \
    || fail "$label is not an absolute plain file" 66
  resolved="$("$readlink_bin" -f -- "$path")" \
    || fail "$label cannot be resolved" 66
  [[ "$resolved" == "$path" ]] || fail "$label resolution differs" 66
  nlink="$("$stat_bin" -c '%h' -- "$path")" \
    || fail "$label link count is unavailable" 66
  [[ "$nlink" == 1 ]] || fail "$label must have one link" 66
}

require_plain_directory() {
  local path="$1" label="$2" resolved
  [[ "$path" == /* && -d "$path" && ! -L "$path" ]] \
    || fail "$label is not an absolute plain directory" 66
  resolved="$("$readlink_bin" -f -- "$path")" \
    || fail "$label cannot be resolved" 66
  [[ "$resolved" == "$path" ]] || fail "$label resolution differs" 66
}

[[ "$mode" == preflight || "$mode" == full ]] \
  || fail 'mode must be preflight or full' 64
[[ "$allocation_job_id" =~ ^[1-9][0-9]*$ \
  && "${SLURM_JOB_ID:-}" == "$allocation_job_id" ]] \
  || fail 'launcher is outside the named owned allocation' 65
allocated_node_count="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
[[ "$allocated_node_count" == 1 ]] \
  || fail 'launcher requires exactly one allocated node' 65

[[ "$python_bin" == "$expected_python" && -x "$python_bin" ]] \
  || fail 'frozen Python lexical identity differs' 66
[[ "$diagnostic_source" == "$expected_diagnostic_source" \
  && "$diagnostic_sha256" == "$expected_diagnostic_sha256" ]] \
  || fail 'diagnostic caller substitution is forbidden' 66
[[ "$source_manifest" == "$expected_source_manifest" \
  && "$source_manifest_sha256" == "$expected_source_manifest_sha256" \
  && "$source_manifest_content_sha256" == "$expected_source_manifest_content_sha256" \
  && "$source_validator_summary_sha256" == "$expected_source_validator_summary_sha256" ]] \
  || fail 'sealed source-manifest argument closure differs' 66
[[ "$visual_checkpoint" == "$expected_checkpoint" \
  && "$checkpoint_manifest" == "$expected_checkpoint_manifest" \
  && "$evaluator_spec" == "$expected_evaluator_spec" \
  && "$evaluator_spec_sha256" == "$expected_evaluator_spec_sha256" \
  && "$visual_scorer_source" == "$expected_visual_scorer" \
  && "$visual_scorer_sha256" == "$expected_visual_scorer_sha256" \
  && "$visual_contract_source" == "$expected_visual_contract" \
  && "$visual_contract_sha256" == "$expected_visual_contract_sha256" ]] \
  || fail 'frozen evaluator lexical path/SHA closure differs' 66
[[ "$legacy_all3_aggregate" == "$expected_legacy_all3" \
  && "$legacy_all3_aggregate_sha256" == "$expected_legacy_all3_sha256" ]] \
  || fail 'frozen r4 all-three caller arguments differ' 66
output_leaf="${output_root#"$diagnostics_root"/}"
readonly output_leaf
[[ "$output_root" == "$diagnostics_root/"* \
  && "$output_leaf" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$ \
  && ! -e "$output_root" && ! -L "$output_root" ]] \
  || fail 'output root must be fresh and lexical under the diagnostics root' 67

require_plain_directory "$staging_root" 'r4 staging root'
require_plain_directory "$expected_checkpoint" 'frozen visual checkpoint'
require_plain_directory "$diagnostics_root" 'diagnostics root'
require_plain_file "$diagnostic_source" 'diagnostic source'
require_plain_file "$expected_source_validator" 'source-manifest validator source'
require_plain_file "$source_manifest" 'sealed source manifest'
require_plain_file "$checkpoint_manifest" 'checkpoint manifest'
require_plain_file "$evaluator_spec" 'evaluator spec'
require_plain_file "$visual_scorer_source" 'visual scorer source'
require_plain_file "$visual_contract_source" 'visual contract source'
require_plain_file "$portable_ffprobe" 'portable ffprobe'
[[ "$(hash_file "$diagnostic_source")" == "$expected_diagnostic_sha256" \
  && "$(hash_file "$expected_source_validator")" == "$expected_source_validator_sha256" \
  && "$(hash_file "$source_manifest")" == "$expected_source_manifest_sha256" \
  && "$(hash_file "$checkpoint_manifest")" == "$expected_checkpoint_manifest_sha256" \
  && "$(hash_file "$evaluator_spec")" == "$expected_evaluator_spec_sha256" \
  && "$(hash_file "$visual_scorer_source")" == "$expected_visual_scorer_sha256" \
  && "$(hash_file "$visual_contract_source")" == "$expected_visual_contract_sha256" \
  && "$(hash_file "$portable_ffprobe")" == "$portable_ffprobe_sha256" ]] \
  || fail 'one fixed source/spec/tool byte identity differs' 66
[[ "$(command -v ffprobe)" == "$portable_ffprobe" ]] \
  || fail 'portable ffprobe is not PATH-first' 66

declare -a pids=()
declare -a cache_paths=()
declare -a cache_identities=()
runtime_scratch=""
runtime_scratch_identity=""
current_uid="$("$id_bin" -u)" || fail 'cannot resolve launcher UID' 66
readonly current_uid

validate_cache_roots() {
  local index path identity owner mode_value resolved
  [[ -n "$runtime_scratch" && -d "$runtime_scratch" && ! -L "$runtime_scratch" ]] \
    || return 1
  resolved="$("$readlink_bin" -f -- "$runtime_scratch")" || return 1
  identity="$("$stat_bin" -c '%d:%i' -- "$runtime_scratch")" || return 1
  owner="$("$stat_bin" -c '%u' -- "$runtime_scratch")" || return 1
  mode_value="$("$stat_bin" -c '%a' -- "$runtime_scratch")" || return 1
  [[ "$resolved" == "$runtime_scratch" \
    && "$identity" == "$runtime_scratch_identity" \
    && "$owner" == "$current_uid" && "$mode_value" == 700 ]] || return 1
  for index in "${!cache_paths[@]}"; do
    path="${cache_paths[index]}"
    [[ "$path" == "$runtime_scratch/"* && -d "$path" && ! -L "$path" ]] || return 1
    resolved="$("$readlink_bin" -f -- "$path")" || return 1
    identity="$("$stat_bin" -c '%d:%i' -- "$path")" || return 1
    owner="$("$stat_bin" -c '%u' -- "$path")" || return 1
    mode_value="$("$stat_bin" -c '%a' -- "$path")" || return 1
    [[ "$resolved" == "$path" && "$identity" == "${cache_identities[index]}" \
      && "$owner" == "$current_uid" && "$mode_value" == 700 ]] || return 1
  done
}

remove_runtime_scratch_exact() {
  [[ -z "$runtime_scratch" ]] && return 0
  validate_cache_roots || return 1
  "$find_bin" "$runtime_scratch" -xdev -depth -mindepth 1 -delete || return 1
  "$rmdir_bin" -- "$runtime_scratch" || return 1
  runtime_scratch=""
  runtime_scratch_identity=""
  cache_paths=()
  cache_identities=()
}

cleanup_on_exit() {
  local status="$?" pid cleanup_failed=0
  trap - EXIT INT TERM HUP
  for pid in "${pids[@]}"; do
    if kill "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  pids=()
  if [[ -n "$runtime_scratch" ]] && ! remove_runtime_scratch_exact; then
    printf 'source8-matrix launcher: unsafe cache cleanup refused; retained %s\n' \
      "$runtime_scratch" >&2
    cleanup_failed=1
  fi
  if [[ "$cleanup_failed" -ne 0 ]]; then
    exit 73
  fi
  exit "$status"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

runtime_scratch="$("$mktemp_bin" -d -- "/tmp/saic-source8-${allocation_job_id}-${mode}.XXXXXXXX")" \
  || fail 'cannot create job-private runtime scratch' 67
[[ "$runtime_scratch" == "/tmp/saic-source8-${allocation_job_id}-${mode}."???????? ]] \
  || fail 'job-private runtime scratch framing differs' 67
require_plain_directory "$runtime_scratch" 'job-private runtime scratch'
runtime_scratch_identity="$("$stat_bin" -c '%d:%i' -- "$runtime_scratch")" \
  || fail 'cannot bind job-private runtime scratch inode' 67
[[ "$("$stat_bin" -c '%u:%a' -- "$runtime_scratch")" == "$current_uid:700" ]] \
  || fail 'job-private runtime scratch ownership/mode differs' 67

register_cache_directory() {
  local path="$1"
  "$mkdir_bin" -- "$path" || fail "cannot create cache directory $path" 67
  require_plain_directory "$path" 'rank-local runtime cache'
  [[ "$("$stat_bin" -c '%u:%a' -- "$path")" == "$current_uid:700" ]] \
    || fail "cache ownership/mode differs for $path" 67
  cache_paths+=("$path")
  cache_identities+=("$("$stat_bin" -c '%d:%i' -- "$path")")
}

prepare_cache_tree() {
  local cache="$1" child
  register_cache_directory "$cache"
  for child in home miopen-user miopen-custom torch-extensions triton torchinductor xdg pycache tmp; do
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
validate_cache_roots || fail 'initial rank-local runtime cache closure differs' 67

run_clean() {
  local cache="$1"
  shift
  "$env_bin" -i \
    PATH="$PATH" LD_LIBRARY_PATH="$runtime_ld_library_path" \
    HOME="$cache/home" TMPDIR="$cache/tmp" XDG_CACHE_HOME="$cache/xdg" \
    MIOPEN_USER_DB_PATH="$cache/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="$cache/miopen-custom" \
    TORCH_EXTENSIONS_DIR="$cache/torch-extensions" \
    TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    PYTHONPYCACHEPREFIX="$cache/pycache" PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    HF_HOME="$cache/xdg/huggingface" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    LC_ALL=C LANG=C "$@"
}

run_clean "$control_cache" "$python_bin" -I -B "$diagnostic_source" build-manifest \
  --source-manifest "$source_manifest" \
  --expected-source-manifest-sha256 "$source_manifest_sha256" \
  --expected-source-manifest-content-sha256 "$source_manifest_content_sha256" \
  --expected-source-validator-summary-sha256 "$source_validator_summary_sha256" \
  --expected-source-sha256 "$diagnostic_sha256" \
  --output-root "$output_root"

require_plain_directory "$output_root" 'fresh diagnostic output root'
output_root_identity="$("$stat_bin" -c '%d:%i' -- "$output_root")" \
  || fail 'cannot bind output-root inode' 67
[[ "$("$stat_bin" -c '%u:%a' -- "$output_root")" == "$current_uid:700" ]] \
  || fail 'fresh output-root ownership/mode differs' 67
readonly output_root_identity
input_manifest="$output_root/input-manifest.json"
require_plain_file "$input_manifest" 'source8 input manifest'
[[ "$("$stat_bin" -c '%a' -- "$input_manifest")" == 400 ]] \
  || fail 'source8 input manifest mode differs' 67
input_manifest_sha256="$(hash_file "$input_manifest")"
readonly input_manifest input_manifest_sha256

shopt -s nullglob dotglob
declare -a expected_files=("$input_manifest")
declare -a json_files=("$input_manifest")
declare -a log_files=()

validate_output_set() {
  local final_modes="$1" path mode_value
  local -a actual=("$output_root"/*)
  declare -A allowed=()
  [[ "$("$readlink_bin" -f -- "$output_root")" == "$output_root" \
    && "$("$stat_bin" -c '%d:%i' -- "$output_root")" == "$output_root_identity" ]] \
    || fail 'output-root path/inode changed' 69
  for path in "${expected_files[@]}"; do
    allowed["$path"]=1
  done
  [[ "${#actual[@]}" -eq "${#expected_files[@]}" ]] \
    || fail 'output artifact count differs' 69
  for path in "${actual[@]}"; do
    [[ -n "${allowed[$path]+present}" ]] \
      || fail "unexpected output artifact: $path" 69
    require_plain_file "$path" 'diagnostic output artifact'
  done
  for path in "${json_files[@]}"; do
    mode_value="$("$stat_bin" -c '%a' -- "$path")"
    [[ "$mode_value" == 400 ]] || fail "JSON receipt mode differs: $path" 69
  done
  for path in "${log_files[@]}"; do
    mode_value="$("$stat_bin" -c '%a' -- "$path")"
    if [[ "$final_modes" == final ]]; then
      [[ "$mode_value" == 400 ]] || fail "sealed log mode differs: $path" 69
    else
      [[ "$mode_value" == 600 ]] || fail "live log mode differs: $path" 69
    fi
  done
}

validate_output_set live
top_log="$output_root/top-$mode.log"
exec 3>"$top_log"
log_files+=("$top_log")
expected_files+=("$top_log")
require_plain_file "$top_log" 'top launcher log'
[[ "$("$stat_bin" -c '%a' -- "$top_log")" == 600 ]] \
  || fail 'top launcher log reservation mode differs' 67
printf 'schema=source8-matrix-top-log-v1 mode=%s allocation=%s source_sha256=%s\n' \
  "$mode" "$allocation_job_id" "$diagnostic_sha256" >&3

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

for rank in 0 1 2 3 4 5 6 7; do
  printf -v rank_label '%02d' "$rank"
  rank_cache="${rank_caches[rank]}"
  log_prefix="$output_root/$mode-rank-$rank_label"
  rank_out="$log_prefix.out"
  rank_err="$log_prefix.err"
  if [[ "$mode" == preflight ]]; then
    rank_receipt="$output_root/preflight-rank-$rank_label.json"
  else
    rank_receipt="$output_root/shard-$rank_label-of-08.json"
  fi
  [[ ! -e "$rank_out" && ! -L "$rank_out" \
    && ! -e "$rank_err" && ! -L "$rank_err" \
    && ! -e "$rank_receipt" && ! -L "$rank_receipt" ]] \
    || fail "rank $rank output reservation is not fresh" 67
  log_files+=("$rank_out" "$rank_err")
  json_files+=("$rank_receipt")
  expected_files+=("$rank_out" "$rank_err" "$rank_receipt")
  if [[ "$mode" == preflight ]]; then
    run_clean "$rank_cache" "ROCR_VISIBLE_DEVICES=$rank" \
      "$python_bin" -I -B "$diagnostic_source" preflight \
      "${common[@]}" --rank "$rank" >"$rank_out" 2>"$rank_err" &
  else
    run_clean "$rank_cache" "ROCR_VISIBLE_DEVICES=$rank" \
      "$python_bin" -I -B "$diagnostic_source" worker \
      "${common[@]}" --rank "$rank" --world-size 8 \
      >"$rank_out" 2>"$rank_err" &
  fi
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
pids=()
[[ "$failed" -eq 0 ]] \
  || fail 'one or more GPU ranks failed; aggregate is forbidden' 70
printf 'all8_rank_processes_completed=true\n' >&3
validate_output_set live
validate_cache_roots || fail 'rank-local runtime cache closure changed' 70

if [[ "$mode" == full ]]; then
  aggregate_out="$output_root/aggregate.out"
  aggregate_err="$output_root/aggregate.err"
  aggregate_receipt="$output_root/aggregate-receipt.json"
  [[ ! -e "$aggregate_out" && ! -L "$aggregate_out" \
    && ! -e "$aggregate_err" && ! -L "$aggregate_err" \
    && ! -e "$aggregate_receipt" && ! -L "$aggregate_receipt" ]] \
    || fail 'aggregate output reservation is not fresh' 67
  log_files+=("$aggregate_out" "$aggregate_err")
  expected_files+=("$aggregate_out" "$aggregate_err")
  # The candidate-bearing r4 bytes have not been hashed or opened above.  The
  # aggregate source first validates all eight sealed rows, then opens them.
  run_clean "$control_cache" "$python_bin" -I -B "$diagnostic_source" aggregate \
    --input-manifest "$input_manifest" \
    --expected-input-manifest-sha256 "$input_manifest_sha256" \
    --expected-source-sha256 "$diagnostic_sha256" \
    --legacy-all3-aggregate "$legacy_all3_aggregate" \
    --expected-legacy-all3-aggregate-sha256 "$legacy_all3_aggregate_sha256" \
    --output-root "$output_root" >"$aggregate_out" 2>"$aggregate_err"
  json_files+=("$aggregate_receipt")
  expected_files+=("$aggregate_receipt")
  printf 'aggregate_completed=true legacy_bytes_opened_only_inside_validated_aggregate=true\n' >&3
fi

validate_output_set live
validate_cache_roots || fail 'runtime cache closure changed before cleanup' 70
remove_runtime_scratch_exact || fail 'exact runtime cache cleanup failed' 73
printf 'rank_local_runtime_cache_cleanup=true\n' >&3
exec 3>&-

for path in "${log_files[@]}"; do
  "$chmod_bin" 0400 -- "$path" || fail "cannot seal log $path" 69
done
validate_output_set final
"$chmod_bin" 0500 -- "$output_root" || fail 'cannot seal output root' 69
[[ "$("$stat_bin" -c '%u:%a' -- "$output_root")" == "$current_uid:500" \
  && "$("$stat_bin" -c '%d:%i' -- "$output_root")" == "$output_root_identity" ]] \
  || fail 'sealed output-root ownership/mode/inode differs' 69

printf 'registered source8 DINO pair matrix %s complete: %s\n' "$mode" "$output_root"
