#!/usr/bin/env bash
# Run the frozen r8 exact60 temporal-representation diagnostic on one owned GPU.

set -euo pipefail
umask 077

if [[ "$#" -ne 17 ]]; then
  echo "usage: $0 <allocation-job-id> <logical-gpu-0..7> <python-bin> <diagnostic.py> <diagnostic-sha256> <cyclic-input-manifest> <cyclic-input-sha256> <terminal-evidence> <visual-checkpoint> <checkpoint-manifest> <evaluator-spec> <evaluator-spec-sha256> <visual-scorer.py> <visual-scorer-sha256> <visual-contract.py> <visual-contract-sha256> <fresh-output-root>" >&2
  exit 64
fi

allocation_job_id="$1"; logical_gpu="$2"; python_bin="$3"
diagnostic_source="$4"; diagnostic_sha256="$5"; input_manifest="$6"
input_manifest_sha256="$7"; terminal_evidence="$8"; visual_checkpoint="$9"
checkpoint_manifest="${10}"; evaluator_spec="${11}"
evaluator_spec_sha256="${12}"; visual_scorer_source="${13}"
visual_scorer_sha256="${14}"; visual_contract_source="${15}"
visual_contract_sha256="${16}"; output_root="${17}"

readonly expected_python=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/runtime/venv-transformers-4.53.2/bin/python
readonly expected_diagnostic_sha=dd2bde7a3aebe1021d9e69611db9e01b56c99e8bab123485b453f46df21f38a9
readonly expected_input_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/allocation-134939-r8-sourcebound60-dinov2-full-28396417-r1/input-manifest.json
readonly expected_input_manifest_sha=28ff1e40f4dd314548616050013afdfb5e2a2a768aba9f0cbd4f00c9f6718c62
readonly expected_terminal_evidence=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/releases/saic-t2v-topup-r8-ddc8a79-r1/evidence-terminal-ddc8a79/saic-exact60-terminal-evidence-135056.json
readonly expected_terminal_evidence_sha=07a6ec7ccbe165d89aa8757985537ef18d62eea5d08e245e452b607dee5bd29a
readonly expected_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/vendor/dinov2-base-f9e44c8
readonly expected_checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/dinov2-base-f9e44c8.sha256
readonly expected_checkpoint_manifest_sha=b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea
readonly expected_evaluator_spec=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json
readonly expected_evaluator_spec_sha=6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736
readonly expected_visual_scorer=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py
readonly expected_visual_scorer_sha=9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39
readonly expected_visual_contract=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py
readonly expected_visual_contract_sha=183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a
readonly portable_ffprobe_dir=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime
readonly portable_ffprobe="$portable_ffprobe_dir/ffprobe"
readonly portable_ffprobe_sha=356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5

if [[ ! "$allocation_job_id" =~ ^[1-9][0-9]*$ || "${SLURM_JOB_ID:-}" != "$allocation_job_id" ]]; then echo "launcher must run inside the named live allocation" >&2; exit 65; fi
if [[ ! "$logical_gpu" =~ ^[0-7]$ ]]; then echo "logical GPU must be one registered index 0..7" >&2; exit 65; fi
allocated_node_count="${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-}}}"
if [[ "$allocated_node_count" != "1" ]]; then echo "launcher requires exactly one allocated node" >&2; exit 65; fi
if [[ "$python_bin" != "$expected_python" || ! -x "$python_bin" ]]; then echo "python identity differs" >&2; exit 66; fi
if [[ "$diagnostic_sha256" != "$expected_diagnostic_sha" ]]; then echo "diagnostic SHA argument differs from hard pin" >&2; exit 66; fi
if [[ "$input_manifest" != "$expected_input_manifest" || "$input_manifest_sha256" != "$expected_input_manifest_sha" || "$terminal_evidence" != "$expected_terminal_evidence" ]]; then echo "r8 input/terminal identity differs" >&2; exit 66; fi
if [[ "$visual_checkpoint" != "$expected_checkpoint" || "$checkpoint_manifest" != "$expected_checkpoint_manifest" || "$evaluator_spec" != "$expected_evaluator_spec" || "$evaluator_spec_sha256" != "$expected_evaluator_spec_sha" || "$visual_scorer_source" != "$expected_visual_scorer" || "$visual_scorer_sha256" != "$expected_visual_scorer_sha" || "$visual_contract_source" != "$expected_visual_contract" || "$visual_contract_sha256" != "$expected_visual_contract_sha" ]]; then echo "frozen evaluator identity differs" >&2; exit 66; fi
for plain_file in "$diagnostic_source" "$input_manifest" "$terminal_evidence" "$checkpoint_manifest" "$evaluator_spec" "$visual_scorer_source" "$visual_contract_source"; do
  if [[ ! -f "$plain_file" || -L "$plain_file" ]]; then echo "sealed input is absent or linked: $plain_file" >&2; exit 66; fi
done
if [[ ! -d "$visual_checkpoint" || -L "$visual_checkpoint" ]]; then echo "visual checkpoint root differs" >&2; exit 66; fi
if [[ "$(sha256sum "$diagnostic_source" | awk '{print $1}')" != "$expected_diagnostic_sha" || "$(sha256sum "$input_manifest" | awk '{print $1}')" != "$expected_input_manifest_sha" || "$(sha256sum "$terminal_evidence" | awk '{print $1}')" != "$expected_terminal_evidence_sha" || "$(sha256sum "$checkpoint_manifest" | awk '{print $1}')" != "$expected_checkpoint_manifest_sha" || "$(sha256sum "$evaluator_spec" | awk '{print $1}')" != "$expected_evaluator_spec_sha" || "$(sha256sum "$visual_scorer_source" | awk '{print $1}')" != "$expected_visual_scorer_sha" || "$(sha256sum "$visual_contract_source" | awk '{print $1}')" != "$expected_visual_contract_sha" ]]; then echo "one sealed source/spec SHA-256 differs" >&2; exit 66; fi
if [[ ! -x "$portable_ffprobe" || -L "$portable_ffprobe" || "$(sha256sum "$portable_ffprobe" | awk '{print $1}')" != "$portable_ffprobe_sha" ]]; then echo "portable ffprobe identity differs" >&2; exit 66; fi
if [[ "$output_root" != /* || "$output_root" == "/" || -e "$output_root" || -L "$output_root" ]]; then echo "output root must be fresh, absolute, and non-root" >&2; exit 67; fi

scratch_parent="${SLURM_TMPDIR:-/tmp}"
if [[ "$scratch_parent" != /* || ! -d "$scratch_parent" || -L "$scratch_parent" || ! -w "$scratch_parent" ]]; then echo "allocation-local scratch parent differs" >&2; exit 67; fi
scratch_parent="$(readlink -f -- "$scratch_parent")"
if [[ -z "$scratch_parent" || "$scratch_parent" == "/" ]]; then echo "scratch parent resolution differs" >&2; exit 67; fi
runtime_prefix="${scratch_parent%/}/saic-r8-temporal-${allocation_job_id}-gpu${logical_gpu}."
runtime_scratch="$(mktemp -d -- "${runtime_prefix}XXXXXXXX")"
if [[ "$runtime_scratch" != "$runtime_prefix"???????? || -L "$runtime_scratch" ]]; then echo "runtime scratch framing differs" >&2; exit 67; fi
chmod 0700 -- "$runtime_scratch"
runtime_identity="$(stat -c '%d:%i' -- "$runtime_scratch")"
current_uid="$(id -u)"
if [[ "$(readlink -f -- "$runtime_scratch")" != "$runtime_scratch" || "$(stat -c '%u:%a' -- "$runtime_scratch")" != "$current_uid:700" ]]; then echo "runtime scratch identity/mode differs" >&2; exit 67; fi

cleanup() {
  local status="$?"
  trap - EXIT INT TERM HUP
  if [[ -n "${runtime_scratch:-}" ]]; then
    if [[ ! -d "$runtime_scratch" || -L "$runtime_scratch" || "$(readlink -f -- "$runtime_scratch")" != "$runtime_scratch" || "$(stat -c '%d:%i' -- "$runtime_scratch")" != "$runtime_identity" || "$(stat -c '%u:%a' -- "$runtime_scratch")" != "$current_uid:700" ]]; then
      echo "unsafe runtime scratch cleanup refused; retained $runtime_scratch" >&2
      exit 73
    fi
    find "$runtime_scratch" -xdev -depth -mindepth 1 -delete
    rmdir -- "$runtime_scratch"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for child in home tmp xdg-config xdg-cache miopen-user miopen-custom torch-extensions triton torchinductor pycache; do
  mkdir -m 0700 -- "$runtime_scratch/$child"
done
export PATH="$portable_ffprobe_dir:${PATH:-/usr/bin:/bin}"
if [[ "$(command -v ffprobe)" != "$portable_ffprobe" ]]; then echo "portable ffprobe is not PATH-first" >&2; exit 66; fi

env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  PATH="$PATH" HOME="$runtime_scratch/home" TMPDIR="$runtime_scratch/tmp" \
  XDG_CONFIG_HOME="$runtime_scratch/xdg-config" XDG_CACHE_HOME="$runtime_scratch/xdg-cache" \
  MIOPEN_USER_DB_PATH="$runtime_scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$runtime_scratch/miopen-custom" \
  TORCH_EXTENSIONS_DIR="$runtime_scratch/torch-extensions" TRITON_CACHE_DIR="$runtime_scratch/triton" \
  TORCHINDUCTOR_CACHE_DIR="$runtime_scratch/torchinductor" PYTHONPYCACHEPREFIX="$runtime_scratch/pycache" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HF_HOME="$runtime_scratch/xdg-cache/huggingface" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false ROCR_VISIBLE_DEVICES="$logical_gpu" \
  "$python_bin" "$diagnostic_source" \
    --input-manifest "$input_manifest" --expected-input-manifest-sha256 "$input_manifest_sha256" \
    --terminal-evidence "$terminal_evidence" --expected-source-sha256 "$diagnostic_sha256" \
    --visual-checkpoint "$visual_checkpoint" --visual-checkpoint-manifest "$checkpoint_manifest" \
    --evaluator-spec "$evaluator_spec" --expected-evaluator-spec-sha256 "$evaluator_spec_sha256" \
    --visual-scorer-source "$visual_scorer_source" --expected-visual-scorer-sha256 "$visual_scorer_sha256" \
    --visual-contract-source "$visual_contract_source" --expected-visual-contract-sha256 "$visual_contract_sha256" \
    --output-root "$output_root"

chmod 0400 "$output_root/aggregate-receipt.json"
chmod 0500 "$output_root"
echo "r8 exact60 frozen temporal-representation diagnostic complete: $output_root authority=zero"
