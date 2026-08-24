#!/usr/bin/env bash
#SBATCH --job-name=saic-ssft-r2v-i0-pair-v2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-289,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

# Minimal deterministic SSFT-v3 follow-up.  The preregistration never assigned
# a high/low sigma boundary, so this launcher does not invent one.  It runs the
# uniquely defined whole-schedule endpoints R00 (visual I0 always off) and R11
# (visual I0 always on), fixing the R2V prompt and every other registered knob.

fail() { echo "[saic-ssft-r2v-pair-v2] ERROR: $*" >&2; exit 2; }
require_abs() { [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute/non-root"; }
require_sha1() { [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be full lowercase SHA-1"; }
require_sha256() { [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be lowercase SHA-256"; }
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() { [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"; }

[[ "${SAIC_SSFT_R2V_PAIR_ACK_DIAGNOSTIC_ONLY:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_R2V_PAIR_ACK_DIAGNOSTIC_ONLY=YES"
[[ "${SAIC_SSFT_R2V_PAIR_ACK_NO_TRAINING_SELECTION_OR_SUCCESS:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_R2V_PAIR_ACK_NO_TRAINING_SELECTION_OR_SUCCESS=YES"
[[ "${SAIC_SSFT_R2V_PAIR_ACK_HIGH_LOW_UNSPECIFIED:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_R2V_PAIR_ACK_HIGH_LOW_UNSPECIFIED=YES"

source_archive="${SAIC_SSFT_R2V_PAIR_SOURCE_ARCHIVE:?set scoped immutable git archive}"
source_archive_sha256="${SAIC_SSFT_R2V_PAIR_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SAIC_SSFT_R2V_PAIR_SOURCE_REVISION:?set archive Git revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
python_bin="${SAIC_SSFT_R2V_PAIR_PYTHON_BIN:?set vace Python executable}"
output_root="${SAIC_SSFT_R2V_PAIR_OUTPUT_ROOT:?set fresh output root}"

readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly upstream_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/ssft-t0-i0-ec4bfb6-j6
readonly source_manifest="${upstream_root}/sealed-inputs/saic_reversible_source_set_v1.json"
readonly event_bank="${upstream_root}/sealed-inputs/saic_pure_t2v_event_bank_v1.json"
readonly source_manifest_sha256=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly event_bank_sha256=623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927
readonly upstream_parent_receipt="${upstream_root}/all8-parent-receipt.json"
readonly upstream_parent_receipt_sha256=286a0ece98f37e100a9f7e1ab9b543c689768d0808284c0efbe68d6a2254fdf9
readonly dog_row=fit-dog-00-7b88a1ca1f804f41
readonly dog_seed=2026082101
readonly dog_source_sha256=4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed
readonly dog_source_clean="${upstream_root}/groups/dog/source.clean-latent.safetensors"
readonly dog_source_clean_receipt="${dog_source_clean}.receipt.json"
readonly dog_source_clean_sha256=e3560e77546d3936f7e7231d5aceb78b8f29ea379b98748c26e4b37a5f277d7a
readonly dog_source_clean_receipt_sha256=bf1a817b6ab2f415311c19b0111f90c7a5537b3243df135d5fbcba06f63c37f1
readonly dog_source_clean_tensor_sha256=86c6b10800cfee13a1c3ca95b388a6ce8f310af6cbb84eea462728d8cd046343
readonly dog_frame0_tensor_sha256=995ac093c20fd07ef7018bdf942a1d7acb971c357effa53d4fddf5ac9e4e1f14
readonly human_row=fit-human-00-a35b590961d24694
readonly human_seed=2026082121
readonly human_source_sha256=6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed
readonly human_source_clean="${upstream_root}/groups/human/source.clean-latent.safetensors"
readonly human_source_clean_receipt="${human_source_clean}.receipt.json"
readonly human_source_clean_sha256=9cab2ac419833a3d8451b24c578c3ca3b466341f7bc17ea40e4067c8ebf6b7e9
readonly human_source_clean_receipt_sha256=ffdc8e4bce3181e0328f0714289b9d03d3c25ff5ac4ae57357d984101ddae0ba
readonly human_source_clean_tensor_sha256=2f7333434783bf681005c62200877aca1f95a395f21f228a2e83689ea0e1b263
readonly human_frame0_tensor_sha256=a6ce585d73bfa7a5e607356c9c06c25ec4f5e1d6f8c3b5e2c83d685b6910c6e1
readonly arms=(R00 R11)

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin output_root; do
  require_abs "${!name}" "$name"
done
require_sha1 "$source_revision" source_revision
require_sha256 "$source_archive_sha256" source_archive_sha256
for name in pinned_checkpoint_tree_sha256 pinned_checkpoint_manifest_sha256 \
  source_manifest_sha256 event_bank_sha256 upstream_parent_receipt_sha256 \
  dog_source_sha256 dog_source_clean_sha256 dog_source_clean_receipt_sha256 \
  dog_source_clean_tensor_sha256 dog_frame0_tensor_sha256 human_source_sha256 \
  human_source_clean_sha256 human_source_clean_receipt_sha256 \
  human_source_clean_tensor_sha256 human_frame0_tensor_sha256; do
  require_sha256 "${!name}" "$name"
done
for path in "$source_archive" "$checkpoint_manifest" "$python_bin" \
  "$source_manifest" "$event_bank" "$upstream_parent_receipt" \
  "$dog_source_clean" "$dog_source_clean_receipt" \
  "$human_source_clean" "$human_source_clean_receipt"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required plain file absent: $path"
done
for path in "$bernini_root" "$veomni_root" "$checkpoint"; do
  [[ -d "$path" && ! -L "$path" ]] || fail "required plain directory absent: $path"
done
[[ -x "$python_bin" && ! -L "$python_bin" ]] || fail "python must be plain executable"
source_archive="$(realpath -e -- "$source_archive")"
bernini_root="$(realpath -e -- "$bernini_root")"
veomni_root="$(realpath -e -- "$veomni_root")"
checkpoint="$(realpath -e -- "$checkpoint")"
checkpoint_manifest="$(realpath -e -- "$checkpoint_manifest")"
python_bin="$(realpath -e -- "$python_bin")"
check_hash "$source_archive" "$source_archive_sha256" source_archive
check_hash "$checkpoint_manifest" "$pinned_checkpoint_manifest_sha256" checkpoint_manifest
check_hash "$source_manifest" "$source_manifest_sha256" upstream_source_manifest
check_hash "$event_bank" "$event_bank_sha256" upstream_event_bank
check_hash "$upstream_parent_receipt" "$upstream_parent_receipt_sha256" upstream_parent_receipt
check_hash "$dog_source_clean" "$dog_source_clean_sha256" dog_source_coordinate
check_hash "$dog_source_clean_receipt" "$dog_source_clean_receipt_sha256" dog_source_receipt
check_hash "$human_source_clean" "$human_source_clean_sha256" human_source_coordinate
check_hash "$human_source_clean_receipt" "$human_source_clean_receipt_sha256" human_source_receipt
for path in "$dog_source_clean" "$dog_source_clean_receipt" "$human_source_clean" "$human_source_clean_receipt" "$source_manifest" "$event_bank"; do
  [[ "$(stat -c '%a:%h' -- "$path")" == 444:1 ]] || fail "upstream input seal differs: $path"
done
[[ "$(git get-tar-commit-id <"$source_archive")" == "$source_revision" ]] || fail "archive revision differs"

output_parent="$(dirname -- "$output_root")"
output_name="${output_root##*/}"
[[ "$(realpath -m -- "$output_root")" == "$output_root" ]] || fail "output root must be canonical"
[[ "$output_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename unsafe"
[[ -d "$output_parent" && ! -L "$output_parent" && -w "$output_parent" ]] || fail "output parent differs"
[[ ! -e "$output_root" && ! -L "$output_root" ]] || fail "output root must be fresh"
[[ "${SLURM_JOB_ID:?Slurm job required}" =~ ^[0-9]+$ ]] || fail "job id differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
scratch_parent="$(realpath -e -- "$scratch_parent")"
task_scratch="$(mktemp -d "${scratch_parent%/}/ssft-r2v-pair-v2-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_inode="$(stat -c '%d:%i' -- "$task_scratch")"
owned_output=false
published=false
output_inode=""
dog_pid=""
human_pid=""
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  for pid in "$dog_pid" "$human_pid"; do [[ "$pid" =~ ^[0-9]+$ ]] && kill "$pid" 2>/dev/null || true; done
  for pid in "$dog_pid" "$human_pid"; do [[ "$pid" =~ ^[0-9]+$ ]] && wait "$pid" 2>/dev/null || true; done
  if [[ "$published" != true && "$owned_output" == true && -d "$output_root" && ! -L "$output_root" ]]; then
    [[ "$(stat -c '%d:%i' -- "$output_root")" == "$output_inode" ]] || cleanup_failed=1
    if [[ "$cleanup_failed" == 0 ]]; then chmod -R u+w -- "$output_root" && rm -rf -- "$output_root" || cleanup_failed=1; fi
  fi
  if [[ -d "$task_scratch" && ! -L "$task_scratch" && "$(stat -c '%d:%i' -- "$task_scratch")" == "$task_scratch_inode" ]]; then
    chmod -R u+w -- "$task_scratch" && rm -rf -- "$task_scratch" || cleanup_failed=1
  else
    cleanup_failed=1
  fi
  if [[ "$status" == 0 && "$cleanup_failed" == 0 && "$published" == true ]]; then
    echo "[saic-ssft-r2v-pair-v2] PASS arms=R00,R11 shared_job132387_source=true shared_frame0=true matched_noise=true exact81=true exact40=true r2v_prompt=true training=false selection=false output=$output_root"
  else
    [[ "$status" != 0 ]] || status=2
    echo "[saic-ssft-r2v-pair-v2] EXIT status=$status published=$published cleanup_verified=$([[ "$cleanup_failed" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "$status"
}
trap cleanup EXIT TERM INT

mkdir -- "$task_scratch/source-tree" "$task_scratch/logs" "$task_scratch/cache"
archive_copy="$task_scratch/source.tar"
cp -- "$source_archive" "$archive_copy"
chmod 0444 -- "$archive_copy"
check_hash "$archive_copy" "$source_archive_sha256" staged_archive
"$python_bin" -B - "$archive_copy" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/infer_saic_source_state_flow_transport_v2.py",
    "methods/bernini_action_editing/infer_saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/materialize_saic_frame0_latent_v1.py",
    "methods/bernini_action_editing/saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/saic_native_source_state_field_v1.py",
    "methods/bernini_action_editing/scripts/auh_infer_saic_fixed_r2v_visual_i0_pair_all8_v2.sbatch",
    "methods/bernini_action_editing/tests/test_materialize_saic_frame0_latent_v1.py",
    "methods/bernini_action_editing/tests/test_infer_saic_source_state_flow_transport_v2.py",
    "methods/bernini_action_editing/tests/test_auh_infer_saic_fixed_r2v_visual_i0_pair_all8_v2.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for item in handle.getmembers():
        path = PurePosixPath(item.name)
        name = path.as_posix().rstrip("/")
        scoped = name in {"methods", "methods/bernini_action_editing"} or name.startswith("methods/bernini_action_editing/")
        if (not name or path.is_absolute() or ".." in path.parts or not scoped or name in seen or item.issym() or item.islnk() or item.isdev() or item.isfifo() or (not item.isfile() and not item.isdir())):
            raise SystemExit("unsafe/duplicate/out-of-scope archive member")
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks fixed-R2V closure: {missing}")
PY
tar --no-same-owner --no-same-permissions -xf "$archive_copy" -C "$task_scratch/source-tree" methods/bernini_action_editing
method_root="$task_scratch/source-tree/methods/bernini_action_editing"
runner="$method_root/infer_saic_source_state_flow_transport_v2.py"
materializer="$method_root/materialize_saic_frame0_latent_v1.py"
launcher="$method_root/scripts/auh_infer_saic_fixed_r2v_visual_i0_pair_all8_v2.sbatch"
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "$launcher")" ]] || fail "running launcher differs from archive"
find "$task_scratch/source-tree" -type f -exec chmod a-w -- {} +
find "$task_scratch/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
for test_file in test_saic_source_state_flow_transport_v1.py \
  test_saic_native_source_state_field_v1.py \
  test_materialize_saic_source_clean_latent_v1.py \
  test_infer_saic_source_state_flow_transport_v1.py \
  test_materialize_saic_frame0_latent_v1.py \
  test_infer_saic_source_state_flow_transport_v2.py \
  test_auh_infer_saic_fixed_r2v_visual_i0_pair_all8_v2.py; do
  PYTHONPATH="$method_root:$bernini_root:$veomni_root" "$python_bin" -B "$method_root/tests/$test_file"
done

mkdir -- "$output_root"
owned_output=true
output_inode="$(stat -c '%d:%i' -- "$output_root")"
mkdir -- "$output_root/groups" "$output_root/groups/dog" "$output_root/groups/human" "$output_root/logs" "$output_root/provenance"
durable_archive="$output_root/provenance/source.tar"
cp -- "$archive_copy" "$durable_archive"
chmod 0444 -- "$durable_archive"
check_hash "$durable_archive" "$source_archive_sha256" durable_archive

run_frame0() (
  set -Eeuo pipefail
  local group="$1" row="$2" source_sha="$3" expected_raw="$4" visible="$5"
  local out="$output_root/groups/$group/reference.frame0-latent.safetensors"
  local cache="$task_scratch/cache/frame0-$group"
  mkdir -p -- "$cache/miopen" "$cache/triton" "$cache/xdg" "$cache/tmp" "$cache/pycache"
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="$visible" MIOPEN_USER_DB_PATH="$cache/miopen" \
    TRITON_CACHE_DIR="$cache/triton" XDG_CACHE_HOME="$cache/xdg" TMPDIR="$cache/tmp" \
    PYTHONPYCACHEPREFIX="$cache/pycache" PYTHONPATH="$method_root:$bernini_root:$veomni_root" \
    "$python_bin" -B "$materializer" \
      --bernini-root "$bernini_root" --veomni-root "$veomni_root" \
      --checkpoint "$checkpoint" --checkpoint-content-manifest "$checkpoint_manifest" \
      --source-manifest "$source_manifest" --expected-source-manifest-sha256 "$source_manifest_sha256" \
      --row-id "$row" --expected-source-video-sha256 "$source_sha" \
      --expected-reference-frame0-tensor-raw-sha256 "$expected_raw" \
      --expected-bernini-commit "$pinned_bernini_commit" --expected-veomni-commit "$pinned_veomni_commit" \
      --expected-checkpoint-tree-sha256 "$pinned_checkpoint_tree_sha256" \
      --method-source-revision "$source_revision" --method-source-archive "$archive_copy" \
      --durable-method-source-archive "$durable_archive" --method-source-archive-sha256 "$source_archive_sha256" \
      --device cuda:0 --output "$out"
)

run_frame0 dog "$dog_row" "$dog_source_sha256" "$dog_frame0_tensor_sha256" 0 >"$output_root/logs/frame0.dog.log" 2>&1 & dog_pid=$!
run_frame0 human "$human_row" "$human_source_sha256" "$human_frame0_tensor_sha256" 4 >"$output_root/logs/frame0.human.log" 2>&1 & human_pid=$!
dog_status=0; human_status=0
wait "$dog_pid" || dog_status=$?; wait "$human_pid" || human_status=$?; dog_pid=""; human_pid=""
if [[ "$dog_status" != 0 || "$human_status" != 0 ]]; then
  tail -n 200 "$output_root/logs/frame0.dog.log" >&2 || true
  tail -n 200 "$output_root/logs/frame0.human.log" >&2 || true
  fail "frame0 materialization failed dog=$dog_status human=$human_status"
fi

read_frame0_binding() {
  "$python_bin" -B - "$1" "$2" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys
from safetensors import safe_open
artifact, expected_raw = Path(sys.argv[1]), sys.argv[2]
receipt = artifact.with_name(artifact.name + ".receipt.json")
for path in (artifact, receipt):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
        raise SystemExit("frame0 seal differs")
with safe_open(str(artifact), framework="pt", device="cpu") as opened:
    tensor = opened.get_tensor("reference_frame0_latent").contiguous()
raw = tensor.view(__import__("torch").uint8).numpy().tobytes()
tensor_sha = hashlib.sha256(raw).hexdigest()
data = json.loads(receipt.read_text())
encoding = data.get("encoding", {})
if (
    data["artifact"]["tensor_raw_sha256"] != tensor_sha
    or encoding.get("expected_job132387_frame0_tensor_raw_sha256") != expected_raw
    or encoding.get("actual_reference_frame0_tensor_raw_sha256") != tensor_sha
    or encoding.get("job132387_frame0_tensor_raw_sha256_match")
    is not (tensor_sha == expected_raw)
):
    raise SystemExit("frame0 raw SHA comparison receipt differs")
print(hashlib.sha256(artifact.read_bytes()).hexdigest(), hashlib.sha256(receipt.read_bytes()).hexdigest(), tensor_sha, str(tensor_sha == expected_raw).lower())
PY
}
read -r dog_frame0_sha dog_frame0_receipt_sha dog_frame0_raw dog_frame0_prior_match <<<"$(read_frame0_binding "$output_root/groups/dog/reference.frame0-latent.safetensors" "$dog_frame0_tensor_sha256")"
read -r human_frame0_sha human_frame0_receipt_sha human_frame0_raw human_frame0_prior_match <<<"$(read_frame0_binding "$output_root/groups/human/reference.frame0-latent.safetensors" "$human_frame0_tensor_sha256")"
for name in dog_frame0_sha dog_frame0_receipt_sha dog_frame0_raw human_frame0_sha human_frame0_receipt_sha human_frame0_raw; do require_sha256 "${!name}" "$name"; readonly "$name=${!name}"; done
for name in dog_frame0_prior_match human_frame0_prior_match; do [[ "${!name}" == true || "${!name}" == false ]] || fail "$name must be boolean"; readonly "$name=${!name}"; done

job_mod=$((SLURM_JOB_ID % 20000)); dog_port=$((20000 + 2 * job_mod)); human_port=$((dog_port + 1))
run_cell() (
  set -Eeuo pipefail
  local group="$1" row="$2" seed="$3" arm="$4" visible="$5" port="$6"
  local source_clean source_receipt source_sha source_receipt_sha source_raw frame0 frame0_receipt frame0_sha frame0_receipt_sha frame0_raw
  if [[ "$group" == dog ]]; then
    source_clean="$dog_source_clean"; source_receipt="$dog_source_clean_receipt"; source_sha="$dog_source_clean_sha256"; source_receipt_sha="$dog_source_clean_receipt_sha256"; source_raw="$dog_source_clean_tensor_sha256"
    frame0="$output_root/groups/dog/reference.frame0-latent.safetensors"; frame0_sha="$dog_frame0_sha"; frame0_receipt_sha="$dog_frame0_receipt_sha"; frame0_raw="$dog_frame0_raw"
  else
    source_clean="$human_source_clean"; source_receipt="$human_source_clean_receipt"; source_sha="$human_source_clean_sha256"; source_receipt_sha="$human_source_clean_receipt_sha256"; source_raw="$human_source_clean_tensor_sha256"
    frame0="$output_root/groups/human/reference.frame0-latent.safetensors"; frame0_sha="$human_frame0_sha"; frame0_receipt_sha="$human_frame0_receipt_sha"; frame0_raw="$human_frame0_raw"
  fi
  frame0_receipt="${frame0}.receipt.json"
  local cache="$task_scratch/cache/run-$group" output="$output_root/groups/$group/$arm.mp4"
  mkdir -p -- "$cache/miopen" "$cache/triton" "$cache/xdg" "$cache/tmp" "$cache/pycache"
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="$visible" MIOPEN_USER_DB_PATH="$cache/miopen" TRITON_CACHE_DIR="$cache/triton" \
    XDG_CACHE_HOME="$cache/xdg" TMPDIR="$cache/tmp" PYTHONPYCACHEPREFIX="$cache/pycache" \
    TORCHELASTIC_ERROR_FILE="$cache/torch-elastic-$arm.json" PYTHONPATH="$method_root:$bernini_root:$veomni_root" \
    "$python_bin" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$port" "$runner" \
      --bernini-root "$bernini_root" --veomni-root "$veomni_root" --checkpoint "$checkpoint" \
      --checkpoint-content-manifest "$checkpoint_manifest" --source-manifest "$source_manifest" \
      --expected-source-manifest-sha256 "$source_manifest_sha256" --event-bank "$event_bank" \
      --expected-event-bank-sha256 "$event_bank_sha256" --row-id "$row" --branch forward --rollout-seed "$seed" --arm "$arm" \
      --source-clean-latent "$source_clean" --source-clean-latent-receipt "$source_receipt" \
      --expected-source-clean-latent-sha256 "$source_sha" --expected-source-clean-latent-receipt-sha256 "$source_receipt_sha" \
      --expected-source-clean-tensor-raw-sha256 "$source_raw" --reference-frame0-latent "$frame0" \
      --reference-frame0-latent-receipt "$frame0_receipt" --expected-reference-frame0-latent-sha256 "$frame0_sha" \
      --expected-reference-frame0-latent-receipt-sha256 "$frame0_receipt_sha" --expected-reference-frame0-tensor-raw-sha256 "$frame0_raw" \
      --output "$output" --expected-bernini-commit "$pinned_bernini_commit" --expected-veomni-commit "$pinned_veomni_commit" \
      --expected-checkpoint-tree-sha256 "$pinned_checkpoint_tree_sha256" --method-source-revision "$source_revision" \
      --method-source-archive "$archive_copy" --durable-method-source-archive "$durable_archive" --method-source-archive-sha256 "$source_archive_sha256"
)

run_wave() {
  local arm="$1"
  echo "[saic-ssft-r2v-pair-v2] START arm=$arm dog=GPU0-3 human=GPU4-7"
  run_cell dog "$dog_row" "$dog_seed" "$arm" 0,1,2,3 "$dog_port" >"$output_root/logs/$arm.dog.log" 2>&1 & dog_pid=$!
  run_cell human "$human_row" "$human_seed" "$arm" 4,5,6,7 "$human_port" >"$output_root/logs/$arm.human.log" 2>&1 & human_pid=$!
  local ds=0 hs=0; wait "$dog_pid" || ds=$?; wait "$human_pid" || hs=$?; dog_pid=""; human_pid=""
  if [[ "$ds" != 0 || "$hs" != 0 ]]; then tail -n 240 "$output_root/logs/$arm.dog.log" >&2 || true; tail -n 240 "$output_root/logs/$arm.human.log" >&2 || true; fail "arm $arm failed dog=$ds human=$hs"; fi
}
for arm in "${arms[@]}"; do run_wave "$arm"; done

"$python_bin" -B - "$output_root" "$source_revision" "$source_archive_sha256" "$SLURM_JOB_ID" \
  "$dog_source_clean_sha256" "$dog_source_clean_receipt_sha256" "$dog_source_clean_tensor_sha256" "$dog_frame0_sha" "$dog_frame0_receipt_sha" "$dog_frame0_raw" \
  "$dog_frame0_tensor_sha256" "$dog_frame0_prior_match" \
  "$human_source_clean_sha256" "$human_source_clean_receipt_sha256" "$human_source_clean_tensor_sha256" "$human_frame0_sha" "$human_frame0_receipt_sha" "$human_frame0_raw" \
  "$human_frame0_tensor_sha256" "$human_frame0_prior_match" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
(root_text, revision, archive_sha, job_id, *pins) = sys.argv[1:]
root = Path(root_text); groups = ("dog", "human"); arms = ("R00", "R11")
pin_keys = ("source_file", "source_receipt", "source_raw", "frame0_file", "frame0_receipt", "frame0_raw", "prior_frame0_raw", "prior_match")
pinmap = {groups[i]: dict(zip(pin_keys, pins[i*8:(i+1)*8])) for i in range(2)}
canonical = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
file_sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
children = []
for group in groups:
    rows = []
    for arm in arms:
        video = root / "groups" / group / f"{arm}.mp4"
        receipt_path = Path(str(video) + ".receipt.json")
        latent = Path(str(video) + ".normalized-clean-latent.safetensors")
        for path in (video, receipt_path, latent):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
                raise SystemExit("runner artifact seal differs")
        raw = receipt_path.read_bytes(); receipt = json.loads(raw)
        if raw != canonical(receipt) + b"\n" or receipt.get("arm", {}).get("arm") != arm:
            raise SystemExit("runner receipt canonical/arm differs")
        unsigned = dict(receipt); declared = unsigned.pop("receipt_digest")
        if hashlib.sha256(canonical(unsigned)).hexdigest() != declared:
            raise SystemExit("runner receipt digest differs")
        if receipt.get("prompt_contract", {}).get("task_name") != "r2v":
            raise SystemExit("runner prompt is not fixed R2V")
        spec = receipt["arm"]
        if spec["candidate_schedule"] != [1] * 40 or spec["anc_enabled"] is not False or spec["aggregation_mode"] != "uniform" or spec["temperature"] is not None or spec["anchor_latent_phase_zero"] is not False:
            raise SystemExit("runner fixed knobs differ")
        transport = receipt["transport"]; source = transport["sealed_source_coordinate"]; frame0 = transport["sealed_frame0_coordinate"]
        p = pinmap[group]
        if source["artifact_sha256"] != p["source_file"] or source["receipt_file_sha256"] != p["source_receipt"] or source["tensor_raw_sha256"] != p["source_raw"] or source.get("reused_from_slurm_job_id") != "132387" or source.get("upstream_coordinate_relabelled") is not False:
            raise SystemExit("Job132387 source coordinate binding differs")
        if frame0["artifact_sha256"] != p["frame0_file"] or frame0["receipt_file_sha256"] != p["frame0_receipt"] or frame0["tensor_raw_sha256"] != p["frame0_raw"] or frame0.get("job132387_ephemeral_i0_tensor_raw_sha256") != p["prior_frame0_raw"] or frame0.get("matches_job132387_ephemeral_i0_tensor") is not (p["prior_match"] == "true"):
            raise SystemExit("shared frame0 coordinate binding differs")
        enabled = arm == "R11"
        if transport.get("reference_passed_to_native_field") is not enabled or transport.get("reference_visual_i0_enabled_all_exact40_cells") is not enabled or transport.get("reference_visual_i0_disabled_all_exact40_cells") is not (not enabled):
            raise SystemExit("whole-schedule visual-I0 treatment differs")
        if any(value is not False for key, value in receipt["authority"].items() if key != "frozen_inference_execution_receipt") or receipt["authority"]["frozen_inference_execution_receipt"] is not True:
            raise SystemExit("runner authority differs")
        rows.append({"arm": arm, "video_sha256": file_sha(video), "receipt_file_sha256": file_sha(receipt_path), "receipt_digest": declared, "candidate_zero_noise_sha256": receipt["noise"]["actual_candidate_zero_subbank_sha256"], "source_coordinate": p["source_raw"], "frame0_coordinate": p["frame0_raw"], "job132387_frame0_coordinate": p["prior_frame0_raw"], "job132387_frame0_exact_match": p["prior_match"] == "true"})
    if rows[0]["candidate_zero_noise_sha256"] != rows[1]["candidate_zero_noise_sha256"]:
        raise SystemExit("R00/R11 matched candidate-zero noise differs")
    children.append({"group": group, "arms": rows, "matched_noise_verified": True})
parent = {
    "schema_version": "bernini-saic-fixed-r2v-visual-i0-pair-all8-receipt-v2",
    "method": "frozen-bernini-fixed-r2v-visual-i0-pair-all8",
    "slurm_job_id": job_id,
    "source_revision": revision,
    "source_archive_sha256": archive_sha,
    "upstream_job132387_root": "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_saic_v1_20260809/runs/ssft-t0-i0-ec4bfb6-j6",
    "arm_order": list(arms),
    "arm_definition": {
        "R00": {"r2v_system_prompt": True, "visual_i0_high": "off", "visual_i0_low": "off", "whole_exact40_off": True},
        "R11": {"r2v_system_prompt": True, "visual_i0_high": "on", "visual_i0_low": "on", "whole_exact40_on": True},
    },
    "unexecuted_preregistered_cells": {
        "R10": "high/low sigma boundary is not uniquely specified",
        "R01": "high/low sigma boundary is not uniquely specified",
    },
    "fixed_contract": {"K": 1, "anc": False, "aggregation": "uniform", "temperature": None, "phase0_anchor": False, "frames": 81, "fps": 25, "steps": 40, "flow_shift": 5.0, "matched_noise": True},
    "groups": children,
    "authority": {"quality": False, "semantic_success": False, "evaluation": False, "selection": False, "training": False, "optimizer": False, "checkpoint_or_lora": False, "production": False},
}
parent["receipt_digest"] = hashlib.sha256(canonical(parent)).hexdigest()
payload = canonical(parent) + b"\n"; path = root / "fixed-r2v-pair-parent-receipt.json"
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
os.chmod(path, 0o444)
if path.read_bytes() != payload: raise SystemExit("parent receipt reopen differs")
print(parent["receipt_digest"])
PY
find "$output_root/logs" -type f -exec chmod 0400 -- {} +
find "$output_root" -type f ! -path "$output_root/logs/*" -exec chmod 0444 -- {} +
find "$output_root" -type d -exec chmod 0500 -- {} +
published=true
