#!/usr/bin/env bash
#SBATCH --job-name=saic-ssft-t0-i0-all8
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

# Frozen SSFT first pilot.  One node is split into two legal WORLD4/Ulysses-SP4
# groups.  In each of two synchronized waves all eight cards are assigned:
# GPUs 0..3 run the fixed dog cell and GPUs 4..7 run the fixed human cell.
# The v1 runner accepts exactly one arm per invocation, so it cannot retain a
# model across T0 and I0.  Consequently each group performs two model loads;
# this is recorded rather than hidden in the terminal receipt.

fail() { echo "[saic-ssft-all8] ERROR: $*" >&2; exit 2; }
require_absolute_nonroot() {
  [[ "$1" == /* && "$1" != / ]] || fail "$2 must be absolute and non-root"
}
require_sha1() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || fail "$2 must be a full lowercase Git SHA-1"
}
require_sha256() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]] || fail "$2 must be a lowercase SHA-256"
}
hash_file() { sha256sum -- "$1" | awk '{print $1}'; }
check_hash() {
  [[ "$(hash_file "$1")" == "$2" ]] || fail "$3 SHA-256 differs"
}

[[ "${SAIC_SSFT_ACK_FROZEN_DIAGNOSTIC_ONLY:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_ACK_FROZEN_DIAGNOSTIC_ONLY=YES"
[[ "${SAIC_SSFT_ACK_NO_SELECTION_TRAINING_OR_SUCCESS_CLAIM:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_ACK_NO_SELECTION_TRAINING_OR_SUCCESS_CLAIM=YES"
[[ "${SAIC_SSFT_ACK_RUNNER_RELOADS_MODEL_PER_ARM:-NO}" == YES ]] || \
  fail "set SAIC_SSFT_ACK_RUNNER_RELOADS_MODEL_PER_ARM=YES"

source_archive="${SAIC_SSFT_SOURCE_ARCHIVE:?set scoped immutable git archive}"
source_archive_sha256="${SAIC_SSFT_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SAIC_SSFT_SOURCE_REVISION:?set archive Git revision}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
python_bin="${SAIC_SSFT_PYTHON_BIN:?set vace Python executable}"
output_root="${SAIC_SSFT_OUTPUT_ROOT:?set fresh final output root}"

readonly pinned_bernini_commit=2d2b4591ac053ec25c6371b01a5a6746679e5793
readonly pinned_veomni_commit=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
readonly pinned_checkpoint_tree_sha256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca
readonly pinned_checkpoint_manifest_sha256=a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831
readonly pinned_source_manifest_raw_sha256=899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9
readonly pinned_source_manifest_content_sha256=9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f
readonly pinned_event_bank_raw_sha256=623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927
readonly pinned_event_bank_content_sha256=3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a
readonly dog_row=fit-dog-00-7b88a1ca1f804f41
readonly dog_seed=2026082101
readonly dog_source_sha256=4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed
readonly human_row=fit-human-00-a35b590961d24694
readonly human_seed=2026082121
readonly human_source_sha256=6e9381d3889437f618e1ec6b694703b10598c4b42d8b361b0442db7780be97ed
readonly pilot_arms=(T0 I0)

for name in source_archive bernini_root veomni_root checkpoint checkpoint_manifest python_bin output_root; do
  require_absolute_nonroot "${!name}" "${name}"
done
require_sha1 "${source_revision}" source_revision
require_sha256 "${source_archive_sha256}" source_archive_sha256
for name in pinned_checkpoint_tree_sha256 pinned_checkpoint_manifest_sha256 pinned_source_manifest_raw_sha256 pinned_source_manifest_content_sha256 pinned_event_bank_raw_sha256 pinned_event_bank_content_sha256 dog_source_sha256 human_source_sha256; do
  require_sha256 "${!name}" "${name}"
done

for name in source_archive checkpoint_manifest; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
[[ -f "${python_bin}" && ! -L "${python_bin}" && -x "${python_bin}" ]] || \
  fail "python_bin must be a plain executable"
source_archive="$(realpath -e -- "${source_archive}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
check_hash "${source_archive}" "${source_archive_sha256}" source_archive
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || \
  fail "source archive revision differs"

output_parent="$(dirname -- "${output_root}")"
output_name="${output_root##*/}"
[[ "${output_parent}" != / ]] || fail "output parent must not be filesystem root"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || \
  fail "output root must be canonical"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "output basename differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" && -w "${output_parent}" ]] || \
  fail "output parent must already be a writable plain directory"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || \
  fail "output root must be fresh; resume and overwrite are forbidden"
[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || \
  fail "Slurm job ID differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
scratch_parent="$(realpath -e -- "${scratch_parent}")"
task_scratch="$(mktemp -d "${scratch_parent%/}/saic-ssft-all8-${SLURM_JOB_ID}.XXXXXX")"
task_scratch_device_inode="$(stat -c '%d:%i' -- "${task_scratch}")"
[[ "${task_scratch_device_inode}" =~ ^[0-9]+:[0-9]+$ ]] || \
  fail "fresh task scratch device/inode identity differs"
readonly task_scratch_device_inode
published=false
owned_output=false
output_device_inode=""
dog_pid=""
human_pid=""

terminate_children() {
  local status="$1" child
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true
  done
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true
  done
  exit "${status}"
}

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  if [[ "${published}" != true && "${owned_output}" == true && -d "${output_root}" && ! -L "${output_root}" ]]; then
    [[ "${output_root}" == "${output_parent}/${output_name}" ]] || cleanup_failed=1
    [[ "${output_device_inode}" =~ ^[0-9]+:[0-9]+$ ]] || cleanup_failed=1
    [[ "$(stat -c '%d:%i' -- "${output_root}")" == "${output_device_inode}" ]] || cleanup_failed=1
    if [[ "${cleanup_failed}" == 0 ]]; then
      chmod -R u+w -- "${output_root}" || cleanup_failed=1
      rm -rf -- "${output_root}" || cleanup_failed=1
    fi
  fi
  case "${task_scratch}" in
    "${scratch_parent%/}/saic-ssft-all8-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    if [[ "$(stat -c '%d:%i' -- "${task_scratch}")" == "${task_scratch_device_inode}" ]]; then
      chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
      [[ "${cleanup_failed}" != 0 ]] || rm -rf -- "${task_scratch}" || cleanup_failed=1
    else
      cleanup_failed=1
    fi
  fi
  if [[ "${status}" == 0 && "${cleanup_failed}" == 0 && "${published}" == true ]]; then
    echo "[saic-ssft-all8] PASS exact81=true fps=25 exact40=true shift=5 arms=T0,I0 topology=dog:0-3+human:4-7 source_materializations=2 consumers=4 model_loads_per_group=2 frozen=true selection=false training=false optimizer=false semantic_success=false output=${output_root}"
  else
    [[ "${status}" != 0 ]] || status=2
    echo "[saic-ssft-all8] EXIT status=${status} published=${published} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${task_scratch}/source-tree" "${task_scratch}/groups" "${task_scratch}/logs"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
check_hash "${archive_copy}" "${source_archive_sha256}" staged_source_archive
chmod 0444 -- "${archive_copy}"

# Reject archives that are not a duplicate-free, link-free scoped
# methods/bernini_action_editing Git tree.  This is before extraction/output/GPU.
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/materialize_saic_source_clean_latent_v1.py",
    "methods/bernini_action_editing/saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/saic_native_source_state_field_v1.py",
    "methods/bernini_action_editing/differential_sampler.py",
    "methods/bernini_action_editing/guided_source_aligned_controller.py",
    "methods/bernini_action_editing/infer_guided_source_aligned_controller_oracle.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_source_aligned_controller_oracle.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
    "methods/bernini_action_editing/build_saic_reversible_source_set_v1.py",
    "methods/bernini_action_editing/saic_pure_t2v_event_bank_v1.py",
    "methods/bernini_action_editing/assets/saic_reversible_source_set_v1.json",
    "methods/bernini_action_editing/assets/saic_pure_t2v_event_bank_v1.json",
    "methods/bernini_action_editing/scripts/auh_infer_saic_source_state_flow_transport_all8_v1.sbatch",
    "methods/bernini_action_editing/tests/test_saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/tests/test_saic_native_source_state_field_v1.py",
    "methods/bernini_action_editing/tests/test_infer_saic_source_state_flow_transport_v1.py",
    "methods/bernini_action_editing/tests/test_materialize_saic_source_clean_latent_v1.py",
    "methods/bernini_action_editing/tests/test_auh_infer_saic_source_state_flow_transport_all8_v1.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        name = path.as_posix().rstrip("/")
        scoped = name in {"methods", "methods/bernini_action_editing"} or name.startswith(
            "methods/bernini_action_editing/"
        )
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or not scoped
            or name in seen
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
            or (not member.isfile() and not member.isdir())
        ):
            raise SystemExit("source archive contains an unsafe, duplicate, or out-of-scope member")
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks frozen SSFT runtime closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" \
  -C "${task_scratch}/source-tree" methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runner="${method_root}/infer_saic_source_state_flow_transport_v1.py"
materializer="${method_root}/materialize_saic_source_clean_latent_v1.py"
launcher="${method_root}/scripts/auh_infer_saic_source_state_flow_transport_all8_v1.sbatch"
source_manifest_archive="${method_root}/assets/saic_reversible_source_set_v1.json"
event_bank_archive="${method_root}/assets/saic_pure_t2v_event_bank_v1.json"
for path in "${runner}" "${materializer}" "${launcher}" "${source_manifest_archive}" "${event_bank_archive}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "authenticated SSFT closure differs"
done
[[ "$(hash_file "${BASH_SOURCE[0]}")" == "$(hash_file "${launcher}")" ]] || \
  fail "running launcher differs from authenticated archive"
check_hash "${source_manifest_archive}" "${pinned_source_manifest_raw_sha256}" source_manifest
check_hash "${event_bank_archive}" "${pinned_event_bank_raw_sha256}" event_bank
find "${task_scratch}/source-tree" -type f -exec chmod a-w -- {} +
find "${task_scratch}/source-tree" -type d -exec chmod a-w -- {} +

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

# External source/checkpoint identities are proven before their packages are
# placed on PYTHONPATH for the CPU runtime-closure tests.
PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" \
  "${source_manifest_archive}" "${event_bank_archive}" \
  "${pinned_source_manifest_content_sha256}" "${pinned_event_bank_content_sha256}" \
  "${dog_row}" "${dog_seed}" "${dog_source_sha256}" \
  "${human_row}" "${human_seed}" "${human_source_sha256}" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

import infer_saic_source_state_flow_transport_v1 as runner
import infer_source_kv_carrier_oracle as source_audit
import train_lora

(
    bernini_root, veomni_root, checkpoint, checkpoint_manifest,
    source_manifest_path, event_bank_path,
    expected_source_content_sha, expected_event_content_sha,
    dog_row, dog_seed, dog_source_sha,
    human_row, human_seed, human_source_sha,
) = sys.argv[1:]
dog_seed, human_seed = int(dog_seed), int(human_seed)

bernini, veomni, bernini_revision, veomni_revision = train_lora.validate_source_trees(
    bernini_root,
    veomni_root,
    expected_bernini_commit=train_lora.BERNINI_OFFICIAL_COMMIT,
    expected_veomni_commit=train_lora.VEOMNI_TESTED_COMMIT,
)
checkpoint_path, config = train_lora.validate_checkpoint(checkpoint)
if int(config.get("num_attention_heads", -1)) != 12 or 12 % 4:
    raise SystemExit("Bernini-R 1.3B/SP4 checkpoint closure differs")
source_audit.validate_checkpoint_content(checkpoint_path, Path(checkpoint_manifest))

manifest = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
events = json.loads(Path(event_bank_path).read_text(encoding="utf-8"))
canonical = lambda value: json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")
if hashlib.sha256(canonical(manifest)).hexdigest() != expected_source_content_sha:
    raise SystemExit("source manifest canonical content SHA-256 differs")
if hashlib.sha256(canonical(events)).hexdigest() != expected_event_content_sha:
    raise SystemExit("event bank canonical content SHA-256 differs")
if events.get("sampling_contract") != {
    "fps": 25,
    "guidance_mode": "t2v_apg",
    "latent_frames": 21,
    "model": "Bernini-R-1.3B-Diffusers",
    "native_arm": "t2v",
    "num_frames": 81,
    "num_inference_steps": 40,
    "same_row_seed_gaussian_across_branches": True,
    "target_initialization": "official_gen_wanx22_fresh_gaussian",
    "ulysses_size": 4,
}:
    raise SystemExit("sealed event sampling contract differs")

for row_id, seed, expected_source_sha in (
    (dog_row, dog_seed, dog_source_sha),
    (human_row, human_seed, human_source_sha),
):
    cell = runner.resolve_sealed_forward_cell(
        manifest, events, row_id=row_id, branch="forward", rollout_seed=seed
    )
    if cell["source_video_sha256"] != expected_source_sha:
        raise SystemExit("fixed source-video content hash differs")
    source = Path(cell["source_video"])
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        raise SystemExit("fixed source video is absent/non-plain")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_source_sha:
        raise SystemExit("fixed source-video bytes changed")

if (bernini_revision, veomni_revision) != (
    runner.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    runner.legacy.trainer.VEOMNI_TESTED_COMMIT,
):
    raise SystemExit("pinned source revision closure differs")
if runner.FRAME_COUNT != 81 or runner.FPS != 25 or runner.NUM_INFERENCE_STEPS != 40 or runner.FLOW_SHIFT != 5.0:
    raise SystemExit("runner exact81/25fps/exact40/shift5 constants differ")
if tuple(runner.ARM_NAMES)[:3] != ("T0", "T1", "I0"):
    raise SystemExit("runner arm registry differs")
PY

# CPU contract closure must pass before output creation and before the first
# torchrun can initialize a GPU/process group.  The just-validated pinned
# Bernini and VeOmni trees are explicit imports; no installed fork may win.
for test_file in \
  test_saic_source_state_flow_transport_v1.py \
  test_saic_native_source_state_field_v1.py \
  test_materialize_saic_source_clean_latent_v1.py \
  test_infer_saic_source_state_flow_transport_v1.py \
  test_auh_infer_saic_source_state_flow_transport_all8_v1.py; do
  PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
    "${python_bin}" -B "${method_root}/tests/${test_file}"
done

# Publish only into a fresh launcher-owned root.  Each runner itself uses
# create-only hard-link publication for its MP4/receipt pair.  On any failure
# this launcher removes only this fresh owned root; no resume path exists.
mkdir -- "${output_root}"
owned_output=true
output_device_inode="$(stat -c '%d:%i' -- "${output_root}")"
[[ "${output_device_inode}" =~ ^[0-9]+:[0-9]+$ ]] || \
  fail "fresh output root device/inode identity differs"
readonly output_device_inode
mkdir -- "${output_root}/sealed-inputs" "${output_root}/groups" "${output_root}/groups/dog" \
  "${output_root}/groups/human" "${output_root}/logs"
durable_archive="${output_root}/method-source.tar"
sealed_source_manifest="${output_root}/sealed-inputs/saic_reversible_source_set_v1.json"
sealed_event_bank="${output_root}/sealed-inputs/saic_pure_t2v_event_bank_v1.json"
cp -- "${archive_copy}" "${durable_archive}"
cp -- "${source_manifest_archive}" "${sealed_source_manifest}"
cp -- "${event_bank_archive}" "${sealed_event_bank}"
chmod 0444 -- "${durable_archive}" "${sealed_source_manifest}" "${sealed_event_bank}"
check_hash "${durable_archive}" "${source_archive_sha256}" durable_source_archive
check_hash "${sealed_source_manifest}" "${pinned_source_manifest_raw_sha256}" durable_source_manifest
check_hash "${sealed_event_bank}" "${pinned_event_bank_raw_sha256}" durable_event_bank

dog_source_clean="${output_root}/groups/dog/source.clean-latent.safetensors"
human_source_clean="${output_root}/groups/human/source.clean-latent.safetensors"
dog_source_clean_receipt="${dog_source_clean}.receipt.json"
human_source_clean_receipt="${human_source_clean}.receipt.json"

# Materialize each selected source coordinate exactly once before either arm.
# These are two concurrent, source-only WORLD1 jobs: physical GPU 0 owns dog
# and physical GPU 4 owns human.  T0 and I0 later consume the same immutable
# artifact/receipt pair for their group; neither runner may encode it again.
materialize_source() (
  set -Eeuo pipefail
  local group_id="$1" row_id="$2" source_sha="$3" visible="$4" output="$5"
  local materializer_scratch="${task_scratch}/materializers/${group_id}"
  mkdir -p -- "${materializer_scratch}/miopen-user" \
    "${materializer_scratch}/miopen-custom" \
    "${materializer_scratch}/torch-extensions" \
    "${materializer_scratch}/triton" "${materializer_scratch}/xdg" \
    "${materializer_scratch}/pycache" "${materializer_scratch}/tmp"
  [[ ! -e "${output}" && ! -L "${output}" ]] || \
    fail "refusing source-clean-latent cache reuse"
  [[ ! -e "${output}.receipt.json" && ! -L "${output}.receipt.json" ]] || \
    fail "refusing source-clean-latent receipt reuse"
  echo "[saic-ssft-all8:${group_id}] MATERIALIZE source-only world=1 row=${row_id} visible=${visible}"
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${visible}" \
    MIOPEN_USER_DB_PATH="${materializer_scratch}/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="${materializer_scratch}/miopen-custom" \
    TORCH_EXTENSIONS_DIR="${materializer_scratch}/torch-extensions" \
    TRITON_CACHE_DIR="${materializer_scratch}/triton" \
    XDG_CACHE_HOME="${materializer_scratch}/xdg" \
    PYTHONPYCACHEPREFIX="${materializer_scratch}/pycache" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    TMPDIR="${materializer_scratch}/tmp" \
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    PYTHONPATH="${method_root}:${bernini_root}:${veomni_root}" \
    "${python_bin}" -B "${materializer}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-manifest "${sealed_source_manifest}" \
      --expected-source-manifest-sha256 "${pinned_source_manifest_raw_sha256}" \
      --row-id "${row_id}" \
      --expected-source-video-sha256 "${source_sha}" \
      --expected-bernini-commit "${pinned_bernini_commit}" \
      --expected-veomni-commit "${pinned_veomni_commit}" \
      --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}" \
      --method-source-revision "${source_revision}" \
      --method-source-archive "${archive_copy}" \
      --durable-method-source-archive "${durable_archive}" \
      --method-source-archive-sha256 "${source_archive_sha256}" \
      --device cuda:0 \
      --output "${output}"
)

materialize_source dog "${dog_row}" "${dog_source_sha256}" 0 "${dog_source_clean}" \
  >"${output_root}/logs/materialize.dog.log" 2>&1 &
dog_pid=$!
materialize_source human "${human_row}" "${human_source_sha256}" 4 "${human_source_clean}" \
  >"${output_root}/logs/materialize.human.log" 2>&1 &
human_pid=$!
dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
dog_pid=""
human_pid=""
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 240 -- "${output_root}/logs/materialize.dog.log" >&2 || true
  tail -n 240 -- "${output_root}/logs/materialize.human.log" >&2 || true
  fail "source-only WORLD1 materialization failed dog=${dog_status} human=${human_status}"
fi

# Reopen both immutable WORLD1 products before any WORLD4 consumer starts and
# derive the exact five runner cache arguments from observed bytes.  The same
# variables are then reused verbatim by both arms in one source group.
mapfile -t materialized_bindings < <(
  "${python_bin}" -B - \
    "${dog_source_clean}" "${dog_source_clean_receipt}" dog "${dog_row}" "${dog_source_sha256}" \
    "${human_source_clean}" "${human_source_clean_receipt}" human "${human_row}" "${human_source_sha256}" \
    "${checkpoint}" "${pinned_checkpoint_tree_sha256}" "${pinned_checkpoint_manifest_sha256}" \
    "${pinned_bernini_commit}" "${pinned_veomni_commit}" \
    "${pinned_source_manifest_raw_sha256}" "${pinned_source_manifest_content_sha256}" \
    "${source_revision}" "${source_archive_sha256}" <<'PY'
from pathlib import Path
import hashlib
import json
import math
import stat
import sys

import torch
from safetensors import safe_open

(
    dog_artifact, dog_receipt, dog_group, dog_row, dog_source_sha,
    human_artifact, human_receipt, human_group, human_row, human_source_sha,
    checkpoint, checkpoint_tree_sha, checkpoint_manifest_sha,
    bernini_commit, veomni_commit,
    source_manifest_raw_sha, source_manifest_content_sha,
    source_revision, archive_sha,
) = sys.argv[1:]

def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def tensor_raw_sha(value):
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()

top_keys = {
    "schema_version", "method", "artifact", "sealed_inputs", "preprocessing",
    "model_closure", "encoding", "runtime", "authority", "receipt_digest",
}
artifact_keys = {
    "schema_version", "path", "file_sha256", "size_bytes", "mode", "tensor_key",
    "tensor_raw_sha256", "shape", "dtype", "metadata",
}
sealed_keys = {
    "accepted_roles", "forbidden_roles", "source_manifest_path",
    "source_manifest_raw_sha256", "source_manifest_content_sha256",
    "source_manifest_schema_version", "source_manifest_dataset_id",
    "source_manifest_bound_files_verified", "row_id", "iid", "analysis_split",
    "actor_family", "source_video_path", "source_video_sha256",
    "source_video_rehashed_after_encode", "source_manifest_terminal_events_verified",
    "optimizer_authorized",
}
preprocessing_keys = {
    "decoded_from_private_byte_snapshot", "frame_count", "fps", "reported_fps",
    "source_input_hw", "source_derived_bucket_hw", "max_pixels", "stride",
    "temporal_policy", "spatial_policy", "resize", "external_shared_i0",
    "source_pixels_shape", "source_pixels_dtype", "source_pixels_raw_sha256",
}
model_keys = {
    "checkpoint_path", "checkpoint_tree_sha256", "checkpoint_content_manifest_audit",
    "bernini_revision", "veomni_revision", "bernini_inference_files",
    "bernini_inference_files_index_sha256", "method_source_revision",
    "method_source_archive_sha256", "runtime_source_index_sha256", "method_provenance",
}
encoding_keys = {
    "encoder_symbol", "callable_module", "callable_name", "callable_qualname",
    "callable_signature", "encoded_in_runner", "full_source_vae_encode_count",
    "total_vae_encode_count", "posterior_statistic", "sampling",
    "torch_inference_mode", "source_pixels_mutated", "source_pixels_before_sha256",
    "source_pixels_after_sha256", "vae_dtype", "vae_eval", "vae_requires_grad",
    "latent_frame_count", "finite",
}
runtime_keys = {
    "device_requested", "world_size", "distributed_initialized", "python_version",
    "torch_version", "hip_version", "diffusers_version", "safetensors_version",
}
authority_keys = {
    "quality_claim_authorized", "semantic_action_success_authorized",
    "ground_truth_authorized", "training_target_authorized", "selection_authorized",
    "optimizer_step_authorized", "checkpoint_or_lora_artifact",
    "production_claim_authorized",
}
metadata_expected = {
    "schema_version": "bernini-saic-source-clean-latent-artifact-v1",
    "coordinate": "bernini_normalized_clean_vae_latent",
    "frame_contract": "exact81_latent21",
    "artifact_role": "saic_common_source_outer_clean_state",
    "source": "sealed_source_video_private_snapshot_wan_vae_mode",
    "posterior": "mode",
    "sampling": "false",
    "authority": "false",
}
accepted_roles_expected = [
    "source_manifest",
    "selected_source_video",
    "checkpoint_and_source_code_provenance",
]
forbidden_roles_expected = [
    "target_video", "mask", "sam_track", "pose", "flow", "motion_donor",
    "pure_t2v_rgb_as_student_target", "pure_t2v_latent_as_student_target",
    "pure_t2v_noise_as_student_input", "natural_language_instruction",
    "source_caption", "target_caption", "event_bank", "branch", "rollout_seed",
    "reference_frame", "shared_i0", "adapter", "lora",
]

def validate_one(artifact_text, receipt_text, group, row, source_sha):
    artifact_path = Path(artifact_text)
    receipt_path = Path(receipt_text)
    for path in (artifact_path, receipt_path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise SystemExit("source-clean materialization is not a plain file")
        if stat.S_IMODE(info.st_mode) != 0o444:
            raise SystemExit("source-clean materialization mode differs")
    raw_receipt = receipt_path.read_bytes()
    receipt = json.loads(raw_receipt.decode("utf-8"))
    if raw_receipt != canonical(receipt) + b"\n" or set(receipt) != top_keys:
        raise SystemExit("source-clean receipt canonical/top-level closure differs")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if object_sha(unsigned) != declared:
        raise SystemExit("source-clean receipt digest differs")
    if (
        receipt["schema_version"] != "bernini-saic-source-clean-latent-receipt-v1"
        or receipt["method"] != "frozen-bernini-source-clean-latent-materializer"
    ):
        raise SystemExit("source-clean receipt schema/method differs")
    artifact = receipt["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != artifact_keys:
        raise SystemExit("source-clean artifact receipt closure differs")
    artifact_file_sha = file_sha(artifact_path)
    receipt_file_sha = file_sha(receipt_path)
    try:
        with safe_open(str(artifact_path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["source_clean_latent"]:
                raise SystemExit("source-clean tensor key differs")
            tensor = opened.get_tensor("source_clean_latent").contiguous()
            metadata = dict(opened.metadata() or {})
    except (OSError, RuntimeError) as error:
        raise SystemExit("cannot reopen source-clean safetensors") from error
    tensor_sha = tensor_raw_sha(tensor)
    if (
        metadata != metadata_expected
        or tensor.dtype != torch.float32
        or tensor.ndim != 5
        or list(tensor.shape[:3]) != [1, 16, 21]
        or not bool(torch.isfinite(tensor).all().item())
        or artifact.get("schema_version") != metadata_expected["schema_version"]
        or artifact.get("path") != str(artifact_path)
        or artifact.get("file_sha256") != artifact_file_sha
        or artifact.get("size_bytes") != artifact_path.stat().st_size
        or artifact.get("mode") != "0444"
        or artifact.get("tensor_key") != "source_clean_latent"
        or artifact.get("tensor_raw_sha256") != tensor_sha
        or artifact.get("shape") != list(tensor.shape)
        or artifact.get("dtype") != "torch.float32"
        or artifact.get("metadata") != metadata_expected
    ):
        raise SystemExit("source-clean artifact bytes/tensor identity differs")
    sealed = receipt["sealed_inputs"]
    if not isinstance(sealed, dict) or set(sealed) != sealed_keys:
        raise SystemExit("source-clean sealed-input closure differs")
    if (
        sealed.get("accepted_roles") != accepted_roles_expected
        or sealed.get("forbidden_roles") != forbidden_roles_expected
        or sealed.get("source_manifest_path")
        != str(artifact_path.parents[2] / "sealed-inputs" / "saic_reversible_source_set_v1.json")
        or sealed.get("source_manifest_raw_sha256") != source_manifest_raw_sha
        or sealed.get("source_manifest_content_sha256") != source_manifest_content_sha
        or sealed.get("source_manifest_schema_version")
        != "bernini-saic-reversible-source-set-v1"
        or sealed.get("source_manifest_dataset_id")
        != "saic-reversible-source-set-exact81-v1"
        or sealed.get("source_manifest_bound_files_verified") is not False
        or sealed.get("row_id") != row
        or sealed.get("iid") != row.rsplit("-", 1)[-1]
        or sealed.get("analysis_split") != "fit"
        or sealed.get("actor_family") != group
        or sealed.get("source_video_sha256") != source_sha
        or sealed.get("source_video_rehashed_after_encode") is not True
        or sealed.get("source_manifest_terminal_events_verified") is not False
        or sealed.get("optimizer_authorized") is not False
    ):
        raise SystemExit("source-clean selected source binding differs")
    source_path = Path(sealed.get("source_video_path", ""))
    if (
        not source_path.is_absolute()
        or not source_path.is_file()
        or source_path.is_symlink()
        or file_sha(source_path) != source_sha
    ):
        raise SystemExit("source-clean selected source bytes differ")
    preprocessing = receipt["preprocessing"]
    if not isinstance(preprocessing, dict) or set(preprocessing) != preprocessing_keys:
        raise SystemExit("source-clean preprocessing closure differs")
    if (
        preprocessing.get("decoded_from_private_byte_snapshot") is not True
        or preprocessing.get("frame_count") != 81
        or preprocessing.get("fps") != 25
        or not math.isclose(float(preprocessing.get("reported_fps")), 25.0, abs_tol=1e-3)
        or preprocessing.get("external_shared_i0") is not False
        or preprocessing.get("source_pixels_dtype") != "torch.float32"
        or preprocessing.get("source_pixels_shape", [])[:3] != [1, 3, 81]
        or preprocessing.get("source_pixels_shape", [])[3:]
        != preprocessing.get("source_derived_bucket_hw")
        or preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
    ):
        raise SystemExit("source-clean exact81/25 preprocessing differs")
    model = receipt["model_closure"]
    if not isinstance(model, dict) or set(model) != model_keys:
        raise SystemExit("source-clean model closure keys differ")
    checkpoint_audit = model.get("checkpoint_content_manifest_audit", {})
    provenance = model.get("method_provenance", {})
    if (
        model.get("checkpoint_path") != checkpoint
        or model.get("checkpoint_tree_sha256") != checkpoint_tree_sha
        or checkpoint_audit.get("manifest_sha256_computed") != checkpoint_manifest_sha
        or checkpoint_audit.get("manifest_sha256_expected") != checkpoint_manifest_sha
        or checkpoint_audit.get("verified_file_count") != 23
        or checkpoint_audit.get("every_file_sha256_verified") is not True
        or model.get("bernini_revision") != bernini_commit
        or model.get("veomni_revision") != veomni_commit
        or model.get("method_source_revision") != source_revision
        or model.get("method_source_archive_sha256") != archive_sha
        or provenance.get("revision") != source_revision
        or provenance.get("archive_sha256") != archive_sha
        or model.get("runtime_source_index_sha256") != provenance.get("runtime_source_index_sha256")
    ):
        raise SystemExit("source-clean checkpoint/method closure differs")
    encoding = receipt["encoding"]
    if not isinstance(encoding, dict) or set(encoding) != encoding_keys:
        raise SystemExit("source-clean encoding closure keys differ")
    if (
        encoding.get("encoder_symbol") != "bernini.pipeline._vae_encode"
        or encoding.get("callable_module") != "bernini.pipeline"
        or encoding.get("callable_name") != "_vae_encode"
        or encoding.get("callable_qualname") != "_vae_encode"
        or encoding.get("callable_signature")
        != "(vae, x: torch.Tensor) -> torch.Tensor"
        or encoding.get("encoded_in_runner") is not False
        or encoding.get("full_source_vae_encode_count") != 1
        or encoding.get("total_vae_encode_count") != 1
        or encoding.get("posterior_statistic") != "latent_dist.mode"
        or encoding.get("sampling") is not False
        or encoding.get("torch_inference_mode") is not True
        or encoding.get("source_pixels_mutated") is not False
        or encoding.get("source_pixels_before_sha256") != encoding.get("source_pixels_after_sha256")
        or encoding.get("vae_dtype") != "torch.float32"
        or encoding.get("vae_eval") is not True
        or encoding.get("vae_requires_grad") is not False
        or encoding.get("latent_frame_count") != 21
        or encoding.get("finite") is not True
    ):
        raise SystemExit("source-clean single mode encode contract differs")
    runtime = receipt["runtime"]
    authority = receipt["authority"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or runtime.get("device_requested") != "cuda:0"
        or runtime.get("world_size") != 1
        or runtime.get("distributed_initialized") is not False
        or not isinstance(authority, dict)
        or set(authority) != authority_keys
        or any(value is not False for value in authority.values())
    ):
        raise SystemExit("source-clean WORLD1/authority closure differs")
    print("\t".join((group, artifact_file_sha, receipt_file_sha, declared, tensor_sha)))

validate_one(dog_artifact, dog_receipt, dog_group, dog_row, dog_source_sha)
validate_one(human_artifact, human_receipt, human_group, human_row, human_source_sha)
PY
)
[[ "${#materialized_bindings[@]}" == 2 ]] || \
  fail "exactly two source materialization bindings are required"
IFS=$'\t' read -r dog_binding_group dog_source_clean_sha256 \
  dog_source_clean_receipt_sha256 dog_source_clean_receipt_digest \
  dog_source_clean_tensor_sha256 <<<"${materialized_bindings[0]}"
IFS=$'\t' read -r human_binding_group human_source_clean_sha256 \
  human_source_clean_receipt_sha256 human_source_clean_receipt_digest \
  human_source_clean_tensor_sha256 <<<"${materialized_bindings[1]}"
[[ "${dog_binding_group}" == dog && "${human_binding_group}" == human ]] || \
  fail "source materialization binding order differs"
for name in dog_source_clean_sha256 dog_source_clean_receipt_sha256 \
  dog_source_clean_receipt_digest dog_source_clean_tensor_sha256 \
  human_source_clean_sha256 human_source_clean_receipt_sha256 \
  human_source_clean_receipt_digest human_source_clean_tensor_sha256; do
  require_sha256 "${!name}" "${name}"
  readonly "${name}=${!name}"
done

job_mod=$((SLURM_JOB_ID % 20000))
dog_port=$((20000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_cell() (
  set -Eeuo pipefail
  local group_id="$1" row_id="$2" seed="$3" arm="$4" visible="$5" port="$6" output="$7"
  local source_clean="$8" source_clean_receipt="$9"
  local source_clean_sha256="${10}" source_clean_receipt_sha256="${11}"
  local source_clean_tensor_sha256="${12}"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" \
    "${group_scratch}/torch-extensions" "${group_scratch}/triton" \
    "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  [[ -z "$(find "${group_scratch}/pycache" -mindepth 1 -print -quit)" ]] || \
    fail "private bytecode prefix must remain empty"
  [[ ! -e "${output}" && ! -L "${output}" ]] || fail "refusing output reuse"
  [[ ! -e "${output}.receipt.json" && ! -L "${output}.receipt.json" ]] || \
    fail "refusing receipt reuse"
  [[ ! -e "${output}.normalized-clean-latent.safetensors" && ! -L "${output}.normalized-clean-latent.safetensors" ]] || \
    fail "refusing normalized clean latent reuse"
  [[ -f "${source_clean}" && ! -L "${source_clean}" ]] || \
    fail "source-clean materialization is absent/non-plain"
  [[ -f "${source_clean_receipt}" && ! -L "${source_clean_receipt}" ]] || \
    fail "source-clean materialization receipt is absent/non-plain"
  [[ "$(stat -c '%a' -- "${source_clean}")" == 444 ]] || \
    fail "source-clean materialization mode differs before consumer"
  [[ "$(stat -c '%a' -- "${source_clean_receipt}")" == 444 ]] || \
    fail "source-clean receipt mode differs before consumer"
  check_hash "${source_clean}" "${source_clean_sha256}" source_clean_before_consumer
  check_hash "${source_clean_receipt}" "${source_clean_receipt_sha256}" source_clean_receipt_before_consumer
  echo "[saic-ssft-all8:${group_id}] START arm=${arm} row=${row_id} seed=${seed} visible=${visible}"
  exec env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    ROCR_VISIBLE_DEVICES="${visible}" \
    MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user" \
    MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom" \
    TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions" \
    TRITON_CACHE_DIR="${group_scratch}/triton" \
    XDG_CACHE_HOME="${group_scratch}/xdg" \
    PYTHONPYCACHEPREFIX="${group_scratch}/pycache" \
    TMPDIR="${group_scratch}/tmp" \
    TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-${arm}.json" \
    "${python_bin}" -B -m torch.distributed.run \
      --nproc_per_node=4 \
      --master_addr=127.0.0.1 \
      --master_port="${port}" \
      "${runner}" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${checkpoint_manifest}" \
      --source-manifest "${sealed_source_manifest}" \
      --expected-source-manifest-sha256 "${pinned_source_manifest_raw_sha256}" \
      --event-bank "${sealed_event_bank}" \
      --expected-event-bank-sha256 "${pinned_event_bank_raw_sha256}" \
      --source-clean-latent "${source_clean}" \
      --source-clean-latent-receipt "${source_clean_receipt}" \
      --expected-source-clean-latent-sha256 "${source_clean_sha256}" \
      --expected-source-clean-latent-receipt-sha256 "${source_clean_receipt_sha256}" \
      --expected-source-clean-tensor-raw-sha256 "${source_clean_tensor_sha256}" \
      --row-id "${row_id}" \
      --branch forward \
      --rollout-seed "${seed}" \
      --arm "${arm}" \
      --output "${output}" \
      --expected-bernini-commit "${pinned_bernini_commit}" \
      --expected-veomni-commit "${pinned_veomni_commit}" \
      --expected-checkpoint-tree-sha256 "${pinned_checkpoint_tree_sha256}" \
      --method-source-revision "${source_revision}" \
      --method-source-archive "${archive_copy}" \
      --durable-method-source-archive "${durable_archive}" \
      --method-source-archive-sha256 "${source_archive_sha256}"
)

run_wave() {
  local arm="$1"
  local dog_output="${output_root}/groups/dog/${arm}.mp4"
  local human_output="${output_root}/groups/human/${arm}.mp4"
  run_cell dog "${dog_row}" "${dog_seed}" "${arm}" "0,1,2,3" "${dog_port}" "${dog_output}" \
    "${dog_source_clean}" "${dog_source_clean_receipt}" \
    "${dog_source_clean_sha256}" "${dog_source_clean_receipt_sha256}" \
    "${dog_source_clean_tensor_sha256}" \
    >"${output_root}/logs/${arm}.dog.log" 2>&1 &
  dog_pid=$!
  run_cell human "${human_row}" "${human_seed}" "${arm}" "4,5,6,7" "${human_port}" "${human_output}" \
    "${human_source_clean}" "${human_source_clean_receipt}" \
    "${human_source_clean_sha256}" "${human_source_clean_receipt_sha256}" \
    "${human_source_clean_tensor_sha256}" \
    >"${output_root}/logs/${arm}.human.log" 2>&1 &
  human_pid=$!
  local dog_status=0 human_status=0
  wait "${dog_pid}" || dog_status=$?
  wait "${human_pid}" || human_status=$?
  dog_pid=""
  human_pid=""
  if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
    tail -n 240 -- "${output_root}/logs/${arm}.dog.log" >&2 || true
    tail -n 240 -- "${output_root}/logs/${arm}.human.log" >&2 || true
    fail "SP4 wave ${arm} failed dog=${dog_status} human=${human_status}"
  fi
}

echo "[saic-ssft-all8] topology=2xWORLD4-SP4 cards=0,1,2,3+4,5,6,7 arms=T0,I0 runner_multi_arm=false model_loads_per_group=2"
for arm in "${pilot_arms[@]}"; do
  run_wave "${arm}"
done

# Reopen every canonical runner receipt and media file, verify all four latent
# identities, create two child group receipts, reopen those, and only then
# atomically create the all8 parent receipt.  None of these receipts selects an
# arm or grants quality, semantic, training, optimizer, or checkpoint authority.
"${python_bin}" -B - \
  "${output_root}" "${source_revision}" "${source_archive_sha256}" \
  "${pinned_bernini_commit}" "${pinned_veomni_commit}" \
  "${checkpoint}" "${pinned_checkpoint_tree_sha256}" "${pinned_checkpoint_manifest_sha256}" \
  "${pinned_source_manifest_raw_sha256}" "${pinned_source_manifest_content_sha256}" \
  "${pinned_event_bank_raw_sha256}" "${pinned_event_bank_content_sha256}" \
  "${dog_row}" "${dog_seed}" "${dog_source_sha256}" \
  "${human_row}" "${human_seed}" "${human_source_sha256}" \
  "${SLURM_JOB_ID}" <<'PY'
from pathlib import Path
import hashlib
import json
import math
import os
import stat
import sys

import torch
from safetensors import safe_open

(
    output_root, source_revision, archive_sha,
    bernini_commit, veomni_commit, checkpoint_path, checkpoint_tree_sha, checkpoint_manifest_sha,
    source_manifest_raw_sha, source_manifest_content_sha,
    event_bank_raw_sha, event_bank_content_sha,
    dog_row, dog_seed, dog_source_sha,
    human_row, human_seed, human_source_sha,
    job_id,
) = sys.argv[1:]
root = Path(output_root)
dog_seed, human_seed = int(dog_seed), int(human_seed)
sha_pattern = __import__("re").compile(r"[0-9a-f]{64}")

def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def tensor_raw_sha(value):
    if (
        not isinstance(value, torch.Tensor)
        or value.numel() <= 0
        or value.requires_grad
        or not value.is_floating_point()
    ):
        raise SystemExit("normalized clean latent tensor closure differs")
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()

def plain_file(path, mode=None):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SystemExit(f"non-plain artifact: {path}")
    actual_mode = stat.S_IMODE(info.st_mode)
    if mode is not None and actual_mode != mode:
        raise SystemExit(f"artifact mode differs: {path}: {actual_mode:o}")
    return info

def read_canonical_receipt(path):
    plain_file(path, 0o444)
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value) + b"\n":
        raise SystemExit(f"receipt is not canonical JSON: {path}")
    declared = value.get("receipt_digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha(unsigned) != declared:
        raise SystemExit(f"receipt digest differs: {path}")
    return value, file_sha(path)

expected_top = {
    "schema_version", "method", "arm", "sealed_inputs", "preprocessing",
    "prompt_contract", "model", "transport", "schedule", "noise",
    "distributed_execution", "output", "runtime_versions", "authority",
    "receipt_digest",
}
false_authority = (
    "quality_authority", "evaluator_authority", "semantic_action_success",
    "identity_preservation_success", "training_authority", "optimizer_authority",
    "training_update_allowed", "optimizer_step_allowed",
    "checkpoint_or_lora_artifact", "production_claim_authorized",
)

materializer_top = {
    "schema_version", "method", "artifact", "sealed_inputs", "preprocessing",
    "model_closure", "encoding", "runtime", "authority", "receipt_digest",
}
materializer_authority = {
    "quality_claim_authorized", "semantic_action_success_authorized",
    "ground_truth_authorized", "training_target_authorized", "selection_authorized",
    "optimizer_step_authorized", "checkpoint_or_lora_artifact",
    "production_claim_authorized",
}
materializer_metadata = {
    "schema_version": "bernini-saic-source-clean-latent-artifact-v1",
    "coordinate": "bernini_normalized_clean_vae_latent",
    "frame_contract": "exact81_latent21",
    "artifact_role": "saic_common_source_outer_clean_state",
    "source": "sealed_source_video_private_snapshot_wan_vae_mode",
    "posterior": "mode",
    "sampling": "false",
    "authority": "false",
}
materializer_accepted_roles = [
    "source_manifest",
    "selected_source_video",
    "checkpoint_and_source_code_provenance",
]
materializer_forbidden_roles = [
    "target_video", "mask", "sam_track", "pose", "flow", "motion_donor",
    "pure_t2v_rgb_as_student_target", "pure_t2v_latent_as_student_target",
    "pure_t2v_noise_as_student_input", "natural_language_instruction",
    "source_caption", "target_caption", "event_bank", "branch", "rollout_seed",
    "reference_frame", "shared_i0", "adapter", "lora",
]

def validate_materialization(group, row, source_sha):
    artifact_path = root / "groups" / group / "source.clean-latent.safetensors"
    receipt_path = artifact_path.with_name(artifact_path.name + ".receipt.json")
    artifact_info = plain_file(artifact_path, 0o444)
    receipt, receipt_file_sha = read_canonical_receipt(receipt_path)
    if set(receipt) != materializer_top:
        raise SystemExit("source materializer receipt top-level closure differs")
    if (
        receipt.get("schema_version") != "bernini-saic-source-clean-latent-receipt-v1"
        or receipt.get("method") != "frozen-bernini-source-clean-latent-materializer"
    ):
        raise SystemExit("source materializer schema/method differs")
    try:
        with safe_open(str(artifact_path), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["source_clean_latent"]:
                raise SystemExit("source materializer tensor key differs")
            tensor = opened.get_tensor("source_clean_latent").contiguous()
            metadata = dict(opened.metadata() or {})
    except (OSError, RuntimeError) as error:
        raise SystemExit("cannot reopen source materialization") from error
    artifact_file_sha = file_sha(artifact_path)
    tensor_sha = tensor_raw_sha(tensor)
    artifact = receipt.get("artifact", {})
    if (
        set(artifact) != {
            "schema_version", "path", "file_sha256", "size_bytes", "mode",
            "tensor_key", "tensor_raw_sha256", "shape", "dtype", "metadata",
        }
        or metadata != materializer_metadata
        or tensor.dtype != torch.float32
        or tensor.ndim != 5
        or list(tensor.shape[:3]) != [1, 16, 21]
        or not bool(torch.isfinite(tensor).all().item())
        or artifact.get("schema_version") != materializer_metadata["schema_version"]
        or artifact.get("path") != str(artifact_path)
        or artifact.get("file_sha256") != artifact_file_sha
        or artifact.get("size_bytes") != artifact_info.st_size
        or artifact.get("mode") != "0444"
        or artifact.get("tensor_key") != "source_clean_latent"
        or artifact.get("tensor_raw_sha256") != tensor_sha
        or artifact.get("shape") != list(tensor.shape)
        or artifact.get("dtype") != "torch.float32"
        or artifact.get("metadata") != materializer_metadata
    ):
        raise SystemExit("source materializer artifact/tensor closure differs")
    sealed = receipt.get("sealed_inputs", {})
    if (
        set(sealed) != {
            "accepted_roles", "forbidden_roles", "source_manifest_path",
            "source_manifest_raw_sha256", "source_manifest_content_sha256",
            "source_manifest_schema_version", "source_manifest_dataset_id",
            "source_manifest_bound_files_verified", "row_id", "iid",
            "analysis_split", "actor_family", "source_video_path",
            "source_video_sha256", "source_video_rehashed_after_encode",
            "source_manifest_terminal_events_verified", "optimizer_authorized",
        }
        or sealed.get("accepted_roles") != materializer_accepted_roles
        or sealed.get("forbidden_roles") != materializer_forbidden_roles
        or sealed.get("source_manifest_path")
        != str(root / "sealed-inputs" / "saic_reversible_source_set_v1.json")
        or sealed.get("source_manifest_raw_sha256") != source_manifest_raw_sha
        or sealed.get("source_manifest_content_sha256") != source_manifest_content_sha
        or sealed.get("source_manifest_schema_version")
        != "bernini-saic-reversible-source-set-v1"
        or sealed.get("source_manifest_dataset_id")
        != "saic-reversible-source-set-exact81-v1"
        or sealed.get("source_manifest_bound_files_verified") is not False
        or sealed.get("row_id") != row
        or sealed.get("source_video_sha256") != source_sha
        or sealed.get("source_video_rehashed_after_encode") is not True
        or sealed.get("source_manifest_terminal_events_verified") is not False
        or sealed.get("optimizer_authorized") is not False
    ):
        raise SystemExit("source materializer selected-input closure differs")
    source_path = Path(sealed.get("source_video_path", ""))
    if (
        not source_path.is_absolute()
        or not source_path.is_file()
        or source_path.is_symlink()
        or file_sha(source_path) != source_sha
    ):
        raise SystemExit("materializer-bound source bytes changed before parent publication")
    preprocessing = receipt.get("preprocessing", {})
    if (
        set(preprocessing) != {
            "decoded_from_private_byte_snapshot", "frame_count", "fps", "reported_fps",
            "source_input_hw", "source_derived_bucket_hw", "max_pixels", "stride",
            "temporal_policy", "spatial_policy", "resize", "external_shared_i0",
            "source_pixels_shape", "source_pixels_dtype", "source_pixels_raw_sha256",
        }
        or preprocessing.get("decoded_from_private_byte_snapshot") is not True
        or preprocessing.get("frame_count") != 81
        or preprocessing.get("fps") != 25
        or not math.isclose(float(preprocessing.get("reported_fps")), 25.0, abs_tol=1e-3)
        or preprocessing.get("external_shared_i0") is not False
        or preprocessing.get("source_pixels_dtype") != "torch.float32"
        or preprocessing.get("source_pixels_shape", [])[:3] != [1, 3, 81]
        or preprocessing.get("source_pixels_shape", [])[3:]
        != preprocessing.get("source_derived_bucket_hw")
        or preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
    ):
        raise SystemExit("source materializer exact81/25 preprocessing differs")
    model = receipt.get("model_closure", {})
    provenance = model.get("method_provenance", {})
    checkpoint_audit = model.get("checkpoint_content_manifest_audit", {})
    if (
        set(model) != {
            "checkpoint_path", "checkpoint_tree_sha256",
            "checkpoint_content_manifest_audit", "bernini_revision", "veomni_revision",
            "bernini_inference_files", "bernini_inference_files_index_sha256",
            "method_source_revision", "method_source_archive_sha256",
            "runtime_source_index_sha256", "method_provenance",
        }
        or model.get("checkpoint_path") != checkpoint_path
        or model.get("checkpoint_tree_sha256") != checkpoint_tree_sha
        or checkpoint_audit.get("manifest_sha256_computed") != checkpoint_manifest_sha
        or checkpoint_audit.get("manifest_sha256_expected") != checkpoint_manifest_sha
        or checkpoint_audit.get("verified_file_count") != 23
        or checkpoint_audit.get("every_file_sha256_verified") is not True
        or model.get("bernini_revision") != bernini_commit
        or model.get("veomni_revision") != veomni_commit
        or model.get("method_source_revision") != source_revision
        or model.get("method_source_archive_sha256") != archive_sha
        or provenance.get("revision") != source_revision
        or provenance.get("archive_sha256") != archive_sha
        or model.get("runtime_source_index_sha256") != provenance.get("runtime_source_index_sha256")
    ):
        raise SystemExit("source materializer model/provenance closure differs")
    encoding = receipt.get("encoding", {})
    if (
        set(encoding) != {
            "encoder_symbol", "callable_module", "callable_name", "callable_qualname",
            "callable_signature", "encoded_in_runner",
            "full_source_vae_encode_count", "total_vae_encode_count",
            "posterior_statistic", "sampling",
            "torch_inference_mode", "source_pixels_mutated",
            "source_pixels_before_sha256", "source_pixels_after_sha256", "vae_dtype",
            "vae_eval", "vae_requires_grad", "latent_frame_count", "finite",
        }
        or encoding.get("encoder_symbol") != "bernini.pipeline._vae_encode"
        or encoding.get("callable_module") != "bernini.pipeline"
        or encoding.get("callable_name") != "_vae_encode"
        or encoding.get("callable_qualname") != "_vae_encode"
        or encoding.get("callable_signature")
        != "(vae, x: torch.Tensor) -> torch.Tensor"
        or encoding.get("encoded_in_runner") is not False
        or encoding.get("full_source_vae_encode_count") != 1
        or encoding.get("total_vae_encode_count") != 1
        or encoding.get("posterior_statistic") != "latent_dist.mode"
        or encoding.get("sampling") is not False
        or encoding.get("torch_inference_mode") is not True
        or encoding.get("source_pixels_mutated") is not False
        or encoding.get("source_pixels_before_sha256") != encoding.get("source_pixels_after_sha256")
        or encoding.get("vae_dtype") != "torch.float32"
        or encoding.get("vae_eval") is not True
        or encoding.get("vae_requires_grad") is not False
        or encoding.get("latent_frame_count") != 21
        or encoding.get("finite") is not True
    ):
        raise SystemExit("source materializer one-encode closure differs")
    runtime = receipt.get("runtime", {})
    authority = receipt.get("authority", {})
    if (
        set(runtime) != {
            "device_requested", "world_size", "distributed_initialized", "python_version",
            "torch_version", "hip_version", "diffusers_version", "safetensors_version",
        }
        or runtime.get("device_requested") != "cuda:0"
        or runtime.get("world_size") != 1
        or runtime.get("distributed_initialized") is not False
        or set(authority) != materializer_authority
        or any(value is not False for value in authority.values())
    ):
        raise SystemExit("source materializer WORLD1/authority closure differs")
    return {
        "schema_version": receipt["schema_version"],
        "method": receipt["method"],
        "row_id": row,
        "source_video_sha256": source_sha,
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": artifact_file_sha,
        "artifact_size": artifact_info.st_size,
        "tensor_raw_sha256": tensor_sha,
        "shape": list(tensor.shape),
        "dtype": "torch.float32",
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha,
        "receipt_content_sha256": receipt["receipt_digest"],
        "source_manifest_raw_sha256": sealed["source_manifest_raw_sha256"],
        "source_manifest_content_sha256": sealed["source_manifest_content_sha256"],
        "checkpoint_tree_sha256": model["checkpoint_tree_sha256"],
        "checkpoint_content_manifest_audit": checkpoint_audit,
        "source_derived_bucket_hw": preprocessing["source_derived_bucket_hw"],
        "method_source_revision": model["method_source_revision"],
        "method_source_archive_sha256": model["method_source_archive_sha256"],
        "runtime_source_index_sha256": model["runtime_source_index_sha256"],
        "world_size": 1,
        "full_source_vae_encode_count": 1,
        "posterior_statistic": "latent_dist.mode",
        "sampling": False,
        "consumed_by_arms": ["T0", "I0"],
        "consumer_count": 2,
        "artifact_reopened_and_verified_before_parent_publication": True,
        "receipt_reopened_and_verified_before_parent_publication": True,
        "terminal_postconsumer_reopen_verified": True,
        "quality_or_training_authority": False,
    }

def validate_runner(group, arm, row, seed, source_sha, materialization):
    media = root / "groups" / group / f"{arm}.mp4"
    clean_latent = media.with_name(
        f"{media.name}.normalized-clean-latent.safetensors"
    )
    receipt_path = media.with_name(media.name + ".receipt.json")
    media_info = plain_file(media, 0o444)
    clean_latent_info = plain_file(clean_latent, 0o444)
    receipt, receipt_file_sha = read_canonical_receipt(receipt_path)
    if set(receipt) != expected_top:
        raise SystemExit("runner receipt top-level closure differs")
    expected_arm = {
        "T0": {
            "arm": "T0", "field_regime": "t2v_apg", "task_name": "t2v",
            "guidance_mode": "t2v_apg", "candidate_schedule": [1] * 40,
            "anc_enabled": False, "aggregation_mode": "uniform",
            "temperature": None, "anchor_latent_phase_zero": False,
        },
        "I0": {
            "arm": "I0", "field_regime": "r2v_apg_source_i0", "task_name": "r2v",
            "guidance_mode": "r2v_apg", "candidate_schedule": [1] * 40,
            "anc_enabled": False, "aggregation_mode": "uniform",
            "temperature": None, "anchor_latent_phase_zero": False,
        },
    }[arm]
    if receipt.get("arm") != expected_arm:
        raise SystemExit("runner arm differs")
    selected = receipt.get("sealed_inputs", {}).get("selected_cell", {})
    if (
        selected.get("row_id") != row
        or selected.get("rollout_seed") != seed
        or selected.get("branch") != "forward"
        or selected.get("source_video_sha256") != source_sha
    ):
        raise SystemExit("runner sealed cell differs")
    sealed = receipt["sealed_inputs"]
    if (
        sealed.get("source_manifest_raw_sha256") != source_manifest_raw_sha
        or sealed.get("source_manifest_content_sha256") != source_manifest_content_sha
        or sealed.get("event_bank_raw_sha256") != event_bank_raw_sha
        or sealed.get("event_bank_content_sha256") != event_bank_content_sha
        or sealed.get("source_manifest_bound_files_verified") is not False
        or sealed.get("source_manifest_terminal_raw_sha256_verified") is not True
        or sealed.get("event_bank_terminal_raw_sha256_verified") is not True
        or sealed.get("selected_source_video_terminal_raw_sha256_verified") is not True
        or sealed.get("optimizer_authorized") is not False
        or sealed.get("source_manifest_terminal_events_verified") is not False
        or sealed.get("event_bank_events_verified") is not False
        or sealed.get("target_video") is not False
        or sealed.get("target_or_oracle_frame") is not False
        or sealed.get("mask_or_swept_tube") is not False
        or sealed.get("pose_flow_track_or_trajectory") is not False
        or sealed.get("motion_donor") is not False
        or sealed.get("external_reference") is not False
    ):
        raise SystemExit("runner sealed manifest identities differ")
    model = receipt["model"]
    provenance = model.get("method_provenance", {})
    checkpoint_content = model.get("checkpoint_content", {})
    if (
        model.get("checkpoint_tree_sha256") != checkpoint_tree_sha
        or model.get("bernini_revision") != bernini_commit
        or model.get("veomni_revision") != veomni_commit
        or provenance.get("revision") != source_revision
        or provenance.get("archive_sha256") != archive_sha
        or model.get("one_arm_per_process_group") is not True
        or model.get("model_once_multi_arm_supported") is not False
        or model.get("unsupported_model_once") is not True
        or checkpoint_content.get("manifest_sha256_computed") != checkpoint_manifest_sha
        or checkpoint_content.get("manifest_sha256_expected") != checkpoint_manifest_sha
        or checkpoint_content.get("verified_file_count") != 23
        or checkpoint_content.get("every_file_sha256_verified") is not True
    ):
        raise SystemExit("runner model/method identity differs")
    source_path = Path(sealed.get("source_video_resolved_path", ""))
    if (
        not source_path.is_absolute()
        or not source_path.is_file()
        or source_path.is_symlink()
        or file_sha(source_path) != source_sha
    ):
        raise SystemExit("runner-bound source bytes changed before parent publication")
    schedule = receipt["schedule"]
    transport = receipt.get("transport", {})
    source_outer_identity = transport.get("source_outer_clean_state", {})
    source_outer_tensor = source_outer_identity.get("identity", {})
    reference_outer_identity = transport.get("reference_frame0")
    sealed_coordinate = transport.get("sealed_source_coordinate", {})
    cert = receipt.get("distributed_execution", {}).get("certificate", {})
    certificate_coordinate = cert.get("sealed_source_coordinate", {})
    expected_coordinate_keys = {
        "schema_version", "artifact_path", "artifact_sha256", "artifact_size",
        "artifact_mode", "receipt_path", "receipt_file_sha256", "receipt_digest",
        "tensor_key", "tensor_raw_sha256", "shape", "dtype",
        "source_manifest_raw_sha256", "source_manifest_content_sha256", "row_id",
        "source_video_sha256", "checkpoint_tree_sha256",
        "checkpoint_content_manifest_audit", "source_derived_bucket_hw",
        "materializer_method_source_revision",
        "materializer_method_source_archive_sha256",
        "materializer_runtime_source_index_sha256",
        "loaded_from_sealed_source_coordinate", "encoded_in_runner",
        "runner_reencoding_verified", "inference_available_source_video",
        "ground_truth", "quality_authority", "semantic_action_success",
        "identity_preservation_success", "training_authority", "optimizer_authority",
        "cpu_to_gpu_byte_exact", "rank0_broadcast_before_renderer",
        "all_rank_identity_after_broadcast", "pre_rollout_rehash",
        "pre_publish_rehash", "terminal_rehash_recorded_in_this_receipt",
        "terminal_rehash_required_for_process_success",
    }
    if (
        not isinstance(sealed_coordinate, dict)
        or set(sealed_coordinate) != expected_coordinate_keys
        or sealed_coordinate != certificate_coordinate
        or sealed_coordinate.get("schema_version")
        != "bernini-saic-source-clean-latent-receipt-v1"
        or sealed_coordinate.get("artifact_path") != materialization["artifact_path"]
        or sealed_coordinate.get("artifact_sha256")
        != materialization["artifact_file_sha256"]
        or sealed_coordinate.get("artifact_size") != materialization["artifact_size"]
        or sealed_coordinate.get("artifact_mode") != "0444"
        or sealed_coordinate.get("receipt_path") != materialization["receipt_path"]
        or sealed_coordinate.get("receipt_file_sha256")
        != materialization["receipt_file_sha256"]
        or sealed_coordinate.get("receipt_digest")
        != materialization["receipt_content_sha256"]
        or sealed_coordinate.get("tensor_key") != "source_clean_latent"
        or sealed_coordinate.get("tensor_raw_sha256")
        != materialization["tensor_raw_sha256"]
        or sealed_coordinate.get("shape") != materialization["shape"]
        or sealed_coordinate.get("dtype") != materialization["dtype"]
        or sealed_coordinate.get("source_manifest_raw_sha256")
        != materialization["source_manifest_raw_sha256"]
        or sealed_coordinate.get("source_manifest_content_sha256")
        != materialization["source_manifest_content_sha256"]
        or sealed_coordinate.get("row_id") != row
        or sealed_coordinate.get("source_video_sha256") != source_sha
        or sealed_coordinate.get("checkpoint_tree_sha256")
        != materialization["checkpoint_tree_sha256"]
        or sealed_coordinate.get("checkpoint_content_manifest_audit")
        != materialization["checkpoint_content_manifest_audit"]
        or sealed_coordinate.get("source_derived_bucket_hw")
        != materialization["source_derived_bucket_hw"]
        or sealed_coordinate.get("materializer_method_source_revision")
        != materialization["method_source_revision"]
        or sealed_coordinate.get("materializer_method_source_archive_sha256")
        != materialization["method_source_archive_sha256"]
        or sealed_coordinate.get("materializer_runtime_source_index_sha256")
        != materialization["runtime_source_index_sha256"]
        or sealed_coordinate.get("loaded_from_sealed_source_coordinate") is not True
        or sealed_coordinate.get("encoded_in_runner") is not False
        or sealed_coordinate.get("runner_reencoding_verified") is not False
        or sealed_coordinate.get("inference_available_source_video") is not True
        or any(
            sealed_coordinate.get(key) is not False
            for key in (
                "ground_truth", "quality_authority", "semantic_action_success",
                "identity_preservation_success", "training_authority", "optimizer_authority",
            )
        )
        or sealed_coordinate.get("cpu_to_gpu_byte_exact") is not True
        or sealed_coordinate.get("rank0_broadcast_before_renderer") is not True
        or sealed_coordinate.get("all_rank_identity_after_broadcast") is not True
        or sealed_coordinate.get("terminal_rehash_recorded_in_this_receipt") is not False
        or sealed_coordinate.get("terminal_rehash_required_for_process_success") is not True
        or transport.get("source_outer_clean_state_loaded_from_sealed_coordinate") is not True
        or transport.get("complete_source_video_vae_encoded_in_runner") is not False
    ):
        raise SystemExit("runner sealed source-coordinate provenance differs")
    expected_rehash_keys = {
        "stage", "artifact_sha256", "receipt_file_sha256", "receipt_digest",
        "tensor_raw_sha256", "retained_descriptor_identity_verified",
        "canonical_path_identity_verified", "mode_0444_verified",
        "canonical_receipt_and_digest_verified", "tensor_reopened_byte_exact",
    }
    for stage in ("pre_rollout", "pre_publish"):
        rehash = sealed_coordinate.get(f"{stage}_rehash", {})
        if (
            not isinstance(rehash, dict)
            or set(rehash) != expected_rehash_keys
            or rehash.get("stage") != stage
            or rehash.get("artifact_sha256") != materialization["artifact_file_sha256"]
            or rehash.get("receipt_file_sha256")
            != materialization["receipt_file_sha256"]
            or rehash.get("receipt_digest") != materialization["receipt_content_sha256"]
            or rehash.get("tensor_raw_sha256") != materialization["tensor_raw_sha256"]
            or any(
                rehash.get(key) is not True
                for key in (
                    "retained_descriptor_identity_verified",
                    "canonical_path_identity_verified", "mode_0444_verified",
                    "canonical_receipt_and_digest_verified", "tensor_reopened_byte_exact",
                )
            )
        ):
            raise SystemExit(f"runner sealed source-coordinate {stage} rehash differs")
    if (
        transport.get("full_source_video_field_tokens") is not False
        or transport.get("reference_is_source_rgb_frame0") is not (arm == "I0")
        or transport.get("reference_is_available_at_inference") is not (arm == "I0")
        or transport.get("reference_independently_vae_encoded") is not (arm == "I0")
        or transport.get("reference_from_temporal_latent_slice") is not False
        or (arm == "T0" and transport.get("reference_frame0") is not None)
        or (arm == "I0" and not isinstance(transport.get("reference_frame0"), dict))
        or source_outer_identity.get("all_rank_exact") is not True
        or source_outer_tensor.get("dtype") != "torch.float32"
        or source_outer_tensor.get("shape", [])[:3] != [1, 16, 21]
        or source_outer_tensor.get("raw_storage_sha256")
        != materialization["tensor_raw_sha256"]
    ):
        raise SystemExit("runner T0/I0 condition-bandwidth contract differs")
    if (
        schedule.get("num_frames") != 81
        or schedule.get("latent_frames") != 21
        or schedule.get("fps") != 25
        or schedule.get("num_inference_steps") != 40
        or schedule.get("flow_shift") != 5.0
    ):
        raise SystemExit("runner exact81/25fps/exact40/shift5 receipt differs")
    output = receipt["output"]
    actual_media_sha = file_sha(media)
    actual_clean_latent_file_sha = file_sha(clean_latent)
    clean_identity = output.get("normalized_clean_latent", {})
    try:
        with safe_open(str(clean_latent), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != ["normalized_clean_latent"]:
                raise SystemExit("normalized clean latent tensor key differs")
            clean_tensor = opened.get_tensor("normalized_clean_latent").contiguous()
            clean_metadata = dict(opened.metadata() or {})
    except (OSError, RuntimeError) as error:
        raise SystemExit("cannot reopen normalized clean latent") from error
    actual_clean_tensor_sha = tensor_raw_sha(clean_tensor)
    expected_clean_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "source_state_flow_transport_endpoint",
        "source": "exact40_transport_edit_clean_before_vae_decode",
        "ground_truth": "false",
        "selected_for_training": "false",
    }
    if (
        output.get("path") != str(media)
        or output.get("sha256") != actual_media_sha
        or output.get("size") != media_info.st_size
        or output.get("frame_count") != 81
        or output.get("fps") != 25
        or output.get("published_video_mode") != "0444"
        or output.get("published_normalized_clean_latent_mode") != "0444"
        or output.get("published_receipt_mode") != "0444"
        or output.get("normalized_clean_latent_is_ground_truth") is not False
        or output.get("normalized_clean_latent_selected_for_training") is not False
        or clean_identity.get("path") != str(clean_latent)
        or clean_identity.get("sha256") != actual_clean_latent_file_sha
        or clean_identity.get("size") != clean_latent_info.st_size
        or clean_identity.get("tensor_key") != "normalized_clean_latent"
        or clean_identity.get("tensor_raw_sha256") != actual_clean_tensor_sha
        or clean_identity.get("shape") != list(clean_tensor.shape)
        or clean_identity.get("shape") != source_outer_tensor.get("shape")
        or clean_identity.get("stored_dtype") != "torch.float32"
        or clean_identity.get("transport_endpoint_before_vae_decode") is not True
        or clean_identity.get("mp4_decode_reencode_used") is not False
        or clean_identity.get("roundtrip_byte_exact_fp32") is not True
        or clean_identity.get("quality_or_selection_authority") is not False
        or clean_tensor.dtype != torch.float32
        or clean_tensor.ndim != 5
        or list(clean_tensor.shape[:3]) != [1, 16, 21]
        or clean_metadata != expected_clean_metadata
    ):
        raise SystemExit("runner media/normalized-clean-latent identity differs")
    authority = receipt["authority"]
    if authority.get("frozen_inference_execution_receipt") is not True or any(
        authority.get(key) is not False for key in false_authority
    ):
        raise SystemExit("runner authority differs")
    adapter = cert.get("native_adapter", {})
    noise = receipt.get("noise", {})
    for key in (
        "source_latent_raw_sha256", "reference_frame0_raw_sha256",
        "generated_latent_raw_sha256", "noise_bank_sha256",
    ):
        if sha_pattern.fullmatch(str(cert.get(key, ""))) is None:
            raise SystemExit(f"runner latent/noise digest malformed: {key}")
    if (
        (arm == "T0" and cert.get("reference_frame0_raw_sha256") != "0" * 64)
        or (arm == "I0" and cert.get("reference_frame0_raw_sha256") == "0" * 64)
        or source_outer_tensor.get("raw_storage_sha256")
        != cert.get("source_latent_raw_sha256")
        or cert.get("source_latent_raw_sha256")
        != materialization["tensor_raw_sha256"]
        or (
            arm == "I0"
            and reference_outer_identity.get("identity", {}).get("raw_storage_sha256")
            != cert.get("reference_frame0_raw_sha256")
        )
    ):
        raise SystemExit("runner reference-frame0 digest closure differs")
    if (
        cert.get("arm") != arm
        or cert.get("row_id") != row
        or cert.get("rollout_seed") != seed
        or cert.get("model_freeze_unchanged") is not True
        or adapter.get("rollout_complete") is not True
        or adapter.get("adapter_failed") is not False
        or adapter.get("failure_stage") is not None
        or adapter.get("initial_full_model_content_audit") is not True
        or adapter.get("final_full_model_content_audit") is not True
        or adapter.get("optimizer_step_allowed") is not False
        or adapter.get("training_update_allowed") is not False
        or adapter.get("semantic_action_success") is not False
        or actual_clean_tensor_sha != cert.get("generated_latent_raw_sha256")
        or noise.get("actual_ordered_noise_bank_sha256") != cert.get("noise_bank_sha256")
        or noise.get("actual_candidate_zero_subbank_sha256") != cert.get("candidate_zero_noise_sha256")
        or noise.get("candidate_schedule") != [1] * 40
        or noise.get("candidate_continuation") != "candidate_zero"
    ):
        raise SystemExit("runner latent execution certificate differs")
    guided_attempt = cert.get("native_guided_query_attempt_count")
    guided_success = cert.get("native_guided_query_success_count")
    raw_attempt = cert.get("native_raw_transformer_forward_attempt_count")
    raw_success = cert.get("native_raw_transformer_forward_success_count")
    expected_guided = 80
    expected_raw = 160 if arm == "T0" else 240
    if (
        type(guided_attempt) is not int
        or guided_attempt != expected_guided
        or guided_success != guided_attempt
        or type(raw_attempt) is not int
        or raw_attempt != expected_raw
        or raw_success != raw_attempt
        or cert.get("core_native_guided_count_reconciled") is not True
        or cert.get("core_native_raw_forward_count_reconciled") is not True
        or adapter.get("guided_query_attempt_count") != guided_attempt
        or adapter.get("guided_query_success_count") != guided_success
        or adapter.get("expected_guided_query_count") != guided_success
        or adapter.get("raw_transformer_forward_attempt_count") != raw_attempt
        or adapter.get("raw_transformer_forward_success_count") != raw_success
        or adapter.get("expected_raw_transformer_forward_count") != raw_success
    ):
        raise SystemExit("runner native attempt/success counter closure differs")
    return {
        "arm": arm,
        "media_path": str(media),
        "media_sha256": actual_media_sha,
        "media_size": media_info.st_size,
        "normalized_clean_latent_path": str(clean_latent),
        "normalized_clean_latent_file_sha256": actual_clean_latent_file_sha,
        "normalized_clean_latent_size": clean_latent_info.st_size,
        "normalized_clean_latent_tensor_raw_sha256": actual_clean_tensor_sha,
        "runner_receipt_path": str(receipt_path),
        "runner_receipt_file_sha256": receipt_file_sha,
        "runner_receipt_content_sha256": receipt["receipt_digest"],
        "source_latent_raw_sha256": cert["source_latent_raw_sha256"],
        "materialized_source_tensor_raw_sha256": materialization[
            "tensor_raw_sha256"
        ],
        "source_latent_equals_single_materialization": True,
        "runner_terminal_source_coordinate_rehash_required": True,
        "runner_process_success_implies_terminal_rehash_all_rank_exact": True,
        "reference_frame0_raw_sha256": cert["reference_frame0_raw_sha256"],
        "generated_latent_raw_sha256": cert["generated_latent_raw_sha256"],
        "noise_bank_sha256": cert["noise_bank_sha256"],
        "candidate_zero_noise_sha256": cert["candidate_zero_noise_sha256"],
        "native_guided_query_attempt_and_success_count": guided_success,
        "native_raw_transformer_forward_attempt_and_success_count": raw_success,
        "receipt_reopened_and_verified": True,
        "media_reopened_and_verified": True,
        "normalized_clean_latent_reopened_and_verified": True,
        "latent_hashes_verified": True,
    }

def publish_receipt(path, value, token):
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing receipt overwrite: {path}")
    payload = dict(value)
    payload["receipt_digest"] = object_sha(payload)
    temporary = path.with_name(f".{path.name}.tmp-{token}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return read_canonical_receipt(path)

groups = (
    ("dog", dog_row, dog_seed, dog_source_sha, "0,1,2,3"),
    ("human", human_row, human_seed, human_source_sha, "4,5,6,7"),
)
group_records = []
for group, row, seed, source_sha, devices in groups:
    materialization = validate_materialization(group, row, source_sha)
    attempts = [
        validate_runner(group, arm, row, seed, source_sha, materialization)
        for arm in ("T0", "I0")
    ]
    if attempts[0]["candidate_zero_noise_sha256"] != attempts[1]["candidate_zero_noise_sha256"]:
        raise SystemExit("T0/I0 candidate-zero noise bytes differ for one source")
    if attempts[0]["source_latent_raw_sha256"] != attempts[1]["source_latent_raw_sha256"]:
        raise SystemExit("T0/I0 source clean latent bytes differ for one source")
    if attempts[0]["source_latent_raw_sha256"] != materialization["tensor_raw_sha256"]:
        raise SystemExit("T0/I0 source clean latent differs from single materialization")
    group_value = {
        "schema_version": "bernini-saic-ssft-world4-child-receipt-v1",
        "group": group,
        "row_id": row,
        "rollout_seed": seed,
        "source_video_sha256": source_sha,
        "visible_physical_gpus": devices,
        "world_size": 4,
        "ulysses_size": 4,
        "pilot_arms_in_fixed_order": ["T0", "I0"],
        "runner_supports_multi_arm_model_reuse": False,
        "model_loads_in_group": 2,
        "source_materialization_count": 1,
        "source_clean_vae_encode_count": 1,
        "source_materialization": materialization,
        "source_materialization_consumers": ["T0", "I0"],
        "attempts": attempts,
        "verification": {
            "runner_receipts_reopened": True,
            "media_reopened": True,
            "normalized_clean_latents_reopened": True,
            "latent_hashes_verified": True,
            "t0_i0_candidate_zero_noise_identical": True,
            "t0_i0_source_clean_latent_identical": True,
            "t0_i0_source_clean_latent_equals_materialized_tensor": True,
            "single_materialization_consumed_by_both_arms": True,
            "runner_source_reencoding": False,
            "runner_terminal_source_coordinate_rehash_enforced_by_process_success": True,
            "launcher_postconsumer_materialization_reopen_verified": True,
            "no_arm_selection": True,
        },
        "authority": {
            "quality_authority": False,
            "selection_authority": False,
            "semantic_action_success": False,
            "identity_preservation_success": False,
            "training_authority": False,
            "optimizer_authority": False,
            "training_update_allowed": False,
            "optimizer_step_allowed": False,
        },
    }
    group_path = root / "groups" / group / "group-receipt.json"
    group_receipt, group_file_sha = publish_receipt(group_path, group_value, f"{job_id}-{group}")
    # Parent consumes a reopened child receipt, not the in-memory value.
    reopened, reopened_file_sha = read_canonical_receipt(group_path)
    if reopened != group_receipt or reopened_file_sha != group_file_sha:
        raise SystemExit("child group receipt reopen differs")
    group_records.append({
        "group": group,
        "child_receipt_path": str(group_path),
        "child_receipt_file_sha256": group_file_sha,
        "child_receipt_content_sha256": reopened["receipt_digest"],
        "source_materialization": reopened["source_materialization"],
        "source_materialization_consumers": reopened[
            "source_materialization_consumers"
        ],
        "attempts": reopened["attempts"],
        "child_receipt_reopened_and_verified": True,
    })

if [item["group"] for item in group_records] != ["dog", "human"]:
    raise SystemExit("exactly two child group receipts are required")
if sum(len(item["attempts"]) for item in group_records) != 4:
    raise SystemExit("exactly four frozen runner artifacts are required")
if len([item["source_materialization"] for item in group_records]) != 2:
    raise SystemExit("exactly two source materializations are required")
all_attempts = [attempt for item in group_records for attempt in item["attempts"]]
if len({attempt["media_path"] for attempt in all_attempts}) != 4 or len({attempt["runner_receipt_path"] for attempt in all_attempts}) != 4:
    raise SystemExit("runner media/receipt paths are not four unique create-only artifacts")
if len({attempt["normalized_clean_latent_path"] for attempt in all_attempts}) != 4:
    raise SystemExit("normalized clean latent paths are not four unique create-only artifacts")

parent = {
    "schema_version": "bernini-saic-ssft-all8-parent-receipt-v1",
    "job_id": job_id,
    "topology": {
        "node_count": 1,
        "allocated_mi210": 8,
        "concurrent_world4_groups_per_wave": 2,
        "dog_physical_gpus": [0, 1, 2, 3],
        "human_physical_gpus": [4, 5, 6, 7],
        "hidden_idle_four_card_group": False,
        "pilot_arms_in_fixed_order": ["T0", "I0"],
        "waves": 2,
        "runner_supports_multi_arm_model_reuse": False,
        "model_loads_per_group": 2,
        "concurrent_source_only_world1_materializers": 2,
        "source_materializer_physical_gpus": {"dog": 0, "human": 4},
    },
    "exact_runtime": {
        "frames": 81,
        "fps": 25,
        "latent_frames": 21,
        "inference_steps": 40,
        "flow_shift": 5.0,
        "frozen_model": True,
    },
    "provenance": {
        "method_source_revision": source_revision,
        "method_source_archive_sha256": archive_sha,
        "bernini_commit": bernini_commit,
        "veomni_commit": veomni_commit,
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
        "source_manifest_raw_sha256": source_manifest_raw_sha,
        "source_manifest_content_sha256": source_manifest_content_sha,
        "event_bank_raw_sha256": event_bank_raw_sha,
        "event_bank_content_sha256": event_bank_content_sha,
    },
    "children": group_records,
    "source_coordinate_policy": {
        "source_materialization_count": 2,
        "source_clean_vae_encode_count": 2,
        "consumer_count": 4,
        "consumers_per_materialization": ["T0", "I0"],
        "materialized_before_any_arm": True,
        "fresh_create_only_no_cache_reuse": True,
        "runner_source_reencoding": False,
        "predecessor_job_132359_cross_arm_source_latent_mismatch_observed": True,
        "cause_attribution_authorized": False,
        "single_materialization_removes_repeated_encode_boundary": True,
        "quality_or_semantic_success_implied": False,
    },
    "verification": {
        "child_group_receipt_count": 2,
        "runner_receipt_count": 4,
        "media_count": 4,
        "normalized_clean_latent_count": 4,
        "source_latent_hash_count": 4,
        "generated_latent_hash_count": 4,
        "source_materialization_count": 2,
        "source_materialization_receipt_count": 2,
        "source_materialization_consumer_count": 4,
        "source_materialization_artifacts_reopened": True,
        "source_materialization_receipts_reopened": True,
        "source_materializations_reopened_after_all_consumers": True,
        "runner_terminal_source_coordinate_rehash_enforced_by_successful_exit": True,
        "runner_source_latents_equal_materialized_tensors": True,
        "child_receipts_reopened": True,
        "runner_receipts_reopened": True,
        "media_hashes_verified": True,
        "normalized_clean_latent_file_and_tensor_hashes_verified": True,
        "latent_hashes_verified": True,
        "candidate_zero_noise_equality_verified_within_each_source": True,
        "source_clean_latent_equality_verified_within_each_source": True,
        "transactional_create_only_publication": True,
    },
    "authority": {
        "quality_authority": False,
        "selection_authority": False,
        "semantic_action_success": False,
        "identity_preservation_success": False,
        "training_authority": False,
        "optimizer_authority": False,
        "training_update_allowed": False,
        "optimizer_step_allowed": False,
        "checkpoint_or_lora_artifact": False,
        "production_claim_authorized": False,
    },
}
parent_path = root / "all8-parent-receipt.json"
parent_receipt, parent_file_sha = publish_receipt(parent_path, parent, f"{job_id}-all8")
reopened_parent, reopened_parent_file_sha = read_canonical_receipt(parent_path)
if reopened_parent != parent_receipt or reopened_parent_file_sha != parent_file_sha:
    raise SystemExit("all8 parent receipt reopen differs")
print(canonical({
    "parent_receipt": str(parent_path),
    "parent_receipt_file_sha256": parent_file_sha,
    "parent_receipt_content_sha256": parent_receipt["receipt_digest"],
    "child_group_receipts": 2,
    "runner_receipts": 4,
    "media": 4,
    "selection_authority": False,
    "training_authority": False,
    "optimizer_authority": False,
    "semantic_action_success": False,
}).decode("utf-8"))
PY

# Terminal source identities are checked after both long GPU waves.  A changed
# archive/manifest makes publication fail closed even if all children exited 0.
check_hash "${source_archive}" "${source_archive_sha256}" source_archive_terminal
check_hash "${archive_copy}" "${source_archive_sha256}" scratch_archive_terminal
check_hash "${durable_archive}" "${source_archive_sha256}" durable_archive_terminal
check_hash "${sealed_source_manifest}" "${pinned_source_manifest_raw_sha256}" source_manifest_terminal
check_hash "${sealed_event_bank}" "${pinned_event_bank_raw_sha256}" event_bank_terminal
check_hash "${checkpoint_manifest}" "${pinned_checkpoint_manifest_sha256}" checkpoint_manifest_terminal
check_hash "${dog_source_clean}" "${dog_source_clean_sha256}" dog_source_clean_terminal
check_hash "${dog_source_clean_receipt}" "${dog_source_clean_receipt_sha256}" dog_source_clean_receipt_terminal
check_hash "${human_source_clean}" "${human_source_clean_sha256}" human_source_clean_terminal
check_hash "${human_source_clean_receipt}" "${human_source_clean_receipt_sha256}" human_source_clean_receipt_terminal
for path in "${dog_source_clean}" "${dog_source_clean_receipt}" \
  "${human_source_clean}" "${human_source_clean_receipt}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source materialization terminal path differs"
  [[ "$(stat -c '%a' -- "${path}")" == 444 ]] || \
    fail "source materialization terminal mode differs"
done
[[ -f "${output_root}/all8-parent-receipt.json" && ! -L "${output_root}/all8-parent-receipt.json" ]] || \
  fail "all8 parent receipt missing"
[[ "$(stat -c '%a' -- "${output_root}/all8-parent-receipt.json")" == 444 ]] || \
  fail "all8 parent receipt is not immutable"

find "${output_root}" -type f -exec chmod a-w -- {} +
find "${output_root}" -type d -exec chmod a-w -- {} +
published=true
