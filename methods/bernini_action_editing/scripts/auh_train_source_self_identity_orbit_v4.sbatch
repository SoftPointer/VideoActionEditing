#!/usr/bin/env bash
#SBATCH --job-name=bern-cio-v4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() {
  echo "[bernini-cio-v4] ERROR: $*" >&2
  exit 2
}

output="${BERNINI_CIO_OUTPUT:?set a fresh absolute BERNINI_CIO_OUTPUT}"
dataset_root="${BERNINI_CIO_DATASET_ROOT:?set the sealed appearance-orbit dataset root}"
dataset_receipt_sha256="${BERNINI_CIO_DATASET_RECEIPT_SHA256:?set BERNINI_CIO_DATASET_RECEIPT_SHA256}"
spec_sha256="${BERNINI_CIO_SPEC_SHA256:?set BERNINI_CIO_SPEC_SHA256}"
mode="${BERNINI_CIO_MODE:?set BERNINI_CIO_MODE=sealed-prefix-canary or complete-cycle}"
max_steps="${BERNINI_CIO_MAX_STEPS:?set BERNINI_CIO_MAX_STEPS}"
prefix_digest="${BERNINI_CIO_PREFIX_DIGEST:-}"
source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
python_bin="${BERNINI_CIO_PYTHON_BIN:?set BERNINI_CIO_PYTHON_BIN to vace Python}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

for name in output dataset_root source_archive source_repository bernini_root veomni_root checkpoint checkpoint_manifest python_bin; do
  value="${!name}"
  [[ "${value}" == /* ]] || fail "${name} must be absolute"
done
[[ "${dataset_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid dataset receipt SHA-256"
[[ "${spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid spec SHA-256"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || fail "max steps must be a positive decimal integer"

mode_args=()
case "${mode}" in
  sealed-prefix-canary)
    (( max_steps >= 1 && max_steps <= 35 )) || fail "sealed prefix must contain 1..35 steps"
    [[ "${prefix_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "sealed prefix requires a SHA-256 digest"
    mode_args+=(
      --expected-prefix-digest "${prefix_digest}"
      --ack-incomplete-cycle-no-scientific-claim
    )
    ;;
  complete-cycle)
    (( max_steps % 36 == 0 )) || fail "complete-cycle steps must be a multiple of 36"
    [[ -z "${prefix_digest}" ]] || fail "complete-cycle mode forbids a prefix digest"
    ;;
  *) fail "BERNINI_CIO_MODE must be sealed-prefix-canary or complete-cycle" ;;
esac

[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$ ]] || fail "unsafe output"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output is create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent is a symlink"

[[ -d "${dataset_root}" && ! -L "${dataset_root}" ]] || fail "dataset root is invalid"
dataset_files="$(find "${dataset_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)"
[[ "${dataset_files}" == $'dataset.parquet\nreceipt.json' ]] || fail "dataset artifact closure differs"
for artifact in dataset.parquet receipt.json; do
  path="${dataset_root}/${artifact}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "dataset ${artifact} is not a plain file"
  [[ -z "$(find "${path}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "dataset ${artifact} must be durable read-only"
done
[[ "$(sha256sum "${dataset_root}/receipt.json" | awk '{print $1}')" == "${dataset_receipt_sha256}" ]] || fail "dataset receipt hash differs"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is invalid"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "source archive must be durable read-only"
[[ -d "${source_repository}/.git" && ! -L "${source_repository}" ]] || fail "source repository is invalid"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root is invalid"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "VeOmni source root is invalid"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root is invalid"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest is invalid"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be a plain executable"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash mismatch"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}")" == commit ]] || fail "source revision is not a commit"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${expected_veomni_commit}" ]] || fail "VeOmni revision mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"
if [[ -d "${bernini_root}/.git" && ! -L "${bernini_root}/.git" ]]; then
  [[ "$(git -C "${bernini_root}" rev-parse HEAD)" == "${expected_bernini_commit}" ]] || fail "Bernini revision mismatch"
  [[ -z "$(git -C "${bernini_root}" status --porcelain --untracked-files=no)" ]] || fail "Bernini tracked source is dirty"
else
  bernini_pinned_files=(
    "bernini/models/renderer.py:fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159"
    "bernini/models/wan_diffusion.py:59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
    "bernini/models/transformer_wan.py:9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
    "bernini/models/scheduler.py:b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97"
    "bernini/training/data.py:29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65"
    "bernini/attention.py:e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055"
    "bernini/parallel/state.py:32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa"
    "bernini/parallel/ops.py:c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30"
    "configs/bernini_renderer_wan21_1p3b/config.json:4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496"
  )
  for binding in "${bernini_pinned_files[@]}"; do
    relative="${binding%%:*}"
    expected_sha256="${binding##*:}"
    pinned_path="${bernini_root}/${relative}"
    [[ -f "${pinned_path}" && ! -L "${pinned_path}" ]] || fail "Bernini pinned file is invalid: ${relative}"
    [[ "$(sha256sum "${pinned_path}" | awk '{print $1}')" == "${expected_sha256}" ]] || fail "Bernini pinned file hash mismatch: ${relative}"
  done
fi
(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid SLURM job id"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-cio-v4-${SLURM_JOB_ID}.XXXXXX")"
mkdir -- \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${task_scratch}/tmp"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}/tmp"

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/bernini-cio-v4-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-cio-v4] refusing unsafe scratch cleanup" >&2; exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

archive_copy="${task_scratch}/method.tar"
git_archive="${task_scratch}/git-method.tar"
cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source archive differs"
git -C "${source_repository}" archive --format=tar --output="${git_archive}" "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${git_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive is not the declared commit subtree"

mkdir -- "${task_scratch}/source"
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
for required in \
  appearance_counterfactual_identity_orbit.py \
  source_self_identity_orbit_v4.py \
  source_self_native_ref_contrastive_v3.py \
  source_self_native_rv2v_guidance.py \
  source_self_native_target_adapter.py \
  source_self_runtime.py \
  train_lora.py \
  train_source_self_identity_orbit_v4.py \
  scripts/auh_train_source_self_identity_orbit_v4.sbatch \
  tools/build_renderer_dataset.py \
  tools/materialize_appearance_counterfactual_identity_orbit.py \
  tools/materialize_ramp_motion_analogy_vae.py \
  tools/materialize_vae.py \
  tests/test_auh_train_source_self_identity_orbit_v4_launcher.py \
  tests/test_source_self_native_rv2v_guidance.py \
  tests/test_train_source_self_identity_orbit_v4_contract.py; do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive is missing ${required}"
done
find "${method_root}" -type f -exec chmod a-w {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
  -s "${method_root}/tests" \
  -p 'test_train_source_self_identity_orbit_v4_contract.py'
PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
  -s "${method_root}/tests" \
  -p 'test_source_self_native_rv2v_guidance.py'

if [[ "${mode}" == "sealed-prefix-canary" ]]; then
  computed_prefix_digest="$(PYTHONPATH="${method_root}" "${python_bin}" -B -c 'import sys; import train_source_self_identity_orbit_v4 as t; print(t.prefix_seal_body(int(sys.argv[1]))["digest"])' "${max_steps}")"
  [[ "${computed_prefix_digest}" == "${prefix_digest}" ]] || fail "prefix digest is not the extracted trainer's exact cycle prefix"
fi

dataset_parquet_before="$(sha256sum "${dataset_root}/dataset.parquet" | awk '{print $1}')"
dataset_receipt_before="$(sha256sum "${dataset_root}/receipt.json" | awk '{print $1}')"

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${method_root}/train_source_self_identity_orbit_v4.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --dataset-root "${dataset_root}" \
  --expected-dataset-receipt-sha256 "${dataset_receipt_sha256}" \
  --expected-materialization-spec-sha256 "${spec_sha256}" \
  --output "${output}" \
  --mode "${mode}" \
  --max-steps "${max_steps}" \
  "${mode_args[@]}" \
  --rho 0 \
  --num-frames 81 \
  --expected-bernini-commit "${expected_bernini_commit}" \
  --expected-veomni-commit "${expected_veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  --ack-pretext-not-action-editing

[[ "$(sha256sum "${dataset_root}/dataset.parquet" | awk '{print $1}')" == "${dataset_parquet_before}" ]] || fail "dataset parquet changed"
[[ "$(sha256sum "${dataset_root}/receipt.json" | awk '{print $1}')" == "${dataset_receipt_before}" ]] || fail "dataset receipt changed"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "durable source archive changed"
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "extracted method tree changed"

"${python_bin}" -B - "${output}" "${mode}" "${max_steps}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from safetensors import safe_open

root = Path(sys.argv[1]).resolve(strict=True)
mode, max_steps, revision, archive_sha = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
expected = {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}
if root.is_symlink() or {path.name for path in root.iterdir()} != expected:
    raise SystemExit("training artifact closure differs")
receipt = json.loads((root / "receipt.json").read_text(encoding="ascii"))
declared = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if hashlib.sha256(payload).hexdigest() != declared:
    raise SystemExit("training receipt digest differs")
if (
    receipt.get("schema_version") != "bernini-counterfactual-identity-orbit-training-receipt-v5"
    or receipt.get("complete") is not True
    or receipt.get("mode") != mode
    or receipt.get("optimizer_steps") != max_steps
    or receipt.get("frame_count") != 81
    or receipt.get("fps") != 25.0
    or receipt.get("rho") != 0.0
    or receipt.get("rho0_original_gaussian_values") is not True
    or receipt.get("action_editing_claim_authorized") is not False
    or receipt.get("scientific_claim_authorized") is not False
    or receipt.get("long_training_automatically_submitted") is not False
):
    raise SystemExit("training receipt scientific/run contract differs")
if receipt.get("method_source_revision") != revision or receipt.get("method_source_archive_sha256") != archive_sha:
    raise SystemExit("training source binding differs")
reference_encoding = receipt.get("dataset", {}).get("reference_encoding_contract", {})
if (
    reference_encoding.get("reference_count") != 4
    or reference_encoding.get("reference_rgb_indices") != [0, 27, 53, 80]
    or reference_encoding.get("full_video_encode_calls_per_row") != 3
    or reference_encoding.get("independent_rgb_reference_encode_calls_per_row") != 12
    or reference_encoding.get("independent_vae_encode_calls_per_row") != 15
    or reference_encoding.get("references_from_full_video_posterior_slice") is not False
    or reference_encoding.get("native_deployment_visual_conditioning")
    != "one_video_plus_four_rgb_refs"
):
    raise SystemExit("training dataset RV2V-4 reference contract differs")
native_refs = receipt.get("objective", {}).get("native_rv2v4_visual_conditioning", {})
native_guidance = receipt.get("objective", {}).get("native_rv2v_guidance", {})
if (
    native_refs.get("native_mode") != "RV2V-4"
    or native_refs.get("video_condition_count") != 1
    or native_refs.get("image_reference_count") != 4
    or native_refs.get("total_visual_condition_count") != 5
    or native_refs.get("vi_video_source_ids") != [1.0]
    or native_refs.get("vi_image_source_ids") != [2.0, 3.0, 4.0, 5.0]
    or native_refs.get("i_image_source_ids") != [1.0, 2.0, 3.0, 4.0]
    or native_refs.get("patch_call_source_ids")
    != [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0]
    or native_refs.get("patch_call_roles")
    != [
        "video:VI", "ref0:VI", "ref0:I", "ref1:VI", "ref1:I",
        "ref2:VI", "ref2:I", "ref3:VI", "ref3:I", "target",
    ]
    or native_refs.get("branch_concat_order")
    != {
        "none": ["target"],
        "V": ["video", "target"],
        "I": ["ref0", "ref1", "ref2", "ref3", "target"],
        "VI": ["video", "ref0", "ref1", "ref2", "ref3", "target"],
    }
    or native_refs.get("latent_concat_dim") != 1
    or native_refs.get("rotary_concat_dim") != 2
    or native_refs.get("source_id_interpolation_used") is not False
    or native_guidance.get("schema_version")
    != "bernini-native-rv2v-guidance-training-v2"
    or native_guidance.get("native_pack_schema")
    != "bernini-source-self-native-ref-contrastive-v4"
    or native_guidance.get("native_rv2v4_reference_contract_digest")
    != native_refs.get("digest")
):
    raise SystemExit("training native RV2V-4 visual pack differs")
distributed = receipt.get("distributed", {})
if (
    distributed.get("world_size") != 8
    or distributed.get("data_parallel_size") != 2
    or distributed.get("ulysses_sequence_parallel_size") != 4
    or distributed.get("all_eight_gpus_used") is not True
):
    raise SystemExit("DP2 x SP4 receipt differs")
checkpointing = receipt.get("gradient_checkpointing", {})
if checkpointing.get("renderer_enabled") is not False or checkpointing.get("transformer_enabled") is not False:
    raise SystemExit("gradient checkpointing was not disabled")
causal = receipt.get("causal_gates", {})
if (
    causal.get("all_dp_gates_passed") is not True
    or causal.get("used_to_change_training_cells") is not False
    or causal.get("heldout_intervention_not_necessarily_heldout_iid") is not True
    or causal.get("wrong_scene_iids_may_be_other_training_cohort_rows") is not True
    or causal.get("cross_iid_generalization_claim_authorized") is not False
):
    raise SystemExit("wrong-scene causal gate receipt differs")
roundtrip = receipt.get("adapter", {}).get("strict_real_file_roundtrip", {})
if (
    receipt.get("adapter", {}).get("changed_by_training") is not True
    or
    roundtrip.get("file_loaded_into_live_adapter") is not True
    or roundtrip.get("strict_key_shape_dtype_value_roundtrip") is not True
):
    raise SystemExit("real adapter roundtrip differs")
adapter = root / "adapter.safetensors"
if hashlib.sha256(adapter.read_bytes()).hexdigest() != receipt["artifacts"]["adapter.safetensors"]:
    raise SystemExit("adapter file hash differs")
with safe_open(str(adapter), framework="pt", device="cpu") as opened:
    metadata = dict(opened.metadata() or {})
    keys = list(opened.keys())
if metadata.get("schema_version") != "bernini-native-target-row-qo-lora-checkpoint-v2" or not keys:
    raise SystemExit("adapter metadata/tensor closure differs")
if (
    metadata.get("gradient_checkpointing_enabled") != "false"
    or metadata.get("rho_hex") != float(0.0).hex()
    or metadata.get("reference_rgb_indices_json") != "[0,27,53,80]"
    or metadata.get("native_rv2v4_reference_contract_digest")
    != native_refs.get("digest")
):
    raise SystemExit("adapter checkpointing/rho metadata differs")
PY

echo "[bernini-cio-v4] PASS mode=${mode} steps=${max_steps} output=${output}"
