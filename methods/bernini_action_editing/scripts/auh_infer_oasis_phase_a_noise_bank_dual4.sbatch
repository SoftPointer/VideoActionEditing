#!/usr/bin/env bash
#SBATCH --job-name=oasis-noise-2xsp4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

# One node runs two independent WORLD4/Ulysses-SP4 processes concurrently.
# GPU 0..3 owns dog fit+confirmation; GPU 4..7 owns human
# fit+confirmation.  Each family process loads frozen Bernini once and emits
# exact12 candidates: 2 source cells x 2 sealed seeds x 3 noise arms.
# This launcher does not run an endpoint scorer or any training process.

set -Eeuo pipefail
umask 027

fail() { echo "[oasis-noise-dual4] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

source_archive="${OASIS_PHASE_A_SOURCE_ARCHIVE:?set immutable git archive}"
source_archive_sha256="${OASIS_PHASE_A_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${OASIS_PHASE_A_SOURCE_REVISION:?set archive git revision}"
manifest="${OASIS_PHASE_A_MANIFEST:?set sealed source-only manifest}"
manifest_sha256="${OASIS_PHASE_A_MANIFEST_SHA256:?set manifest SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
checkpoint_tree_sha256="${BERNINI_CHECKPOINT_TREE_SHA256:?set checkpoint tree SHA-256}"
output_root="${OASIS_PHASE_A_NOISE_BANK_OUTPUT:?set fresh noise-bank output root}"
python_bin="${OASIS_PHASE_A_PYTHON_BIN:?set vace Python executable}"

expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

for name in source_archive manifest bernini_root veomni_root checkpoint checkpoint_manifest output_root python_bin; do
  absolute "${!name}" "${name}"
done
for name in source_archive_sha256 manifest_sha256 checkpoint_manifest_sha256 checkpoint_tree_sha256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${checkpoint_manifest_sha256}" == "${expected_checkpoint_manifest_sha256}" ]] || \
  fail "checkpoint content manifest is outside preregistration"
[[ "${checkpoint_tree_sha256}" == "${expected_checkpoint_tree_sha256}" ]] || \
  fail "checkpoint tree is outside preregistration"

source_archive="$(realpath -e -- "${source_archive}")"
manifest="$(realpath -e -- "${manifest}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for name in source_archive manifest checkpoint_manifest python_bin; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must be a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"

output_parent="$(dirname -- "${output_root}")"
output_name="${output_root##*/}"
[[ "${output_root}" != / && "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "output root is unsafe"
mkdir -p -- "${output_parent}"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_name}" ]] || fail "output root is not canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"

for pair in \
  "${source_archive}:${source_archive_sha256}" \
  "${manifest}:${manifest_sha256}" \
  "${checkpoint_manifest}:${checkpoint_manifest_sha256}"; do
  path="${pair%%:*}"
  digest="${pair##*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${digest}" ]] || \
    fail "sealed input SHA differs: ${path}"
done
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || \
  fail "source archive revision differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/oasis-noise-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/oasis-noise-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[oasis-noise-dual4] PASS all8=true topology=dog-SP4+human-SP4 exact81=true exact40=true candidates=24 guidance=fixed-native scoring=false training=false output=${output_root}"
  else
    echo "[oasis-noise-dual4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true
  done
  for child in "${dog_pid}" "${human_pid}"; do
    [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || \
  fail "scratch source archive hash differs after copy"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || \
  fail "scratch source archive revision differs after copy"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
    "methods/bernini_action_editing/infer_oasis_phase_a_noise_bank.py",
    "methods/bernini_action_editing/oasis_phase_a_core.py",
    "methods/bernini_action_editing/oasis_phase_a_manifest.py",
    "methods/bernini_action_editing/motion_null_appearance_noise.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
    "methods/bernini_action_editing/tests/test_motion_null_appearance_noise.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_noise_bank.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_core.py",
    "methods/bernini_action_editing/tests/test_oasis_phase_a_manifest.py",
    "methods/bernini_action_editing/tests/test_auh_oasis_phase_a_noise_bank_launcher.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise SystemExit("unsafe source archive member")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks OASIS noise-bank closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
method_tree_before="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

for test_file in \
  test_motion_null_appearance_noise.py \
  test_oasis_phase_a_noise_bank.py \
  test_oasis_phase_a_core.py \
  test_oasis_phase_a_manifest.py \
  test_auh_oasis_phase_a_noise_bank_launcher.py; do
  # Execute by filename.  ``python -m unittest /absolute/path`` interprets
  # dots in an AUH username (for example peiyuan.zhu) as module separators.
  PYTHONPATH="${method_root}" "${python_bin}" -B \
    "${method_root}/tests/${test_file}"
done

mkdir -- "${output_root}"
dog_output="${output_root}/dog_sit_hold"
human_output="${output_root}/human_stand_hold"
dog_log="${output_root}/dog_sit_hold.log"
human_log="${output_root}/human_stand_hold.log"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((32000 + 2 * job_mod))
human_port=$((dog_port + 1))
[[ "${dog_port}" -ne "${human_port}" ]] || fail "rendezvous ports collided"

launch_family() (
  set -Eeuo pipefail
  local family="$1" visible_gpus="$2" master_port="$3" family_output="$4"
  local family_scratch="${task_scratch}/groups/${family}"
  mkdir -p -- "${family_scratch}/cache/miopen-user" \
    "${family_scratch}/cache/miopen-custom" \
    "${family_scratch}/cache/torch-extensions" \
    "${family_scratch}/cache/triton" \
    "${family_scratch}/cache/xdg" \
    "${family_scratch}/cache/pycache" \
    "${family_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${family_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${family_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${family_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${family_scratch}/cache/triton"
  export XDG_CACHE_HOME="${family_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${family_scratch}/cache/pycache"
  export TMPDIR="${family_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${family_scratch}/torch-elastic-error.json"
  echo "[oasis-noise-dual4:${family}] START visible=${visible_gpus} WORLD4/SP4 cells=2 seeds=2 arms=3 guidance=fixed-native-rv2v"
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${method_root}/infer_oasis_phase_a_noise_bank.py" \
    --manifest "${manifest}" \
    --expected-manifest-sha256 "${manifest_sha256}" \
    --family "${family}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" \
    --output-dir "${family_output}"
)

echo "[oasis-noise-dual4] allocation=8xMI210 topology=dog:0,1,2,3+human:4,5,6,7 exact81=true exact40=true noise_arms=rho0,.05,.10 guidance=fixed-native-rv2v endpoint_scorer=false training=false"
launch_family dog_sit_hold "0,1,2,3" "${dog_port}" "${dog_output}" >"${dog_log}" 2>&1 &
dog_pid=$!
launch_family human_stand_hold "4,5,6,7" "${human_port}" "${human_output}" >"${human_log}" 2>&1 &
human_pid=$!
dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 200 -- "${dog_log}" >&2 || true
  tail -n 200 -- "${human_log}" >&2 || true
  fail "one or more SP4 family banks failed"
fi

for pair in \
  "${source_archive}:${source_archive_sha256}" \
  "${manifest}:${manifest_sha256}" \
  "${checkpoint_manifest}:${checkpoint_manifest_sha256}"; do
  path="${pair%%:*}"
  digest="${pair##*:}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${digest}" ]] || \
    fail "sealed input changed during bank: ${path}"
done
method_tree_after="$(find "${method_root}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "read-only extracted method tree changed"

"${python_bin}" -B - \
  "${manifest}" "${manifest_sha256}" "${output_root}" "${method_root}" \
  "${source_revision}" "${source_archive_sha256}" \
  "${checkpoint_tree_sha256}" "${checkpoint_manifest_sha256}" <<'PY'
import hashlib, json, os, sys
from itertools import product
from pathlib import Path

(
    manifest_path,
    manifest_sha,
    root_arg,
    method_root,
    source_revision,
    source_archive_sha,
    checkpoint_tree_sha,
    checkpoint_manifest_sha,
) = sys.argv[1:]
sys.path.insert(0, method_root)
import oasis_phase_a_core as core
import oasis_phase_a_manifest as contract
import infer_oasis_phase_a_noise_bank as runtime

root = Path(root_arg).resolve(strict=True)
sealed = contract.load_phase_a_manifest(manifest_path, manifest_sha, verify_files=True)

def check_digest(value, field):
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if contract.object_sha256(unsigned) != declared:
        raise SystemExit(f"digest differs: {field}")

def check_file(artifact, digest_field):
    path = Path(artifact["path"])
    try:
        path.resolve(strict=True).relative_to(root)
    except Exception as error:
        raise SystemExit(f"artifact escaped output root: {path}") from error
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"artifact is not a plain file: {path}")
    if contract.file_sha256(path) != artifact[digest_field]:
        raise SystemExit(f"artifact SHA differs: {path}")

family_receipt_digests = {}
family_receipt_file_sha256 = {}
all_candidate_ids = set()
all_source_ids = set()
all_triplet_audit_digests = set()
for family in contract.FAMILY_ORDER:
    family_root = root / family
    receipt_path = family_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    check_digest(receipt, "receipt_digest")
    if (
        receipt.get("schema_version") != runtime.SCHEMA_VERSION
        or receipt.get("method") != runtime.METHOD
        or receipt.get("family") != family
        or receipt.get("manifest_file_sha256") != manifest_sha
        or receipt.get("manifest_digest") != sealed.manifest_digest
        or receipt.get("analysis_splits") != list(contract.SPLIT_ORDER)
        or receipt.get("seed_order") != list(sealed.seed_order)
        or receipt.get("source_cell_count") != 2
        or receipt.get("rollout_count") != 12
        or len(receipt.get("matched_triplet_audits", [])) != 4
        or receipt.get("noise_arm_order") != list(runtime.NOISE_ARM_ORDER)
        or receipt.get("method_source_revision") != source_revision
        or receipt.get("method_source_archive_sha256") != source_archive_sha
        or receipt.get("topology") != {
            "world_size": 4,
            "ulysses_sequence_parallel_size": 4,
            "all_four_ranks_reported_bit_exact_tensors": True,
        }
    ):
        raise SystemExit(f"family root contract differs: {family}")
    checkpoint = receipt.get("checkpoint", {})
    if not (
        checkpoint.get("tree_sha256") == checkpoint_tree_sha
        and checkpoint.get("content_manifest_file_sha256")
        == checkpoint_manifest_sha
        and checkpoint.get("frozen_unchanged") is True
        and isinstance(checkpoint.get("content_before_and_after"), dict)
    ):
        raise SystemExit(f"checkpoint provenance differs: {family}")
    boundary = receipt.get("composable_sampling_boundary", {})
    if not (
        boundary.get("temporary_mutation_surface") == [
            "bernini.models.wan_diffusion.randn_tensor"
        ]
        and boundary.get("official_randn_tensor_always_called_first") is True
        and boundary.get("sample_method_replaced") is False
        and boundary.get("sample_one_step_replaced") is False
        and boundary.get("native_cfg_or_apg_replaced") is False
        and boundary.get("native_scheduler_replaced") is False
        and boundary.get("guidance_policy") == "fixed_native_rv2v_no_ablation"
    ):
        raise SystemExit(f"composable wrapper boundary differs: {family}")
    operator_contract = receipt.get("noise_operator_contract", {})
    if not (
        operator_contract.get("runtime_callable")
        == "motion_null_appearance_noise.build_motion_null_appearance_noise"
        and operator_contract.get("same_callable_required_for_any_future_training")
        is True
        and operator_contract.get("alternate_training_noise_builder_authorized")
        is False
        and operator_contract.get("operator_self_registers_sampler_hook") is False
        and operator_contract.get("operator_self_registers_launcher") is False
        and operator_contract.get("dedicated_frozen_inference_integration_executed")
        is True
        and operator_contract.get("training_integration_executed") is False
    ):
        raise SystemExit(f"operator train/inference parity differs: {family}")
    information_flow = receipt.get("information_flow", {})
    if not (
        information_flow.get("external_media_inputs_source_video_only") is True
        and information_flow.get("text_input") == "complete_action_caption"
        and information_flow.get("source_frame_order_consumed_by_carrier") is False
        and information_flow.get("full_video_latent_consumed_by_carrier") is False
        and information_flow.get("target_mask_flow_pose_track_trajectory_consumed")
        is False
        and information_flow.get("proposal_media_latent_or_motion_donor_consumed")
        is False
        and information_flow.get("external_action_scorer_or_calibration_consumed")
        is False
    ):
        raise SystemExit(f"source/text information-flow contract differs: {family}")
    if any(
        receipt.get(field) is not expected
        for field, expected in (
            ("legacy_pair_v5_native_bank_authorized", False),
            ("endpoint_action_source_scoring_authorized", False),
            ("endpoint_selection_performed", False),
            ("external_action_scorer_consumed", False),
            ("optimizer_or_training_authorized", False),
            ("training_performed", False),
            ("scientific_action_editing_success_claim", False),
        )
    ):
        raise SystemExit(f"authority closure differs: {family}")
    expected_cells = sealed.cells_for_family(family)
    expected_cell_by_id = {cell.sample_id: cell for cell in expected_cells}
    expected_source_ids = {cell.sample_id for cell in expected_cells}
    observed_source_ids = {row["sample_id"] for row in receipt["source_cells"]}
    if observed_source_ids != expected_source_ids:
        raise SystemExit(f"source-cell coverage differs: {family}")
    source_row_by_id = {}
    for source_row in receipt["source_cells"]:
        cell = expected_cell_by_id[source_row["sample_id"]]
        source_row_by_id[cell.sample_id] = source_row
        if any(
            source_row.get(field) != expected
            for field, expected in (
                ("family", cell.family),
                ("analysis_split", cell.analysis_split),
                ("source_video_path", str(cell.source_video)),
                ("source_video_sha256", cell.source_video_sha256),
                ("edit_instruction_sha256", cell.edit_instruction_sha256),
                (
                    "source_instruction_binding_digest",
                    cell.source_instruction_binding_digest,
                ),
                ("sample_digest", cell.sample_digest),
            )
        ):
            raise SystemExit(f"source/instruction provenance differs: {cell.sample_id}")
        conditioning = source_row.get("source_conditioning")
        conditioning_digest = source_row.get("source_conditioning_digest")
        if not (
            isinstance(conditioning, dict)
            and contract.object_sha256(conditioning) == conditioning_digest
            and conditioning.get("sample_digest") == cell.sample_digest
            and conditioning.get("source_video_sha256") == cell.source_video_sha256
            and conditioning.get("edit_instruction_sha256")
            == cell.edit_instruction_sha256
            and conditioning.get("source_instruction_binding_digest")
            == cell.source_instruction_binding_digest
            and conditioning.get("source_frame_set_digest")
            == source_row.get("source_frame_set_digest")
        ):
            raise SystemExit(f"source conditioning digest differs: {cell.sample_id}")
        frame_artifact = source_row["source_frame_set_artifact"]
        if not (
            frame_artifact.get("frame_set_digest")
            == source_row.get("source_frame_set_digest")
            and frame_artifact.get("frame_order_consumed_by_carrier") is False
            and frame_artifact.get("full_video_latent_consumed_by_carrier") is False
            and len(frame_artifact.get("tensor_keys", [])) == 4
        ):
            raise SystemExit(f"source frame-set artifact differs: {cell.sample_id}")
        check_file(frame_artifact, "file_sha256")
    expected_keys = set(product(expected_source_ids, sealed.seed_order, runtime.NOISE_ARM_ORDER))
    observed_keys = set()
    matched_parent = {}
    rollout_rows_by_cell_seed = {}
    for row in receipt["rollouts"]:
        check_digest(row, "rollout_digest")
        key = (row["sample_id"], row["seed"], row["noise_arm"])
        if key in observed_keys:
            raise SystemExit(f"duplicate rollout: {key}")
        observed_keys.add(key)
        rollout_rows_by_cell_seed.setdefault((row["sample_id"], row["seed"]), []).append(row)
        arm = row["noise_arm"]
        rho = runtime.NOISE_RHO_BY_ARM[arm]
        if row.get("source_carrier_rho") != rho:
            raise SystemExit(f"arm/rho differs: {key}")
        operator_receipt = row["operator_receipt"]
        if contract.object_sha256(operator_receipt) != row["operator_receipt_digest"]:
            raise SystemExit(f"operator receipt digest differs: {key}")
        if not (
            operator_receipt.get("schema_version")
            == runtime.appearance_noise.SCHEMA_VERSION
            and operator_receipt.get("trainer_integration_executed") is False
            and operator_receipt.get("operator_self_registers_sampler_hook") is False
            and operator_receipt.get("operator_self_registers_launcher") is False
            and operator_receipt.get("ablation_only") is True
            and operator_receipt.get("scientific_claim_authorized") is False
        ):
            raise SystemExit(f"pure operator registration receipt differs: {key}")
        runtime_binding = row.get("operator_runtime_binding", {})
        if not (
            runtime_binding.get("callable")
            == "motion_null_appearance_noise.build_motion_null_appearance_noise"
            and runtime_binding.get("integration_owner")
            == "infer_oasis_phase_a_noise_bank._sample_with_oasis_noise_arm"
            and runtime_binding.get("official_randn_called_first") is True
            and runtime_binding.get("inference_integration_executed") is True
            and runtime_binding.get("operator_self_registered_sampler_hook") is False
        ):
            raise SystemExit(f"operator runtime binding differs: {key}")
        diagnostics = operator_receipt.get("diagnostics", {})
        if (
            diagnostics.get("rho") != rho
            or diagnostics.get("source_temporal_indices_consumed") is not False
            or diagnostics.get("source_temporal_phase_consumed") is not False
            or diagnostics.get("source_spatial_phase_consumed") is not False
            or diagnostics.get("carrier_strict_temporal_dc") is not True
        ):
            raise SystemExit(f"motion-null diagnostics differ: {key}")
        native_sampling = row.get("native_sampling", {})
        if not (
            native_sampling.get("num_frames") == 81
            and native_sampling.get("num_inference_steps") == 40
            and native_sampling.get("guidance_mode") == "rv2v"
            and native_sampling.get("guidance_policy")
            == "fixed_native_rv2v_no_ablation"
            and native_sampling.get("guidance_implementation_replaced") is False
            and native_sampling.get("sample_one_step_replaced") is False
            and native_sampling.get("scheduler_replaced") is False
        ):
            raise SystemExit(f"native sampling differs: {key}")
        parent = row["parent_official_gaussian_raw_value_sha256"]
        parent_key = (row["sample_id"], row["seed"])
        if parent_key in matched_parent and matched_parent[parent_key] != parent:
            raise SystemExit(f"matched parent Gaussian differs: {parent_key}")
        matched_parent[parent_key] = parent
        baseline = row["baseline_artifact"]
        injected = row["sampler_initial_noise_artifact"]
        if baseline.get("raw_value_sha256") != parent:
            raise SystemExit(f"parent artifact binding differs: {key}")
        check_file(baseline, "file_sha256")
        check_file(injected, "file_sha256")
        endpoint = row["endpoint"]
        check_file(endpoint, "sha256")
        check_file(endpoint["normalized_clean_latent"], "sha256")
        active = rho > 0.0
        if not active:
            if not (
                row.get("external_initial_noise_injection") is False
                and row.get("rho_zero_exact_native_object_forwarded") is True
                and baseline["raw_value_sha256"] == injected["raw_value_sha256"]
                and diagnostics.get("rho_zero_exact_object_alias") is True
            ):
                raise SystemExit(f"rho0 exact-native control differs: {key}")
        elif not (
            row.get("external_initial_noise_injection") is True
            and row.get("rho_zero_exact_native_object_forwarded") is False
            and row.get("active_noise_parent_matches_official_control") is True
            and baseline["raw_value_sha256"] != injected["raw_value_sha256"]
            and diagnostics.get("source_conditioned_non_gaussian") is True
        ):
            raise SystemExit(f"active source carrier differs: {key}")
        if any(
            row.get(field) is not expected
            for field, expected in (
                ("endpoint_candidate_only", True),
                ("legacy_pair_v5_native_rollout_schema_compatible", False),
                ("external_action_scorer_consumed", False),
                ("action_source_scoring_performed", False),
                ("endpoint_selection_performed", False),
                ("optimizer_or_training_authorized", False),
                ("training_performed", False),
                ("scientific_action_editing_success_claim", False),
            )
        ):
            raise SystemExit(f"rollout authority differs: {key}")
        all_candidate_ids.add(row["candidate_id"])
    if observed_keys != expected_keys:
        raise SystemExit(f"exact12 family coverage differs: {family}")
    declared_audits = receipt.get("matched_triplet_audits", [])
    audit_by_key = {
        (row.get("sample_id"), row.get("seed")): row for row in declared_audits
    }
    if len(audit_by_key) != 4 or len(declared_audits) != 4:
        raise SystemExit(f"matched-triplet audit coverage differs: {family}")
    for cell in expected_cells:
        source_row = source_row_by_id[cell.sample_id]
        for seed in sealed.seed_order:
            audit = core.validate_matched_rollout_triplet(
                rollout_rows_by_cell_seed[(cell.sample_id, seed)],
                sample_id=cell.sample_id,
                sample_digest=cell.sample_digest,
                source_video_sha256=cell.source_video_sha256,
                edit_instruction_sha256=cell.edit_instruction_sha256,
                source_conditioning_digest=source_row["source_conditioning_digest"],
                source_frame_set_digest=source_row["source_frame_set_digest"],
                family=cell.family,
                analysis_split=cell.analysis_split,
                seed=seed,
            )
            declared_audit = audit_by_key[(cell.sample_id, seed)]
            if declared_audit != {
                "sample_id": audit.sample_id,
                "sample_digest": audit.sample_digest,
                "seed": audit.seed,
                "candidate_ids": list(audit.candidate_ids),
                "audit_digest": audit.audit_digest,
            }:
                raise SystemExit(
                    f"matched source/instruction/seed audit differs: {cell.sample_id}:{seed}"
                )
    all_source_ids.update(observed_source_ids)
    family_receipt_digests[family] = receipt["receipt_digest"]
    family_receipt_file_sha256[family] = contract.file_sha256(receipt_path)
    all_triplet_audit_digests.update(
        row["audit_digest"] for row in receipt["matched_triplet_audits"]
    )

if (
    len(all_candidate_ids) != 24
    or len(all_triplet_audit_digests) != 8
    or all_source_ids != {cell.sample_id for cell in sealed.samples}
):
    raise SystemExit("dual-SP4 exact24/four-source coverage differs")
unsigned = {
    "schema_version": "bernini-oasis-phase-a-source-set-noise-dual-sp4-stage-v2",
    "manifest_file_sha256": manifest_sha,
    "manifest_digest": sealed.manifest_digest,
    "family_order": list(contract.FAMILY_ORDER),
    "family_receipt_digests": family_receipt_digests,
    "family_receipt_file_sha256": family_receipt_file_sha256,
    "method_source_revision": source_revision,
    "method_source_archive_sha256": source_archive_sha,
    "checkpoint_tree_sha256": checkpoint_tree_sha,
    "checkpoint_content_manifest_file_sha256": checkpoint_manifest_sha,
    "source_cell_count": 4,
    "candidate_count": 24,
    "candidate_id_set_digest": contract.object_sha256(sorted(all_candidate_ids)),
    "matched_triplet_count": 8,
    "matched_triplet_audit_set_digest": contract.object_sha256(
        sorted(all_triplet_audit_digests)
    ),
    "topology": "two_concurrent_independent_WORLD4_Ulysses_SP4_on_all8",
    "dedicated_launcher_executed": True,
    "guidance_policy": "fixed_native_rv2v_no_ablation",
    "endpoint_scoring_performed": False,
    "endpoint_selection_performed": False,
    "external_action_scorer_consumed": False,
    "optimizer_or_training_authorized": False,
    "training_performed": False,
    "scientific_action_editing_success_claim": False,
}
stage = {**unsigned, "receipt_digest": contract.object_sha256(unsigned)}
stage_path = root / "dual-sp4-receipt.json"
payload = contract.canonical_json_bytes(stage) + b"\n"
with stage_path.open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
print("BERNINI_OASIS_SOURCE_SET_NOISE_DUAL_SP4_EXACT24_OK")
PY

echo "[oasis-noise-dual4] COMPLETE candidates=24 source_cells=4 arms=official+.05+.10 guidance=fixed-native-rv2v endpoint_scorer=false training=false success_claim=false"
