#!/usr/bin/env bash
#SBATCH --job-name=bernini-srcset-noise-2x4
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One allocation, two isolated WORLD4/SP4 groups.  GPUs 0-3 decode the fixed
# dog cell and GPUs 4-7 decode the fixed human cell.  Each group renders the
# same five exact40/exact81 arms; no arm is scored, ranked, or selected.

source_archive="${SOURCE_SET_NOISE_SOURCE_ARCHIVE:?set source archive}"
source_archive_sha256="${SOURCE_SET_NOISE_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
source_revision="${SOURCE_SET_NOISE_SOURCE_REVISION:?set source revision}"
cell_spec="${SOURCE_SET_NOISE_CELL_SPEC:?set sealed core2 spec}"
cell_spec_sha256="${SOURCE_SET_NOISE_CELL_SPEC_SHA256:?set spec SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${SOURCE_SET_NOISE_OUTPUT_ROOT:?set fresh output root}"
python_bin="${SOURCE_SET_NOISE_PYTHON_BIN:?set vace Python executable}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
authoring_spec_sha256="204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"

fail() {
  echo "[source-set-noise-dual4] ERROR: $*" >&2
  exit 2
}

absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for name in \
  source_archive cell_spec bernini_root veomni_root checkpoint \
  checkpoint_manifest output_root python_bin
do
  absolute "${!name}" "${name}"
done
for name in \
  source_archive_sha256 cell_spec_sha256 checkpoint_manifest_sha256 \
  checkpoint_tree_sha256 authoring_spec_sha256
do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"

source_archive="$(realpath -e -- "${source_archive}")"
cell_spec="$(realpath -e -- "${cell_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

for path in "${source_archive}" "${cell_spec}" "${checkpoint_manifest}" "${python_bin}"
do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain file required: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"
do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain directory required: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || \
  fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || \
  fail "source archive revision differs"
[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || \
  fail "cell spec changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || \
  fail "checkpoint manifest changed"

[[ "${output_root}" != / ]] || fail "output cannot be root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || \
  fail "unsafe output basename"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || \
  fail "output path is not canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output must be fresh"

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
export NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || \
  fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/source-set-noise-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/source-set-noise-${SLURM_JOB_ID}."*) ;;
    *) cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${status}" == 0 ]]; then status=2; fi
  if [[ "${status}" == 0 ]]; then
    echo "[source-set-noise-dual4] PASS all8=true topology=dog-SP4+human-SP4 exact81=true exact40=true arms=5 selection=false output=${output_root}"
  else
    echo "[source-set-noise-dual4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${dog_pid}" "${human_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
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
  fail "scratch source archive changed"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_orderless_source_frame_set_noise_canary.py",
    "methods/bernini_action_editing/orderless_source_frame_set_noise.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/tools/build_renderer_dataset.py",
    "methods/bernini_action_editing/tools/materialize_vae.py",
    "methods/bernini_action_editing/assets/orderless_source_frame_set_noise_core2_v1.json",
    "methods/bernini_action_editing/assets/pair_v5_t2v_calibration_first8_authoring_v1.json",
    "methods/bernini_action_editing/scripts/auh_infer_orderless_source_frame_set_noise_canary_dual4.sbatch",
    "methods/bernini_action_editing/tests/test_orderless_source_frame_set_noise.py",
    "methods/bernini_action_editing/tests/test_infer_orderless_source_frame_set_noise_canary.py",
    "methods/bernini_action_editing/tests/test_auh_infer_orderless_source_frame_set_noise_canary_launcher.py",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("source archive member escaped repository")
        normalized = path.as_posix().lstrip("./")
        if normalized.startswith("methods/bernini_action_editing/"):
            if member.issym() or member.islnk() or member.isfifo() or member.isdev():
                raise SystemExit("method closure contains link/device")
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks source-set canary closure: {missing}")
PY

tar --no-same-owner --no-same-permissions \
  -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_orderless_source_frame_set_noise_canary.py"
staged_launcher="${method_root}/scripts/auh_infer_orderless_source_frame_set_noise_canary_dual4.sbatch"
staged_spec="${method_root}/assets/orderless_source_frame_set_noise_core2_v1.json"
staged_authoring_spec="${method_root}/assets/pair_v5_t2v_calibration_first8_authoring_v1.json"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher absent"
[[ -f "${staged_spec}" && ! -L "${staged_spec}" ]] || fail "staged spec absent"
[[ -f "${staged_authoring_spec}" && ! -L "${staged_authoring_spec}" ]] || \
  fail "staged authoring spec absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_sha256}" ]] || \
  fail "executed launcher differs from archive"
[[ "$(sha256sum "${staged_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || \
  fail "archive spec differs"
[[ "$(sha256sum "${staged_authoring_spec}" | awk '{print $1}')" == "${authoring_spec_sha256}" ]] || \
  fail "archive authoring spec differs"

for test_file in \
  test_orderless_source_frame_set_noise.py \
  test_infer_orderless_source_frame_set_noise_canary.py \
  test_auh_infer_orderless_source_frame_set_noise_canary_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
cp -- "${cell_spec}" "${output_root}/sealed-core2-spec.json"
chmod 0444 -- "${output_root}/sealed-core2-spec.json"
cp -- "${staged_authoring_spec}" "${output_root}/sealed-first8-authoring-spec.json"
chmod 0444 -- "${output_root}/sealed-first8-authoring-spec.json"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((34000 + 2 * job_mod))
human_port=$((dog_port + 1))

launch_group() (
  set -Eeuo pipefail
  local label="$1"
  local visible_gpus="$2"
  local master_port="$3"
  local group_scratch="${task_scratch}/groups/${label}"
  local group_output="${output_root}/${label}"
  mkdir -p -- \
    "${group_scratch}/cache/miopen-user" \
    "${group_scratch}/cache/miopen-custom" \
    "${group_scratch}/cache/torch-extensions" \
    "${group_scratch}/cache/triton" \
    "${group_scratch}/cache/xdg" \
    "${group_scratch}/cache/pycache" \
    "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/cache/triton"
  export XDG_CACHE_HOME="${group_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/cache/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic.json"
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${label}-source-set-noise"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --cell-spec "${output_root}/sealed-core2-spec.json" \
    --expected-cell-spec-sha256 "${cell_spec_sha256}" \
    --authoring-spec "${output_root}/sealed-first8-authoring-spec.json" \
    --expected-authoring-spec-sha256 "${authoring_spec_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --output-dir "${group_output}" \
    --num-inference-steps 40 \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}"
)

echo "[source-set-noise-dual4] allocation=8xMI210 dog=0,1,2,3 human=4,5,6,7 arms=official+correct/wrong-rho005+correct/wrong-rho010"
launch_group dog 0,1,2,3 "${dog_port}" \
  >"${output_root}/logs/dog.out" 2>"${output_root}/logs/dog.err" &
dog_pid=$!
launch_group human 4,5,6,7 "${human_port}" \
  >"${output_root}/logs/human.out" 2>"${output_root}/logs/human.err" &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || \
  fail "WORLD4 group failed: dog=${dog_status} human=${human_status}"

"${python_bin}" -B - \
  "${output_root}" "${source_revision}" "${source_archive_sha256}" \
  "${launcher_sha256}" "${cell_spec_sha256}" "${authoring_spec_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
revision, archive_sha, launcher_sha, spec_sha, authoring_sha = sys.argv[2:]
expected_arms = [
    "official_gaussian",
    "correct_source_rho005",
    "wrong_source_rho005",
    "correct_source_rho010",
    "wrong_source_rho010",
]
expected_gate_names = [
    "same_rho_correct_vs_wrong_carrier_source_specificity",
    "old_motion_direction_and_order_leakage_nonincrease",
    "target_action_nonregression_vs_official_gaussian",
    "identity_blur_flicker_camera_quality_nonregression_vs_official_gaussian",
]
sha256_pattern = re.compile(r"[0-9a-f]{64}")


def canonical_object_digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_child_artifacts(receipt, *, label, child_root, expected_arms):
    rows = []
    outputs = receipt["outputs"]
    for arm in expected_arms:
        output = outputs[arm]
        rows.append((f"output/{arm}/mp4", output, "sha256"))
        rows.append(
            (
                f"output/{arm}/normalized-clean-latent",
                output["normalized_clean_latent"],
                "sha256",
            )
        )
    rows.append(
        (
            "official-parent-gaussian",
            receipt["official_parent_gaussian"],
            "file_sha256",
        )
    )
    frame_sets = receipt["source_conditioning"]["frame_set_artifacts"]
    for role in ("correct", "wrong"):
        rows.append((f"frame-set/{role}", frame_sets[role], "file_sha256"))
    for arm_row in receipt["arms"]:
        arm = arm_row["arm"]
        rows.append(
            (
                f"arm/{arm}/sampler-initial-noise",
                arm_row["sampler_initial_noise_artifact"],
                "file_sha256",
            )
        )
        prototype = arm_row["source_set_prototype_artifact"]
        carrier = arm_row["temporal_dc_carrier_artifact"]
        if arm == "official_gaussian":
            if prototype is not None or carrier is not None:
                raise SystemExit(f"{label} rho0 unexpectedly has carrier artifacts")
        else:
            if not isinstance(prototype, dict) or not isinstance(carrier, dict):
                raise SystemExit(f"{label}/{arm} active carrier artifacts absent")
            rows.append((f"arm/{arm}/source-set-prototype", prototype, "file_sha256"))
            rows.append((f"arm/{arm}/temporal-dc-carrier", carrier, "file_sha256"))
    if len(rows) != 26:
        raise SystemExit(f"{label} generated artifact closure differs: {len(rows)}")

    child_root = child_root.resolve(strict=True)
    verified = {}
    seen_paths = set()
    for artifact_label, node, hash_key in rows:
        if not isinstance(node, dict):
            raise SystemExit(f"{label}/{artifact_label} receipt node differs")
        raw_path = node.get("path")
        expected_sha = node.get(hash_key)
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or not isinstance(expected_sha, str)
            or sha256_pattern.fullmatch(expected_sha) is None
        ):
            raise SystemExit(f"{label}/{artifact_label} path/hash declaration differs")
        requested = Path(raw_path)
        if not requested.is_file() or requested.is_symlink():
            raise SystemExit(f"{label}/{artifact_label} is not a plain file")
        resolved = requested.resolve(strict=True)
        if resolved != requested:
            raise SystemExit(f"{label}/{artifact_label} path is not canonical")
        try:
            resolved.relative_to(child_root)
        except ValueError as error:
            raise SystemExit(
                f"{label}/{artifact_label} escaped child output root"
            ) from error
        if str(resolved) in seen_paths:
            raise SystemExit(f"{label}/{artifact_label} aliases another artifact")
        observed_sha = file_sha256(resolved)
        if observed_sha != expected_sha:
            raise SystemExit(f"{label}/{artifact_label} SHA-256 differs")
        seen_paths.add(str(resolved))
        verified[artifact_label] = {
            "path": str(resolved),
            "sha256": observed_sha,
        }
    return {
        "verified_artifact_count": len(verified),
        "verified_artifact_set_digest": canonical_object_digest(verified),
    }


children = {}
for label in ("dog", "human"):
    path = root / label / "receipt.json"
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed_receipt_digest = receipt.get("receipt_digest")
    unsigned_child = dict(receipt)
    unsigned_child.pop("receipt_digest", None)
    if (
        not isinstance(claimed_receipt_digest, str)
        or sha256_pattern.fullmatch(claimed_receipt_digest) is None
        or canonical_object_digest(unsigned_child) != claimed_receipt_digest
    ):
        raise SystemExit(f"{label} receipt object digest differs")
    sampling = receipt.get("sampling", {})
    interpretation = receipt.get("interpretation", {})
    audit = receipt.get("matched_five_arm_audit", {})
    source_conditioning = receipt.get("source_conditioning", {})
    authoring = receipt.get("upstream_authoring_provenance", {})
    gate_contract = receipt.get("scientific_gate_contract", {})
    gates = gate_contract.get("gates", [])
    arms = receipt.get("arms", [])
    outputs = receipt.get("outputs", {})
    if not (
        receipt.get("schema_version")
        == "bernini-orderless-source-frame-set-noise-canary-receipt-v1"
        and receipt.get("stage") == "isolated_exact40_exact81_five_arm_canary"
        and receipt.get("cell_spec", {}).get("file_sha256") == spec_sha
        and receipt.get("cell_spec", {}).get("cell", {}).get("cell_id") == label
        and authoring.get("file_sha256") == authoring_sha
        and authoring.get("same_action_family_id_verified") is True
        and receipt.get("arm_order") == expected_arms
        and [row.get("arm") for row in arms] == expected_arms
        and len(arms) == 5
        and set(outputs) == set(expected_arms)
        and sampling.get("exact40") is True
        and sampling.get("exact81") is True
        and sampling.get("frame_count") == 81
        and sampling.get("num_inference_steps") == 40
        and sampling.get("same_source_action_seed_scheduler_guidance_across_five_arms") is True
        and audit.get("one_official_parent_gaussian_across_five_arms") is True
        and audit.get("correct_and_wrong_carriers_differ") is True
        and audit.get("wrong_source_is_carrier_only") is True
        and audit.get("wrong_source_is_action_family_matched_source_specificity_control") is True
        and audit.get("wrong_source_is_pure_identity_control") is False
        and source_conditioning.get("caller_selection_indices_consumed") is True
        and source_conditioning.get("operator_received_frame_indices") is False
        and source_conditioning.get("operator_set_sequence_order_consumed") is False
        and source_conditioning.get("unordered_pose_occupancy_leakage_possible") is True
        and source_conditioning.get("wrong_source_is_pure_identity_control") is False
        and gate_contract.get("identity_orbit_decoded_role") == "excluded_external_no_go"
        and gate_contract.get("identity_orbit_adapter_loaded") is False
        and gate_contract.get("identity_adapter_loaded") is False
        and gate_contract.get("identity_adapter_used_as_baseline") is False
        and gate_contract.get("identity_adapter_used_as_prior") is False
        and gate_contract.get("all_gates_evaluated") is False
        and gate_contract.get("arm_selection_authorized") is False
        and [row.get("name") for row in gates] == expected_gate_names
        and all(row.get("evaluated") is False for row in gates)
        and interpretation.get("candidate_count") == 5
        and interpretation.get("training_performed") is False
        and interpretation.get("training_harness_executed") is False
        and interpretation.get("validation_only_train_lora_module_imported") is True
        and interpretation.get("trainer_instantiated") is False
        and interpretation.get("critic_loaded") is False
        and interpretation.get("reward_computed") is False
        and interpretation.get("best_arm_selected") is False
        and interpretation.get("ranking_performed") is False
        and interpretation.get("source_pose_occupancy_leakage_possible") is True
        and receipt.get("runtime_source", {}).get("revision") == revision
        and receipt.get("runtime_source", {}).get("archive_sha256") == archive_sha
        and receipt.get("runtime_source", {}).get("launcher_sha256") == launcher_sha
    ):
        raise SystemExit(f"{label} receipt contract differs")
    if label == "human" and not (
        source_conditioning.get("wrong_source_geometry_confound_present") is True
        and interpretation.get(
            "human_wrong_source_geometry_and_initial_pose_confound_acknowledged"
        ) is True
        and source_conditioning.get("wrong_source_metadata", {}).get(
            "native_bucket_matches_target_cell_bucket"
        ) is False
        and source_conditioning.get("wrong_source_metadata", {}).get(
            "direct_resize_anisotropic_aspect_distortion_fraction", 0.0
        ) > 0.01
    ):
        raise SystemExit("human source-specificity control confound is not sealed")
    artifact_verification = verify_child_artifacts(
        receipt,
        label=label,
        child_root=root / label,
        expected_arms=expected_arms,
    )
    parent_hashes = {
        row.get("official_parent_gaussian_raw_value_sha256") for row in arms
    }
    if len(parent_hashes) != 1:
        raise SystemExit(f"{label} five arms lack one official Gaussian")
    for arm, output in outputs.items():
        video = Path(output.get("path", ""))
        if (
            arm not in expected_arms
            or not video.is_file()
            or video.is_symlink()
            or output.get("frame_count") != 81
            or output.get("fps") != 25
        ):
            raise SystemExit(f"{label}/{arm} output differs")
    raw = path.read_bytes()
    children[label] = {
        "receipt_path": str(path),
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_digest": claimed_receipt_digest,
        **artifact_verification,
        "official_parent_gaussian_raw_value_sha256": next(iter(parent_hashes)),
        "candidate_count": 5,
        "world_size": 4,
        "ulysses_size": 4,
    }

unsigned = {
    "schema_version": "bernini-orderless-source-frame-set-noise-all8-dual4-receipt-v1",
    "method": "frozen-bernini-orderless-source-frame-set-noise-canary-all8",
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "launcher_sha256": launcher_sha,
    "cell_spec_sha256": spec_sha,
    "upstream_authoring_spec_sha256": authoring_sha,
    "topology": {
        "allocated_gpu_count": 8,
        "groups": {
            "dog": {"visible_gpus": [0, 1, 2, 3], "world_size": 4, "ulysses_size": 4},
            "human": {"visible_gpus": [4, 5, 6, 7], "world_size": 4, "ulysses_size": 4},
        },
        "groups_ran_concurrently": True,
    },
    "arm_order": expected_arms,
    "total_candidate_count": 10,
    "children": children,
    "training_performed": False,
    "training_harness_executed": False,
    "validation_only_train_lora_module_imported": True,
    "trainer_instantiated": False,
    "critic_loaded": False,
    "reward_computed": False,
    "ranking_performed": False,
    "best_arm_selected": False,
    "identity_orbit_or_identity_adapter_baseline_or_prior": False,
    "scientific_gates_executed": False,
    "scientific_claim_authorized": False,
}
payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
receipt = {**unsigned, "receipt_digest": hashlib.sha256(payload).hexdigest()}
target = root / "all8-receipt.json"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
descriptor = os.open(target, flags, 0o444)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
except Exception:
    try:
        os.close(descriptor)
    except OSError:
        pass
    raise
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
PY

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || \
  fail "source archive changed during run"
[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || \
  fail "cell spec changed during run"
[[ "$(sha256sum "${output_root}/sealed-core2-spec.json" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || \
  fail "sealed output cell spec changed during run"
[[ "$(sha256sum "${output_root}/sealed-first8-authoring-spec.json" | awk '{print $1}')" == "${authoring_spec_sha256}" ]] || \
  fail "sealed output authoring spec changed during run"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || \
  fail "checkpoint manifest changed during run"
