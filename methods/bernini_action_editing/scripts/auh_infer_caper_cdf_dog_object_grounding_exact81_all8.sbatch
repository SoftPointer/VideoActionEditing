#!/usr/bin/env bash
#SBATCH --job-name=caper-cdf-dog-grounding
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=05:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# One preregistered wave uses the whole node concurrently.  The historical
# seed is WORLD4/SP4 on GPUs 0-3 and the fresh seed is WORLD4/SP4 on GPUs 4-7.
# These are same-source seed replicates, not independent identity samples.

repo_root="${CAPER_CDF_DOG_REPO_ROOT:?set clean repository root}"
source_archive="${CAPER_CDF_DOG_SOURCE_ARCHIVE:?set git archive}"
source_archive_sha256="${CAPER_CDF_DOG_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
registry="${CAPER_CDF_DOG_REGISTRY:?set sealed CDF dog registry}"
registry_sha256="${CAPER_CDF_DOG_REGISTRY_SHA256:?set registry SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${CAPER_CDF_DOG_OUTPUT_ROOT:?set fresh output root}"
python_bin="${CAPER_CDF_DOG_PYTHON_BIN:?set vace Python executable}"
source_video="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4"
source_video_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
canonical_registry_sha256="f91327227384d4d29308d43895fe71d2fa9b4666438b9ec99bf6c65e7b7283c8"

fail() {
  echo "[caper-cdf-dog-object-grounding] ERROR: $*" >&2
  exit 2
}

for value in \
  "${repo_root}" "${source_archive}" "${registry}" "${bernini_root}" \
  "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" \
  "${output_root}" "${python_bin}" "${source_video}"
do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for value in "${source_archive_sha256}" "${registry_sha256}" "${source_video_sha256}"
do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 input differs"
done
[[ "${registry_sha256}" == "${canonical_registry_sha256}" ]] || fail "canonical registry digest differs"

repo_root="$(realpath -e -- "${repo_root}")"
source_archive="$(realpath -e -- "${source_archive}")"
registry="$(realpath -e -- "${registry}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -d "${repo_root}" && ! -L "${repo_root}" ]] || fail "repository root differs"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${registry}" && ! -L "${registry}" ]] || fail "registry differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python runtime differs"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "source video path differs"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "actual source video bytes differ"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/infer_caper_cdf_dog_object_grounding_exact81_canary.py"
repo_registry="${method_root}/assets/caper_cdf_dog_object_grounding_exact81_v1.json"
repo_launcher="${method_root}/scripts/auh_infer_caper_cdf_dog_object_grounding_exact81_all8.sbatch"
running_launcher="$(realpath -e -- "$0")"
repo_closure=(
  "${runtime}"
  "${repo_registry}"
  "${method_root}/infer_t2v_v2v_branch_homotopy_canary.py"
  "${method_root}/t2v_v2v_branch_homotopy_v1.py"
  "${method_root}/t2v_v2v_branch_homotopy_runtime_v1.py"
  "${method_root}/infer_native_branch_homotopy_canary.py"
  "${method_root}/native_branch_homotopy_v1.py"
  "${method_root}/native_branch_homotopy_runtime_v1.py"
  "${method_root}/infer_native_identity_generation_canary.py"
  "${method_root}/infer_native_self_guided_action_field_canary.py"
  "${method_root}/self_guided_action_field_v1.py"
  "${method_root}/source_self_native_ref_contrastive_v3.py"
  "${method_root}/tri_branch_unipc.py"
  "${method_root}/infer_lora.py"
  "${method_root}/train_lora.py"
  "${method_root}/infer_source_kv_carrier_oracle.py"
  "${method_root}/infer_source_value_residual_oracle.py"
  "${method_root}/source_kv_replay.py"
  "${method_root}/source_kv_route_batches.py"
  "${method_root}/source_value_residual.py"
  "${method_root}/assets/t2v_v2v_branch_homotopy_core4_v1.json"
  "${method_root}/assets/native_branch_homotopy_core4_v1.json"
  "${method_root}/assets/wrong_family_prompt_swap_pilot_registry_v1.json"
  "${method_root}/tests/test_t2v_v2v_branch_homotopy_v1.py"
  "${method_root}/tests/test_t2v_v2v_branch_homotopy_runtime_v1.py"
  "${method_root}/tests/test_infer_t2v_v2v_branch_homotopy_canary.py"
  "${method_root}/tests/test_infer_caper_cdf_dog_object_grounding_exact81_canary.py"
  "${method_root}/tests/test_auh_infer_caper_cdf_dog_object_grounding_exact81_launcher.py"
  "${repo_launcher}"
)
closure=("${repo_closure[@]}" "${registry}")
for path in "${closure[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source closure file missing: ${path}"
done
[[ "$(sha256sum "${running_launcher}" | awk '{print $1}')" == "$(sha256sum "${repo_launcher}" | awk '{print $1}')" ]] || fail "running launcher differs"
[[ "$(sha256sum "${repo_registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "repository registry differs"

source_revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "archive/repository revision differs"
for path in "${repo_closure[@]}"; do
  relative="${path#"${repo_root}/"}"
  [[ "${relative}" != "${path}" ]] || fail "repository closure path escaped"
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null || fail "closure path is untracked: ${relative}"
done
git -C "${repo_root}" diff --quiet -- "${repo_closure[@]}" || fail "runtime closure has unstaged edits"
git -C "${repo_root}" diff --cached --quiet -- "${repo_closure[@]}" || fail "runtime closure has staged edits"
launcher_sha256="$(sha256sum "${repo_launcher}" | awk '{print $1}')"
source_closure_sha256="$(sha256sum "${closure[@]}" | sha256sum | awk '{print $1}')"

[[ "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output path is not canonical"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe output basename"

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

for test_file in \
  test_t2v_v2v_branch_homotopy_v1.py \
  test_t2v_v2v_branch_homotopy_runtime_v1.py \
  test_infer_t2v_v2v_branch_homotopy_canary.py \
  test_infer_caper_cdf_dog_object_grounding_exact81_canary.py \
  test_auh_infer_caper_cdf_dog_object_grounding_exact81_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/caper-cdf-dog-${SLURM_JOB_ID}.XXXXXX")"
historical_pid=""
fresh_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/caper-cdf-dog-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[caper-cdf-dog-object-grounding] PASS all8=true one-wave=true exact81=true exact40=true automatic-success=false output=${output_root}"
  fi
  exit "${status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${historical_pid}" "${fresh_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${historical_pid}" "${fresh_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
cp -- "${registry}" "${output_root}/sealed-caper-cdf-dog-object-grounding-exact81-v1.json"
chmod 0444 -- "${output_root}/sealed-caper-cdf-dog-object-grounding-exact81-v1.json"

job_mod=$((SLURM_JOB_ID % 10000))
historical_port=$((40000 + 2 * job_mod))
fresh_port=$((historical_port + 1))

launch_group() (
  set -Eeuo pipefail
  local label="$1"
  local visible="$2"
  local port="$3"
  local group_scratch="${task_scratch}/${label}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" "${group_scratch}/torch-extensions" "${group_scratch}/triton" "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
  export TMPDIR="${group_scratch}/tmp"
  export BERNINI_OUTPUT_TRANSACTION_ID="caper-cdf-dog-${SLURM_JOB_ID}-${label}"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" \
    --registry "${output_root}/sealed-caper-cdf-dog-object-grounding-exact81-v1.json" \
    --expected-registry-sha256 "${registry_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_root}/${label}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --runtime-source-closure-sha256 "${source_closure_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}"
)

launch_group "cdf-dog-historical-seed-2027" 0,1,2,3 "${historical_port}" >"${output_root}/logs/cdf-dog-historical-seed-2027.log" 2>&1 &
historical_pid=$!
launch_group "cdf-dog-fresh-seed-2026081701" 4,5,6,7 "${fresh_port}" >"${output_root}/logs/cdf-dog-fresh-seed-2026081701.log" 2>&1 &
fresh_pid=$!
historical_status=0
fresh_status=0
wait "${historical_pid}" || historical_status=$?
wait "${fresh_pid}" || fresh_status=$?
historical_pid=""
fresh_pid=""
[[ "${historical_status}" == 0 ]] || fail "historical-seed WORLD4/SP4 failed with ${historical_status}"
[[ "${fresh_status}" == 0 ]] || fail "fresh-seed WORLD4/SP4 failed with ${fresh_status}"

"${python_bin}" -B - "${output_root}" "${registry_sha256}" "${source_revision}" "${source_archive_sha256}" "${source_closure_sha256}" "${launcher_sha256}" "${source_video}" "${source_video_sha256}" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import av
import torch
from safetensors import safe_open

root = Path(sys.argv[1]).resolve(strict=True)
registry_sha, revision, archive_sha, closure_sha, launcher_sha = sys.argv[2:7]
source = Path(sys.argv[7])
source_sha = sys.argv[8]
arms = [
    "native-source-video-only-v2v-endpoint",
    "pure-target-only-t2v-endpoint",
    "t2v-v2v-branch-homotopy-095-075",
]
cells = [
    ("cdf-dog-historical-seed-2027", 2027, "historical_comparison_seed"),
    ("cdf-dog-fresh-seed-2026081701", 2026081701, "fresh_preregistered_seed"),
]
caption = "Make the dog pick up the bone and hold it in its mouth."
schedule_sha = "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
event_stages = ["approach", "contact", "grip", "lift", "hold"]
correspondences = [
    "source_dog_identity",
    "source_bone_identity",
    "source_dog_mouth_anatomical_identity",
]

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def object_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()

def plain_child_file(cell_root: Path, declared: object, name: str) -> Path:
    requested = Path(declared) if isinstance(declared, str) else Path(".")
    expected = cell_root / name
    if not requested.is_absolute() or requested != expected or requested.is_symlink() or requested.resolve(strict=True) != requested or not requested.is_file():
        raise SystemExit(f"{cell_root.name}/{name} path differs")
    return requested

def video_contract(path: Path) -> dict[str, object]:
    frames = 0
    observed_hw = set()
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise SystemExit(f"{path} stream count differs")
        stream = streams[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        for frame in container.decode(stream):
            frames += 1
            observed_hw.add((int(frame.height), int(frame.width)))
    if frames != 81 or abs(fps - 25.0) > 1.0e-6 or observed_hw != {(496, 480)}:
        raise SystemExit(f"{path} exact81 media differs")
    return {"decoded_frame_count": frames, "decoded_fps": fps, "decoded_hw": [496, 480]}

def tensor_raw_sha(path: Path, key: str) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [key]:
            raise SystemExit(f"{path} key differs")
        tensor = opened.get_tensor(key).contiguous()
    if tensor.dtype != torch.float32 or list(tensor.shape) != [1, 16, 21, 62, 60]:
        raise SystemExit(f"{path} tensor geometry differs")
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()

if source.is_symlink() or source.resolve(strict=True) != source or file_sha(source) != source_sha:
    raise SystemExit("actual source path/hash differs in postflight")

children = []
for cell, seed, role in cells:
    cell_root = root / cell
    receipt_path = cell_root / "receipt.json"
    if receipt_path.is_symlink() or receipt_path.resolve(strict=True) != receipt_path:
        raise SystemExit(f"{cell} receipt path differs")
    row = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(row)
    declared_digest = unsigned.pop("receipt_digest", None)
    if object_sha(unsigned) != declared_digest:
        raise SystemExit(f"{cell} receipt seal differs")
    input_row = row.get("input", {})
    sampling = row.get("sampling", {})
    revisions = row.get("source_revisions", {})
    if (
        row.get("schema_version") != "bernini-caper-cdf-dog-object-grounding-exact81-receipt-v1"
        or row.get("method") != "frozen-bernini-caper-cdf-dog-object-grounding-exact81-canary"
        or row.get("cell_id") != cell
        or row.get("wave_id") != "wave1-same-source-seed-replication"
        or row.get("cohort") != "cdf_dog_same_source_seed_replication"
        or input_row.get("source_video") != str(source)
        or input_row.get("source_video_sha256") != source_sha
        or input_row.get("target_action_caption") != caption
        or input_row.get("target_video") is not False
        or input_row.get("custom_initial_noise") is not False
        or input_row.get("mask_track_pose_flow") is not False
        or row.get("training_performed") is not False
        or row.get("optimizer_created") is not False
        or row.get("parameter_update") is not False
        or row.get("scientific_or_action_editing_claim_authorized") is not False
        or row.get("automatic_object_event_success_claim_authorized") is not False
        or row.get("same_source_replication_identity_aggregation_authorized") is not False
        or row.get("single_example_conclusion_authorized") is not False
        or revisions.get("runtime_method") != revision
        or revisions.get("runtime_source_archive_sha256") != archive_sha
        or revisions.get("runtime_source_closure_sha256") != closure_sha
        or revisions.get("launcher_source_sha256") != launcher_sha
    ):
        raise SystemExit(f"{cell} receipt boundary differs")
    replication = row.get("replication_design", {})
    grounding = row.get("object_grounding_evaluation", {})
    event = grounding.get("ordered_object_event_gate", {})
    correspondence = grounding.get("source_correspondence_gate", {})
    if (
        replication.get("replicate_role") != role
        or replication.get("same_source_iid") != "cdf-dog-288545b9c031491a"
        or replication.get("same_source_cells") != [name for name, _, _ in cells]
        or replication.get("same_source_cells_are_seed_replicates_not_independent_identities") is not True
        or replication.get("aggregate_as_independent_identities_authorized") is not False
        or replication.get("seed_search_or_posthoc_selection_authorized") is not False
        or grounding.get("gate_status") != "not_automatically_adjudicated"
        or grounding.get("observed_outcome_recorded") is not False
        or grounding.get("automatic_success_claim") is not False
        or grounding.get("manual_review_required") is not True
        or grounding.get("receipt_records_contract_not_outcome") is not True
        or grounding.get("automatic_success_claim_authorized") is not False
        or grounding.get("single_example_success_claim_authorized") is not False
        or event.get("required_stage_order") != event_stages
        or event.get("automatic_adjudication_performed") is not False
        or event.get("automatic_success_claim_authorized") is not False
        or correspondence.get("required_correspondences") != correspondences
        or correspondence.get("automatic_adjudication_performed") is not False
        or correspondence.get("automatic_success_claim_authorized") is not False
    ):
        raise SystemExit(f"{cell} object-grounding disclosure differs")
    schedule = sampling.get("schedule_receipt", {})
    if (
        sampling.get("frame_count") != 81
        or sampling.get("latent_phases") != 21
        or sampling.get("num_inference_steps") != 40
        or sampling.get("fps") != 25
        or sampling.get("seed") != seed
        or sampling.get("arm_order") != arms
        or sampling.get("same_official_gaussian_all_arms") is not True
        or sampling.get("flow_shift_from_renderer_config") != 5.0
        or sampling.get("schedule_sha256") != schedule_sha
        or schedule.get("digest") != schedule_sha
        or sampling.get("high_endpoint_step_indices") != list(range(9))
        or sampling.get("transition_step_indices") != list(range(9, 26))
        or sampling.get("low_endpoint_step_indices") != list(range(26, 40))
    ):
        raise SystemExit(f"{cell} exact81/exact40 sampling differs")
    conditions = row.get("conditions", {})
    if (
        conditions.get("source_v2v_full_source_video_count") != 1
        or conditions.get("source_v2v_source_reference_count") != 0
        or conditions.get("pure_t2v_full_source_video_count") != 0
        or conditions.get("pure_t2v_source_reference_count") != 0
        or conditions.get("pure_t2v_visual_conditions_all_none") is not True
        or conditions.get("homotopy_host_full_source_video_count") != 1
    ):
        raise SystemExit(f"{cell} condition contract differs")
    freeze = row.get("freeze_certificate", {})
    checkpoint_identity = row.get("checkpoint", {})
    if (
        freeze.get("base_frozen") is not True
        or freeze.get("model_eval") is not True
        or freeze.get("adapter_modules_absent") is not True
        or freeze.get("exact_parameter_and_buffer_bytes_hashed") is not True
        or checkpoint_identity.get("manifest_sha256_computed") != "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
        or checkpoint_identity.get("manifest_sha256_expected") != "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
        or checkpoint_identity.get("verified_file_count") != 23
        or checkpoint_identity.get("every_file_sha256_verified") is not True
        or not isinstance(checkpoint_identity.get("verified_entries_digest"), str)
        or len(checkpoint_identity["verified_entries_digest"]) != 64
        or revisions.get("bernini") != "2d2b4591ac053ec25c6371b01a5a6746679e5793"
        or revisions.get("veomni") != "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
        or revisions.get("wan_diffusion_sha256") != "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
        or not isinstance(revisions.get("inference_files"), dict)
        or not revisions["inference_files"]
    ):
        raise SystemExit(f"{cell} checkpoint/vendor/freeze certificate differs")
    traces = row.get("runtime_traces", {})
    if not isinstance(traces, dict) or set(traces) != set(arms):
        raise SystemExit(f"{cell} trace arms differ")
    hom = traces[arms[2]]
    if (
        hom.get("steps") != 40
        or hom.get("transformer_forwards") != 160
        or hom.get("low_source_v2v_forwards") != 80
        or hom.get("high_pure_t2v_forwards") != 80
        or hom.get("original_scheduler_calls") != 40
        or hom.get("optimizer_created") is not False
        or hom.get("parameters_updated") is not False
    ):
        raise SystemExit(f"{cell} homotopy trace differs")
    noise_hashes = set()
    live_outputs = {}
    for arm in arms:
        output = row.get("outputs", {}).get(arm, {})
        video = plain_child_file(cell_root, output.get("path"), f"{arm}.mp4")
        if file_sha(video) != output.get("sha256"):
            raise SystemExit(f"{cell}/{arm} video hash differs")
        live_outputs[arm] = {"sha256": output["sha256"], **video_contract(video)}
        clean_row = output.get("normalized_clean_latent", {})
        clean = plain_child_file(cell_root, clean_row.get("path"), f"{arm}.normalized-clean-latent.safetensors")
        clean_raw = tensor_raw_sha(clean, "normalized_clean_latent")
        if file_sha(clean) != clean_row.get("sha256") or clean_raw != row["generated_identities"][arm]["identity"]["raw_storage_sha256"]:
            raise SystemExit(f"{cell}/{arm} clean latent differs")
        noise_row = row.get("initial_noise_artifacts", {}).get(arm, {})
        noise = plain_child_file(cell_root, noise_row.get("path"), f"{arm}.official-initial-gaussian.safetensors")
        noise_raw = tensor_raw_sha(noise, "official_initial_gaussian")
        if file_sha(noise) != noise_row.get("sha256") or noise_raw != noise_row.get("raw_value_sha256"):
            raise SystemExit(f"{cell}/{arm} noise differs")
        noise_hashes.add(noise_raw)
    if len(noise_hashes) != 1 or next(iter(noise_hashes)) != sampling.get("official_gaussian_raw_sha256"):
        raise SystemExit(f"{cell} within-cell Gaussian equality differs")
    children.append({
        "cell_id": cell,
        "seed": seed,
        "replicate_role": role,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha(receipt_path),
        "receipt_digest": declared_digest,
        "live_output_postflight": live_outputs,
    })

master = {
    "schema_version": "bernini-caper-cdf-dog-object-grounding-exact81-all8-receipt-v1",
    "method": "frozen-bernini-caper-cdf-dog-object-grounding-exact81-canary",
    "topology": "one_wave_two_concurrent_world4_sp4_groups_on_one_8gpu_node",
    "all8_used": True,
    "one_concurrent_wave": True,
    "cell_order": [name for name, _, _ in cells],
    "seeds": [seed for _, seed, _ in cells],
    "arm_order": arms,
    "registry_file_sha256": registry_sha,
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "runtime_source_closure_sha256": closure_sha,
    "launcher_source_sha256": launcher_sha,
    "source_video": str(source),
    "source_video_sha256": source_sha,
    "target_action_caption": caption,
    "exact81": True,
    "fps": 25,
    "bucket_hw": [496, 480],
    "latent_shape": [1, 16, 21, 62, 60],
    "exact40_shift5": True,
    "schedule_sha256": schedule_sha,
    "training_performed": False,
    "optimizer_created": False,
    "parameter_update": False,
    "same_source_cells_are_seed_replicates_not_independent_identities": True,
    "aggregate_as_independent_identities_authorized": False,
    "single_example_conclusion_authorized": False,
    "automatic_object_event_success_claim_authorized": False,
    "ordered_object_event_gate": event_stages,
    "required_source_correspondences": correspondences,
    "object_grounding_outcome_recorded": False,
    "manual_review_required": True,
    "children": children,
}
master["receipt_digest"] = object_sha(master)
target = root / "all8-receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=".all8-receipt.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
PY
