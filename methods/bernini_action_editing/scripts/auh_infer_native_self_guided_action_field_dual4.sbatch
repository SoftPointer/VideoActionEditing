#!/usr/bin/env bash
#SBATCH --job-name=bernini-action-field
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# GPUs 0-3 run dog and GPUs 4-7 run human as independent WORLD4/Ulysses4
# groups.  Each group compares native RV2V and two fixed frozen T2V action-
# field scales at exact81/exact40.  No training or parameter update occurs.

repo_root="${ACTION_FIELD_REPO_ROOT:?set clean repository root}"
source_archive="${ACTION_FIELD_SOURCE_ARCHIVE:?set git archive}"
source_archive_sha256="${ACTION_FIELD_SOURCE_ARCHIVE_SHA256:?set git archive SHA-256}"
registry="${ACTION_FIELD_REGISTRY:?set sealed action-field core2 registry}"
registry_sha256="${ACTION_FIELD_REGISTRY_SHA256:?set registry SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${ACTION_FIELD_OUTPUT_ROOT:?set fresh output root}"
python_bin="${ACTION_FIELD_PYTHON_BIN:?set vace Python executable}"

fail() {
  echo "[self-guided-action-field] ERROR: $*" >&2
  exit 2
}

for value in \
  "${repo_root}" "${source_archive}" "${registry}" "${bernini_root}" \
  "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" \
  "${output_root}" "${python_bin}"
do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
for value in "${source_archive_sha256}" "${registry_sha256}"
do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "SHA-256 input differs"
done

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
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/infer_native_self_guided_action_field_canary.py"
core="${method_root}/self_guided_action_field_v1.py"
repo_launcher="${method_root}/scripts/auh_infer_native_self_guided_action_field_dual4.sbatch"
running_launcher="$(realpath -e -- "$0")"
repo_closure=(
  "${runtime}"
  "${core}"
  "${method_root}/infer_native_identity_generation_canary.py"
  "${method_root}/tri_branch_unipc.py"
  "${method_root}/infer_lora.py"
  "${method_root}/train_lora.py"
  "${method_root}/infer_source_kv_carrier_oracle.py"
  "${method_root}/infer_source_value_residual_oracle.py"
  "${method_root}/source_kv_replay.py"
  "${method_root}/source_kv_route_batches.py"
  "${method_root}/source_value_residual.py"
  "${method_root}/tests/test_self_guided_action_field_v1.py"
  "${method_root}/tests/test_infer_native_self_guided_action_field_canary.py"
  "${method_root}/tests/test_auh_native_self_guided_action_field_launcher.py"
  "${repo_launcher}"
)
closure=("${repo_closure[@]}" "${registry}")
for path in "${closure[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "source closure file missing: ${path}"
done
[[ "$(sha256sum "${running_launcher}" | awk '{print $1}')" == "$(sha256sum "${repo_launcher}" | awk '{print $1}')" ]] || fail "running launcher differs"

source_revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive/repository revision differs"
for path in "${repo_closure[@]}"; do
  relative="${path#"${repo_root}/"}"
  [[ "${relative}" != "${path}" ]] || fail "repository closure path escaped"
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null || fail "repository closure path is untracked: ${relative}"
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
  test_self_guided_action_field_v1.py \
  test_infer_native_self_guided_action_field_canary.py \
  test_auh_native_self_guided_action_field_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/self-guided-action-field-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/self-guided-action-field-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[self-guided-action-field] PASS all8=true exact81=true exact40=true training=false output=${output_root}"
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

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
cp -- "${registry}" "${output_root}/sealed-action-field-core2.json"
chmod 0444 -- "${output_root}/sealed-action-field-core2.json"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((40000 + 2 * job_mod))
human_port=$((dog_port + 1))

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
  export BERNINI_OUTPUT_TRANSACTION_ID="self-guided-action-field-${SLURM_JOB_ID}-${label}"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" \
    --registry "${output_root}/sealed-action-field-core2.json" \
    --expected-registry-sha256 "${registry_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_root}/${label}" \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}"
)

launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/logs/dog.log" 2>&1 &
dog_pid=$!
launch_group human 4,5,6,7 "${human_port}" >"${output_root}/logs/human.log" 2>&1 &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 ]] || fail "dog SP4 failed with ${dog_status}"
[[ "${human_status}" == 0 ]] || fail "human SP4 failed with ${human_status}"

"${python_bin}" -B - "${output_root}" "${registry_sha256}" "${source_revision}" "${source_archive_sha256}" "${source_closure_sha256}" <<'PY'
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
registry_sha, revision, archive_sha, closure_sha = sys.argv[2:]

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def object_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()

def plain_child_file(cell_root: Path, declared: object, expected_name: str) -> Path:
    if not isinstance(declared, str):
        raise SystemExit(f"{expected_name} path is not text")
    requested = Path(declared)
    expected = cell_root / expected_name
    if (
        not requested.is_absolute()
        or requested != expected
        or requested.is_symlink()
        or requested.resolve(strict=True) != requested
        or not requested.is_file()
    ):
        raise SystemExit(f"{expected_name} path differs")
    return requested

def decoded_video_contract(path: Path, *, cell: str, arm: str, height: int, width: int) -> dict[str, object]:
    frame_count = 0
    observed_hw: set[tuple[int, int]] = set()
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise SystemExit(f"{cell}/{arm} must contain one video stream")
        stream = streams[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        for frame in container.decode(stream):
            frame_count += 1
            observed_hw.add((int(frame.height), int(frame.width)))
    if frame_count != 81 or abs(fps - 25.0) > 1.0e-6 or observed_hw != {(height, width)}:
        raise SystemExit(
            f"{cell}/{arm} decoded media differs: frames={frame_count} fps={fps} hw={sorted(observed_hw)}"
        )
    return {"decoded_frame_count": frame_count, "decoded_fps": fps, "decoded_hw": [height, width]}

def safetensor_raw_sha(path: Path, *, key: str, expected_shape: list[int]) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [key]:
            raise SystemExit(f"{path.name} safetensors key differs")
        tensor = opened.get_tensor(key).contiguous()
    if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape:
        raise SystemExit(f"{path.name} tensor geometry differs")
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()

arms = ["native-rv2v", "action-field-075", "action-field-150"]
children = []
for cell in ("dog", "human"):
    cell_root = root / cell
    path = cell_root / "receipt.json"
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise SystemExit(f"{cell} receipt path differs")
    row = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(row)
    declared = unsigned.pop("receipt_digest")
    if object_sha(unsigned) != declared:
        raise SystemExit(f"{cell} receipt seal differs")
    if (
        row.get("schema_version")
        != "bernini-self-guided-action-field-canary-receipt-v3"
        or row.get("method")
        != "frozen-bernini-guided-t2v-action-field-canary"
        or row.get("cell_id") != cell
        or row.get("sampling", {}).get("arms") != arms
    ):
        raise SystemExit(f"{cell} receipt contract differs")
    sampling = row.get("sampling", {})
    if (
        sampling.get("frame_count") != 81
        or sampling.get("num_inference_steps") != 40
        or sampling.get("guidance_mode") != "v2v_apg"
    ):
        raise SystemExit(f"{cell} exact81/exact40/v2v_apg differs")
    if row.get("training_performed") is not False or row.get("parameter_update") is not False:
        raise SystemExit(f"{cell} frozen boundary differs")
    action_field = row.get("action_field")
    if not isinstance(action_field, dict) or set(action_field) != set(arms):
        raise SystemExit(f"{cell} action-field receipt set differs")
    native_field = action_field["native-rv2v"]
    if native_field != {
        "native_baseline": True,
        "frozen_t2v_teacher_forwards": 0,
        "effective_scale": 0.0,
    }:
        raise SystemExit(f"{cell} native action-field receipt differs")
    expected_call_graph = [
        "native-negative",
        "native-action",
        "t2v-uncond",
        "t2v-target",
        "t2v-source",
    ]
    for arm, scale in (("action-field-075", 0.75), ("action-field-150", 1.5)):
        field = action_field[arm]
        if not isinstance(field, dict):
            raise SystemExit(f"{cell}/{arm} action-field row differs")
        trace = field.get("trace")
        if (
            field.get("effective_scale") != scale
            or field.get("native_rv2v_steps") != 40
            or field.get("native_rv2v_forwards") != 80
            or field.get("frozen_t2v_teacher_forwards") != 120
            or field.get("original_scheduler_calls") != 40
            or field.get("per_step_call_graph") != expected_call_graph
            or field.get("teacher_field") != "t2v_apg_target_minus_t2v_apg_source"
            or field.get("composition_boundary")
            != "after_official_apg_before_original_scheduler_step"
            or field.get("injection_divided_by_omega_txt") is not False
            or not isinstance(trace, list)
            or len(trace) != 40
        ):
            raise SystemExit(f"{cell}/{arm} action-field summary differs")
        for index, step in enumerate(trace):
            if (
                not isinstance(step, dict)
                or step.get("step_index") != index
                or step.get("native_official_apg_exact_parity") is not True
                or step.get("t2v_teacher_forwards") != 3
                or step.get("original_scheduler_calls") != 1
                or step.get("injection_divided_by_omega_txt") is not False
            ):
                raise SystemExit(f"{cell}/{arm} action-field step {index} differs")
    freeze = row.get("freeze_certificate")
    freeze_digest = freeze.get("state_content_sha256") if isinstance(freeze, dict) else None
    if (
        not isinstance(freeze, dict)
        or freeze.get("base_frozen") is not True
        or freeze.get("model_eval") is not True
        or freeze.get("adapter_modules_absent") is not True
        or freeze.get("exact_parameter_and_buffer_bytes_hashed") is not True
        or not isinstance(freeze_digest, str)
        or len(freeze_digest) != 64
        or any(character not in "0123456789abcdef" for character in freeze_digest)
    ):
        raise SystemExit(f"{cell} freeze certificate differs")
    live_outputs = {}
    live_clean = {}
    live_noise = {}
    latent_shape = list(row.get("generated_identities", {}).get(arms[0], {}).get("identity", {}).get("shape", []))
    if len(latent_shape) != 5 or latent_shape[:3] != [1, 16, 21]:
        raise SystemExit(f"{cell} generated latent geometry differs")
    for arm in arms:
        output = row.get("outputs", {}).get(arm, {})
        video = plain_child_file(cell_root, output.get("path"), f"{arm}.mp4")
        video_sha = file_sha(video)
        if video_sha != output.get("sha256"):
            raise SystemExit(f"{cell}/{arm} live MP4 hash differs")
        live_outputs[arm] = {
            "sha256": video_sha,
            **decoded_video_contract(
                video,
                cell=cell,
                arm=arm,
                height=int(output.get("height", -1)),
                width=int(output.get("width", -1)),
            ),
        }

        clean_row = output.get("normalized_clean_latent", {})
        clean = plain_child_file(
            cell_root,
            clean_row.get("path"),
            f"{arm}.normalized-clean-latent.safetensors",
        )
        clean_container_sha = file_sha(clean)
        if clean_container_sha != clean_row.get("sha256"):
            raise SystemExit(f"{cell}/{arm} live clean-latent container hash differs")
        clean_raw_sha = safetensor_raw_sha(
            clean,
            key="normalized_clean_latent",
            expected_shape=latent_shape,
        )
        declared_generated = row["generated_identities"][arm]["identity"]["raw_storage_sha256"]
        if clean_raw_sha != declared_generated:
            raise SystemExit(f"{cell}/{arm} live clean-latent value hash differs")
        live_clean[arm] = {
            "container_sha256": clean_container_sha,
            "raw_value_sha256": clean_raw_sha,
        }

        noise_row = row.get("initial_noise_artifacts", {}).get(arm, {})
        noise = plain_child_file(
            cell_root,
            noise_row.get("path"),
            f"{arm}.official-initial-gaussian.safetensors",
        )
        noise_container_sha = file_sha(noise)
        if noise_container_sha != noise_row.get("sha256"):
            raise SystemExit(f"{cell}/{arm} live Gaussian container hash differs")
        noise_raw_sha = safetensor_raw_sha(
            noise,
            key="official_initial_gaussian",
            expected_shape=latent_shape,
        )
        if noise_raw_sha != noise_row.get("raw_value_sha256"):
            raise SystemExit(f"{cell}/{arm} live Gaussian value hash differs")
        live_noise[arm] = {
            "container_sha256": noise_container_sha,
            "raw_value_sha256": noise_raw_sha,
        }
    children.append({
        "cell_id": cell,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha(path),
        "receipt_digest": declared,
        "action_field_trace_digest": object_sha(action_field),
        "freeze_certificate_digest": object_sha(freeze),
        "live_output_postflight": live_outputs,
        "live_clean_latent_postflight": live_clean,
        "live_initial_gaussian_postflight": live_noise,
    })

receipt = {
    "schema_version": "bernini-self-guided-action-field-all8-receipt-v3",
    "method": "frozen-bernini-guided-t2v-action-field-canary",
    "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
    "all8_used": True,
    "cell_order": ["dog", "human"],
    "registry_file_sha256": registry_sha,
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "runtime_source_closure_sha256": closure_sha,
    "exact81": True,
    "exact40": True,
    "guidance_mode": "v2v_apg",
    "training_performed": False,
    "parameter_update": False,
    "generated_owner_media_consumed": False,
    "source_rich_initial_noise": False,
    "children": children,
}
receipt["receipt_digest"] = object_sha(receipt)
target = root / "all8-receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=".all8-receipt.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
PY
