#!/usr/bin/env bash
#SBATCH --job-name=caper-native-siblings
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

# One public fit source x K=4 seeds x four same-seed sibling arms.  Each cell
# stays inside one WORLD4/SP4 process and runs target/noop/incomplete/
# phase-order-violation as four complete stock full-source v2v_apg trajectories.
# Two isolated SP4 groups process two cells concurrently for two waves.

repo_root="${CAPER_NATIVE_SIBLING_REPO_ROOT:?set clean repository root}"
source_archive="${CAPER_NATIVE_SIBLING_SOURCE_ARCHIVE:?set git archive}"
source_archive_sha256="${CAPER_NATIVE_SIBLING_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
registry="${CAPER_NATIVE_SIBLING_REGISTRY:?set sealed sibling registry}"
registry_sha256="${CAPER_NATIVE_SIBLING_REGISTRY_SHA256:?set registry SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${CAPER_NATIVE_SIBLING_OUTPUT_ROOT:?set fresh output root}"
python_bin="${CAPER_NATIVE_SIBLING_PYTHON_BIN:?set vace Python executable}"

canonical_registry_sha256="09b1bdc13a91ded9fe853f9582499da310995c0bc4d48e50599f52cb0de154e0"
source_id="7b88a1ca1f804f41"
source_video="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/7b88a1ca1f804f41/samples/7b88a1ca1f804f41/source_video.mp4"
source_video_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
schedule_sha256="46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
arms=(target noop incomplete phase-order-violation)
cells=(
  fit-7b88a1ca1f804f41-s2026081801
  fit-7b88a1ca1f804f41-s2026081802
  fit-7b88a1ca1f804f41-s2026081803
  fit-7b88a1ca1f804f41-s2026081804
)

fail() {
  echo "[caper-native-siblings] ERROR: $*" >&2
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
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "public fit source differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"
[[ "$(sha256sum "${source_video}" | awk '{print $1}')" == "${source_video_sha256}" ]] || fail "public fit source bytes differ"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/infer_caper_native_counterfactual_siblings_v1.py"
repo_registry="${method_root}/assets/caper_native_counterfactual_siblings_sit_v1.json"
repo_launcher="${method_root}/scripts/auh_infer_caper_native_counterfactual_siblings_all8.sbatch"
runner_test="${method_root}/tests/test_infer_caper_native_counterfactual_siblings_v1.py"
launcher_test="${method_root}/tests/test_auh_infer_caper_native_counterfactual_siblings_all8_launcher.py"
running_launcher="$(realpath -e -- "$0")"
repo_closure=(
  "${runtime}"
  "${repo_registry}"
  "${repo_launcher}"
  "${runner_test}"
  "${launcher_test}"
  "${method_root}/infer_caper_native_kseed_population_v1.py"
  "${method_root}/infer_t2v_v2v_branch_homotopy_canary.py"
  "${method_root}/infer_t2v_v2v_branch_homotopy_v1.py"
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

PYTHONPATH="${method_root}" "${python_bin}" -B "${runner_test}"
PYTHONPATH="${method_root}" "${python_bin}" -B "${launcher_test}"

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/caper-native-siblings-${SLURM_JOB_ID}.XXXXXX")"
left_pid=""
right_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/caper-native-siblings-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[caper-native-siblings] PASS all8=true source=${source_id} k=4 arms=4 rollouts=16 output=${output_root}"
  else
    echo "[caper-native-siblings] NO-GO source=${source_id} output=${output_root} status=${status}" >&2
  fi
  exit "${status}"
}

terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${left_pid}" "${right_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then kill "${child}" 2>/dev/null || true; fi
  done
  for child in "${left_pid}" "${right_pid}"; do
    if [[ "${child}" =~ ^[0-9]+$ ]]; then wait "${child}" 2>/dev/null || true; fi
  done
  exit "${status}"
}

trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs" "${output_root}/attempts"
cp -- "${registry}" "${output_root}/sealed-caper-native-counterfactual-siblings-sit-v1.json"
chmod 0444 -- "${output_root}/sealed-caper-native-counterfactual-siblings-sit-v1.json"

job_mod=$((SLURM_JOB_ID % 5000))
base_port=$((40000 + 2 * job_mod))

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
  export BERNINI_OUTPUT_TRANSACTION_ID="caper-native-siblings-${SLURM_JOB_ID}-${label}"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" \
    --registry "${output_root}/sealed-caper-native-counterfactual-siblings-sit-v1.json" \
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

write_attempt_receipt() {
  local cell="$1"
  local process_exit_code="$2"
  local visible="$3"
  local wave_index="$4"
  local log_path="${output_root}/logs/${cell}.log"
  "${python_bin}" -B - "${output_root}" "${cell}" "${process_exit_code}" "${visible}" "${wave_index}" "${log_path}" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve(strict=True)
cell, exit_text, visible, wave_text = sys.argv[2:6]
log_path = Path(sys.argv[6]).resolve(strict=True)
exit_code = int(exit_text)
wave_index = int(wave_text)
arms = ["target", "noop", "incomplete", "phase-order-violation"]

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def object_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

if log_path.is_symlink() or not log_path.is_file():
    raise SystemExit("attempt log differs")
seed = int(cell.rsplit("-s", 1)[1])
cell_root = root / cell
child_receipt = cell_root / "receipt.json"
child_ok = exit_code == 0 and child_receipt.is_file() and not child_receipt.is_symlink()
child_sha = file_sha(child_receipt) if child_ok else None
child_digest = None
if child_ok:
    child = json.loads(child_receipt.read_text(encoding="utf-8"))
    unsigned = dict(child)
    child_digest = unsigned.pop("receipt_digest", None)
    child_ok = (
        object_sha(unsigned) == child_digest
        and child.get("cell_id") == cell
        and child.get("arm_order") == arms
        and child.get("all_four_sibling_arms_complete") is True
    )
partial = []
if cell_root.exists() and cell_root.is_dir() and not cell_root.is_symlink():
    for path in sorted(cell_root.iterdir(), key=lambda item: item.name):
        if path.is_file() and not path.is_symlink():
            partial.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha(path)})
attempt_success = exit_code == 0 and child_ok
row = {
    "schema_version": "bernini-caper-native-counterfactual-sibling-attempt-receipt-v1",
    "cell_id": cell,
    "source_id": "7b88a1ca1f804f41",
    "seed": seed,
    "arm_order": arms,
    "expected_arm_count": 4,
    "wave_index": wave_index,
    "world_size": 4,
    "ulysses_size": 4,
    "visible_gpu_slots": visible,
    "process_exit_code": exit_code,
    "attempt_success": attempt_success,
    "attempt_status": "completed_success" if attempt_success else "completed_failure",
    "cell_process_attempt_recorded_even_on_failure": True,
    "all_four_arm_outcomes_closed": attempt_success,
    "unobserved_or_incomplete_arm_outcomes_possible": not attempt_success,
    "seed_discarded": False,
    "retry_or_replacement_seed_authorized": False,
    "log": {"path": str(log_path), "size_bytes": log_path.stat().st_size, "sha256": file_sha(log_path)},
    "cell_receipt_path": str(child_receipt) if child_receipt.exists() else None,
    "cell_receipt_file_sha256": child_sha,
    "cell_receipt_digest": child_digest,
    "partial_or_complete_cell_artifacts": partial,
}
row["receipt_digest"] = object_sha(row)
target = root / "attempts" / f"{cell}.json"
with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{cell}.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, target)
fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try: os.fsync(fd)
finally: os.close(fd)
PY
}

for ((index=0; index<${#cells[@]}; index+=2)); do
  left="${cells[index]}"
  right="${cells[index+1]}"
  wave=$((index / 2))
  left_port=$((base_port + 2 * wave))
  right_port=$((left_port + 1))
  launch_group "${left}" 0,1,2,3 "${left_port}" >"${output_root}/logs/${left}.log" 2>&1 &
  left_pid=$!
  launch_group "${right}" 4,5,6,7 "${right_port}" >"${output_root}/logs/${right}.log" 2>&1 &
  right_pid=$!
  left_status=0
  right_status=0
  wait "${left_pid}" || left_status=$?
  wait "${right_pid}" || right_status=$?
  left_pid=""
  right_pid=""
  write_attempt_receipt "${left}" "${left_status}" 0,1,2,3 "${wave}"
  write_attempt_receipt "${right}" "${right_status}" 4,5,6,7 "${wave}"
done

"${python_bin}" -B - "${output_root}" "${registry_sha256}" "${source_revision}" "${source_archive_sha256}" "${source_closure_sha256}" "${launcher_sha256}" "${source_video}" "${source_video_sha256}" "${schedule_sha256}" <<'PY'
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
source_video = Path(sys.argv[7])
source_sha = sys.argv[8]
schedule_sha = sys.argv[9]
method = "frozen-bernini-caper-native-same-seed-counterfactual-siblings-v1"
cell_schema = "bernini-caper-native-counterfactual-sibling-cell-receipt-v1"
arm_schema = "bernini-caper-native-counterfactual-sibling-arm-receipt-v1"
attempt_schema = "bernini-caper-native-counterfactual-sibling-attempt-receipt-v1"
arms = ["target", "noop", "incomplete", "phase-order-violation"]
cells = [f"fit-7b88a1ca1f804f41-s{seed}" for seed in range(2026081801, 2026081805)]
seeds = list(range(2026081801, 2026081805))
registry_path = root / "sealed-caper-native-counterfactual-siblings-sit-v1.json"

def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def object_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

def sealed_json(path: Path) -> tuple[dict[str, object], str]:
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise SystemExit(f"sealed JSON path differs: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(row)
    declared = unsigned.pop("receipt_digest", None)
    if object_sha(unsigned) != declared:
        raise SystemExit(f"receipt seal differs: {path}")
    return row, str(declared)

def plain_child(cell_root: Path, declared: object, name: str) -> Path:
    requested = Path(declared) if isinstance(declared, str) else Path(".")
    expected = cell_root / name
    if not requested.is_absolute() or requested != expected or requested.is_symlink() or requested.resolve(strict=True) != requested or not requested.is_file():
        raise SystemExit(f"{cell_root.name}/{name} path differs")
    return requested

def tensor_raw_sha(path: Path, key: str, expected_shape: list[int]) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [key]:
            raise SystemExit(f"{path} tensor key differs")
        tensor = opened.get_tensor(key).contiguous()
    if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all().item()):
        raise SystemExit(f"{path} tensor geometry differs")
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()

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
    if frames != 81 or abs(fps - 25.0) > 1.0e-6 or observed_hw != {(480, 496)}:
        raise SystemExit(f"{path} exact81 media differs")
    return {"decoded_frame_count": frames, "decoded_fps": fps, "decoded_hw": [480, 496]}

if registry_path.is_symlink() or file_sha(registry_path) != registry_sha:
    raise SystemExit("sealed registry differs")
registry = json.loads(registry_path.read_text(encoding="utf-8"))
if registry.get("arm_order") != arms or registry.get("population_design", {}).get("cell_order") != cells:
    raise SystemExit("registry population order differs")
if source_video.is_symlink() or source_video.resolve(strict=True) != source_video or file_sha(source_video) != source_sha:
    raise SystemExit("public source bytes differ")

attempts = []
children = []
failed_attempts = []
source_latent_hashes = set()
checkpoint_digests = set()
live_schedule_digests = set()

attempt_paths = sorted((root / "attempts").glob("*.json"), key=lambda path: path.name)
if [path.stem for path in attempt_paths] != sorted(cells):
    raise SystemExit("every K=4 cell must have one attempt receipt")

for cell, seed in zip(cells, seeds):
    attempt_path = root / "attempts" / f"{cell}.json"
    attempt, attempt_digest = sealed_json(attempt_path)
    if (
        attempt.get("schema_version") != attempt_schema
        or attempt.get("cell_id") != cell
        or attempt.get("seed") != seed
        or attempt.get("arm_order") != arms
        or attempt.get("expected_arm_count") != 4
        or attempt.get("cell_process_attempt_recorded_even_on_failure") is not True
        or attempt.get("all_four_arm_outcomes_closed")
        is not (attempt.get("attempt_success") is True)
        or attempt.get("unobserved_or_incomplete_arm_outcomes_possible")
        is not (attempt.get("attempt_success") is not True)
        or attempt.get("seed_discarded") is not False
        or attempt.get("retry_or_replacement_seed_authorized") is not False
        or (attempt.get("attempt_success") is True and attempt.get("process_exit_code") != 0)
    ):
        raise SystemExit(f"{cell} attempt boundary differs")
    attempt_summary = {
        "cell_id": cell,
        "path": str(attempt_path),
        "file_sha256": file_sha(attempt_path),
        "receipt_digest": attempt_digest,
        "attempt_success": attempt.get("attempt_success"),
        "process_exit_code": attempt.get("process_exit_code"),
        "all_four_arm_outcomes_closed": attempt.get("all_four_arm_outcomes_closed"),
        "unobserved_or_incomplete_arm_outcomes_possible": attempt.get(
            "unobserved_or_incomplete_arm_outcomes_possible"
        ),
    }
    attempts.append(attempt_summary)
    if attempt.get("attempt_success") is not True:
        failed_attempts.append(attempt_summary)
        continue

    cell_root = root / cell
    receipt_path = cell_root / "receipt.json"
    receipt, receipt_digest = sealed_json(receipt_path)
    if (
        attempt.get("cell_receipt_path") != str(receipt_path)
        or attempt.get("cell_receipt_file_sha256") != file_sha(receipt_path)
        or attempt.get("cell_receipt_digest") != receipt_digest
    ):
        raise SystemExit(f"{cell} attempt/cell receipt binding differs")
    if (
        receipt.get("schema_version") != cell_schema
        or receipt.get("method") != method
        or receipt.get("cell_id") != cell
        or receipt.get("source_id") != "7b88a1ca1f804f41"
        or receipt.get("seed") != seed
        or receipt.get("arm_order") != arms
        or receipt.get("expected_arm_count") != 4
        or receipt.get("complete_arm_count") != 4
        or receipt.get("all_four_sibling_arms_complete") is not True
        or receipt.get("training_performed") is not False
        or receipt.get("optimizer_created") is not False
        or receipt.get("parameter_update") is not False
        or receipt.get("preference_admission_performed") is not False
        or receipt.get("partial_population_scientific_claim_authorized") is not False
    ):
        raise SystemExit(f"{cell} cell receipt boundary differs")
    input_row = receipt.get("input", {})
    shared = receipt.get("shared_contract", {})
    if (
        input_row.get("source_video") != str(source_video)
        or input_row.get("source_video_sha256") != source_sha
        or input_row.get("target_video") is not False
        or input_row.get("custom_initial_noise") is not False
        or input_row.get("mask_track_pose_flow_trajectory") is not False
        or shared.get("same_source_latent_all_arms") is not True
        or shared.get("same_checkpoint_all_arms") is not True
        or shared.get("same_negative_embedding_object_within_each_rank") is not True
        or shared.get("same_official_gaussian_value_all_arms") is not True
        or shared.get("same_exact40_schedule_all_arms") is not True
        or shared.get("shared_high_sigma_prefix") is not False
        or shared.get("shared_prefix_steps") != 0
        or shared.get("independent_complete_stock_trajectory_per_arm") is not True
        or shared.get("positive_embedding_shared_across_arms") is not False
        or shared.get("external_initial_noise_injection") is not False
    ):
        raise SystemExit(f"{cell} shared sibling contract differs")

    source_artifact = receipt.get("source_condition_artifact", {})
    source_path = plain_child(cell_root, source_artifact.get("path"), "source.normalized-clean-latent.safetensors")
    source_raw = tensor_raw_sha(source_path, "normalized_clean_latent", [1, 16, 21, 60, 62])
    source_identity = shared.get("source_identity_before", {}).get("identity", {})
    if (
        file_sha(source_path) != source_artifact.get("sha256")
        or source_artifact.get("artifact_role") != "source_video_condition"
        or source_raw != source_identity.get("raw_storage_sha256")
        or shared.get("source_identity_after") != shared.get("source_identity_before")
    ):
        raise SystemExit(f"{cell} shared source latent differs")
    source_latent_hashes.add(source_raw)
    checkpoint_digests.add(object_sha(receipt.get("checkpoint")))
    live_schedule_digests.add(shared.get("live_schedule_digest"))

    arm_rows = receipt.get("arm_receipts")
    if not isinstance(arm_rows, list) or [row.get("arm") for row in arm_rows] != arms:
        raise SystemExit(f"{cell} ordered arm receipt closure differs")
    gaussian_hashes = set()
    cell_outputs = []
    for arm_index, (arm, pointer) in enumerate(zip(arms, arm_rows)):
        arm_path = plain_child(cell_root, pointer.get("path"), f"{arm}.receipt.json")
        arm_receipt, arm_digest = sealed_json(arm_path)
        if file_sha(arm_path) != pointer.get("file_sha256") or arm_digest != pointer.get("receipt_digest"):
            raise SystemExit(f"{cell}/{arm} arm receipt pointer differs")
        if (
            arm_receipt.get("schema_version") != arm_schema
            or arm_receipt.get("method") != method
            or arm_receipt.get("cell_id") != cell
            or arm_receipt.get("arm") != arm
            or arm_receipt.get("arm_index") != arm_index
            or arm_receipt.get("seed") != seed
            or arm_receipt.get("source_video_sha256") != source_sha
            or arm_receipt.get("training_performed") is not False
            or arm_receipt.get("optimizer_created") is not False
            or arm_receipt.get("parameter_update") is not False
            or arm_receipt.get("preference_admission_performed") is not False
        ):
            raise SystemExit(f"{cell}/{arm} arm boundary differs")
        sampling = arm_receipt.get("sampling", {})
        live = sampling.get("live_schedule", {})
        if (
            sampling.get("guidance_mode") != "v2v_apg"
            or sampling.get("positive_task") != "mv2v"
            or sampling.get("frame_count") != 81
            or sampling.get("latent_phases") != 21
            or sampling.get("num_inference_steps") != 40
            or sampling.get("fps") != 25
            or sampling.get("seed") != seed
            or sampling.get("official_gaussian_is_sampler_initial_noise") is not True
            or sampling.get("official_gaussian_captured_by_read_only_observer") is not True
            or sampling.get("observer_injects_or_replaces_noise") is not False
            or sampling.get("external_initial_noise_injection") is not False
            or sampling.get("shared_high_sigma_prefix") is not False
            or sampling.get("independent_complete_stock_trajectory") is not True
            or sampling.get("flow_shift") != 5.0
            or sampling.get("omega_text") != 4.0
            or sampling.get("eta") != 0.5
            or sampling.get("norm_threshold") != 50.0
            or sampling.get("momentum") != 0.0
            or sampling.get("schedule_sha256") != schedule_sha
            or live.get("pinned_schedule_sha256") != schedule_sha
            or live.get("completed_step_index") != 40
        ):
            raise SystemExit(f"{cell}/{arm} native exact40 contract differs")
        trace = arm_receipt.get("runtime_trace", {})
        if (
            trace.get("native_endpoint") is not True
            or trace.get("denoiser_or_scheduler_field_hook_installed") is not False
            or trace.get("initial_noise_observer_installed") is not True
            or trace.get("initial_noise_observer_read_only") is not True
            or trace.get("initial_noise_observer_replaces_or_injects_noise") is not False
            or trace.get("expected_transformer_forwards_per_step_from_pinned_vendor") != 2
            or trace.get("source_video_count") != 1
            or trace.get("source_reference_count") != 0
            or trace.get("shared_prefix_steps") != 0
            or trace.get("complete_independent_stock_steps") != 40
        ):
            raise SystemExit(f"{cell}/{arm} stock trace differs")

        output = arm_receipt.get("output", {})
        video = plain_child(cell_root, output.get("path"), f"{arm}.mp4")
        clean_row = output.get("normalized_clean_latent", {})
        clean = plain_child(cell_root, clean_row.get("path"), f"{arm}.normalized-clean-latent.safetensors")
        noise_row = arm_receipt.get("official_initial_gaussian", {})
        noise = plain_child(cell_root, noise_row.get("path"), f"{arm}.official-initial-gaussian.safetensors")
        clean_raw = tensor_raw_sha(clean, "normalized_clean_latent", [1, 16, 21, 60, 62])
        noise_raw = tensor_raw_sha(noise, "official_initial_gaussian", [1, 16, 21, 60, 62])
        if (
            file_sha(video) != output.get("sha256")
            or file_sha(clean) != clean_row.get("sha256")
            or file_sha(noise) != noise_row.get("sha256")
            or clean_raw != arm_receipt.get("generated_identity", {}).get("identity", {}).get("raw_storage_sha256")
            or noise_raw != noise_row.get("raw_value_sha256")
            or noise_raw != sampling.get("official_gaussian_raw_sha256")
            or arm_receipt.get("source_condition_artifact") != source_artifact
        ):
            raise SystemExit(f"{cell}/{arm} live artifact identity differs")
        gaussian_hashes.add(noise_raw)
        cell_outputs.append({
            "arm": arm,
            "arm_receipt_path": str(arm_path),
            "arm_receipt_file_sha256": file_sha(arm_path),
            "arm_receipt_digest": arm_digest,
            "mp4_sha256": output.get("sha256"),
            "clean_latent_raw_sha256": clean_raw,
            "official_gaussian_raw_sha256": noise_raw,
            **video_contract(video),
        })
    if len(gaussian_hashes) != 1 or next(iter(gaussian_hashes)) != shared.get("official_gaussian_raw_sha256"):
        raise SystemExit(f"{cell} four-arm official Gaussian equality differs")
    children.append({
        "cell_id": cell,
        "seed": seed,
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha(receipt_path),
        "receipt_digest": receipt_digest,
        "shared_source_latent_raw_sha256": source_raw,
        "shared_official_gaussian_raw_sha256": next(iter(gaussian_hashes)),
        "arms": cell_outputs,
    })

cross_k_invariants = {
    "same_source_video_bytes_all_cells": True,
    "each_cell_uses_one_source_latent_for_all_four_arms": len(children) == 4,
    "cross_seed_source_latent_byte_identity_required": False,
    "observed_distinct_source_latent_raw_sha256_count": len(source_latent_hashes),
    "same_checkpoint_all_k4_cells": len(children) == 4 and len(checkpoint_digests) == 1,
    "same_live_exact40_schedule_all_k4_cells": len(children) == 4 and len(live_schedule_digests) == 1,
}
population_complete = (
    len(attempts) == 4
    and len(failed_attempts) == 0
    and len(children) == 4
    and cross_k_invariants["same_source_video_bytes_all_cells"] is True
    and cross_k_invariants["each_cell_uses_one_source_latent_for_all_four_arms"] is True
    and cross_k_invariants["same_checkpoint_all_k4_cells"] is True
    and cross_k_invariants["same_live_exact40_schedule_all_k4_cells"] is True
)
master = {
    "schema_version": "bernini-caper-native-counterfactual-sibling-population-receipt-v1",
    "method": method,
    "population_decision": "PASS_COMPLETE" if population_complete else "NO_GO_INCOMPLETE_OR_FAILED_ATTEMPTS",
    "population_complete": population_complete,
    "topology": "one_node_two_concurrent_world4_sp4_groups_two_waves",
    "all8_used": True,
    "source_id": "7b88a1ca1f804f41",
    "source_video": str(source_video),
    "source_video_sha256": source_sha,
    "k": 4,
    "seeds": seeds,
    "cell_order": cells,
    "arm_order": arms,
    "arms_per_cell": 4,
    "expected_rollout_count": 16,
    "complete_rollout_count": len(children) * 4,
    "every_registered_cell_process_attempt_recorded": len(attempts) == 4,
    "every_registered_seed_and_arm_outcome_closed": population_complete,
    "seed_filtering_or_best_of_k_authorized": False,
    "replacement_seed_authorized": False,
    "shared_high_sigma_prefix": False,
    "official_gaussian_equal_value_verified_for_all_k4_cells": len(children) == 4,
    "independent_complete_stock_v2v_apg_trajectory_verified_for_all_arms": len(children) == 4,
    "required_output_contract_exact81": True,
    "all_completed_outputs_verified_exact81": population_complete,
    "fps": 25,
    "latent_shape": [1, 16, 21, 60, 62],
    "required_trajectory_contract_exact40_shift5": True,
    "all_completed_trajectories_verified_exact40_shift5": population_complete,
    "schedule_sha256": schedule_sha,
    "cross_k_invariants": cross_k_invariants,
    "registry_file_sha256": registry_sha,
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "runtime_source_closure_sha256": closure_sha,
    "launcher_source_sha256": launcher_sha,
    "attempts": attempts,
    "failed_attempts": failed_attempts,
    "children": children,
    "training_performed": False,
    "optimizer_created": False,
    "parameter_update": False,
    "preference_admission_performed": False,
    "partial_population_scientific_claim_authorized": False,
}
master["receipt_digest"] = object_sha(master)
target = root / "population-receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=".population-receipt.", suffix=".tmp", delete=False) as handle:
    temporary = Path(handle.name)
    handle.write(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, target)
fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try: os.fsync(fd)
finally: os.close(fd)
print(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False), flush=True)
raise SystemExit(0 if population_complete else 3)
PY
