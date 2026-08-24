#!/usr/bin/env bash
#SBATCH --job-name=caper-native-kseed
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

# Default/first invocation renders the entire 4-source x 4-seed fit bank.
# Lockbox rendering is a separate invocation and needs both an explicit
# acknowledgement and a sealed threshold-freeze receipt bound to the complete
# fit-population master receipt.  Every registered cell gets an attempt
# receipt.  A failed child is retained as a failed attempt, remaining waves
# continue, and the root population receipt becomes NO-GO rather than silently
# dropping that seed.

repo_root="${CAPER_NATIVE_KSEED_REPO_ROOT:?set clean repository root}"
source_archive="${CAPER_NATIVE_KSEED_SOURCE_ARCHIVE:?set git archive}"
source_archive_sha256="${CAPER_NATIVE_KSEED_SOURCE_ARCHIVE_SHA256:?set archive SHA-256}"
registry="${CAPER_NATIVE_KSEED_REGISTRY:?set sealed population registry}"
registry_sha256="${CAPER_NATIVE_KSEED_REGISTRY_SHA256:?set registry SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${CAPER_NATIVE_KSEED_OUTPUT_ROOT:?set fresh output root}"
python_bin="${CAPER_NATIVE_KSEED_PYTHON_BIN:?set vace Python executable}"
population_phase="${CAPER_NATIVE_KSEED_PHASE:-fit}"
canonical_registry_sha256="f700eba0d7097943a1c26d7663815a7bdfbf5cd8786c07da2189f716921a7da4"
native_arm="native-source-video-only-v2v-endpoint"

fail() {
  echo "[caper-native-kseed] ERROR: $*" >&2
  exit 2
}

[[ "${population_phase}" == fit || "${population_phase}" == lockbox ]] || fail "phase must be fit or lockbox"
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
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(sha256sum "${registry}" | awk '{print $1}')" == "${registry_sha256}" ]] || fail "registry changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime="${method_root}/infer_caper_native_kseed_population_v1.py"
repo_registry="${method_root}/assets/caper_native_kseed_population_sit_v1.json"
repo_launcher="${method_root}/scripts/auh_infer_caper_native_kseed_population_all8.sbatch"
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
  "${method_root}/tests/test_t2v_v2v_branch_homotopy_v1.py"
  "${method_root}/tests/test_t2v_v2v_branch_homotopy_runtime_v1.py"
  "${method_root}/tests/test_infer_t2v_v2v_branch_homotopy_canary.py"
  "${method_root}/tests/test_infer_caper_native_kseed_population_v1.py"
  "${method_root}/tests/test_auh_infer_caper_native_kseed_population_all8_launcher.py"
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

fit_cells=(
  fit-7b88a1ca1f804f41-s2026081801 fit-7b88a1ca1f804f41-s2026081802
  fit-7b88a1ca1f804f41-s2026081803 fit-7b88a1ca1f804f41-s2026081804
  fit-841b5e0080a1441d-s2026081801 fit-841b5e0080a1441d-s2026081802
  fit-841b5e0080a1441d-s2026081803 fit-841b5e0080a1441d-s2026081804
  fit-402059390cdb4f50-s2026081801 fit-402059390cdb4f50-s2026081802
  fit-402059390cdb4f50-s2026081803 fit-402059390cdb4f50-s2026081804
  fit-3be4072a63144b8f-s2026081801 fit-3be4072a63144b8f-s2026081802
  fit-3be4072a63144b8f-s2026081803 fit-3be4072a63144b8f-s2026081804
)
lockbox_cells=(
  lockbox-6a7ebea80ba64f18-s2026081901 lockbox-6a7ebea80ba64f18-s2026081902
  lockbox-ac87fea937864bd5-s2026081901 lockbox-ac87fea937864bd5-s2026081902
)
fit_source_paths=(
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/7b88a1ca1f804f41/samples/7b88a1ca1f804f41/source_video.mp4"
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/841b5e0080a1441d/samples/841b5e0080a1441d/source_video.mp4"
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/402059390cdb4f50/samples/402059390cdb4f50/source_video.mp4"
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/3be4072a63144b8f/samples/3be4072a63144b8f/source_video.mp4"
)
fit_source_hashes=(
  4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed
  5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a
  210732166ae851bae57135d55798d79ac24b83503b81fadf47e3285b182abdce
  792c506404fba2cc2e88239f52bbb2e88c16af9be9842ec6194234806147b28f
)
lockbox_source_paths=(
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/6a7ebea80ba64f18/samples/6a7ebea80ba64f18/source_video.mp4"
  "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/goku_action_wan22_20260730T043022Z/fullmotion_next1000_v17_20260803T133300Z/wan_next1000_v17/samples/ac87fea937864bd5/samples/ac87fea937864bd5/source_video.mp4"
)
lockbox_source_hashes=(
  2bcf9bc195167c5450ac36026bac0477104814a602bf0caf2a241d302deebcf7
  12f766286bfe1409c7ee3a7bff5a88e6c135c44f20a551919a7801a31dd10ebe
)

threshold_freeze=""
threshold_freeze_sha256=""
fit_population_receipt=""
fit_population_receipt_sha256=""
lockbox_args=()
if [[ "${population_phase}" == fit ]]; then
  [[ -z "${CAPER_NATIVE_KSEED_ENABLE_LOCKBOX:-}" ]] || fail "fit phase refuses lockbox switch"
  [[ -z "${CAPER_NATIVE_KSEED_THRESHOLD_FREEZE_RECEIPT:-}" ]] || fail "fit phase refuses threshold-freeze receipt"
  [[ -z "${CAPER_NATIVE_KSEED_FIT_POPULATION_RECEIPT:-}" ]] || fail "fit phase refuses prior fit receipt"
  cells=("${fit_cells[@]}")
  source_paths=("${fit_source_paths[@]}")
  source_hashes=("${fit_source_hashes[@]}")
else
  [[ "${CAPER_NATIVE_KSEED_ENABLE_LOCKBOX:-}" == "thresholds-frozen-open-lockbox-v1" ]] || fail "explicit lockbox acknowledgement differs"
  threshold_freeze="${CAPER_NATIVE_KSEED_THRESHOLD_FREEZE_RECEIPT:?set sealed threshold-freeze receipt}"
  threshold_freeze_sha256="${CAPER_NATIVE_KSEED_THRESHOLD_FREEZE_SHA256:?set threshold-freeze SHA-256}"
  fit_population_receipt="${CAPER_NATIVE_KSEED_FIT_POPULATION_RECEIPT:?set complete fit master receipt}"
  for value in "${threshold_freeze}" "${fit_population_receipt}"; do [[ "${value}" == /* ]] || fail "lockbox authority paths must be absolute"; done
  [[ "${threshold_freeze_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "threshold-freeze SHA differs"
  threshold_freeze="$(realpath -e -- "${threshold_freeze}")"
  fit_population_receipt="$(realpath -e -- "${fit_population_receipt}")"
  [[ -f "${threshold_freeze}" && ! -L "${threshold_freeze}" ]] || fail "threshold-freeze receipt differs"
  [[ -f "${fit_population_receipt}" && ! -L "${fit_population_receipt}" ]] || fail "fit master receipt differs"
  [[ "$(sha256sum "${threshold_freeze}" | awk '{print $1}')" == "${threshold_freeze_sha256}" ]] || fail "threshold-freeze bytes differ"
  fit_population_receipt_sha256="$(sha256sum "${fit_population_receipt}" | awk '{print $1}')"
  declared_fit_sha="$("${python_bin}" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["fit_population_receipt_sha256"])' "${threshold_freeze}")"
  [[ "${declared_fit_sha}" == "${fit_population_receipt_sha256}" ]] || fail "threshold freeze does not bind supplied fit master receipt"
  cells=("${lockbox_cells[@]}")
  source_paths=("${lockbox_source_paths[@]}")
  source_hashes=("${lockbox_source_hashes[@]}")
  lockbox_args=(
    --lockbox-second-stage-enabled
    --threshold-freeze-receipt "${output_root}/sealed-threshold-freeze-receipt.json"
    --expected-threshold-freeze-sha256 "${threshold_freeze_sha256}"
  )
fi

# Fit preflight deliberately never opens lockbox source bytes.  Lockbox source
# verification occurs only after the sealed second-stage authority above.
[[ "${#source_paths[@]}" == "${#source_hashes[@]}" ]] || fail "source check arrays differ"
for index in "${!source_paths[@]}"; do
  path="${source_paths[index]}"
  expected="${source_hashes[index]}"
  [[ -f "${path}" && ! -L "${path}" ]] || fail "phase source path differs: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || fail "phase source bytes differ: ${path}"
done

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
  test_infer_caper_native_kseed_population_v1.py \
  test_auh_infer_caper_native_kseed_population_all8_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
done

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/caper-native-kseed-${SLURM_JOB_ID}.XXXXXX")"
left_pid=""
right_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/caper-native-kseed-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[caper-native-kseed] PASS phase=${population_phase} all8=true complete-population=true output=${output_root}"
  else
    echo "[caper-native-kseed] NO-GO phase=${population_phase} output=${output_root} status=${status}" >&2
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
cp -- "${registry}" "${output_root}/sealed-caper-native-kseed-population-sit-v1.json"
chmod 0444 -- "${output_root}/sealed-caper-native-kseed-population-sit-v1.json"
if [[ "${population_phase}" == lockbox ]]; then
  cp -- "${threshold_freeze}" "${output_root}/sealed-threshold-freeze-receipt.json"
  cp -- "${fit_population_receipt}" "${output_root}/bound-fit-population-receipt.json"
  chmod 0444 -- "${output_root}/sealed-threshold-freeze-receipt.json" "${output_root}/bound-fit-population-receipt.json"
fi

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
  export BERNINI_OUTPUT_TRANSACTION_ID="caper-native-kseed-${SLURM_JOB_ID}-${label}"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${port}" \
    "${runtime}" \
    --registry "${output_root}/sealed-caper-native-kseed-population-sit-v1.json" \
    --expected-registry-sha256 "${registry_sha256}" \
    --population-phase "${population_phase}" \
    --cell-id "${label}" \
    "${lockbox_args[@]}" \
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
  "${python_bin}" -B - "${output_root}" "${cell}" "${population_phase}" "${process_exit_code}" "${visible}" "${wave_index}" "${log_path}" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve(strict=True)
cell, phase, exit_text, visible, wave_text = sys.argv[2:7]
log_path = Path(sys.argv[7]).resolve(strict=True)
exit_code = int(exit_text)
wave_index = int(wave_text)

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def object_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

if log_path.is_symlink() or not log_path.is_file():
    raise SystemExit("attempt log differs")
parts = cell.split("-")
source_id = parts[1]
seed = int(parts[-1][1:])
cell_root = root / cell
child_receipt = cell_root / "receipt.json"
child_ok = exit_code == 0 and child_receipt.is_file() and not child_receipt.is_symlink()
child_sha = file_sha(child_receipt) if child_ok else None
child_digest = None
if child_ok:
    child = json.loads(child_receipt.read_text(encoding="utf-8"))
    unsigned = dict(child)
    child_digest = unsigned.pop("receipt_digest", None)
    child_ok = object_sha(unsigned) == child_digest and child.get("cell_id") == cell
partial = []
if cell_root.exists() and cell_root.is_dir() and not cell_root.is_symlink():
    for path in sorted(cell_root.iterdir(), key=lambda p: p.name):
        if path.is_file() and not path.is_symlink():
            partial.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha(path)})
attempt_success = exit_code == 0 and child_ok
row = {
    "schema_version": "bernini-caper-native-kseed-attempt-receipt-v1",
    "cell_id": cell,
    "population_phase": phase,
    "source_id": source_id,
    "seed": seed,
    "wave_index": wave_index,
    "world_size": 4,
    "ulysses_size": 4,
    "visible_gpu_slots": visible,
    "process_exit_code": exit_code,
    "attempt_success": attempt_success,
    "attempt_status": "completed_success" if attempt_success else "completed_failure",
    "seed_attempt_recorded_even_on_failure": True,
    "seed_discarded": False,
    "retry_or_replacement_seed_authorized": False,
    "log": {"path": str(log_path), "size_bytes": log_path.stat().st_size, "sha256": file_sha(log_path)},
    "child_receipt_path": str(child_receipt) if child_receipt.exists() else None,
    "child_receipt_file_sha256": child_sha,
    "child_receipt_digest": child_digest,
    "partial_or_complete_cell_artifacts": partial,
}
row["receipt_digest"] = object_sha(row)
target = root / "attempts" / f"{cell}.json"
with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{cell}.", suffix=".tmp", delete=False) as f:
    tmp = Path(f.name)
    f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, target)
fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try: os.fsync(fd)
finally: os.close(fd)
PY
}

overall_process_failure=0
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
  if [[ "${left_status}" != 0 || "${right_status}" != 0 ]]; then
    overall_process_failure=1
  fi
done

"${python_bin}" -B - "${output_root}" "${population_phase}" "${registry_sha256}" "${source_revision}" "${source_archive_sha256}" "${source_closure_sha256}" "${launcher_sha256}" "${threshold_freeze_sha256}" "${fit_population_receipt_sha256}" <<'PY'
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
phase, registry_sha, revision, archive_sha, closure_sha, launcher_sha, freeze_sha, fit_master_sha = sys.argv[2:10]
native_arm = "native-source-video-only-v2v-endpoint"
registry_path = root / "sealed-caper-native-kseed-population-sit-v1.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
design = registry["population_design"][phase]
expected_cells = design["cell_order"]
sources = {row["source_id"]: row for row in registry["sources"] if row["split"] == phase}

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def object_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()

def plain_exact(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise SystemExit(f"plain artifact differs: {path}")
    return path

def video_contract(path: Path, expected_hw: list[int]) -> dict[str, object]:
    frames = 0
    observed = set()
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise SystemExit(f"{path} stream count differs")
        stream = streams[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        for frame in container.decode(stream):
            frames += 1
            observed.add((int(frame.height), int(frame.width)))
    if frames != 81 or abs(fps - 25.0) > 1.0e-6 or observed != {tuple(expected_hw)}:
        raise SystemExit(f"{path} exact81 media differs")
    return {"decoded_frame_count": frames, "decoded_fps": fps, "decoded_hw": expected_hw}

def tensor_raw_sha(path: Path, key: str, expected_shape: list[int]) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [key]:
            raise SystemExit(f"{path} key differs")
        tensor = opened.get_tensor(key).contiguous()
    if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all().item()):
        raise SystemExit(f"{path} tensor differs")
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()

if file_sha(registry_path) != registry_sha:
    raise SystemExit("sealed registry hash differs")
if phase == "fit":
    if freeze_sha or fit_master_sha or (root / "sealed-threshold-freeze-receipt.json").exists() or (root / "bound-fit-population-receipt.json").exists():
        raise SystemExit("fit root unexpectedly contains lockbox authority")
else:
    freeze_path = plain_exact(root / "sealed-threshold-freeze-receipt.json")
    fit_master_path = plain_exact(root / "bound-fit-population-receipt.json")
    if file_sha(freeze_path) != freeze_sha or file_sha(fit_master_path) != fit_master_sha:
        raise SystemExit("lockbox authority artifact bytes differ")
attempt_paths = sorted((root / "attempts").glob("*.json"), key=lambda p: p.name)
if [path.stem for path in attempt_paths] != sorted(expected_cells):
    raise SystemExit("attempt ledger does not exhaust the registered phase population")

children = []
failed_attempts = []
for cell in expected_cells:
    attempt_path = root / "attempts" / f"{cell}.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_unsigned = dict(attempt)
    attempt_digest = attempt_unsigned.pop("receipt_digest", None)
    if (
        object_sha(attempt_unsigned) != attempt_digest
        or attempt.get("cell_id") != cell
        or attempt.get("population_phase") != phase
        or attempt.get("seed_attempt_recorded_even_on_failure") is not True
        or attempt.get("seed_discarded") is not False
        or attempt.get("retry_or_replacement_seed_authorized") is not False
    ):
        raise SystemExit(f"{cell} attempt receipt differs")
    if not attempt.get("attempt_success"):
        failed_attempts.append({
            "cell_id": cell,
            "source_id": attempt.get("source_id"),
            "seed": attempt.get("seed"),
            "process_exit_code": attempt.get("process_exit_code"),
            "attempt_receipt_sha256": file_sha(attempt_path),
            "log_sha256": attempt.get("log", {}).get("sha256"),
            "partial_or_complete_cell_artifacts": attempt.get("partial_or_complete_cell_artifacts", []),
        })
        continue
    cell_root = root / cell
    receipt_path = plain_exact(cell_root / "receipt.json")
    row = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(row)
    declared = unsigned.pop("receipt_digest", None)
    source = sources[row["input"]["source_id"]]
    if (
        object_sha(unsigned) != declared
        or row.get("schema_version") != "bernini-caper-native-kseed-cell-receipt-v1"
        or row.get("method") != "frozen-bernini-caper-native-full-source-v2v-kseed-population-v1"
        or row.get("cell_id") != cell
        or row.get("population_phase") != phase
        or row.get("training_performed") is not False
        or row.get("optimizer_created") is not False
        or row.get("parameter_update") is not False
        or row.get("seed_filtering_or_best_of_k_authorized") is not False
        or row.get("a66e_generated_by_this_runner") is not False
        or row.get("registry", {}).get("sha256") != registry_sha
        or row.get("registry", {}).get("complete_phase_cell_order") != expected_cells
        or row.get("input", {}).get("source_video_sha256") != source["source_video_sha256"]
        or row.get("input", {}).get("target_video") is not False
        or any(row.get("input", {}).get(key) is not False for key in ("mask", "track", "pose", "flow", "trajectory", "first_frame_anchor", "source_reference"))
    ):
        raise SystemExit(f"{cell} receipt boundary differs")
    sampling = row.get("sampling", {})
    if (
        sampling.get("arm") != native_arm
        or sampling.get("guidance_mode") != "v2v_apg"
        or sampling.get("frame_count") != 81
        or sampling.get("latent_phases") != 21
        or sampling.get("num_inference_steps") != 40
        or sampling.get("fps") != 25
        or sampling.get("flow_shift_from_renderer_config") != 5.0
        or sampling.get("schedule_sha256") != "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
        or sampling.get("official_gaussian_is_sampler_initial_noise") is not True
        or sampling.get("official_gaussian_captured_by_read_only_observer") is not True
        or sampling.get("observer_injects_or_replaces_noise") is not False
        or sampling.get("external_initial_noise_injection") is not False
    ):
        raise SystemExit(f"{cell} sampling differs")
    expected_seed = int(cell.rsplit("-s", 1)[1])
    if sampling.get("seed") != expected_seed:
        raise SystemExit(f"{cell} seed differs")
    if phase == "fit" and row.get("threshold_freeze") is not None:
        raise SystemExit(f"{cell} fit consumed lockbox authority")
    if phase == "lockbox":
        freeze = row.get("threshold_freeze", {})
        if (
            freeze.get("artifact_sha256") != freeze_sha
            or freeze.get("fit_population_receipt_sha256") != fit_master_sha
            or freeze.get("method_and_thresholds_frozen_before_lockbox") is not True
            or freeze.get("lockbox_media_opened_before_freeze_is_false") is not True
        ):
            raise SystemExit(f"{cell} threshold freeze differs")
    source_path = plain_exact(Path(source["source_video"]))
    if file_sha(source_path) != source["source_video_sha256"]:
        raise SystemExit(f"{cell} source bytes differ")
    output = row.get("outputs", {}).get(native_arm, {})
    video = plain_exact(cell_root / f"{native_arm}.mp4")
    if output.get("path") != str(video) or file_sha(video) != output.get("sha256"):
        raise SystemExit(f"{cell} output video differs")
    live_video = video_contract(video, source["bucket_hw"])
    clean_row = output.get("normalized_clean_latent", {})
    clean = plain_exact(cell_root / f"{native_arm}.normalized-clean-latent.safetensors")
    clean_raw = tensor_raw_sha(clean, "normalized_clean_latent", source["latent_shape"])
    if clean_row.get("path") != str(clean) or file_sha(clean) != clean_row.get("sha256") or clean_raw != row["generated_identity"]["identity"]["raw_storage_sha256"]:
        raise SystemExit(f"{cell} clean latent differs")
    noise_row = row.get("official_initial_gaussian", {})
    noise = plain_exact(cell_root / f"{native_arm}.official-initial-gaussian.safetensors")
    noise_raw = tensor_raw_sha(noise, "official_initial_gaussian", source["latent_shape"])
    if noise_row.get("path") != str(noise) or file_sha(noise) != noise_row.get("sha256") or noise_raw != noise_row.get("raw_value_sha256") or noise_raw != sampling.get("official_gaussian_raw_sha256"):
        raise SystemExit(f"{cell} official Gaussian differs")
    children.append({
        "cell_id": cell,
        "source_id": source["source_id"],
        "seed": expected_seed,
        "attempt_receipt_sha256": file_sha(attempt_path),
        "child_receipt_path": str(receipt_path),
        "child_receipt_file_sha256": file_sha(receipt_path),
        "child_receipt_digest": declared,
        "mp4_sha256": output["sha256"],
        "normalized_clean_latent_sha256": clean_row["sha256"],
        "official_gaussian_sha256": noise_row["sha256"],
        "official_gaussian_raw_sha256": noise_raw,
        "live_video_postflight": live_video,
    })

population_complete = not failed_attempts and len(children) == len(expected_cells)
master = {
    "schema_version": "bernini-caper-native-kseed-population-all8-receipt-v1",
    "method": "frozen-bernini-caper-native-full-source-v2v-kseed-population-v1",
    "population_phase": phase,
    "topology": "waves_of_two_concurrent_world4_sp4_groups_on_one_8gpu_node",
    "all8_used_each_wave": True,
    "registered_cell_order": expected_cells,
    "registered_cell_count": len(expected_cells),
    "attempt_receipt_count": len(attempt_paths),
    "every_registered_seed_attempt_recorded": len(attempt_paths) == len(expected_cells),
    "successful_cell_count": len(children),
    "failed_cell_count": len(failed_attempts),
    "population_complete": population_complete,
    "population_decision": "PASS_COMPLETE" if population_complete else "NO_GO_INCOMPLETE_OR_FAILED_ATTEMPTS",
    "partial_population_scientific_claim_authorized": False,
    "seed_filtering_or_best_of_k_authorized": False,
    "retry_or_replacement_seed_authorized": False,
    "registry_file_sha256": registry_sha,
    "action_family_id": "animal-sit-and-hold",
    "action_taxonomy_sha256": "cd25fe7e95b2d43a2b33a5b02bad44833405155c28ef2402a2b2dab11da04396",
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "runtime_source_closure_sha256": closure_sha,
    "launcher_source_sha256": launcher_sha,
    "threshold_freeze_receipt_sha256": freeze_sha or None,
    "bound_fit_population_receipt_sha256": fit_master_sha or None,
    "native_arm": native_arm,
    "exact81": True,
    "fps": 25,
    "exact40_shift5": True,
    "training_performed": False,
    "optimizer_created": False,
    "parameter_update": False,
    "a66e_posthoc_negative_generated": False,
    "children": children,
    "failed_attempts": failed_attempts,
}
master["receipt_digest"] = object_sha(master)
target = root / f"{phase}-population-receipt.json"
with tempfile.NamedTemporaryFile(dir=root, prefix=f".{phase}-population-receipt.", suffix=".tmp", delete=False) as f:
    tmp = Path(f.name)
    f.write(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, target)
fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try: os.fsync(fd)
finally: os.close(fd)
print(json.dumps(master, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
PY

population_complete="$("${python_bin}" -B -c 'import json,sys; print("true" if json.load(open(sys.argv[1]))["population_complete"] else "false")' "${output_root}/${population_phase}-population-receipt.json")"
if [[ "${overall_process_failure}" != 0 || "${population_complete}" != true ]]; then
  exit 3
fi
