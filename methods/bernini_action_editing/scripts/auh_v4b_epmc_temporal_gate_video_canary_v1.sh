#!/usr/bin/env bash
# Frozen in-allocation controller for the v4-B -> Bernini EPMC video canary.
# It never submits a job.  The caller must provide one exact single-node,
# four-MI210 Slurm step and a hash-pinned read-only method source archive.

set -Eeuo pipefail
umask 077

readonly tag="v4b-epmc-temporal-gate-video-canary-v1"
readonly expected_iid="7b88a1ca1f804f41"
readonly expected_source_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
readonly expected_anchor_sha256="8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015"
readonly expected_instruction_sha256="105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"
readonly expected_feature_receipt_sha256="8ff8f5fd5be36cb67ce40d5558a4406bdf70cbe9b72b0c43c71fa3abe8f6ad9c"
readonly expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
readonly expected_checkpoint_file_count=23
readonly expected_bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
readonly release_sealed=false
readonly expected_method_source_archive_sha256="0000000000000000000000000000000000000000000000000000000000000000"
readonly expected_method_tree_sha256="0000000000000000000000000000000000000000000000000000000000000000"
readonly expected_materializer_sha256="502af9a1c00cb06942193b4dc5848228d581b54df6142ff3868972fef0678d38"
readonly expected_runner_sha256="71a8be8ca60950245ecf9802cd6608c519c72f5180e542872733fa6991fd17ab"
readonly expected_builder_sha256="a0366b4e9a338e2ed9f4658825be070372917143a42d4d1e76f9277ca74a1bbc"
readonly expected_test_sha256="3807b9ee5d670959442cfef52ee3dcb58267ff34a0e9cb6502ba7c04d82ee163"

fail() {
  echo "[${tag}] ERROR: $*" >&2
  exit 2
}

# Deliberate NO-GO.  A release authority must replace both zero digests and set
# release_sealed=true in one reviewed controller revision.  Until then this
# exits before reading run parameters, allocating scratch, loading a checkpoint,
# importing Bernini, or touching an output path.
[[ "${release_sealed}" == true ]] || fail "UNSEALED CONTROLLER: diagnostic canary is NO-GO"

actual_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

method_tree_digest() {
  "${python_bin}" -B - "$1" <<'PY'
from pathlib import Path
import hashlib, json, stat, sys
root = Path(sys.argv[1]).resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    rel = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise SystemExit(f"method tree symlink: {rel}")
    if stat.S_ISDIR(mode):
        if path.name == "__pycache__":
            raise SystemExit("method tree contains __pycache__")
        continue
    if not stat.S_ISREG(mode) or path.suffix in (".pyc", ".pyo"):
        raise SystemExit(f"unsupported method member: {rel}")
    if mode & 0o222:
        raise SystemExit(f"writable method member: {rel}")
    rows.append({"path": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
print(hashlib.sha256(raw).hexdigest())
PY
}

validate_checkpoint_content() {
  local manifest="$1"
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest differs"
  [[ "$(actual_sha256 "${manifest}")" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest SHA differs"
  [[ "$(wc -l < "${manifest}")" -eq "${expected_checkpoint_file_count}" ]] || fail "checkpoint manifest count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains symlink"
  [[ "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)" -eq "${expected_checkpoint_file_count}" ]] || fail "checkpoint file count differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content differs"
}

gpu_idle_gate() {
  [[ "${SLURM_JOB_ID:?controller requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid SLURM_JOB_ID"
  [[ "${SLURM_STEP_ID:?controller requires a real Slurm step}" =~ ^[0-9]+$ ]] || fail "invalid SLURM_STEP_ID"
  [[ "${SLURM_NNODES:-${SLURM_STEP_NUM_NODES:-}}" == "1" ]] || fail "controller requires one Slurm node"
  [[ "${SLURM_GPUS_ON_NODE:-}" == "4" ]] || fail "Slurm step must expose exactly four GPUs"
  local visible="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  local physical_list="${SLURM_STEP_GPUS:-}"
  [[ -n "${visible}" && -n "${physical_list}" ]] || fail "visible/physical GPU authority is absent"
  IFS=',' read -r -a visible_devices <<< "${visible}"
  IFS=',' read -r -a devices <<< "${physical_list}"
  [[ "${#visible_devices[@]}" -eq 4 && "${#devices[@]}" -eq 4 ]] || fail "visible/physical lists are not WORLD4"
  local seen="," device
  for device in "${devices[@]}"; do
    [[ "${device}" =~ ^[0-9]+$ ]] || fail "SLURM_STEP_GPUS must contain physical integer IDs"
    [[ "${seen}" != *",${device},"* ]] || fail "duplicate physical GPU identifier"
    seen="${seen}${device},"
  done
  "${python_bin}" -B - "${devices[@]}" <<'PY'
import json, os, subprocess, sys, torch
physical = [int(x) for x in sys.argv[1:]]
if len(physical) != 4 or len(set(physical)) != 4:
    raise SystemExit("physical GPU identity closure differs")
if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
    raise SystemExit("ROCm torch is unavailable")
if torch.cuda.device_count() != 4:
    raise SystemExit("torch logical device count is not WORLD4")
names = [torch.cuda.get_device_name(i) for i in range(4)]
if any("MI210" not in name for name in names):
    raise SystemExit(f"logical GPU is not MI210: {names}")
raw = subprocess.check_output(
    ["rocm-smi", "--showuniqueid", "--showproductname", "--showuse", "--showmeminfo", "vram", "--json"],
    text=True,
)
data = json.loads(raw)
uuids = []
for index in physical:
    row = data.get(f"card{index}")
    if not isinstance(row, dict):
        raise SystemExit(f"rocm-smi lacks card{index}")
    joined = " ".join(str(v) for v in row.values())
    if "MI210" not in joined:
        raise SystemExit(f"physical card{index} is not MI210")
    unique_values = [str(v) for k, v in row.items() if "unique" in k.lower()]
    if len(unique_values) != 1 or not unique_values[0] or unique_values[0] in ("0", "N/A"):
        raise SystemExit(f"physical card{index} unique ID differs")
    uuids.append(unique_values[0])
    use_values = [str(v).strip().rstrip("%") for k, v in row.items() if "gpu use" in k.lower()]
    if len(use_values) != 1 or float(use_values[0]) != 0.0:
        raise SystemExit(f"physical card{index} is already compute-active")
    used_values = [str(v).split()[0] for k, v in row.items() if "used" in k.lower() and "vram" in k.lower()]
    if len(used_values) != 1 or int(float(used_values[0])) != 0:
        raise SystemExit(f"physical card{index} already has VRAM allocated")
if len(set(uuids)) != 4:
    raise SystemExit("MI210 unique IDs are not distinct")
print(json.dumps({"world_size": 4, "logical_names": names, "physical_cards": physical, "unique_ids": uuids}, sort_keys=True))
PY
}

readonly python_bin="${V4B_EPMC_PYTHON:?set V4B_EPMC_PYTHON}"
readonly source_archive="${V4B_EPMC_METHOD_SOURCE_ARCHIVE:?set V4B_EPMC_METHOD_SOURCE_ARCHIVE}"
readonly source_archive_sha256="${V4B_EPMC_METHOD_SOURCE_ARCHIVE_SHA256:?set V4B_EPMC_METHOD_SOURCE_ARCHIVE_SHA256}"
readonly method_revision="${V4B_EPMC_METHOD_REVISION:?set V4B_EPMC_METHOD_REVISION}"
readonly v4b_receipt="${V4B_EPMC_V4B_RECEIPT:?set V4B_EPMC_V4B_RECEIPT}"
readonly v4b_receipt_sha256="${V4B_EPMC_V4B_RECEIPT_SHA256:?set V4B_EPMC_V4B_RECEIPT_SHA256}"
readonly fold1_checkpoint="${V4B_EPMC_FOLD1_CHECKPOINT:?set V4B_EPMC_FOLD1_CHECKPOINT}"
readonly fold1_checkpoint_sha256="${V4B_EPMC_FOLD1_CHECKPOINT_SHA256:?set V4B_EPMC_FOLD1_CHECKPOINT_SHA256}"
readonly feature_root="${V4B_EPMC_FEATURE_ROOT:?set V4B_EPMC_FEATURE_ROOT}"
readonly bernini_root="${V4B_EPMC_BERNINI_ROOT:?set V4B_EPMC_BERNINI_ROOT}"
readonly bernini_archive="${V4B_EPMC_BERNINI_ARCHIVE:?set V4B_EPMC_BERNINI_ARCHIVE}"
readonly veomni_root="${V4B_EPMC_VEOMNI_ROOT:?set V4B_EPMC_VEOMNI_ROOT}"
readonly checkpoint="${V4B_EPMC_BERNINI_CHECKPOINT:?set V4B_EPMC_BERNINI_CHECKPOINT}"
readonly source_video="${V4B_EPMC_SOURCE_VIDEO:?set V4B_EPMC_SOURCE_VIDEO}"
readonly anchor_video_ref="${V4B_EPMC_ANCHOR_VIDEO_REF:?set V4B_EPMC_ANCHOR_VIDEO_REF}"
readonly instruction="${V4B_EPMC_INSTRUCTION:?set V4B_EPMC_INSTRUCTION}"
readonly output_root="${V4B_EPMC_OUTPUT_ROOT:?set V4B_EPMC_OUTPUT_ROOT}"

[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "python is not a plain executable"
for file in "${source_archive}" "${v4b_receipt}" "${fold1_checkpoint}" "${bernini_archive}" "${source_video}" "${anchor_video_ref}"; do
  [[ "${file}" == /* && -f "${file}" && ! -L "${file}" ]] || fail "required absolute plain file missing: ${file}"
done
for directory in "${feature_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ "${directory}" == /* && -d "${directory}" && ! -L "${directory}" ]] || fail "required absolute real directory missing: ${directory}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ && "${method_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "method source identity differs"
[[ "${source_archive_sha256}" == "${expected_method_source_archive_sha256}" ]] || fail "method release archive authority differs"
[[ "${v4b_receipt_sha256}" =~ ^[0-9a-f]{64}$ && "${fold1_checkpoint_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "v4-B artifact pins differ"
[[ -z "$(find "${source_archive}" -maxdepth 0 -perm /222 -print -quit)" ]] || fail "method archive must be read-only"
[[ "$(actual_sha256 "${source_archive}")" == "${source_archive_sha256}" ]] || fail "method source archive SHA differs"
[[ "$(actual_sha256 "${bernini_archive}")" == "${expected_bernini_archive_sha256}" ]] || fail "Bernini source archive SHA differs"
[[ "$(actual_sha256 "${v4b_receipt}")" == "${v4b_receipt_sha256}" ]] || fail "v4-B receipt SHA differs"
[[ "$(actual_sha256 "${fold1_checkpoint}")" == "${fold1_checkpoint_sha256}" ]] || fail "fold-1 checkpoint SHA differs"
[[ "$(actual_sha256 "${feature_root}/feature_extraction_receipt.json")" == "${expected_feature_receipt_sha256}" ]] || fail "feature receipt SHA differs"
[[ "$(actual_sha256 "${source_video}")" == "${expected_source_sha256}" ]] || fail "source video SHA differs"
[[ "$(actual_sha256 "${anchor_video_ref}")" == "${expected_anchor_sha256}" ]] || fail "detached anchor video SHA differs"
[[ "$(printf '%s' "${instruction}" | sha256sum | awk '{print $1}')" == "${expected_instruction_sha256}" ]] || fail "instruction SHA differs"
[[ "${output_root}" == /* && "${output_root}" != / && ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root must be fresh, absolute, and non-root"
readonly output_name="${output_root##*/}"
[[ "${output_name}" =~ ^[A-Za-z0-9_-]+$ ]] || fail "output basename is unsafe"
readonly output_parent="$(dirname -- "${output_root}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

readonly scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
readonly task_scratch="$(mktemp -d "${scratch_parent%/}/${tag}-${SLURM_JOB_ID:?}.${SLURM_STEP_ID:?}.XXXXXX")"
cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch}" in
    "${scratch_parent%/}/${tag}-${SLURM_JOB_ID}.${SLURM_STEP_ID}."*) ;;
    *) echo "[${tag}] unsafe scratch cleanup target: ${task_scratch}" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  echo "[${tag}] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p "${task_scratch}/source" "${task_scratch}/input" "${task_scratch}/pycache" "${task_scratch}/miopen-user" "${task_scratch}/miopen-custom" "${task_scratch}/torch-extensions" "${task_scratch}/triton"
readonly archive_copy="${task_scratch}/method.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod a-w -- "${archive_copy}"
[[ "$(actual_sha256 "${archive_copy}")" == "${source_archive_sha256}" ]] || fail "private archive copy differs"
"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
with tarfile.open(sys.argv[1], "r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        allowed_parent = path.parts in (("methods",), ("methods", "bernini_action_editing"))
        allowed_child = path.parts[:2] == ("methods", "bernini_action_editing")
        if not member.name or path.is_absolute() or ".." in path.parts or not (allowed_parent or allowed_child):
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
readonly source_root="${task_scratch}/source"
readonly method_root="${source_root}/methods/bernini_action_editing"
for required in \
  materialize_v4b_epmc_gate_state_v1.py \
  infer_v4b_epmc_temporal_gate_canary_v1.py \
  tools/build_v4b_epmc_temporal_gate_review_v1.py \
  tests/test_v4b_epmc_temporal_gate_canary_v1.py \
  scripts/auh_v4b_epmc_temporal_gate_video_canary_v1.sh \
  semantic_anchor_temporal_convae_v4b_fast.py \
  semantic_anchor_linear_frontier_v4_fast.py \
  semantic_anchor_action_sequence_vae_v2.py \
  semantic_action_cvae_canary_v1.py \
  infer_fewshot_motion_code.py \
  fewshot_motion_branch.py \
  fewshot_episode_io.py \
  fewshot_privileged_motion_code.py \
  fewshot_proposal_motion_carrier.py \
  counterfactual_proposal_motion_runtime.py \
  counterfactual_proposal_motion_branch.py \
  counterfactual_proposal_motion_rebinding.py \
  infer_counterfactual_proposal_motion_oracle.py \
  infer_lora.py \
  infer_source_kv_carrier_oracle.py \
  infer_source_value_residual_oracle.py \
  source_kv_route_batches.py \
  source_kv_replay.py \
  source_value_residual.py \
  action_preservation_decoded_eval_model_authority_v2.py \
  self_generated_action_preservation_v2.py \
  train_lora.py \
  tools/build_renderer_dataset.py \
  tools/materialize_vae.py \
  audits/bernini_r13_ff4c5d4_checkpoint.sha256
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "extracted method tree contains symlink"
find "${method_root}" -type f -exec chmod a-w -- {} +
readonly controller_self="$(realpath -- "$0")"
[[ -f "${controller_self}" && ! -L "${controller_self}" && "$(stat -c '%h' "${controller_self}")" == 1 ]] || fail "executing controller identity differs"
cmp -s -- "${controller_self}" "${method_root}/scripts/auh_v4b_epmc_temporal_gate_video_canary_v1.sh" || fail "executing controller is not the archived controller"
[[ "$(actual_sha256 "${method_root}/materialize_v4b_epmc_gate_state_v1.py")" == "${expected_materializer_sha256}" ]] || fail "materializer source pin differs"
[[ "$(actual_sha256 "${method_root}/infer_v4b_epmc_temporal_gate_canary_v1.py")" == "${expected_runner_sha256}" ]] || fail "runner source pin differs"
[[ "$(actual_sha256 "${method_root}/tools/build_v4b_epmc_temporal_gate_review_v1.py")" == "${expected_builder_sha256}" ]] || fail "builder source pin differs"
[[ "$(actual_sha256 "${method_root}/tests/test_v4b_epmc_temporal_gate_canary_v1.py")" == "${expected_test_sha256}" ]] || fail "test source pin differs"
readonly method_tree_sha256_pre="$(method_tree_digest "${method_root}")"
[[ "${method_tree_sha256_pre}" == "${expected_method_tree_sha256}" ]] || fail "frozen method tree authority differs"
validate_checkpoint_content "${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"

readonly staged_source="${task_scratch}/input/source.mp4"
readonly staged_anchor="${task_scratch}/input/anchor.mp4"
cp -- "${source_video}" "${staged_source}"
cp -- "${anchor_video_ref}" "${staged_anchor}"
chmod a-w -- "${staged_source}" "${staged_anchor}"
[[ "$(actual_sha256 "${staged_source}")" == "${expected_source_sha256}" ]] || fail "staged source differs"
[[ "$(actual_sha256 "${staged_anchor}")" == "${expected_anchor_sha256}" ]] || fail "staged anchor differs"

unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export PYTHONPATH="${source_root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"
cd "${source_root}"

for optimize in normal optimized; do
  if [[ "${optimize}" == normal ]]; then
    "${python_bin}" -B -m unittest methods.bernini_action_editing.tests.test_v4b_epmc_temporal_gate_canary_v1 -v
  else
    "${python_bin}" -O -B -m unittest methods.bernini_action_editing.tests.test_v4b_epmc_temporal_gate_canary_v1 -v
  fi
done

mkdir -p -- "${output_root}"
readonly gate_state="${output_root}/v4b-fold1-${expected_iid}-gate-state.json"
"${python_bin}" -B "${method_root}/materialize_v4b_epmc_gate_state_v1.py" \
  --v4b-receipt "${v4b_receipt}" \
  --expected-v4b-receipt-sha256 "${v4b_receipt_sha256}" \
  --fold1-checkpoint "${fold1_checkpoint}" \
  --expected-fold1-checkpoint-sha256 "${fold1_checkpoint_sha256}" \
  --feature-root "${feature_root}" \
  --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
  --output "${gate_state}"
readonly gate_state_sha256="$(actual_sha256 "${gate_state}")"

run_seed() {
  local seed="$1" output_dir="$2"
  [[ "${seed}" == 2028 || "${seed}" == 2029 ]] || fail "render seed differs"
  [[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "render output already exists"
  gpu_idle_gate
  BERNINI_OUTPUT_TRANSACTION_ID="v4b-epmc-s${seed}" \
    "${python_bin}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
      "${method_root}/infer_v4b_epmc_temporal_gate_canary_v1.py" \
      --bernini-root "${bernini_root}" \
      --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" \
      --checkpoint-content-manifest "${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256" \
      --source-video "${staged_source}" \
      --instruction "${instruction}" \
      --gate-state "${gate_state}" \
      --expected-gate-state-sha256 "${gate_state_sha256}" \
      --output-dir "${output_dir}" \
      --render-seed "${seed}" \
      --expected-source-sha256 "${expected_source_sha256}" \
      --expected-instruction-sha256 "${expected_instruction_sha256}" \
      --method-source-revision "${method_revision}" \
      --method-source-archive-sha256 "${source_archive_sha256}"
  [[ -f "${output_dir}/receipt.json" && ! -L "${output_dir}/receipt.json" ]] || fail "seed ${seed} receipt missing"
  [[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 6 ]] || fail "seed ${seed} output count differs"
  for arm in B0 zero correct reverse shuffle; do
    [[ -f "${output_dir}/${arm}.mp4" && ! -L "${output_dir}/${arm}.mp4" ]] || fail "seed ${seed} ${arm} missing"
  done
}

readonly seed2028_dir="${output_root}/seed2028"
readonly seed2029_dir="${output_root}/seed2029"
run_seed 2028 "${seed2028_dir}"
run_seed 2029 "${seed2029_dir}"
readonly seed2028_receipt_sha256="$(actual_sha256 "${seed2028_dir}/receipt.json")"
readonly seed2029_receipt_sha256="$(actual_sha256 "${seed2029_dir}/receipt.json")"
readonly review_dir="${output_root}/review"
"${python_bin}" -B "${method_root}/tools/build_v4b_epmc_temporal_gate_review_v1.py" \
  --seed2028-dir "${seed2028_dir}" \
  --seed2029-dir "${seed2029_dir}" \
  --expected-seed2028-receipt-sha256 "${seed2028_receipt_sha256}" \
  --expected-seed2029-receipt-sha256 "${seed2029_receipt_sha256}" \
  --source-video-ref "${staged_source}" \
  --anchor-video-ref "${staged_anchor}" \
  --expected-source-sha256 "${expected_source_sha256}" \
  --expected-anchor-sha256 "${expected_anchor_sha256}" \
  --instruction "${instruction}" \
  --output-dir "${review_dir}"
[[ -f "${review_dir}/index.html" && -f "${review_dir}/packet.json" ]] || fail "HTML review closure missing"
chmod -R a-w -- "${seed2028_dir}" "${seed2029_dir}"
readonly method_tree_sha256_post="$(method_tree_digest "${method_root}")"
[[ "${method_tree_sha256_post}" == "${method_tree_sha256_pre}" ]] || fail "method tree changed during execution"
echo "[${tag}] COMPLETE diagnostic_only=true world=4xMI210 seeds=2028,2029 method_tree_sha256=${method_tree_sha256_pre} review=${review_dir}/index.html"
