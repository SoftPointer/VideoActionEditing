#!/usr/bin/env bash
#SBATCH --job-name=bern-egntc-k2
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=02:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_EGNTC_SOURCE_ARCHIVE:?set BERNINI_EGNTC_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256:?set BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_EGNTC_SOURCE_REVISION:?set BERNINI_EGNTC_SOURCE_REVISION}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
preview_manifest="${BERNINI_EGNTC_PREVIEW_MANIFEST:?set BERNINI_EGNTC_PREVIEW_MANIFEST}"
vae_index="${BERNINI_EGNTC_VAE_INDEX:?set BERNINI_EGNTC_VAE_INDEX}"
mode="${BERNINI_EGNTC_MODE:?set BERNINI_EGNTC_MODE=smoke or formal}"
output_dir="${BERNINI_EGNTC_OUTPUT_DIR:?set BERNINI_EGNTC_OUTPUT_DIR}"
python_bin="${BERNINI_EGNTC_PYTHON_BIN:?set BERNINI_EGNTC_PYTHON_BIN}"

expected_config_sha256="a46d18fce025b0cd3b30a6505514b817cd5d96c43d305d6202405a952eed2446"
expected_preview_sha256="49506e003f86f319ebe8a5e843d19c88cef75e84cd4250968da283bb19252e47"
expected_vae_index_sha256="d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
expected_ffprobe_sha256="c2739db02e1e3abedebd5b7d84dc253432884dc797d4e16102df8362f158c23f"
expected_pyav_version="13.1.0"
bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_manifest_file_count=23
optimizer_steps_per_round="${BERNINI_EGNTC_OPTIMIZER_STEPS_PER_ROUND:-25}"
learning_rate="${BERNINI_EGNTC_LEARNING_RATE:-0.02}"
max_grad_norm="${BERNINI_EGNTC_MAX_GRAD_NORM:-1.0}"
training_seed="${BERNINI_EGNTC_TRAINING_SEED:-20260809}"
rollout_seed="${BERNINI_EGNTC_ROLLOUT_SEED:-2029}"

fail() {
  echo "[bernini-egntc] ERROR: $*" >&2
  exit 2
}

validate_checkpoint_content() {
  local manifest="$1"
  local actual_count
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || fail "checkpoint manifest is not plain"
  [[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"
  [[ "$(wc -l < "${manifest}")" -eq "${checkpoint_manifest_file_count}" ]] || fail "checkpoint manifest count differs"
  [[ -z "$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type l -print -quit)" ]] || fail "checkpoint contains non-cache symlink"
  actual_count="$(find "${checkpoint}" -path "${checkpoint}/.cache" -prune -o -type f -print | wc -l)"
  [[ "${actual_count}" -eq "${checkpoint_manifest_file_count}" ]] || fail "checkpoint file set differs"
  (cd "${checkpoint}" && sha256sum --strict --status -c "${manifest}") || fail "checkpoint content hash differs"
}

for value_name in source_archive bernini_root veomni_root checkpoint preview_manifest vae_index output_dir python_bin; do
  value="${!value_name}"
  [[ "${value}" == /* ]] || fail "${value_name} must be absolute"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive SHA-256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not plain"
[[ "$(stat -c '%A' "${source_archive}")" != *w* ]] || fail "source archive must be read-only"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini root"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "invalid VeOmni root"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid checkpoint"
[[ -f "${preview_manifest}" && ! -L "${preview_manifest}" ]] || fail "invalid preview manifest"
[[ -f "${vae_index}" && ! -L "${vae_index}" ]] || fail "invalid VAE index"
[[ -f "${python_bin}" && -x "${python_bin}" ]] || fail "invalid Python"
export BERNINI_EPMC_PYTHON_BIN="${python_bin}"
[[ "$("${BERNINI_EPMC_PYTHON_BIN}" -B -c 'import av; print(av.__version__)')" == "${expected_pyav_version}" ]] || fail "Python must provide pinned PyAV ${expected_pyav_version}"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
[[ "${output_dir}" != / ]] || fail "output cannot be root"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
case "${mode}" in
  smoke) mode_args=(--engineering-smoke) ;;
  formal) mode_args=() ;;
  *) fail "mode must be smoke or formal" ;;
esac

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${preview_manifest}" | awk '{print $1}')" == "${expected_preview_sha256}" ]] || fail "preview hash differs"
[[ "$(sha256sum "${vae_index}" | awk '{print $1}')" == "${expected_vae_index_sha256}" ]] || fail "VAE index hash differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit differs"

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

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-egntc-${mode}-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  status=$?
  trap - EXIT TERM INT
  case "${task_scratch}" in
    "${scratch_parent%/}/bernini-egntc-${mode}-${SLURM_JOB_ID}."*) ;;
    *) echo "[bernini-egntc] unsafe scratch path: ${task_scratch}" >&2; exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  [[ ! -e "${task_scratch}" ]] || status=2
  echo "[bernini-egntc] exit=${status} mode=${mode} output=${output_dir}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/input" \
  "${task_scratch}/bin" \
  "${task_scratch}/pycache" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

cp -- "${preview_manifest}" "${task_scratch}/input/preview_manifest.jsonl"
cp -- "${vae_index}" "${task_scratch}/input/vae_index.jsonl"
cp -- "${source_archive}" "${task_scratch}/source.tar"
chmod a-w -- "${task_scratch}/input/preview_manifest.jsonl" "${task_scratch}/input/vae_index.jsonl" "${task_scratch}/source.tar"
[[ "$(sha256sum "${task_scratch}/source.tar" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged source archive hash differs"
[[ "$(git get-tar-commit-id <"${task_scratch}/source.tar")" == "${source_revision}" ]] || fail "staged source archive revision differs"
tar --no-same-owner --no-same-permissions -xf "${task_scratch}/source.tar" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"

for required in \
  generator_native_trajectory_controller.py \
  train_generator_native_trajectory_controller.py \
  infer_generator_native_trajectory_controller.py \
  train_lora.py \
  train_fewshot_motion_code.py \
  fewshot_episode_io.py \
  fewshot_episode_parallel.py \
  fewshot_teacher_objective.py \
  inference_sigma_strata.py \
  tri_branch_unipc.py \
  tools/ffprobe_pyav_compat.py \
  audits/bernini_r13_ff4c5d4_checkpoint.sha256 \
  configs/epmc_sit_turn_head_k2_v1.json \
  scripts/auh_train_generator_native_trajectory_controller_canary.sbatch \
  tests/test_generator_native_trajectory_controller.py \
  tests/test_train_generator_native_trajectory_controller_contract.py \
  tests/test_infer_generator_native_trajectory_controller_contract.py \
  tests/test_auh_train_generator_native_trajectory_controller_launcher.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks ${required}"
done
[[ -z "$(find "${method_root}" -type l -print -quit)" ]] || fail "source archive contains symlink"
ffprobe_helper="${method_root}/tools/ffprobe_pyav_compat.py"
[[ "$(sha256sum "${ffprobe_helper}" | awk '{print $1}')" == "${expected_ffprobe_sha256}" ]] || fail "ffprobe shim hash differs"
cp -- "${ffprobe_helper}" "${task_scratch}/bin/ffprobe"
chmod 0555 "${task_scratch}/bin/ffprobe"
[[ "$(stat -c '%a' "${task_scratch}/bin/ffprobe")" == 555 ]] || fail "ffprobe shim mode differs"
export PATH="${task_scratch}/bin:${PATH}"
hash -r
[[ "$(command -v ffprobe)" == "${task_scratch}/bin/ffprobe" ]] || fail "ffprobe shim not active"
[[ "${BERNINI_EPMC_PYTHON_BIN}" == "${python_bin}" ]] || fail "ffprobe runtime Python binding differs"
find "${method_root}" -type f -exec chmod a-w -- {} +

k2_config="${method_root}/configs/epmc_sit_turn_head_k2_v1.json"
[[ "$(sha256sum "${k2_config}" | awk '{print $1}')" == "${expected_config_sha256}" ]] || fail "K2 config hash differs"
checkpoint_manifest="${method_root}/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
validate_checkpoint_content "${checkpoint_manifest}"
PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${bernini_root}" "${veomni_root}" "${bernini_commit}" "${veomni_commit}" <<'PY'
from pathlib import Path
import sys

import train_lora

bernini, veomni, bernini_revision, veomni_revision = train_lora.validate_source_trees(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_bernini_commit=sys.argv[3],
    expected_veomni_commit=sys.argv[4],
)
if bernini_revision != sys.argv[3] or veomni_revision != sys.argv[4]:
    raise SystemExit("source tree revision differs")
print(
    "[bernini-egntc] SOURCE_TREES_OK "
    f"bernini={bernini_revision} veomni={veomni_revision}"
)
PY
mkdir -p -- "$(dirname -- "${output_dir}")"

echo "[bernini-egntc] host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[bernini-egntc] revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-egntc] mode=${mode} world=8 DP=2 Ulysses=4 supports_parallel=2 frames=81 latent_phases=21"
echo "[bernini-egntc] train_rounds=$([[ "${mode}" == formal ]] && echo 2 || echo 1) formal_postfit_prototype_rollout=$([[ "${mode}" == formal ]] && echo true || echo false)"
echo "[bernini-egntc] per_rollout_steps=40 forwards_per_support_rollout=120 full_resolution_cache=1x16x21x60x62"
echo "[bernini-egntc] preview_only=true scientific_claim=false deployable=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in \
  test_generator_native_trajectory_controller.py \
  test_train_generator_native_trajectory_controller_contract.py \
  test_infer_generator_native_trajectory_controller_contract.py \
  test_auh_train_generator_native_trajectory_controller_launcher.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${method_root}/train_generator_native_trajectory_controller.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --k2-config "${k2_config}" \
  --expected-k2-config-sha256 "${expected_config_sha256}" \
  --preview-manifest "${task_scratch}/input/preview_manifest.jsonl" \
  --vae-index "${task_scratch}/input/vae_index.jsonl" \
  --output "${output_dir}" \
  --num-frames 81 \
  --k-shot 2 \
  --num-inference-steps 40 \
  --optimizer-steps-per-round "${optimizer_steps_per_round}" \
  --learning-rate "${learning_rate}" \
  --max-grad-norm "${max_grad_norm}" \
  --seed "${training_seed}" \
  --rollout-seed "${rollout_seed}" \
  --teacher-sigma-index 20 \
  --ack-preview-experimental-only \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}" \
  "${mode_args[@]}"

for artifact in diagnostics.pt controller.safetensors run.receipt.json controller.receipt.json; do
  [[ -f "${output_dir}/${artifact}" && ! -L "${output_dir}/${artifact}" ]] || fail "missing ${artifact}"
done
[[ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 4 ]] || fail "unexpected output file count"

PYTHONPATH="${method_root}" "${python_bin}" -B - \
  "${output_dir}" "${mode}" "${source_revision}" \
  "${source_archive_sha256}" "${checkpoint_tree_sha256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import infer_generator_native_trajectory_controller as inference
import train_generator_native_trajectory_controller as training


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(sys.argv[1])
mode = sys.argv[2]
source_revision = sys.argv[3]
source_archive_sha256 = sys.argv[4]
checkpoint_tree_sha256 = sys.argv[5]
run_path = root / "run.receipt.json"
companion_path = root / "controller.receipt.json"
diagnostics_path = root / "diagnostics.pt"
state_path = root / "controller.safetensors"
run = json.loads(run_path.read_text(encoding="utf-8"))
companion = json.loads(companion_path.read_text(encoding="utf-8"))
require(isinstance(run, dict), "run receipt is not an object")
declared_run_digest = run.get("receipt_sha256")
unsigned_run = dict(run)
unsigned_run.pop("receipt_sha256", None)
require(
    declared_run_digest == inference.object_sha256(unsigned_run),
    "run receipt self digest differs",
)
run_file_sha256 = file_sha256(run_path)
require(
    run.get("schema_version") == training.RUN_RECEIPT_SCHEMA
    and run.get("method") == training.METHOD_NAME,
    "run schema/method differs",
)
require(
    run.get("diagnostics_sha256") == file_sha256(diagnostics_path),
    "diagnostics hash differs",
)
geometry = run.get("exact_geometry", {})
expected_rounds = 1 if mode == "smoke" else 2
expected_evaluation_rollouts = 0 if mode == "smoke" else 1
require(
    geometry.get("rgb_frames") == 81
    and geometry.get("latent_phases") == 21
    and geometry.get("unipc_steps_per_round") == 40
    and geometry.get("transformer_forwards_per_support_round") == 120
    and geometry.get("rounds") == expected_rounds
    and geometry.get("evaluation_only_full_rollouts") == expected_evaluation_rollouts
    and geometry.get("total_transformer_forwards_per_support")
    == 120 * (expected_rounds + expected_evaluation_rollouts),
    "run geometry differs",
)
require(
    run.get("deployable") is False and run.get("diagnostic_only") is True,
    "canary must remain diagnostic/non-deployable",
)
expected_gate = "NOT_EVALUATED_ENGINEERING_SMOKE" if mode == "smoke" else None
if expected_gate is not None:
    require(run.get("representability_gate") == expected_gate, "smoke gate differs")
if mode == "formal":
    require(run.get("representability_gate") in ("GO", "NO_GO"), "formal gate differs")
    require(
        run.get("post_refit_prototype_evaluation_rollout") is True,
        "formal run lacks post-refit prototype rollout",
    )
gate = run.get("gate", {})
require(
    isinstance(gate, dict)
    and gate.get("representability_gate") == run.get("representability_gate")
    and gate.get("deployable") is False
    and gate.get("diagnostic_only") is True,
    "run gate fields differ",
)
distributed = run.get("distributed", {})
require(
    distributed.get("world_size") == 8
    and distributed.get("ulysses_size") == 4
    and distributed.get("data_parallel_size") == 2
    and distributed.get("support_parallel") is True
    and distributed.get("sp_groups") == [[0, 1, 2, 3], [4, 5, 6, 7]]
    and distributed.get("dp_groups") == [[0, 4], [1, 5], [2, 6], [3, 7]]
    and distributed.get("cross_support_gradient_sync") is False,
    "WORLD8 DP2xSP4 receipt differs",
)
support_iids = run.get("support_iids")
supports = run.get("support")
evaluations = run.get("support_evaluations")
require(
    isinstance(support_iids, list)
    and len(support_iids) == 2
    and len(set(support_iids)) == 2
    and isinstance(supports, list)
    and len(supports) == 2
    and isinstance(evaluations, list)
    and len(evaluations) == 2,
    "K=2 support envelope differs",
)
for support_index, (iid, support, evaluation) in enumerate(
    zip(support_iids, supports, evaluations)
):
    require(
        support.get("iid") == iid
        and support.get("support_index") == support_index + 1
        and support.get("dp_rank") == support_index
        and support.get("sp_ranks")
        == ([0, 1, 2, 3] if support_index == 0 else [4, 5, 6, 7]),
        "support topology/identity differs",
    )
    rounds = support.get("rounds")
    require(isinstance(rounds, list) and len(rounds) == expected_rounds, "round count differs")
    for round_index, round_receipt in enumerate(rounds, start=1):
        expected_policy = "official_action" if round_index == 1 else "learned_controller"
        trace = round_receipt.get("trace", {})
        require(
            round_receipt.get("round_index") == round_index
            and round_receipt.get("execution_policy") == expected_policy
            and round_receipt.get("step_count") == 40
            and round_receipt.get("transformer_forwards") == 120
            and round_receipt.get("target_seen_by_callback") is False
            and round_receipt.get("target_used_by_rollout") is False
            and trace.get("original_unipc_scheduler_calls") == 40
            and trace.get("official_action_apg_certified_each_step") is True,
            "support rollout trace differs",
        )
        controller_execution = round_receipt.get("controller_execution")
        if round_index == 1:
            require(controller_execution is None, "official-action round used a controller")
        else:
            inference.egntc.validate_controller_receipt(
                controller_execution, require_complete=True
            )
    prototype_rollout = support.get("prototype_evaluation_rollout")
    require(
        evaluation.get("iid") == iid
        and evaluation.get("evaluation_is_post_refit_prototype_on_policy")
        is (mode == "formal"),
        "support evaluation identity/policy differs",
    )
    if mode == "smoke":
        require(prototype_rollout is None, "smoke unexpectedly has prototype rollout")
    else:
        prototype_trace = prototype_rollout.get("trace", {})
        require(
            prototype_rollout.get("round_index") == 3
            and prototype_rollout.get("execution_policy") == "learned_controller"
            and prototype_rollout.get("step_count") == 40
            and prototype_rollout.get("transformer_forwards") == 120
            and prototype_trace.get("original_unipc_scheduler_calls") == 40,
            "formal prototype rollout trace differs",
        )
        inference.egntc.validate_controller_receipt(
            prototype_rollout.get("controller_execution"), require_complete=True
        )

versions = run.get("versions", {})
require(
    versions.get("method_source_revision") == source_revision
    and versions.get("method_source_archive_sha256") == source_archive_sha256
    and versions.get("checkpoint_tree_sha256") == checkpoint_tree_sha256,
    "run provenance differs",
)
require(
    run.get("controller_contract") == inference.egntc.controller_contract(),
    "controller runtime contract differs",
)
source_paths = {
    "runner": Path(training.__file__),
    "controller": Path(inference.egntc.__file__),
    "episode_trainer": Path(training.episode_train.__file__),
    "tri_branch_unipc": Path(training.tri.__file__),
}
require(
    run.get("source_file_sha256")
    == {name: file_sha256(path) for name, path in source_paths.items()},
    "executed source file hashes differ",
)

state_sha256 = file_sha256(state_path)
companion_sha256 = file_sha256(companion_path)
bundle = inference.load_controller_bundle(
    state_path,
    companion_path,
    expected_state_sha256=state_sha256,
    expected_receipt_sha256=companion_sha256,
    allow_diagnostic_no_go=True,
)
provenance = companion.get("training_provenance", {})
require(
    provenance.get("training_run_receipt_sha256") == run_file_sha256
    and provenance.get("support_iids") == support_iids
    and provenance.get("representability_gate") == run.get("representability_gate")
    and provenance.get("deployable") is False
    and bundle.representability_gate == run.get("representability_gate")
    and bundle.deployable is False,
    "companion/run provenance differs",
)
state = companion.get("controller_state", {})
prototype = run.get("prototype", {})
require(
    state.get("filename") == state_path.name
    and state.get("file_sha256") == state_sha256
    and prototype.get("filename") == state_path.name
    and prototype.get("sha256") == state_sha256
    and prototype.get("tensor_keys") == [inference.CONTROLLER_TENSOR_KEY]
    and prototype.get("shape") == [36]
    and prototype.get("dtype") == "torch.float32",
    "controller checkpoint/run state differs",
)
print(
    "[bernini-egntc] STRONG_AUDIT_OK "
    f"mode={mode} gate={run.get('representability_gate')} "
    f"world=8 dp=2 sp=4 deployable=false"
)
PY

echo "[bernini-egntc] COMPLETE mode=${mode} output=${output_dir}"
