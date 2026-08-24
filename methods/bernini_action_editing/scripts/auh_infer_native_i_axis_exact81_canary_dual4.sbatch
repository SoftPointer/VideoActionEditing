#!/usr/bin/env bash
#SBATCH --job-name=bernini-i-axis-2xsp4
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=36:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Two isolated WORLD4/SP4 jobs use all eight GPUs concurrently.  Dog runs on
# 0--3 and human on 4--7; each group runs seven arms for two sealed seeds.

repo_root="${NATIVE_I_AXIS_REPO_ROOT:?set repository root}"
cell_spec="${NATIVE_I_AXIS_CELL_SPEC:?set sealed core2 spec}"
cell_spec_sha256="${NATIVE_I_AXIS_CELL_SPEC_SHA256:?set spec SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set Bernini checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint manifest}"
output_root="${NATIVE_I_AXIS_OUTPUT_ROOT:?set fresh output root}"
python_bin="${NATIVE_I_AXIS_PYTHON_BIN:?set vace Python executable}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() {
  echo "[native-i-axis-dual4] ERROR: $*" >&2
  exit 2
}

for value in "${repo_root}" "${cell_spec}" "${bernini_root}" "${veomni_root}" "${checkpoint}" "${checkpoint_manifest}" "${output_root}" "${python_bin}"; do
  [[ "${value}" == /* ]] || fail "all paths must be absolute"
done
[[ "${cell_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "spec SHA-256 differs"

repo_root="$(realpath -e -- "${repo_root}")"
cell_spec="$(realpath -e -- "${cell_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
for path in "${cell_spec}" "${checkpoint_manifest}" "${python_bin}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain file required: ${path}"
done
for path in "${repo_root}" "${bernini_root}" "${veomni_root}" "${checkpoint}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain directory required: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "cell spec changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed"

method_root="${repo_root}/methods/bernini_action_editing"
runtime_path="${method_root}/infer_native_i_axis_exact81_canary.py"
core_path="${method_root}/native_i_axis_guidance.py"
running_launcher_path="$(realpath -e -- "$0")"
repo_launcher_path="${method_root}/scripts/auh_infer_native_i_axis_exact81_canary_dual4.sbatch"
closure_files=(
  "${runtime_path}"
  "${core_path}"
  "${method_root}/infer_native_identity_generation_canary.py"
  "${method_root}/infer_orderless_source_frame_set_noise_canary.py"
  "${method_root}/tri_branch_unipc.py"
  "${method_root}/infer_lora.py"
  "${method_root}/train_lora.py"
  "${method_root}/infer_source_kv_carrier_oracle.py"
  "${method_root}/infer_source_value_residual_oracle.py"
  "${cell_spec}"
  "${repo_launcher_path}"
)
for path in "${closure_files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "method source absent: ${path}"
done
[[ -f "${running_launcher_path}" && ! -L "${running_launcher_path}" ]] || fail "running Slurm launcher is not a plain file"
[[ "$(sha256sum "${running_launcher_path}" | awk '{print $1}')" == "$(sha256sum "${repo_launcher_path}" | awk '{print $1}')" ]] || fail "running Slurm launcher differs from repository launcher"
source_revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "repository revision differs"
launcher_sha256="$(sha256sum "${running_launcher_path}" | awk '{print $1}')"
source_closure_sha256="$(sha256sum "${closure_files[@]}" | sha256sum | awk '{print $1}')"

[[ "${output_root}" != / ]] || fail "output cannot be root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe output basename"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
output_parent="$(realpath -e -- "${output_parent}")"
[[ "${output_root}" == "${output_parent}/${output_root##*/}" ]] || fail "output is not canonical"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output must be fresh"

for test_file in test_native_i_axis_guidance.py test_infer_native_i_axis_exact81_canary.py test_auh_infer_native_i_axis_exact81_canary_launcher.py; do
  [[ -f "${method_root}/tests/${test_file}" && ! -L "${method_root}/tests/${test_file}" ]] || fail "test absent: ${test_file}"
  PYTHONPATH="${method_root}" "${python_bin}" -B "${method_root}/tests/${test_file}"
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

[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/native-i-axis-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/native-i-axis-${SLURM_JOB_ID}."*) ;;
    *) exit 2 ;;
  esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}"
    rm -rf -- "${task_scratch}"
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[native-i-axis-dual4] PASS all8=true dog-SP4=true human-SP4=true exact81=true exact40=true candidates=28 output=${output_root}"
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
cp -- "${cell_spec}" "${output_root}/sealed-native-i-axis-core2.json"
chmod 0444 -- "${output_root}/sealed-native-i-axis-core2.json"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((36000 + 2 * job_mod))
human_port=$((dog_port + 1))

launch_group() (
  set -Eeuo pipefail
  local label="$1"
  local visible_gpus="$2"
  local master_port="$3"
  local group_scratch="${task_scratch}/${label}"
  mkdir -p -- "${group_scratch}/miopen-user" "${group_scratch}/miopen-custom" "${group_scratch}/torch-extensions" "${group_scratch}/triton" "${group_scratch}/xdg" "${group_scratch}/pycache" "${group_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${group_scratch}/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${group_scratch}/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${group_scratch}/torch-extensions"
  export TRITON_CACHE_DIR="${group_scratch}/triton"
  export XDG_CACHE_HOME="${group_scratch}/xdg"
  export PYTHONPYCACHEPREFIX="${group_scratch}/pycache"
  export TMPDIR="${group_scratch}/tmp"
  exec "${python_bin}" -B -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --cell-spec "${output_root}/sealed-native-i-axis-core2.json" \
    --expected-cell-spec-sha256 "${cell_spec_sha256}" \
    --cell-id "${label}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --output-dir "${output_root}/${label}" \
    --num-inference-steps 40 \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-closure-sha256 "${source_closure_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}"
)

echo "[native-i-axis-dual4] GPUs0-3=dog GPUs4-7=human arms=N-C,N-W,G-C,G-W,G-P,G-D,G-S seeds=2"
launch_group dog 0,1,2,3 "${dog_port}" >"${output_root}/logs/dog.out" 2>"${output_root}/logs/dog.err" &
dog_pid=$!
launch_group human 4,5,6,7 "${human_port}" >"${output_root}/logs/human.out" 2>"${output_root}/logs/human.err" &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "WORLD4 group failed dog=${dog_status} human=${human_status}"

"${python_bin}" -B - "${output_root}" "${source_revision}" "${source_closure_sha256}" "${launcher_sha256}" "${cell_spec_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
revision, closure_sha, launcher_sha, spec_sha = sys.argv[2:]
arms = ["N-C", "N-W", "G-C", "G-W", "G-P", "G-D", "G-S"]
gated = {"G-C", "G-W", "G-P", "G-D", "G-S"}


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


children = {}
for label in ("dog", "human"):
    path = root / label / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.get("receipt_digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    sampling = receipt.get("sampling", {})
    interpretation = receipt.get("interpretation", {})
    candidates = receipt.get("candidates", [])
    traces = receipt.get("traces", {})
    outputs = receipt.get("outputs", {})
    seeds = sampling.get("seeds", [])
    if not (
        digest(unsigned) == claimed
        and receipt.get("schema_version") == "bernini-native-i-axis-exact81-canary-receipt-v1"
        and receipt.get("cell_spec", {}).get("file_sha256") == spec_sha
        and receipt.get("cell_spec", {}).get("cell", {}).get("cell_id") == label
        and receipt.get("runtime_source", {}).get("revision") == revision
        and receipt.get("runtime_source", {}).get("closure_sha256") == closure_sha
        and receipt.get("runtime_source", {}).get("launcher_sha256") == launcher_sha
        and sampling.get("exact40") is True and sampling.get("exact81") is True
        and sampling.get("arm_order") == arms and len(seeds) == 2
        and len(candidates) == 14 and len(traces) == 14 and len(outputs) == 14
        and interpretation.get("training_performed") is False
        and interpretation.get("optimizer") is None
        and interpretation.get("apg") is False
        and interpretation.get("best_arm_selected") is False
    ):
        raise SystemExit(f"{label} child receipt gate failed")
    noise_by_seed = {seed: set() for seed in seeds}
    seen = set()
    for candidate in candidates:
        arm, seed, key = candidate["arm"], candidate["seed"], candidate["candidate_key"]
        if arm not in arms or seed not in seeds or key != f"seed-{seed}__{arm}" or key in seen:
            raise SystemExit(f"{label} candidate registry differs")
        seen.add(key)
        noise_by_seed[seed].add(candidate["official_initial_gaussian_raw_value_sha256"])
        steps = traces[key]["steps"]
        expected_forwards = 200 if arm in {"G-C", "G-W", "G-P", "G-S"} else 160
        if not (
            len(steps) == 40
            and traces[key]["observed_transformer_forwards"] == expected_forwards
            and [row["step_index"] for row in steps if row["gate_active"]] == ([33, 34, 35, 36, 37] if arm in gated else [])
            and all(steps[index]["final_native_parity"] for index in (38, 39))
            and all(steps[index]["scheduler_received_original_model_output_object"] for index in (38, 39))
            and all(row["native_formula_exact_parity"] for row in steps)
            and all(row["original_scheduler_call_count"] == 1 for row in steps)
        ):
            raise SystemExit(f"{label}/{key} step gate failed")
        if arm in {"N-C", "N-W", "G-D"} and any(row["native_velocity_raw_sha256"] != row["executed_velocity_raw_sha256"] for row in steps):
            raise SystemExit(f"{label}/{key} native control changed velocity")
        output = outputs[key]
        mp4 = Path(output["path"])
        latent_node = output["normalized_clean_latent"]
        latent = Path(latent_node["path"])
        if not (
            mp4.is_file() and latent.is_file()
            and not mp4.is_symlink() and not latent.is_symlink()
            and file_sha(mp4) == output["sha256"]
            and file_sha(latent) == latent_node["sha256"]
            and output["frame_count"] == 81 and output["fps"] == 25
            and latent_node["native_sampler_before_vae_decode"] is True
        ):
            raise SystemExit(f"{label}/{key} artifact gate failed")
    if len(seen) != 14 or any(len(values) != 1 for values in noise_by_seed.values()):
        raise SystemExit(f"{label} common Gaussian gate failed")
    children[label] = {
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha(path),
        "receipt_digest": claimed,
        "candidate_count": 14,
        "world_size": 4,
        "ulysses_size": 4,
    }

unsigned_parent = {
    "schema_version": "bernini-native-i-axis-exact81-all8-dual4-receipt-v1",
    "method": "frozen-bernini-native-i-axis-late-gated-canary-all8",
    "runtime_source_revision": revision,
    "runtime_source_closure_sha256": closure_sha,
    "launcher_sha256": launcher_sha,
    "cell_spec_sha256": spec_sha,
    "topology": {
        "allocated_gpu_count": 8,
        "groups": {
            "dog": {"visible_gpus": [0, 1, 2, 3], "world_size": 4, "ulysses_size": 4},
            "human": {"visible_gpus": [4, 5, 6, 7], "world_size": 4, "ulysses_size": 4},
        },
        "groups_ran_concurrently": True,
    },
    "arm_order": arms,
    "seed_count_per_cell": 2,
    "total_candidate_count": 28,
    "children": children,
    "exact40": True,
    "exact81": True,
    "training_performed": False,
    "optimizer": None,
    "apg": False,
    "mask_pose_flow_track_trajectory": False,
    "ranking_performed": False,
    "best_arm_selected": False,
    "scientific_claim_authorized": False,
}
parent = {**unsigned_parent, "receipt_digest": digest(unsigned_parent)}
target = root / "all8-receipt.json"
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(json.dumps(parent, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(parent, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
PY

[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "cell spec changed during run"
[[ "$(sha256sum "${output_root}/sealed-native-i-axis-core2.json" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "sealed output spec changed"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest changed during run"
source_closure_after="$(sha256sum "${closure_files[@]}" | sha256sum | awk '{print $1}')"
[[ "${source_closure_after}" == "${source_closure_sha256}" ]] || fail "runtime source closure changed during run"
