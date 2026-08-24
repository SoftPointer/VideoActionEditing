#!/usr/bin/env bash
#SBATCH --job-name=bernini-orbit-role-2x4
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

# Exact40 only: WORLD4 GPUs 0-3 run the preregistered dog cell while WORLD4
# GPUs 4-7 run the preregistered human cell.  Each group generates one fresh
# frozen T2V donor, then decodes matched base/orbit-adapter exact81 outputs.

source_archive="${ROLE_COMPOSITION_SOURCE_ARCHIVE:?set ROLE_COMPOSITION_SOURCE_ARCHIVE}"
source_archive_sha256="${ROLE_COMPOSITION_SOURCE_ARCHIVE_SHA256:?set ROLE_COMPOSITION_SOURCE_ARCHIVE_SHA256}"
source_revision="${ROLE_COMPOSITION_SOURCE_REVISION:?set ROLE_COMPOSITION_SOURCE_REVISION}"
cell_spec="${ROLE_COMPOSITION_CELL_SPEC:?set ROLE_COMPOSITION_CELL_SPEC}"
cell_spec_sha256="${ROLE_COMPOSITION_CELL_SPEC_SHA256:?set ROLE_COMPOSITION_CELL_SPEC_SHA256}"
adapter="${IDENTITY_ORBIT_ADAPTER:?set IDENTITY_ORBIT_ADAPTER}"
adapter_sha256="${IDENTITY_ORBIT_ADAPTER_SHA256:?set IDENTITY_ORBIT_ADAPTER_SHA256}"
training_receipt="${IDENTITY_ORBIT_TRAINING_RECEIPT:?set IDENTITY_ORBIT_TRAINING_RECEIPT}"
training_receipt_sha256="${IDENTITY_ORBIT_TRAINING_RECEIPT_SHA256:?set IDENTITY_ORBIT_TRAINING_RECEIPT_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_root="${ROLE_COMPOSITION_OUTPUT_ROOT:?set ROLE_COMPOSITION_OUTPUT_ROOT}"
python_bin="${ROLE_COMPOSITION_PYTHON_BIN:?set ROLE_COMPOSITION_PYTHON_BIN}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"

fail() {
  echo "[orbit-role-dual4] ERROR: $*" >&2
  exit 2
}

require_absolute() {
  local value="$1"
  local label="$2"
  [[ "${value}" == /* ]] || fail "${label} must be absolute"
}

for name in \
  source_archive cell_spec adapter training_receipt bernini_root veomni_root \
  checkpoint checkpoint_manifest output_root python_bin
do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${cell_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "cell spec SHA-256 differs"
[[ "${adapter_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "adapter SHA-256 differs"
[[ "${training_receipt_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "training receipt SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
cell_spec="$(realpath -e -- "${cell_spec}")"
adapter="$(realpath -e -- "${adapter}")"
training_receipt="$(realpath -e -- "${training_receipt}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"
launcher_path="$(realpath -e -- "$0")"

for path in "${source_archive}" "${cell_spec}" "${adapter}" "${training_receipt}" "${checkpoint_manifest}" "${python_bin}"
do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "plain input file required: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint}"
do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "plain input directory required: ${path}"
done
[[ -x "${python_bin}" ]] || fail "Python runtime is not executable"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive changed"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${cell_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "cell spec changed"
[[ "$(sha256sum "${adapter}" | awk '{print $1}')" == "${adapter_sha256}" ]] || fail "adapter changed"
[[ "$(sha256sum "${training_receipt}" | awk '{print $1}')" == "${training_receipt_sha256}" ]] || fail "training receipt changed"

[[ "${output_root}" != / ]] || fail "output cannot be filesystem root"
[[ "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe output basename"
[[ "$(realpath -m -- "${output_root}")" == "${output_root}" ]] || fail "output must be canonical"
output_parent="$(dirname -- "${output_root}")"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "refusing reused output"

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

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Slurm job ID"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/orbit-role-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""
human_pid=""

cleanup() {
  local status=$?
  local cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in
    "${scratch_parent%/}/orbit-role-${SLURM_JOB_ID}."*) ;;
    *) echo "[orbit-role-dual4] refusing unsafe scratch cleanup" >&2; cleanup_failed=1 ;;
  esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  if [[ "${cleanup_failed}" != 0 && "${status}" == 0 ]]; then status=2; fi
  if [[ "${status}" == 0 ]]; then
    echo "[orbit-role-dual4] PASS all8=true topology=dog-WORLD4+human-WORLD4 exact81=true exact40=true output=${output_root}"
  else
    echo "[orbit-role-dual4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "staged archive changed"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/infer_identity_orbit_role_composition_v1.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_native_multivideo_motion_donor_oracle.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_native_target_adapter.py",
    "methods/bernini_action_editing/source_self_runtime.py",
    "methods/bernini_action_editing/source_self_identity_orbit_v4.py",
    "methods/bernini_action_editing/train_source_self_identity_orbit_v4.py",
    "methods/bernini_action_editing/appearance_counterfactual_identity_orbit.py",
    "methods/bernini_action_editing/tri_branch_unipc.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/tools/materialize_appearance_counterfactual_identity_orbit.py",
    "methods/bernini_action_editing/assets/identity_orbit_heldout_role_composition_core2_v1.json",
    "methods/bernini_action_editing/scripts/auh_infer_identity_orbit_role_composition_dual4.sbatch",
}
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit("archive member escaped repository")
        normalized = path.as_posix().lstrip("./")
        if normalized.startswith("methods/bernini_action_editing/"):
            if member.issym() or member.islnk() or member.isfifo() or member.isdev():
                raise SystemExit("method closure contains a link/device")
            seen.add(normalized)
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"archive lacks role-composition closure: {missing}")
PY

tar --no-same-owner --no-same-permissions \
  -xf "${archive_copy}" -C "${task_scratch}/source-tree" \
  methods/bernini_action_editing
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
runtime_path="${method_root}/infer_identity_orbit_role_composition_v1.py"
staged_launcher="${method_root}/scripts/auh_infer_identity_orbit_role_composition_dual4.sbatch"
staged_spec="${method_root}/assets/identity_orbit_heldout_role_composition_core2_v1.json"
[[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] || fail "staged runtime is absent"
[[ -f "${staged_launcher}" && ! -L "${staged_launcher}" ]] || fail "staged launcher is absent"
[[ -f "${staged_spec}" && ! -L "${staged_spec}" ]] || fail "staged core2 spec is absent"
find "${method_root}" -type f -exec chmod a-w -- {} +
launcher_sha256="$(sha256sum "${staged_launcher}" | awk '{print $1}')"
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${launcher_sha256}" ]] || fail "executed launcher differs from archive"
[[ "$(sha256sum "${staged_spec}" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "archive core2 spec differs"

mkdir -- "${output_root}"
mkdir -- "${output_root}/logs"
cp -- "${cell_spec}" "${output_root}/sealed-core2-spec.json"
chmod 0444 -- "${output_root}/sealed-core2-spec.json"
[[ "$(sha256sum "${output_root}/sealed-core2-spec.json" | awk '{print $1}')" == "${cell_spec_sha256}" ]] || fail "published spec changed"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((30000 + 2 * job_mod))
human_port=$((dog_port + 1))

launch_group() (
  set -Eeuo pipefail
  local label="$1"
  local cell_id="$2"
  local visible_gpus="$3"
  local master_port="$4"
  local group_scratch="${task_scratch}/groups/${label}"
  local group_output="${output_root}/${label}"
  mkdir -p -- \
    "${group_scratch}/cache/miopen-user" "${group_scratch}/cache/miopen-custom" \
    "${group_scratch}/cache/torch-extensions" "${group_scratch}/cache/triton" \
    "${group_scratch}/cache/xdg" "${group_scratch}/cache/pycache" "${group_scratch}/tmp"
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
  export BERNINI_OUTPUT_TRANSACTION_ID="${SLURM_JOB_ID}-${label}-orbit-role-exact40"
  exec "${python_bin}" -B -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port="${master_port}" \
    "${runtime_path}" \
    --cell-spec "${output_root}/sealed-core2-spec.json" \
    --expected-cell-spec-sha256 "${cell_spec_sha256}" \
    --cell-id "${cell_id}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --adapter "${adapter}" \
    --adapter-training-receipt "${training_receipt}" \
    --expected-adapter-sha256 "${adapter_sha256}" \
    --expected-adapter-training-receipt-sha256 "${training_receipt_sha256}" \
    --output-dir "${group_output}" \
    --num-inference-steps 40 \
    --runtime-source-revision "${source_revision}" \
    --runtime-source-archive-sha256 "${source_archive_sha256}" \
    --launcher-source-sha256 "${launcher_sha256}" \
    --expected-bernini-commit "${bernini_commit}" \
    --expected-veomni-commit "${veomni_commit}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"
)

launch_group dog dog-fit 0,1,2,3 "${dog_port}" \
  >"${output_root}/logs/dog.out" 2>"${output_root}/logs/dog.err" &
dog_pid=$!
launch_group human human-fit 4,5,6,7 "${human_port}" \
  >"${output_root}/logs/human.out" 2>"${output_root}/logs/human.err" &
human_pid=$!

dog_status=0
human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
[[ "${dog_status}" == 0 && "${human_status}" == 0 ]] || fail "WORLD4 group failed: dog=${dog_status} human=${human_status}"

"${python_bin}" -B - \
  "${output_root}" "${source_revision}" "${source_archive_sha256}" \
  "${launcher_sha256}" "${cell_spec_sha256}" "${adapter_sha256}" \
  "${training_receipt_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
revision, archive_sha, launcher_sha, spec_sha, adapter_sha, training_sha = sys.argv[2:]
rows = {}
for label, cell_id in (("dog", "dog-fit"), ("human", "human-fit")):
    path = root / label / "receipt.json"
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} receipt absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    sampling = receipt.get("sampling", {})
    composition = receipt.get("composition", {})
    routes = composition.get("route_traces", {})
    outputs = receipt.get("outputs", {})
    adapter = receipt.get("adapter", {})
    publication = adapter.get("publication", {})
    loaded = adapter.get("loaded", {})
    if not (
        receipt.get("schema_version") == "bernini-identity-orbit-heldout-role-composition-receipt-v1"
        and receipt.get("stage") == "matched_exact40_qualitative_pilot"
        and receipt.get("cell_spec", {}).get("cell", {}).get("cell_id") == cell_id
        and receipt.get("cell_spec", {}).get("file_sha256") == spec_sha
        and sampling.get("exact81") is True
        and sampling.get("frame_count") == 81
        and sampling.get("latent_phases") == 21
        and sampling.get("num_inference_steps") == 40
        and composition.get("rho") == 0.0
        and composition.get("same_official_gaussian_base_and_adapter") is True
        and composition.get("same_donor_refs_text_scheduler_guidance_base_and_adapter") is True
        and routes.get("base", {}).get("adapter_explicitly_disabled_on_every_native_coordinate") is True
        and routes.get("base", {}).get("step_count") == 40
        and routes.get("base", {}).get("guidance_forward_count") == 160
        and routes.get("orbit-adapter", {}).get("adapter_active_on_every_native_coordinate") is True
        and routes.get("orbit-adapter", {}).get("step_count") == 40
        and routes.get("orbit-adapter", {}).get("guidance_forward_count") == 160
        and publication.get("adapter_sha256") == adapter_sha
        and publication.get("training_receipt_sha256") == training_sha
        and loaded.get("strict_file_to_live_load") is True
        and loaded.get("all_parameters_frozen_after_load") is True
        and receipt.get("fresh_t2v_donor", {}).get("condition_only") is True
        and receipt.get("fresh_t2v_donor", {}).get("external_media") is False
        and receipt.get("interpretation", {}).get("older_calibration_bank_consumed") is False
        and receipt.get("interpretation", {}).get("training_performed") is False
        and receipt.get("runtime_source", {}).get("revision") == revision
        and receipt.get("runtime_source", {}).get("archive_sha256") == archive_sha
        and receipt.get("runtime_source", {}).get("launcher_sha256") == launcher_sha
    ):
        raise SystemExit(f"{label} receipt contract differs")
    if set(outputs) != {"t2v-donor", "base", "orbit-adapter"}:
        raise SystemExit(f"{label} output arm closure differs")
    for arm, output in outputs.items():
        media = Path(output["path"])
        if (
            output.get("frame_count") != 81
            or output.get("fps") != 25
            or not media.is_file()
            or media.is_symlink()
            or hashlib.sha256(media.read_bytes()).hexdigest() != output.get("sha256")
        ):
            raise SystemExit(f"{label}/{arm} decoded exact81 artifact differs")
    rows[label] = {
        "cell_id": cell_id,
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "receipt_digest": receipt["receipt_digest"],
        "outputs": outputs,
    }

aggregate = {
    "schema_version": "bernini-identity-orbit-heldout-role-composition-dual4-receipt-v1",
    "runtime_source_revision": revision,
    "runtime_source_archive_sha256": archive_sha,
    "launcher_sha256": launcher_sha,
    "cell_spec_sha256": spec_sha,
    "adapter_sha256": adapter_sha,
    "adapter_training_receipt_sha256": training_sha,
    "topology": {
        "allocated_gpus": 8,
        "groups": {"dog": [0, 1, 2, 3], "human": [4, 5, 6, 7]},
        "world_size_per_group": 4,
        "ulysses_size_per_group": 4,
        "all_eight_gpus_used": True,
    },
    "exact81": True,
    "exact40": True,
    "rho": 0.0,
    "base_adapter_matched_pairs": 2,
    "groups": rows,
    "scientific_claim_authorized": False,
}
payload = json.dumps(aggregate, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
aggregate["receipt_digest"] = hashlib.sha256(payload).hexdigest()
path = root / "receipt.json"
if path.exists() or path.is_symlink():
    raise SystemExit("aggregate receipt path reused")
with path.open("x", encoding="ascii") as handle:
    json.dump(aggregate, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(path, 0o400)
PY

[[ "$(sha256sum "${adapter}" | awk '{print $1}')" == "${adapter_sha256}" ]] || fail "adapter changed after decode"
[[ "$(sha256sum "${training_receipt}" | awk '{print $1}')" == "${training_receipt_sha256}" ]] || fail "training receipt changed after decode"
echo "[orbit-role-dual4] outputs=${output_root}/dog/{t2v-donor,base,orbit-adapter}.mp4,${output_root}/human/{t2v-donor,base,orbit-adapter}.mp4"
