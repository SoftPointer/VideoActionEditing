#!/usr/bin/env bash
#SBATCH --job-name=cage-vjp-2xsp4
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Read-only Phase-A CAGE reward probe.  Two frozen Bernini WORLD4/Ulysses-SP4
# processes run concurrently: the preregistered dog-fit row on GPUs 0--3 and
# the preregistered human-fit row on GPUs 4--7.  Each arm consumes one sealed
# native exact81 clean latent plus that same candidate's official Gaussian and
# the matching action+9-caption cell.  It performs the implemented 3-sigma x
# 10-branch scan (30 frozen shared_step calls) and two selected input-VJP
# replays.  It creates no optimizer, takes no parameter update, and consumes no
# target, proposal media, donor, mask, flow, pose, track, or trajectory.
#
# IMPORTANT capability boundary: probe_cage_native_candidate_vjp.py does not
# implement a scan over all 30 transformer blocks and does not select/seal a
# trainable module band.  The strong postflight records both facts as false and
# refuses to authorize a training step.  A later native block-probe must supply
# that separate receipt; 30 branch/sigma model calls are not "30 blocks".

source_archive="${CAGE_PROBE_SOURCE_ARCHIVE:?set CAGE_PROBE_SOURCE_ARCHIVE}"
source_archive_sha256="${CAGE_PROBE_SOURCE_ARCHIVE_SHA256:?set CAGE_PROBE_SOURCE_ARCHIVE_SHA256}"
source_revision="${CAGE_PROBE_SOURCE_REVISION:?set CAGE_PROBE_SOURCE_REVISION}"
population_spec="${CAGE_PROBE_NATIVE_POPULATION_SPEC:?set CAGE_PROBE_NATIVE_POPULATION_SPEC}"
population_spec_sha256="${CAGE_PROBE_NATIVE_POPULATION_SPEC_SHA256:?set CAGE_PROBE_NATIVE_POPULATION_SPEC_SHA256}"
t2v_spec="${CAGE_PROBE_T2V_CORE4_V2_SPEC:?set CAGE_PROBE_T2V_CORE4_V2_SPEC}"
t2v_spec_sha256="${CAGE_PROBE_T2V_CORE4_V2_SPEC_SHA256:?set CAGE_PROBE_T2V_CORE4_V2_SPEC_SHA256}"
rollout_root="${CAGE_PROBE_NATIVE_ROLLOUT_ROOT:?set CAGE_PROBE_NATIVE_ROLLOUT_ROOT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_commit="${BERNINI_OFFICIAL_COMMIT:?set BERNINI_OFFICIAL_COMMIT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
veomni_commit="${BERNINI_VEOMNI_COMMIT:?set BERNINI_VEOMNI_COMMIT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_tree_sha256="${BERNINI_CHECKPOINT_TREE_SHA256:?set BERNINI_CHECKPOINT_TREE_SHA256}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${CAGE_PROBE_OUTPUT_DIR:?set CAGE_PROBE_OUTPUT_DIR}"
python_bin="${CAGE_PROBE_PYTHON_BIN:?set CAGE_PROBE_PYTHON_BIN}"

readonly pinned_population_sha256="525d727951ee05d7aac27f47d294e3604996781106dfc710087d4029a1bbd8f0"
readonly pinned_t2v_sha256="a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
readonly pinned_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
readonly pinned_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
readonly pinned_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
readonly dog_candidate_id="pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080901"
readonly human_candidate_id="pair5-native-core4-v1-a35b590961d24694-action-s2026080901"

fail() { echo "[cage-vjp-dual4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive population_spec t2v_spec rollout_root bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${population_spec_sha256}" == "${pinned_population_sha256}" ]] || fail "native population is not pinned core4-v1"
[[ "${t2v_spec_sha256}" == "${pinned_t2v_sha256}" ]] || fail "T2V prompt authority is not pinned core4-v2 exact40"
[[ "${bernini_commit}" == "${pinned_bernini_commit}" ]] || fail "Bernini commit differs"
[[ "${veomni_commit}" == "${pinned_veomni_commit}" ]] || fail "VeOmni commit differs"
[[ "${checkpoint_tree_sha256}" == "${pinned_checkpoint_tree_sha256}" ]] || fail "checkpoint tree identity differs"
[[ "${checkpoint_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "checkpoint manifest SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
population_spec="$(realpath -e -- "${population_spec}")"
t2v_spec="$(realpath -e -- "${t2v_spec}")"
rollout_root="$(realpath -e -- "${rollout_root}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${population_spec}" && ! -L "${population_spec}" ]] || fail "native population spec differs"
[[ -f "${t2v_spec}" && ! -L "${t2v_spec}" ]] || fail "T2V core4-v2 spec differs"
[[ -d "${rollout_root}" && ! -L "${rollout_root}" ]] || fail "native exact81 rollout root differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini source root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni source root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${population_spec}" | awk '{print $1}')" == "${population_spec_sha256}" ]] || fail "native population spec hash differs"
[[ "$(sha256sum "${t2v_spec}" | awk '{print $1}')" == "${t2v_spec_sha256}" ]] || fail "T2V core4-v2 spec hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"

[[ "${output_dir}" != / ]] || fail "output cannot be root"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
mkdir -p -- "$(dirname -- "${output_dir}")"
[[ -d "$(dirname -- "${output_dir}")" && ! -L "$(dirname -- "${output_dir}")" ]] || fail "output parent differs"

# Bind the launcher itself to the exact40/81f bank geometry and to one fit row
# from each action family before any GPU process or output directory exists.
"${python_bin}" -B - \
  "${population_spec}" "${population_spec_sha256}" \
  "${t2v_spec}" "${t2v_spec_sha256}" \
  "${dog_candidate_id}" "${human_candidate_id}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def reject_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"sealed JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def reject_constant(token):
    raise SystemExit(f"sealed JSON contains non-finite constant {token}")


def load(path_text, expected_sha, label):
    path = Path(path_text)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit(f"{label} raw SHA-256 differs")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} root differs")
    return value


population = load(sys.argv[1], sys.argv[2], "native population")
t2v = load(sys.argv[3], sys.argv[4], "T2V core4-v2 spec")
dog_id, human_id = sys.argv[5:7]
if population.get("schema_version") != "pair-v5-native-rv2v4-rollout-spec-v1":
    raise SystemExit("native population schema differs")
if t2v.get("schema_version") != "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2":
    raise SystemExit("T2V prompt authority schema differs")
native_sampling = population.get("sampling_contract", {})
if (
    native_sampling.get("condition_mode") != "rv2v4"
    or native_sampling.get("num_frames") != 81
    or native_sampling.get("latent_frames") != 21
    or native_sampling.get("num_inference_steps") != 40
    or native_sampling.get("source_reference_indices") != [0, 27, 53, 80]
    or native_sampling.get("target_initialization")
    != "official_gen_wanx22_fresh_gaussian"
):
    raise SystemExit("native population is not sealed exact40/81f RV2V-4")
native_closure = population.get("semantic_input_closure", {})
for key in (
    "target_video", "t2v_proposal_media", "donor_video",
    "external_reference", "mask", "flow", "pose", "track", "trajectory",
):
    if native_closure.get(key) is not False:
        raise SystemExit(f"native semantic closure differs: {key}")
sampling = t2v.get("sampling_contract", {})
if (
    sampling.get("num_frames") != 81
    or sampling.get("latent_frames") != 21
    or sampling.get("num_inference_steps") != 40
    or sampling.get("semantic_branch_order")
    != [
        "action", "noop", "incomplete", "reverse", "shuffle",
        "wrong_actor", "wrong_object", "camera_only", "appearance_only",
        "generic_wrong_motion",
    ]
):
    raise SystemExit("T2V bank is not sealed exact40/81f action+9")
groups = population.get("groups")
if (
    not isinstance(groups, list)
    or [row.get("group_id") for row in groups] != ["sp4-a", "sp4-b"]
    or [row.get("visible_gpus") for row in groups]
    != [[0, 1, 2, 3], [4, 5, 6, 7]]
    or [len(row.get("candidates", [])) for row in groups] != [4, 4]
):
    raise SystemExit("native exact81 population topology differs")
native_rows = {
    row.get("candidate_id"): (group.get("group_id"), row)
    for group in groups
    for row in group.get("candidates", [])
}
expected_native = {
    dog_id: ("sp4-a", 2026080901),
    human_id: ("sp4-b", 2026080901),
}
for candidate_id, (group_id, seed) in expected_native.items():
    actual = native_rows.get(candidate_id)
    if actual is None or actual[0] != group_id or actual[1].get("seed") != seed:
        raise SystemExit(f"preregistered native row differs: {candidate_id}")
t2v_groups = t2v.get("groups")
if (
    not isinstance(t2v_groups, list)
    or [group.get("group_id") for group in t2v_groups] != ["sp4-a", "sp4-b"]
    or [group.get("visible_gpus") for group in t2v_groups]
    != [[0, 1, 2, 3], [4, 5, 6, 7]]
    or [len(group.get("candidates", [])) for group in t2v_groups] != [20, 20]
):
    raise SystemExit("T2V exact40 group topology differs")
t2v_rows = [
    row
    for group in t2v_groups
    for row in group.get("candidates", [])
]
if len(t2v_rows) != 40 or len({row.get("candidate_id") for row in t2v_rows}) != 40:
    raise SystemExit("T2V bank is not exact40")
action_by_caption = {
    row.get("full_t2v_caption_utf8_sha256"): row
    for row in t2v_rows
    if row.get("semantic_branch") == "action"
}
expected_actions = {
    dog_id: ("dog-sit-facing-camera", "fit"),
    human_id: ("human-rise-to-stand", "fit"),
}
for candidate_id, (family, split) in expected_actions.items():
    native = native_rows[candidate_id][1]
    action = action_by_caption.get(native.get("complete_caption_sha256"))
    if (
        action is None
        or action.get("full_t2v_caption") != native.get("complete_caption")
        or action.get("action_family_id") != family
        or action.get("analysis_split") != split
        or action.get("geometry_contract")
        != "decode_exact81_for_bucket_shape_only_never_encode_or_condition"
    ):
        raise SystemExit(f"native/prompt family binding differs: {candidate_id}")
closure = t2v.get("semantic_input_closure", {})
for key in (
    "proposal_media_as_condition", "proposal_media_as_student_input",
    "proposal_media_as_noise", "proposal_media_as_donor_or_pseudo_target",
    "target_video", "mask", "flow", "pose", "track", "trajectory",
):
    if closure.get(key) is not False:
        raise SystemExit(f"T2V semantic closure differs: {key}")
print("CAGE_NATIVE_DUAL4_SEALED_EXACT40_81F_PREFLIGHT_OK")
PY

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/cage-vjp-dual4-${SLURM_JOB_ID}.XXXXXX")"
dog_pid=""; human_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/cage-vjp-dual4-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[cage-vjp-dual4] PASS topology=two-independent-SP4 all8=true exact40_prompt_bank=true exact81_native=true frozen=true update=false branch_sigma_scan_calls=30 transformer_block_scan=false sealed_trainable_band=false output=${output_dir}"
  else
    echo "[cage-vjp-dual4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${dog_pid}" "${human_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/arms"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "scratch source archive hash differs after copy"
[[ "$(git get-tar-commit-id <"${archive_copy}")" == "${source_revision}" ]] || fail "scratch source archive revision differs after copy"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/cage_candidate_action_energy_vjp.py",
    "methods/bernini_action_editing/dclr_runtime_contract.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/inference_sigma_strata.py",
    "methods/bernini_action_editing/mace_candidate_action_energy.py",
    "methods/bernini_action_editing/pair_v5_action_adapter.py",
    "methods/bernini_action_editing/pair_v5_native_bridge.py",
    "methods/bernini_action_editing/pair_v5_native_rollout_spec.py",
    "methods/bernini_action_editing/pair_v5_native_rv2v_action_score_v3.py",
    "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
    "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
    "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
    "methods/bernini_action_editing/probe_cage_native_candidate_vjp.py",
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_native_target_adapter.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/validate_pair_v5_t2v_calibration_mainline_v3.py",
}
if len(required) != 26:
    raise SystemExit("native CAGE audited Python import closure is not exact26")
seen = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
        ):
            raise SystemExit("source archive member escaped or is not a plain tree entry")
        seen.add(path.as_posix().lstrip("./"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"source archive lacks native CAGE probe closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
probe_source="${method_root}/probe_cage_native_candidate_vjp.py"
[[ -f "${probe_source}" && ! -L "${probe_source}" ]] || fail "extracted probe source differs"
find "${method_root}" -type f -exec chmod a-w -- {} +

mkdir -- "${output_dir}"
echo "[cage-vjp-dual4] allocation=8xMI210 topology=dog-fit:0,1,2,3+human-fit:4,5,6,7 world_per_arm=4 ulysses_per_arm=4"
echo "[cage-vjp-dual4] mode=no-update exact81=true native_schedule_indices=20,28,33 action_plus_hard_negatives=10 scan_calls=30 input_vjp_replays=2"
echo "[cage-vjp-dual4] BLOCKER transformer_30_block_parameter_scan=false sealed_trainable_band_receipt=false one_step_training_authorized=false"

job_mod=$((SLURM_JOB_ID % 10000))
dog_port=$((30000 + 2 * job_mod))
human_port=$((dog_port + 1))

run_arm() (
  set -Eeuo pipefail
  local arm_id="$1" visible_gpus="$2" master_port="$3" candidate_id="$4"
  local arm_scratch="${task_scratch}/arms/${arm_id}"
  mkdir -p -- "${arm_scratch}/cache/miopen-user" "${arm_scratch}/cache/miopen-custom" \
    "${arm_scratch}/cache/torch-extensions" "${arm_scratch}/cache/triton" \
    "${arm_scratch}/cache/xdg" "${arm_scratch}/cache/pycache" "${arm_scratch}/tmp"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export ROCR_VISIBLE_DEVICES="${visible_gpus}"
  export MIOPEN_USER_DB_PATH="${arm_scratch}/cache/miopen-user"
  export MIOPEN_CUSTOM_CACHE_DIR="${arm_scratch}/cache/miopen-custom"
  export TORCH_EXTENSIONS_DIR="${arm_scratch}/cache/torch-extensions"
  export TRITON_CACHE_DIR="${arm_scratch}/cache/triton"
  export XDG_CACHE_HOME="${arm_scratch}/cache/xdg"
  export PYTHONPYCACHEPREFIX="${arm_scratch}/cache/pycache"
  export TMPDIR="${arm_scratch}/tmp"
  export TORCHELASTIC_ERROR_FILE="${arm_scratch}/torch-elastic-error.json"
  "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${probe_source}" \
    --population-spec "${population_spec}" \
    --expected-population-spec-sha256 "${population_spec_sha256}" \
    --t2v-spec "${t2v_spec}" \
    --expected-t2v-spec-sha256 "${t2v_spec_sha256}" \
    --rollout-root "${rollout_root}" \
    --candidate-id "${candidate_id}" \
    --bernini-root "${bernini_root}" \
    --expected-bernini-commit "${bernini_commit}" \
    --veomni-root "${veomni_root}" \
    --expected-veomni-commit "${veomni_commit}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-content-manifest "${checkpoint_manifest}" \
    --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
    --output-dir "${output_dir}/${arm_id}" \
    --target-margin 0.0 --temperature 1.0 --ack-probe-not-training
)

run_arm dog-fit "0,1,2,3" "${dog_port}" "${dog_candidate_id}" >"${output_dir}/dog-fit.log" 2>&1 & dog_pid=$!
run_arm human-fit "4,5,6,7" "${human_port}" "${human_candidate_id}" >"${output_dir}/human-fit.log" 2>&1 & human_pid=$!
dog_status=0; human_status=0
wait "${dog_pid}" || dog_status=$?
wait "${human_pid}" || human_status=$?
if [[ "${dog_status}" != 0 || "${human_status}" != 0 ]]; then
  tail -n 200 -- "${output_dir}/dog-fit.log" >&2 || true
  tail -n 200 -- "${output_dir}/human-fit.log" >&2 || true
  fail "one or more frozen SP4 probe arms failed"
fi

# Independent postflight: authenticate both canonical receipts and FP32
# safetensors, bind all identities back to the sealed source/spec/checkpoint,
# and create an aggregate receipt that explicitly blocks training because the
# native 30-block module scan and band selection are not implemented here.
"${python_bin}" -B - \
  "${population_spec}" "${population_spec_sha256}" \
  "${t2v_spec}" "${t2v_spec_sha256}" \
  "${checkpoint_manifest_sha256}" "${checkpoint_tree_sha256}" \
  "${source_revision}" "${source_archive_sha256}" \
  "${bernini_commit}" "${veomni_commit}" \
  "${output_dir}" "${probe_source}" \
  "${dog_candidate_id}" "${human_candidate_id}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open


(
    population_path,
    population_sha,
    t2v_path,
    t2v_sha,
    checkpoint_manifest_sha,
    checkpoint_tree_sha,
    source_revision,
    source_archive_sha,
    bernini_commit,
    veomni_commit,
    output_text,
    probe_source_text,
    dog_id,
    human_id,
) = sys.argv[1:]
output_root = Path(output_text).resolve(strict=True)
probe_source = Path(probe_source_text).resolve(strict=True)


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"postflight plain receipt absent: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"postflight receipt is not strict ASCII JSON: {path}: {error}") from error
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise SystemExit(f"postflight receipt is not canonical: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def verify_embedded(value, field, label):
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    observed = hashlib.sha256(canonical(unsigned)).hexdigest()
    if declared != observed:
        raise SystemExit(f"{label} embedded digest differs")
    return declared


expected = {
    "dog-fit": {
        "candidate_id": dog_id,
        "group_id": "sp4-a",
        "family": "dog-sit-facing-camera",
    },
    "human-fit": {
        "candidate_id": human_id,
        "group_id": "sp4-b",
        "family": "human-rise-to-stand",
    },
}
method_sha = file_sha256(probe_source)
arm_results = []
for arm_id, contract in expected.items():
    arm_root = output_root / arm_id
    receipt_path = arm_root / "cage-native-candidate-vjp-receipt.json"
    cotangent_path = arm_root / "cage-native-candidate-cotangent.safetensors"
    receipt, receipt_file_sha = load_canonical(receipt_path)
    receipt_digest = verify_embedded(receipt, "receipt_digest", f"{arm_id} receipt")
    for nested, field, label in (
        (receipt.get("contract", {}), "digest", "contract"),
        (receipt.get("execution", {}), "digest", "execution"),
        (receipt.get("vjp", {}), "digest", "VJP"),
        (receipt.get("output", {}), "artifact_digest", "cotangent artifact"),
    ):
        verify_embedded(nested, field, f"{arm_id} {label}")
    if receipt.get("schema_version") != "bernini-cage-native-candidate-vjp-probe-v1":
        raise SystemExit(f"{arm_id} schema differs")
    candidate = receipt.get("candidate", {})
    if (
        candidate.get("candidate_id") != contract["candidate_id"]
        or candidate.get("group_id") != contract["group_id"]
        or candidate.get("analysis_split") != "fit"
        or candidate.get("action_family_id") != contract["family"]
        or candidate.get("seed") != 2026080901
    ):
        raise SystemExit(f"{arm_id} candidate/family binding differs")
    sealed = receipt.get("sealed_specs", {})
    if (
        Path(sealed.get("native_population_path", "")).resolve(strict=True)
        != Path(population_path).resolve(strict=True)
        or sealed.get("native_population_sha256") != population_sha
        or Path(sealed.get("t2v_core4_v2_path", "")).resolve(strict=True)
        != Path(t2v_path).resolve(strict=True)
        or sealed.get("t2v_core4_v2_sha256") != t2v_sha
    ):
        raise SystemExit(f"{arm_id} sealed spec binding differs")
    coordinates = receipt.get("coordinates", [])
    if (
        [row.get("schedule_index") for row in coordinates] != [20, 28, 33]
        or [row.get("native_scheduler_timestep") for row in coordinates]
        != [833, 682, 516]
        or any(row.get("legacy_1000_sigma_timestep_rejected") is not True for row in coordinates)
    ):
        raise SystemExit(f"{arm_id} native schedule coordinates differ")
    execution = receipt.get("execution", {})
    if (
        execution.get("scan_shared_step_calls") != 30
        or execution.get("scan_patch_vae_latent_calls") != 3
        or execution.get("scan_all_branches_all_sigmas_under_no_grad") is not True
        or execution.get("scan_shared_x_sigma_per_ten_branches") is not True
        or execution.get("replay_shared_step_calls") != 2
        or execution.get("replay_patch_vae_latent_calls") != 2
        or execution.get("replay_input_graph_enabled") is not True
        or execution.get("pure_t2v_proposal_media_consumed") is not False
    ):
        raise SystemExit(f"{arm_id} 30-scan/2-replay execution differs")
    vjp = receipt.get("vjp", {})
    if (
        vjp.get("gradient_finite") is not True
        or vjp.get("gradient_nonzero") is not True
        or not isinstance(vjp.get("gradient_norm"), (int, float))
        or not math.isfinite(vjp["gradient_norm"])
        or vjp["gradient_norm"] <= 0
        or vjp.get("replay_call_order", [None])[0] != "action"
    ):
        raise SystemExit(f"{arm_id} candidate cotangent differs")
    frozen = receipt.get("frozen_runtime", {})
    checkpoint_identity = frozen.get("checkpoint_identity", {})
    if (
        frozen.get("freeze_certificates_identical") is not True
        or frozen.get("freeze_before") != frozen.get("freeze_after")
        or frozen.get("bernini_revision") != bernini_commit
        or frozen.get("veomni_revision") != veomni_commit
        or checkpoint_identity.get("manifest_sha256_computed")
        != checkpoint_manifest_sha
        or checkpoint_identity.get("manifest_sha256_expected")
        != checkpoint_manifest_sha
        or checkpoint_identity.get("every_file_sha256_verified") is not True
        or frozen.get("freeze_after", {}).get("optimizer_created") is not False
        or frozen.get("freeze_after", {}).get("optimizer_steps") != 0
    ):
        raise SystemExit(f"{arm_id} frozen runtime certificate differs")
    cli = receipt.get("cli", {})
    if (
        cli.get("candidate_id") != contract["candidate_id"]
        or cli.get("expected_checkpoint_tree_sha256") != checkpoint_tree_sha
        or receipt.get("method_source_sha256") != method_sha
        or receipt.get("training_performed") is not False
        or receipt.get("optimizer_created") is not False
        or receipt.get("optimizer_steps") != 0
        or receipt.get("scientific_action_editing_success_claim") is not False
    ):
        raise SystemExit(f"{arm_id} no-update/source/checkpoint contract differs")
    closure = receipt.get("input_closure", {})
    for key in (
        "pure_t2v_generated_media_consumed", "pure_t2v_predecode_latent_consumed",
        "pure_t2v_gaussian_consumed", "target_video", "donor", "mask", "flow",
        "pose", "track", "trajectory",
    ):
        if closure.get(key) is not False:
            raise SystemExit(f"{arm_id} forbidden input closure differs: {key}")
    artifact = receipt.get("output", {})
    if (
        not cotangent_path.is_file()
        or cotangent_path.is_symlink()
        or Path(artifact.get("path", "")).resolve(strict=True) != cotangent_path.resolve(strict=True)
        or artifact.get("sha256") != file_sha256(cotangent_path)
        or artifact.get("tensor_key") != "candidate_action_cotangent"
        or artifact.get("shape", [])[:3] != [1, 16, 21]
        or artifact.get("stored_dtype") != "torch.float32"
        or artifact.get("roundtrip_byte_exact_fp32") is not True
        or artifact.get("proposal_media_consumed") is not False
    ):
        raise SystemExit(f"{arm_id} exact81 cotangent artifact differs")
    with safe_open(str(cotangent_path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != ["candidate_action_cotangent"]:
            raise SystemExit(f"{arm_id} cotangent key closure differs")
        tensor = opened.get_tensor("candidate_action_cotangent")
        metadata = dict(opened.metadata() or {})
    if (
        tensor.dtype != torch.float32
        or list(tensor.shape) != artifact["shape"]
        or not bool(torch.isfinite(tensor).all().item())
        or not bool(torch.count_nonzero(tensor).item())
        or metadata.get("schema_version") != "bernini-cage-native-candidate-cotangent-v1"
        or metadata.get("proposal_media_consumed") != "false"
    ):
        raise SystemExit(f"{arm_id} cotangent tensor/metadata differs")
    arm_results.append(
        {
            "arm_id": arm_id,
            "candidate_id": contract["candidate_id"],
            "action_family_id": contract["family"],
            "analysis_split": "fit",
            "receipt_file_sha256": receipt_file_sha,
            "receipt_digest": receipt_digest,
            "cotangent_sha256": artifact["sha256"],
            "gradient_norm": vjp["gradient_norm"],
            "selected_coordinate_id": vjp.get("selected_coordinate_id"),
            "selected_negative_branch": vjp.get("selected_negative_branch"),
        }
    )

aggregate = {
    "schema_version": "bernini-cage-native-candidate-vjp-dual4-postflight-v1",
    "source": {
        "archive_sha256": source_archive_sha,
        "git_revision": source_revision,
        "probe_source_sha256": method_sha,
    },
    "sealed_inputs": {
        "native_population_sha256": population_sha,
        "t2v_core4_v2_spec_sha256": t2v_sha,
        "checkpoint_content_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_tree_sha256": checkpoint_tree_sha,
        "bernini_commit": bernini_commit,
        "veomni_commit": veomni_commit,
        "t2v_bank_candidate_count": 40,
        "native_num_frames": 81,
        "native_latent_frames": 21,
        "native_inference_steps": 40,
    },
    "topology": {
        "node_count": 1,
        "allocated_mi210": 8,
        "independent_world4_sp4_arms": 2,
        "visible_gpu_partitions": [[0, 1, 2, 3], [4, 5, 6, 7]],
    },
    "probe": {
        "arms": arm_results,
        "action_family_count": 2,
        "schedule_indices": [20, 28, 33],
        "action_plus_hard_negative_branches": 10,
        "frozen_branch_sigma_scan_calls_per_arm": 30,
        "selected_input_vjp_replays_per_arm": 2,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
    },
    "capability_boundary": {
        "thirty_branch_sigma_calls_are_transformer_block_scan": False,
        "transformer_30_block_parameter_scan_performed": False,
        "sealed_trainable_band_receipt_produced": False,
        "one_step_training_authorized": False,
        "blocker": (
            "probe_cage_native_candidate_vjp.py exposes only the frozen "
            "candidate-clean input-VJP probe; a separate native all-30-block "
            "parameter-VJP scan and sealed band receipt are still required"
        ),
    },
    "scientific_action_editing_success_claim": False,
}
aggregate["receipt_digest"] = hashlib.sha256(canonical(aggregate)).hexdigest()
postflight = output_root / "cage-native-candidate-vjp-dual4-postflight.json"
raw = canonical(aggregate) + b"\n"
descriptor = os.open(postflight, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
print("CAGE_NATIVE_CANDIDATE_VJP_DUAL4_STRONG_POSTFLIGHT_OK", len(arm_results))
PY

[[ -f "${output_dir}/cage-native-candidate-vjp-dual4-postflight.json" ]] || fail "dual4 postflight receipt absent"
