#!/usr/bin/env bash
#SBATCH --job-name=pair7-phasea-fit
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() { echo "[pair-v7-phase-a] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

[[ "${PAIR_V7_ROOT_REVIEWED_PHASE_A:-NO}" == YES ]] || \
  fail "set PAIR_V7_ROOT_REVIEWED_PHASE_A=YES only after fit-evidence/runtime review"
[[ "${PAIR_V7_ACK_NO_MUTATION_NO_SUCCESS:-NO}" == YES ]] || \
  fail "set PAIR_V7_ACK_NO_MUTATION_NO_SUCCESS=YES to acknowledge measurement-only semantics"

runtime_archive="${PAIR_V7_PHASE_A_RUNTIME_ARCHIVE:?set immutable Phase-A git archive}"
runtime_archive_sha256="${PAIR_V7_PHASE_A_RUNTIME_ARCHIVE_SHA256:?set runtime archive SHA-256}"
runtime_revision="${PAIR_V7_PHASE_A_RUNTIME_REVISION:?set runtime archive revision}"
cast_archive="${PAIR_V7_CAST_V4_METHOD_ARCHIVE:?set immutable CAST-v4 d5e87ca method archive}"
cast_archive_sha256="${PAIR_V7_CAST_V4_METHOD_ARCHIVE_SHA256:?set CAST-v4 archive SHA-256}"
cast_revision="${PAIR_V7_CAST_V4_METHOD_REVISION:?set CAST-v4 method revision}"
fit_manifest="${PAIR_V7_FIT_ONLY_MANIFEST:?set sealed DP2 fit-only manifest}"
fit_manifest_sha256="${PAIR_V7_FIT_ONLY_MANIFEST_SHA256:?set fit-only manifest SHA-256}"
fit_evidence="${PAIR_V7_FIT_ONLY_EVIDENCE:?set sealed fit-only/no-update evidence}"
fit_evidence_sha256="${PAIR_V7_FIT_ONLY_EVIDENCE_SHA256:?set fit-only evidence SHA-256}"
cast_group_a="${PAIR_V7_CAST_V4_GROUP_A_RECEIPT:?set CAST-v4 group-A receipt}"
cast_group_a_sha256="${PAIR_V7_CAST_V4_GROUP_A_RECEIPT_SHA256:?set group-A receipt SHA-256}"
cast_group_b="${PAIR_V7_CAST_V4_GROUP_B_RECEIPT:?set CAST-v4 group-B receipt}"
cast_group_b_sha256="${PAIR_V7_CAST_V4_GROUP_B_RECEIPT_SHA256:?set group-B receipt SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint content manifest SHA-256}"
output="${PAIR_V7_PHASE_A_OUTPUT:?set fresh absolute measurement-receipt output}"
python_bin="${PAIR_V7_PHASE_A_PYTHON_BIN:?set vace Python executable}"

# These are scientific inputs, not caller-selectable alternatives.
readonly expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
readonly expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
readonly expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
readonly expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
readonly expected_cast_revision="d5e87caf8f58a63c2ff902386d6e19100f7fcf34"
readonly expected_cast_archive_sha256="a15a95ea9bcee5cbe331ef975ce2cef38e6c164df644d8d053a87b5fe71c7470"
readonly expected_cast_group_a_sha256="bc20af168fdb29f72854d6c4c0f978eb29ba6642c1a5cbbeab69f33bf068ea1a"
readonly expected_cast_group_b_sha256="4a95e306c560b37b52f204c8d818774d43b648ff6fc79e5c704c9e69cdc2ce75"

for name in runtime_archive cast_archive fit_manifest fit_evidence cast_group_a cast_group_b bernini_root veomni_root checkpoint checkpoint_manifest output python_bin; do
  absolute "${!name}" "${name}"
done
for name in runtime_archive_sha256 cast_archive_sha256 fit_manifest_sha256 fit_evidence_sha256 cast_group_a_sha256 cast_group_b_sha256 checkpoint_manifest_sha256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} must be lowercase SHA-256"
done
for name in runtime_revision cast_revision; do
  [[ "${!name}" =~ ^[0-9a-f]{40}$ ]] || fail "${name} must be lowercase Git commit"
done
[[ "${checkpoint_manifest_sha256}" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest pin differs"
[[ "${cast_revision}" == "${expected_cast_revision}" ]] || fail "CAST-v4 revision differs"
[[ "${cast_archive_sha256}" == "${expected_cast_archive_sha256}" ]] || fail "CAST-v4 archive pin differs"
[[ "${cast_group_a_sha256}" == "${expected_cast_group_a_sha256}" ]] || fail "CAST-v4 group-A pin differs"
[[ "${cast_group_b_sha256}" == "${expected_cast_group_b_sha256}" ]] || fail "CAST-v4 group-B pin differs"

runtime_archive="$(realpath -e -- "${runtime_archive}")"
cast_archive="$(realpath -e -- "${cast_archive}")"
fit_manifest="$(realpath -e -- "${fit_manifest}")"
fit_evidence="$(realpath -e -- "${fit_evidence}")"
cast_group_a="$(realpath -e -- "${cast_group_a}")"
cast_group_b="$(realpath -e -- "${cast_group_b}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for name in runtime_archive cast_archive fit_manifest fit_evidence cast_group_a cast_group_b checkpoint_manifest python_bin; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must resolve to a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must resolve to a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"

sealed_inputs=(
  "${runtime_archive}" "${runtime_archive_sha256}"
  "${cast_archive}" "${cast_archive_sha256}"
  "${fit_manifest}" "${fit_manifest_sha256}"
  "${fit_evidence}" "${fit_evidence_sha256}"
  "${cast_group_a}" "${cast_group_a_sha256}"
  "${cast_group_b}" "${cast_group_b_sha256}"
  "${checkpoint_manifest}" "${checkpoint_manifest_sha256}"
)
for ((index=0; index<${#sealed_inputs[@]}; index+=2)); do
  path="${sealed_inputs[index]}"
  digest="${sealed_inputs[index+1]}"
  [[ "$(sha256_file "${path}")" == "${digest}" ]] || fail "sealed input bytes differ: ${path}"
done
[[ "$(git get-tar-commit-id <"${runtime_archive}")" == "${runtime_revision}" ]] || fail "runtime archive revision differs"
[[ "$(git get-tar-commit-id <"${cast_archive}")" == "${cast_revision}" ]] || fail "CAST-v4 archive revision differs"

[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output must be create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent is a symlink"

(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed before Phase-A"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v7-phase-a-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v7-phase-a-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
  if [[ -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || status=2
    rm -rf -- "${task_scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${task_scratch}/source" "${task_scratch}/cache" "${task_scratch}/tmp" "${task_scratch}/pycache"
archive_copy="${task_scratch}/runtime-source.tar"
cp -- "${runtime_archive}" "${archive_copy}"
chmod a-w -- "${archive_copy}"
[[ "$(sha256_file "${archive_copy}")" == "${runtime_archive_sha256}" ]] || fail "staged runtime archive differs"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

required = {
    "methods/bernini_action_editing/audit_pair_v7_phase_a_geometry.py",
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
    "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
    "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
    "methods/bernini_action_editing/pair_v5_t2v_guidance_distill.py",
    "methods/bernini_action_editing/pair_v7_dual_coordinate_nullspace_transport.py",
    "methods/bernini_action_editing/pair_v7_fit_only_geometry_authority.py",
    "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
    "methods/bernini_action_editing/source_kv_replay.py",
    "methods/bernini_action_editing/source_kv_route_batches.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
    "methods/bernini_action_editing/source_self_native_target_adapter.py",
    "methods/bernini_action_editing/source_self_runtime.py",
    "methods/bernini_action_editing/source_value_residual.py",
    "methods/bernini_action_editing/train_lora.py",
    "methods/bernini_action_editing/tests/test_pair_v7_dual_coordinate_nullspace_transport.py",
    "methods/bernini_action_editing/tests/test_pair_v7_fit_only_geometry_authority.py",
    "methods/bernini_action_editing/tests/test_audit_pair_v7_phase_a_geometry.py",
    "methods/bernini_action_editing/tests/test_auh_audit_pair_v7_phase_a_launcher.py",
    "methods/bernini_action_editing/tests/test_pair_v5_action_adapter.py",
    "methods/bernini_action_editing/scripts/auh_audit_pair_v7_phase_a_dp2sp4.sbatch",
}
seen: set[str] = set()
with tarfile.open(sys.argv[1], "r:*") as handle:
    for member in handle.getmembers():
        pure = PurePosixPath(member.name)
        name = pure.as_posix()
        if name.startswith("./"):
            name = name[2:]
        if (
            not name
            or pure.is_absolute()
            or ".." in pure.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
            or not (member.isfile() or member.isdir())
            or name in seen
        ):
            raise SystemExit("unsafe or duplicate runtime archive member")
        seen.add(name.rstrip("/"))
missing = sorted(required - seen)
if missing:
    raise SystemExit(f"runtime archive lacks fit-only Phase-A closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -d "${method_root}" && ! -L "${method_root}" ]] || fail "extracted method root differs"
find "${method_root}" -type f -exec chmod a-w -- {} +
method_tree_before="$(find "${method_root}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"

export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/cache/triton"
export XDG_CACHE_HOME="${task_scratch}/cache/xdg"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export TMPDIR="${task_scratch}/tmp"
export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"

for test_name in \
  test_pair_v7_dual_coordinate_nullspace_transport.py \
  test_pair_v7_fit_only_geometry_authority.py \
  test_audit_pair_v7_phase_a_geometry.py \
  test_auh_audit_pair_v7_phase_a_launcher.py \
  test_pair_v5_action_adapter.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
    -s "${method_root}/tests" -p "${test_name}"
done

# The audit CLI retains compatibility flag names; both values below are the new
# sealed fit-only manifest/evidence and carry no optimizer or population GO.
common_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"
  --action-event-manifest "${fit_manifest}"
  --expected-action-event-manifest-sha256 "${fit_manifest_sha256}"
  --cagd-validator-evidence "${fit_evidence}"
  --expected-cagd-validator-evidence-sha256 "${fit_evidence_sha256}"
  --scorer-group-receipt "${cast_group_a}"
  --scorer-group-receipt "${cast_group_b}"
  --expected-scorer-group-receipt-sha256 "${cast_group_a_sha256}"
  --expected-scorer-group-receipt-sha256 "${cast_group_b_sha256}"
  --runtime-source-archive "${archive_copy}"
  --runtime-source-archive-sha256 "${runtime_archive_sha256}"
  --runtime-source-revision "${runtime_revision}"
  --evidence-method-source-archive "${cast_archive}"
  --evidence-method-source-archive-sha256 "${cast_archive_sha256}"
  --evidence-method-source-revision "${cast_revision}"
  --output "${output}"
  --schedule-index 33
  --source-noise-seed 20260808
  --source-carrier-mode none
  --expected-bernini-commit "${expected_bernini_commit}"
  --expected-veomni-commit "${expected_veomni_commit}"
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
  --ack-root-reviewed-phase-a-launch
  --ack-no-parameter-mutation-no-success-claim
)

echo "[pair-v7-phase-a] preflight: fit-only geometry measurement, schedule=33, no update/success claim"
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/audit_pair_v7_phase_a_geometry.py" \
  "${common_args[@]}" --preflight-only

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
echo "[pair-v7-phase-a] allocation=8xMI210 topology=WORLD8/DP2xSP4 exact81=true fps=25 schedule=33"
PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "${method_root}/audit_pair_v7_phase_a_geometry.py" \
  "${common_args[@]}"

for ((index=0; index<${#sealed_inputs[@]}; index+=2)); do
  path="${sealed_inputs[index]}"
  digest="${sealed_inputs[index+1]}"
  [[ "$(sha256_file "${path}")" == "${digest}" ]] || fail "sealed input changed during Phase-A: ${path}"
done
[[ "$(sha256_file "${archive_copy}")" == "${runtime_archive_sha256}" ]] || fail "staged runtime archive changed"
(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed after Phase-A"
method_tree_after="$(find "${method_root}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "read-only extracted runtime tree changed"

"${python_bin}" -B - \
  "${output}" \
  "${fit_manifest_sha256}" \
  "${runtime_archive_sha256}" \
  "${runtime_revision}" \
  "${cast_archive_sha256}" \
  "${cast_revision}" \
  "${checkpoint_manifest_sha256}" \
  "${expected_checkpoint_tree_sha256}" \
  "${bernini_root}" \
  "${veomni_root}" \
  "${expected_bernini_commit}" \
  "${expected_veomni_commit}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(
    output_value,
    fit_manifest_sha,
    runtime_archive_sha,
    runtime_revision,
    cast_archive_sha,
    cast_revision,
    checkpoint_manifest_sha,
    checkpoint_tree_sha,
    bernini_root,
    veomni_root,
    bernini_revision,
    veomni_revision,
) = sys.argv[1:]

SHA256 = re.compile(r"[0-9a-f]{64}")
NO_UPDATE = {
    "global_population_go": False,
    "optimizer_authorized": False,
    "parameter_update_authorized": False,
    "action_success_claimed": False,
}


def require(value, message):
    if not value:
        raise SystemExit(message)


def canonical_digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def check_seal(value, label):
    require(isinstance(value, dict), f"{label} must be an object")
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    require(
        isinstance(declared, str)
        and SHA256.fullmatch(declared) is not None
        and canonical_digest(unsigned) == declared,
        f"{label} digest differs",
    )
    return declared


def check_no_update(value, label):
    for field, expected in NO_UPDATE.items():
        require(value.get(field) is expected, f"{label}.{field} differs")


raw_root = Path(output_value)
require(not raw_root.is_symlink(), "Phase-A output must not be a symlink")
root = raw_root.resolve(strict=True)
require(root.is_dir(), "Phase-A output must be a plain directory")
entries = list(root.iterdir())
require(
    len(entries) == 1
    and entries[0].name == "receipt.json"
    and entries[0].is_file()
    and not entries[0].is_symlink(),
    "Phase-A artifact closure must be receipt.json only",
)
receipt = json.loads(entries[0].read_text(encoding="ascii"))
check_seal(receipt, "Phase-A receipt")
check_no_update(receipt, "Phase-A receipt")

for field in (
    "optimizer_constructed",
    "optimizer_step_called",
    "candidate_delta_constructed",
    "parameter_add_called",
    "parameter_mutation_performed",
    "parameter_update_authorized",
    "scientific_action_editing_success_claim",
):
    require(receipt.get(field) is False, f"Phase-A receipt.{field} must remain false")
require(receipt.get("audit_complete") is True, "Phase-A audit is incomplete")
require(
    receipt.get("schema_version")
    == "bernini-pair-v7-phase-a-geometry-audit-v3",
    "Phase-A receipt schema differs",
)
require(receipt.get("geometry_audit_performed") is True, "geometry measurement was not performed")
require(receipt.get("topology") == "WORLD8-DP2xUlysses-SP4", "WORLD8 DP2xSP4 topology differs")
require(receipt.get("single_exact40_cell") is True, "single-cell closure differs")
require(receipt.get("frame_count") == 81, "run is not exact81")
require(
    receipt.get("parameter_state_sha256")
    == receipt.get("post_audit_parameter_state_sha256"),
    "fixed Action-LoRA gauge changed",
)

schedule = receipt.get("schedule_policy")
require(
    isinstance(schedule, dict)
    and schedule.get("schedule_index") == 33
    and schedule.get("first_phase_a_schedule_index") == 33
    and schedule.get("is_preregistered_first_phase_a_cell") is True
    and schedule.get("single_fit_only_geometry_cell") is True
    and schedule.get("model_callbacks_authorized") is True
    and schedule.get("gradient_audit_authorized") is True
    and schedule.get("parameter_update_authorized") is False,
    "fixed schedule-33 measurement policy differs",
)

checkpoint = receipt.get("checkpoint")
require(
    isinstance(checkpoint, dict)
    and checkpoint.get("tree_sha256") == checkpoint_tree_sha
    and checkpoint.get("content_manifest_sha256") == checkpoint_manifest_sha
    and checkpoint.get("post_audit_unchanged") is True,
    "checkpoint postflight closure differs",
)

runtime_source = receipt.get("runtime_source")
cast_source = receipt.get("evidence_method_source")
require(
    isinstance(runtime_source, dict)
    and runtime_source.get("revision") == runtime_revision
    and runtime_source.get("archive_sha256") == runtime_archive_sha
    and runtime_source.get("post_audit_unchanged") is True,
    "runtime archive receipt differs",
)
require(
    isinstance(cast_source, dict)
    and cast_source.get("revision") == cast_revision
    and cast_source.get("archive_sha256") == cast_archive_sha
    and cast_source.get("post_audit_unchanged") is True,
    "CAST-v4 archive receipt differs",
)

action_manifest = receipt.get("action_manifest")
require(
    isinstance(action_manifest, dict)
    and action_manifest.get("file_sha256") == fit_manifest_sha
    and len(action_manifest.get("candidate_ids", [])) == 2
    and len(set(action_manifest.get("candidate_ids", []))) == 2
    and len(action_manifest.get("action_families", [])) == 2
    and len(set(action_manifest.get("action_families", []))) == 2
    and action_manifest.get("global_population_authority_consumed") is False
    and action_manifest.get("cast_v4_receipts_consumed_as_fit_provenance_only") is True
    and action_manifest.get("population_scorer_optimizer_authority_consumed") is False,
    "fit-only DP2 manifest receipt differs",
)

authority = action_manifest
validation = authority.get("fit_only_geometry_authority_validation_receipt")
check_seal(validation, "fit-only authority validation receipt")
check_no_update(validation, "fit-only authority validation receipt")
require(
    validation.get("authority_scope") == "fit_only_read_only_gradient_geometry"
    and validation.get("geometry_measurement_authorized") is True
    and validation.get("manifest_file_sha256") == fit_manifest_sha
    and validation.get("checkpoint_tree_sha256") == checkpoint_tree_sha
    and validation.get("fit_event_count") == 2
    and validation.get("fit_event_ids") == action_manifest.get("candidate_ids")
    and validation.get("fit_action_families") == action_manifest.get("action_families")
    and validation.get("schedule_indices") == [33]
    and validation.get("first_schedule_index") == 33
    and validation.get("confirmation_population_consumed") is False
    and validation.get("population_scorer_receipts_consumed") is True
    and validation.get("population_scorer_receipts_role")
    == "authenticate_selected_fit_action_scores_only_never_population_go"
    and validation.get("legacy_optimizer_authority_consumed") is False
    and validation.get("cast_v4_candidate_receipt_count") == 40
    and validation.get("external_evidence_file_count") == 49
    and validation.get("all_external_evidence_files_bound_for_postflight") is True
    and authority.get("external_evidence_files_rehashed_post_audit") is True,
    "fit-only authority/external-evidence closure differs",
)
require(
    action_manifest.get("fit_only_geometry_authority_digest")
    == validation["receipt_digest"],
    "fit-only authority digest binding differs",
)
method_binding = validation.get("cast_v4_method_archive")
require(
    isinstance(method_binding, dict)
    and method_binding.get("file_sha256") == cast_archive_sha
    and method_binding.get("git_archive_revision") == cast_revision
    and method_binding.get("required_member_closure_present") is True
    and isinstance(validation.get("cast_v4_group_receipt_digests"), list)
    and len(validation["cast_v4_group_receipt_digests"]) == 2
    and set(validation.get("selected_action_score_by_event", {}))
    == set(action_manifest["candidate_ids"]),
    "CAST-v4 fit provenance closure differs",
)
boundaries = validation.get("negative_boundaries")
require(isinstance(boundaries, list) and len(boundaries) == 2, "negative-boundary closure differs")
boundary_by_id = {row.get("boundary_id"): row for row in boundaries}
legacy_boundary = boundary_by_id.get("d541801_v3_confirmation_no_optimizer_go")
sail_boundary = boundary_by_id.get("sail_prior_frozen_intervention_no_success")
require(
    isinstance(legacy_boundary, dict)
    and legacy_boundary.get("required_boolean_observations")
    == {"optimizer_authorized": False, "gates.confirmation_overall": False}
    and legacy_boundary.get("inherited_as_population_or_update_authority") is False,
    "legacy d541801 NO-GO boundary differs",
)
require(
    isinstance(sail_boundary, dict)
    and sail_boundary.get("required_boolean_observations")
    == {
        "postflight_complete": True,
        "all_six_mp4_exact81": True,
        "scientific_claim_authorized": False,
        "action_editing_success_claim_authorized": False,
        "training_performed": False,
        "source_condition_in_live_query": False,
    }
    and sail_boundary.get("inherited_as_population_or_update_authority") is False,
    "SAIL no-success boundary differs",
)
sail_children = sail_boundary.get("child_receipts")
require(
    isinstance(sail_children, list)
    and len(sail_children) == 2
    and {row.get("role") for row in sail_children} == {"dog", "human"},
    "SAIL dog/human child closure differs",
)
for child in sail_children:
    require(
        child.get("mechanism_probe_only") is True
        and child.get("editor_parameter_or_update_authorized") is False
        and child.get("training_performed") is False
        and child.get("scientific_claim_authorized") is False
        and child.get("action_editing_success_claim_authorized") is False
        and child.get("source_condition_in_live_query") is False
        and child.get("measured_vjp_observations")
        == {
            "real_sp4_autograd_collective_observed": True,
            "replica_consensus_observed": True,
            "same_x_sigma_object_for_action_noop": True,
        }
        and SHA256.fullmatch(str(child.get("file_sha256"))) is not None
        and SHA256.fullmatch(str(child.get("receipt_digest"))) is not None,
        "SAIL child no-success/measured-VJP boundary differs",
    )

source = receipt.get("correct_source_coordinates")
decoded = source.get("decoded_source_receipts_by_arm") if isinstance(source, dict) else None
require(
    isinstance(source, dict)
    and source.get("manifest_file_sha256") == fit_manifest_sha
    and source.get("correct_source_media_bound_and_loaded") is True
    and source.get("source_frame_count") == 81
    and source.get("source_fps") == 25.0
    and source.get("deployment_visual_condition") == "source_video_only_V"
    and source.get("image_reference_count") == 0
    and source.get("reference_indices") == []
    and source.get("sp4_source_receipt_consensus_per_arm") is True
    and source.get("wrong_source_fields_present") is False
    and source.get("legacy_source_binding_manifest_consumed") is False
    and isinstance(decoded, list)
    and len(decoded) == 2,
    "correct-source exact81/25fps closure differs",
)
require([row.get("arm_index") for row in decoded] == [0, 1], "decoded-source arm order differs")
require(len(set(source.get("sample_ids", []))) == 2, "correct-source sample IDs differ")

decoded_by_arm = {}
for row in decoded:
    arm = row.get("arm_index")
    source_receipt = row.get("source_receipt")
    require(row.get("sp4_receipt_consensus") is True, "SP4 decoded-source consensus differs")
    check_seal(source_receipt, f"decoded source arm {arm}")
    check_no_update(source_receipt, f"decoded source arm {arm}")
    require(
        source_receipt.get("frame_count") == 81
        and row.get("source_sample_id") == source["sample_ids"][arm]
        and source_receipt.get("fps") == 25.0
        and source_receipt.get("deployment_visual_condition") == "source_video_only_V"
        and source_receipt.get("image_reference_count") == 0
        and source_receipt.get("reference_indices") == []
        and SHA256.fullmatch(str(source_receipt.get("video_sha256"))) is not None
        and SHA256.fullmatch(str(source_receipt.get("clean_latent_sha256"))) is not None
        and isinstance(source_receipt.get("reference_latent_sha256"), list)
        and source_receipt["reference_latent_sha256"] == [],
        f"decoded source arm {arm} payload differs",
    )
    video_path = Path(str(source_receipt.get("video_path")))
    require(video_path.is_absolute() and video_path.is_file() and not video_path.is_symlink(), f"source arm {arm} file differs")
    digest = hashlib.sha256()
    with video_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    require(digest.hexdigest() == source_receipt["video_sha256"], f"source arm {arm} bytes changed")
    decoded_by_arm[arm] = source_receipt

ranks = receipt.get("rank_runtime_provenance")
require(isinstance(ranks, list) and len(ranks) == 8, "WORLD8 runtime provenance differs")
require({row.get("rank") for row in ranks} == set(range(8)), "WORLD8 rank closure differs")
prompt_receipt_by_arm = {}
for arm, expected_ranks in ((0, {0, 1, 2, 3}), (1, {4, 5, 6, 7})):
    rows = [row for row in ranks if row.get("arm_index") == arm]
    require(
        len(rows) == 4
        and {row.get("rank") for row in rows} == expected_ranks
        and {row.get("sp_rank") for row in rows} == set(range(4))
        and all(row.get("source_receipt") == decoded_by_arm[arm] for row in rows)
        and all(row.get("source_receipt_digest") == decoded_by_arm[arm]["receipt_digest"] for row in rows),
        f"source arm {arm} SP4 rank consensus differs",
    )
    prompt_receipt = rows[0].get("prompt_receipt")
    prompt_digest = check_seal(prompt_receipt, f"prompt receipt arm {arm}")
    check_no_update(prompt_receipt, f"prompt receipt arm {arm}")
    require(
        all(row.get("prompt_receipt") == prompt_receipt for row in rows)
        and all(row.get("prompt_receipt_digest") == prompt_digest for row in rows)
        and prompt_receipt.get("deployment_v2v_prompt_authority")
        == "infer_lora.build_training_prompt"
        and prompt_receipt.get("task_prefix_applied_exactly_once") is True
        and all(
            SHA256.fullmatch(str(prompt_receipt.get(field))) is not None
            for field in (
                "raw_caption_bank_sha256",
                "t2v_task_prompt_bank_sha256",
                "deployment_v2v_task_prompt_bank_sha256",
                "deployment_v2v_system_prompt_sha256",
            )
        ),
        f"prompt receipt arm {arm} authority/SP4 closure differs",
    )
    prompt_receipt_by_arm[arm] = prompt_receipt


def check_sp4_bundle(
    bundle,
    label,
    expected_gradient,
    *,
    expected_rank_schema,
    identity_protocol_digest=None,
    expected_sketch_index=None,
    expected_source_coordinate_digest=None,
    expected_deployment_v_pack_digest=None,
    expected_negative_prompt_embedding_sha256=None,
    expected_positive_prompt_embedding_sha256_by_branch=None,
):
    check_seal(bundle, label)
    check_no_update(bundle, label)
    rows = bundle.get("sp_rank_receipts")
    require(
        bundle.get("schema_version")
        == "bernini-pair-v7-phase-a-sp4-vjp-bundle-v1"
        and bundle.get("sp_size") == 4
        and bundle.get("all_four_rank_local_vjps_bound") is True
        and bundle.get("sp4_arithmetic_average_bound") is True
        and bundle.get("parameter_mutation_performed") is False
        and bundle.get("averaged_gradient_sha256") == expected_gradient
        and isinstance(rows, list)
        and len(rows) == 4
        and [row.get("sp_rank") for row in rows] == [0, 1, 2, 3],
        f"{label} SP4 closure differs",
    )
    digests = []
    for index, row in enumerate(rows):
        digests.append(check_seal(row, f"{label} rank {index}"))
        check_no_update(row, f"{label} rank {index}")
        require(
            row.get("schema_version") == expected_rank_schema
            and row.get("gradient_sha256") == expected_gradient,
            f"{label} rank schema/gradient differs",
        )
        if identity_protocol_digest is not None:
            require(
                row.get("identity_deployment_protocol_digest")
                == identity_protocol_digest
                and row.get("sketch_index") == expected_sketch_index,
                f"{label} deployment-protocol/sketch binding differs",
            )
            require(
                row.get("source_coordinate_receipt_digest")
                == expected_source_coordinate_digest,
                f"{label} source-coordinate binding differs",
            )
            pack = row.get("deployment_v_pack_receipt")
            pack_digest = check_seal(pack, f"{label} rank {index} deployment V pack")
            check_no_update(pack, f"{label} rank {index} deployment V pack")
            require(
                row.get("deployment_v_pack_receipt_digest") == pack_digest
                and pack_digest == expected_deployment_v_pack_digest
                and pack.get("schema_version")
                == "bernini-pair-v7-deployment-v-pack-v1"
                and pack.get("deployment_entrypoint") == "infer_lora.py:model.sample"
                and pack.get("visual_condition") == "source_video_only_V"
                and pack.get("image_reference_count") == 0
                and pack.get("patch_call_roles") == ["source_video", "noisy_target"]
                and pack.get("patch_call_source_ids") == [1.0, 0.0]
                and pack.get("concat_order") == ["video", "target"]
                and type(pack.get("condition_tokens")) is int
                and pack.get("condition_tokens") > 0
                and pack.get("target_tokens") == pack.get("condition_tokens")
                and pack.get("total_tokens") == 2 * pack.get("condition_tokens"),
                f"{label} deployment V-pack binding differs",
            )
            require(
                row.get("negative_prompt_embedding_sha256")
                == expected_negative_prompt_embedding_sha256
                and row.get("positive_prompt_embedding_sha256_by_branch")
                == expected_positive_prompt_embedding_sha256_by_branch,
                f"{label} prompt-embedding binding differs",
            )
    require(bundle.get("sp_rank_receipt_digests") == digests, f"{label} rank digest list differs")


actions = receipt.get("dp_action_gradient_metadata")
require(isinstance(actions, list) and len(actions) == 2, "DP2 action-gradient metadata differs")
for index, row in enumerate(actions):
    check_sp4_bundle(
        row.get("gradient_computation_sp4_bundle"),
        f"action arm {index}",
        row.get("gradient_sha256"),
        expected_rank_schema="bernini-pair-v7-phase-a-action-gradient-v1",
    )
identity_protocol = receipt.get("identity_deployment_protocol")
identity_protocol_digest = check_seal(
    identity_protocol, "identity deployment protocol"
)
check_no_update(identity_protocol, "identity deployment protocol")
require(
    identity_protocol.get("schema_version")
    == "bernini-pair-v7-identity-deployment-protocol-v1"
    and identity_protocol.get("authority")
    == "VideoEdit_infer_lora_frozen_deployment_contract"
    and identity_protocol.get("positive_prompt_authority")
    == "infer_lora.build_training_prompt"
    and identity_protocol.get("negative_prompt_authority")
    == "infer_lora.DEFAULT_NEGATIVE_PROMPT"
    and identity_protocol.get("guidance_mode") == "v2v_apg"
    and identity_protocol.get("visual_condition") == "source_video_only_V"
    and identity_protocol.get("image_reference_count") == 0
    and identity_protocol.get("forward_order_per_field")
    == ["V_negative", "V_positive"]
    and identity_protocol.get("omega_txt") == 4.0
    and identity_protocol.get("eta") == 0.5
    and identity_protocol.get("norm_threshold") == 50.0
    and identity_protocol.get("momentum") == 0.0
    and identity_protocol.get("num_inference_steps") == 40
    and identity_protocol.get("schedule_index") == 33
    and identity_protocol.get("timestep") == 516
    and identity_protocol.get("sigma_source")
    == "scheduler.sigmas[33]_cpu_fp32"
    and identity_protocol.get("sigma_float32_be_hex") == "3f042120"
    and identity_protocol.get("fresh_zero_momentum_history_equivalent") is True
    and identity_protocol.get("old_diff_vjp_coefficient") == 0.0
    and identity_protocol.get("sealed_source_noised_coordinate") is True
    and identity_protocol.get("single_cell_local_field_geometry") is True
    and identity_protocol.get("full_sampler_trajectory_equivalent") is False
    and identity_protocol.get("vendor_apg_helper_used") is True
    and identity_protocol.get("alignment_scope")
    == "visual_pack_sampler_parameters_and_post_APG_operator"
    and identity_protocol.get(
        "action_lora_scope_is_method_specific_not_infer_lora_peft_scope"
    )
    is True
    and identity_protocol.get("forwarded_visual_branches") == ["V"],
    "frozen infer_lora deployment-aligned V-only APG protocol differs",
)
identities = receipt.get("identity_probe_metadata")
require(
    isinstance(identities, list) and len(identities) == 16,
    "DP2x2-familyxK4 identity VJP metadata differs",
)
required_identity_families = {
    "deploy_noop_identity",
    "deploy_camera_delta",
}
require(
    {row.get("arm_index") for row in identities} == {0, 1}
    and {row.get("family") for row in identities} == required_identity_families,
    "deployment identity arm/family closure differs",
)
require(
    all(
        row.get("identity_deployment_protocol_digest")
        == identity_protocol_digest
        for row in ranks
    ),
    "WORLD8 deployment-protocol consensus differs",
)
for arm in (0, 1):
    for family in required_identity_families:
        group = [
            row
            for row in identities
            if row.get("arm_index") == arm and row.get("family") == family
        ]
        require(
            len(group) == 4
            and {row.get("sketch_index") for row in group} == set(range(4))
            and len({row.get("feature_sketch_sha256") for row in group}) == 4
            and all(
                SHA256.fullmatch(str(row.get("feature_sketch_sha256")))
                is not None
                for row in group
            )
            and all(
                row.get("identity_deployment_protocol_digest")
                == identity_protocol_digest
                for row in group
            )
            and len(
                {row.get("source_coordinate_receipt_digest") for row in group}
            )
            == 1
            and len({row.get("deployment_v_pack_receipt_digest") for row in group})
            == 1
            and len({row.get("negative_prompt_embedding_sha256") for row in group})
            == 1
            and all(
                SHA256.fullmatch(
                    str(row.get("deployment_v_pack_receipt_digest"))
                )
                is not None
                and SHA256.fullmatch(
                    str(row.get("negative_prompt_embedding_sha256"))
                )
                is not None
                for row in group
            ),
            f"identity arm {arm} family {family} K4/protocol closure differs",
        )
        for row in group:
            positive_prompt_embeddings = row.get(
                "positive_prompt_embedding_sha256_by_branch"
            )
            expected_prompt_branches = (
                {"noop"}
                if family == "deploy_noop_identity"
                else {"noop", "camera_only"}
            )
            require(
                isinstance(positive_prompt_embeddings, dict)
                and set(positive_prompt_embeddings) == expected_prompt_branches
                and all(
                    SHA256.fullmatch(str(value)) is not None
                    for value in positive_prompt_embeddings.values()
                ),
                f"identity arm {arm} family {family} positive prompt closure differs",
            )
            check_sp4_bundle(
                row.get("identity_vjp_sp4_bundle"),
                f"identity arm {arm} family {family} sketch {row['sketch_index']}",
                row.get("gradient_sha256"),
                expected_rank_schema="bernini-pair-v7-phase-a-identity-vjp-v2",
                identity_protocol_digest=identity_protocol_digest,
                expected_sketch_index=row["sketch_index"],
                expected_source_coordinate_digest=row[
                    "source_coordinate_receipt_digest"
                ],
                expected_deployment_v_pack_digest=row[
                    "deployment_v_pack_receipt_digest"
                ],
                expected_negative_prompt_embedding_sha256=row[
                    "negative_prompt_embedding_sha256"
                ],
                expected_positive_prompt_embedding_sha256_by_branch=(
                    positive_prompt_embeddings
                ),
            )

for arm in (0, 1):
    arm_rows = [row for row in identities if row.get("arm_index") == arm]
    noop_embeddings = {
        row["positive_prompt_embedding_sha256_by_branch"]["noop"]
        for row in arm_rows
    }
    require(
        len(noop_embeddings) == 1
        and len({row.get("negative_prompt_embedding_sha256") for row in arm_rows})
        == 1
        and len({row.get("deployment_v_pack_receipt_digest") for row in arm_rows})
        == 1,
        f"identity arm {arm} shared deployment prompt/pack closure differs",
    )

identity_coordinate_by_arm = {
    arm: {
        row.get("source_coordinate_receipt_digest")
        for row in identities
        if row.get("arm_index") == arm
    }
    for arm in (0, 1)
}
require(
    all(len(values) == 1 for values in identity_coordinate_by_arm.values())
    and len(set.union(*identity_coordinate_by_arm.values())) == 2,
    "DP2 identity source-coordinate assignment differs",
)

solver = receipt.get("world_union_solver_authority")
solver_digest = check_seal(solver, "WORLD8 union solver authority receipt")
check_no_update(solver, "WORLD8 union solver authority receipt")
solver_input = solver.get("input_receipt")
solver_input_digest = check_seal(
    solver_input, "WORLD8 union solver input receipt"
)
check_no_update(solver_input, "WORLD8 union solver input receipt")
require(
    solver.get("schema_version")
    == "bernini-pair-v7-phase-a-world-union-authority-v1"
    and solver.get("world_size") == 8
    and solver.get("authoritative_world_rank") == 0
    and solver.get("solver_execution_count") == 1
    and solver.get("solver_device") == "cpu"
    and solver.get("solver_dtype")
    == "FP64_geometry_from_exact_FP32_gradients"
    and solver.get("replicated_gpu_solver_used") is False
    and solver.get("world_input_digest_consensus") is True
    and solver.get("result_broadcast_to_world") is True
    and solver.get("world_result_digest_consensus") is True
    and solver.get("input_receipt_digest") == solver_input_digest
    and SHA256.fullmatch(str(solver.get("world_result_digest"))) is not None
    and solver.get("geometry_audit_passed")
    is receipt.get("geometry_audit_passed"),
    "WORLD8 rank-0 CPU union authority differs",
)
require(
    solver_input.get("schema_version")
    == "bernini-pair-v7-phase-a-world-union-input-v1"
    and solver_input.get("topology") == "WORLD8-DP2xUlysses-SP4"
    and isinstance(solver_input.get("action_rows"), list)
    and len(solver_input["action_rows"]) == 2
    and isinstance(solver_input.get("identity_rows"), list)
    and len(solver_input["identity_rows"]) == 16
    and solver_input.get("unprojected_gradient_bytes_bound") is True
    and solver_input.get("world_input_digest_consensus_required_before_solver")
    is True,
    "WORLD8 union input byte binding differs",
)

union = receipt.get("union_projection_receipt")
union_digest = check_seal(union, "union projection receipt")
check_no_update(union, "union projection receipt")
require(
    isinstance(union, dict)
    and union.get("schema_version")
    == "bernini-pair-v7-phase-a-dp2-union-projection-v2"
    and union.get("local_project_then_average") is False
    and union.get("projection_count_after_union") == 1,
    "union projection order differs",
)
require(
    union.get("identity_probe_union_count") == 16
    and union.get("identity_required_family_count") == 2
    and union.get("identity_sketches_per_source_family") == 4
    and union.get("identity_source_coordinate_count") == 2,
    "union identity K4/source/family closure differs",
)
rank_gates = union.get("identity_source_family_rank_gate")
source_coordinate_digests = {
    row.get("source_coordinate_receipt_digest") for row in identities
}
require(
    len(source_coordinate_digests) == 2
    and all(SHA256.fullmatch(str(value)) is not None for value in source_coordinate_digests)
    and isinstance(rank_gates, list)
    and len(rank_gates) == 4
    and {
        (row.get("source_coordinate_receipt_digest"), row.get("family"))
        for row in rank_gates
    }
    == {
        (coordinate, family)
        for coordinate in source_coordinate_digests
        for family in required_identity_families
    },
    "union source-family rank-gate closure differs",
)
for row in rank_gates:
    require(
        row.get("probe_count") == 4
        and row.get("minimum_required_rank") == 3
        and type(row.get("effective_rank")) is int
        and 0 <= row["effective_rank"] <= 4
        and isinstance(row.get("gram_eigenvalues"), list)
        and len(row["gram_eigenvalues"]) <= 4,
        "union source-family effective-rank evidence differs",
    )
require(
    union.get("identity_minimum_global_effective_rank") == 8
    and type(union.get("identity_global_effective_rank")) is int
    and 0 <= union["identity_global_effective_rank"] <= 16
    and union.get("geometry_audit_passed")
    is receipt.get("geometry_audit_passed"),
    "union global effective-rank/result binding differs",
)
if receipt.get("geometry_audit_passed") is True:
    require(
        all(row["effective_rank"] >= 3 for row in rank_gates)
        and union["identity_global_effective_rank"] >= 8,
        "passing geometry lacks the preregistered identity-rank strength",
    )
transport = receipt.get("nullspace_transport_receipt")
transport_digest = check_seal(transport, "nullspace transport receipt")
check_no_update(transport, "nullspace transport receipt")
require(
    union.get("transport_receipt_digest") == transport_digest
    and solver.get("union_projection_receipt_digest") == union_digest
    and solver.get("nullspace_transport_receipt_digest") == transport_digest
    and solver.get("world_result_digest")
    == canonical_digest(
        {
            "input_receipt_digest": solver_input_digest,
            "geometry_audit_passed": receipt.get("geometry_audit_passed"),
            "union_projection_receipt_digest": union_digest,
            "nullspace_transport_receipt_digest": transport_digest,
        }
    )
    and union_digest == union.get("receipt_digest")
    and union.get("transport_geometry_authorized")
    is transport.get("geometry_authorized")
    and transport.get("identity_probe_count") == 16
    and transport.get("identity_family_counts")
    == {"deploy_noop_identity": 8, "deploy_camera_delta": 8}
    and transport.get("identity_effective_rank")
    == union.get("identity_global_effective_rank")
    and transport.get("schema_version")
    == "bernini-pair-v7-nullspace-transport-receipt-v2",
    "union/nullspace sealed receipt binding differs",
)

trees = receipt.get("model_source_trees")
require(
    isinstance(trees, dict)
    and trees.get("tracked_file_closure_rehashed_after_audit") is True
    and trees.get("post_audit_unchanged") is True
    and trees.get("preflight_receipt") == trees.get("postflight_receipt"),
    "Bernini/VeOmni source-tree postflight differs",
)
tree_receipt = trees["preflight_receipt"]
check_seal(tree_receipt, "model source-tree receipt")
check_no_update(tree_receipt, "model source-tree receipt")
require(
    tree_receipt.get("schema_version") == "bernini-pair-v7-source-tree-binding-v2"
    and tree_receipt.get("tracked_or_extracted_source_bytes_verified") is True,
    "source-tree bytes were not verified",
)
for label, root_value, revision in (
    ("bernini", bernini_root, bernini_revision),
    ("veomni", veomni_root, veomni_revision),
):
    value = tree_receipt.get(label)
    require(
        isinstance(value, dict)
        and value.get("root") == root_value
        and value.get("revision") == revision
        and isinstance(value.get("file_count"), int)
        and value.get("file_count") > 0
        and isinstance(value.get("regular_file_count"), int)
        and isinstance(value.get("tracked_relative_symlink_count"), int)
        and value.get("regular_file_count")
        + value.get("tracked_relative_symlink_count")
        == value.get("file_count")
        and value.get("tracked_relative_symlink_link_and_target_bytes_verified")
        is True
        and SHA256.fullmatch(str(value.get("file_entries_digest"))) is not None
        and value.get("closure_mode") in {"git_tracked_files", "extracted_source_and_config_files"},
        f"{label} source-tree binding differs",
    )

print(
    json.dumps(
        {
            "measurement_receipt_verified": True,
            "geometry_audit_passed": receipt.get("geometry_audit_passed"),
            "schedule_index": 33,
            "exact81": True,
            "fps": 25,
            "world8_dp2_sp4": True,
            "parameter_update": False,
            "action_success_claim": False,
        },
        sort_keys=True,
    )
)
PY

echo "[pair-v7-phase-a] MEASUREMENT_COMPLETE_ONLY output=${output} schedule=33 exact81=true fps=25 parameter_update=false action_success_claim=false"
