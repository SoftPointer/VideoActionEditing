#!/usr/bin/env bash
#SBATCH --job-name=pair7-phasea2-8x64
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=08:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 027

fail() { echo "[pair-v7-phase-a2] ERROR: $*" >&2; exit 2; }
absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

[[ "${PAIR_V7_ROOT_REVIEWED_PHASE_A2:-NO}" == YES ]] || \
  fail "set PAIR_V7_ROOT_REVIEWED_PHASE_A2=YES only after reviewing the immutable runtime archive"
[[ "${PAIR_V7_ACK_PHASE_A2_READ_ONLY:-NO}" == YES ]] || \
  fail "set PAIR_V7_ACK_PHASE_A2_READ_ONLY=YES to acknowledge measurement-only semantics"
[[ "${PAIR_V7_ACK_PHASE_B_REMEASURE:-NO}" == YES ]] || \
  fail "acknowledge that a GO requires a separate Phase-B job to remeasure and apply in memory"

runtime_archive="${PAIR_V7_PHASE_A2_RUNTIME_ARCHIVE:?set immutable Phase-A2 git archive}"
runtime_archive_sha256="${PAIR_V7_PHASE_A2_RUNTIME_ARCHIVE_SHA256:?set runtime archive SHA-256}"
runtime_revision="${PAIR_V7_PHASE_A2_RUNTIME_REVISION:?set runtime archive commit}"
cast_archive="${PAIR_V7_CAST_V4_METHOD_ARCHIVE:?set immutable CAST-v4 method archive}"
cast_archive_sha256="${PAIR_V7_CAST_V4_METHOD_ARCHIVE_SHA256:?set CAST-v4 archive SHA-256}"
cast_revision="${PAIR_V7_CAST_V4_METHOD_REVISION:?set CAST-v4 method revision}"
cast_root="${PAIR_V7_CAST_V4_ROOT_SPEC:?set CAST-v4 root spec}"
cast_root_sha256="${PAIR_V7_CAST_V4_ROOT_SPEC_SHA256:?set CAST-v4 root SHA-256}"
plan="${PAIR_V7_MULTICONDITION_PREREGISTRATION:?set sealed Phase-A2 preregistration}"
plan_sha256="${PAIR_V7_MULTICONDITION_PREREGISTRATION_SHA256:?set preregistration SHA-256}"
cast_group_a="${PAIR_V7_CAST_V4_GROUP_A_RECEIPT:?set CAST-v4 group-A receipt}"
cast_group_a_sha256="${PAIR_V7_CAST_V4_GROUP_A_RECEIPT_SHA256:?set group-A receipt SHA-256}"
cast_group_b="${PAIR_V7_CAST_V4_GROUP_B_RECEIPT:?set CAST-v4 group-B receipt}"
cast_group_b_sha256="${PAIR_V7_CAST_V4_GROUP_B_RECEIPT_SHA256:?set group-B receipt SHA-256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set pinned Bernini source root}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set pinned VeOmni source root}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set pinned Bernini-R 1.3B checkpoint}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set checkpoint content manifest}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set checkpoint manifest SHA-256}"
output="${PAIR_V7_PHASE_A2_OUTPUT:?set fresh absolute receipt-only output}"
python_bin="${PAIR_V7_PHASE_A2_PYTHON_BIN:?set vace Python executable}"
torchrun_nnodes="${PAIR_V7_TORCHRUN_NNODES:-1}"
torchrun_nproc_per_node="${PAIR_V7_TORCHRUN_NPROC_PER_NODE:-8}"
torchrun_node_rank="${PAIR_V7_TORCHRUN_NODE_RANK:-0}"
torchrun_master_addr="${PAIR_V7_TORCHRUN_MASTER_ADDR:-127.0.0.1}"
torchrun_master_port="${PAIR_V7_TORCHRUN_MASTER_PORT:-29500}"
visible_devices="${PAIR_V7_VISIBLE_DEVICES:-}"

readonly expected_plan_sha256="cfd065c6ad84c4598b76ae5d0c390fe69bcbcaaad643045af4ac719c030f52df"
readonly expected_plan_digest="ad8a9d6ae462c28f48bf8c20cb649903ccca3941af6660e0cab4be214eba1790"
readonly expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
readonly expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
readonly expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
readonly expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
readonly expected_cast_revision="d5e87caf8f58a63c2ff902386d6e19100f7fcf34"
readonly expected_cast_archive_sha256="a15a95ea9bcee5cbe331ef975ce2cef38e6c164df644d8d053a87b5fe71c7470"
readonly expected_cast_root_sha256="a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
readonly expected_cast_group_a_sha256="bc20af168fdb29f72854d6c4c0f978eb29ba6642c1a5cbbeab69f33bf068ea1a"
readonly expected_cast_group_b_sha256="4a95e306c560b37b52f204c8d818774d43b648ff6fc79e5c704c9e69cdc2ce75"

for name in runtime_archive cast_archive cast_root plan cast_group_a cast_group_b bernini_root veomni_root checkpoint checkpoint_manifest output python_bin; do
  absolute "${!name}" "${name}"
done
for name in runtime_archive_sha256 cast_archive_sha256 cast_root_sha256 plan_sha256 cast_group_a_sha256 cast_group_b_sha256 checkpoint_manifest_sha256; do
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || fail "${name} must be lowercase SHA-256"
done
for name in runtime_revision cast_revision; do
  [[ "${!name}" =~ ^[0-9a-f]{40}$ ]] || fail "${name} must be lowercase Git commit"
done
[[ "${plan_sha256}" == "${expected_plan_sha256}" ]] || fail "Phase-A2 plan pin differs"
[[ "${checkpoint_manifest_sha256}" == "${expected_checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest pin differs"
[[ "${cast_revision}" == "${expected_cast_revision}" ]] || fail "CAST-v4 revision differs"
[[ "${cast_archive_sha256}" == "${expected_cast_archive_sha256}" ]] || fail "CAST-v4 archive pin differs"
[[ "${cast_root_sha256}" == "${expected_cast_root_sha256}" ]] || fail "CAST-v4 root pin differs"
[[ "${cast_group_a_sha256}" == "${expected_cast_group_a_sha256}" ]] || fail "group-A pin differs"
[[ "${cast_group_b_sha256}" == "${expected_cast_group_b_sha256}" ]] || fail "group-B pin differs"

runtime_archive="$(realpath -e -- "${runtime_archive}")"
cast_archive="$(realpath -e -- "${cast_archive}")"
cast_root="$(realpath -e -- "${cast_root}")"
plan="$(realpath -e -- "${plan}")"
cast_group_a="$(realpath -e -- "${cast_group_a}")"
cast_group_b="$(realpath -e -- "${cast_group_b}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

for name in runtime_archive cast_archive cast_root plan cast_group_a cast_group_b checkpoint_manifest python_bin; do
  [[ -f "${!name}" && ! -L "${!name}" ]] || fail "${name} must resolve to a plain file"
done
for name in bernini_root veomni_root checkpoint; do
  [[ -d "${!name}" && ! -L "${!name}" ]] || fail "${name} must resolve to a plain directory"
done
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${torchrun_nnodes}" =~ ^[0-9]+$ && "${torchrun_nnodes}" -ge 1 ]] || fail "torchrun node count differs"
[[ "${torchrun_nproc_per_node}" =~ ^[0-9]+$ && "${torchrun_nproc_per_node}" -ge 1 ]] || fail "torchrun process count differs"
[[ "${torchrun_node_rank}" =~ ^[0-9]+$ && "${torchrun_node_rank}" -lt "${torchrun_nnodes}" ]] || fail "torchrun node rank differs"
[[ "${torchrun_master_port}" =~ ^[0-9]+$ && "${torchrun_master_port}" -ge 1024 && "${torchrun_master_port}" -le 65535 ]] || fail "torchrun master port differs"
if [[ "${torchrun_nnodes}" == 1 ]]; then
  [[ "${torchrun_nproc_per_node}" == 8 && "${torchrun_node_rank}" == 0 ]] || \
    fail "single-node topology must be 1x8"
elif [[ "${torchrun_nnodes}" == 2 ]]; then
  [[ "${torchrun_nproc_per_node}" == 4 ]] || fail "two-node topology must be 2x4"
  [[ "${torchrun_master_addr}" != 127.0.0.1 && -n "${visible_devices}" ]] || \
    fail "two-node topology requires explicit master address and GPU quartet"
else
  fail "only the registered 1x8 and 2x4 WORLD8 topologies are allowed"
fi
[[ "${output}" != / && "${output##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output"
[[ "$(realpath -m -- "${output}")" == "${output}" ]] || fail "output must be canonical"
[[ ! -e "${output}" && ! -L "${output}" ]] || fail "output must be create-only"
mkdir -p -- "$(dirname -- "${output}")"
[[ ! -L "$(dirname -- "${output}")" ]] || fail "output parent is a symlink"

sealed_inputs=(
  "${runtime_archive}" "${runtime_archive_sha256}"
  "${cast_archive}" "${cast_archive_sha256}"
  "${cast_root}" "${cast_root_sha256}"
  "${plan}" "${plan_sha256}"
  "${cast_group_a}" "${cast_group_a_sha256}"
  "${cast_group_b}" "${cast_group_b_sha256}"
  "${checkpoint_manifest}" "${checkpoint_manifest_sha256}"
)
for ((index=0; index<${#sealed_inputs[@]}; index+=2)); do
  [[ "$(sha256_file "${sealed_inputs[index]}")" == "${sealed_inputs[index+1]}" ]] || \
    fail "sealed input bytes differ: ${sealed_inputs[index]}"
done
[[ "$(git get-tar-commit-id <"${runtime_archive}")" == "${runtime_revision}" ]] || fail "runtime archive revision differs"
[[ "$(git get-tar-commit-id <"${cast_archive}")" == "${cast_revision}" ]] || fail "CAST archive revision differs"
(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed before Phase-A2"

"${python_bin}" -B - "${plan}" "${expected_plan_digest}" <<'PY'
import json
from pathlib import Path
import sys
row=json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
if row.get("preregistration_digest") != sys.argv[2]:
    raise SystemExit("embedded Phase-A2 preregistration digest differs")
if row.get("primary_schedule_indices") != [16,35] or row.get("event_count") != 4:
    raise SystemExit("Phase-A2 plan factorial differs")
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
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v7-phase-a2-${SLURM_JOB_ID}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v7-phase-a2-${SLURM_JOB_ID}."*) ;; *) exit 2 ;; esac
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
# The TOCTOU bank authority hashes absolute external-file paths.  All nodes in
# a multi-node WORLD must therefore bind the shared immutable archive path,
# while each node still extracts and executes its independently verified local
# copy.  Single-node jobs retain the historical staged-path binding.
runtime_binding_archive="${archive_copy}"
if [[ "${torchrun_nnodes}" == 2 ]]; then
  runtime_binding_archive="${runtime_archive}"
fi

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required={
 "methods/bernini_action_editing/audit_pair_v7_phase_a2_multicondition_geometry.py",
 "methods/bernini_action_editing/audit_pair_v7_phase_a_geometry.py",
 "methods/bernini_action_editing/pair_v7_multicondition_geometry_authority.py",
 "methods/bernini_action_editing/pair_v7_multicondition_nullspace_transport.py",
 "methods/bernini_action_editing/pair_v7_dual_coordinate_nullspace_transport.py",
 "methods/bernini_action_editing/pair_v7_fit_only_geometry_authority.py",
 "methods/bernini_action_editing/pair_v5_t2v_guidance_distill.py",
 "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
 "methods/bernini_action_editing/source_self_runtime.py",
 "methods/bernini_action_editing/postflight_pair_v7_phase_a2_multicondition.py",
 "methods/bernini_action_editing/tests/test_audit_pair_v7_phase_a2_multicondition_geometry.py",
 "methods/bernini_action_editing/tests/test_pair_v7_multicondition_geometry_authority.py",
 "methods/bernini_action_editing/tests/test_pair_v7_multicondition_nullspace_transport.py",
 "methods/bernini_action_editing/scripts/auh_audit_pair_v7_phase_a2_multicondition_dp2sp4.sbatch",
}
seen=set()
with tarfile.open(sys.argv[1],"r:*") as handle:
    for member in handle.getmembers():
        pure=PurePosixPath(member.name)
        name=pure.as_posix().lstrip("./").rstrip("/")
        if (not name or pure.is_absolute() or ".." in pure.parts or member.issym()
                or member.islnk() or member.isdev() or member.isfifo() or name in seen):
            raise SystemExit("unsafe or duplicate runtime archive member")
        seen.add(name)
missing=sorted(required-seen)
if missing:
    raise SystemExit(f"runtime archive lacks Phase-A2 closure: {missing}")
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
  test_pair_v7_multicondition_geometry_authority.py \
  test_pair_v7_multicondition_nullspace_transport.py \
  test_audit_pair_v7_phase_a2_multicondition_geometry.py; do
  PYTHONPATH="${method_root}" "${python_bin}" -B -m unittest discover -v \
    -s "${method_root}/tests" -p "${test_name}"
done

common_args=(
  --bernini-root "${bernini_root}"
  --veomni-root "${veomni_root}"
  --checkpoint "${checkpoint}"
  --checkpoint-content-manifest "${checkpoint_manifest}"
  --expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"
  --expected-checkpoint-tree-sha256 "${expected_checkpoint_tree_sha256}"
  --multicondition-preregistration "${plan}"
  --expected-preregistration-sha256 "${plan_sha256}"
  --cast-method-archive "${cast_archive}"
  --cast-method-archive-sha256 "${cast_archive_sha256}"
  --cast-method-revision "${cast_revision}"
  --cast-root-spec "${cast_root}"
  --expected-cast-root-spec-sha256 "${cast_root_sha256}"
  --scorer-group-receipt "${cast_group_a}"
  --scorer-group-receipt "${cast_group_b}"
  --expected-scorer-group-receipt-sha256 "${cast_group_a_sha256}"
  --expected-scorer-group-receipt-sha256 "${cast_group_b_sha256}"
  --runtime-source-archive "${runtime_binding_archive}"
  --runtime-source-archive-sha256 "${runtime_archive_sha256}"
  --runtime-source-revision "${runtime_revision}"
  --output "${output}"
  --expected-bernini-commit "${expected_bernini_commit}"
  --expected-veomni-commit "${expected_veomni_commit}"
  --ack-root-reviewed-phase-a2-launch
  --ack-no-parameter-mutation-no-success-claim
)

echo "[pair-v7-phase-a2] preflight fit+confirmation x s16/s35, exact81, no update"
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/audit_pair_v7_phase_a2_multicondition_geometry.py" \
  "${common_args[@]}" --preflight-only

if [[ -n "${visible_devices}" ]]; then
  [[ "${visible_devices}" == 0,1,2,3 ]] || fail "registered two-node GPU quartet must be physical 0,1,2,3"
  export ROCR_VISIBLE_DEVICES="${visible_devices}"
  export HIP_VISIBLE_DEVICES="${visible_devices}"
  export CUDA_VISIBLE_DEVICES="${visible_devices}"
  export GPU_DEVICE_ORDINAL="${visible_devices}"
else
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
fi
echo "[pair-v7-phase-a2] allocation=8xMI210 topology=WORLD8/DP2xSP4 nodes=${torchrun_nnodes} local_processes=${torchrun_nproc_per_node} node_rank=${torchrun_node_rank} action=8 identity=64"
if [[ "${torchrun_nnodes}" == 1 ]]; then
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    "${method_root}/audit_pair_v7_phase_a2_multicondition_geometry.py" \
    "${common_args[@]}"
else
  PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=4 \
    --node_rank="${torchrun_node_rank}" \
    --master_addr="${torchrun_master_addr}" \
    --master_port="${torchrun_master_port}" \
    "${method_root}/audit_pair_v7_phase_a2_multicondition_geometry.py" \
    "${common_args[@]}"
fi

[[ -d "${output}" && ! -L "${output}" ]] || fail "Phase-A2 output directory is absent"
[[ -f "${output}/receipt.json" && ! -L "${output}/receipt.json" ]] || fail "Phase-A2 receipt is absent"
[[ "$(find "${output}" -mindepth 1 -maxdepth 1 -printf '%f\n')" == receipt.json ]] || fail "Phase-A2 output must contain only receipt.json"

for ((index=0; index<${#sealed_inputs[@]}; index+=2)); do
  [[ "$(sha256_file "${sealed_inputs[index]}")" == "${sealed_inputs[index+1]}" ]] || \
    fail "sealed input changed during Phase-A2: ${sealed_inputs[index]}"
done
[[ "$(sha256_file "${archive_copy}")" == "${runtime_archive_sha256}" ]] || fail "staged runtime archive changed"
(
  cd "${checkpoint}"
  sha256sum --strict --status -c "${checkpoint_manifest}"
) || fail "checkpoint content verification failed after Phase-A2"
method_tree_after="$(find "${method_root}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${method_tree_before}" == "${method_tree_after}" ]] || fail "read-only extracted runtime tree changed"

# Re-run the complete plan + 2 group + 40 child + checkpoint preflight after
# measurement so the launcher independently closes the same TOCTOU envelope.
PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/audit_pair_v7_phase_a2_multicondition_geometry.py" \
  "${common_args[@]}" --preflight-only

PYTHONPATH="${method_root}" "${python_bin}" -B \
  "${method_root}/postflight_pair_v7_phase_a2_multicondition.py" \
  --receipt "${output}/receipt.json" \
  --expected-runtime-archive-sha256 "${runtime_archive_sha256}"

"${python_bin}" -B - "${output}/receipt.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1],"r",encoding="ascii"))
go=row.get("primary_replication_go")
if type(go) is not bool:
    raise SystemExit("Phase-A2 GO/NO-GO type differs")
if go:
    print("[pair-v7-phase-a2] GO receipt only; DO NOT update here. Submit a separately root-authorized Phase-B job that remeasures 8+64 and applies the solved direction in memory.")
else:
    print("[pair-v7-phase-a2] scientific NO-GO; no update is authorized.")
PY
