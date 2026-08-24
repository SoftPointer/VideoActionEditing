#!/usr/bin/env bash
#SBATCH --job-name=pair5-native-d541-v3
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=24:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Score one sealed native RV2V action-only population with two independent
# persistent Ulysses-SP4 frozen-Bernini workers.  Each candidate uses its own
# predecode clean latent and its own captured official Gaussian.  Global MACE
# runs at exact40 index 33 / native model timestep 516.  Its formal scalar is
# the exact d541801 live-FP32 two-log/first-argmin expression, validated against
# a separate hash-pinned d541801 source archive.  T2V media, RV2V MP4,
# source condition latents, targets, masks, flow, pose, tracks, and trajectories
# never enter the scorer.

source_archive="${PAIR_V5_NATIVE_SCORE_SOURCE_ARCHIVE:?set PAIR_V5_NATIVE_SCORE_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_NATIVE_SCORE_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_NATIVE_SCORE_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_NATIVE_SCORE_SOURCE_REVISION:?set PAIR_V5_NATIVE_SCORE_SOURCE_REVISION}"
formal_v3_source_archive="${PAIR_V5_T2V_V3_SOURCE_ARCHIVE:?set PAIR_V5_T2V_V3_SOURCE_ARCHIVE}"
formal_v3_source_archive_sha256="${PAIR_V5_T2V_V3_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_T2V_V3_SOURCE_ARCHIVE_SHA256}"
formal_v3_source_revision="${PAIR_V5_T2V_V3_SOURCE_REVISION:-d541801a162796aacde34c2bfc2b1f0472d954d2}"
population_spec="${PAIR_V5_NATIVE_POPULATION_SPEC:?set PAIR_V5_NATIVE_POPULATION_SPEC}"
population_spec_sha256="${PAIR_V5_NATIVE_POPULATION_SPEC_SHA256:?set PAIR_V5_NATIVE_POPULATION_SPEC_SHA256}"
rollout_root="${PAIR_V5_NATIVE_ROLLOUT_ROOT:?set PAIR_V5_NATIVE_ROLLOUT_ROOT}"
t2v_bank_spec="${PAIR_V5_T2V_BANK_SPEC:?set PAIR_V5_T2V_BANK_SPEC}"
t2v_bank_spec_sha256="${PAIR_V5_T2V_BANK_SPEC_SHA256:?set PAIR_V5_T2V_BANK_SPEC_SHA256}"
t2v_bank_receipt="${PAIR_V5_T2V_BANK_RECEIPT:?set PAIR_V5_T2V_BANK_RECEIPT}"
t2v_bank_receipt_sha256="${PAIR_V5_T2V_BANK_RECEIPT_SHA256:?set PAIR_V5_T2V_BANK_RECEIPT_SHA256}"
t2v_score_root="${PAIR_V5_T2V_SCORE_ROOT:?set PAIR_V5_T2V_SCORE_ROOT}"
t2v_calibration_root="${PAIR_V5_T2V_CALIBRATION_ROOT:?set PAIR_V5_T2V_CALIBRATION_ROOT}"
t2v_calibration_receipt_sha256="${PAIR_V5_T2V_CALIBRATION_RECEIPT_SHA256:?set PAIR_V5_T2V_CALIBRATION_RECEIPT_SHA256}"
t2v_preregistration_sha256="${PAIR_V5_T2V_PREREGISTRATION_SHA256:?set PAIR_V5_T2V_PREREGISTRATION_SHA256}"
checkpoint_tree_sha256="${BERNINI_CHECKPOINT_TREE_SHA256:?set BERNINI_CHECKPOINT_TREE_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
checkpoint_manifest_sha256="${BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256}"
output_dir="${PAIR_V5_NATIVE_SCORE_OUTPUT_DIR:?set PAIR_V5_NATIVE_SCORE_OUTPUT_DIR}"
python_bin="${PAIR_V5_NATIVE_SCORE_PYTHON_BIN:?set PAIR_V5_NATIVE_SCORE_PYTHON_BIN}"

fail() { echo "[pair-v5-native-d541-v3] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive formal_v3_source_archive population_spec rollout_root t2v_bank_spec t2v_bank_receipt t2v_score_root t2v_calibration_root bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
for value in "${source_archive_sha256}" "${formal_v3_source_archive_sha256}" "${population_spec_sha256}" "${t2v_bank_spec_sha256}" "${t2v_bank_receipt_sha256}" "${t2v_calibration_receipt_sha256}" "${t2v_preregistration_sha256}" "${checkpoint_tree_sha256}" "${checkpoint_manifest_sha256}"; do
  [[ "${value}" =~ ^[0-9a-f]{64}$ ]] || fail "a required SHA-256 differs"
done
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${formal_v3_source_revision}" == d541801a162796aacde34c2bfc2b1f0472d954d2 ]] || fail "formal v3 source revision must be exact d541801"

source_archive="$(realpath -e -- "${source_archive}")"
formal_v3_source_archive="$(realpath -e -- "${formal_v3_source_archive}")"
population_spec="$(realpath -e -- "${population_spec}")"
rollout_root="$(realpath -e -- "${rollout_root}")"
t2v_bank_spec="$(realpath -e -- "${t2v_bank_spec}")"
t2v_bank_receipt="$(realpath -e -- "${t2v_bank_receipt}")"
t2v_score_root="$(realpath -e -- "${t2v_score_root}")"
t2v_calibration_root="$(realpath -e -- "${t2v_calibration_root}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${formal_v3_source_archive}" && ! -L "${formal_v3_source_archive}" ]] || fail "formal v3 source archive differs"
[[ -f "${population_spec}" && ! -L "${population_spec}" ]] || fail "population spec differs"
[[ -d "${rollout_root}" && ! -L "${rollout_root}" ]] || fail "rollout root differs"
[[ -f "${t2v_bank_spec}" && ! -L "${t2v_bank_spec}" ]] || fail "T2V bank spec differs"
[[ -f "${t2v_bank_receipt}" && ! -L "${t2v_bank_receipt}" ]] || fail "T2V bank receipt differs"
[[ -d "${t2v_score_root}" && ! -L "${t2v_score_root}" ]] || fail "T2V score root differs"
[[ -d "${t2v_calibration_root}" && ! -L "${t2v_calibration_root}" ]] || fail "T2V calibration root differs"
[[ -f "${t2v_calibration_root}/calibration-receipt-v3.json" && ! -L "${t2v_calibration_root}/calibration-receipt-v3.json" ]] || fail "T2V calibration receipt differs"
[[ -f "${t2v_calibration_root}/preregistration-v3.json" && ! -L "${t2v_calibration_root}/preregistration-v3.json" ]] || fail "T2V preregistration differs"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source revision/archive differs"
[[ "$(sha256sum "${formal_v3_source_archive}" | awk '{print $1}')" == "${formal_v3_source_archive_sha256}" ]] || fail "formal v3 source archive hash differs"
[[ "$(git get-tar-commit-id <"${formal_v3_source_archive}")" == "${formal_v3_source_revision}" ]] || fail "formal v3 source revision/archive differs"
[[ "$(sha256sum "${population_spec}" | awk '{print $1}')" == "${population_spec_sha256}" ]] || fail "population spec hash differs"
[[ "$(sha256sum "${t2v_bank_spec}" | awk '{print $1}')" == "${t2v_bank_spec_sha256}" ]] || fail "T2V bank spec hash differs"
[[ "$(sha256sum "${t2v_bank_receipt}" | awk '{print $1}')" == "${t2v_bank_receipt_sha256}" ]] || fail "T2V bank receipt hash differs"
[[ "$(sha256sum "${t2v_calibration_root}/calibration-receipt-v3.json" | awk '{print $1}')" == "${t2v_calibration_receipt_sha256}" ]] || fail "T2V calibration receipt hash differs"
[[ "$(sha256sum "${t2v_calibration_root}/preregistration-v3.json" | awk '{print $1}')" == "${t2v_preregistration_sha256}" ]] || fail "T2V preregistration hash differs"
[[ "$(sha256sum "${checkpoint_manifest}" | awk '{print $1}')" == "${checkpoint_manifest_sha256}" ]] || fail "checkpoint manifest hash differs"

[[ "${output_dir}" != / ]] || fail "output cannot be root"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
mkdir -p -- "$(dirname -- "${output_dir}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-native-d541-v3-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-native-d541-v3-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-native-d541-v3] PASS topology=2xSP4 exact81=true native_timestep=516 d541801_v3=true candidate_own_xsigma=true safe_pareto_action_only=true output=${output_dir}"
  else
    echo "[pair-v5-native-d541-v3] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
  fi
  exit "${status}"
}
terminate_children() {
  local status="$1"
  trap - TERM INT
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && kill "${child}" 2>/dev/null || true; done
  for child in "${sp4_a_pid}" "${sp4_b_pid}"; do [[ "${child}" =~ ^[0-9]+$ ]] && wait "${child}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'terminate_children 143' TERM
trap 'terminate_children 130' INT

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/formal-v3-source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
formal_v3_archive_copy="${task_scratch}/formal-v3-source.tar"
cp -- "${source_archive}" "${archive_copy}"
cp -- "${formal_v3_source_archive}" "${formal_v3_archive_copy}"
chmod 0444 -- "${archive_copy}"
chmod 0444 -- "${formal_v3_archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
 "methods/bernini_action_editing/pair_v5_native_rv2v_action_score_d541801_v3.py",
 "methods/bernini_action_editing/pair_v5_native_rollout_spec.py",
 "methods/bernini_action_editing/validate_pair_v5_t2v_calibration_d541801_v3.py",
 "methods/bernini_action_editing/pair_v5_t2v_score_d541801_v3_compat.py",
 "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
 "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
 "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
 "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
 "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
 "methods/bernini_action_editing/infer_lora.py",
 "methods/bernini_action_editing/train_lora.py",
 "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
 "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
 "methods/bernini_action_editing/source_kv_replay.py",
 "methods/bernini_action_editing/source_kv_route_batches.py",
 "methods/bernini_action_editing/source_value_residual.py",
 "methods/bernini_action_editing/pair_v5_native_bridge.py",
 "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
 "methods/bernini_action_editing/mace_candidate_action_energy.py",
 "methods/bernini_action_editing/pair_v5_action_adapter.py",
 "methods/bernini_action_editing/inference_sigma_strata.py",
 "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
 "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
 "methods/bernini_action_editing/source_self_native_target_adapter.py",
 "methods/bernini_action_editing/dclr_runtime_contract.py",
}
seen=set()
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("archive member escaped or is not a plain tree entry")
  seen.add(path.as_posix().lstrip("./"))
missing=sorted(required-seen)
if missing: raise SystemExit(f"archive lacks PAIR-v5 native scorer closure: {missing}")
PY

"${python_bin}" -B - "${formal_v3_archive_copy}" <<'PY'
from pathlib import PurePosixPath
import hashlib,sys,tarfile
required = {
 "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py": "3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e",
 "methods/bernini_action_editing/mace_candidate_action_energy.py": "0cd4b2c86aa9ccdd353010a8750eeecfebfb28425b44616e9fd52810dd90e986",
}
seen={}
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("formal v3 archive member escaped or is not a plain tree entry")
  normalized=path.as_posix().lstrip("./")
  if normalized in required:
   extracted=handle.extractfile(member)
   if extracted is None: raise SystemExit("formal v3 critical source is not a regular file")
   seen[normalized]=hashlib.sha256(extracted.read()).hexdigest()
if seen!=required: raise SystemExit(f"formal v3 critical source closure differs: {seen}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
tar --no-same-owner --no-same-permissions -xf "${formal_v3_archive_copy}" -C "${task_scratch}/formal-v3-source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
formal_v3_method_root="${task_scratch}/formal-v3-source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
find "${formal_v3_method_root}" -type f -exec chmod a-w -- {} +
mkdir -- "${output_dir}"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_port=$((30000 + 2 * job_mod))
sp4_b_port=$((sp4_a_port + 1))

run_group() (
  set -Eeuo pipefail
  local group_id="$1" visible_gpus="$2" master_port="$3"
  local group_scratch="${task_scratch}/groups/${group_id}"
  mkdir -p -- "${group_scratch}/cache/miopen-user" "${group_scratch}/cache/miopen-custom" \
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
  export TORCHELASTIC_ERROR_FILE="${group_scratch}/torch-elastic-error.json"
  "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${master_port}" \
    "${method_root}/pair_v5_native_rv2v_action_score_d541801_v3.py" \
    --population-spec "${population_spec}" --expected-population-spec-sha256 "${population_spec_sha256}" \
    --rollout-root "${rollout_root}" \
    --t2v-bank-spec "${t2v_bank_spec}" --expected-t2v-bank-spec-sha256 "${t2v_bank_spec_sha256}" \
    --t2v-bank-receipt "${t2v_bank_receipt}" \
    --expected-t2v-bank-receipt-sha256 "${t2v_bank_receipt_sha256}" \
    --t2v-score-root "${t2v_score_root}" --t2v-calibration-root "${t2v_calibration_root}" \
    --expected-t2v-calibration-receipt-sha256 "${t2v_calibration_receipt_sha256}" \
    --expected-t2v-preregistration-sha256 "${t2v_preregistration_sha256}" \
    --formal-v3-method-root "${formal_v3_method_root}" \
    --formal-v3-source-revision "${formal_v3_source_revision}" \
    --formal-v3-source-archive-sha256 "${formal_v3_source_archive_sha256}" \
    --checkpoint-tree-sha256 "${checkpoint_tree_sha256}" --group-id "${group_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
    --output-dir "${output_dir}/${group_id}" --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}" \
    --ack-action-metric-is-not-action-success
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 200 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 200 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more native SP4 score groups failed"
fi

"${python_bin}" -B - "${method_root}" "${population_spec}" "${population_spec_sha256}" \
  "${t2v_bank_spec}" "${t2v_bank_spec_sha256}" \
  "${t2v_bank_receipt}" "${t2v_bank_receipt_sha256}" "${t2v_score_root}" \
  "${t2v_calibration_root}" "${t2v_calibration_receipt_sha256}" \
  "${t2v_preregistration_sha256}" "${checkpoint_tree_sha256}" \
  "${formal_v3_method_root}" "${formal_v3_source_revision}" \
  "${formal_v3_source_archive_sha256}" "${rollout_root}" "${output_dir}" <<'PY'
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
import pair_v5_native_rollout_spec as population_contract
import pair_v5_native_rv2v_action_score_d541801_v3 as scorer

spec,spec_sha=population_contract.load_sealed_spec(sys.argv[2],sys.argv[3])
bank_spec=sys.argv[4]; bank_spec_sha=sys.argv[5]
bank_receipt=sys.argv[6]; bank_receipt_sha=sys.argv[7]; score_root=sys.argv[8]
calibration_root=sys.argv[9]; calibration_sha=sys.argv[10]; prereg_sha=sys.argv[11]
checkpoint_tree=sys.argv[12]; formal_method_root=sys.argv[13]
formal_revision=sys.argv[14]; formal_archive_sha=sys.argv[15]
rollout_root=sys.argv[16]; root=Path(sys.argv[17])
bundle=scorer.load_mainline_calibration_bundle(
 t2v_bank_spec=bank_spec,
 t2v_bank_spec_sha256=bank_spec_sha,
 t2v_bank_receipt=bank_receipt,
 t2v_bank_receipt_sha256=bank_receipt_sha,
 t2v_score_root=score_root,
 t2v_calibration_root=calibration_root,
 t2v_calibration_receipt_sha256=calibration_sha,
 t2v_preregistration_sha256=prereg_sha,
 checkpoint_tree_sha256=checkpoint_tree,
 formal_v3_method_root=formal_method_root,
 formal_v3_source_revision=formal_revision,
 formal_v3_source_archive_sha256=formal_archive_sha,
 python_executable=sys.executable,
)
expected={g["group_id"]:[c["candidate_id"] for c in g["candidates"]] for g in spec["groups"]}
contexts={}
for group_id in ("sp4-a","sp4-b"):
 _,rows=scorer.load_native_group_population(
  population_spec_path=sys.argv[2],
  population_spec_sha256=spec_sha,
  rollout_root=rollout_root,
  group_id=group_id,
  calibration_bundle=bundle,
 )
 contexts.update({row["candidate"]["candidate_id"]:row for row in rows})
score_digests=[]; safe_digests=[]; x_by_cell={}; ordered=[]
for group_id in ("sp4-a","sp4-b"):
 group_path=root/group_id/scorer.GROUP_FILENAME.format(group_id=group_id)
 group=json.loads(group_path.read_bytes()); unsigned=dict(group); declared=unsigned.pop("receipt_digest",None)
 if hashlib.sha256(scorer.canonical_json_bytes(unsigned)).hexdigest()!=declared: raise SystemExit("group receipt digest differs")
 if group.get("candidate_order")!=expected[group_id] or group.get("checkpoint_tree_sha256")!=checkpoint_tree: raise SystemExit("group population/checkpoint binding differs")
 if group.get("formal_t2v_score_schema")!=bundle["authorization"]["formal_score_schema"] or group.get("formal_t2v_score_scalar_definition")!=bundle["authorization"]["formal_score_scalar_definition"]: raise SystemExit("group formal d541801-v3 score authority differs")
 if group.get("formal_v3_source_revision")!=formal_revision or group.get("formal_v3_source_archive_sha256")!=formal_archive_sha or group.get("formal_v3_source_binding_digest")!=bundle["authorization"]["formal_v3_source_binding_digest"]: raise SystemExit("group formal d541801-v3 source binding differs")
 if group.get("active_repository_action_scalar_consumed") is not False or group.get("decimal_or_log1p_action_scalar_consumed") is not False: raise SystemExit("group consumed non-v3 action scalar")
 for candidate_id in expected[group_id]:
  candidate_root=root/group_id/candidate_id
  score_path=candidate_root/scorer.SCORE_FILENAME
  safe_path=candidate_root/scorer.SAFE_PARETO_ACTION_FILENAME
  score=scorer.validate_score_receipt(json.loads(score_path.read_bytes()))
  safe=scorer.validate_safe_pareto_action_record(json.loads(safe_path.read_bytes()))
  context=contexts[candidate_id]
  clean=scorer.frozen_mace_runtime._load_exact81_tensor(
   context["native_artifacts"]["predecode_clean_latent"],
   key="normalized_clean_latent",label=f"{candidate_id} postflight clean",
  )
  epsilon=scorer.frozen_mace_runtime._load_exact81_tensor(
   context["native_artifacts"]["official_initial_gaussian"],
   key="official_initial_gaussian",label=f"{candidate_id} postflight Gaussian",
  )
  sigma=torch.tensor([scorer.PILOT_SIGMA],dtype=torch.float32)
  scorer.verify_score_against_context(
   score,row=context,calibration_bundle=bundle,clean=clean,epsilon=epsilon,sigma=sigma,
  )
  if safe["native_score_receipt_digest"]!=score["receipt_digest"]: raise SystemExit("safe-Pareto/score binding differs")
  if score["population"]["spec_raw_sha256"]!=spec_sha: raise SystemExit("score population binding differs")
  if score["calibration"]["mainline_authorization_digest"]!=bundle["authorization"]["authorization_digest"]: raise SystemExit("score mainline calibration binding differs")
  if score["calibration"]["formal_score_provenance_set_digest"]!=bundle["formal_score_provenance_set_digest"]: raise SystemExit("score formal T2V provenance binding differs")
  if score["calibration"]["formal_score_schema"]!=scorer.formal_v3_compat.FORMAL_SCORE_SCHEMA: raise SystemExit("score formal d541801-v3 schema binding differs")
  if score["calibration"]["formal_v3_source_revision"]!=formal_revision or score["calibration"]["formal_v3_source_archive_sha256"]!=formal_archive_sha: raise SystemExit("score formal d541801-v3 source binding differs")
  if score["rollout"]["checkpoint_tree_sha256"]!=checkpoint_tree: raise SystemExit("score checkpoint binding differs")
  cell=score["candidate"]["calibration_group_id"]
  xhash=score["artifacts"]["candidate_own_x_sigma_tensor_sha256"]
  if xhash in x_by_cell.setdefault(cell,set()): raise SystemExit("two candidates in one cell reused x_sigma")
  x_by_cell[cell].add(xhash)
  ordered.append(candidate_id); score_digests.append(score["receipt_digest"]); safe_digests.append(safe["record_digest"])
 if group["score_receipt_digests"]!=score_digests[-len(expected[group_id]):]: raise SystemExit("group score digest order differs")
 if group["safe_pareto_record_digests"]!=safe_digests[-len(expected[group_id]):]: raise SystemExit("group safe digest order differs")
unsigned={
 "schema_version":"bernini-pair-v5-native-rv2v-global-action-score-bank-d541801-v3",
 "population_spec_raw_sha256":spec_sha,
 "mainline_calibration_authorization_digest":bundle["authorization"]["authorization_digest"],
 "formal_t2v_score_provenance_set_digest":bundle["formal_score_provenance_set_digest"],
 "formal_t2v_score_schema":bundle["authorization"]["formal_score_schema"],
 "formal_t2v_score_scalar_definition":bundle["authorization"]["formal_score_scalar_definition"],
 "formal_v3_source_revision":formal_revision,
 "formal_v3_source_archive_sha256":formal_archive_sha,
 "formal_v3_source_binding_digest":bundle["authorization"]["formal_v3_source_binding_digest"],
 "calibration_receipt_digest":bundle["calibration"]["receipt_digest"],
 "family_mapping_set_digest":bundle["family_mapping_set_digest"],
 "checkpoint_tree_sha256":checkpoint_tree,
 "candidate_order":ordered,
 "candidate_order_digest":scorer.object_sha256(ordered),
 "score_receipt_digests":score_digests,
 "safe_pareto_record_digests":safe_digests,
 "candidate_count":len(ordered),
 "two_independent_sp4_groups":True,
 "candidate_own_x_sigma_verified":True,
 "d541801_live_fp32_action_energy_verified":True,
 "active_repository_action_scalar_consumed":False,
 "decimal_or_log1p_action_scalar_consumed":False,
 "safe_pareto_still_requires_identity_camera_temporal_metrics":True,
 "optimizer_authorized":False,
 "scientific_action_editing_claim":False,
}
bank={**unsigned,"receipt_digest":scorer.object_sha256(unsigned)}
path=root/"pair-v5-native-rv2v-action-score-bank-d541801-v3.json"
if path.exists() or path.is_symlink(): raise SystemExit("refusing score-bank overwrite")
path.write_bytes(scorer.canonical_json_bytes(bank)+b"\n"); os.chmod(path,0o400)
print("PAIR_V5_NATIVE_RV2V_ACTION_SCORE_D541801_V3_DUAL4_STRONG_AUDIT_OK",len(ordered),bank["receipt_digest"])
PY
