#!/usr/bin/env bash
#SBATCH --job-name=pair5-srcbound-v1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=06:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# Two independent rank-4 evaluator groups.  Every rank owns exactly one of the
# sealed current-family action-only RV2V candidates.  This is data-parallel
# post-video scoring, not Bernini/Ulysses inference.  No target/proposal/donor,
# caption condition, mask, flow, pose, track, or trajectory enters the scorer.

source_archive="${PAIR_V5_SOURCE_BOUND_SOURCE_ARCHIVE:?set PAIR_V5_SOURCE_BOUND_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_SOURCE_BOUND_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_SOURCE_BOUND_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_SOURCE_BOUND_SOURCE_REVISION:?set PAIR_V5_SOURCE_BOUND_SOURCE_REVISION}"
rollout_spec="${PAIR_V5_SOURCE_BOUND_ROLLOUT_SPEC:?set PAIR_V5_SOURCE_BOUND_ROLLOUT_SPEC}"
rollout_spec_sha256="${PAIR_V5_SOURCE_BOUND_ROLLOUT_SPEC_SHA256:?set PAIR_V5_SOURCE_BOUND_ROLLOUT_SPEC_SHA256}"
evaluator_spec="${PAIR_V5_SOURCE_BOUND_EVALUATOR_SPEC:?set PAIR_V5_SOURCE_BOUND_EVALUATOR_SPEC}"
evaluator_spec_sha256="${PAIR_V5_SOURCE_BOUND_EVALUATOR_SPEC_SHA256:?set PAIR_V5_SOURCE_BOUND_EVALUATOR_SPEC_SHA256}"
rollout_output="${PAIR_V5_SOURCE_BOUND_ROLLOUT_OUTPUT_DIR:?set PAIR_V5_SOURCE_BOUND_ROLLOUT_OUTPUT_DIR}"
visual_checkpoint="${PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT:?set PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT}"
visual_checkpoint_manifest="${PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT_MANIFEST:?set PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT_MANIFEST}"
visual_checkpoint_manifest_sha256="${PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT_MANIFEST_SHA256:?set PAIR_V5_SOURCE_BOUND_VISUAL_CHECKPOINT_MANIFEST_SHA256}"
output_dir="${PAIR_V5_SOURCE_BOUND_OUTPUT_DIR:?set PAIR_V5_SOURCE_BOUND_OUTPUT_DIR}"
python_bin="${PAIR_V5_SOURCE_BOUND_PYTHON_BIN:?set PAIR_V5_SOURCE_BOUND_PYTHON_BIN}"

fail() { echo "[pair-v5-source-bound-v1] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive rollout_spec evaluator_spec rollout_output visual_checkpoint visual_checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 differs"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision differs"
[[ "${rollout_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "rollout spec SHA-256 differs"
[[ "${evaluator_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "evaluator spec SHA-256 differs"
[[ "${visual_checkpoint_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "visual checkpoint manifest SHA-256 differs"

source_archive="$(realpath -e -- "${source_archive}")"
rollout_spec="$(realpath -e -- "${rollout_spec}")"
evaluator_spec="$(realpath -e -- "${evaluator_spec}")"
rollout_output="$(realpath -e -- "${rollout_output}")"
visual_checkpoint="$(realpath -e -- "${visual_checkpoint}")"
visual_checkpoint_manifest="$(realpath -e -- "${visual_checkpoint_manifest}")"
python_bin_parent="$(dirname -- "${python_bin}")"
python_bin_name="$(basename -- "${python_bin}")"
[[ "$(realpath -e -- "${python_bin_parent}")/${python_bin_name}" == "${python_bin}" ]] \
  || fail "Python invocation path must have a canonical parent"
python_bin_target="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive differs"
[[ -f "${rollout_spec}" && ! -L "${rollout_spec}" ]] || fail "rollout spec differs"
[[ -f "${evaluator_spec}" && ! -L "${evaluator_spec}" ]] || fail "evaluator spec differs"
[[ -d "${rollout_output}" && ! -L "${rollout_output}" ]] || fail "rollout output differs"
[[ -d "${visual_checkpoint}" && ! -L "${visual_checkpoint}" ]] || fail "visual checkpoint differs"
[[ -f "${visual_checkpoint_manifest}" && ! -L "${visual_checkpoint_manifest}" ]] || fail "visual checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python invocation differs"
[[ -x "${python_bin_target}" && -f "${python_bin_target}" && ! -L "${python_bin_target}" ]] \
  || fail "Python executable target differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${rollout_spec}" | awk '{print $1}')" == "${rollout_spec_sha256}" ]] || fail "rollout spec hash differs"
[[ "$(sha256sum "${evaluator_spec}" | awk '{print $1}')" == "${evaluator_spec_sha256}" ]] || fail "evaluator spec hash differs"
[[ "$(sha256sum "${visual_checkpoint_manifest}" | awk '{print $1}')" == "${visual_checkpoint_manifest_sha256}" ]] || fail "visual checkpoint manifest hash differs"

[[ "${output_dir}" != / ]] || fail "output cannot be root"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output must be canonical"
[[ "${output_dir##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename differs"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing output reuse"
mkdir -p -- "$(dirname -- "${output_dir}")"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-source-bound-v1-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-source-bound-v1-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-source-bound-v1] PASS topology=2xrank4 candidates=8 raw_evidence=true absolute_thresholds=false output=${output_dir}"
  else
    echo "[pair-v5-source-bound-v1] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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

mkdir -p -- "${task_scratch}/source-tree" "${task_scratch}/groups"
archive_copy="${task_scratch}/method-source.tar"
cp -- "${source_archive}" "${archive_copy}"
chmod 0444 -- "${archive_copy}"

"${python_bin}" -B - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys, tarfile
required = {
 "methods/bernini_action_editing/pair_v5_source_bound_preservation_evaluator_v1.py",
 "methods/bernini_action_editing/score_pair_v5_source_bound_preservation_v1.py",
 "methods/bernini_action_editing/author_pair_v5_source_bound_preservation_spec_v1.py",
 "methods/bernini_action_editing/scripts/auh_score_pair_v5_source_bound_preservation_v1_dual4.sbatch",
 "methods/bernini_action_editing/tests/test_pair_v5_source_bound_preservation_evaluator_v1.py",
 "methods/bernini_action_editing/tests/test_score_pair_v5_source_bound_preservation_v1_launcher.py",
}
seen=set()
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("source archive member escaped or is not a plain tree entry")
  seen.add(path.as_posix().lstrip("./"))
missing=sorted(required-seen)
if missing: raise SystemExit(f"archive lacks source-bound evaluator closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +

# Validate the evaluator spec against the archived implementation, exact
# rollout order, runtime versions, method archive, and checkpoint manifest
# before creating any durable output.
"${python_bin}" -B - \
  "${method_root}" "${rollout_spec}" "${rollout_spec_sha256}" \
  "${evaluator_spec}" "${evaluator_spec_sha256}" \
  "${source_revision}" "${source_archive_sha256}" \
  "${visual_checkpoint_manifest_sha256}" <<'PY'
from pathlib import Path
import json, sys
sys.path.insert(0, sys.argv[1])
import pair_v5_source_bound_preservation_evaluator_v1 as contract
import score_pair_v5_source_bound_preservation_v1 as scorer
rollout, rollout_sha = contract.load_current_family_rollout_spec(sys.argv[2], sys.argv[3])
spec, _ = contract.load_evaluator_spec(
    sys.argv[4], sys.argv[5], normalized_rollout=rollout,
    rollout_spec_raw_sha256=rollout_sha,
    implementation_path=Path(sys.argv[1]) / "score_pair_v5_source_bound_preservation_v1.py",
    contract_path=Path(sys.argv[1]) / "pair_v5_source_bound_preservation_evaluator_v1.py",
)
if spec["method_source_revision"] != sys.argv[6] or spec["method_source_archive_sha256"] != sys.argv[7]:
    raise SystemExit("evaluator method-source binding differs")
if spec["model"]["checkpoint_manifest_sha256"] != sys.argv[8]:
    raise SystemExit("evaluator checkpoint-manifest binding differs")
observed_runtime = scorer.runtime_versions()
if observed_runtime != spec["runtime_versions"]:
    print(
        json.dumps(
            {"expected_runtime": spec["runtime_versions"], "observed_runtime": observed_runtime},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    raise SystemExit("evaluator runtime-version binding differs")
if len(spec["candidate_order"]) != 8 or len(spec["source_order"]) != 4:
    raise SystemExit("evaluator current-family population differs")
print("PAIR_V5_SOURCE_BOUND_PREFLIGHT_OK candidates=8 sources=4 thresholds=none")
PY

mkdir -- "${output_dir}"
echo "[pair-v5-source-bound-v1] semantic_media=candidate,correct_source,wrong_source target=false proposal=false donor=false mask=false flow=false pose=false track=false trajectory=false"
echo "[pair-v5-source-bound-v1] evidence=raw_dino_global+dense_fixed_grid+spatial_layout_viewpoint_proxy+temporal_translation_diagnostic+quality_diagnostic ref_off=not_applicable"

job_mod=$((SLURM_JOB_ID % 10000))
sp4_a_port=$((38000 + 2 * job_mod))
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
    "${method_root}/score_pair_v5_source_bound_preservation_v1.py" \
    --root-spec "${rollout_spec}" --expected-root-spec-sha256 "${rollout_spec_sha256}" \
    --evaluator-spec "${evaluator_spec}" --expected-evaluator-spec-sha256 "${evaluator_spec_sha256}" \
    --rollout-output-dir "${rollout_output}" --group-id "${group_id}" \
    --checkpoint "${visual_checkpoint}" --checkpoint-content-manifest "${visual_checkpoint_manifest}" \
    --output-dir "${output_dir}/${group_id}" --method-source-revision "${source_revision}" \
    --method-source-archive-sha256 "${source_archive_sha256}"
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more rank4 evaluator groups failed"
fi

"${python_bin}" -B - \
  "${method_root}" "${rollout_spec}" "${rollout_spec_sha256}" \
  "${evaluator_spec}" "${evaluator_spec_sha256}" "${output_dir}" <<'PY'
from pathlib import Path
import hashlib,json,os,sys
sys.path.insert(0, sys.argv[1])
import pair_v5_source_bound_preservation_evaluator_v1 as contract

rollout, rollout_sha = contract.load_current_family_rollout_spec(sys.argv[2], sys.argv[3])
spec, _ = contract.load_evaluator_spec(
    sys.argv[4], sys.argv[5], normalized_rollout=rollout,
    rollout_spec_raw_sha256=rollout_sha,
    implementation_path=Path(sys.argv[1]) / "score_pair_v5_source_bound_preservation_v1.py",
    contract_path=Path(sys.argv[1]) / "pair_v5_source_bound_preservation_evaluator_v1.py",
)
root=Path(sys.argv[6]); seen=[]; eligible=0
groups=[]; group_file_hashes={}; candidates=[]; candidate_file_hashes={}
for group_id in contract.EXPECTED_GROUPS:
 group_root=root/group_id
 group_path=group_root/f"pair-v5-source-bound-preservation-{group_id}-v1.json"
 group_raw=group_path.read_bytes(); group_value=json.loads(group_raw)
 group=contract.validate_group_receipt(
  group_value,evaluator_spec=spec,evaluator_spec_raw_sha256=sys.argv[5]
 )
 wanted=[cid for cid in spec["candidate_order"] if spec["candidate_group_by_id"][cid]==group_id]
 if group["candidate_order"]!=wanted or group["candidate_count"]!=4:
  raise SystemExit(f"group candidate closure differs: {group_id}")
 groups.append(group)
 group_file_hashes[group_id]=hashlib.sha256(group_raw).hexdigest()
 group_eligible=0
 for cid in wanted:
  path=group_root/f"{cid}.json"
  raw=path.read_bytes(); value=json.loads(raw)
  row=contract.validate_candidate_receipt(
   value,evaluator_spec=spec,evaluator_spec_raw_sha256=sys.argv[5]
  )
  if hashlib.sha256(raw).hexdigest()!=group["candidate_receipt_file_sha256_by_id"][cid]:
   raise SystemExit(f"candidate receipt file hash differs: {cid}")
  if row["receipt_digest"]!=group["candidate_receipt_digest_by_id"][cid]:
   raise SystemExit(f"candidate receipt object digest differs: {cid}")
  if row["candidate_id"]!=cid or not row["evidence_valid"] or row["absolute_source_preservation_pass_claim"]:
   raise SystemExit(f"candidate evidence state differs: {cid}")
  eligible += int(row["eligible_for_downstream_calibration"])
  group_eligible += int(row["eligible_for_downstream_calibration"])
  seen.append(cid)
  candidates.append(row)
  candidate_file_hashes[cid]=hashlib.sha256(raw).hexdigest()
 if group["eligible_for_downstream_calibration_count"]!=group_eligible:
  raise SystemExit(f"group eligible count differs: {group_id}")
if seen!=spec["candidate_order"] or len(seen)!=8:
 raise SystemExit("strong audit candidate order/coverage differs")
if eligible != 8:
 raise SystemExit("all eight provenance-valid candidates must remain eligible")
topology={
 "group_world_size":4,"group_ulysses_size":4,
 "groups":{gid:contract.EXPECTED_GROUP_GPUS[gid] for gid in contract.EXPECTED_GROUPS},
 "total_physical_gpus":8,"concurrent_disjoint_groups":True,
}
completion=contract.make_root_receipt(
 evaluator_spec=spec,evaluator_spec_raw_sha256=sys.argv[5],
 group_receipts=groups,group_receipt_file_sha256_by_id=group_file_hashes,
 candidate_receipts=candidates,candidate_receipt_file_sha256_by_id=candidate_file_hashes,
 topology=topology,
)
completion_path=root/"pair-v5-source-bound-preservation-root-v1.json"
raw=contract.canonical_json_bytes(completion)+b"\n"
temporary_path=root/f".pair-v5-source-bound-preservation-root-v1.{os.getpid()}.tmp"
with temporary_path.open("xb") as handle:
 handle.write(raw); handle.flush(); os.fsync(handle.fileno())
os.chmod(temporary_path,0o400)
# Publishing a fully fsynced inode with link(2) is atomic and refuses an
# existing final name; unlike replacing the final file, it cannot overwrite.
os.link(temporary_path,completion_path)
os.unlink(temporary_path)
directory_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
try: os.fsync(directory_fd)
finally: os.close(directory_fd)
sealed=json.loads(completion_path.read_bytes())
contract.validate_root_receipt(
 sealed,evaluator_spec=spec,evaluator_spec_raw_sha256=sys.argv[5]
)
if completion_path.read_bytes()!=raw:
 raise SystemExit("root completion receipt changed after durable write")
print(f"PAIR_V5_SOURCE_BOUND_STRONG_AUDIT_OK candidates=8 eligible_for_calibration={eligible} root_receipt={completion_path} absolute_pass_claims=0")
PY
