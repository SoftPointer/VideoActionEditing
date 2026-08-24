#!/usr/bin/env bash
#SBATCH --job-name=pair5-rv2v4-2x4
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

# A sealed JSON spec assigns candidates to two independent SP4 groups.  Each
# group loads frozen Bernini once per candidate (serial within group); the two
# groups run concurrently.  This intentionally favors auditability over model
# load amortization for the first PAIR-v5 rollout bank.

source_archive="${PAIR_V5_SOURCE_ARCHIVE:?set PAIR_V5_SOURCE_ARCHIVE}"
source_archive_sha256="${PAIR_V5_SOURCE_ARCHIVE_SHA256:?set PAIR_V5_SOURCE_ARCHIVE_SHA256}"
source_revision="${PAIR_V5_SOURCE_REVISION:?set PAIR_V5_SOURCE_REVISION}"
rollout_spec="${PAIR_V5_ROLLOUT_SPEC:?set PAIR_V5_ROLLOUT_SPEC}"
rollout_spec_sha256="${PAIR_V5_ROLLOUT_SPEC_SHA256:?set PAIR_V5_ROLLOUT_SPEC_SHA256}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:?set BERNINI_CHECKPOINT_CONTENT_MANIFEST}"
output_dir="${PAIR_V5_ROLLOUT_OUTPUT_DIR:?set PAIR_V5_ROLLOUT_OUTPUT_DIR}"
python_bin="${PAIR_V5_PYTHON_BIN:?set PAIR_V5_PYTHON_BIN}"

fail() { echo "[pair-v5-rv2v4-dual4] ERROR: $*" >&2; exit 2; }
require_absolute() { [[ "$1" == /* ]] || fail "$2 must be absolute"; }

for name in source_archive rollout_spec bernini_root veomni_root checkpoint checkpoint_manifest output_dir python_bin; do
  require_absolute "${!name}" "${name}"
done
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "source archive SHA-256 is invalid"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "source revision is invalid"
[[ "${rollout_spec_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "rollout spec SHA-256 is invalid"

source_archive="$(realpath -e -- "${source_archive}")"
rollout_spec="$(realpath -e -- "${rollout_spec}")"
bernini_root="$(realpath -e -- "${bernini_root}")"
veomni_root="$(realpath -e -- "${veomni_root}")"
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_manifest="$(realpath -e -- "${checkpoint_manifest}")"
python_bin="$(realpath -e -- "${python_bin}")"

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "source archive is not a plain file"
[[ -f "${rollout_spec}" && ! -L "${rollout_spec}" ]] || fail "rollout spec is not a plain file"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "Bernini root differs"
[[ -d "${veomni_root}" && ! -L "${veomni_root}" ]] || fail "VeOmni root differs"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "checkpoint root differs"
[[ -f "${checkpoint_manifest}" && ! -L "${checkpoint_manifest}" ]] || fail "checkpoint manifest differs"
[[ -x "${python_bin}" && -f "${python_bin}" ]] || fail "Python executable differs"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive hash differs"
[[ "$(git get-tar-commit-id <"${source_archive}")" == "${source_revision}" ]] || fail "source archive revision differs"
[[ "$(sha256sum "${rollout_spec}" | awk '{print $1}')" == "${rollout_spec_sha256}" ]] || fail "sealed rollout spec hash differs"

[[ "${output_dir}" != / ]] || fail "output directory cannot be root"
output_basename="${output_dir##*/}"
[[ "${output_basename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "output basename is unsafe"
[[ "$(realpath -m -- "${output_dir}")" == "${output_dir}" ]] || fail "output directory must be canonical"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse rollout output directory"
output_parent="$(dirname -- "${output_dir}")"
mkdir -p -- "${output_parent}"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false MODELING_BACKEND=hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

[[ "${SLURM_JOB_ID:?launcher requires SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
task_scratch="$(mktemp -d "${scratch_parent%/}/pair-v5-rv2v4-${SLURM_JOB_ID}.XXXXXX")"
sp4_a_pid=""; sp4_b_pid=""

cleanup() {
  local status=$? cleanup_failed=0
  trap - EXIT TERM INT
  case "${task_scratch:-}" in "${scratch_parent%/}/pair-v5-rv2v4-${SLURM_JOB_ID}."*) ;; *) cleanup_failed=1 ;; esac
  if [[ "${cleanup_failed}" == 0 && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w -- "${task_scratch}" || cleanup_failed=1
    rm -rf -- "${task_scratch}" || cleanup_failed=1
  fi
  [[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]] || cleanup_failed=1
  [[ "${cleanup_failed}" == 0 || "${status}" != 0 ]] || status=2
  if [[ "${status}" == 0 ]]; then
    echo "[pair-v5-rv2v4-dual4] PASS topology=two-independent-SP4 mode=rv2v4 frames=81 refs=0,27,53,80 steps=40 output=${output_dir}"
  else
    echo "[pair-v5-rv2v4-dual4] EXIT status=${status} cleanup_verified=$([[ "${cleanup_failed}" == 0 ]] && echo true || echo false)" >&2
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
 "methods/bernini_action_editing/pair_v5_native_rollout_spec.py",
 "methods/bernini_action_editing/infer_pair_v5_native_rv2v4_rollout.py",
 "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
 "methods/bernini_action_editing/infer_lora.py",
 "methods/bernini_action_editing/train_lora.py",
 "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
 "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
 "methods/bernini_action_editing/source_kv_replay.py",
 "methods/bernini_action_editing/source_kv_route_batches.py",
 "methods/bernini_action_editing/source_value_residual.py",
 "methods/bernini_action_editing/tools/materialize_vae.py",
}
seen=set()
with tarfile.open(sys.argv[1], "r:*") as handle:
 for member in handle.getmembers():
  path=PurePosixPath(member.name)
  if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
   raise SystemExit("archive member escaped or is not a plain tree entry")
  seen.add(path.as_posix().lstrip("./"))
missing=sorted(required-seen)
if missing: raise SystemExit(f"archive lacks PAIR-v5 rollout closure: {missing}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source-tree"
method_root="${task_scratch}/source-tree/methods/bernini_action_editing"
find "${method_root}" -type f -exec chmod a-w -- {} +
plan_root="${task_scratch}/plan"
"${python_bin}" -B "${method_root}/pair_v5_native_rollout_spec.py" \
  --spec "${rollout_spec}" --expected-sha256 "${rollout_spec_sha256}" --output-dir "${plan_root}" \
  >"${task_scratch}/plan-materialization.json"

mkdir -- "${output_dir}"
echo "[pair-v5-rv2v4-dual4] semantic_inputs=source_video,complete_caption target=false t2v_proposal=false donor=false external_reference=false mask=false flow=false pose=false track=false trajectory=false"
echo "[pair-v5-rv2v4-dual4] allocation=8xMI210 topology=sp4-a:0,1,2,3+sp4-b:4,5,6,7 exact81=true steps=40 seed_source=sealed_spec guidance_source=sealed_spec"

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
  local candidate_spec candidate_id candidate_output
  while IFS= read -r candidate_spec; do
    [[ -f "${candidate_spec}" && ! -L "${candidate_spec}" ]] || fail "candidate envelope differs"
    candidate_id="$("${python_bin}" -B - "${candidate_spec}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["candidate"]["candidate_id"])
PY
)"
    [[ "${candidate_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail "candidate ID differs"
    candidate_output="${output_dir}/${candidate_id}"
    [[ ! -e "${candidate_output}" && ! -L "${candidate_output}" ]] || fail "refusing candidate output reuse"
    echo "[pair-v5-rv2v4-dual4:${group_id}] START candidate=${candidate_id} visible=${visible_gpus}"
    "${python_bin}" -B -m torch.distributed.run --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="${master_port}" \
      "${method_root}/infer_pair_v5_native_rv2v4_rollout.py" \
      --candidate-spec "${candidate_spec}" --expected-root-spec-sha256 "${rollout_spec_sha256}" \
      --output-dir "${candidate_output}" --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
      --checkpoint "${checkpoint}" --checkpoint-content-manifest "${checkpoint_manifest}" \
      --method-source-revision "${source_revision}" --method-source-archive-sha256 "${source_archive_sha256}"
  done < <(find "${plan_root}/${group_id}" -mindepth 1 -maxdepth 1 -type f -name '*.json' -print | sort)
)

run_group sp4-a "0,1,2,3" "${sp4_a_port}" >"${output_dir}/sp4-a.log" 2>&1 & sp4_a_pid=$!
run_group sp4-b "4,5,6,7" "${sp4_b_port}" >"${output_dir}/sp4-b.log" 2>&1 & sp4_b_pid=$!
sp4_a_status=0; sp4_b_status=0
wait "${sp4_a_pid}" || sp4_a_status=$?
wait "${sp4_b_pid}" || sp4_b_status=$?
if [[ "${sp4_a_status}" != 0 || "${sp4_b_status}" != 0 ]]; then
  tail -n 160 -- "${output_dir}/sp4-a.log" >&2 || true
  tail -n 160 -- "${output_dir}/sp4-b.log" >&2 || true
  fail "one or more SP4 rollout groups failed"
fi

"${python_bin}" -B - "${rollout_spec}" "${rollout_spec_sha256}" "${output_dir}" "${method_root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
import pair_v5_native_rollout_spec as contract
spec,digest=contract.load_sealed_spec(sys.argv[1],sys.argv[2])
root=Path(sys.argv[3]); expected={c["candidate_id"] for g in spec["groups"] for c in g["candidates"]}
seen=set()
for cid in sorted(expected):
 p=root/cid/"pair-v5-rollout-receipt.json"
 value=json.loads(p.read_bytes()); declared=value.pop("receipt_digest")
 if hashlib.sha256(contract.canonical_json_bytes(value)).hexdigest()!=declared: raise SystemExit(f"PAIR receipt digest differs: {cid}")
 value["receipt_digest"]=declared
 if value["root_spec_raw_sha256"]!=digest: raise SystemExit(f"root binding differs: {cid}")
 if value["sampling_contract"]!=contract.SAMPLING_CONTRACT: raise SystemExit(f"sampling differs: {cid}")
 if value["semantic_input_closure"]!=contract.SEMANTIC_INPUT_CLOSURE: raise SystemExit(f"input closure differs: {cid}")
 topology=value.get("runtime_topology",{})
 wanted_visible=",".join(str(x) for x in value["visible_gpus"])
 if topology!={"world_size":4,"ulysses_size":4,"rocr_visible_devices":wanted_visible}: raise SystemExit(f"runtime SP4 binding differs: {cid}")
 for artifact in (value["artifacts"]["mp4"],value["artifacts"]["predecode_clean_latent"],value["artifacts"]["official_initial_gaussian"]):
  q=Path(artifact["path"])
  if not q.is_file() or q.is_symlink() or hashlib.sha256(q.read_bytes()).hexdigest()!=artifact["sha256"]: raise SystemExit(f"artifact differs: {cid}")
 seen.add(cid)
if seen!=expected: raise SystemExit("candidate coverage differs")
print("PAIR_V5_NATIVE_RV2V4_DUAL4_STRONG_AUDIT_OK",len(seen))
PY
