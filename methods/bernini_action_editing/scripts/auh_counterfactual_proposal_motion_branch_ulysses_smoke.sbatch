#!/usr/bin/env bash
#SBATCH --job-name=bern-v11-cpmr-xsp
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=01:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail
umask 077

source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
receipt_output="${BERNINI_V11_CPMR_SMOKE_RECEIPT:?set BERNINI_V11_CPMR_SMOKE_RECEIPT}"
bernini_root="${BERNINI_OFFICIAL_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591}"
veomni_root="${BERNINI_VEOMNI_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11}"
dog_source="${BERNINI_V11_DOG_SOURCE:-/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/288545b9c031491a/source.mp4}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_revision="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_revision="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
dog_sha256="5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
schema="bernini-cpmr-v11-motion-branch-ulysses-smoke-v1"

fail() {
  echo "[bernini-v11-cpmr-xsp] ERROR: $*" >&2
  exit 2
}

[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ -d "${source_repository}/.git" ]] || fail "source repository is not a Git checkout"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini snapshot root"
[[ -d "${veomni_root}/.git" && ! -L "${veomni_root}" ]] || fail "invalid VeOmni checkout"
[[ -f "${dog_source}" && ! -L "${dog_source}" ]] || fail "invalid dog source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${receipt_output}" == /*.json ]] || fail "receipt must be an absolute JSON path"
[[ ! -e "${receipt_output}" && ! -L "${receipt_output}" ]] || fail "refusing to overwrite receipt"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_revision}" ]] || fail "VeOmni revision differs"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"
[[ "$(sha256sum "${dog_source}" | awk '{print $1}')" == "${dog_sha256}" ]] || fail "dog source SHA256 differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "source revision is not a commit"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-v11-cpmr-xsp-${SLURM_JOB_ID}.XXXXXX")"
cleanup_task_scratch() {
  local expected_prefix="${scratch_parent%/}/bernini-v11-cpmr-xsp-${SLURM_JOB_ID}."
  if [[ -n "${task_scratch:-}" && "${task_scratch}" == "${expected_prefix}"* && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    chmod -R u+w "${task_scratch}" 2>/dev/null || true
    rm -rf -- "${task_scratch}"
  fi
}
trap cleanup_task_scratch EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

method_archive="${task_scratch}/method-${source_revision}.tar"
source_root="${task_scratch}/source"
run_log="${task_scratch}/torchrun.log"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/pycache" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${receipt_output%/*}"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

# Execute only source bytes reconstructed from the declared immutable commit.
git -C "${source_repository}" archive --format=tar \
  -o "${method_archive}" \
  "${source_revision}" \
  methods/bernini_action_editing/counterfactual_proposal_motion_branch.py \
  methods/bernini_action_editing/counterfactual_proposal_motion_branch_ulysses_smoke.py
method_archive_sha256="$(sha256sum "${method_archive}" | awk '{print $1}')"
[[ "${method_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA256"

"${python_bin}" - "${method_archive}" "${source_revision}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive, revision = sys.argv[1:]
required = {
    "methods/bernini_action_editing/counterfactual_proposal_motion_branch.py",
    "methods/bernini_action_editing/counterfactual_proposal_motion_branch_ulysses_smoke.py",
}
with tarfile.open(archive, mode="r:*") as handle:
    if handle.pax_headers.get("comment") != revision:
        raise SystemExit("git-archive embedded revision differs")
    members = handle.getmembers()
    files = set()
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
        if member.isfile():
            files.add(member.name)
if files != required:
    raise SystemExit(f"archive file set differs: {sorted(files)!r}")
PY

tar --no-same-owner --no-same-permissions -xf "${method_archive}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
[[ -f "${method_root}/counterfactual_proposal_motion_branch.py" && ! -L "${method_root}/counterfactual_proposal_motion_branch.py" ]] || fail "archive lacks motion branch"
[[ -f "${method_root}/counterfactual_proposal_motion_branch_ulysses_smoke.py" && ! -L "${method_root}/counterfactual_proposal_motion_branch_ulysses_smoke.py" ]] || fail "archive lacks smoke"
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "extracted source contains symlink"
[[ -z "$(find "${source_root}" -type d -name __pycache__ -print -quit)" ]] || fail "extracted source contains pycache"
[[ -z "$(find "${source_root}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print -quit)" ]] || fail "extracted source contains bytecode"
chmod -R a-w "${source_root}"
chmod a-w "${method_archive}"
[[ -z "$(find "${source_root}" -perm /222 -print -quit)" ]] || fail "private source remains writable"
[[ ! -w "${method_archive}" ]] || fail "private archive remains writable"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-v11-cpmr-xsp] source_revision=${source_revision} archive_sha256=${method_archive_sha256}"
echo "[bernini-v11-cpmr-xsp] Bernini=${bernini_revision} VeOmni=${veomni_revision}"
echo "[bernini-v11-cpmr-xsp] dog=81f global_q=39060 local_q=9765 carrier=1344 heads=12x128 gate=0.10"
echo "[bernini-v11-cpmr-xsp] video_claim=false science_claim=false lora_claim=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/counterfactual_proposal_motion_branch_ulysses_smoke.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --dog-source-video "${dog_source}" \
  --expected-bernini-commit "${bernini_revision}" \
  --expected-veomni-commit "${veomni_revision}" \
  --method-revision "${source_revision}" \
  --method-archive "${method_archive}" \
  --expected-method-archive-sha256 "${method_archive_sha256}" | tee "${run_log}"

# Atomic-at-creation semantics: O_EXCL guarantees an existing receipt is never
# replaced.  A failed write may leave a partial new file, but never overwrites
# prior evidence; reruns must choose a new explicit output path.
"${python_bin}" - "${run_log}" "${receipt_output}" "${schema}" "${source_revision}" "${method_archive_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

log_value, output_value, schema, revision, archive_sha = sys.argv[1:]
matches = []
with open(log_value, encoding="utf-8", errors="strict") as handle:
    for line in handle:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == schema:
            matches.append(value)
if len(matches) != 1:
    raise SystemExit(f"expected one canonical smoke receipt, found {len(matches)}")
receipt = matches[0]
stored = receipt.get("receipt_digest")
payload = dict(receipt)
payload.pop("receipt_digest", None)
encoded_payload = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode()
if stored != hashlib.sha256(encoded_payload).hexdigest():
    raise SystemExit("receipt digest differs")
verified = receipt.get("verified_claims", {})
runtime = receipt.get("runtime", {})
expected_binding = {
    "expected_block_indices": [0],
    "observed_block_indices": [0],
    "completed": True,
    "consumed": True,
    "aborted": False,
    "bound_tensor_released": True,
}
expected_wrapper_stats = {
    "block_index": 0,
    "base_calls": 2,
    "motion_calls": 1,
    "zero_gate_delegations": 1,
    "inactive_delegations": 0,
    "no_context_delegations": 0,
    "branch_delegations": 0,
    "branch_counts": {"final_render:positive": 2},
}
expected_motion_stats = {
    "block_index": 0,
    "motion_calls": 3,
    "project_qkv_calls": 3,
    "varlen_calls": 3,
    "explicit_custom_collective_calls": 0,
    "measured_custom_collective_calls": None,
    "last_metadata": {
        "origin_q": 39060,
        "local_q": 9765,
        "carrier_k": 1344,
        "heads": 12,
        "head_dim": 128,
        "cu_q": [0, 9765],
        "cu_k": [0, 1344],
        "max_q": 9765,
        "max_k": 1344,
        "rank_q_len": 9765,
        "explicit_custom_collectives": 0,
        "measured_custom_collectives": None,
    },
}
checks = (
    receipt.get("method_revision") == revision,
    receipt.get("method_archive_sha256") == archive_sha,
    receipt.get("scientific_claim") is False,
    receipt.get("video_claim") is False,
    receipt.get("video_quality_claim") is False,
    receipt.get("training_claim") is False,
    receipt.get("lora_claim") is False,
    receipt.get("full_transformer_forward_claim") is False,
    receipt.get("gradient_checkpoint_training_claim") is False,
    receipt.get("frozen_single_block_engineering_only") is True,
    verified and all(item is True for item in verified.values()),
    runtime.get("world_size") == 4,
    runtime.get("ulysses_size") == 4,
    runtime.get("attention_backend") == "fa2",
    runtime.get("global_q") == 39060,
    runtime.get("local_q") == 9765,
    runtime.get("carrier_kv") == 1344,
    runtime.get("heads") == 12,
    runtime.get("head_dim") == 128,
    runtime.get("active_gate") == 0.10,
    runtime.get("a2a_proxy_calls") == {
        "gather_heads_scatter_seq": 0,
        "gather_seq_scatter_heads": 0,
    },
    runtime.get("reference_only_dist_all_gather_calls_per_rank") == 2,
    runtime.get("gathered_reference_parity", {}).get("allclose") is True,
    runtime.get("zero_gate_conditioned_encoder_binding") == expected_binding,
    runtime.get("active_conditioned_encoder_binding") == expected_binding,
    runtime.get("wrapper_statistics") == expected_wrapper_stats,
    runtime.get("motion_statistics") == expected_motion_stats,
    len(runtime.get("rank_summaries", [])) == 4,
)
if not all(checks):
    raise SystemExit("smoke receipt contract validation failed")
output = Path(output_value)
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(output, 0o400)
PY

[[ -f "${receipt_output}" && ! -L "${receipt_output}" ]] || fail "receipt is missing"
[[ ! -w "${receipt_output}" ]] || fail "receipt is unexpectedly writable"
echo "[bernini-v11-cpmr-xsp] receipt=${receipt_output}"
echo "[bernini-v11-cpmr-xsp] receipt_sha256=$(sha256sum "${receipt_output}" | awk '{print $1}')"
