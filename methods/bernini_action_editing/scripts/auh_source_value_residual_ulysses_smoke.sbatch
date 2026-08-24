#!/usr/bin/env bash
#SBATCH --job-name=bern-v10-ulysses
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

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
receipt_output="${BERNINI_V10_SMOKE_RECEIPT:?set BERNINI_V10_SMOKE_RECEIPT}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_revision="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_revision="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

fail() {
  echo "[bernini-v10-ulysses] ERROR: $*" >&2
  exit 2
}

[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive SHA256"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method revision"
[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "method archive is not a plain file"
[[ -d "${source_repository}/.git" ]] || fail "method source repository is not a Git checkout"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini snapshot root"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root is not a Git checkout"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ "${receipt_output}" == /*.json ]] || fail "receipt output must be an absolute JSON path"
[[ ! -e "${receipt_output}" && ! -L "${receipt_output}" ]] || fail "refusing to overwrite smoke receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive SHA256 differs"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_revision}" ]] || fail "VeOmni revision differs"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

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
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-v10-ulysses-${SLURM_JOB_ID}.XXXXXX")"
cleanup_task_scratch() {
  local expected_prefix="${scratch_parent%/}/bernini-v10-ulysses-${SLURM_JOB_ID}."
  if [[ -n "${task_scratch:-}" && "${task_scratch}" == "${expected_prefix}"* && -d "${task_scratch}" && ! -L "${task_scratch}" ]]; then
    # The method tree is deliberately mode-read-only while Python runs.  Restore
    # owner write bits only inside this validated private scratch root so the
    # EXIT trap can remove it after success, failure, TERM, or INT.
    chmod -R u+w "${task_scratch}" 2>/dev/null || true
    rm -rf -- "${task_scratch}"
  fi
}
trap cleanup_task_scratch EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
source_root="${task_scratch}/source"
run_log="${task_scratch}/torchrun.log"
mkdir -p \
  "${source_root}" \
  "${task_scratch}/pycache-prefix" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${receipt_output%/*}"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache-prefix"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"
export TMPDIR="${task_scratch}"

cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private archive copy differs"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method revision is not a commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "archive bytes do not bind the declared revision"

"${python_bin}" - "${archive_copy}" "${source_revision}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

archive, revision = sys.argv[1:]
with tarfile.open(archive, mode="r:*") as handle:
    if handle.pax_headers.get("comment") != revision:
        raise SystemExit("git-archive embedded revision differs")
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY

tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${source_root}"
method_root="${source_root}/methods/bernini_action_editing"
for required in \
  source_value_residual_ulysses_smoke.py \
  source_value_residual.py \
  source_kv_replay.py
do
  [[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]] || fail "archive lacks plain ${required}"
done
[[ -z "$(find "${source_root}" -type l -print -quit)" ]] || fail "extracted method contains a symlink"
[[ -z "$(find "${source_root}" -type d -name __pycache__ -print -quit)" ]] || fail "extracted method contains __pycache__"
[[ -z "$(find "${source_root}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print -quit)" ]] || fail "extracted method contains bytecode"

# The smoke itself verifies that the three executed source files and archive
# cannot change during execution.  Make both the private archive and extracted
# tree mode-read-only before Python starts.
chmod -R a-w "${source_root}"
chmod a-w "${archive_copy}"
[[ -z "$(find "${source_root}" -perm /222 -print -quit)" ]] || fail "private method tree remains mode-writable"
[[ ! -w "${archive_copy}" ]] || fail "private archive remains mode-writable"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-v10-ulysses] revision=${source_revision} archive_sha256=${source_archive_sha256}"
echo "[bernini-v10-ulysses] Bernini_snapshot=${bernini_revision} VeOmni=${veomni_revision}"
echo "[bernini-v10-ulysses] 4xMI210 FA2 Ulysses=4 fixed_gate=0.25 video_claim=false"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/source_value_residual_ulysses_smoke.py" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --expected-bernini-commit "${bernini_revision}" \
  --expected-veomni-commit "${veomni_revision}" \
  --method-revision "${source_revision}" \
  --method-archive "${archive_copy}" \
  --expected-method-archive-sha256 "${source_archive_sha256}" \
  --gate 0.25 | tee "${run_log}"

"${python_bin}" - "${run_log}" "${receipt_output}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

log_path, output_value = sys.argv[1:]
output = Path(output_value)
matches = []
with open(log_path, encoding="utf-8", errors="strict") as handle:
    for line in handle:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == "bernini-counterfactual-source-value-residual-v10-ulysses-smoke-v2":
            matches.append(value)
if len(matches) != 1:
    raise SystemExit(f"expected one canonical smoke receipt, found {len(matches)}")
receipt = matches[0]
stored = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
if stored != hashlib.sha256(payload).hexdigest():
    raise SystemExit("smoke receipt digest differs")
receipt["receipt_digest"] = stored
encoded = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
except BaseException:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    raise
PY

[[ -f "${receipt_output}" && ! -L "${receipt_output}" ]] || fail "smoke receipt is missing"
receipt_sha_round1="$(sha256sum "${receipt_output}" | awk '{print $1}')"
"${python_bin}" - "${receipt_output}" "${source_revision}" "${source_archive_sha256}" <<'PY'
import hashlib
import json
import sys

path, revision, archive_sha = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    receipt = json.load(handle)
runtime = receipt.get("runtime", {})
manifest = receipt.get("source_manifest", {})
certificate = receipt.get("source_manifest_certificate", {})
provenance = receipt.get("method_provenance", {})
verified = receipt.get("verified_claims", {})
bytecode = manifest.get("bytecode_policy", {})
sequence = runtime.get("input_sequence_order", {})
qkv = runtime.get("projected_qkv_head_order", {})
reference = runtime.get("direct_reference", {})
checks = (
    receipt.get("schema_version") == "bernini-counterfactual-source-value-residual-v10-ulysses-smoke-v2",
    receipt.get("scientific_video_claim") is False,
    receipt.get("learnable_gate_claim") is False,
    receipt.get("fixed_gate_oracle_core_only") is True,
    receipt.get("four_rank_engineering_certificate") is True,
    receipt.get("method_revision") == revision,
    receipt.get("method_archive_sha256") == archive_sha,
    provenance.get("revision") == revision,
    provenance.get("archive_sha256") == archive_sha,
    provenance.get("archive_member_sha256") == provenance.get("runtime_source_sha256"),
    receipt.get("bernini_git_checkout_claimed") is False,
    manifest.get("method", {}).get("revision") == revision,
    manifest.get("method", {}).get("archive_sha256") == archive_sha,
    set(manifest.get("method", {}).get("archive_member_sha256", {})) == {"smoke", "v10_core", "v9_runtime"},
    manifest.get("bernini_snapshot", {}).get("git_checkout_claimed") is False,
    len(manifest.get("bernini_snapshot", {}).get("files", {})) == 4,
    manifest.get("veomni", {}).get("tracked_worktree_clean") is True,
    len(manifest.get("veomni", {}).get("files", {})) == 7,
    isinstance(manifest.get("diffusers", {}).get("attention_source_sha256"), str),
    bytecode.get("dont_write_bytecode") is True,
    bytecode.get("absolute_environment_runtime_exact") is True,
    bytecode.get("outside_audited_source_roots") is True,
    bytecode.get("shared_source_pycache_importable") is False,
    bytecode.get("method_root_pycache_absent") is True,
    certificate.get("pre_post_exact") is True,
    certificate.get("bytecode_policy_pre_post_exact") is True,
    certificate.get("pre", {}).get("all_four_ranks_exact") is True,
    certificate.get("post", {}).get("all_four_ranks_exact") is True,
    verified and all(value is True for value in verified.values()),
    runtime.get("world_size") == 4,
    runtime.get("ulysses_size") == 4,
    runtime.get("attention_backend") == "fa2",
    runtime.get("heads") == 12,
    runtime.get("heads_per_rank") == 3,
    runtime.get("full_source_tokens") == 16,
    runtime.get("full_pair_tokens") == 32,
    runtime.get("main_operator") == "full_k_value",
    runtime.get("main_gate") == 0.25,
    sequence.get("carrier", {}).get("exact") is True,
    sequence.get("pair", {}).get("exact") is True,
    all(qkv.get(span, {}).get(name, {}).get("allclose") is True for span in ("source", "pair") for name in ("query", "key", "value")),
    qkv.get("cached_source_value", {}).get("allclose") is True,
    reference.get("residual_output", {}).get("allclose") is True,
    reference.get("residual_input_gradient", {}).get("allclose") is True,
    reference.get("zero_output", {}).get("allclose") is True,
    reference.get("zero_input_gradient", {}).get("allclose") is True,
    reference.get("direct_gate_gradient_finite_nonzero") is True,
    reference.get("residual_specific_gradient_finite_nonzero") is True,
    reference.get("zero_gate_local_official_byte_exact_all_ranks") is True,
    runtime.get("all_rank_residual_digest_equal") is True,
    runtime.get("all_rank_zero_digest_equal") is True,
    runtime.get("all_rank_attention_parameters_exact") is True,
    runtime.get("all_rank_receipt_payload_exact") is True,
    runtime.get("all_rank_pre_inventory_digest_exact") is True,
    runtime.get("all_rank_final_receipt_digest_exact") is True,
    len(runtime.get("rank_summaries", [])) == 4,
)
stored = receipt.pop("receipt_digest", None)
payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
checks += (stored == hashlib.sha256(payload).hexdigest(),)
raise SystemExit(0 if all(checks) else 1)
PY

[[ "$(sha256sum "${receipt_output}" | awk '{print $1}')" == "${receipt_sha_round1}" ]] || fail "receipt changed during audit"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_revision}" ]] || fail "VeOmni revision changed"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source changed"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive changed"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-v10-ulysses] PASS receipt_sha256=${receipt_sha_round1} video_claim=false"
