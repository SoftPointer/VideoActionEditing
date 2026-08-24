#!/usr/bin/env bash
#SBATCH --job-name=bern-full644-dsa
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=12:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:?set BERNINI_ACTION_SOURCE_ARCHIVE}"
source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:?set BERNINI_ACTION_SOURCE_ARCHIVE_SHA256}"
source_revision="${BERNINI_ACTION_SOURCE_REVISION:?set BERNINI_ACTION_SOURCE_REVISION}"
source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:?set BERNINI_ACTION_SOURCE_REPOSITORY}"
bernini_root="${BERNINI_OFFICIAL_ROOT:?set BERNINI_OFFICIAL_ROOT}"
bernini_archive="${BERNINI_OFFICIAL_ARCHIVE:?set BERNINI_OFFICIAL_ARCHIVE}"
veomni_root="${BERNINI_VEOMNI_ROOT:?set BERNINI_VEOMNI_ROOT}"
checkpoint="${BERNINI_ACTION_CHECKPOINT:?set BERNINI_ACTION_CHECKPOINT}"
adapter_checkpoint="${BERNINI_ACTION_ADAPTER_CHECKPOINT:?set BERNINI_ACTION_ADAPTER_CHECKPOINT}"
source_video="${BERNINI_ACTION_SOURCE_VIDEO:?set BERNINI_ACTION_SOURCE_VIDEO}"
instruction="${BERNINI_ACTION_INSTRUCTION:?set BERNINI_ACTION_INSTRUCTION}"
output_video="${BERNINI_ACTION_INFERENCE_OUTPUT:?set BERNINI_ACTION_INFERENCE_OUTPUT}"
block_scope="${BERNINI_DIRECTED_ATTN_BLOCKS:?set BERNINI_DIRECTED_ATTN_BLOCKS to mid, late, or all}"
num_steps="${BERNINI_ACTION_INFERENCE_STEPS:-40}"
seed="${BERNINI_ACTION_INFERENCE_SEED:-2027}"
python_bin="${BERNINI_ACTION_PYTHON:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"
veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
bernini_archive_sha256="08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670"
full644_adapter_sha256="9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"

fail() {
  echo "[bernini-full644-directed] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid method source archive"
[[ -d "${source_repository}/.git" ]] || fail "invalid method source repository"
[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]] || fail "invalid Bernini source root"
[[ -f "${bernini_archive}" && ! -L "${bernini_archive}" ]] || fail "invalid Bernini source archive"
[[ -d "${veomni_root}/.git" ]] || fail "VeOmni root must be a Git checkout"
[[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] || fail "invalid base checkpoint"
[[ -d "${adapter_checkpoint}" && ! -L "${adapter_checkpoint}" ]] || fail "invalid full644 adapter checkpoint"
[[ -f "${adapter_checkpoint}/adapter/adapter_config.json" && ! -L "${adapter_checkpoint}/adapter/adapter_config.json" ]] || fail "adapter config missing"
[[ -f "${adapter_checkpoint}/adapter/adapter_model.safetensors" && ! -L "${adapter_checkpoint}/adapter/adapter_model.safetensors" ]] || fail "adapter weights missing"
[[ -f "${adapter_checkpoint}/receipt.json" && ! -L "${adapter_checkpoint}/receipt.json" ]] || fail "training receipt missing"
[[ -f "${source_video}" && ! -L "${source_video}" ]] || fail "invalid source video"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -n "${instruction//[[:space:]]/}" ]] || fail "instruction is empty"
[[ "${block_scope}" == mid || "${block_scope}" == late || "${block_scope}" == all ]] || fail "block scope must be mid, late, or all"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid method archive hash"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid method source revision"
[[ "${num_steps}" == 40 ]] || fail "the oracle probe is fixed to 40 inference steps"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be non-negative"
[[ "${output_video}" == /*.mp4 ]] || fail "output must be an absolute .mp4 path"
[[ ! -e "${output_video}" && ! -L "${output_video}" ]] || fail "refusing to overwrite output"
[[ ! -e "${output_video}.receipt.json" && ! -L "${output_video}.receipt.json" ]] || fail "refusing to overwrite receipt"
[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive hash mismatch"
[[ "$(sha256sum "${bernini_archive}" | awk '{print $1}')" == "${bernini_archive_sha256}" ]] || fail "Bernini archive hash mismatch"
[[ "$(sha256sum "${adapter_checkpoint}/adapter/adapter_model.safetensors" | awk '{print $1}')" == "${full644_adapter_sha256}" ]] || fail "full644 adapter hash mismatch"
[[ "$(git -C "${veomni_root}" rev-parse HEAD)" == "${veomni_commit}" ]] || fail "VeOmni commit mismatch"
[[ -z "$(git -C "${veomni_root}" status --porcelain --untracked-files=no)" ]] || fail "VeOmni tracked source is dirty"

training_source_revision="$(jq -er '.method_source_revision' "${adapter_checkpoint}/receipt.json")" || fail "training source revision missing"
training_source_archive_sha256="$(jq -er '.method_source_archive_sha256' "${adapter_checkpoint}/receipt.json")" || fail "training source archive identity missing"
[[ "${training_source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid training source revision"
[[ "${training_source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid training source archive hash"

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

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-full644-directed-${SLURM_JOB_ID}.XXXXXX")"
archive_copy="${task_scratch}/method.tar"
archive_from_revision="${task_scratch}/method-from-revision.tar"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton" \
  "${output_video%/*}"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

cp -- "${source_archive}" "${archive_copy}"
[[ "$(sha256sum "${archive_copy}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "private method archive hash mismatch"
[[ "$(git -C "${source_repository}" cat-file -t "${source_revision}^{commit}")" == commit ]] || fail "method source revision is not a repository commit"
git -C "${source_repository}" archive --format=tar \
  -o "${archive_from_revision}" \
  "${source_revision}" methods/bernini_action_editing
[[ "$(sha256sum "${archive_from_revision}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "method archive bytes are not the declared Git revision"
"${python_bin}" - "${archive_copy}" <<'PY'
from pathlib import PurePosixPath
import sys
import tarfile

with tarfile.open(sys.argv[1], mode="r:*") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("empty method source archive")
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk() or member.isfifo() or member.isdev():
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name!r}")
PY
tar --no-same-owner --no-same-permissions -xf "${archive_copy}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/infer_lora.py" ]] || fail "archive lacks legacy full644 inference harness"
[[ -f "${method_root}/directed_source_attention.py" ]] || fail "archive lacks directed attention processor"
[[ -f "${method_root}/infer_directed_attn_oracle.py" ]] || fail "archive lacks oracle wrapper"
[[ -f "${method_root}/tests/test_directed_source_attention.py" ]] || fail "archive lacks directed processor tests"
[[ -f "${method_root}/tests/test_infer_directed_attn_oracle.py" ]] || fail "archive lacks oracle tests"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
echo "[bernini-full644-directed] source_revision=${source_revision}"
echo "[bernini-full644-directed] method_archive_sha256=${source_archive_sha256}"
echo "[bernini-full644-directed] blocks=${block_scope} frames=81 steps=${num_steps} seed=${seed}"
echo "[bernini-full644-directed] adapter=${adapter_checkpoint}"
echo "[bernini-full644-directed] adapter_sha256=${full644_adapter_sha256}"
echo "[bernini-full644-directed] training_source_revision=${training_source_revision} training_source_archive_sha256=${training_source_archive_sha256}"
echo "[bernini-full644-directed] source_sha256=$(sha256sum "${source_video}" | awk '{print $1}')"
echo "[bernini-full644-directed] output=${output_video}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

for pattern in test_directed_source_attention.py test_infer_directed_attn_oracle.py test_infer_lora_contract.py
do
  PYTHONPATH="${method_root}" "${python_bin}" -m unittest discover \
    -s "${method_root}/tests" -p "${pattern}" -v
done

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${method_root}/infer_directed_attn_oracle.py" \
  --directed-attn-blocks "${block_scope}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --adapter-checkpoint "${adapter_checkpoint}" \
  --source-video "${source_video}" \
  --instruction "${instruction}" \
  --output "${output_video}" \
  --num-inference-steps "${num_steps}" \
  --seed "${seed}" \
  --expected-bernini-commit "${bernini_commit}" \
  --expected-veomni-commit "${veomni_commit}" \
  --expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}" \
  --method-source-revision "${source_revision}" \
  --method-source-archive-sha256 "${source_archive_sha256}"

receipt_path="${output_video}.receipt.json"
[[ -f "${output_video}" && ! -L "${output_video}" ]] || fail "output video missing"
[[ -f "${receipt_path}" && ! -L "${receipt_path}" ]] || fail "output receipt missing"
"${python_bin}" -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); scope,run_rev,run_arc,train_rev,train_arc=sys.argv[2:]; o=r.get("oracle", {}); a=r.get("adapter", {}); d=o.get("directed_source_attention", {}); rt=d.get("runtime", {}); rows=rt.get("per_block", []); c=o.get("runtime_execution_certificate", {}); block_ok=bool(rows) and all(x.get("call_count")==80 and x.get("ulysses_observed") is True and isinstance(x.get("source_sequence_length"),int) and x.get("source_sequence_length")>0 and x.get("target_sequence_length")==x.get("source_sequence_length") and x.get("full_sequence_length")==x.get("source_sequence_length")+x.get("target_sequence_length") for x in rows); checks=(r.get("untrained_oracle") is True,r.get("source_and_instruction_only") is True,r.get("production_claim_forbidden") is True,r.get("scientific_claim_forbidden") is True,r.get("scientific_claim_authorized") is False,r.get("method_source_revision")==run_rev,r.get("method_source_archive_sha256")==run_arc,a.get("training_global_step")==644,o.get("training_method_source_revision")==train_rev,o.get("training_method_source_archive_sha256")==train_arc,o.get("zero_training") is True,o.get("trained_parameters")==0,d.get("block_selection")==scope,rt.get("restored") is True,c.get("validated") is True,c.get("expected_calls_per_selected_block")==80,c.get("ulysses_observed_on_every_selected_block") is True,c.get("selected_block_count")==len(rows),block_ok); raise SystemExit(0 if all(checks) else 1)' "${receipt_path}" "${block_scope}" "${source_revision}" "${source_archive_sha256}" "${training_source_revision}" "${training_source_archive_sha256}" || fail "oracle receipt audit failed"
rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[bernini-full644-directed] PASS"
