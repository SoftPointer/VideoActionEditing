#!/bin/bash
#SBATCH --job-name=seer-fm160-r1
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318
#SBATCH --output=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813/logs/seer-fm160-r1-%j.out
#SBATCH --error=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813/logs/seer-fm160-r1-%j.err

set -Eeuo pipefail
umask 077

EXP=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_seer_event_erasure_v1_20260813
METHOD_ARCHIVE="${EXP}/staging/seer-event-erasure-method-r1.tar"
METHOD_ROOT="${EXP}/staging/seer-event-erasure-b0-r1/methods/bernini_action_editing"
TRAINER="${METHOD_ROOT}/train_seer_event_erasure_fm.py"
WRAPPER="${EXP}/staging/seer-train-runner-r1/rank-wrapper.sh"
RAW="${EXP}/data/event-erasure-raw-r2"
VAE="${EXP}/data/event-erasure-vae-r1"
FINAL="${EXP}/data/event-erasure-final-r1"
OUTPUT="${EXP}/runs/seer-same-state-fm-160step-r1"
PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python
TORCHRUN=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/torchrun
BERNINI=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
VEOMNI=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
CHECKPOINT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4

METHOD_ARCHIVE_SHA256=ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822
TRAINER_SHA256=4afea87301a05cbd44bfe906b404efc4c8ce355f4c16459fff104bd80c01379e
WRAPPER_SHA256=5d2e38e790620f500a251a0febe620f98abe4e8591cb453a55438359dcb7e738
OWNER_SHA256=42e0421404db59c198c6881ea4866344061e3b1ed03edb3c79f61efca68bb1f1
ROUTING_SHA256=6c69cb748e6776d105f7c6c15caf3bd13179f4b09d9819e744de82aa1af77847
SUMMARY_SHA256=69b4587abd78459fdabf6fa41c4ed5e3f9c6f291b7516db9faf2dbd981e9f4a2
MANIFEST_SHA256=5ae25aefff8f2b3583954be6010e696bae3c21df7aa9281a35ae88123c3e9aaa
METHOD_REVISION=6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a
BERNINI_COMMIT=2d2b4591ac053ec25c6371b01a5a6746679e5793
VEOMNI_COMMIT=f90b3dc6fbb0ce693745223cc7a94064123dbf4d
CHECKPOINT_TREE_SHA256=6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca

fail() {
  echo "SEER_FM160_PREFLIGHT_FAILED: $*" >&2
  exit 1
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual=""
  test -f "${path}" || fail "${label} is absent: ${path}"
  read -r actual _ < <(sha256sum -- "${path}")
  test "${actual}" = "${expected}" || fail "${label} sha256 differs: ${actual}"
}

test "${SLURM_NNODES:?}" = 1 || fail "requires exactly one node"
test "${SLURM_NTASKS:?}" = 1 || fail "requires exactly one Slurm task"
test "${SLURM_CPUS_PER_TASK:?}" = 32 || fail "requires exactly 32 CPUs"
test "${SLURM_JOB_NUM_NODES:?}" = 1 || fail "job node count differs"
test ! -e "${OUTPUT}" || fail "fresh output already exists: ${OUTPUT}"
test -x "${PYTHON}" || fail "pinned Python is not executable"
test -x "${TORCHRUN}" || fail "pinned torchrun is not executable"
test -x "${WRAPPER}" || fail "pinned rank wrapper is not executable"
test -d "${BERNINI}" || fail "pinned Bernini checkout is absent"
test -d "${VEOMNI}" || fail "pinned VeOmni checkout is absent"
test -d "${CHECKPOINT}" || fail "pinned base checkpoint is absent"
test -d "${VAE}/shards" || fail "sealed VAE shards are absent"

require_sha256 "${METHOD_ARCHIVE}" "${METHOD_ARCHIVE_SHA256}" "method archive"
require_sha256 "${TRAINER}" "${TRAINER_SHA256}" "sealed trainer"
require_sha256 "${WRAPPER}" "${WRAPPER_SHA256}" "successful-r3 rank wrapper"
require_sha256 "${METHOD_ROOT}/assets/seer_owner_core2_v1.json" "${OWNER_SHA256}" "owner specification"
require_sha256 "${RAW}/full_pair.routing.jsonl" "${ROUTING_SHA256}" "full-pair routing"
require_sha256 "${FINAL}/vae.summary.json" "${SUMMARY_SHA256}" "VAE summary"
require_sha256 "${FINAL}/seer_dataset_manifest.json" "${MANIFEST_SHA256}" "SEER dataset manifest"

cache_base="${SLURM_TMPDIR:-/tmp}/seer-fm160-single4-${SLURM_JOB_ID}"
test ! -e "${cache_base}" || fail "rank cache root already exists: ${cache_base}"
mkdir -m 0700 "${cache_base}"
cleanup() {
  find "${cache_base}" -xdev -mindepth 1 -delete 2>/dev/null || true
  rmdir "${cache_base}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The hash-bound wrapper creates rank-{0,1,2,3} beneath this node-local root
# and gives every rank isolated HOME/TMP/XDG/MIOpen/Triton/extension caches.
export SEER_CACHE_ROOT="${cache_base}"
export SEER_PYTHON="${PYTHON}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

"${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --max-restarts=0 \
  --no-python \
  "${WRAPPER}" \
  "${TRAINER}" \
  --bernini-root "${BERNINI}" \
  --veomni-root "${VEOMNI}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-bernini-commit "${BERNINI_COMMIT}" \
  --expected-veomni-commit "${VEOMNI_COMMIT}" \
  --expected-checkpoint-tree-sha256 "${CHECKPOINT_TREE_SHA256}" \
  --preprocessed-parquet-dir "${VAE}/shards" \
  --dataset-summary "${FINAL}/vae.summary.json" \
  --routing-jsonl "${RAW}/full_pair.routing.jsonl" \
  --expected-routing-jsonl-sha256 "${ROUTING_SHA256}" \
  --output "${OUTPUT}" \
  --max-steps 160 \
  --save-every 40 \
  --learning-rate 1e-6 \
  --seer-owner-spec "${METHOD_ROOT}/assets/seer_owner_core2_v1.json" \
  --expected-seer-owner-spec-sha256 "${OWNER_SHA256}" \
  --seer-dataset-manifest "${FINAL}/seer_dataset_manifest.json" \
  --expected-seer-manifest-sha256 "${MANIFEST_SHA256}" \
  --method-source-revision "${METHOD_REVISION}" \
  --method-source-archive-sha256 "${METHOD_ARCHIVE_SHA256}"

for step in 00000040 00000080 00000120 00000160; do
  checkpoint_dir="${OUTPUT}/checkpoint-${step}"
  test -f "${checkpoint_dir}/adapter/adapter_model.safetensors" || fail "checkpoint ${step} adapter is absent"
  test -f "${checkpoint_dir}/adapter/adapter_config.json" || fail "checkpoint ${step} adapter config is absent"
  test -f "${checkpoint_dir}/optimizer.pt" || fail "checkpoint ${step} optimizer state is absent"
  test -f "${checkpoint_dir}/receipt.json" || fail "checkpoint ${step} receipt is absent"
done

"${PYTHON}" -B - "${OUTPUT}" "${METHOD_REVISION}" "${METHOD_ARCHIVE_SHA256}" \
  "${ROUTING_SHA256}" "${SUMMARY_SHA256}" "${MANIFEST_SHA256}" "${OWNER_SHA256}" \
  "${BERNINI_COMMIT}" "${VEOMNI_COMMIT}" "${CHECKPOINT_TREE_SHA256}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys


output = Path(sys.argv[1])
(
    revision,
    archive_sha,
    routing_sha,
    summary_sha,
    manifest_sha,
    owner_sha,
    bernini_commit,
    veomni_commit,
    checkpoint_tree_sha,
) = sys.argv[2:]
receipt_path = output / "checkpoint-00000160" / "receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"SEER_FM160_POSTFLIGHT_FAILED: {message}")


require(receipt.get("schema_version") == "bernini-seer-same-state-fm-training-receipt-v1", "receipt schema differs")
require(receipt.get("method") == "self-generated-event-erasure-same-state-fm-motion-copy-v1", "method differs")
require(receipt.get("global_step") == 160 and receipt.get("max_steps") == 160, "final step differs")
require(receipt.get("bernini_commit") == bernini_commit, "Bernini revision differs")
require(receipt.get("veomni_commit") == veomni_commit, "VeOmni revision differs")
require(receipt.get("checkpoint", {}).get("tree_sha256") == checkpoint_tree_sha, "checkpoint tree differs")
distributed = receipt.get("distributed", {})
require(distributed.get("world_size") == 4 and distributed.get("ulysses_size") == 4, "WORLD4 contract differs")
immutable = receipt.get("immutable_contract", {})
value = immutable.get("value", {})
require(immutable.get("method_source_archive_sha256") == archive_sha, "archive binding differs")
require(value.get("method_source_archive_sha256") == archive_sha, "inner archive binding differs")
require(value.get("method_source_revision") == revision, "method revision differs")
require(value.get("expected_routing_jsonl_sha256") == routing_sha, "routing binding differs")
require(value.get("dataset_summary_sha256") == summary_sha, "summary binding differs")
require(value.get("expected_seer_manifest_sha256") == manifest_sha, "manifest binding differs")
require(value.get("expected_seer_owner_spec_sha256") == owner_sha, "owner binding differs")
require(value.get("learning_rate") == 1e-6, "learning rate differs")
require(receipt.get("dataset", {}).get("rows") == 2, "sealed row count differs")
require(receipt.get("dataset", {}).get("routing", {}).get("file_sha256") == routing_sha, "receipt routing differs")
require(receipt.get("dataset", {}).get("summary", {}).get("sha256") == summary_sha, "receipt summary differs")
seer = receipt.get("seer", {})
require(seer.get("dataset_manifest_sha256") == manifest_sha, "receipt manifest differs")
require(seer.get("owner_spec_sha256") == owner_sha, "receipt owner differs")
require(seer.get("training_completion_is_method_success") is False, "receipt overclaims method success")
evidence = receipt.get("parameter_update_evidence", {})
require(evidence.get("engineering_execution_success") is True, "engineering update success is absent")
require(evidence.get("exact_parameter_bytes_changed") is True, "parameter bytes did not change")
require(evidence.get("method_success_claimed") is False, "parameter update overclaims method success")
require(evidence.get("initial_trainable_parameter_digest") != evidence.get("final_trainable_parameter_digest"), "parameter digest did not change")
gradient_norm = evidence.get("final_preclip_gradient_norm")
require(isinstance(gradient_norm, (int, float)) and math.isfinite(gradient_norm) and gradient_norm > 0, "final gradient norm is not positive finite")
require(receipt.get("production_claim_forbidden") is True and receipt.get("scientific_claim_authorized") is False, "claim boundary differs")

latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
require(latest == {"checkpoint": str(output / "checkpoint-00000160"), "global_step": 160}, "latest pointer differs")

payload = dict(receipt)
declared_digest = payload.pop("receipt_digest", None)
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
require(declared_digest == hashlib.sha256(canonical).hexdigest(), "receipt digest differs")
print(f"SEER_FM160_POSTFLIGHT_PASS receipt_sha256={hashlib.sha256(receipt_path.read_bytes()).hexdigest()}")
PY

echo "SEER_FM160_SINGLE4_PASS output=${OUTPUT}"
