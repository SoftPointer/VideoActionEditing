#!/usr/bin/env bash
# Runs inside the sole node279 Slurm step. No retry, promotion, or parent action.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly release_root="${experiment_root}/releases/pre-d0-paired2-1211e56185aa"
readonly launch_root="${experiment_root}/launchers/pre-d0-paired2-1211e56185aa-v2"
readonly output="${experiment_root}/runs/pre_d0_engineering_paired2-1211e56185aa-r1"
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=f20fb289d582d79631cc4cd23206735448c51f660b8f9fb67274e27786f37b69
readonly runner="${release_root}/train_action_edit_large_lora_0817_v1.py"
readonly runner_sha=1211e56185aabec4c1444f05316727864ad92d052c32dc10daddcbc66abcbb6d
readonly rank_exec="${launch_root}/auh_pre_d0_paired2_node279_rank_exec_v1.sh"
readonly rank_exec_sha=bac05573c035ce6770665304f54970ed1fefc2aa76f2801f1e4cb34929f5b638
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
readonly dataset_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c

fail() {
  printf 'PRE_D0 node279 step refused: %s\n' "$*" >&2
  exit 96
}

[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
[[ "$(hostname -s)" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm step is absent"
[[ "${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail "step node count differs"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
[[ -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root differs"
[[ "$(stat -c %a "${release_root}")" == 555 ]] || fail "release root mode differs"
[[ -f "${release_manifest}" && ! -L "${release_manifest}" ]] || fail "release manifest differs"
[[ "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ -f "${runner}" && ! -L "${runner}" ]] || fail "runner file differs"
[[ "$(sha256sum "${runner}" | awk '{print $1}')" == "${runner_sha}" ]] || fail "runner SHA differs"
[[ -x "${rank_exec}" && ! -L "${rank_exec}" ]] || fail "rank wrapper differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ ! -e "${output}" ]] || fail "PRE_D0 output is not fresh"

"${python_bin}" -I -B -c '
from pathlib import Path
import sys
manifest, expected, root = sys.argv[1:]
sys.path.insert(0, root)
import train_action_edit_large_lora_0817_v1 as runner
receipt = runner.validate_release_manifest(
    Path(manifest), expected_sha256=expected, method_root=Path(root)
)
assert receipt["member_count"] == 8
print("PASS frozen PRE_D0 release closure", receipt["member_set_sha256"], flush=True)
' "${release_manifest}" "${release_manifest_sha}" "${release_root}"

"${python_bin}" -I -B -c '
import torch
assert torch.__version__ == "2.7.1+rocm6.3", torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("MI210" in name for name in names), names
print("PASS node279 exact8 MI210 runtime", names, flush=True)
'

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
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
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

exec "${python_bin}" -I -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  --no_python "${rank_exec}" "${runner}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --checkpoint "${checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --preprocessed-parquet-dir "${dataset_root}/shards" \
  --dataset-summary "${dataset_root}/dataset_summary.json" \
  --output "${output}" \
  --max-steps 2 \
  --learning-rate 0.0001 \
  --max-grad-norm 1.0 \
  --seed 20260817 \
  --expected-bernini-commit 2d2b4591ac053ec25c6371b01a5a6746679e5793 \
  --expected-veomni-commit f90b3dc6fbb0ce693745223cc7a94064123dbf4d \
  --expected-checkpoint-tree-sha256 6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca \
  --expected-checkpoint-content-manifest-sha256 a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831 \
  --workspace-base-revision c00ce9ff9a5ea9ad15a94a756014f165ecb7fab1 \
  --expected-runner-source-sha256 "${runner_sha}" \
  --release-manifest "${release_manifest}" \
  --expected-release-manifest-sha256 "${release_manifest_sha}" \
  --ack-pre-d0-engineering-only \
  --ack-legacy-target-quality-unqualified \
  --ack-no-d0-or-scientific-claim \
  --ack-fresh-base-disposable
