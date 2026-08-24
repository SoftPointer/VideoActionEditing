#!/usr/bin/env bash
# Sole node279 Slurm-step payload for one immutable PRE_D0 Level-A P2 attempt.
# It never retries and never mutates parent allocation 140846.

set -Eeuo pipefail
umask 077

readonly job_id=140846
readonly node=auh7-1b-gpu-279
readonly python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly experiment_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_action_editing_0817
readonly tag=fresh-world8-level-a-r2-p2-launchbound-v2
readonly release_root="${experiment_root}/releases/${tag}"
readonly launch_root="${experiment_root}/launchers/${tag}"
readonly run_root="${experiment_root}/runs/${tag}"
readonly step_self="${launch_root}/auh_fresh_world8_level_a_r2_p2_node279_step_v1.sh"
readonly release_manifest="${release_root}/RELEASE_MANIFEST.json"
readonly release_manifest_sha=f9e9f8542ec701cc9890fed919695980b989fd6d731eb914a5588edb1de4eeaa
readonly driver="${release_root}/action_edit_fresh_world8_level_a_driver_0817_v1.py"
readonly driver_sha=6435c6bb06a79cfcb407c137571404e5962e0de50e8082e7bd600e4618c05ea4
readonly consumer_sha=8bf0a9e48e0b2443a8e2f8e0744d08591226a167ac6ace45ee513481f5a97b3a
readonly product_sha=b16d8aef25b35df13e8294ef387e4d334170af65c2f43ece9894142d7cadac14
readonly consumer="${release_root}/action_edit_checkpoint_consumer_0817_v1.py"
readonly product="${release_root}/infer_action_edit_product_abi_0817_v1.py"
readonly rank_exec="${launch_root}/auh_fresh_world8_level_a_r2_p2_node279_rank_exec_v1.sh"
readonly rank_exec_sha=37a099285453265f3442da10d14b577f3e31fd60ac01b9399356d2d9966b00c8
readonly r2_release_manifest="${experiment_root}/releases/pre-d0-paired2-edf3d1d2a77c-r2/RELEASE_MANIFEST.json"
readonly campaign_receipt="${experiment_root}/runs/pre_d0_engineering_paired2-edf3d1d2a77c-r2/receipt.json"
readonly checkpoint_dir="${experiment_root}/runs/pre_d0_engineering_paired2-edf3d1d2a77c-r2/checkpoints/checkpoint-00000002"
readonly bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
readonly veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
readonly base_checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
readonly checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256

fail() {
  printf 'Level-A P2 node279 step refused: %s\n' "$*" >&2
  exit 96
}

[[ $# == 5 ]] || fail "exact launch authority argv differs"
readonly attempt_label="$1"
[[ "${attempt_label}" == A || "${attempt_label}" == B ]] || fail "attempt label must be A or B"
readonly launch_authority_core="$2"
readonly launch_authority_core_sha="$3"
readonly attempt_intent="$4"
readonly attempt_intent_sha="$5"
readonly output_root="${run_root}/${attempt_label}"
[[ "${launch_authority_core}" == "${launch_root}/LAUNCH_AUTHORITY_CORE.json" ]] || fail "launch authority core path differs"
[[ "${launch_authority_core_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "launch authority core SHA differs"
[[ "${attempt_intent}" == "${experiment_root}/attempts/${tag}/${attempt_label}/STARTED/intent.json" ]] || fail "attempt intent path differs"
[[ "${attempt_intent_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "attempt intent SHA differs"
for authority_file in "${launch_authority_core}" "${attempt_intent}"; do
  [[ -f "${authority_file}" && ! -L "${authority_file}" ]] || fail "node-visible authority file differs: ${authority_file}"
  [[ "$(stat -c %a "${authority_file}")" == 444 ]] || fail "node-visible authority mode differs: ${authority_file}"
  [[ "$(stat -c %h "${authority_file}")" == 1 ]] || fail "node-visible authority link count differs: ${authority_file}"
done
[[ "$(sha256sum "${launch_authority_core}" | awk '{print $1}')" == "${launch_authority_core_sha}" ]] || fail "node-visible launch authority SHA differs"
[[ "$(sha256sum "${attempt_intent}" | awk '{print $1}')" == "${attempt_intent_sha}" ]] || fail "node-visible attempt intent SHA differs"

[[ "$0" == "${step_self}" ]] || fail "step payload must be invoked by its frozen absolute path"
[[ -x "${step_self}" && ! -L "${step_self}" ]] || fail "step payload file differs"
[[ "$(stat -c %a "${step_self}")" == 555 ]] || fail "step payload mode differs"
[[ "$(stat -c %h "${step_self}")" == 1 ]] || fail "step payload link count differs"
[[ "${SLURM_JOB_ID:-}" == "${job_id}" ]] || fail "parent job identity differs"
[[ "$(hostname -s)" == "${node}" ]] || fail "physical node differs"
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || fail "numeric Slurm step is absent"
[[ "${SLURM_STEP_NUM_NODES:-${SLURM_NNODES:-}}" == 1 ]] || fail "step node count differs"
[[ "${SLURM_STEP_NUM_TASKS:-${SLURM_NTASKS:-}}" == 1 ]] || fail "outer step task count differs"
readonly current_step="${job_id}.${SLURM_STEP_ID}"
readonly node_numeric_steps="$(
  /usr/bin/squeue --steps -w "${node}" -h -j "${job_id}" -o '%i' |
    awk -v prefix="${job_id}." \
      'index($0, prefix) == 1 && substr($0, length(prefix) + 1) ~ /^[0-9]+$/ { print $0 }' |
    LC_ALL=C sort
)"
[[ "${node_numeric_steps}" == "${current_step}" ]] || \
  fail "node279 numeric child closure differs: ${node_numeric_steps}"
[[ -x "${python_bin}" && ! -L "${python_bin}" ]] || fail "pinned Python differs"
[[ -d "${release_root}" && ! -L "${release_root}" ]] || fail "release root differs"
[[ "$(stat -c %a "${release_root}")" == 555 ]] || fail "release root mode differs"
readonly expected_release_entries=$'RELEASE_MANIFEST.json\naction_edit_checkpoint_consumer_0817_v1.py\naction_edit_fresh_world8_level_a_driver_0817_v1.py\naction_plan_predictor_v1.py\nclean_source_visual_context_stage_b_contract_v1.py\ninfer_action_edit_product_abi_0817_v1.py\ninference_sigma_strata.py\npacked_preservation_lora_v2.py\npacked_preservation_release_v2.py\nsource_self_runtime.py\ntrain_action_edit_large_lora_0817_v1.py\ntrain_lora.py'
readonly observed_release_entries="$(find "${release_root}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)"
[[ "${observed_release_entries}" == "${expected_release_entries}" ]] || fail "release exact root closure differs"
for pinned_file in "${release_manifest}" "${driver}" "${consumer}" "${product}"; do
  [[ -f "${pinned_file}" && ! -L "${pinned_file}" ]] || fail "release pinned file differs: ${pinned_file}"
  [[ "$(stat -c %a "${pinned_file}")" == 444 ]] || fail "release pinned file mode differs: ${pinned_file}"
  [[ "$(stat -c %h "${pinned_file}")" == 1 ]] || fail "release pinned file link count differs: ${pinned_file}"
done
[[ "$(sha256sum "${release_manifest}" | awk '{print $1}')" == "${release_manifest_sha}" ]] || fail "release manifest SHA differs"
[[ "$(sha256sum "${driver}" | awk '{print $1}')" == "${driver_sha}" ]] || fail "driver SHA differs"
[[ "$(sha256sum "${consumer}" | awk '{print $1}')" == "${consumer_sha}" ]] || fail "consumer SHA differs"
[[ "$(sha256sum "${product}" | awk '{print $1}')" == "${product_sha}" ]] || fail "product bridge SHA differs"
[[ -x "${rank_exec}" && ! -L "${rank_exec}" ]] || fail "rank wrapper differs"
[[ "$(stat -c %a "${rank_exec}")" == 555 ]] || fail "rank wrapper mode differs"
[[ "$(stat -c %h "${rank_exec}")" == 1 ]] || fail "rank wrapper link count differs"
[[ "$(sha256sum "${rank_exec}" | awk '{print $1}')" == "${rank_exec_sha}" ]] || fail "rank wrapper SHA differs"
[[ -f "${r2_release_manifest}" && ! -L "${r2_release_manifest}" ]] || fail "r2 release manifest differs"
[[ -f "${campaign_receipt}" && ! -L "${campaign_receipt}" ]] || fail "r2 campaign receipt differs"
[[ -d "${checkpoint_dir}" && ! -L "${checkpoint_dir}" ]] || fail "P2 checkpoint directory differs"
[[ -d "${run_root}" && ! -L "${run_root}" ]] || fail "run latch root differs"
[[ "$(stat -c %a "${run_root}")" == 700 ]] || fail "run latch root mode differs"
[[ ! -e "${output_root}" ]] || fail "attempt output root already exists"
mkdir -m 0700 "${output_root}" || fail "attempt output root claim failed"

"${python_bin}" -I -B -c '
import torch
assert torch.__version__ == "2.7.1+rocm6.3", torch.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("MI210" in name for name in names), names
print("PASS node279 exact8 MI210 Level-A runtime", names, flush=True)
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
  --no_python "${rank_exec}" "${driver}" run \
  --attempt-label "${attempt_label}" \
  --launch-authority-core "${launch_authority_core}" \
  --expected-launch-authority-core-sha256 "${launch_authority_core_sha}" \
  --attempt-intent "${attempt_intent}" \
  --expected-attempt-intent-sha256 "${attempt_intent_sha}" \
  --release-manifest "${release_manifest}" \
  --expected-release-manifest-sha256 "${release_manifest_sha}" \
  --expected-driver-source-sha256 "${driver_sha}" \
  --expected-consumer-source-sha256 "${consumer_sha}" \
  --expected-product-source-sha256 "${product_sha}" \
  --r2-release-manifest "${r2_release_manifest}" \
  --campaign-receipt "${campaign_receipt}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --bernini-root "${bernini_root}" \
  --veomni-root "${veomni_root}" \
  --base-checkpoint "${base_checkpoint}" \
  --checkpoint-content-manifest "${checkpoint_manifest}" \
  --output-root "${output_root}"
