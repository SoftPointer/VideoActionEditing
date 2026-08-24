#!/usr/bin/env bash
set -euo pipefail

run_dir="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/source-owned-role-locator-v15b-e00-sp4-r6-null64-67fd8211-ff71de79-r2"
python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"

test "$(hostname)" = "auh7-1b-gpu-292"
test -d "${run_dir}"
test ! -e "${run_dir}/output/e00_v15b_r6_probe_receipt.json"
test ! -e "${run_dir}/output/e00_v15b_r6_affinity.safetensors"

mkdir -p "${run_dir}/output" "${run_dir}/miopen"
for rank in 0 1 2 3; do
  mkdir -p \
    "${run_dir}/miopen/rank_${rank}/miopen-user" \
    "${run_dir}/miopen/rank_${rank}/miopen-custom"
done

export HIP_VISIBLE_DEVICES="0,1,2,3"
export MODELING_BACKEND="hf"
export V15B_MIOPEN_CACHE_ROOT="${run_dir}/miopen"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="1"
export NCCL_DEBUG="WARN"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export PYTHONPATH="${run_dir}:${PYTHONPATH:-}"

cd "${run_dir}"
exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --master_port=29661 \
  probe_source_owned_role_locator_v15b_r6_sp4.py \
  --runtime-adapter auh_source_owned_role_locator_v15_adapter:create_auh_bernini_source_role_adapter \
  --adapter-config '{}' \
  --role-asset "${run_dir}/assets/interaction_e00_source_instance_role_token_spans_v15b.json" \
  --null-registry "${run_dir}/assets/interaction_e00_null_token_span_registry_v15b_r6.json" \
  --event-id pour-liquid-into-cup \
  --block-indices 4 9 14 19 24 \
  --diagnostics-output "${run_dir}/output/e00_v15b_r6_affinity.safetensors" \
  --output "${run_dir}/output/e00_v15b_r6_probe_receipt.json"
