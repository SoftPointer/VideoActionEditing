#!/usr/bin/env bash
#SBATCH --job-name=wan22-i2v-shard
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:mi210:8
#SBATCH --time=12:00:00
#SBATCH --requeue

# Freeze the socket-only RCCL transport that passed the exact Wan Ulysses
# ALLTOALL diagnostic, then delegate to the ordinary recoverable full runner.

set -Eeuo pipefail
umask 077

source_snapshot="${MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT:?set MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT}"
full_script="${source_snapshot}/methods/motive/scripts/auh_wan22_i2v_full.sbatch"

if [[ -L "${full_script}" || ! -f "${full_script}" ]]; then
  echo "[wan22-parallel-full] missing regular full runner: ${full_script}" >&2
  exit 2
fi

export NCCL_IB_DISABLE=1
unset NCCL_IB_HCA NCCL_IB_GID_INDEX
export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET

echo "[wan22-parallel-full] frozen_transport=socket interface=bond0"
exec bash "${full_script}"
