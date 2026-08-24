#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 EXPECTED_PHYSICAL_UNIQUE_ID TAG COMMAND..." >&2
  exit 64
fi

expected_unique_id="$1"
run_tag="$2"
shift 2

if [[ ! "$expected_unique_id" =~ ^[0-9a-f]{16}$ ]]; then
  echo "invalid expected physical GPU unique id" >&2
  exit 65
fi
if [[ ! "$run_tag" =~ ^[a-z0-9_-]+$ ]]; then
  echo "invalid run tag" >&2
  exit 66
fi
if [[
  -z "${SLURM_JOB_ID:-}"
  || -z "${SLURM_STEP_ID:-}"
  || "${SLURM_NTASKS:-}" != "5"
  || "${SLURM_NNODES:-}" != "1"
  || ! "${SLURM_PROCID:-}" =~ ^[0-4]$
]]; then
  echo "wrapper requires the exact five-task, one-node Slurm layout" >&2
  exit 67
fi

sync_root="/tmp/anchor-v2-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${run_tag}"

if [[ "$SLURM_PROCID" == "0" ]]; then
  mkdir -m 0700 "$sync_root"
  : > "$sync_root/ready"
fi

for _ in $(seq 1 240); do
  [[ -f "$sync_root/ready" ]] && break
  sleep 0.25
done
if [[ ! -f "$sync_root/ready" ]]; then
  echo "rank0 did not publish the synchronization root" >&2
  exit 68
fi

if [[ "$SLURM_PROCID" != "4" ]]; then
  for _ in $(seq 1 7200); do
    if [[ -f "$sync_root/done" ]]; then
      exit 0
    fi
    if [[ -f "$sync_root/fail" ]]; then
      exit 1
    fi
    sleep 1
  done
  echo "rank4 timed out without a terminal marker" >&2
  exit 69
fi

terminal_marker() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    : > "$sync_root/done"
  else
    printf '%s\n' "$rc" > "$sync_root/fail"
  fi
}
trap terminal_marker EXIT

cache_root="$sync_root/rank4-cache"
mkdir -m 0700 "$cache_root"
for leaf in tmp xdg pycache miopen-user miopen-custom triton inductor extensions; do
  mkdir -m 0700 "$cache_root/$leaf"
done

export TMPDIR="$cache_root/tmp"
export TMP="$cache_root/tmp"
export TEMP="$cache_root/tmp"
export XDG_CACHE_HOME="$cache_root/xdg"
export PYTHONPYCACHEPREFIX="$cache_root/pycache"
export MIOPEN_USER_DB_PATH="$cache_root/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="$cache_root/miopen-custom"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export TORCH_EXTENSIONS_DIR="$cache_root/extensions"
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

if [[ "${ROCR_VISIBLE_DEVICES:-}" != "0" ]]; then
  echo "rank4 must expose exactly Slurm logical device 0" >&2
  exit 70
fi

/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12 -B -c \
  '
import sys
import torch
import uuid

expected = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("GPU gate requires exactly one available logical device")
if type(torch.version.hip) is not str or not torch.version.hip:
    raise SystemExit("GPU gate requires a ROCm PyTorch build")
properties = torch.cuda.get_device_properties(0)
if properties.name != "AMD Instinct MI210":
    raise SystemExit(f"unexpected GPU name: {properties.name!r}")
actual = uuid.UUID(str(properties.uuid)).bytes.decode("ascii").lower()
if actual != expected:
    raise SystemExit(f"physical GPU unique id differs: {actual} != {expected}")
print("GPU_GATE_PASS", actual, torch.version.hip, file=sys.stderr, flush=True)
' \
  "$expected_unique_id"

"$@"
