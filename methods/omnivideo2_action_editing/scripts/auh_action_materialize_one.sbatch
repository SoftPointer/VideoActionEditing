#!/usr/bin/env bash
#SBATCH --job-name=marp-mat-a0b
#SBATCH --partition=faculty
#SBATCH --qos=stqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:mi210:4
#SBATCH --time=04:00:00
#SBATCH --exclude=auh7-1b-gpu-185,auh7-1b-gpu-187,auh7-1b-gpu-195,auh7-1b-gpu-233,auh7-1b-gpu-318

set -Eeuo pipefail

source_archive="${MARP_SOURCE_ARCHIVE:?set MARP_SOURCE_ARCHIVE}"
source_archive_sha256="${MARP_SOURCE_ARCHIVE_SHA256:?set MARP_SOURCE_ARCHIVE_SHA256}"
source_revision="${MARP_SOURCE_REVISION:?set MARP_SOURCE_REVISION}"
preview_manifest="${MARP_PREVIEW_MANIFEST:?set MARP_PREVIEW_MANIFEST}"
sample_id="${MARP_SAMPLE_ID:?set MARP_SAMPLE_ID}"
output_dir="${MARP_MATERIALIZED_OUTPUT:?set MARP_MATERIALIZED_OUTPUT}"
temporal_mode="${MARP_TEMPORAL_MODE:-full_81_25fps}"
spatial_profile="${MARP_SPATIAL_PROFILE:-full_480p}"
omnivideo_root="${MARP_OMNIVIDEO_ROOT:?set MARP_OMNIVIDEO_ROOT}"
omnivideo_commit="${MARP_OMNIVIDEO_COMMIT:-adcee0a4a5b439ad3615f825298221b21177d4e3}"
qwen_checkpoint="${MARP_QWEN_CHECKPOINT:?set MARP_QWEN_CHECKPOINT}"
vae_checkpoint="${MARP_VAE_CHECKPOINT:?set MARP_VAE_CHECKPOINT}"
umt5_checkpoint="${MARP_UMT5_CHECKPOINT:?set MARP_UMT5_CHECKPOINT}"
umt5_tokenizer="${MARP_UMT5_TOKENIZER:?set MARP_UMT5_TOKENIZER}"
python_bin="${MARP_PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"

fail() {
  echo "[marp-materialize] ERROR: $*" >&2
  exit 2
}

[[ -f "${source_archive}" && ! -L "${source_archive}" ]] || fail "invalid source archive"
[[ -f "${preview_manifest}" && ! -L "${preview_manifest}" ]] || fail "invalid preview manifest"
[[ -d "${omnivideo_root}/.git" ]] || fail "Omni root is not a Git checkout"
[[ -d "${qwen_checkpoint}" && ! -L "${qwen_checkpoint}" ]] || fail "invalid Qwen checkpoint"
[[ -f "${vae_checkpoint}" && ! -L "${vae_checkpoint}" ]] || fail "invalid VAE checkpoint"
[[ -f "${umt5_checkpoint}" && ! -L "${umt5_checkpoint}" ]] || fail "invalid UMT5 checkpoint"
[[ -d "${umt5_tokenizer}" && ! -L "${umt5_tokenizer}" ]] || fail "invalid UMT5 tokenizer"
[[ -x "${python_bin}" ]] || fail "Python is not executable"
[[ -d "${output_dir%/*}" ]] || fail "output parent does not exist"
[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "output already exists"
[[ "${source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid source archive digest"
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid source revision"
[[ "${omnivideo_commit}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid Omni commit"
[[ "${sample_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || fail "unsafe sample ID"
[[ "${temporal_mode}" == "full_81_25fps" || "${temporal_mode}" == "smoke_41_12p5fps" ]] || fail "invalid temporal mode"
[[ "${spatial_profile}" == "full_480p" || "${spatial_profile}" == "motion_384p" ]] || fail "invalid spatial profile"

[[ "$(sha256sum "${source_archive}" | awk '{print $1}')" == "${source_archive_sha256}" ]] || fail "source archive digest mismatch"
[[ "$(sha256sum "${vae_checkpoint}" | awk '{print $1}')" == "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981" ]] || fail "VAE digest mismatch"
[[ "$(git -C "${omnivideo_root}" rev-parse HEAD)" == "${omnivideo_commit}" ]] || fail "Omni commit mismatch"
[[ -z "$(git -C "${omnivideo_root}" status --porcelain --untracked-files=no)" ]] || fail "Omni tracked files are dirty"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

task_scratch="${SLURM_TMPDIR:-/tmp}/marp-materialize-${SLURM_JOB_ID}"
mkdir -p \
  "${task_scratch}/source" \
  "${task_scratch}/miopen-user" \
  "${task_scratch}/miopen-custom" \
  "${task_scratch}/torch-extensions" \
  "${task_scratch}/triton"
export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"
export TRITON_CACHE_DIR="${task_scratch}/triton"

tar -xf "${source_archive}" -C "${task_scratch}/source"
method_root="${task_scratch}/source/methods/omnivideo2_action_editing"
[[ -f "${method_root}/tools/materialize_action_payloads.py" ]] || fail "archive lacks materializer"

hostname
date -u +%Y-%m-%dT%H:%M:%SZ
printf '[marp-materialize] source_revision=%s\n' "${source_revision}"
printf '[marp-materialize] source_archive_sha256=%s\n' "${source_archive_sha256}"
printf '[marp-materialize] sample_id=%s\n' "${sample_id}"
printf '[marp-materialize] temporal_mode=%s\n' "${temporal_mode}"
printf '[marp-materialize] spatial_profile=%s\n' "${spatial_profile}"
rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true

"${python_bin}" "${method_root}/tools/materialize_action_payloads.py" \
  --preview-manifest "${preview_manifest}" \
  --output-dir "${output_dir}" \
  --sample-id "${sample_id}" \
  --omni-root "${omnivideo_root}" \
  --qwen-checkpoint "${qwen_checkpoint}" \
  --vae-checkpoint "${vae_checkpoint}" \
  --umt5-checkpoint "${umt5_checkpoint}" \
  --umt5-tokenizer "${umt5_tokenizer}" \
  --motion-tokens 16 \
  --device cuda:0 \
  --t5-device cuda:0 \
  --qwen-device-map balanced_low_0 \
  --qwen-video-frames 6 \
  --temporal-mode "${temporal_mode}" \
  --spatial-profile "${spatial_profile}" \
  --no-qwen-flash-attention \
  --min-crop-retention 0.8 \
  --allow-preview-exploration

PYTHONPATH="${method_root}" "${python_bin}" - \
  "${output_dir}" "${sample_id}" "${source_revision}" "${source_archive_sha256}" \
  "${temporal_mode}" "${spatial_profile}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from action import ActionLatentDataset, DataConfig

root = Path(sys.argv[1])
sample_id = sys.argv[2]
temporal_mode = sys.argv[5]
spatial_profile = sys.argv[6]
expected_temporal = {
    "full_81_25fps": {
        "indices": list(range(81)),
        "frames": 81,
        "fps": 25.0,
        "subsampled": False,
    },
    "smoke_41_12p5fps": {
        "indices": list(range(0, 81, 2)),
        "frames": 41,
        "fps": 12.5,
        "subsampled": True,
    },
}[temporal_mode]
expected_hw = {
    "full_480p": {(60, 104), (104, 60)},
    "motion_384p": {(48, 80), (80, 48)},
}[spatial_profile]
expected_video_hw = {
    "full_480p": (480, 832),
    "motion_384p": (384, 640),
}[spatial_profile]
data_config = DataConfig.from_mapping(
    {
        "video_num_frames": expected_temporal["frames"],
        "video_fps": expected_temporal["fps"],
        "video_height": expected_video_hw[0],
        "video_width": expected_video_hw[1],
        "temporal_mode": temporal_mode,
        "spatial_profile": spatial_profile,
        "allow_transpose": True,
        "smoke_only": expected_temporal["subsampled"],
        "require_materialization_metadata": True,
    }
)
dataset = ActionLatentDataset(
    root / "manifest.jsonl",
    payload_root=root / "payloads",
    expected_motion_tokens=16,
    expected_latent_shapes=data_config.expected_latent_shapes,
    expected_data_config=data_config,
    allowed_task_types=("action_edit",),
    allow_preview=True,
)
if len(dataset) != 1:
    raise RuntimeError(f"expected one materialized row, got {len(dataset)}")
sample = dataset[0]
if sample["sample_id"] != sample_id:
    raise RuntimeError("materialized sample identity differs")
receipt_path = root / "materialization.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected_t = (expected_temporal["frames"] - 1) // 4 + 1
for key, expected in {
    "temporal_mode": temporal_mode,
    "spatial_profile": spatial_profile,
    "temporal_indices": expected_temporal["indices"],
    "source_frame_count": 81,
    "materialized_frame_count": expected_temporal["frames"],
    "source_fps": 25.0,
    "materialized_fps": expected_temporal["fps"],
    "temporal_subsampled": expected_temporal["subsampled"],
}.items():
    if receipt.get(key) != expected:
        raise RuntimeError(
            f"materialization receipt {key} differs: "
            f"expected={expected!r}, actual={receipt.get(key)!r}"
        )
for name in ("source_latent", "target_latent"):
    shape = tuple(sample[name].shape)
    if shape[:2] != (16, expected_t) or shape[2:] not in expected_hw:
        raise RuntimeError(
            f"{name} shape {shape} differs from {temporal_mode}/{spatial_profile}"
        )
verification = {
    "format": "marp-action-real-preview-materialization-verification-v1",
    "complete": True,
    "preview_only": True,
    "scientific_quality_not_tested": True,
    "sample_id": sample_id,
    "source_revision": sys.argv[3],
    "source_archive_sha256": sys.argv[4],
    "temporal_mode": temporal_mode,
    "spatial_profile": spatial_profile,
    "temporal_indices_verified": True,
    "temporal_subsampled": expected_temporal["subsampled"],
    "materialization_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    "source_latent_shape": list(sample["source_latent"].shape),
    "target_latent_shape": list(sample["target_latent"].shape),
    "source_vlm_shape": list(sample["source_vlm_context"].shape),
    "target_motion_shape": list(sample["target_motion_tokens"].shape),
    "provenance_sha256": sample["provenance_sha256"],
    "target_motion_tokens_usage": receipt["target_motion_tokens_usage"],
}
path = root / "verification.json"
path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(verification, indent=2, sort_keys=True))
PY

rocm-smi --showmeminfo vram --showuse 2>/dev/null || true
echo "[marp-materialize] PASS"
