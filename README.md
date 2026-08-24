# VideoActionEditing

Research code for source-conditioned video action editing. The primary path
trains and evaluates a mask-free LoRA on the open Bernini-R 1.3B renderer:

```text
source video + edit instruction -> edited video
```

The inference contract does not accept a target video, segmentation mask,
track, pose, trajectory, edited first frame, or external motion reference.
Targets and privileged evaluators are confined to training or offline scoring.

This repository contains research and reproducibility code, including negative
results and fail-closed experiment gates. It is not a packaged production
service, and the presence of an experimental trainer does not imply that its
data or checkpoint has passed a scientific or production-quality gate.

## Repository layout

| Path | Purpose |
| --- | --- |
| `methods/bernini_action_editing/` | Main Bernini-R action-editing trainers, inference runtimes, experiment controllers, tools, schemas, and tests |
| `action_data_construction/` | Read-only MEV adjacent-event candidate construction and Qwen visual audit |
| `methods/omnivideo2_action_editing/` | OmniVideo2 action-editing baselines and training prototypes |
| `methods/action_editing_baselines/` | Shared source-plus-instruction baseline contract |
| `methods/motive/` | Motion-centric data audit, action representation, and attribution package |
| `methods/action_anchor_target_gap_audit/` | Action-anchor versus frozen-source target-gap audit |
| `methods/action_matcher_reward_audit/` | Order-aware action-matcher and reward audit |
| `methods/semantic_moments_reward_audit/` | SemanticMoments reward suitability audit |
| `tests/` | Cross-module action-editing contract tests |
| `md/action_editing/` | Seven path-stable JSON authority contracts required by hash-locked launchers and tests; general notes and experiment outputs are excluded |

Model weights, datasets, videos, adapters, experiment outputs, and cluster logs
are intentionally not included.

## Environment

The mainline environment uses Linux and Python 3.11. Install the PyTorch build
for the target accelerator first, then install the Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install a CUDA or ROCm PyTorch build first; see ENVIRONMENT.md.
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The project imports the official Bernini and VeOmni source trees at runtime.
The audited mainline pins are:

- Bernini: `2d2b4591ac053ec25c6371b01a5a6746679e5793`
- VeOmni: `f90b3dc6fbb0ce693745223cc7a94064123dbf4d`
- `ByteDance/Bernini-R-1.3B-Diffusers` revision:
  `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`

See [ENVIRONMENT.md](ENVIRONMENT.md) for CUDA, ROCm, model-download, and
version-parity instructions. Copy `.env.example` to a private shell setup or
export the same variables manually; the launchers do not require committing
local paths or credentials.

## Quick validation

Most contract tests use synthetic fixtures and do not load a model:

```bash
python -m unittest discover -s action_data_construction/tests -v

PYTHONPATH=methods/bernini_action_editing \
  python -m unittest discover \
  -s methods/bernini_action_editing/tests \
  -p 'test_train_lora_contract.py' -v

PYTHONPATH=methods/bernini_action_editing \
  python -m unittest discover \
  -s methods/bernini_action_editing/tests \
  -p 'test_infer_lora_contract.py' -v

python -m unittest discover \
  -s methods/omnivideo2_action_editing/tests -v

python -m pip install -e 'methods/motive[action-repr]'
python -m unittest discover -s methods/motive/tests -v
```

GPU integration tests and model-backed Qwen audits require the optional
environment described in [ENVIRONMENT.md](ENVIRONMENT.md).

## Data construction

MEV data are treated as read-only. The pipeline writes metadata containing
absolute references to source and target videos; it does not copy, crop,
transcode, or mutate source media.

```bash
export MEV_SOURCE_ROOT=/path/to/read-only/MEV
export ACTION_DATA_ROOT=/path/to/action-data-output

python action_data_construction/build_candidates.py \
  --source-root "$MEV_SOURCE_ROOT" \
  --output-root "$ACTION_DATA_ROOT" \
  --smoke-count 8
```

The source of semantic truth is the target event caption in
`annotations/mev.json`. Qwen is used to audit visual compatibility and cannot
overwrite the authoritative instruction. See
[action_data_construction/README.md](action_data_construction/README.md).

## Canonical Bernini-R pipeline

### 1. Build the renderer parquet

```bash
python methods/bernini_action_editing/tools/build_renderer_dataset.py \
  --preview-manifest /path/to/action_preview.jsonl \
  --output-parquet /path/to/raw/action.parquet \
  --receipt-path /path/to/raw/receipt.json \
  --acknowledge-preview-only
```

Use `--acknowledge-broader-natural-release` only when the broader release
contract has been reviewed. These flags record an explicit experimental-data
acknowledgement; they do not grant scientific or production authority.

### 2. Materialize exact-81-frame VAE rows

```bash
python methods/bernini_action_editing/tools/materialize_vae.py \
  --raw-parquet /path/to/raw/action.parquet \
  --raw-receipt /path/to/raw/receipt.json \
  --raw-job-done /path/to/raw/job_done.json \
  --checkpoint "$BERNINI_ACTION_CHECKPOINT" \
  --output-root /path/to/vae_rows \
  --device cuda:0 \
  --max-pixels 245760 \
  --stride 16 \
  --min-target-retention 0.98

python methods/bernini_action_editing/tools/finalize_vae_dataset.py \
  --raw-receipt /path/to/raw/receipt.json \
  --raw-job-done /path/to/raw/job_done.json \
  --materialized-root /path/to/vae_rows \
  --output-index /path/to/vae_rows/dataset_index.jsonl \
  --output-summary /path/to/vae_rows/dataset_summary.json
```

### 3. Train the LoRA

Four-way sequence parallelism is the audited cluster path:

```bash
torchrun --standalone --nproc_per_node=4 \
  methods/bernini_action_editing/train_lora.py \
  --bernini-root "$BERNINI_OFFICIAL_ROOT" \
  --veomni-root "$BERNINI_VEOMNI_ROOT" \
  --checkpoint "$BERNINI_ACTION_CHECKPOINT" \
  --preprocessed-parquet-dir /path/to/vae_rows \
  --dataset-summary /path/to/vae_rows/dataset_summary.json \
  --output /path/to/train_output \
  --num-frames 81 \
  --max-steps 400 \
  --method-source-revision '<40-hex-git-revision>' \
  --method-source-archive-sha256 '<64-hex-source-archive-sha256>'
```

The trainer validates the pinned upstream source, model tree, dataset summary,
LoRA scope, distributed initialization, and receipt identities before updating
weights. The AUH templates under `methods/bernini_action_editing/scripts/`
provide the complete source-archive and cluster contract.

### 4. Run source-only inference

```bash
torchrun --standalone --nproc_per_node=4 \
  methods/bernini_action_editing/infer_lora.py \
  --bernini-root "$BERNINI_OFFICIAL_ROOT" \
  --veomni-root "$BERNINI_VEOMNI_ROOT" \
  --checkpoint "$BERNINI_ACTION_CHECKPOINT" \
  --adapter-checkpoint /path/to/checkpoint-00000400 \
  --source-video /path/to/source.mp4 \
  --instruction 'Make the person sit down while preserving identity and scene.' \
  --output /absolute/path/to/edited.mp4 \
  --num-inference-steps 40 \
  --seed 42 \
  --method-source-revision '<40-hex-git-revision>' \
  --method-source-archive-sha256 '<64-hex-source-archive-sha256>'
```

The output path must be new. The runtime publishes a hash-bound receipt next
to the generated video and rejects target-video or external-control leakage.

## Research branches

`methods/bernini_action_editing/` also contains versioned experimental paths
for causal tangent editing, source carriers, low-dimensional motion codes,
self-generated rewards, hidden-event geometry, role rebinding, source-anchor
training, action-representation learning, and online anchor attention. Each
branch has paired tests and launch scripts. Version suffixes are meaningful;
do not silently substitute a newer script into a frozen experiment receipt.

## Reproducibility and safety notes

- Keep datasets, checkpoints, generated media, and credentials outside Git.
- Match the Transformers and PEFT versions recorded by a training receipt at
  inference time.
- Treat Qwen judgments and proxy rewards as audit evidence, not calibrated
  human labels.
- A no-target source row may be used for representation learning or source
  preservation, but not as a flow-matching target.
- Cluster launchers are templates. Review partitions, GPU types, paths, and
  resource limits before submission on another system.

## Upstream projects

- [Bernini](https://github.com/bytedance/Bernini)
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni)
- [Bernini-R 1.3B Diffusers checkpoint](https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers)

This repository does not vendor those projects or redistribute their weights.
