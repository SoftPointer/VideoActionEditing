# Environment setup

This document describes the mainline Bernini-R action-editing environment.
Many files in this repository are frozen research experiments; when a script
declares a stricter runtime version, that script and its receipt take priority
over the general environment below.

## Supported platform

- Linux x86_64
- Python 3.11
- FFmpeg and `ffprobe` available on `PATH`
- NVIDIA CUDA or AMD ROCm accelerator
- Four GPUs for the audited Ulysses training and inference path
- Sufficient local storage for the approximately 29 GB model plus datasets,
  VAE rows, checkpoints, and generated videos

CPU-only environments can run most schema, contract, and orchestration tests,
but they cannot run model materialization, training, or inference.

## Pinned upstream identities

The canonical trainer validates these identities:

| Component | Identity |
| --- | --- |
| Bernini source | `2d2b4591ac053ec25c6371b01a5a6746679e5793` |
| VeOmni source | `f90b3dc6fbb0ce693745223cc7a94064123dbf4d` |
| Bernini-R 1.3B model revision | `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce` |
| PyTorch generation | `2.7.1` with a backend-specific CUDA or ROCm wheel |
| Diffusers | `0.38.0` |
| Transformers | `5.5.4` |
| PEFT for the audited full644 adapter | `0.19.1` |

Clone the source dependencies outside this repository:

```bash
mkdir -p /path/to/third_party

git clone https://github.com/bytedance/Bernini.git \
  /path/to/third_party/Bernini
git -C /path/to/third_party/Bernini checkout \
  2d2b4591ac053ec25c6371b01a5a6746679e5793

git clone https://github.com/ByteDance-Seed/VeOmni.git \
  /path/to/third_party/VeOmni
git -C /path/to/third_party/VeOmni checkout \
  f90b3dc6fbb0ce693745223cc7a94064123dbf4d
```

Keep both checkouts clean. The launchers and trainers reject mismatched or
dirty tracked source when a frozen experiment requires exact provenance.

## Python environment

Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### NVIDIA CUDA

Install the PyTorch build that matches the host driver and CUDA toolkit. The
official Bernini environment uses a PyTorch 2.7.1 generation stack; obtain the
appropriate command from the PyTorch installation selector instead of letting
`requirements.txt` choose an arbitrary GPU wheel.

Example shape only:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/<matching-cuda-index>
```

### AMD ROCm

The AUH launchers were exercised on MI210 GPUs with ROCm 6.3 and a
`torch==2.7.1+rocm6.3` build supplied by that cluster. Install the wheel or
module approved for the target ROCm installation. Do not install a CUDA wheel
into the same environment.

Verify the backend before installing the remaining packages:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('HIP:', getattr(torch.version, 'hip', None))
print('CUDA:', getattr(torch.version, 'cuda', None))
PY
```

Install the repository dependencies after PyTorch:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e /path/to/third_party/VeOmni
```

Bernini is imported from its checkout via `--bernini-root`; it does not need to
be copied into this repository. If the pinned Bernini checkout provides an
installation command for its exact revision, follow it without allowing pip
to replace the already selected PyTorch backend.

## Model download

Install the Hugging Face CLI and download the exact revision:

```bash
python -m pip install 'huggingface_hub[cli]'
hf download ByteDance/Bernini-R-1.3B-Diffusers \
  --revision ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce \
  --local-dir /path/to/models/Bernini-R-1.3B-Diffusers
```

The project code validates a known checkpoint tree digest for frozen runs.
Do not modify files inside a checkpoint directory used by those runs.

## Project paths

Copy the example file and replace placeholders locally:

```bash
cp .env.example .env
```

`.env` is ignored by Git. The shell launchers use exported variables, so either
source the file explicitly or export the values in a scheduler job:

```bash
set -a
source .env
set +a
```

Never commit access tokens, private dataset paths that should not be disclosed,
or model credentials.

## Optional environments

### Motive

```bash
python -m pip install -e methods/motive
python -m pip install -e 'methods/motive[action-repr]'
```

For local Qwen audits:

```bash
python -m pip install -e 'methods/motive[qwen]'
```

### Qwen visual audit

The Qwen3-VL-32B data-audit jobs require `transformers`, `accelerate`, Pillow,
and `qwen-vl-utils`, plus enough aggregate device memory for the selected
parallel layout. Qwen outputs are pseudo-label evidence and remain pending
human qualification unless a separate review contract says otherwise.

### OmniVideo2 and external evaluators

OmniVideo2, TEAM, SemanticMoments, SAM2, V-JEPA2, VideoPrism, and similar
evaluators are external projects. Their source and checkpoints are not vendored
here. Install only the evaluator required by the selected audit and preserve
its exact checkpoint identity in the generated receipt.

## Version parity

Adapter inference must use the same Transformers and PEFT versions recorded by
the training receipt. Some later experimental branches pin a newer
Transformers version than the canonical upstream environment. The branch's
fail-closed version check is intentional; create a separate virtual environment
instead of upgrading a completed run in place.

## Troubleshooting

- If VeOmni installation replaces PyTorch, recreate the environment and install
  VeOmni with `--no-deps` after installing the accelerator-specific wheel.
- If `torch.cuda.is_available()` is false on ROCm, verify the cluster module,
  device visibility variables, and wheel source.
- If a script reports a source or checkpoint hash mismatch, do not bypass the
  check. Recreate the pinned checkout or checkpoint directory.
- If a receipt reports a Transformers/PEFT mismatch, run inference in an
  environment matching that receipt.
- Model-backed integration tests are expected to fail or skip when checkpoints
  and GPUs are unavailable; contract-only tests should still run on CPU.
