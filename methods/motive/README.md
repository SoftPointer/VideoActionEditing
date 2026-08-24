# Motive: motion-centric data audit and action representation

`methods/motive` is a runnable Motive-inspired research prototype for video
motion filtering, action representation, and model-conditioned attribution. It
is not an official reproduction of any unpublished implementation detail.

The package separates two objects that should not be conflated:

1. A camera-compensated geometric motion descriptor used for conservative
   filtering, clustering, retrieval, and representation learning.
2. A model-conditioned gradient fingerprint tied to a specific checkpoint,
   parameter subset, timestep, noise realization, VAE posterior, and loss.

Neither object is a causal guarantee for a single video, and neither is an
inference-time action-control code by default.

## Features

- Bounded frame sampling and Farneback dense flow.
- Partial-affine/RANSAC global camera-motion estimation.
- Object residual flow, scene-cut detection, and conservative motion classes.
- Temporal-spatial HOOF and source-to-target action-delta descriptors.
- Motion-weighted latent losses and streaming CountSketch gradient projection.
- Single-query ranking and multi-query percentile-budget voting.
- Hash-bound archives with checkpoint, parameter, loss, randomness, and
  projection provenance.
- Rule, feature, and Qwen cascades with explicit `auto_keep`, `review`, and
  `auto_reject` states.
- Blind Qwen visual audit with strict JSON validation and fail-closed repair.
- Human-review templates and review/calibration reports.
- Action-representation training with split-only fitting, variance guards, and
  checkpoint/digest validation.
- Deterministic Goku/MEV manifest preparation and cluster orchestration tools.

Qwen evidence is treated as a pseudo-label. It cannot replace a human review
contract or silently overwrite authoritative dataset annotations.

## Install

```bash
python -m pip install -e methods/motive
```

Optional gradient attribution and action representation:

```bash
python -m pip install -e 'methods/motive[attribution,action-repr]'
```

Optional local Qwen audit:

```bash
python -m pip install -e 'methods/motive[qwen]'
```

The package requires Python 3.10 or newer. The main repository environment uses
Python 3.11; see `../../ENVIRONMENT.md` for the shared GPU setup.

## Test

```bash
python -m unittest discover -s methods/motive/tests -v
```

Most tests use synthetic arrays or small generated videos. Model-backed Qwen,
generator, and attribution integration tests require their optional checkpoints
and accelerator environment.

## CLI entry points

Installing the package exposes:

- `motive-audit`
- `motive-rank`
- `motive-goku-manifest`
- `motive-cascade`
- `motive-qwen-filter`
- `motive-train-action-repr`
- `motive-human-review`
- `motive-prepare-r5-pilot`
- `motive-r5-features`
- `motive-train-source-aware-r5`
- `motive-r5-gate`

Run any command with `--help` for its current schema and required paths.

## Minimal audit example

```bash
motive-audit \
  --input /path/to/pairs.jsonl \
  --root /path/to/read-only/videos \
  --output-dir /path/to/audit-output
```

For a source manifest built from Goku-style data:

```bash
PYTHONPATH=methods/motive python -m motive.goku_manifest \
  --dataset-root /path/to/dataset \
  --output /path/to/sample_manifest.jsonl \
  --sample-size 500 \
  --seed 260108828 \
  --semantic-classes continuous_action motion_suppression
```

Start with a bounded sample. Do not recursively decode a very large media tree
on a login node.

## Data and authority rules

- Source media are read-only; outputs go to a separate directory.
- Caption text is not used as hidden visual decision evidence unless a contract
  explicitly authorizes that field.
- Automation outputs retain their schema, model, prompt, and digest identities.
- Repaired or fallback visual outputs are routed to review rather than treated
  as automatic authority.
- A fixed hash order is reproducible, but is not automatically equivalent to a
  repeated randomized sample.
- No experiment result authorizes renderer training unless its independent
  admission gate explicitly passes.
