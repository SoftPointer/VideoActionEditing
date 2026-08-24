# Bernini-R action editing

This directory contains the main action-editing implementation for the open
`ByteDance/Bernini-R-1.3B-Diffusers` renderer and a large set of versioned
research branches built around the same source-only inference contract.

```text
training:  clean source latent + instruction + noisy target latent -> target velocity
inference: source video + instruction -> edited video
```

No external segmentation mask, swept tube, track, pose, trajectory, target
caption, target feature, or edited first frame is accepted at inference. The
renderer-internal `vae_latents_mask` is a packed-token loss selector, not a
spatial editing mask.

## Frozen identities

- Bernini source commit:
  `2d2b4591ac053ec25c6371b01a5a6746679e5793`
- VeOmni source commit:
  `f90b3dc6fbb0ce693745223cc7a94064123dbf4d`
- Bernini-R 1.3B checkpoint revision:
  `ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce`
- Audited checkpoint tree SHA-256:
  `6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca`

The 1.3B release is a single-transformer Wan2.1 renderer. The canonical path
keeps `transformer_1`, skips `transformer_2`, trains over the full flow interval,
and uses the native `mv2v` shift of `5.0`.

## Exact video contract

The audited data path uses exactly 81 frames at 25 FPS. VAE materialization
does not temporally subsample the input. Each pair is mapped to a source-aspect
bucket with a shared first-frame tensor and a maximum area of 245,760 pixels.
An 81-frame input produces 21 Wan VAE latent frames.

## Canonical pipeline

The stable baseline is organized around these files:

- `tools/build_renderer_dataset.py`: validates a preview manifest and emits
  Bernini renderer parquet rows.
- `tools/materialize_vae.py`: creates exact-81-frame source/target VAE rows.
- `tools/finalize_vae_dataset.py`: verifies every shard and freezes the dataset
  index and summary.
- `train_lora.py`: trains the source-conditioned attention LoRA with strict
  upstream, dataset, distributed, and receipt validation.
- `infer_lora.py`: reloads and merges an adapter, accepts one source video and
  one instruction, and emits an 81-frame result plus a hash-bound receipt.
- `exact_local_video_materializer_v1.py`: exact frame and geometry loading used
  by inference.
- `action_preservation_decoded_eval_model_authority_v2.py`: model authority and
  consumption checks used by retained evaluation paths.

Primary cluster templates:

- `scripts/auh_prepare_raw.sbatch`
- `scripts/auh_materialize_smoke.sbatch`
- `scripts/auh_materialize_full.sbatch`
- `scripts/auh_train_lora.sbatch`
- `scripts/auh_infer_lora.sbatch`

Review every scheduler directive, path, partition, and GPU request before using
these templates on a different cluster.

## Direct validation

From the repository root:

```bash
PYTHONPATH=methods/bernini_action_editing \
  python -m unittest discover \
  -s methods/bernini_action_editing/tests \
  -p 'test_train_lora_contract.py' -v

PYTHONPATH=methods/bernini_action_editing \
  python -m unittest discover \
  -s methods/bernini_action_editing/tests \
  -p 'test_infer_lora_contract.py' -v

PYTHONPATH=methods/bernini_action_editing \
  python -m unittest discover \
  -s methods/bernini_action_editing/tests \
  -p 'test_materialize_vae.py' -v
```

These suites use synthetic fixtures and validate contracts without loading the
full checkpoint. GPU integration tests require the environment and external
assets described in the repository-level `ENVIRONMENT.md`.

## Research code map

The directory preserves versioned experiments because their filenames,
schemas, launchers, and tests are part of each experiment's provenance.

- Delta and tangent methods: C2FR, projected delta LoRA, prior-guided tangent,
  differential sampling, and source-relative motion fields.
- Source-carrier methods: source K/V replay, source spectral bridges,
  source-noised carriers, and native multi-video routes.
- Low-dimensional action methods: few-shot privileged motion codes, semantic
  action codecs, and action-representation adapters.
- Self-generated reward methods: DCLR, HAT, DMIQ, MACE, self-guided fields,
  preference objectives, and factorial action banks.
- Hidden-event geometry: temporal quotients, latent event critics, trajectory
  probes, and causal localization.
- Role and identity methods: source-role localization, null-space observers,
  role rebinding, and source graph preservation.
- Source-anchor and online-attention methods: SAIC, action-anchor distillation,
  dynamic/static anchor attention, target-T0 action representation, and
  checkpoint visual-quality gates.
- Evaluation and release tooling: decoded preservation evaluation, target-blind
  gates, post-video routing, source-bound scoring, and immutable receipt checks.

`spt_v2/` contains a self-contained source-preserving transport branch with its
own contracts, trainers, inference entry points, and launch scripts.

## Version discipline

Version suffixes are not cosmetic. Later files may change the optimizer,
conditioning route, authority boundary, topology, or receipt schema. A frozen
run must use the exact source paths and hashes named by its release contract.
Do not replace an older entry point with a newer one solely because the newer
filename has a larger version number.

Many experiments are falsification or engineering canaries. A successful job
or finite gradient does not by itself establish action correctness, identity
preservation, held-out generalization, or scientific authority.

## Data and model boundaries

- Source datasets and generated media remain outside the repository.
- Target videos may supervise training or be consumed by an offline evaluator,
  but must not be read by the source-only inference process.
- No-target rows are valid for source preservation, representation learning,
  or self-generated anchors, not flow-matching SFT.
- Adapter inference must match the Transformers and PEFT versions recorded in
  the training receipt.
- Model, source, adapter, and output paths must be plain files/directories rather
  than symlinks when a fail-closed contract requires it.

See the repository-level `README.md` for the end-to-end command sequence and
`ENVIRONMENT.md` for installation details.
