# OmniVideo2 mask-free action editing

This directory contains the implementation for the corrected mask-free
action-editing research line, referred to as MARP-Omni in the source and
experiment identifiers.

The default contract is:

```text
source video + action instruction
→ untouched full-source OmniVideo2 condition
→ source-conditioned predicted motion tokens
→ complete-target rectified-flow supervision
→ edited video
```

It does not use a source/target mask, actor track, swept tube, erased source,
latent splice, or region projection. Target-derived motion tokens are
privileged training labels for the planner only. They must never be supplied
to the renderer; both training and inference render from planner predictions.

Status: the mask-free core contracts, preview-manifest join, official-encoder
materializer, and four-rank trainer are implemented. AUH jobs `123109`,
`123119`, `123171`, and `123192` are historical 41-frame/12.5-FPS real-payload
engineering records; they do not validate the current 81-frame mainline.
`123108` is a synthetic official-checkpoint DDP record. In the first 81-frame
diagnostic chain, `123264` completed full-480p materialization, `123265` failed
closed before forward because the old config confused four active keys with
26 active rows, `123266` reached a node with no usable HIP GPU before sample
loading, and dependent job `123267` was cancelled. Corrected full-480p and
motion-384p chains `123306`--`123309` all completed successfully. Both kept
all 81 frames at 25 FPS and completed official-checkpoint four-rank
forward/backward/optimizer, nonzero LoRA/planner gradients, adapter save, and
strict clean-base reload. The observed maximum reserved memory was 17.79 GB
for full-480p and 12.32 GB for motion-384p; full-480p did not OOM. This is a
single-preview-payload, one-step engineering result; no decoded video has yet
established action-editing quality. Historical run notes and generated audit
receipts are intentionally omitted from this code-focused repository.

## Directory layout

```text
action/
  config.py       closed experiment configuration
  dataset.py      mask-free, digest-bound latent payload contract
  flow.py         full-target Wan/DiffSynth rectified-flow primitives
  planner.py      source-only K×2048 temporal motion-plan predictor
  omni.py         strict official 1.3B loader, LoRA, source token budgeting

configs/
  marp_1_3b.json  81-frame 480p mainline at context 9216
  marp_1_3b_81f_640x384_ctx6144.json
                   81-frame spatial-downscale control at context 6144
  marp_81f_fullres_real_one_step.json
                   81-frame full-resolution four-rank feasibility config
  marp_81f_motion384_real_one_step.json
                   81-frame motion-resolution four-rank feasibility config
  marp_smoke_41f_real_one_step.json
                   explicit 41-frame engineering smoke only

tools/
  build_action_preview_manifest.py
                   fail-closed v17 Qwen/Wan/natural preview join
  materialize_action_payloads.py
                   official Qwen/UMT5/Wan-VAE preview payload materializer

tests/
  test_action_*.py
  test_build_action_preview_manifest.py

pact/             withdrawn tube-first implementation; legacy/oracle only

audits/           immutable-hash AUH engineering evidence and claim boundary
```

The independent upstream checkout remains at `methods/Omni-Video`. This
wrapper imports it at runtime rather than modifying the nested repository.

## Main method invariants

### Untouched source

The official source VAE latent is passed to OmniVideo2's Visual Context
Adapter without actor-dependent modification. Every v2 training config sets
`require_uncompressed_source=true`: if Qwen + T5 + plan + full source exceed
the configured context budget, the trainer fails before model execution and
reports that row's nonvisual, visual, total and budget token counts. The action
trainer never invokes source pooling or truncation.

The pinned official `special_tokens.pkl` serializes six delimiter tensors
(40 rows). The current 1.3B unified forward consumes four of them, occupying
26 context rows (`6+6+7+7`), not four rows. Checked-in configs require exactly 26, the trainer recomputes
the count from the trusted checkpoint, and run/done/adapter receipts plus the
strict verifier bind the contract ID, all six serialized shapes (40 rows), the
four active entries (26 rows), and both transformer/special-token file digests.
The special-token digest is checked before the upstream pickle is unpickled.
The loader also requires the tracked-clean upstream revision
`adcee0a4a5b439ad3615f825298221b21177d4e3`; transformer hashing and safe
deserialization share one open file descriptor, and provenance rechecks the
digest after model loading. Adapter checkpoints use an exact closed field set,
including the activation contract and CPU/device RNG states.

The loader also binds each real payload to its provenance-declared temporal
mode, FPS, exact frame indices, subsampling flag and spatial bucket. It writes
one `context_preflight.jsonl` row per sample. The feasibility configs use
Omni's official-compatible `fixed_budget` context padding. Because upstream
does not mask those padding positions, `batch_exact` is exposed only as an
explicit ablation and is never selected implicitly.

### Complete target

For complete target latent `x0`, the core uses:

```text
x_t      = (1 - sigma) * x0 + sigma * eps
v_target = eps - x0
loss     = mean squared velocity error over every target latent element
```

The implementation has no mask argument and no partial endpoint. The
standalone FP32 schedule follows the audited Wan/DiffSynth rational shift and
batch-shared discrete timestep convention without importing the legacy PACT
mask stack.

### Predicted motion plan only

`TemporalMotionPlanPredictor` receives only the source VLM context. The context
is expected to have been produced by Qwen from the source video together with
the edit instruction. It predicts `K` tokens of width 2048.

`target_motion_tokens` exist only in the offline payload and are detached
inside `motion_plan_loss`. The planner `forward` method has no target-token
argument. The trainer must append only the predicted tokens to Omni's VLM
condition. Feeding payload target tokens to the renderer is target leakage and
must fail validation.

The current V0 predictor directly predicts target-plan tokens. Calling this a
mathematical residual requires a separate residual implementation/ablation;
the safe current description is that the predicted tokens are a residual
additional condition on top of Omni's untouched native source condition.

### Action-only adaptation

The official Omni model, Visual Context Adapter, `vlm_norm`, and `vlm_proj`
are frozen. `enable_action_lora` injects FP32 LoRA weights only into a closed
Wan scope. The initial main scope is all self-/cross-attention `q/k/v/o`;
attention+FFN and smaller scopes are ablations.

The action adapter is a separately named/loadable artifact. Native Omni
generation and appearance editing must run without loading it. In mixed-task
training, `native_replay` keeps action LoRA on but omits motion-plan tokens, so
ordinary Omni edit pairs provide a real retention gradient. The engineering-
only `native_isolation_probe` omits the plan and sets LoRA exactly to zero; it
verifies the unadapted path and must not be counted as replay supervision.

### Preview data is rejected by default

`ActionLatentDataset` rejects any row with `preview_only=true` unless an
explicit exploratory flag is supplied. The override does not authorize
production training and must propagate `preview_only` to every run receipt and
checkpoint.

Setting `preview_only=false` is not sufficient to authorize a row. A
non-preview provenance record must also be training-authorized,
training-allowed, production-eligible and post-video accepted, and must bind
both a signed-release file and its successful verification receipt by
SHA-256. The receipt is a closed, parsed record whose `sample_id`, release
digest and canonical accepted release-row digest must all match the provenance
and current manifest row. Missing, changed, cross-sample or self-inconsistent
release artifacts fail before payload loading.

The latent loader validates the closed output of a trusted upstream signature
verifier; it does not establish that verifier's public-key/trust root itself.
The future production materializer must invoke the project release verifier
before emitting this receipt. The current materializer is preview-only and
cannot manufacture a production-authorized row.

## Preview manifest from the current v17 run

The builder joins only fully committed per-IID artifacts and verifies their
declared hashes. It selects the first exploratory cohort with one dynamic
actor, locked source/target camera, `preserve_static`, and high-confidence
Qwen census/plan.

```bash
python tools/build_action_preview_manifest.py \
  --qwen-passed-dir "$V17_RUN/qwen_next1000_v17/passed" \
  --wan-root "$V17_RUN/wan_next1000_v17" \
  --natural-root "$V17_RUN/natural_motion_instruction_v5_20260804T222100Z" \
  --instruction-source natural \
  --output-manifest /runs/marp_preview/preview_manifest.jsonl \
  --summary-output /runs/marp_preview/summary.json
```

The output is deliberately marked:

```text
preview_only=true
training_authorized=false
training_use_forbidden=true
production_eligible=false
post_video_acceptance=pending
```

This tool does not approve a generated video and does not produce VAE, Qwen,
UMT5, or motion-token payloads. A production materializer must consume a
separate post-video accepted and signed release.

Counts under the Wan root may change while upstream generation is active, so do
not hard-code the number of joined preview rows in an experiment config.

## Offline action payload

`ActionLatentDataset` accepts the closed
`omnivideo2-action-latents-v1` payload:

```python
{
    "format": "omnivideo2-action-latents-v1",
    "sample_id": str,
    "encoder_contract": dict,
    "source_latent": Tensor[16, T, H, W],
    "target_latent": Tensor[16, T, H, W],
    "text_context": Tensor[L_text, 4096],
    "source_vlm_context": Tensor[L_source, 2048],
    "target_motion_tokens": Tensor[K, 2048],
    "task_type": str,
    "preview_only": bool,
}
```

The manifest binds both payload and provenance sidecar by SHA-256 and must
agree on `sample_id`, `task_type`, and `preview_only`. The sidecar in turn
binds its payload, source/target/I0 media, preview row, instruction, target
caption, motion-only teacher text, encoder checkpoints/contracts, preprocessing
and tensor digests. Payloads are CPU-only, finite, safely loaded with
`weights_only=True`, and neither payload nor provenance can escape its root
through absolute paths, `..`, or symlinks. A batch cannot mix encoder
contracts. `target_motion_tokens_usage` must be exactly `planner_loss_only`.

No mask, tube, track, source-erasure tensor, or spliced target is part of this
schema.

## Real materialization requirements

For the v17 81-frame, 25-FPS pairs, the default materializer keeps every frame
at fixed indices `0,1,...,80` and retains 25 FPS. This is important because the
Omni/Wan renderer receives no physical-FPS condition: a 41-frame stride-2 input
is not model-equivalent to the complete sequence even if the saved container
has the same endpoint span. The old `0,2,...,80` path is available only through
the explicit `smoke_41_12p5fps` profile.

Temporal and spatial choices are independent. `full_480p` uses the
480×832/832×480 buckets; `motion_384p` uses 384×640/640×384 while retaining all
81 frames. The former pairs with the 9,216-context full-resolution config; the
latter is a 6,144-context spatial ablation and never enables temporal
subsampling implicitly.

Source and target can have different original resolutions. They use the same
orientation/bucket policy and the same deterministic center-crop/resize
algorithm. Both processed videos, not only target, are overwritten at frame
zero by the same resized lossless float32 I0, clamped to the VAE contract
`[-1,1]`. The default fails if any source/target/I0 crop retains less than
80% of its input pixels.

The current 108-row AUH preview audit found no orientation mismatches, but the
minimum/median/90th-percentile joint crop retention is approximately
0.603/0.729/0.953; only 22 rows meet the 0.8 gate. Consequently the initial
real overfit cohort is 16--22 rows, not all 108. The first encoder probe uses
IID `a0b66487ab68498a` (about 0.978 retention).

The real materializer must bind:

- Wan VAE checkpoint, normalization, posterior mode, frame indices and crop;
- Qwen checkpoint, motion prompt, feature layer and token resampler;
- UMT5 checkpoint, target-caption/instruction order and padding;
- source/target media and conditioning-frame hashes;
- preview or production authorization state.

Natural-v5 is the editing instruction. It is not the target caption. The
materializer predicts the target caption through the official source-aware
Qwen reasoner. Separately, target video is converted to a strict three-line
motion-only record, then encoded with text-only Qwen and deterministically
pooled to `K×2048`; those tokens are planner labels only.

The first feasibility job keeps all 81 renderer/VAE frames but asks the Qwen
video processor for six sampled frames. This is recorded in the encoder
contract and is not claimed to be an adequate motion teacher. A Qwen-frame
coverage ablation is required before quality training because short action
atoms can fall between those samples.

For the first AUH probe, run one Python process across four visible MI210s.
Qwen uses `balanced_low_0`, UMT5/VAE use `cuda:0`, and flash attention is off
until the installed version is independently qualified:

```bash
python tools/materialize_action_payloads.py \
  --preview-manifest "$PREVIEW" \
  --output-dir "$OUTPUT" \
  --sample-id a0b66487ab68498a \
  --omni-root "$OMNI" \
  --qwen-checkpoint "$QWEN" \
  --vae-checkpoint "$WANROOT/Wan2.1_VAE.pth" \
  --umt5-checkpoint "$WANROOT/models_t5_umt5-xxl-enc-bf16.pth" \
  --umt5-tokenizer "$WANROOT/google/umt5-xxl" \
  --qwen-device-map balanced_low_0 \
  --t5-device cuda:0 \
  --temporal-mode full_81_25fps \
  --spatial-profile full_480p \
  --no-qwen-flash-attention \
  --min-crop-retention 0.8 \
  --allow-preview-exploration
```

The historical 41-frame probe for `a0b66487ab68498a` produced source/target
latents `[16,11,104,60]`, source Qwen features `[128,2048]`, and planner labels
`[16,2048]`. A subsequent four-rank one-step run kept all 4,290 source visual
tokens (`compressed=false`) and gave nonzero LoRA/planner gradients. It remains
an engineering record, not the current training default.

This sample also exposes the next data gate: the target/Qwen motion record
clearly covers lowering the hands but does not reliably cover every requested
torso/hip-motion atom. Materialization success cannot replace post-video
execution review or motion-teacher coverage review.

## Training and DDP expectations

The action trainer is expected to run under `torchrun`/RCCL with a
`DistributedSampler`, one model replica per GPU, synchronized failure handling,
rank-zero-only artifact publication, and `no_sync` during non-final gradient
accumulation microsteps.

Only the following tensors may be trainable:

```text
action LoRA FP32 master weights
motion planner FP32 weights
```

The base transformer runs in BF16. Checkpoints must be adapter-only and bind
the exact official transformer, special tokens, config, manifest, encoder
contract, world size, preview status, source revision, and step/RNG state.

When the end-to-end entry and synthetic fixture are present, the intended smoke
sequence is:

```bash
python tools/build_action_synthetic_fixture.py --output-dir /runs/marp_fixture

torchrun --standalone --nproc_per_node=4 train_omnivideo2_action.py \
  --config /runs/marp_fixture/configs/marp_one_step.json \
  --manifest /runs/marp_fixture/manifest.jsonl \
  --payload-root /runs/marp_fixture/payloads \
  --omnivideo-root ../Omni-Video \
  --checkpoint-dir /models/OmniVideo2-1.3B \
  --output-dir /runs/marp_ddp_one_step \
  --allow-preview-exploration
```

Do not treat this command as complete until every referenced entry exists and
its unit/integration tests pass. The production default must omit
`--allow-preview-exploration`.

## Tests

The mask-free unit tests do not require model downloads:

```bash
KMP_INIT_AT_FORK=FALSE OMP_NUM_THREADS=1 \
python -m unittest \
  tests.test_action_config \
  tests.test_action_context \
  tests.test_action_dataset \
  tests.test_action_flow \
  tests.test_action_omni \
  tests.test_action_planner \
  tests.test_action_runtime \
  tests.test_action_task_gate \
  tests.test_action_synthetic_fixture \
  tests.test_build_action_preview_manifest \
  tests.test_materialize_action_payloads \
  tests.test_verify_action_real_ddp_smoke -v
```

The important negative tests are as valuable as the happy path:

- config objects reject unknown or missing fields;
- payload/manifest/provenance schemas reject extra mask/tube fields and
  content-level tensor/text digest mismatches;
- preview rows are rejected without explicit exploratory authorization;
- non-preview rows reject forged authorization flags and missing/changed
  signed-release or verification-receipt files;
- target motion tokens are detached planner labels;
- planner forward accepts no target representation;
- full-target loss supervises every latent element;
- source token budgeting preserves exact latent frame zero;
- mainline context budgeting rejects any source temporal compression;
- checkpoint loading rejects a changed official model or special token file.

## Legacy PACT boundary

The old tube-first implementation, configs, tests and receipts remain for
reproducibility under `pact/` and the historical PACT tools. The MARP renderer
and payload schema never import mask/tube objects. For checkpoint compatibility,
the current code does reuse audited encoder-contract constants from
`pact.dataset` and `LoRALinear` from `pact.lora`; these are shared utilities,
not spatial conditions, and should be extracted into a neutral module in a
later cleanup.

The PACT design is withdrawn from the deployable path. Mask/tube components may
be used only as explicitly labeled oracle diagnostics with a different
condition budget. They must not be reported as the deployable source+text
method.

AUH job `119646` remains evidence that the historical 300-module PACT path
could take one synthetic official-checkpoint optimizer step and strictly reload
its adapters. It does not validate MARP, real payloads, multi-GPU DDP, action
quality, tracking, or decoded video inference.

## Evidence and licensing boundaries

Until a real-payload overfit and held-out evaluation are complete, the only
supported claims are about contracts, checkpoint connectivity, distributed
execution, finite loss/gradients, and artifact provenance. Do not claim action
success, local preservation, convergence, or native-prior retention from a
synthetic smoke.

The Hugging Face metadata for the weights states Apache-2.0, while the checked
upstream GitHub snapshot historically lacked the license file referenced by
its README. Confirm upstream code and derivative-weight terms before
redistributing trained adapters.
