# Self-Predicted Phase-Transport LoRA (SPT-v2)

SPT-v2 is an independent Bernini prototype. It does not modify or replace the
CDF-v1 files.

Its runtime inputs are only the complete source video and edit instruction.
The source's 21 clean latent phases form an immutable token bank. A small
student planner predicts dense `(dt,dy,dx)` retrieval offsets and a three-way
`preserve / transport / generate` gate. The default `phase_query_v2` planner
keeps the full unpadded T5 sequence: 21 learned queries plus fixed sinusoidal
phase timestamps cross-attend to all text tokens in two layers. Their states
FiLM-modulate a source volume with explicit normalized `(t,y,x)` channels.
Bernini supplies only the generate candidate. The resulting clean estimate is
projected back to raw flow velocity before integration, so training and
inference operate on the same quantity.

The paired-latent oracle is deliberately modest: local soft matching among
source latent candidates. It uses no SAM, optical-flow model, pose estimator,
track, mask, or first-frame anchor. It is legal only for training and oracle
diagnostics. The student planner's Python signature accepts `source` and
`instruction_tokens`; there is no target argument.

The main-path oracle also applies a conservative innovation budget. Before
budgeting it identifies every unmatched generate candidate, ranks candidates
independently in each `(batch, phase)` plane by
`min(zero_cost, best_nonzero_cost)`, and retains at most
`floor(0.12 * H * W)` cells. Every rejected generate candidate becomes
`preserve`; it is never relabelled as transport. Receipts and audit reports
record pre-budget generate coverage, post-budget coverage, rejected coverage,
and the observed per-phase maximum. An unbounded generate teacher exists only
behind an explicit diagnostic-ablation flag and cannot enter the default
training path.

## Falsification order

1. Run `oracle_diagnostic.py` on held-out pairs. Reject SPT if the proxy cannot
   improve target reconstruction over copying while keeping the generate gate
   sparse.
2. Run oracle execution through the sampler. This tests the executor separately
   from text-to-plan prediction.
3. Train the small planner plus a conservative Bernini `q/out` LoRA and compare
   oracle-plan with student-plan outputs.
4. Run exact no-op, wrong-instruction, instruction-strength, identity, temporal
   consistency, and action-success tests on held-out videos.

## Current integration boundary

The source/target latent geometry and loss are implemented. `SPTTrainingHead`
returns a differentiable loss for joint optimization with LoRA parameters. A
production AUH trainer must save the planner state beside the PEFT adapter and
must insert both parameter groups into distributed gradient reduction. CDF's
existing trainer cannot be reused unchanged because it intentionally rejects
all non-LoRA trainable parameters and saves only PEFT state. Silently hiding the
planner inside `modules_to_save` would weaken its exact-scope audit, so v2 needs
an explicit checkpoint schema.

## Planner-only training

`train_student.py` implements the first complete training stage. Bernini is
frozen and contributes only its pinned tokenizer/T5 instruction embedding.
Each optimizer step builds an action oracle from the local source/target VAE
pair, calls the student with source+instruction tokens only, and also distills
an exact preserve plan for the no-op instruction. Four-rank launches are true
replicated data parallel: rank `r` consumes
`(global_step * world_size + r) % training_rows`, then gradients are explicitly
averaged. Ulysses is disabled for this planner-only stage. The paired oracle
uses all 64 packed latent channels. Action gate distillation independently
normalizes every teacher-present class before averaging, so a sparse generate
region is not diluted by the much larger preserve region; no-op calibration
keeps ordinary preserve cross-entropy. Checkpoints contain `planner.safetensors`,
`planner_config.json`, `optimizer.pt`, and a digest-bound `receipt.json`; both
architecture identity and teacher budget are immutable, so the former globally pooled planner
checkpoint cannot be resumed as `phase_query_v2`.

AUH launches use `scripts/auh_train_student.sbatch`. Required environment
variables are `BERNINI_SPT_SOURCE_ROOT`, `BERNINI_OFFICIAL_ROOT`,
`BERNINI_VEOMNI_ROOT`, `BERNINI_ACTION_CHECKPOINT`,
`BERNINI_ACTION_PARQUET_DIR`, `BERNINI_ACTION_DATASET_SUMMARY`, and
`BERNINI_SPT_TRAIN_OUTPUT`. `BERNINI_SPT_MAX_STEPS`,
`BERNINI_SPT_SAVE_EVERY`, and `BERNINI_SPT_RESUME` are optional. A diagnostic
overfit may set `BERNINI_SPT_TRAIN_PREFIX_ROWS=8`; the default remains the full
validated dataset. Receipts record the full row count and every selected
`iid`/identity hash, and label a strict prefix as `diagnostic_subset=true`.
