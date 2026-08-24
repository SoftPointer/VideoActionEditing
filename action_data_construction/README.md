# MEV to video action-editing data construction

This directory converts the read-only MEV multi-event dataset into two forms
of action-editing metadata. Semantic authority comes from
`annotations/mev.json`; decoded video is used only to determine whether a pair
is visually usable. Source videos are referenced by absolute path and are not
copied, cropped, transcoded, linked, or modified.

## Modes

### Paired adjacent-event candidates

For each UUID, the pipeline constructs `(event_i, event_{i+1})` pairs. The
target event caption from `mev.json` is wrapped by a deterministic template to
form the editing instruction. Qwen cannot overwrite it.

Because the target is a real continuation rather than a counterfactual filmed
from the source initial state, every pair is labeled `continuation-derived`.
The deterministic layer rejects missing/short/low-quality media and unsafe
source-state transitions. Qwen3-VL-32B then compares source `S0 -> Sn` with
target `T0 -> Tn`. A pair is retained only when the initial states align, the
target action is visible, identity/scene/camera are sufficiently preserved,
and the target does not strictly depend on the completed source outcome.

### Event-1 no-target source pool

The row with `event_id=1` for each UUID is written to
`metadata/no_target_sources.jsonl` with `target=null`. These rows may be used
for self-generated action anchors, representation learning, or source
preservation. They must not enter flow-matching SFT without a qualified target.

## Outputs

Candidate construction publishes:

- `metadata/raw_paired_candidates.jsonl`
- `metadata/rule_rejected_pairs.jsonl`
- `metadata/qwen_audit_queue.jsonl`
- `metadata/qwen_smoke8.jsonl`
- `metadata/no_target_sources.jsonl`
- `metadata/build_summary.json`

Each Qwen run writes create-only per-pair results and terminal receipts under
`runs/<run-id>/`. Finalization publishes:

- `final_metadata/paired_training_candidates.jsonl`
- `final_metadata/paired_training_candidates.csv`
- `final_metadata/paired_qwen_rejected.jsonl`
- `final_metadata/paired_uncertain_review.jsonl`
- `final_metadata/finalization_summary.json`

The v2 accepted state is
`qwen-visual-accepted-annotation-instruction-pending-human`. It is an
automatically screened candidate, not a two-reviewer SFT authority.

## Annotation metadata

- `metadata_annotation_v2/paired_annotation_semantics.jsonl`: full global
  prompt, source/target event annotations, deterministic instruction, and SHA
  provenance for 17,246 adjacent pairs.
- `metadata_annotation_v2/no_target_sources_annotation_v2.jsonl`: annotation
  metadata for 8,015 event-1 sources.
- `metadata_annotation_v2/annotation_extraction_summary.json`: source
  `mev.json` identity and output digests.
- `runs/<full-run>/final_metadata_annotation_v2/`: visually accepted pairs
  whose instruction authority remains the annotation caption.

## Quick start

```bash
export PROJECT=/path/to/VideoActionEditing/action_data_construction
export SOURCE=/path/to/read-only/MEV

python "$PROJECT/build_candidates.py" \
  --source-root "$SOURCE" \
  --output-root /path/to/action-data-output \
  --smoke-count 8
```

For the cluster Qwen audit, set `MEV_AUDIT_OUTPUT_ROOT` to a writable run
directory and review the scheduler directives before submission:

```bash
export MEV_AUDIT_PHASE=smoke
export MEV_AUDIT_OUTPUT_ROOT=/path/to/action-data-output/runs/smoke_<UTC>
sbatch --export=ALL "$PROJECT/slurm/qwen_audit_8gpu.sbatch"
```

Training loaders read `source_video_path` and `target_video_path` directly.
The primary `instruction` comes from the target event caption. A Qwen wording
proposal is retained only as `non_authoritative_qwen_instruction_proposal`.

See [the dataset contract](docs/DATASET_CONTRACT.md),
[annotation policy](docs/ANNOTATION_SEMANTICS_POLICY.md),
[Qwen audit policy](docs/QWEN_AUDIT_POLICY.md),
[operations guide](docs/OPERATIONS.md), and
[full-v5 receipt](docs/RUN_20260817_FULL_V5.md).
