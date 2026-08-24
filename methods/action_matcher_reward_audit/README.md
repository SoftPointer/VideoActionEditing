# Order-aware action matcher reward audit

This directory audits sequence matchers as candidate evaluators for dual-anchor
action editing.  It does not authorize a training reward.

Evidence levels are kept separate:

- `run_sequence_audit.py` operates on frozen DINO frame sequences.  Its
  `otam_style_centered` score is an algorithmic relaxed-boundary diagnostic,
  not an official trained OTAM checkpoint.
- TRX has an author repository but no directly published checkpoint in its
  README.  No randomly initialized TRX result may be reported as TRX efficacy.
- TEAM has an author repository and public pretrained checkpoints; its official
  SSV2-Small ViT 1-shot checkpoint is evaluated by the separate TEAM runner.

The objective open-set protocol calibrates a threshold on a hash-defined fit
split and evaluates it on disjoint videos.  Order-preserving speed warps are
positives; reverse, shuffle, no-op, and incomplete/tail-hold variants are
negatives.  SimMotion labels and project branch names are reported only as
their original weak authorities, never as action correctness truth.

The fit-only controlled threshold is also applied unchanged to the real
cross-video pairs.  This reports `both_accepted`, `neither_accepted`, and the
two one-sided outcomes explicitly, so a forced pairwise winner cannot be
mistaken for an absolute same-action decision.

`operational_reward.py` is the concrete fast-path prototype that follows from
the audit.  It is a group-relative Best-of-N reranker, not an open-set
same-action classifier.  It combines SemanticMoments M3 as a motion-set axis
with phase-aligned DINO order, forward-vs-reverse contrast, and activity
matching.  Required axes are aggregated by their minimum within-pool
percentile, with explicit hard gates and abstention.  There is no VLM fallback:
an abstained group produces zero training update and is kept/resampled at
inference.

`run_operational_reward_audit.py` replays that exact contract on the frozen
182-video feature population.  `select_training_pair` only exposes raw-axis
Pareto-dominant pairs, so a large score on one event axis cannot compensate for
a regression on another.  Source preservation remains a separate hard
partial order in `methods/bernini_action_editing/saic_rollout_preference_set_v1.py`.
