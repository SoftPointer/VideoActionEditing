# SemanticMoments action-reward audit

This directory audits the official SemanticMoments implementation as a motion
reward candidate for dual-anchor video action editing.  The audit is
read-only with respect to the generator/editor: it extracts frozen DINOv2
features, evaluates held-out video pairs, and writes diagnostic receipts.

The experiment has three data layers:

1. SimMotion-Real's 40 human-curated `ref/positive/negative` triplets;
2. the project's exact-60 SAIC self-generated `forward/reverse/noop` bank;
3. exact frame-order counterfactuals (reverse, shuffle, truncation/hold) for all
   videos, including the two audited owner action anchors.

SimMotion-Real is reported both as the weak triplet-only preference test and
with the official repository's stricter retrieval rule: the positive must beat
the paired negative and every video from the other 39 triplets (119 total
non-query candidates).  The audit does not claim the paper's Kinetics-400
external-distractor protocol.

The official moment formula is loaded directly from
`semantic_moments/embedders/base.py`.  Each extracted video must match a local
recomposition of the official `(alpha1, alpha2, alpha3) = (1, 8, 4)` embedding
within numerical tolerance.  M1, M2, M3, M23, and M123 are reported separately
to expose appearance leakage and the contribution of higher moments.

The AUH validation uses a locally frozen Hugging Face DINOv2-B/14 checkpoint
(768 dimensions) for network-independent reproducibility.  This is a formula
and reward-suitability audit, not a reproduction of the repository's default
DINOv2-L/14-register headline configuration.

Commands:

```bash
python run_audit.py build-manifest ...
python run_audit.py extract-shard --shard-index 0 --num-shards 8 ...
python run_audit.py analyze ...
```

The output has no authority to select training data, create preference pairs,
or update a model.  Project branch names are generation contracts, not human
event labels; they are reported as a project-domain stress test rather than
ground truth.
