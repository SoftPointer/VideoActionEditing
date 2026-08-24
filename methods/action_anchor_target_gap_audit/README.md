# MEV action-anchor vs frozen-base target-gap audit

This package answers one deliberately narrow question on 16 clean MEV test
pairs: given the same frozen Bernini-R 1.3B model, content caption, seed,
81-frame geometry, and 40-step sampler, is a text-to-video self-generated
action anchor or a full-source-conditioned frozen RV2V output closer to the
real adjacent target **in action**?

The selected instruction and generation caption are assembled from the
original `mev.json` annotations stored in the paired metadata. Video decoding
is not used to invent the instruction. The protected MEV source tree is read
only; generated videos and all receipts live under a separate experiment root.

## Superseded v1 protocol

Qwen3-VL-32B is the primary judge. It sees chronological rows for source,
real target, Candidate A, and Candidate B, and scores action semantics,
temporal order, and completion separately. A/B is swapped in a second pass.
The per-pass action score is the minimum of those three axes, so appearance or
one strong action axis cannot compensate for wrong order or non-completion. A
sample winner is accepted only when both slot orders agree.

Official SemanticMoments with frozen DINOv2-B is computed for M1, M2, M3,
M23, and M123. Its result is diagnostic only: temporal moments are invariant to
frame permutation and reversal and therefore cannot establish action direction
or completion.

This protocol and its HTML are retained as failure-analysis evidence, but its
candidate winners are invalid. The joint four-row contact sheet leaked source
and target layout/identity into action judgment, and sparse still frames caused
Qwen to hallucinate transitions such as a full spin, phone pickup/placement,
and clap. SemanticMoments was also dominated by orderless appearance/pose
similarity. Do not use the v1 winner counts as an experimental conclusion.

## Corrected v2 calibration protocol

`manual_action_contracts_v2.json` freezes the user's full-video labels and
decomposes every requested action into required visible transitions and
forbidden source-like substitutes. This 16-sample run is explicitly a
calibration/resubstitution audit, not an independent test set.

`corrected_eval.py` gives Qwen one anonymous native video at a time. Source and
real target are never shown beside an anchor or frozen-base candidate. The
first stage is instruction-free: Qwen records a fixed 12-checkpoint trace of
orientation, pose, hands/objects, and continuity without knowing the requested
action. The second stage sees only that trace (never the video or candidate
role) and checks every required predicate, completion, observability, and every
forbidden behavior with a noncompensatory minimum. Candidate evaluation is
repeated with the predicate list reversed, and a winner is accepted only when
both passes agree. If both candidates have the same strict minimum gate, their
mean atomic evidence coverage is used only as a relative tie-break; it cannot
turn an incomplete action into a strict pass. The same two-stage judge
separately receives forward real target, source no-op, reversed target, and
three-block-shuffled target as temporal controls.

InternVideo2-CLIP-S replaces SemanticMoments only as a calibrated auxiliary
diagnostic. It compares candidate-video affinity to target-action text against
source-action text. It is admitted for candidate ranking only if the forward
target beats source no-op, reversed target, and shuffled target on at least
12/16 samples by a preregistered margin; otherwise it is reported as rejected
diagnostic evidence and cannot vote on the candidate winner.

## Source-residue-aware v3 calibration

`source_residue_eval.py` reuses the frozen instruction-free v2 traces; no video
is decoded or viewed again. Only after a trace is frozen does Qwen receive the
v2 target-only judgment and, in a separate prompt, the verbatim MEV source-event
caption plus one manually specified harmful source-behavior predicate. The
source-only judge never receives the target instruction, candidate role, or any
comparison video. Its dedicated system prompt says that there is no video in
this stage, predicate ids are opaque, mirror ambiguity must follow the predicate
definition, and identity/scene/layout/camera/clothing/object presence or the
initial pose never establish residue. It also makes temporal quantifiers hard:
for an any-occurrence predicate, one clear matching checkpoint permanently
triggers the violation even if the subject later stops or completes the target;
for a persistence predicate, one initial pose is insufficient. Harmful source
persistence/replay is appended to the independently frozen target components as
an inverse noncompensatory gate: visible residue gives 0, uncertainty 2, and no
residue 4. Thus target evidence cannot compensate for retaining or replaying the
source action.

`representation_eval.py` adds two further control-gated diagnostics:

- official VideoPrism-LvT-B evaluates
  `cos(video, target-action-text) - cos(video, source-action-text)`;
- official V-JEPA2 ViT-L spatially pools its ordered temporal tokens, removes
  their time-constant component, and concatenates directed residuals at strides
  1, 2, and 4. Candidate-to-real-target similarity is contrasted against
  candidate-to-source similarity. A global mean-token baseline is reported
  separately.

Every metric is denied candidate voting unless real-target forward exceeds
source no-op, reversed target, and shuffled target by `0.005` on at least 12 of
16 pairs. Results from a rejected metric remain visible as diagnostics but do
not enter the final candidate decision.

On AUH, one Qwen3-VL-32B instance uses four MI210 cards but has a large host
memory loading peak, so run at most one instance per node. The resumable shard
entry point is:

```bash
ROCR_VISIBLE_DEVICES=0,1,2,3 \
  methods/action_anchor_target_gap_audit/run_auh_source_residue_qwen_v3_shard.sh \
  SHARD_INDEX NUM_SHARDS
```

Different nodes may run disjoint shards concurrently. Do not start two model
loads on the same node; even disjoint GPU sets can exceed the node's host RAM.

## Main commands

```bash
python -m methods.action_anchor_target_gap_audit.audit build-manifest \
  --metadata /path/to/paired_training_candidates.jsonl \
  --selection methods/action_anchor_target_gap_audit/selected_pair_prefixes.json \
  --experiment-root /path/to/fresh/experiment \
  --output /path/to/fresh/experiment/manifest.json

python -m unittest discover \
  -s methods/action_anchor_target_gap_audit/tests -v
```

`run_auh_pipeline.sbatch` performs dual-4-GPU generation, dual-4-GPU Qwen
evaluation, eight-way SemanticMoments extraction, and final aggregation. The
real target path appears only in evaluator inputs, never in a generation
command.

`run_auh_review_html.sbatch` builds a portable offline review bundle after the
evaluation. It retimes source, real target, T2V anchor, and frozen-base RV2V to
the same review-only 81-frame/25-fps timeline, then writes an HTML page with
per-sample synchronized play, seek, frame-step, and speed controls. All derived
media is written outside the protected MEV tree and does not change evaluation
inputs or scores.

`run_auh_corrected_eval.sbatch` runs the v2 control construction, native-video
Qwen audit, and pinned official InternVideo2-CLIP-S calibration. Set
`ACTION_GAP_V2_MODE=smoke` for the single `81533c9e56ec` spin canary; use
`ACTION_GAP_V2_MODE=full` only after that output is reviewed.

If a frozen action contract is corrected after discovering a coordinate or
layout confound, rerun only that pair and use `merge-qwen-repair` to atomically
replace its complete eight-record packet in the finished shard. The merger
rejects partial repairs, duplicate pair/role/pass keys, and record-count drift.
Long shards can also use `qwen-evaluate --resume`: existing records are keyed
by pair, role, and pass; out-of-shard records, duplicate keys, excess passes,
or changed video digests are rejected before missing roles are generated.
`ACTION_GAP_V2_MODE=qwen-repair` accepts a comma-separated list in
`ACTION_GAP_V2_REPAIR_PREFIXES`, evaluates all requested pairs in one model
load, and writes a standalone repair packet for the atomic merger.
For contract-only changes, prefer `ACTION_GAP_V2_MODE=qwen-rescore`: it freezes
and validates the original instruction-free 12-checkpoint trace, reruns only
the trace-to-contract text judgment, and records the previous prompt/output
digests and observation. This isolates contract correction from a new visual
inference sample.
