# Dataset contract

## 1. Source and immutability boundary

- Source and output roots must not overlap.
- Source videos are not copied, linked, cropped, transcoded, or modified.
- Metadata references `MEV/videos/<original_filename>`.
- `build_summary.json` binds the `events.csv` SHA-256 and records the media
  inventory as `relative_name + size + mtime_ns`.
- GPU jobs verify the same inventory before and after execution.
- Annotation v2 binds the original `annotations/mev.json` SHA-256 directly.

The audited source contained 25,262 events. Expanded captions/global prompts
from `events.csv` matched `mev.json`, but semantic provenance still points to
the original JSON rather than the derived CSV.

## 2. Split isolation

Splits are assigned by UUID with:

```text
sha256("mev-action-edit-split-v1\\0" + uuid)
```

The mapping is deterministic 90/5/5 train/validation/test. Every paired and
no-target row from one UUID must remain in the same split; events, adjacent
pairs, and generated seeds cannot cross splits.

## 3. Paired candidates

The primitive unit is a consecutive `(event_i, event_{i+1})` pair from one
UUID. Its semantic class is permanently `continuation-derived` because the
target is a real next event, not a counterfactual recorded from the source
initial state.

The deterministic layer rejects a pair when:

- media are missing or event IDs are not consecutive;
- either clip is shorter than one second or longer than twenty seconds;
- conservative VBench consistency, flicker, smoothness, or imaging-quality
  thresholds fail;
- the source clip contains subject appearance/disappearance or lighting change.

Environment, focus, occlusion, camera, and state-change words are advisory and
cannot automatically accept or reject a pair. Every remaining pair requires a
visual audit.

Qwen acceptance requires:

- `target_initial_matches` in `{source_start,both}`;
- a valid `source_state_change_class`;
- `source_enables_target=no`;
- `initial_state_compatibility=aligned`;
- `dependency_level` in `{none,weak}`;
- `target_action_quality=clear_action`;
- `preservation` in `{same_identity_scene_camera,minor_change}`;
- `confidence` in `{medium,high}`;
- a non-unknown visible target action.

Retained rows still declare:

```text
is_strict_counterfactual_ground_truth=false
semantic_truth_class=continuation-derived
qualification_status=qwen-auto-accepted-pending-human
training_use=sft_candidate_pending_human_qualification
```

This is a high-precision candidate pool, not a claim of real counterfactual
ground truth. Formal training gates may still require a human-review receipt.

## 4. No-target sources

Only an explicit `event_id=1` may represent a UUID. A UUID without event 1 is
written to `source_anomalies.jsonl`; the smallest observed event ID is not used
as an implicit replacement.

Allowed uses:

- self-generated action anchors;
- action-representation contrastive learning;
- source preservation and semantic no-op training.

Forbidden use: flow-matching SFT without a qualified target. Multiple generated
seeds or anchors do not create independent source rows.

## 5. Instruction authority

The instruction is generated deterministically from the target event caption:

```text
Edit the action so that <target caption as a lower-case clause>.
```

It does not infer a new action from video or add semantics absent from the
annotation. The full global prompt, source event, target event, JSON pointer,
and event digest remain in annotation-v2 metadata. Qwen evaluates visual
usability only; its wording proposal is separate and non-authoritative.

## 6. Training consumption

The CSV is a convenience view.
`final_metadata_annotation_v2/paired_training_candidates.jsonl` is the
authoritative field set. Training should split by UUID, consume only
instructions with `instruction_source=mev.json target event caption`, and make
an explicit policy decision about `pending-human` rows. No-target metadata must
use a separate sampler and must not be packed with paired flow targets.
