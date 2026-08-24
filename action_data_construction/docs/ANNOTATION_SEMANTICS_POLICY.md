# `mev.json` semantic-authority contract

## Principle

The semantic authority for an editing instruction is the target event
`caption` in the read-only source file `MEV/annotations/mev.json`. It is not a
free-form Qwen description of the decoded video.

Qwen is restricted to:

1. checking initial-state compatibility between source `S0 -> Sn` and target
   `T0 -> Tn`;
2. rejecting continuations whose target strictly depends on the completed
   source outcome or preparation action;
3. checking identity, background, camera, and visible temporal action quality;
4. publishing an auditable visual description as evidence.

Qwen's `action_instruction` is retained as
`non_authoritative_qwen_instruction_proposal` and cannot replace the training
row's `instruction`.

## Bound fields

Each pair binds the following data from `mev.json`:

- the full `global_prompt`, including scene, style, shot, camera, lighting,
  subject profile, and short/middle/long captions;
- complete source and target event annotations, including caption, timing,
  filename, state flags, focus object, camera description, and VBench fields;
- source JSON SHA-256, stable JSON pointer, and event-annotation SHA-256;
- source and target action captions;
- the deterministic editing instruction.

The instruction template performs no generation:

```text
target caption: The woman slices a raw chicken breast into strips.
instruction:    Edit the action so that the woman slices a raw chicken breast into strips.
```

It adds no subject, object, action, or state. Any later rewrite must live in a
separate derived field and retain the original caption and receipt.

## Full-v5 compatibility

Existing Qwen runs remain valid for visual pair auditing; their queues and
terminal receipts are not rewritten. `paired_annotation_semantics.jsonl`
joins `mev.json` semantics through the existing pair ID and candidate digest.
The v2 finalizer performs a strict join and writes an independent
`final_metadata_annotation_v2` output. The legacy output in which Qwen wording
was treated as instruction authority is not published as the v2 authority.
