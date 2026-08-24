# Qwen3-VL-32B audit policy

## Objective

Adjacent events are continuous, but continuity does not imply editability. A
common failure is a source in which a subject moves from state A to state B,
followed by a target that starts at B. Such a target depends on the completed
source outcome and cannot represent an edited video beginning from source state
`S0`.

The audit uniformly samples eight ordered frames per clip:

```text
SOURCE: S0 -> S1 -> ... -> S7
TARGET: T0 -> T1 -> ... -> T7
```

The model evaluates:

1. whether `S0` and `T0` align in identity, object state/ownership, spatial
   state, scene, and camera;
2. whether `T0` aligns mainly with `Sn`, indicating source-outcome dependency;
3. whether the source starts, prepares, or enables the target action;
4. whether the target contains visible temporal action rather than a static
   endpoint or residual consequence;
5. whether identity, scene, and camera are preserved sufficiently for a
   source-conditioned edit.

The model must first publish `source_initial_state`, `source_final_state`, and
`target_initial_state`, then derive `target_initial_matches` and
`source_enables_target`. `source_end_only` or `source_enables_target=yes`
forces strict dependency and rejection.

The reverse consistency rule is also enforced. A strict label without source-
end, enabling, or shifted-state evidence is normalized from the structured
state fields. If all acceptance fields pass but the free-form verdict says
reject, the verdict is deterministically normalized to accept. Every change is
recorded in `audit_normalizations`; raw model attempts remain intact.

When `source_state_change_class=none`, `S0=Sn`; therefore
`target_initial_matches=source_end_only`, `source_enables_target=yes`, and
`shifted_by_source_outcome` are impossible. The implementation normalizes this
case to `both/no/aligned` before deriving dependency and verdict.

## Fail-closed behavior

- Invalid schema is retried at most once; a second failure creates an error
  terminal receipt.
- Insufficient visual evidence produces `uncertain`, never automatic accept.
- Strict dependency must reject.
- Cross-field acceptance requirements are revalidated in code.
- Each pair publishes a create-only result and SHA-bound terminal receipt.
- Qwen evidence is a pseudo-label, not a human review or calibrated probability.

## Runtime topology

The smoke profile uses one eight-MI210 node split into two persistent four-card
workers. The full profile uses four such nodes and eight global workers. Rows
are assigned by worker rank with a deterministic stride, and every pair writes
its own immutable output rather than appending to a shared file.

The smoke gate requires eight valid terminal receipts before full submission.
Explicit reject and uncertain rows are valid full-run outcomes; only missing
terminals and I/O/model/schema errors are engineering failures.
