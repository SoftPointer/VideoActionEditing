# Action-editing shared-8 baselines

This directory runs Lucy, Bernini-R 1.3B, and OmniVideo2-1.3B on one frozen
eight-case source-video-plus-instruction contract. The input manifest has no
target, mask, track, pose, trajectory, reference-media, or shared-I0 field.
Ground-truth targets live in a separate reference manifest and are available
only to post-generation evaluation.

The cases reuse the prior Lucy action experiment's five held-out test rows and
three validation-extension rows. They are IID-disjoint from both the earlier
Lucy project training manifests and the 644-pair Bernini training membership
(`0/8` overlap, frozen in `manifests/goku_legacy_shared8_exposure.json`). They
are useful for an engineering comparison because the earlier Lucy
`e1_plain_lora` outputs and checkpoint are already hash-audited. They are not
production-accepted data, the split is not content-disjoint, the labels were
not human accepted, and eight examples cannot establish scientific superiority.

The primary comparison is intentionally explicit about unequal adaptation:

- `lucy_official_base`: the official Lucy-Edit 1.1 Dev editing checkpoint,
  without a project action adapter;
- `bernini_full644_lora_step644`: the completed 644-pair Bernini attention-LoRA
  run, adapter SHA-256
  `9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774`;
- `omnivideo2_official_base`: the official 1.3B editing checkpoint. The MARP
  checkpoints are one-step plumbing canaries and must not be reported as a
  trained quality baseline.

The existing `lucy_e1_action_lora_step100` outputs may be attached only as an
auxiliary diagnostic. That 64-pair, 100-step adapter has checkpoint SHA-256
`ea29c242fd970868cb0aa1f4bb59ef464bbbbd04d6ac744a1fb14188c08446ab`
and was previously judged `0/8` for action correctness; it is not labeled as
the official Lucy baseline.

All generated videos must contain exactly 81 frames and be encoded at 25 FPS.
The same per-case seed is used across model families. Samplers and spatial
geometry remain model-native and are recorded rather than falsely presented as
identical: Lucy and Bernini preserve the source aspect ratio within their
native pixel budgets, and OmniVideo2 uses its native 832x480 or 480x832 bucket.
Lucy buckets both pixel dimensions to multiples of 32: its expanded-timestep
path and Wan patch embed disagree on odd latent grids even though the current
Diffusers input check only requires a multiple of 16.
Lucy uses the official BF16 pipeline with the model-card FP32 VAE. OmniVideo2
must account for all 8,190 visual source tokens inside a fixed 9,216-token
context and refuses the official entry point's silent truncation behavior.

AUH launchers are under `scripts/`. `run_shared8.py` loads one row by array
index, refuses manifests containing privileged fields, invokes only the chosen
model's source-only entry point, validates the output, and writes an atomic
receipt. `finalize_shared8.py` later checks all model receipts and joins the
separate references for evaluation without making them inference inputs.

The completed 2026-08-05 AUH run included a blinded review, per-case findings,
and explicit claim limits. Generated review notes and machine-readable audit
receipts are intentionally omitted from this code-focused repository.
