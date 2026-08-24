#!/usr/bin/env python3
"""Fresh v15 routed-teacher training with a same-caption dynamic/static anchor.

This entry point deliberately reuses the audited v1 optimizer, distributed,
Q/K-route, and checkpoint machinery.  It replaces only the experiment contract:
the frozen self-generated donor is contrasted against its phase-zero-tiled
static state under the same caption, RNG seed, scheduler, and timestep.

The resulting receipts authorize an engineering experiment only.  They do not
authorize a scientific claim or treat donor RGB/latents as flow-matching targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_v1 as base


METHOD = "bernini-online-anchor-dynamic-static-routed-teacher-v15"
RECEIPT_SCHEMA = "bernini-online-anchor-dynamic-static-routed-teacher-receipt-v15"
OBJECTIVE = "real_source_target_owned_routed_teacher_delta_v14r2"
ROUTE_OPERATOR = "self_target_owned_activity_kernel25_v14r2"
REPLAY_COMBINE_MODE = "action_priority_pcgrad_010"
NOISE_RECOVERY_ATOL = 2.0e-5
ALLOWED_STEPS = (2, 8, 32)


_BASE_VALIDATE_ARGS = base.validate_args
_BASE_CHECKPOINT_RECEIPT = base.checkpoint_receipt
_BASE_LOAD_MANIFEST = base.pairs.load_manifest


def _empty_runtime_audit() -> dict[str, Any]:
    return {
        "batch_pair_count": 0,
        "same_caption_tokenization_all": True,
        "same_timestep_tensor_all": True,
        "same_noise_rng_seed_all": True,
        "phase0_tiled_static_all": True,
        "dynamic_post_phase0_difference_all": True,
        "recovered_gaussian_fp32_max_abs_error": 0.0,
        "manifest_training_order": None,
        "manifest_ordered_iids": (),
        "target_iids": set(),
        "target_events": set(),
        "donor_iids": set(),
        "donor_events": set(),
    }


_RUNTIME_AUDIT = _empty_runtime_audit()


def fail(message: str) -> None:
    base.fail(message)


def build_parser() -> argparse.ArgumentParser:
    return base.build_parser()


def validate_args(args: argparse.Namespace) -> None:
    """Apply all audited base gates, changing only action/noop to dynamic/static."""

    if args.profile != "dynamic_static":
        fail("v15 requires the dynamic_static anchor profile")

    # The base validator rejects routed teachers for every profile except
    # action_noop.  Validate a shadow namespace to retain every other gate,
    # then enforce the complete v15 profile below.
    shadow = argparse.Namespace(**vars(args))
    shadow.profile = "action_noop"
    _BASE_VALIDATE_ARGS(shadow)

    exact = {
        "training_objective": OBJECTIVE,
        "route_operator": ROUTE_OPERATOR,
        "routed_teacher_mode": "same_action_route_only",
        "training_interface": "first_phase_caption_i2v",
        "teacher_delta_mode": "raw",
        "source_variant": "not_applicable",
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "source_reconstruction_prompt": "action",
    }
    for name, expected in exact.items():
        if getattr(args, name) != expected:
            fail(f"v15 requires --{name.replace('_', '-')}={expected}")
    if args.max_steps not in ALLOWED_STEPS:
        fail("v15 staged training permits max-steps 2, 8, or 32 only")
    if float(args.route_strength) != 0.25:
        fail("v15 requires student route strength 0.25")
    if float(args.teacher_route_strength) != 0.50:
        fail("v15 requires teacher route strength 0.50")
    if float(args.paired_target_fm_weight) != 0.0:
        fail("v15 forbids synthetic target flow matching")
    if float(args.source_reconstruction_weight) != 0.025:
        fail("v15 requires the audited base replay scale 0.025")
    if float(args.learning_rate) != 1.0e-5:
        fail("v15 requires learning rate 1e-5")
    if bool(args.gradient_diagnostic_only):
        fail("v15 S2/S8/S32 runs are training, not diagnostic-only runs")

    checkpoint_parts = {part.lower() for part in Path(args.checkpoint).parts}
    if any(part.startswith("checkpoint-") or part.startswith("train_") for part in checkpoint_parts):
        fail("v15 must start from the frozen base checkpoint, not a prior adapter output")
    if "v15" not in str(Path(args.output)).lower():
        fail("v15 output path must carry an explicit v15 namespace")


def load_manifest_event_interleaved_v15(path: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Put one variant from every event before revisiting an event.

    The upstream manifest is event-major.  Since the optimizer selects
    ``rows[global_step % 32]``, that ordering makes an eight-step run see only
    two events.  v15 uses v0/e00..e07, then v1/e00..e07, and so on.
    """

    manifest, rows = _BASE_LOAD_MANIFEST(path)
    event_order: list[str] = []
    registry: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        event_id = row.get("event_id")
        variant_id = row.get("variant_id")
        iid = row.get("iid")
        if not all(isinstance(value, str) and value for value in (event_id, variant_id, iid)):
            fail("v15 manifest row identity is incomplete")
        if event_id not in event_order:
            event_order.append(event_id)
        key = (event_id, variant_id)
        if key in registry:
            fail("v15 manifest event/variant row is duplicated")
        registry[key] = row
    variants = ("v0", "v1", "v2", "v3")
    if len(event_order) != 8 or len(rows) != 32 or set(registry) != {
        (event_id, variant_id)
        for event_id in event_order
        for variant_id in variants
    }:
        fail("v15 requires exact Complex8 x four-variant manifest closure")
    reordered = [
        registry[(event_id, variant_id)]
        for variant_id in variants
        for event_id in event_order
    ]
    first_eight_events = [str(row["event_id"]) for row in reordered[:8]]
    if first_eight_events != event_order or len(set(first_eight_events)) != 8:
        fail("v15 event-interleaved manifest prefix differs")
    _RUNTIME_AUDIT["manifest_training_order"] = "variant_major_event_interleaved_v15"
    _RUNTIME_AUDIT["manifest_ordered_iids"] = tuple(str(row["iid"]) for row in reordered)
    return manifest, reordered


def _require_same_text_condition(action: Mapping[str, Any], contrast: Mapping[str, Any]) -> None:
    import torch

    for field in ("input_ids", "attention_mask", "t5_input_lens"):
        left = action.get(field)
        right = contrast.get(field)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            fail(f"v15 anchor text field is absent: {field}")
        if not torch.equal(left, right):
            fail(f"v15 dynamic/static anchor changed caption tokenization: {field}")


def _target_velocity_spatial(
    batch: Mapping[str, Any], *, spatial_shape: Sequence[int], label: str
) -> Any:
    import torch

    velocity = batch.get("target_velocity")
    if not isinstance(velocity, torch.Tensor):
        fail(f"v15 {label} target velocity is absent")
    try:
        result = base.data.patches_to_spatial(
            velocity, spatial_shape=spatial_shape
        ).float()
    except Exception as error:
        raise base.OnlineAnchorTrainingError(
            f"v15 {label} target velocity geometry differs"
        ) from error
    if not bool(torch.isfinite(result).all().item()):
        fail(f"v15 {label} target velocity is non-finite")
    return result


def build_anchor_batches(
    *,
    target_row: Mapping[str, Any],
    donor: Mapping[str, Any],
    profile: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    captions: Optional[Mapping[tuple[str, str], Mapping[str, str]]] = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, ...]]:
    """Build and audit the v15 same-caption dynamic/static donor contrast."""

    import torch

    if profile != "dynamic_static":
        fail("v15 anchor builder rejects every profile except dynamic_static")
    if captions is None:
        fail("v15 requires full-scene caption-I2V authoring")

    dynamic, _, _, _, _ = base.pairs.load_row_tensors(donor)
    if (
        not isinstance(dynamic, torch.Tensor)
        or dynamic.ndim != 5
        or int(dynamic.shape[0]) != 1
        or int(dynamic.shape[1]) != 16
        or int(dynamic.shape[2]) != 21
        or not bool(torch.isfinite(dynamic).all().item())
    ):
        fail("v15 dynamic donor must be one finite 16x21 latent trajectory")
    dynamic = dynamic.contiguous()
    static = dynamic[:, :, :1].expand_as(dynamic).clone().contiguous()
    if not torch.equal(static[:, :, 0], dynamic[:, :, 0]):
        fail("v15 static contrast changed the donor-owned phase zero")
    if not torch.equal(static, static[:, :, :1].expand_as(static)):
        fail("v15 static contrast is not an exact phase-zero tile")
    if torch.equal(dynamic[:, :, 1:], static[:, :, 1:]):
        fail("v15 donor has no post-phase-zero dynamic/static state contrast")

    key = (str(donor["event_id"]), str(donor["variant_id"]))
    try:
        action_prompt = captions[key]["target"]
    except (KeyError, TypeError) as error:
        raise base.OnlineAnchorTrainingError(
            "v15 donor action caption is absent"
        ) from error
    if not isinstance(action_prompt, str) or not action_prompt.strip():
        fail("v15 donor action caption must be non-empty")

    # Both calls deliberately receive the identical caption and integer seed.
    # build_transform reseeds immediately before the one shared scheduler path.
    action = transform(
        base.data.make_sample(
            instruction=action_prompt,
            source_blob=None,
            target_blob=base._blob(dynamic, mean, std),
        ),
        seed,
    )
    contrast = transform(
        base.data.make_sample(
            instruction=action_prompt,
            source_blob=None,
            target_blob=base._blob(static, mean, std),
        ),
        seed,
    )

    _require_same_text_condition(action, contrast)
    action_t = action.get("timesteps")
    contrast_t = contrast.get("timesteps")
    if (
        not isinstance(action_t, torch.Tensor)
        or not isinstance(contrast_t, torch.Tensor)
        or not torch.equal(action_t, contrast_t)
    ):
        fail("v15 dynamic/static anchor did not share the exact timestep tensor")

    shape = tuple(map(int, dynamic.shape))
    action_velocity = _target_velocity_spatial(
        action, spatial_shape=shape, label="dynamic"
    )
    contrast_velocity = _target_velocity_spatial(
        contrast, spatial_shape=shape, label="static"
    )
    action_clean = dynamic.to(
        device=action_velocity.device, dtype=action_velocity.dtype
    )
    contrast_clean = static.to(
        device=contrast_velocity.device, dtype=contrast_velocity.dtype
    )
    # Bernini flow-matching target_velocity is epsilon - x_clean.  Recovering
    # epsilon on both branches catches any accidental RNG/scheduler divergence.
    action_noise = action_velocity + action_clean
    contrast_noise = contrast_velocity + contrast_clean
    noise_error = float(
        (action_noise - contrast_noise).abs().max().detach().cpu().item()
    )
    if not torch.allclose(
        action_noise, contrast_noise, rtol=0.0, atol=NOISE_RECOVERY_ATOL
    ):
        fail(
            "v15 dynamic/static anchor did not recover the same Gaussian noise "
            f"(max_abs_error={noise_error:.9g})"
        )

    _RUNTIME_AUDIT["batch_pair_count"] += 1
    _RUNTIME_AUDIT["target_iids"].add(str(target_row["iid"]))
    _RUNTIME_AUDIT["target_events"].add(str(target_row["event_id"]))
    _RUNTIME_AUDIT["donor_iids"].add(str(donor["iid"]))
    _RUNTIME_AUDIT["donor_events"].add(str(donor["event_id"]))
    _RUNTIME_AUDIT["recovered_gaussian_fp32_max_abs_error"] = max(
        float(_RUNTIME_AUDIT["recovered_gaussian_fp32_max_abs_error"]),
        noise_error,
    )
    return action, contrast, shape


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    """Extend the audited base receipt with the v15 state-contrast contract."""

    receipt = _BASE_CHECKPOINT_RECEIPT(**kwargs)
    if int(_RUNTIME_AUDIT["batch_pair_count"]) <= 0:
        fail("v15 cannot checkpoint before a dynamic/static pair was audited")
    if _RUNTIME_AUDIT["manifest_training_order"] != "variant_major_event_interleaved_v15":
        fail("v15 cannot checkpoint without the event-interleaved manifest order")
    contract = receipt.get("training_contract")
    if not isinstance(contract, dict):
        fail("v15 base checkpoint receipt has no mutable training contract")

    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_held_out_evaluation"
    )
    contract.update(
        {
            "method": METHOD,
            "profile": "dynamic_static",
            "anchor_contrast_profile_is_state_not_caption": True,
            "anchor_action_caption_equals_static_caption": True,
            "caption_difference_used_as_anchor_supervision": False,
            "anchor_dynamic_clean_state": "self_generated_full_dynamic_t2v_latent",
            "anchor_static_clean_state": "same_donor_phase0_tiled_over_21_latent_phases",
            "anchor_phase0_is_exactly_donor_owned_on_both_branches": True,
            "anchor_dynamic_and_static_post_phase0_clean_states_differ": True,
            "anchor_dynamic_and_static_share_exact_noise_rng_seed_scheduler_and_timestep": True,
            "anchor_recovered_gaussian_agrees_within_fp32_tolerance": True,
            "anchor_recovered_gaussian_fp32_tolerance": NOISE_RECOVERY_ATOL,
            "anchor_recovered_gaussian_fp32_max_abs_error": float(
                _RUNTIME_AUDIT["recovered_gaussian_fp32_max_abs_error"]
            ),
            "anchor_dynamic_static_pairs_audited": int(
                _RUNTIME_AUDIT["batch_pair_count"]
            ),
            "training_manifest_order": _RUNTIME_AUDIT["manifest_training_order"],
            "training_manifest_ordered_iids_sha256": base.legacy.object_sha256(
                list(_RUNTIME_AUDIT["manifest_ordered_iids"])
            ),
            "actual_distinct_target_iid_count": len(_RUNTIME_AUDIT["target_iids"]),
            "actual_distinct_target_iids": sorted(_RUNTIME_AUDIT["target_iids"]),
            "actual_distinct_target_event_count": len(_RUNTIME_AUDIT["target_events"]),
            "actual_distinct_target_events": sorted(_RUNTIME_AUDIT["target_events"]),
            "actual_distinct_cross_appearance_donor_iid_count": len(
                _RUNTIME_AUDIT["donor_iids"]
            ),
            "actual_distinct_cross_appearance_donor_iids": sorted(
                _RUNTIME_AUDIT["donor_iids"]
            ),
            "actual_distinct_cross_appearance_donor_event_count": len(
                _RUNTIME_AUDIT["donor_events"]
            ),
            "anchor_qk_support_uses_phase0_relative_action_noop_contrast": False,
            "anchor_qk_support_uses_phase0_relative_same_caption_dynamic_static_contrast": True,
            "self_generated_intermediate_supervision": (
                "detached_frozen_donor_qk_temporal_route_support"
            ),
            "self_generated_rgb_or_latent_used_as_flow_matching_target": False,
            "student_supervision_is_target_coordinate_routed_teacher_delta": True,
            "starts_from_frozen_base_checkpoint_not_prior_adapter": True,
            "scientific_claim_authorized": False,
            "claim_scope": (
                "engineering_training_run_only_non_scientific_until_held_out_evaluation"
            ),
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT

    _RUNTIME_AUDIT = _empty_runtime_audit()
    original_validate = base.validate_args
    original_builder = base.build_anchor_batches
    original_receipt = base.checkpoint_receipt
    original_load_manifest = base.pairs.load_manifest
    base.validate_args = validate_args
    base.build_anchor_batches = build_anchor_batches
    base.checkpoint_receipt = checkpoint_receipt
    base.pairs.load_manifest = load_manifest_event_interleaved_v15
    try:
        return base.main(argv)
    finally:
        base.validate_args = original_validate
        base.build_anchor_batches = original_builder
        base.checkpoint_receipt = original_receipt
        base.pairs.load_manifest = original_load_manifest


if __name__ == "__main__":
    raise SystemExit(main())
