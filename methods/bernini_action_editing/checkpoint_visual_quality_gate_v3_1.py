#!/usr/bin/env python3
"""Conservative v3.1 confirmation policy for the checkpoint quality gate.

V3.1 is an independent evidence schema.  It does not read, rewrite, or
upgrade v3 reports.  It deliberately reuses v3's all-frame, two-scale feature
extractor, then applies a new decision policy that prevents a single
resolution-dependent measurement from becoming a hard artifact claim.

In particular, neither a chroma high-pass excess at only one scale nor a
base-salient tile-retention loss at only one scale can hard-fail a sample.
Hard artifact families require either agreement of the same artifact evidence
at both analysis scales or two genuinely different artifact evidence kinds.
Unsupported base-relative structural divergence remains ``unresolved``.  The
three outcomes remain ``pass``, ``unresolved``, and ``fail``; only ``pass`` is
publishable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import checkpoint_visual_quality_gate_v3 as v3


SCHEMA_VERSION = "bernini-checkpoint-visual-quality-gate-v3.1"
FEATURE_EXTRACTOR_SCHEMA_VERSION = v3.SCHEMA_VERSION
FEATURE_EXTRACTOR_TOOL_SHA256 = v3._file_sha256(
    Path(v3.__file__).resolve(strict=True)
)
TOOL_SHA256 = v3._file_sha256(Path(__file__).resolve(strict=True))
EXPECTED_FRAME_COUNT = v3.EXPECTED_FRAME_COUNT
ANALYSIS_SCALES = v3.ANALYSIS_SCALES
QualityGateError = v3.QualityGateError
QualityThresholds = v3.QualityThresholds
THRESHOLDS = v3.THRESHOLDS
decode_video_exact81_multiscale = v3.decode_video_exact81_multiscale
write_json_atomic = v3.write_json_atomic


def _evidence_count(evidence: Mapping[str, bool]) -> int:
    return sum(bool(value) for value in evidence.values())


def _cross_scale_evidence(
    per_scale: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    evidence_key: str,
) -> list[str]:
    scales = list(per_scale)
    if len(scales) != len(ANALYSIS_SCALES):
        raise QualityGateError("v3.1 decision requires both analysis scales")
    names = set(per_scale[scales[0]][family][evidence_key])
    for scale in scales[1:]:
        names &= set(per_scale[scale][family][evidence_key])
    return sorted(
        name
        for name in names
        if all(
            bool(per_scale[scale][family][evidence_key].get(name))
            for scale in scales
        )
    )


def _family_row(
    per_scale: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    cross_scale_evidence: Sequence[str],
    any_raw: bool,
) -> dict[str, Any]:
    same_scale_confirmed = [
        scale
        for scale, decisions in per_scale.items()
        if bool(decisions[family]["triggered"])
    ]
    triggered = bool(same_scale_confirmed or cross_scale_evidence)
    unresolved_scales = [
        scale
        for scale, decisions in per_scale.items()
        if bool(decisions[family].get("raw_candidate_triggered"))
        and not bool(decisions[family]["triggered"])
    ]
    return {
        "triggered": triggered,
        "triggered_scales": same_scale_confirmed,
        "cross_scale_confirmed_evidence": list(cross_scale_evidence),
        "unresolved": bool(any_raw and not triggered),
        "unresolved_scales": unresolved_scales if not triggered else [],
        "confirmation_rule": (
            "same evidence kind at both scales OR at least two independent "
            "evidence kinds at one scale"
        ),
        "per_scale": {
            scale: decisions[family] for scale, decisions in per_scale.items()
        },
    }


def _decision(scale_features: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply v3.1's cross-scale/independent-evidence confirmation policy."""

    expected_keys = {
        v3._scale_key(width, height) for width, height in ANALYSIS_SCALES
    }
    if set(scale_features) != expected_keys:
        raise QualityGateError(
            f"v3.1 decision scales must be exactly {sorted(expected_keys)}"
        )

    per_scale: dict[str, dict[str, Any]] = {}
    for key, features in scale_features.items():
        values = features["scalars"]

        low_kurtosis_and_flat_spectrum = bool(
            values["hp_kurtosis_median"] < THRESHOLDS.hp_kurtosis_max
            and values["spectral_flatness_median"]
            < THRESHOLDS.spectral_flatness_max
        )
        hp_energy_excess = bool(
            values["frame_hp_rms_retention_to_base_p10"] > 1.15
        )
        strong_hp_energy_excess = bool(
            values["frame_hp_rms_retention_to_base_p10"] > 1.50
        )
        chroma_hp_excess = bool(
            values["chroma_hp_ratio_to_base"]
            > THRESHOLDS.chroma_hp_ratio_max
        )
        temporal_residual_excess = bool(
            values["motion_compensated_temporal_residual_ratio_to_base"]
            > THRESHOLDS.motion_compensated_residual_ratio_max
        )
        noise_evidence = {
            # Luma distribution plus energy is one spatial evidence kind.
            "distributional_luma_noise": bool(
                low_kurtosis_and_flat_spectrum and hp_energy_excess
            ),
            # Chroma is independent of the luma-distribution measurement.
            "chroma_highpass_excess": chroma_hp_excess,
            # Temporal residual plus strong luma HP is a temporal/spatial kind.
            "temporal_residual_with_strong_luma_excess": bool(
                temporal_residual_excess and strong_hp_energy_excess
            ),
        }
        noise_raw = bool(any(noise_evidence.values()) or temporal_residual_excess)
        noise_confirmed_here = _evidence_count(noise_evidence) >= 2

        blur_evidence = {
            "global_frequency_loss": bool(
                values["frame_hp_rms_retention_to_base_p10"]
                < THRESHOLDS.frame_hp_retention_p10_min
                and values["frame_laplacian_var_retention_to_base_p10"]
                < THRESHOLDS.frame_laplacian_var_retention_p10_min
            ),
            "salient_edge_retention_loss": bool(
                values["base_salient_low_retention_frame_fraction"]
                >= THRESHOLDS.salient_bad_frame_fraction
            ),
        }
        blur_raw = bool(any(blur_evidence.values()))
        blur_confirmed_here = _evidence_count(blur_evidence) >= 2

        low_windowed = bool(
            values["candidate_base_windowed_ssim_mean"]
            < THRESHOLDS.windowed_ssim_to_base_min
        )
        low_global_edge = bool(
            values["candidate_base_global_ssim_mean"]
            < THRESHOLDS.global_ssim_to_base_min
            and values["candidate_base_edge_correlation_mean"]
            < THRESHOLDS.edge_correlation_to_base_min
        )
        structure_raw_evidence = {
            "low_windowed_ssim": low_windowed,
            "low_global_ssim_and_edge_correlation": low_global_edge,
        }
        structure_raw = bool(any(structure_raw_evidence.values()))
        structure_support_evidence = {
            "salient_structure_retention_loss": bool(
                values["structure_support_low_retention_frame_fraction"]
                >= THRESHOLDS.structure_support_bad_frame_fraction
            ),
            "confirmed_noise_artifact": noise_confirmed_here,
            "confirmed_blur_artifact": blur_confirmed_here,
        }
        # Base-relative SSIM/edge metrics are mutually correlated and are not
        # counted as independent artifact support.  Two separate support kinds
        # are required for a one-scale route-off hard claim.
        structure_confirmed_here = bool(
            structure_raw and _evidence_count(structure_support_evidence) >= 2
        )

        duplicate_prevalence = bool(
            values["candidate_near_duplicate_transition_fraction"]
            >= THRESHOLDS.near_duplicate_fraction_min
        )
        duplicate_excess = bool(
            values["near_duplicate_fraction_excess_over_base"]
            >= THRESHOLDS.near_duplicate_excess_over_base_min
        )
        freeze_evidence = {
            "near_duplicate_prevalence": duplicate_prevalence,
            "near_duplicate_excess_over_base": duplicate_excess,
        }
        # The excess term contains candidate prevalence, so it is not treated
        # as an independent second kind at a single scale.  Cross-scale
        # agreement below is required for a hard freeze claim.
        freeze_raw = bool(any(freeze_evidence.values()))

        per_scale[key] = {
            "NOISE": {
                "triggered": noise_confirmed_here,
                "raw_candidate_triggered": noise_raw,
                "unresolved": bool(noise_raw and not noise_confirmed_here),
                "independent_evidence_count": _evidence_count(noise_evidence),
                "artifact_evidence": noise_evidence,
                "raw_conditions": {
                    "low_kurtosis_and_low_spectral_flatness": low_kurtosis_and_flat_spectrum,
                    "distribution_has_hp_excess_over_base": hp_energy_excess,
                    "strong_hp_excess_over_base": strong_hp_energy_excess,
                    "chroma_hp_ratio_above_1p50": chroma_hp_excess,
                    "motion_compensated_residual_ratio_above_1p80": temporal_residual_excess,
                },
            },
            "BLUR": {
                "triggered": blur_confirmed_here,
                "raw_candidate_triggered": blur_raw,
                "unresolved": bool(blur_raw and not blur_confirmed_here),
                "independent_evidence_count": _evidence_count(blur_evidence),
                "artifact_evidence": blur_evidence,
                "raw_conditions": {
                    "p10_hp_below_0p55_and_laplacian_below_0p25": blur_evidence[
                        "global_frequency_loss"
                    ],
                    "salient_tile_bad_frame_fraction_at_least_0p20": blur_evidence[
                        "salient_edge_retention_loss"
                    ],
                },
            },
            "ROUTEOFF_STRUCTURE": {
                "triggered": structure_confirmed_here,
                "raw_candidate_triggered": structure_raw,
                "unresolved": bool(structure_raw and not structure_confirmed_here),
                "independent_evidence_count": _evidence_count(
                    structure_support_evidence
                ),
                "raw_structure_evidence": structure_raw_evidence,
                "artifact_support_evidence": structure_support_evidence,
                "base_structure_reference_eligible": bool(
                    values["base_source_global_ssim_mean"]
                    >= THRESHOLDS.base_source_global_ssim_for_structure_reference_min
                ),
                "raw_conditions": {
                    "windowed_ssim_below_0p70": low_windowed,
                    "global_ssim_below_0p85_and_edge_corr_below_0p50": low_global_edge,
                },
            },
            "FREEZE": {
                "triggered": False,
                "raw_candidate_triggered": freeze_raw,
                "unresolved": freeze_raw,
                "independent_evidence_count": 1 if duplicate_prevalence and duplicate_excess else 0,
                "artifact_evidence": freeze_evidence,
                "raw_conditions": {
                    "candidate_near_duplicate_fraction_at_least_0p90": duplicate_prevalence,
                    "near_duplicate_excess_over_base_at_least_0p50": duplicate_excess,
                },
            },
        }

    noise_cross = _cross_scale_evidence(
        per_scale, family="NOISE", evidence_key="artifact_evidence"
    )
    blur_cross = _cross_scale_evidence(
        per_scale, family="BLUR", evidence_key="artifact_evidence"
    )
    freeze_cross_raw = _cross_scale_evidence(
        per_scale, family="FREEZE", evidence_key="artifact_evidence"
    )
    freeze_cross = (
        ["near_duplicate_prevalence_and_excess_over_base"]
        if {
            "near_duplicate_prevalence",
            "near_duplicate_excess_over_base",
        }.issubset(freeze_cross_raw)
        else []
    )

    # Route-off structure is only cross-scale confirmed when both raw
    # structure divergence and the *same genuine artifact support* persist at
    # both scales.  Merely repeating low SSIM at two scales is not an artifact:
    # a legitimate large edit repeats that difference too.
    structure_raw_both = all(
        bool(decisions["ROUTEOFF_STRUCTURE"]["raw_candidate_triggered"])
        for decisions in per_scale.values()
    )
    structure_support_cross = _cross_scale_evidence(
        per_scale,
        family="ROUTEOFF_STRUCTURE",
        evidence_key="artifact_support_evidence",
    )
    routeoff_cross = (
        structure_support_cross if structure_raw_both else []
    )

    raw_by_family = {
        family: any(
            bool(decisions[family]["raw_candidate_triggered"])
            for decisions in per_scale.values()
        )
        for family in ("NOISE", "BLUR", "ROUTEOFF_STRUCTURE", "FREEZE")
    }
    families = {
        "NOISE": _family_row(
            per_scale,
            family="NOISE",
            cross_scale_evidence=noise_cross,
            any_raw=raw_by_family["NOISE"],
        ),
        "BLUR": _family_row(
            per_scale,
            family="BLUR",
            cross_scale_evidence=blur_cross,
            any_raw=raw_by_family["BLUR"],
        ),
        "ROUTEOFF_STRUCTURE": _family_row(
            per_scale,
            family="ROUTEOFF_STRUCTURE",
            cross_scale_evidence=routeoff_cross,
            any_raw=raw_by_family["ROUTEOFF_STRUCTURE"],
        ),
        "FREEZE": _family_row(
            per_scale,
            family="FREEZE",
            cross_scale_evidence=freeze_cross,
            any_raw=raw_by_family["FREEZE"],
        ),
    }

    failure_codes = [
        f"quality_{name.lower()}"
        for name, row in families.items()
        if row["triggered"]
    ]
    unresolved_codes = [
        f"quality_{name.lower()}_requires_external_verifier"
        for name, row in families.items()
        if row["unresolved"] and not row["triggered"]
    ]
    if failure_codes:
        outcome = "fail"
    elif unresolved_codes:
        outcome = "unresolved"
    else:
        outcome = "pass"
    return {
        "outcome": outcome,
        "passed": outcome == "pass",
        "hard_artifact_failure": outcome == "fail",
        "unresolved": outcome == "unresolved",
        "failure_codes": failure_codes,
        "unresolved_codes": unresolved_codes,
        "decision_rule": (
            "confirmed(NOISE OR BLUR OR ROUTEOFF_STRUCTURE OR FREEZE); "
            "confirmation requires cross-scale agreement or two independent "
            "artifact evidence kinds"
        ),
        "family_combination": "non-compensating OR after family confirmation",
        "single_scale_chroma_hp_alone_can_hard_fail": False,
        "single_scale_salient_retention_alone_can_hard_fail": False,
        "evidence_families": families,
    }


def evaluate_visual_quality(
    source_frames_by_scale: Mapping[str, Any],
    candidate_frames_by_scale: Mapping[str, Any],
    *,
    frozen_base_frames_by_scale: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate one aligned source/candidate/base triplet under v3.1."""

    prefix = {
        "schema_version": SCHEMA_VERSION,
        "tool_sha256": TOOL_SHA256,
        "feature_extractor_schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
        "feature_extractor_tool_sha256": FEATURE_EXTRACTOR_TOOL_SHA256,
        "fail_closed": True,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "required_analysis_scales": [
            {"width": width, "height": height}
            for width, height in ANALYSIS_SCALES
        ],
        "thresholds": asdict(THRESHOLDS),
        "confirmation_policy": {
            "cross_scale_or_two_independent_evidence_required": True,
            "single_scale_chroma_hp_alone_hard_fail_forbidden": True,
            "single_scale_salient_retention_alone_hard_fail_forbidden": True,
            "unsupported_base_relative_structure_is_unresolved": True,
        },
        "metadata": v3._json_safe(metadata or {}),
    }
    try:
        inputs = v3._validate_scale_maps(
            source_frames_by_scale,
            candidate_frames_by_scale,
            frozen_base_frames_by_scale,
        )
        features = {}
        for key, (source, candidate, frozen_base) in inputs.items():
            features[key] = v3._scale_features(
                source, candidate, frozen_base, scale_key=key
            )
        decision = _decision(features)
    except (QualityGateError, TypeError, ValueError, FloatingPointError) as error:
        return {
            **prefix,
            "status": "error",
            "passed": False,
            "publishable": False,
            "input_contract_passed": False,
            "failure_codes": ["quality_gate_input_contract_violation"],
            "error": str(error),
        }

    return v3._json_safe(
        {
            **prefix,
            "status": decision["outcome"],
            "passed": decision["passed"],
            "publishable": decision["passed"],
            "hard_artifact_failure": decision["hard_artifact_failure"],
            "unresolved": decision["unresolved"],
            "input_contract_passed": True,
            "failure_codes": decision["failure_codes"],
            "unresolved_codes": decision["unresolved_codes"],
            "decision": decision,
            "features": {
                "metric_scope": (
                    "all 81 frames and all 80 adjacent transitions at both scales"
                ),
                "all_frames_evaluated": True,
                "all_transitions_evaluated": True,
                "evaluated_frame_count_per_scale": EXPECTED_FRAME_COUNT,
                "evaluated_transition_count_per_scale": EXPECTED_FRAME_COUNT - 1,
                "scales": features,
            },
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--checkpoint-label")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        source, source_identity = decode_video_exact81_multiscale(
            args.source, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        candidate, candidate_identity = decode_video_exact81_multiscale(
            args.candidate, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        frozen_base, base_identity = decode_video_exact81_multiscale(
            args.frozen_base, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
        )
        report = evaluate_visual_quality(
            source,
            candidate,
            frozen_base_frames_by_scale=frozen_base,
            metadata={
                "sample_id": args.sample_id,
                "checkpoint_step": args.checkpoint_step,
                "checkpoint_label": args.checkpoint_label
                or f"checkpoint-{args.checkpoint_step:08d}",
                "inputs": {
                    "source": source_identity,
                    "candidate": candidate_identity,
                    "frozen_base": base_identity,
                },
            },
        )
        write_json_atomic(args.output, report)
        return 0 if report["passed"] else 2
    except (QualityGateError, OSError, ValueError) as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "tool_sha256": TOOL_SHA256,
            "feature_extractor_schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
            "feature_extractor_tool_sha256": FEATURE_EXTRACTOR_TOOL_SHA256,
            "status": "error",
            "passed": False,
            "publishable": False,
            "fail_closed": True,
            "input_contract_passed": False,
            "failure_codes": ["quality_gate_runtime_error"],
            "metadata": {
                "sample_id": args.sample_id,
                "checkpoint_step": args.checkpoint_step,
                "checkpoint_label": args.checkpoint_label,
            },
            "error": str(error),
        }
        write_json_atomic(args.output, report)
        return 2


__all__ = [
    "ANALYSIS_SCALES",
    "EXPECTED_FRAME_COUNT",
    "FEATURE_EXTRACTOR_SCHEMA_VERSION",
    "FEATURE_EXTRACTOR_TOOL_SHA256",
    "QualityGateError",
    "QualityThresholds",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "TOOL_SHA256",
    "decode_video_exact81_multiscale",
    "evaluate_visual_quality",
    "write_json_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
