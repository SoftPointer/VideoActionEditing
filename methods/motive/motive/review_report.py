"""Build a calibration report from a human-review-merged JSONL manifest.

This module deliberately treats human ``uncertain`` verdicts as unresolved,
not as negative labels.  Wilson intervals and gate decisions are computed
from unweighted, conclusive human audits only.  Inverse-probability-weighted
point estimates are reported separately only when every audited row carries
strictly validated balanced-sample v1 provenance; binomial Wilson intervals
are never attached to those weighted estimates.

Run as::

    python -m motive.review_report \
        --input human_reviewed.jsonl \
        --output calibration_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPORT_SCHEMA_VERSION = "motive-action-review-report-v1"
REPORT_VERSION = 1
HUMAN_REVIEW_SCHEMA = "motive-action-human-review-v1"
KEEP_PRECISION_LOWER_THRESHOLD = 0.70
REJECT_POSITIVE_CONTAMINATION_UPPER_THRESHOLD = 0.15
MIN_CONCLUSIVE_PER_GATE = 100
MAX_UNCERTAIN_FRACTION = 0.20
WILSON_Z_95 = 1.959963984540054
BALANCED_SAMPLE_SCHEME = "family-tier-capped-round-robin-hash"
BALANCED_SAMPLE_VERSION = "v1"

POSITIVE_VERDICTS = frozenset({"valid_action", "valid_suppression"})
UNCERTAIN_VERDICT = "uncertain"
NEGATIVE_VERDICTS = frozenset(
    {
        "endpoint_only",
        "appearance_only",
        "camera_motion",
        "background_motion",
        "static",
        "instruction_mismatch",
        "artifact",
    }
)
ALL_HUMAN_VERDICTS = (
    POSITIVE_VERDICTS | NEGATIVE_VERDICTS | {UNCERTAIN_VERDICT}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPTIONAL_REVIEW_TEXT_FIELDS = (
    "action_signature",
    "notes",
    "event_type",
    "actor",
    "actor_valid",
    "instruction_aligned",
    "complete_temporal_event",
    "source_action",
    "target_action",
    "direction",
    "speed",
    "phase",
    "contact_or_interaction",
    "camera_motion",
    "preservation_ok",
    "review_confidence",
    "secondary_reviewer",
    "adjudication",
)


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            yield line_number, row


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite intentionally")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_human_review(
    row: Mapping[str, Any],
    *,
    context: str,
) -> str:
    review = row.get("human_review")
    if not isinstance(review, dict):
        raise ValueError(f"{context} human_review must be an object")

    required = {
        "schema_version",
        "verdict",
        "reviewer",
        "label_source_sha256",
    }
    missing = sorted(required - set(review))
    if missing:
        raise ValueError(f"{context} human_review is missing {missing}")
    if review["schema_version"] != HUMAN_REVIEW_SCHEMA:
        raise ValueError(f"{context} has unsupported human_review schema")

    verdict = review["verdict"]
    if not isinstance(verdict, str) or verdict not in ALL_HUMAN_VERDICTS:
        raise ValueError(f"{context} has invalid human_review verdict")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(
            f"{context} human_review reviewer must be a non-empty string"
        )
    label_digest = review["label_source_sha256"]
    if (
        not isinstance(label_digest, str)
        or _SHA256_RE.fullmatch(label_digest) is None
    ):
        raise ValueError(
            f"{context} human_review label_source_sha256 is invalid"
        )

    for field in _OPTIONAL_REVIEW_TEXT_FIELDS:
        if field in review and not isinstance(review[field], str):
            raise ValueError(
                f"{context} human_review {field} must be a string"
            )
    frame_values: dict[str, int | None] = {}
    for field in ("event_start_frame", "event_end_frame"):
        value = review.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{context} human_review {field} must be a "
                "non-negative integer or null"
            )
        frame_values[field] = value
    start = frame_values["event_start_frame"]
    end = frame_values["event_end_frame"]
    if start is not None and end is not None and end < start:
        raise ValueError(
            f"{context} human_review event_end_frame precedes "
            "event_start_frame"
        )
    return verdict


def _human_outcome(verdict: str) -> str:
    if verdict in POSITIVE_VERDICTS:
        return "positive"
    if verdict == UNCERTAIN_VERDICT:
        return "uncertain"
    if verdict in NEGATIVE_VERDICTS:
        return "negative"
    raise AssertionError(f"unvalidated human verdict: {verdict!r}")


def _final_decision(row: Mapping[str, Any], *, context: str) -> str:
    final = row.get("final_triage")
    if final is None:
        rule = row.get("auto_rule")
        if isinstance(rule, dict) and rule.get("tier") == "reject":
            return "rule_reject"
        raise ValueError(
            f"{context} lacks final_triage and is not a rule-stage reject"
        )
    if not isinstance(final, dict):
        raise ValueError(f"{context} final_triage must be an object")
    decision = final.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise ValueError(
            f"{context} final_triage.decision must be a non-empty string"
        )
    return decision.strip()


def _qwen_visual_verdict(row: Mapping[str, Any]) -> str:
    evidence = row.get("qwen_evidence")
    if evidence is None:
        return "not_available"
    if not isinstance(evidence, dict):
        return "malformed_record"
    visual = evidence.get("visual")
    if visual is None:
        return "not_available"
    if not isinstance(visual, dict):
        return "malformed_record"
    status = visual.get("status")
    if status != "ok":
        if isinstance(status, str) and status.strip():
            return f"status:{status.strip()}"
        return "malformed_record"
    result = visual.get("result")
    if not isinstance(result, dict):
        return "malformed_record"
    verdict = result.get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return "malformed_record"
    return verdict.strip()


def _inclusion_probability(
    row: Mapping[str, Any],
    *,
    context: str,
) -> float | None:
    provenance = row.get("sampling_provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        raise ValueError(
            f"{context} sampling_provenance must be an object"
        )
    if "inclusion_probability" not in provenance:
        return None
    probability = provenance["inclusion_probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
    ):
        raise ValueError(
            f"{context} sampling_provenance.inclusion_probability "
            "must be numeric"
        )
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ValueError(
            f"{context} sampling_provenance.inclusion_probability "
            "must satisfy 0 < p <= 1"
        )
    return probability


def _balanced_sample_provenance(
    row: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Classify and validate one balanced-sample provenance record.

    Unknown designs are classified rather than interpreted.  A record that
    explicitly claims the supported v1 design is validated strictly so that a
    bare or self-inconsistent propensity can never enable population IPW.
    """

    provenance = row.get("sampling_provenance")
    probability = _inclusion_probability(row, context=context)
    if provenance is None:
        return {
            "status": "missing_provenance",
            "inclusion_probability": probability,
        }
    # _inclusion_probability has already checked this, but retaining the
    # assertion makes the narrowing below explicit for type checkers/readers.
    if not isinstance(provenance, dict):
        raise AssertionError("unreachable non-object sampling_provenance")

    scheme = provenance.get("scheme")
    if scheme is None:
        return {
            "status": "missing_scheme",
            "inclusion_probability": probability,
        }
    if not isinstance(scheme, str) or scheme != BALANCED_SAMPLE_SCHEME:
        return {
            "status": "unsupported_scheme",
            "scheme": scheme if isinstance(scheme, str) else None,
            "inclusion_probability": probability,
        }

    version = provenance.get("version")
    if version != BALANCED_SAMPLE_VERSION:
        return {
            "status": "unsupported_version",
            "scheme": scheme,
            "version": version if isinstance(version, str) else None,
            "inclusion_probability": probability,
        }

    required = {
        "scheme",
        "version",
        "seed",
        "stratum",
        "stratum_population",
        "stratum_selected",
        "inclusion_probability",
        "inverse_probability_weight",
        "within_stratum_rank",
    }
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError(
            f"{context} supported sampling_provenance is missing {missing}"
        )

    seed = provenance["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"{context} sampling_provenance.seed must be an integer"
        )
    stratum = provenance["stratum"]
    if (
        not isinstance(stratum, str)
        or not stratum
        or stratum != stratum.strip()
    ):
        raise ValueError(
            f"{context} sampling_provenance.stratum must be a canonical "
            "non-empty string"
        )

    population = provenance["stratum_population"]
    selected = provenance["stratum_selected"]
    rank = provenance["within_stratum_rank"]
    for field, value in (
        ("stratum_population", population),
        ("stratum_selected", selected),
        ("within_stratum_rank", rank),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{context} sampling_provenance.{field} must be a "
                "positive integer"
            )
    if selected > population:
        raise ValueError(
            f"{context} sampling_provenance.stratum_selected must not "
            "exceed stratum_population"
        )
    if rank > selected:
        raise ValueError(
            f"{context} sampling_provenance.within_stratum_rank must not "
            "exceed stratum_selected"
        )

    # probability is guaranteed present and valid by the required-field check
    # plus _inclusion_probability above.
    if probability is None:
        raise AssertionError("supported provenance probability is missing")
    expected_probability = selected / population
    if not math.isclose(
        probability,
        expected_probability,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{context} sampling_provenance.inclusion_probability must "
            "equal stratum_selected / stratum_population"
        )

    weight = provenance["inverse_probability_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(
            f"{context} sampling_provenance.inverse_probability_weight "
            "must be numeric"
        )
    weight = float(weight)
    if not math.isfinite(weight) or weight < 1.0:
        raise ValueError(
            f"{context} sampling_provenance.inverse_probability_weight "
            "must be finite and at least one"
        )
    expected_weight = 1.0 / probability
    if not math.isclose(
        weight,
        expected_weight,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{context} sampling_provenance.inverse_probability_weight "
            "must equal 1 / inclusion_probability"
        )

    return {
        "status": "supported_valid",
        "scheme": scheme,
        "version": version,
        "seed": seed,
        "stratum": stratum,
        "stratum_population": population,
        "stratum_selected": selected,
        "inclusion_probability": probability,
        "inverse_probability_weight": weight,
        "within_stratum_rank": rank,
    }


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float] | None:
    """Return the two-sided Wilson score interval, or ``None`` for ``n=0``."""

    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or successes < 0
        or trials < 0
        or successes > trials
    ):
        raise ValueError("successes/trials must satisfy 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be finite and positive")
    if trials == 0:
        return None
    estimate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (estimate + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _unweighted_metric(
    *,
    name: str,
    decision: str,
    outcomes: Mapping[str, int],
    threshold_bound: str,
    threshold_value: float,
) -> dict[str, Any]:
    positive = int(outcomes.get("positive", 0))
    negative = int(outcomes.get("negative", 0))
    uncertain = int(outcomes.get("uncertain", 0))
    conclusive = positive + negative
    audited = conclusive + uncertain
    uncertain_fraction = uncertain / audited if audited else None
    interval = wilson_interval(positive, conclusive)
    estimate = positive / conclusive if conclusive else None
    if (
        interval is None
        or conclusive < MIN_CONCLUSIVE_PER_GATE
        or uncertain_fraction is None
        or uncertain_fraction > MAX_UNCERTAIN_FRACTION
    ):
        assessment = "insufficient"
    elif threshold_bound == "lower":
        assessment = (
            "pass" if interval[0] >= threshold_value else "fail"
        )
    elif threshold_bound == "upper":
        assessment = (
            "pass" if interval[1] <= threshold_value else "fail"
        )
    else:
        raise ValueError(f"unsupported threshold bound: {threshold_bound}")
    return {
        "metric": name,
        "final_triage_decision": decision,
        "numerator_human_positive": positive,
        "denominator_conclusive": conclusive,
        "human_negative": negative,
        "human_uncertain_excluded": uncertain,
        "audited_rows": audited,
        "uncertain_fraction": uncertain_fraction,
        "estimate": estimate,
        "wilson_95_ci": (
            {"lower": interval[0], "upper": interval[1]}
            if interval is not None
            else None
        ),
        "preregistered_threshold": {
            "bound": threshold_bound,
            "operator": ">=" if threshold_bound == "lower" else "<=",
            "value": threshold_value,
        },
        "adequacy_requirements": {
            "minimum_conclusive": MIN_CONCLUSIVE_PER_GATE,
            "maximum_uncertain_fraction": MAX_UNCERTAIN_FRACTION,
        },
        "assessment": assessment,
    }


def _weighted_metric(outcomes: Mapping[str, float]) -> dict[str, Any]:
    positive = float(outcomes.get("positive", 0.0))
    negative = float(outcomes.get("negative", 0.0))
    uncertain = float(outcomes.get("uncertain", 0.0))
    conclusive = positive + negative
    return {
        "ht_positive_total": positive,
        "ht_negative_total": negative,
        "ht_uncertain_total": uncertain,
        "ht_conclusive_total": conclusive,
        "ratio_estimate": positive / conclusive if conclusive else None,
    }


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _sorted_float_counter(
    counter: Mapping[str, float],
) -> dict[str, float]:
    return {key: float(counter[key]) for key in sorted(counter)}


def build_report(input_path: Path) -> dict[str, Any]:
    """Validate and summarize one human-review-merged JSONL file."""

    input_path = input_path.expanduser()
    input_sha256 = _file_digest(input_path)
    seen_iids: set[str] = set()
    human_verdicts: Counter[str] = Counter()
    human_outcomes: Counter[str] = Counter()
    decision_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    cross_verdicts: dict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    cross_outcomes: dict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    weighted_outcomes: Counter[str] = Counter()
    weighted_decision_outcomes: dict[str, Counter[str]] = defaultdict(
        Counter
    )
    provenance_statuses: Counter[str] = Counter()
    supported_provenance_rows = 0
    supported_design_seeds: set[int] = set()
    stratum_configs: dict[str, tuple[int, int, int]] = {}
    stratum_observed: Counter[str] = Counter()
    stratum_observed_ranks: dict[str, set[int]] = defaultdict(set)
    rows_with_probability = 0
    total_rows = 0

    for line_number, row in _iter_jsonl(input_path):
        context = f"{input_path}:{line_number}"
        iid = row.get("iid")
        if not isinstance(iid, str) or not iid.strip():
            raise ValueError(f"{context} iid must be a non-empty string")
        if iid in seen_iids:
            raise ValueError(f"{context} has duplicate iid={iid}")
        seen_iids.add(iid)

        verdict = _validate_human_review(row, context=context)
        outcome = _human_outcome(verdict)
        decision = _final_decision(row, context=context)
        qwen_verdict = _qwen_visual_verdict(row)
        sampling = _balanced_sample_provenance(row, context=context)
        sampling_status = str(sampling["status"])
        probability = sampling.get("inclusion_probability")

        total_rows += 1
        human_verdicts[verdict] += 1
        human_outcomes[outcome] += 1
        decision_outcomes[decision][outcome] += 1
        provenance_statuses[sampling_status] += 1
        cell = (decision, qwen_verdict)
        cross_verdicts[cell][verdict] += 1
        cross_outcomes[cell][outcome] += 1
        if probability is not None:
            rows_with_probability += 1
        if sampling_status == "supported_valid":
            supported_provenance_rows += 1
            seed = int(sampling["seed"])
            supported_design_seeds.add(seed)
            if len(supported_design_seeds) > 1:
                raise ValueError(
                    f"{context} supported sampling_provenance mixes seeds "
                    "within one report"
                )

            stratum = str(sampling["stratum"])
            population = int(sampling["stratum_population"])
            selected = int(sampling["stratum_selected"])
            config = (seed, population, selected)
            previous_config = stratum_configs.setdefault(stratum, config)
            if previous_config != config:
                raise ValueError(
                    f"{context} has inconsistent sampling_provenance "
                    f"configuration for stratum={stratum!r}"
                )

            rank = int(sampling["within_stratum_rank"])
            if rank in stratum_observed_ranks[stratum]:
                raise ValueError(
                    f"{context} duplicates sampling_provenance "
                    f"within_stratum_rank={rank} for stratum={stratum!r}"
                )
            stratum_observed_ranks[stratum].add(rank)
            stratum_observed[stratum] += 1
            if stratum_observed[stratum] > selected:
                raise ValueError(
                    f"{context} has more reviewed rows than "
                    "sampling_provenance.stratum_selected for "
                    f"stratum={stratum!r}"
                )

            weight = float(sampling["inverse_probability_weight"])
            weighted_outcomes[outcome] += weight
            weighted_decision_outcomes[decision][outcome] += weight

    if _file_digest(input_path) != input_sha256:
        raise RuntimeError(f"{input_path} changed while it was being read")

    cross_cells = []
    for decision, qwen_verdict in sorted(cross_verdicts):
        verdict_counts = cross_verdicts[(decision, qwen_verdict)]
        outcome_counts = cross_outcomes[(decision, qwen_verdict)]
        cross_cells.append(
            {
                "final_triage_decision": decision,
                "qwen_visual_verdict": qwen_verdict,
                "rows": int(sum(verdict_counts.values())),
                "human_verdicts": _sorted_counter(verdict_counts),
                "human_outcomes": {
                    key: int(outcome_counts.get(key, 0))
                    for key in ("positive", "negative", "uncertain")
                },
            }
        )

    metrics = {
        "auto_keep_precision": _unweighted_metric(
            name="auto_keep_precision",
            decision="auto_keep",
            outcomes=decision_outcomes.get("auto_keep", {}),
            threshold_bound="lower",
            threshold_value=KEEP_PRECISION_LOWER_THRESHOLD,
        ),
        "auto_reject_positive_contamination": _unweighted_metric(
            name="auto_reject_positive_contamination",
            decision="auto_reject",
            outcomes=decision_outcomes.get("auto_reject", {}),
            threshold_bound="upper",
            threshold_value=REJECT_POSITIVE_CONTAMINATION_UPPER_THRESHOLD,
        ),
        "rule_reject_positive_contamination": _unweighted_metric(
            name="rule_reject_positive_contamination",
            decision="rule_reject",
            outcomes=decision_outcomes.get("rule_reject", {}),
            threshold_bound="upper",
            threshold_value=REJECT_POSITIVE_CONTAMINATION_UPPER_THRESHOLD,
        ),
    }

    rows_without_probability = total_rows - rows_with_probability
    weighting: dict[str, Any] = {
        "rows_with_inclusion_probability": rows_with_probability,
        "rows_without_inclusion_probability": rows_without_probability,
        "rows_with_valid_supported_provenance": supported_provenance_rows,
        "rows_total": total_rows,
        "supported_scheme": BALANCED_SAMPLE_SCHEME,
        "supported_version": BALANCED_SAMPLE_VERSION,
        "provenance_statuses": _sorted_counter(provenance_statuses),
        "estimator": (
            "ratio_of_horvitz_thompson_totals_hajek_ipw"
        ),
        "confidence_intervals": None,
        "ci_note": (
            "No confidence interval is attached to IPW estimates. The "
            "Wilson 95% intervals and gates above apply only to the "
            "unweighted binomial human audit."
        ),
        "scope_note": (
            "Population IPW is emitted only for validated balanced-sample v1 "
            "provenance on every reviewed row. A reviewed subset may contain "
            "fewer rows than stratum_selected, so provenance completeness "
            "does not prove that every assigned review was completed or "
            "adjust for reviewer nonresponse."
        ),
    }
    if total_rows > 0 and supported_provenance_rows == total_rows:
        ht_total = float(sum(weighted_outcomes.values()))
        weighting.update(
            {
                "availability": "complete",
                "sampling_design": {
                    "scheme": BALANCED_SAMPLE_SCHEME,
                    "version": BALANCED_SAMPLE_VERSION,
                    "seed": next(iter(supported_design_seeds)),
                    "strata": {
                        stratum: {
                            "population": config[1],
                            "selected": config[2],
                            "reviewed": int(stratum_observed[stratum]),
                        }
                        for stratum, config in sorted(
                            stratum_configs.items()
                        )
                    },
                },
                "ht_estimated_total": ht_total,
                "ht_outcome_totals": _sorted_float_counter(
                    weighted_outcomes
                ),
                "outcome_proportions": {
                    key: (
                        float(weighted_outcomes.get(key, 0.0)) / ht_total
                        if ht_total
                        else None
                    )
                    for key in ("positive", "negative", "uncertain")
                },
                "metrics": {
                    "auto_keep_precision": _weighted_metric(
                        weighted_decision_outcomes.get("auto_keep", {})
                    ),
                    "auto_reject_positive_contamination": _weighted_metric(
                        weighted_decision_outcomes.get(
                            "auto_reject", {}
                        )
                    ),
                    "rule_reject_positive_contamination": _weighted_metric(
                        weighted_decision_outcomes.get(
                            "rule_reject", {}
                        )
                    ),
                },
            }
        )
    elif (
        provenance_statuses.get("unsupported_scheme", 0)
        or provenance_statuses.get("unsupported_version", 0)
    ):
        weighting.update(
            {
                "availability": "unsupported",
                "reason": (
                    "unknown_or_mixed_sampling_scheme_or_version; no "
                    "population IPW estimate is reported"
                ),
            }
        )
    elif supported_provenance_rows or rows_with_probability:
        weighting.update(
            {
                "availability": "incomplete",
                "reason": (
                    "missing_or_partial_supported_sampling_provenance; no "
                    "population IPW estimate is reported"
                ),
            }
        )
    elif provenance_statuses.get("missing_scheme", 0):
        weighting.update(
            {
                "availability": "incomplete",
                "reason": (
                    "sampling_provenance_scheme_is_missing; no population "
                    "IPW estimate is reported"
                ),
            }
        )
    else:
        weighting.update(
            {
                "availability": "not_available",
                "reason": "no inclusion_probability values",
            }
        )

    covers_rule_rejects = "rule_reject" in decision_outcomes
    covers_downstream = bool(
        {"auto_keep", "review", "auto_reject"} & set(decision_outcomes)
    )
    if covers_rule_rejects and covers_downstream:
        population_scope = "mixed_rule_reject_and_post_rule_candidate_audits"
    elif covers_rule_rejects:
        population_scope = "rule_stage_reject_audit"
    else:
        population_scope = "post_rule_candidate_pool"

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
        "stage": "human_review_calibration",
        "scope": {
            "population": population_scope,
            "covers_downstream_feature_qwen_fusion": covers_downstream,
            "covers_rule_stage_rejects": covers_rule_rejects,
            "combined_cascade_sensitivity": None,
            "false_negative_rate": None,
            "combination_note": (
                "A combined cascade sensitivity requires compatible "
                "population weights and design-based uncertainty across the "
                "rule-reject and post-rule samples; this report keeps their "
                "within-reject positive-contamination metrics separate. "
                "Those metrics use rejected rows as their denominator and "
                "must not be called a false-negative or miss rate."
            ),
        },
        "input": str(input_path),
        "input_sha256": input_sha256,
        "rows": total_rows,
        "label_semantics": {
            "positive_verdicts": sorted(POSITIVE_VERDICTS),
            "negative_verdicts": sorted(NEGATIVE_VERDICTS),
            "uncertain_verdict": UNCERTAIN_VERDICT,
            "uncertain_policy": (
                "reported separately and excluded from positive/negative "
                "metric denominators"
            ),
        },
        "human_verdicts": _sorted_counter(human_verdicts),
        "human_outcomes": {
            key: int(human_outcomes.get(key, 0))
            for key in ("positive", "negative", "uncertain")
        },
        "cross_tabulation": {
            "dimensions": [
                "final_triage.decision",
                "qwen_evidence.visual.result.verdict",
                "human_review.verdict",
            ],
            "qwen_unavailable_buckets": [
                "not_available",
                "malformed_record",
                "status:<non-ok-status>",
            ],
            "cells": cross_cells,
        },
        "metrics": metrics,
        "inverse_probability_weighting": weighting,
        "statistical_note": (
            "Wilson 95% confidence intervals assume an unweighted binomial "
            "audit and describe only the audited sample distribution. They "
            "are not design-based intervals for the unequal-probability "
            "candidate population, are never transferred to IPW point "
            "estimates, do not cover stages absent from the input, and cannot "
            "be combined into an end-to-end cascade interval."
        ),
    }


def generate(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different paths")
    report = build_report(input_path)
    _atomic_write_json(
        output_path,
        report,
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(
        f"[motive-review-report] rows={report['rows']} "
        f"keep={report['metrics']['auto_keep_precision']['assessment']} "
        f"reject={report['metrics']['auto_reject_positive_contamination']['assessment']} "
        f"rule_reject={report['metrics']['rule_reject_positive_contamination']['assessment']} "
        f"output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report human calibration of final triage and Qwen visual verdicts."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Human-review-merged JSONL manifest.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
