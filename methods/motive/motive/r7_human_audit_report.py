"""Strict, design-based report for the R7 human audit.

The report consumes the immutable output directory produced by
``r7_human_audit_sample`` and the two JSONL manifests produced by
``human_review merge`` (primary and secondary reviewer slots).  It validates
the complete byte/hash/provenance chain before computing any statistic.

Only the two pre-registered ``sampling_mode == "probability"`` cohorts are
used for population inference.  Family-coverage and priority-review rows are
purposive case-finding observations and are never allowed into a population
rate.  Missing and human-``uncertain`` outcomes remain visible and are
propagated as worst-case lower/upper bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import r7_human_audit_policy as _policy_module
from . import r7_human_audit_sample as _sample_module
from .human_review import (
    OPTIONAL_REVIEW_TEXT_FIELDS,
    R7_ASSIGNMENT_FIELD,
    R7_ASSIGNMENT_SCHEMA,
    R7_MEDIA_FIELD,
    R7_REVIEW_ITEM_DIGEST_FIELDS,
    R7_RATE_AUDIT_REVIEW_SCHEMA,
    _review_item_digest as _human_review_item_digest,
    _validated_r7_contract,
    normalize_reviewer_id,
)
from .r7_human_audit_policy import (
    implementation_bundle_payload,
    implementation_bundle_sha256,
    policy_payload,
    policy_sha256,
)
from .r7_human_audit_sample import (
    DESIGN_VERSION,
    DONE_NAME,
    DONE_SCHEMA,
    HUMAN_REVIEW_SCHEMA,
    LEDGER_ROW_SCHEMA,
    PRIMARY_REVIEW_NAME,
    REVIEWER_ASSIGNMENTS_NAME,
    ROW_SCHEMA,
    SAMPLED_MANIFEST_NAME,
    SAMPLING_LEDGER_NAME,
    SECONDARY_MANIFEST_NAME,
    SECONDARY_REVIEW_NAME,
    SOURCE_DONE_NAME,
    SOURCE_SUMMARY_NAME,
    SUMMARY_NAME,
    SUMMARY_SCHEMA,
    _validate_source_directory,
    validate_media_binding,
)
from .train_action_repr import (
    HUMAN_APPROVED_VERDICTS,
    HUMAN_REJECTED_VERDICTS,
)


REPORT_SCHEMA = "motive-r7-human-audit-report-v4"
REPORT_VERSION = 4
INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA = (
    "motive-r7-independent-review-process-attestation-v1"
)
_INDEPENDENT_REVIEWER_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "sample_artifact_digest",
        "assignment_set_digest",
        "primary_reviewer_id",
        "secondary_reviewer_id",
        "primary_labels_sha256",
        "secondary_labels_sha256",
        "distinct_humans_attested",
        "secondary_blinded_to_primary_until_completion",
        "attestor_id",
        "timestamp",
    }
)
MERGE_SUMMARY_SCHEMA = HUMAN_REVIEW_SCHEMA
_DEFAULT_POLICY = policy_payload()
Z_95 = float(_DEFAULT_POLICY["wilson_interval"]["z"])
MIN_CONCLUSIVE_PER_COHORT = int(
    _DEFAULT_POLICY["population_gate"][
        "min_conclusive_per_probability_cohort"
    ]
)
MAX_UNRESOLVED_FRACTION = float(
    _DEFAULT_POLICY["population_gate"][
        "max_design_weighted_unresolved_fraction"
    ]
)
MIN_POSITIVE_PRECISION_95_LCB = float(
    _DEFAULT_POLICY["population_gate"][
        "min_pseudo_positive_precision_95_lcb"
    ]
)
MAX_PSEUDO_NEGATIVE_FNR_95_UCB = float(
    _DEFAULT_POLICY["population_gate"][
        "max_pseudo_negative_false_negative_rate_95_ucb"
    ]
)
MIN_DOUBLE_CONCLUSIVE = int(
    _DEFAULT_POLICY["double_review_gate"]["min_conclusive_pairs"]
)
MAX_DOUBLE_UNRESOLVED_FRACTION = float(
    _DEFAULT_POLICY["double_review_gate"][
        "max_unresolved_pair_fraction"
    ]
)
MIN_DOUBLE_RAW_AGREEMENT = float(
    _DEFAULT_POLICY["double_review_gate"][
        "min_exact_raw_agreement_wilson_95_lcb"
    ]
)
MIN_DOUBLE_COHEN_KAPPA = float(
    _DEFAULT_POLICY["double_review_gate"][
        "min_exact_cohen_kappa_bootstrap_95_lcb"
    ]
)
KAPPA_BOOTSTRAP_SEED = int(
    _DEFAULT_POLICY["kappa_bootstrap"]["seed"]
)
KAPPA_BOOTSTRAP_DRAWS = int(
    _DEFAULT_POLICY["kappa_bootstrap"]["draws"]
)
SIMULTANEOUS_INTERVAL_ALPHA = float(
    _DEFAULT_POLICY["finite_population_interval"]["familywise_alpha"]
)

PROBABILITY_COHORTS = (
    "pseudo_positive",
    "pseudo_negative",
)
COHORT_ESTIMANDS = {
    "pseudo_positive": "pseudo_positive_precision",
    "pseudo_negative": "pseudo_negative_false_negative_rate",
}
POSITIVE_VERDICTS = frozenset(HUMAN_APPROVED_VERDICTS)
UNCERTAIN_VERDICT = "uncertain"
NEGATIVE_VERDICTS = frozenset(HUMAN_REJECTED_VERDICTS) - {
    UNCERTAIN_VERDICT
}
ALL_VERDICTS = (
    POSITIVE_VERDICTS | NEGATIVE_VERDICTS | {UNCERTAIN_VERDICT}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_ARTIFACTS = frozenset(
    {
        SAMPLED_MANIFEST_NAME,
        SECONDARY_MANIFEST_NAME,
        SAMPLING_LEDGER_NAME,
        PRIMARY_REVIEW_NAME,
        SECONDARY_REVIEW_NAME,
        REVIEWER_ASSIGNMENTS_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    }
)
_HASHED_SAMPLE_OUTPUTS = frozenset(_SAMPLE_ARTIFACTS - {DONE_NAME})
_TEMPLATE_TEXT_FIELDS = (
    "verdict",
    "reviewer",
    "action_signature",
    *OPTIONAL_REVIEW_TEXT_FIELDS,
    "notes",
)
_TEMPLATE_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "input_digest",
        "review_item_digest",
        "prompt",
        "src_video",
        "tgt_video",
        "event_start_frame",
        "event_end_frame",
        R7_ASSIGNMENT_FIELD,
        R7_MEDIA_FIELD,
        *_TEMPLATE_TEXT_FIELDS,
    }
)
_MERGED_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "verdict",
        "reviewer",
        "action_signature",
        "notes",
        "review_item_digest",
        "label_source_sha256",
        *OPTIONAL_REVIEW_TEXT_FIELDS,
        "event_start_frame",
        "event_end_frame",
        "review_instance_id",
        "annotator_slot",
        "assigned_reviewer_id",
        "assignment_set_digest",
        "policy_sha256",
        "media_binding_sha256",
        "label_scope",
        "direct_training_supervision_allowed",
        "training_authorized",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _require_int(
    value: Any,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{context} must be an integer >= {minimum}"
        )
    return value


def _require_finite(
    value: Any,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{context} is below {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{context} is above {maximum}")
    return converted


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(
    path: Path,
    *,
    canonical: bool,
    allow_empty: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            if canonical and line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no rows")
    return rows


def _safe_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{context} must not be a symlink")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{context} is not a regular file")
    return resolved


def _iid(row: Mapping[str, Any], *, context: str) -> str:
    iid = row.get("iid")
    if (
        not isinstance(iid, str)
        or not iid
        or iid.strip() != iid
        or "\x00" in iid
    ):
        raise ValueError(f"{context} has an invalid IID")
    return iid


def _review_item_digest(row: Mapping[str, Any], *, context: str) -> str:
    return _human_review_item_digest(dict(row), context=context)


def _expected_template(row: Mapping[str, Any]) -> dict[str, Any]:
    iid = _iid(row, context="sample manifest")
    template: dict[str, Any] = {
        "schema_version": HUMAN_REVIEW_SCHEMA,
        "iid": iid,
        "input_digest": _require_sha256(
            row.get("input_digest"),
            context=f"iid={iid} input_digest",
        ),
        "verdict": "",
        "reviewer": "",
        "action_signature": "",
        "event_type": "",
        "actor": "",
        "actor_valid": "",
        "instruction_aligned": "",
        "complete_temporal_event": "",
        "source_action": "",
        "target_action": "",
        "direction": "",
        "speed": "",
        "phase": "",
        "contact_or_interaction": "",
        "camera_motion": "",
        "preservation_ok": "",
        "event_start_frame": "",
        "event_end_frame": "",
        "review_confidence": "",
        "secondary_reviewer": "",
        "adjudication": "",
        "notes": "",
    }
    for field in ("prompt", "src_video", "tgt_video"):
        value = row.get(field)
        if not isinstance(value, str):
            raise ValueError(f"iid={iid} has invalid {field}")
        template[field] = value
    for field in (R7_ASSIGNMENT_FIELD, R7_MEDIA_FIELD):
        value = row.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"iid={iid} lacks immutable {field}")
        template[field] = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    template["review_item_digest"] = _review_item_digest(
        template,
        context=f"iid={iid}",
    )
    return template


def _validated_policy_commit(
    *,
    summary: Mapping[str, Any],
    done: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Verify that sampling committed the exact current immutable policy."""

    current = policy_payload()
    current_digest = policy_sha256()
    current_module_sha = _file_digest(
        Path(_policy_module.__file__).resolve(strict=True)
    )
    if summary.get("policy") != current or done.get("policy") != current:
        raise ValueError("sample policy payload differs from current policy")
    for context, value in (
        ("sample summary policy_sha256", summary.get("policy_sha256")),
        ("sample done policy_sha256", done.get("policy_sha256")),
    ):
        if _require_sha256(value, context=context) != current_digest:
            raise ValueError("sample policy digest differs from current policy")
    if _object_digest(current) != current_digest:
        raise RuntimeError("current policy digest implementation differs")
    for context, value in (
        (
            "sample summary policy_module_sha256",
            summary.get("policy_module_sha256"),
        ),
        (
            "sample done policy_module_sha256",
            done.get("policy_module_sha256"),
        ),
    ):
        if _require_sha256(value, context=context) != current_module_sha:
            raise ValueError("sample policy implementation differs")
    return current, current_digest


def _validate_sample_directory(
    sample_dir: Path,
    *,
    expected_sample_artifact_digest: str,
    expected_implementation_bundle_digest: str,
    expected_source_artifact_digest: str,
    expected_source_input_sha256: str,
) -> dict[str, Any]:
    external_anchor = _require_sha256(
        expected_sample_artifact_digest,
        context="expected sample artifact digest",
    )
    external_implementation_anchor = _require_sha256(
        expected_implementation_bundle_digest,
        context="expected implementation bundle digest",
    )
    external_source_anchor = _require_sha256(
        expected_source_artifact_digest,
        context="expected source artifact digest",
    )
    external_source_input_anchor = _require_sha256(
        expected_source_input_sha256,
        context="expected source fused-input digest",
    )
    current_implementation_bundle = _implementation_bundle()
    if (
        current_implementation_bundle.get("bundle_sha256")
        != external_implementation_anchor
    ):
        raise ValueError(
            "current implementation bundle differs from the external "
            "expected digest"
        )
    expanded = sample_dir.expanduser()
    if expanded.is_symlink():
        raise ValueError("sample directory must not be a symlink")
    root = expanded.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    actual = {entry.name for entry in root.iterdir()}
    if actual != _SAMPLE_ARTIFACTS:
        raise ValueError(
            "sample artifact set differs: "
            f"missing={sorted(_SAMPLE_ARTIFACTS - actual)}, "
            f"extra={sorted(actual - _SAMPLE_ARTIFACTS)}"
        )
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name in sorted(_SAMPLE_ARTIFACTS):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sample artifact is not a regular file: {name}")
        paths[name] = path
        digests[name] = _file_digest(path)

    summary = _load_json(paths[SUMMARY_NAME])
    done = _load_json(paths[DONE_NAME])
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("sample summary schema/status differs")
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("sample done schema/status differs")
    if (
        summary.get("design_version") != DESIGN_VERSION
        or done.get("design_version") != DESIGN_VERSION
    ):
        raise ValueError("sample design version differs")
    seed = _require_int(summary.get("seed"), context="sample seed")
    implementation_sha = _require_sha256(
        summary.get("implementation_sha256"),
        context="sample implementation_sha256",
    )
    if done.get("implementation_sha256") != implementation_sha:
        raise ValueError("sample implementation provenance differs")
    current_sampler_sha = _file_digest(
        Path(_sample_module.__file__).resolve(strict=True)
    )
    if implementation_sha != current_sampler_sha:
        raise ValueError(
            "sample was not produced by the current sampler implementation"
        )
    frozen_policy, frozen_policy_sha = _validated_policy_commit(
        summary=summary,
        done=done,
    )
    if (
        summary.get("implementation_bundle")
        != current_implementation_bundle
        or done.get("implementation_bundle")
        != current_implementation_bundle
        or summary.get("expected_implementation_bundle_digest")
        != external_implementation_anchor
        or done.get("expected_implementation_bundle_digest")
        != external_implementation_anchor
        or summary.get(
            "implementation_bundle_external_anchor_verified"
        )
        is not True
        or done.get(
            "implementation_bundle_external_anchor_verified"
        )
        is not True
    ):
        raise ValueError(
            "sample implementation bundle/external anchor commit differs"
        )
    sampling_policy = frozen_policy["sampling_design"]
    expected_formal_design = {
        "seed": int(sampling_policy["seed"]),
        "positive_sample_target": int(
            sampling_policy["positive_sample_target"]
        ),
        "pseudo_negative_sample_target": int(
            sampling_policy["pseudo_negative_sample_target"]
        ),
        "review_sample_target": int(
            sampling_policy["review_sample_target"]
        ),
        "double_review_fraction": float(
            sampling_policy["double_review_fraction"]
        ),
    }
    sampling_design_commit = summary.get("sampling_design")
    if (
        not isinstance(sampling_design_commit, Mapping)
        or done.get("sampling_design") != sampling_design_commit
        or sampling_design_commit.get("immutable_policy_design")
        != expected_formal_design
    ):
        raise ValueError("sample sampling-design commit differs")
    requested_sampling_design = sampling_design_commit.get("requested")
    if not isinstance(requested_sampling_design, Mapping):
        raise ValueError("sample requested sampling design is invalid")
    requested_sampling_design = dict(requested_sampling_design)
    expected_requested_fields = set(expected_formal_design)
    if set(requested_sampling_design) != expected_requested_fields:
        raise ValueError("sample requested sampling-design fields differ")
    for field in (
        "seed",
        "positive_sample_target",
        "pseudo_negative_sample_target",
        "review_sample_target",
    ):
        _require_int(
            requested_sampling_design.get(field),
            context=f"sample requested design {field}",
            minimum=0 if field == "seed" else 1,
        )
    _require_finite(
        requested_sampling_design.get("double_review_fraction"),
        context="sample requested double-review fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if summary.get("seed") != requested_sampling_design["seed"]:
        raise ValueError("sample seed differs from requested design")
    output_sha = done.get("output_sha256")
    summary_outputs = summary.get("outputs")
    if not isinstance(output_sha, Mapping):
        raise ValueError("sample done lacks output_sha256")
    if set(output_sha) != _HASHED_SAMPLE_OUTPUTS:
        raise ValueError("sample done output artifact set differs")
    if not isinstance(summary_outputs, Mapping):
        raise ValueError("sample summary lacks outputs")
    expected_summary_outputs = _HASHED_SAMPLE_OUTPUTS - {SUMMARY_NAME}
    if set(summary_outputs) != expected_summary_outputs:
        raise ValueError("sample summary output artifact set differs")
    for name in sorted(_HASHED_SAMPLE_OUTPUTS):
        expected = _require_sha256(
            output_sha.get(name),
            context=f"sample done {name} digest",
        )
        if expected != digests[name]:
            raise ValueError(f"sample done digest differs: {name}")
        if name != SUMMARY_NAME:
            binding = summary_outputs.get(name)
            if not isinstance(binding, Mapping):
                raise ValueError(f"sample summary lacks binding: {name}")
            if (
                _require_sha256(
                    binding.get("sha256"),
                    context=f"sample summary {name} digest",
                )
                != digests[name]
            ):
                raise ValueError(f"sample summary digest differs: {name}")
    recomputed_artifact_digest = _object_digest(
        {
            name: str(output_sha[name])
            for name in sorted(_HASHED_SAMPLE_OUTPUTS)
        }
    )
    if done.get("artifact_digest") != recomputed_artifact_digest:
        raise ValueError("sample done artifact_digest differs")
    if recomputed_artifact_digest != external_anchor:
        raise ValueError(
            "sample artifact digest differs from the external expected anchor"
        )
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("sample summary lacks source provenance")
    source_artifact_digest = _require_sha256(
        source.get("artifact_digest"),
        context="sample source artifact_digest",
    )
    source_input_sha256 = _require_sha256(
        source.get("fused_input_sha256"),
        context="sample source fused-input digest",
    )
    if (
        done.get("source_artifact_digest") != source_artifact_digest
        or done.get("source_input_sha256") != source_input_sha256
        or source.get("expected_artifact_digest")
        != external_source_anchor
        or source.get("expected_fused_input_sha256")
        != external_source_input_anchor
        or source.get("external_anchor_verified") is not True
        or done.get("expected_source_artifact_digest")
        != external_source_anchor
        or done.get("expected_source_input_sha256")
        != external_source_input_anchor
        or done.get("source_external_anchor_verified") is not True
    ):
        raise ValueError("sample source artifact binding differs")
    source_directory = source.get("directory")
    if not isinstance(source_directory, str) or not source_directory:
        raise ValueError("sample summary source directory is invalid")
    source_root = Path(source_directory)
    (
        live_rows_by_bucket,
        live_source_summary,
        live_source_done,
    ) = _validate_source_directory(source_root)
    live_source_root = source_root.expanduser().resolve(strict=True)
    live_source_summary_sha = _file_digest(
        live_source_root / SOURCE_SUMMARY_NAME
    )
    live_source_done_sha = _file_digest(
        live_source_root / SOURCE_DONE_NAME
    )
    live_source_input = live_source_summary.get("input")
    if not isinstance(live_source_input, Mapping):
        raise ValueError("live source summary lacks fused-input provenance")
    live_source_input_sha = _require_sha256(
        live_source_input.get("sha256"),
        context="live source fused-input digest",
    )
    if (
        source.get("summary_sha256") != live_source_summary_sha
        or done.get("source_summary_sha256") != live_source_summary_sha
        or source.get("done_sha256") != live_source_done_sha
        or done.get("source_done_sha256") != live_source_done_sha
        or live_source_done.get("artifact_digest") != source_artifact_digest
        or source_artifact_digest != external_source_anchor
        or live_source_done.get("input_sha256") != live_source_input_sha
        or live_source_input_sha != source_input_sha256
        or source_input_sha256 != external_source_input_anchor
    ):
        raise ValueError("live source commit differs from sampled provenance")
    live_pseudo_negatives = [
        row
        for row in live_rows_by_bucket["negative"]
        if row["r7_expansion_manifest"].get("negative_role")
        == "pseudo_negative"
    ]
    live_audit_only = [
        row
        for row in live_rows_by_bucket["negative"]
        if row["r7_expansion_manifest"].get("negative_role")
        == "audit_only"
    ]
    live_populations = {
        "pseudo_positive": len(live_rows_by_bucket["positive"]),
        "pseudo_negative": len(live_pseudo_negatives),
        "priority_review_source": len(live_rows_by_bucket["review"]),
        "audit_only_excluded": len(live_audit_only),
    }
    live_source_by_iid = {
        str(row["iid"]): row
        for rows in live_rows_by_bucket.values()
        for row in rows
    }
    if source.get("populations") != live_populations:
        raise ValueError("sample source population commit differs from live source")
    if (
        live_source_summary.get("status") != "complete"
        or live_source_done.get("status") != "complete"
    ):
        raise ValueError("live source commit is incomplete")
    for field in (
        "split_assigned",
        "human_labels_asserted",
        "formal_evidence",
        "training_authorized",
    ):
        if done.get(field) is not False:
            raise ValueError(f"sample done {field} does not fail closed")
    semantics = summary.get("semantics")
    if not isinstance(semantics, Mapping):
        raise ValueError("sample summary lacks semantics")
    if (
        semantics.get("family_coverage_population_inference_allowed")
        is not False
        or semantics.get("priority_review_population_inference_allowed")
        is not False
        or semantics.get("label_scope") != "rate_audit_only"
        or semantics.get("direct_training_supervision_allowed") is not False
        or semantics.get("training_authorized") is not False
    ):
        raise ValueError("sample inference/training semantics differ")
    if (
        done.get("label_scope") != "rate_audit_only"
        or done.get("direct_training_supervision_allowed") is not False
    ):
        raise ValueError("sample label/training scope differs")

    manifest_rows = _load_jsonl(
        paths[SAMPLED_MANIFEST_NAME],
        canonical=True,
        allow_empty=False,
    )
    secondary_rows = _load_jsonl(
        paths[SECONDARY_MANIFEST_NAME],
        canonical=True,
        allow_empty=True,
    )
    ledger_rows = _load_jsonl(
        paths[SAMPLING_LEDGER_NAME],
        canonical=True,
        allow_empty=False,
    )
    primary_templates = _load_jsonl(
        paths[PRIMARY_REVIEW_NAME],
        canonical=True,
        allow_empty=False,
    )
    secondary_templates = _load_jsonl(
        paths[SECONDARY_REVIEW_NAME],
        canonical=True,
        allow_empty=True,
    )
    assignments = _load_jsonl(
        paths[REVIEWER_ASSIGNMENTS_NAME],
        canonical=True,
        allow_empty=False,
    )

    output_rows = {
        SAMPLED_MANIFEST_NAME: len(manifest_rows),
        SECONDARY_MANIFEST_NAME: len(secondary_rows),
        SAMPLING_LEDGER_NAME: len(ledger_rows),
        PRIMARY_REVIEW_NAME: len(primary_templates),
        SECONDARY_REVIEW_NAME: len(secondary_templates),
        REVIEWER_ASSIGNMENTS_NAME: len(assignments),
    }
    for name, rows in output_rows.items():
        binding = summary_outputs[name]
        if binding.get("rows") != rows:
            raise ValueError(f"sample summary row count differs: {name}")
    if done.get("rows") != len(manifest_rows):
        raise ValueError("sample done row count differs")

    media_commit = summary.get("media")
    if not isinstance(media_commit, Mapping):
        raise ValueError("sample summary lacks media commit")
    if (
        media_commit.get("schema_version") != "motive-r7-media-commit-v1"
        or media_commit.get("selected_iids") != len(manifest_rows)
    ):
        raise ValueError("sample media commit metadata differs")
    media_bytes_bound = media_commit.get("media_bytes_bound")
    if not isinstance(media_bytes_bound, bool):
        raise ValueError("sample media bound flag differs")
    expected_media_mode = (
        "formal_bound" if media_bytes_bound else "diagnostic_unbound"
    )
    if media_commit.get("mode") != expected_media_mode:
        raise ValueError("sample media mode differs")
    design_matches_policy = (
        requested_sampling_design == expected_formal_design
    )
    expected_design_mode = (
        "formal_policy_locked"
        if media_bytes_bound
        else "diagnostic_customizable"
    )
    if (
        sampling_design_commit.get("mode") != expected_design_mode
        or sampling_design_commit.get("matches_immutable_policy")
        is not design_matches_policy
        or (media_bytes_bound and not design_matches_policy)
    ):
        raise ValueError(
            "formal sample sampling design is not policy locked"
        )
    recorded_data_root = media_commit.get("data_root")
    if media_bytes_bound:
        if not isinstance(recorded_data_root, str) or not recorded_data_root:
            raise ValueError("formal sample lacks a data root")
        data_root: Path | None = Path(recorded_data_root)
        resolved_data_root = _sample_module._validated_data_root(data_root)
        if str(resolved_data_root) != recorded_data_root:
            raise ValueError("sample data root is not canonical")
    else:
        if recorded_data_root is not None:
            raise ValueError("diagnostic sample asserts a data root")
        resolved_data_root = None
    if (
        done.get("media_bytes_bound") is not media_bytes_bound
        or done.get("data_root") != recorded_data_root
        or done.get("formal_gate_input_eligible") is not media_bytes_bound
        or semantics.get("formal_gate_input_eligible") is not media_bytes_bound
    ):
        raise ValueError("sample media eligibility commit differs")
    expected_bound_files = 2 * len(manifest_rows) if media_bytes_bound else 0
    if media_commit.get("bound_files") != expected_bound_files:
        raise ValueError("sample media bound-file count differs")
    media_binding_set_digest = _require_sha256(
        media_commit.get("media_binding_set_digest"),
        context="sample media binding set digest",
    )
    if done.get("media_binding_set_digest") != media_binding_set_digest:
        raise ValueError("sample media binding set provenance differs")

    assignment_commit = summary.get("reviewer_assignment")
    if not isinstance(assignment_commit, Mapping):
        raise ValueError("sample summary lacks reviewer assignment commit")
    if (
        assignment_commit.get("schema_version")
        != "motive-r7-review-assignment-set-v1"
        or assignment_commit.get("core_digest_excludes_assignment_set_digest")
        is not True
    ):
        raise ValueError("sample reviewer assignment contract differs")
    primary_reviewer = normalize_reviewer_id(
        assignment_commit.get("primary_reviewer_id"),
        context="sample primary",
    )
    secondary_reviewer = normalize_reviewer_id(
        assignment_commit.get("secondary_reviewer_id"),
        context="sample secondary",
    )
    if (
        assignment_commit.get("primary_reviewer_id") != primary_reviewer
        or assignment_commit.get("secondary_reviewer_id")
        != secondary_reviewer
    ):
        raise ValueError("sample reviewer IDs are not normalized")
    if secondary_rows and primary_reviewer == secondary_reviewer:
        raise ValueError("double-review reviewer IDs must differ")
    assignment_set_digest = _require_sha256(
        assignment_commit.get("assignment_set_digest"),
        context="sample assignment set digest",
    )
    if (
        done.get("assignment_set_digest") != assignment_set_digest
        or done.get("primary_reviewer_id") != primary_reviewer
        or done.get("secondary_reviewer_id") != secondary_reviewer
    ):
        raise ValueError("sample reviewer assignment provenance differs")

    by_iid: dict[str, dict[str, Any]] = {}
    probability_strata: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    probability_ranks: dict[tuple[str, str], set[int]] = defaultdict(set)
    expected_secondary_iids: list[str] = []
    expected_ledger: list[dict[str, Any]] = []
    primary_assignment_cores: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows, start=1):
        iid = _iid(row, context=f"sample manifest row {index}")
        if iid in by_iid:
            raise ValueError(f"sample manifest duplicates IID={iid}")
        if "human_review" in row:
            raise ValueError(f"sample manifest already labels IID={iid}")
        sampled_source_row = dict(row)
        for field in (
            "r7_human_audit_sampling",
            R7_ASSIGNMENT_FIELD,
            R7_MEDIA_FIELD,
        ):
            sampled_source_row.pop(field, None)
        if sampled_source_row != live_source_by_iid.get(iid):
            raise ValueError(
                f"iid={iid} sampled source row differs from live source"
            )
        _require_sha256(
            row.get("input_digest"),
            context=f"iid={iid} input_digest",
        )
        contract = _validated_r7_contract(
            row,
            context=f"sample manifest row {index}",
        )
        if contract is None:
            raise ValueError(f"iid={iid} lacks the formal R7 review contract")
        assignment, _media = contract
        expected_assignment = _sample_module._assignment_core(
            iid=iid,
            slot="primary",
            assigned_reviewer_id=primary_reviewer,
            seed=seed,
            policy_digest=frozen_policy_sha,
        )
        if {
            key: value
            for key, value in assignment.items()
            if key != "assignment_set_digest"
        } != expected_assignment:
            raise ValueError(f"iid={iid} primary review assignment differs")
        if assignment.get("assignment_set_digest") != assignment_set_digest:
            raise ValueError(f"iid={iid} assignment-set binding differs")
        primary_assignment_cores.append(expected_assignment)
        validated_media_bound = validate_media_binding(
            row,
            expected_data_root=resolved_data_root,
            allow_diagnostic_unbound=not media_bytes_bound,
        )
        if validated_media_bound is not media_bytes_bound:
            raise ValueError(f"iid={iid} media binding mode differs")
        sampling = row.get("r7_human_audit_sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError(f"iid={iid} lacks sampling provenance")
        if (
            sampling.get("schema_version") != ROW_SCHEMA
            or sampling.get("design_version") != DESIGN_VERSION
            or sampling.get("seed") != seed
            or sampling.get("sample_order") != index
        ):
            raise ValueError(f"iid={iid} sampling provenance differs")
        for field in (
            "split_assigned",
            "human_label",
            "training_eligible",
        ):
            if sampling.get(field) is not False:
                raise ValueError(f"iid={iid} sampling {field} differs")
        cohort = sampling.get("cohort")
        mode = sampling.get("sampling_mode")
        if not isinstance(cohort, str) or not cohort:
            raise ValueError(f"iid={iid} has invalid cohort")
        if cohort in PROBABILITY_COHORTS:
            if mode != "probability":
                raise ValueError(
                    f"iid={iid} probability cohort has mode={mode!r}"
                )
            if sampling.get("estimand") != COHORT_ESTIMANDS[cohort]:
                raise ValueError(f"iid={iid} estimand differs")
            sid = _require_sha256(
                sampling.get("stratum_id"),
                context=f"iid={iid} stratum_id",
            )
            key = sampling.get("stratum_key")
            if (
                not isinstance(key, list)
                or not key
                or any(not isinstance(value, str) or not value for value in key)
            ):
                raise ValueError(f"iid={iid} has invalid stratum key")
            population = _require_int(
                sampling.get("stratum_population"),
                context=f"iid={iid} stratum population",
                minimum=1,
            )
            sample = _require_int(
                sampling.get("stratum_sample_size"),
                context=f"iid={iid} stratum sample",
                minimum=1,
            )
            if sample > population:
                raise ValueError(f"iid={iid} stratum n exceeds N")
            probability = _require_finite(
                sampling.get("selection_probability"),
                context=f"iid={iid} selection probability",
                minimum=0.0,
                maximum=1.0,
            )
            weight = _require_finite(
                sampling.get("design_weight"),
                context=f"iid={iid} design weight",
                minimum=1.0,
            )
            if (
                not math.isclose(
                    probability,
                    sample / population,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    weight,
                    population / sample,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"iid={iid} probability/weight differs")
            rank = _require_int(
                sampling.get("within_stratum_rank"),
                context=f"iid={iid} within-stratum rank",
                minimum=1,
            )
            if rank > sample:
                raise ValueError(f"iid={iid} within-stratum rank exceeds n")
            stratum = (cohort, sid)
            config = {
                "stratum_key": key,
                "population": population,
                "sample": sample,
                "probability": probability,
                "weight": weight,
            }
            existing = probability_strata.setdefault(stratum, config)
            if existing != config:
                raise ValueError(f"stratum configuration differs: {stratum}")
            if rank in probability_ranks[stratum]:
                raise ValueError(f"stratum rank duplicates: {stratum}")
            probability_ranks[stratum].add(rank)
        else:
            expected_purposive_modes = {
                "pseudo_positive_family_coverage":
                    "purposive_family_coverage",
                "pseudo_negative_family_coverage":
                    "purposive_family_coverage",
                "priority_review": "purposive_casefinding",
            }
            if expected_purposive_modes.get(cohort) != mode:
                raise ValueError(
                    f"iid={iid} cohort/mode combination is unsupported"
                )
            if mode not in {
                "purposive_casefinding",
                "purposive_family_coverage",
            }:
                raise ValueError(f"iid={iid} has unsupported sampling mode")
            if (
                sampling.get("estimand") is not None
                or sampling.get("selection_probability") is not None
                or sampling.get("design_weight") is not None
            ):
                raise ValueError(
                    f"iid={iid} purposive row asserts population inference"
                )
        manifest_label = row.get("r7_expansion_manifest")
        if not isinstance(manifest_label, Mapping):
            raise ValueError(f"iid={iid} lacks expansion label")
        source_bucket = manifest_label.get("bucket")
        expected_source_buckets = {
            "pseudo_positive": "positive",
            "pseudo_negative": "negative",
            "pseudo_positive_family_coverage": "positive",
            "pseudo_negative_family_coverage": "negative",
            "priority_review": "review",
        }
        if source_bucket != expected_source_buckets.get(cohort):
            raise ValueError(f"iid={iid} cohort/source bucket differs")
        expected_ledger.append(
            {
                "schema_version": LEDGER_ROW_SCHEMA,
                "iid": iid,
                "input_digest": row["input_digest"],
                "source_bucket": source_bucket,
                **dict(sampling),
            }
        )
        if sampling.get("double_review") is True:
            expected_secondary_iids.append(iid)
        elif sampling.get("double_review") is not False:
            raise ValueError(f"iid={iid} has invalid double_review flag")
        by_iid[iid] = row

    for stratum, config in probability_strata.items():
        if probability_ranks[stratum] != set(
            range(1, int(config["sample"]) + 1)
        ):
            raise ValueError(f"stratum sampled rows are incomplete: {stratum}")
        if int(config["sample"]) < int(config["population"]) and int(
            config["sample"]
        ) < 2:
            raise ValueError(
                f"non-census stratum has no estimable variance: {stratum}"
            )
    if ledger_rows != expected_ledger:
        raise ValueError("sampling ledger differs from sampled manifest")
    if [str(row.get("iid")) for row in secondary_rows] != (
        expected_secondary_iids
    ):
        raise ValueError("secondary manifest order/subset differs")
    expected_secondary: list[dict[str, Any]] = []
    secondary_assignment_cores: list[dict[str, Any]] = []
    for index, secondary_row in enumerate(secondary_rows, start=1):
        iid = _iid(
            secondary_row,
            context=f"secondary manifest row {index}",
        )
        primary_row = by_iid[iid]
        secondary_contract = _validated_r7_contract(
            secondary_row,
            context=f"secondary manifest row {index}",
        )
        if secondary_contract is None:
            raise ValueError(f"secondary IID={iid} lacks R7 contract")
        secondary_assignment, _secondary_media = secondary_contract
        expected_assignment = _sample_module._assignment_core(
            iid=iid,
            slot="secondary",
            assigned_reviewer_id=secondary_reviewer,
            seed=seed,
            policy_digest=frozen_policy_sha,
        )
        if {
            key: value
            for key, value in secondary_assignment.items()
            if key != "assignment_set_digest"
        } != expected_assignment:
            raise ValueError(f"iid={iid} secondary review assignment differs")
        if (
            secondary_assignment.get("assignment_set_digest")
            != assignment_set_digest
        ):
            raise ValueError(f"iid={iid} secondary assignment-set binding differs")
        secondary_assignment_cores.append(expected_assignment)
        primary_base = dict(primary_row)
        primary_base.pop(R7_ASSIGNMENT_FIELD)
        secondary_base = dict(secondary_row)
        secondary_base.pop(R7_ASSIGNMENT_FIELD)
        if secondary_base != primary_base:
            raise ValueError(
                f"secondary IID={iid} differs outside its slot assignment"
            )
        validated_media_bound = validate_media_binding(
            secondary_row,
            expected_data_root=resolved_data_root,
            allow_diagnostic_unbound=not media_bytes_bound,
        )
        if validated_media_bound is not media_bytes_bound:
            raise ValueError(f"secondary IID={iid} media mode differs")
        expected_secondary.append(
            {
                **primary_base,
                R7_ASSIGNMENT_FIELD: {
                    **expected_assignment,
                    "assignment_set_digest": assignment_set_digest,
                },
            }
        )
    recomputed_assignment_set_digest = _object_digest(
        {
            "schema_version": "motive-r7-review-assignment-set-v1",
            "design_version": DESIGN_VERSION,
            "seed": seed,
            "assignments": [
                *primary_assignment_cores,
                *secondary_assignment_cores,
            ],
        }
    )
    if recomputed_assignment_set_digest != assignment_set_digest:
        raise ValueError("review assignment-set digest differs")
    recomputed_media_set_digest = _object_digest(
        [
            {
                "iid": row["iid"],
                R7_MEDIA_FIELD: row[R7_MEDIA_FIELD],
            }
            for row in manifest_rows
        ]
    )
    if recomputed_media_set_digest != media_binding_set_digest:
        raise ValueError("media binding-set digest differs")
    expected_primary_templates = [
        _expected_template(row) for row in manifest_rows
    ]
    expected_secondary_templates = [
        _expected_template(row) for row in expected_secondary
    ]
    if primary_templates != expected_primary_templates:
        raise ValueError("primary blind template differs from manifest")
    if secondary_templates != expected_secondary_templates:
        raise ValueError("secondary blind template differs from manifest")

    expected_assignments: list[dict[str, Any]] = []
    for cores, templates, template_name in (
        (
            primary_assignment_cores,
            expected_primary_templates,
            PRIMARY_REVIEW_NAME,
        ),
        (
            secondary_assignment_cores,
            expected_secondary_templates,
            SECONDARY_REVIEW_NAME,
        ),
    ):
        for core, template in zip(cores, templates, strict=True):
            expected_assignments.append(
                {
                    **core,
                    "assignment_set_digest": assignment_set_digest,
                    "blind_template": template_name,
                    "review_item_digest": template["review_item_digest"],
                }
            )
    if assignments != expected_assignments:
        raise ValueError("reviewer assignments differ")
    if (
        assignment_commit.get("assignments") != len(expected_assignments)
    ):
        raise ValueError("sample assignment task count differs")

    blind_contract = summary.get("blind_review_contract")
    if not isinstance(blind_contract, Mapping):
        raise ValueError("sample summary lacks blind-review contract")
    if (
        blind_contract.get("reviewer_facing_files")
        != [PRIMARY_REVIEW_NAME, SECONDARY_REVIEW_NAME]
        or blind_contract.get("automation_hints_included") is not False
        or blind_contract.get("assignment_set_digest")
        != assignment_set_digest
        or blind_contract.get("policy_sha256") != frozen_policy_sha
        or blind_contract.get("media_bytes_bound") is not media_bytes_bound
        or blind_contract.get("review_item_digest_fields")
        != list(R7_REVIEW_ITEM_DIGEST_FIELDS)
        or blind_contract.get(
            "secondary_rows_are_distinct_review_instances"
        )
        is not True
        or blind_contract.get(
            "secondary_must_be_reviewed_independently_before_adjudication"
        )
        is not True
        or blind_contract.get("merge_manifests")
        != {
            "primary": SAMPLED_MANIFEST_NAME,
            "secondary": SECONDARY_MANIFEST_NAME,
        }
    ):
        raise ValueError("sample blind-review contract differs")

    selected = summary.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("sample summary lacks selected counts")
    expected_selected = Counter(
        str(row["r7_human_audit_sampling"]["cohort"])
        for row in manifest_rows
    )
    if (
        selected.get("pseudo_positive_probability")
        != expected_selected["pseudo_positive"]
        or selected.get("pseudo_negative_probability")
        != expected_selected["pseudo_negative"]
        or selected.get("pseudo_positive_family_coverage")
        != expected_selected["pseudo_positive_family_coverage"]
        or selected.get("pseudo_negative_family_coverage")
        != expected_selected["pseudo_negative_family_coverage"]
        or selected.get("priority_review")
        != expected_selected["priority_review"]
        or selected.get("total") != len(manifest_rows)
        or selected.get("double_review") != len(expected_secondary)
        or selected.get("review_tasks") != len(expected_assignments)
    ):
        raise ValueError("sample selected counts differ")

    designs = summary.get("designs")
    if not isinstance(designs, Mapping):
        raise ValueError("sample summary lacks designs")
    for cohort in PROBABILITY_COHORTS:
        design = designs.get(cohort)
        if not isinstance(design, Mapping):
            raise ValueError(f"sample summary lacks {cohort} design")
        expected_configs = {
            sid: config
            for (candidate_cohort, sid), config in probability_strata.items()
            if candidate_cohort == cohort
        }
        strata = design.get("strata")
        if (
            design.get("sampling_mode") != "probability"
            or not isinstance(strata, list)
            or design.get("sample")
            != sum(int(config["sample"]) for config in expected_configs.values())
            or design.get("population")
            != sum(
                int(config["population"])
                for config in expected_configs.values()
            )
        ):
            raise ValueError(f"sample summary {cohort} design differs")
        observed_configs: dict[str, dict[str, Any]] = {}
        for entry in strata:
            if not isinstance(entry, Mapping):
                raise ValueError(f"sample summary {cohort} stratum invalid")
            sid = _require_sha256(
                entry.get("stratum_id"),
                context=f"sample summary {cohort} stratum ID",
            )
            observed_configs[sid] = {
                "stratum_key": entry.get("stratum_key"),
                "population": entry.get("population"),
                "sample": entry.get("sample"),
                "probability": entry.get("selection_probability"),
                "weight": entry.get("design_weight"),
            }
        if observed_configs != expected_configs:
            raise ValueError(f"sample summary {cohort} strata differ")
        if design.get("population") != live_populations[cohort]:
            raise ValueError(
                f"sample summary {cohort} population differs from live source"
            )

    verification_design = (
        expected_formal_design
        if media_bytes_bound
        else requested_sampling_design
    )
    verification_summary = _sample_module.build_human_audit_sample(
        source_dir=live_source_root,
        output_dir=root,
        data_root=resolved_data_root,
        primary_reviewer_id=primary_reviewer,
        secondary_reviewer_id=secondary_reviewer,
        expected_implementation_bundle_digest=(
            external_implementation_anchor
        ),
        expected_source_artifact_digest=external_source_anchor,
        expected_source_input_sha256=external_source_input_anchor,
        diagnostic_unbound_media=not media_bytes_bound,
        positive_sample=int(
            verification_design["positive_sample_target"]
        ),
        pseudo_negative_sample=int(
            verification_design["pseudo_negative_sample_target"]
        ),
        review_sample=int(
            verification_design["review_sample_target"]
        ),
        double_review_fraction=float(
            verification_design["double_review_fraction"]
        ),
        seed=int(verification_design["seed"]),
        resume=True,
    )
    if verification_summary.get("resume_verified") is not True:
        raise RuntimeError("sampler verification-only replay was not verified")

    for name, expected_digest in digests.items():
        if _file_digest(paths[name]) != expected_digest:
            raise RuntimeError(
                f"sample artifact changed while being read: {name}"
            )
    return {
        "root": root,
        "summary": summary,
        "done": done,
        "digests": digests,
        "manifest_rows": manifest_rows,
        "secondary_rows": secondary_rows,
        "primary_templates": expected_primary_templates,
        "secondary_templates": expected_secondary_templates,
        "by_iid": by_iid,
        "artifact_digest": done["artifact_digest"],
        "expected_artifact_digest": external_anchor,
        "policy": frozen_policy,
        "policy_sha256": frozen_policy_sha,
        "media_bytes_bound": media_bytes_bound,
        "data_root": resolved_data_root,
        "media_binding_set_digest": media_binding_set_digest,
        "assignment_set_digest": assignment_set_digest,
        "primary_reviewer_id": primary_reviewer,
        "secondary_reviewer_id": secondary_reviewer,
        "implementation_bundle":
            current_implementation_bundle,
        "implementation_bundle_sha256":
            external_implementation_anchor,
        "source_artifact_digest": external_source_anchor,
        "source_input_sha256": external_source_input_anchor,
        "sampling_design": dict(sampling_design_commit),
    }


def _resolve_recorded_path(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path is invalid")
    return _safe_file(Path(value), context=context)


def _normalise_frame(value: Any, *, context: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a non-negative integer or blank")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} must be a non-negative integer or blank"
        ) from error
    if integer < 0 or str(integer) != str(value).strip():
        raise ValueError(
            f"{context} must be a non-negative integer or blank"
        )
    return integer


def _expected_merged_review(
    label: Mapping[str, Any],
    *,
    labels_sha256: str,
    context: str,
) -> dict[str, Any] | None:
    verdict = label["verdict"].strip()
    if not verdict:
        return None
    if verdict not in ALL_VERDICTS:
        raise ValueError(f"{context} has invalid verdict={verdict!r}")
    reviewer = label["reviewer"].strip()
    if not reviewer:
        raise ValueError(f"{context} reviewer is required")
    r7_contract = _validated_r7_contract(label, context=context)
    if r7_contract is None:
        raise ValueError(f"{context} lacks the R7 review contract")
    assignment, media = r7_contract
    if reviewer != assignment["assigned_reviewer_id"]:
        raise ValueError(
            f"{context} reviewer differs from the assigned reviewer ID"
        )
    start = _normalise_frame(
        label["event_start_frame"],
        context=f"{context} event_start_frame",
    )
    end = _normalise_frame(
        label["event_end_frame"],
        context=f"{context} event_end_frame",
    )
    if start is not None and end is not None and end < start:
        raise ValueError(f"{context} event_end_frame precedes event_start_frame")
    review: dict[str, Any] = {
        "schema_version": R7_RATE_AUDIT_REVIEW_SCHEMA,
        "verdict": verdict,
        "reviewer": reviewer,
        "action_signature": label["action_signature"].strip(),
        "notes": label["notes"].strip(),
        "review_item_digest": label["review_item_digest"],
        "label_source_sha256": labels_sha256,
    }
    review.update(
        {
            field: label[field].strip()
            for field in OPTIONAL_REVIEW_TEXT_FIELDS
        }
    )
    review.update(
        {
            "event_start_frame": start,
            "event_end_frame": end,
        }
    )
    review.update(
        {
            "review_instance_id": assignment["review_instance_id"],
            "annotator_slot": assignment["annotator_slot"],
            "assigned_reviewer_id": assignment["assigned_reviewer_id"],
            "assignment_set_digest": assignment[
                "assignment_set_digest"
            ],
            "policy_sha256": assignment["policy_sha256"],
            "media_binding_sha256": _object_digest(media),
            "label_scope": "rate_audit_only",
            "direct_training_supervision_allowed": False,
            "training_authorized": False,
        }
    )
    return review


def _validate_merge(
    *,
    slot: str,
    merged_path: Path,
    manifest_path: Path,
    manifest_rows: Sequence[dict[str, Any]],
    templates: Sequence[dict[str, Any]],
    externally_attested_labels_path: Path | None,
    expected_labels_sha256: str | None,
) -> dict[str, Any]:
    has_external_path = externally_attested_labels_path is not None
    has_external_sha = expected_labels_sha256 is not None
    if has_external_path != has_external_sha:
        raise ValueError(
            f"{slot} external labels path and digest must be supplied together"
        )
    if manifest_rows and not has_external_path:
        raise ValueError(
            f"{slot} assigned tasks require external label-byte attestation"
        )
    if has_external_path:
        assert externally_attested_labels_path is not None
        assert expected_labels_sha256 is not None
        externally_attested_labels: Path | None = _safe_file(
            externally_attested_labels_path,
            context=f"{slot} externally attested labels",
        )
        external_labels_sha: str | None = _require_sha256(
            expected_labels_sha256,
            context=f"{slot} expected labels digest",
        )
        if (
            _file_digest(externally_attested_labels)
            != external_labels_sha
        ):
            raise ValueError(
                f"{slot} labels differ from the external expected digest"
            )
    else:
        externally_attested_labels = None
        external_labels_sha = None
    merged = _safe_file(merged_path, context=f"{slot} merged output")
    summary_path = merged.with_suffix(merged.suffix + ".summary.json")
    summary_path = _safe_file(
        summary_path,
        context=f"{slot} merge summary",
    )
    summary_sha = _file_digest(summary_path)
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version") != MERGE_SUMMARY_SCHEMA
        or summary.get("stage") != "merge"
    ):
        raise ValueError(f"{slot} merge summary schema/stage differs")
    recorded_manifest = _resolve_recorded_path(
        summary.get("manifest"),
        context=f"{slot} recorded manifest",
    )
    if recorded_manifest != manifest_path.resolve(strict=True):
        raise ValueError(f"{slot} merge used the wrong manifest")
    manifest_sha = _require_sha256(
        summary.get("manifest_sha256"),
        context=f"{slot} manifest_sha256",
    )
    if manifest_sha != _file_digest(manifest_path):
        raise ValueError(f"{slot} manifest provenance differs")
    recorded_output = _resolve_recorded_path(
        summary.get("output"),
        context=f"{slot} recorded output",
    )
    if recorded_output != merged:
        raise ValueError(f"{slot} recorded output path differs")
    output_sha = _require_sha256(
        summary.get("output_sha256"),
        context=f"{slot} output_sha256",
    )
    if output_sha != _file_digest(merged):
        raise ValueError(f"{slot} merged output digest differs")
    labels_path = _resolve_recorded_path(
        summary.get("labels"),
        context=f"{slot} labels",
    )
    if (
        externally_attested_labels is not None
        and labels_path != externally_attested_labels
    ):
        raise ValueError(
            f"{slot} merge provenance used different label bytes/path"
        )
    labels_sha = _require_sha256(
        summary.get("labels_sha256"),
        context=f"{slot} labels_sha256",
    )
    if labels_sha != _file_digest(labels_path):
        raise ValueError(f"{slot} labels digest differs")
    if (
        external_labels_sha is not None
        and labels_sha != external_labels_sha
    ):
        raise ValueError(f"{slot} externally attested labels digest differs")

    expected_by_iid: dict[str, dict[str, Any]] = {}
    template_by_iid: dict[str, dict[str, Any]] = {}
    for row, template in zip(manifest_rows, templates, strict=True):
        iid = _iid(row, context=f"{slot} manifest")
        expected_by_iid[iid] = row
        if template != _expected_template(row):
            raise ValueError(f"{slot} template binding differs for IID={iid}")
        template_by_iid[iid] = template

    label_rows = _load_jsonl(
        labels_path,
        canonical=False,
        allow_empty=True,
    )
    labels_by_iid: dict[str, dict[str, Any]] = {}
    expected_reviews: dict[str, dict[str, Any]] = {}
    incomplete = 0
    for line_number, label in enumerate(label_rows, start=1):
        context = f"{slot} labels row {line_number}"
        iid = _iid(label, context=context)
        if iid in labels_by_iid:
            raise ValueError(f"{slot} labels duplicate IID={iid}")
        if iid not in expected_by_iid:
            raise ValueError(f"{slot} labels contain unknown IID={iid}")
        if set(label) != _TEMPLATE_FIELDS:
            raise ValueError(
                f"{context} fields differ from the blind template"
            )
        if label.get("schema_version") != HUMAN_REVIEW_SCHEMA:
            raise ValueError(f"{context} schema differs")
        for field in _TEMPLATE_TEXT_FIELDS:
            if not isinstance(label.get(field), str):
                raise ValueError(f"{context} {field} must be a string")
        expected_template = template_by_iid[iid]
        for field in (
            "schema_version",
            "iid",
            "input_digest",
            "prompt",
            "src_video",
            "tgt_video",
            "review_item_digest",
            R7_ASSIGNMENT_FIELD,
            R7_MEDIA_FIELD,
        ):
            if label.get(field) != expected_template[field]:
                raise ValueError(
                    f"{context} immutable template field differs: {field}"
                )
        if (
            _review_item_digest(label, context=context)
            != label["review_item_digest"]
        ):
            raise ValueError(f"{context} review_item_digest differs")
        expected_review = _expected_merged_review(
            label,
            labels_sha256=labels_sha,
            context=context,
        )
        if expected_review is None:
            incomplete += 1
        else:
            expected_reviews[iid] = expected_review
        labels_by_iid[iid] = label

    merged_rows = _load_jsonl(
        merged,
        canonical=False,
        allow_empty=True,
    )
    reviews: dict[str, dict[str, Any]] = {}
    verdict_counts: Counter[str] = Counter()
    for line_number, row in enumerate(merged_rows, start=1):
        context = f"{slot} merged row {line_number}"
        iid = _iid(row, context=context)
        if iid in reviews:
            raise ValueError(f"{slot} merged output duplicates IID={iid}")
        if iid not in expected_reviews:
            raise ValueError(f"{slot} merged output has unexpected IID={iid}")
        review = row.get("human_review")
        if not isinstance(review, dict):
            raise ValueError(f"{context} lacks human_review")
        if set(review) != _MERGED_REVIEW_FIELDS:
            raise ValueError(f"{context} human_review fields differ")
        base = dict(row)
        base.pop("human_review")
        if base != expected_by_iid[iid]:
            raise ValueError(f"{context} changed the sampled manifest row")
        if review != expected_reviews[iid]:
            raise ValueError(f"{context} human label provenance differs")
        reviews[iid] = review
        verdict_counts[str(review["verdict"])] += 1
    if set(reviews) != set(expected_reviews):
        raise ValueError(f"{slot} merged output omits completed labels")

    if summary.get("completed") != len(merged_rows):
        raise ValueError(f"{slot} completed count differs")
    if summary.get("incomplete") != incomplete:
        raise ValueError(f"{slot} incomplete count differs")
    if summary.get("verdicts") != dict(sorted(verdict_counts.items())):
        raise ValueError(f"{slot} verdict counts differ")
    if manifest_rows:
        first_contract = _validated_r7_contract(
            manifest_rows[0],
            context=f"{slot} merge manifest",
        )
        if first_contract is None:
            raise ValueError(f"{slot} merge manifest lacks R7 contract")
        assignment, media = first_contract
        expected_contract_summary = {
            "r7_contract_bound": True,
            "media_bytes_bound":
                media.get("media_bytes_bound") is True,
            "assignment_set_digest":
                assignment["assignment_set_digest"],
            "policy_sha256": assignment["policy_sha256"],
            "annotator_slots": [slot],
            "label_scope": "rate_audit_only",
            "direct_training_supervision_allowed": False,
            "training_authorized": False,
        }
        expected_digest_fields = list(R7_REVIEW_ITEM_DIGEST_FIELDS)
    else:
        expected_contract_summary = {
            "r7_contract_bound": False,
            "media_bytes_bound": False,
            "assignment_set_digest": None,
            "policy_sha256": None,
            "annotator_slots": [],
        }
        expected_digest_fields = [
            "schema_version",
            "iid",
            "input_digest",
            "prompt",
            "src_video",
            "tgt_video",
        ]
    if summary.get("review_item_digest_fields") != expected_digest_fields:
        raise ValueError(f"{slot} review-item digest provenance differs")
    for field, expected in expected_contract_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"{slot} review-item provenance differs: {field}")
    if _file_digest(merged) != output_sha:
        raise RuntimeError(f"{slot} merged output changed while being read")
    if _file_digest(labels_path) != labels_sha:
        raise RuntimeError(f"{slot} labels changed while being read")
    if _file_digest(summary_path) != summary_sha:
        raise RuntimeError(f"{slot} merge summary changed while being read")

    assigned = len(manifest_rows)
    missing_label_rows = assigned - len(labels_by_iid)
    return {
        "slot": slot,
        "path": merged,
        "sha256": output_sha,
        "summary_path": summary_path,
        "summary_sha256": summary_sha,
        "labels_path": labels_path,
        "labels_sha256": labels_sha,
        "expected_labels_sha256": external_labels_sha,
        "external_hash_attestation_verified":
            external_labels_sha is not None,
        "assigned": assigned,
        "label_rows": len(label_rows),
        "completed": len(reviews),
        "blank": incomplete,
        "missing_label_rows": missing_label_rows,
        "reviews": reviews,
        "verdict_counts": dict(sorted(verdict_counts.items())),
    }


def _validate_independent_reviewer_attestation(
    *,
    attestation_path: Path | None,
    expected_attestation_sha256: str | None,
    sample: Mapping[str, Any],
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an externally anchored declaration of the review process.

    This attestation binds the two completed label files and the immutable
    assignment/sample commits.  It is a process declaration, not a
    cryptographic proof of either reviewer's real-world identity.
    """

    has_path = attestation_path is not None
    has_sha = expected_attestation_sha256 is not None
    if has_path != has_sha:
        raise ValueError(
            "independent-reviewer attestation path and digest must be "
            "supplied together"
        )
    if not has_path:
        return {
            "schema_version":
                INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA,
            "status": "not_supplied",
            "required": bool(sample["secondary_rows"]),
            "path": None,
            "sha256": None,
            "expected_sha256": None,
            "external_process_attestation_verified": False,
            "independent_humans_attested": False,
            "secondary_blinded_to_primary_until_completion_attested":
                False,
            "cryptographic_reviewer_identity_verified": False,
            "reason": "external_process_attestation_not_supplied",
        }

    assert attestation_path is not None
    assert expected_attestation_sha256 is not None
    path = _safe_file(
        attestation_path,
        context="independent-reviewer process attestation",
    )
    expected_sha = _require_sha256(
        expected_attestation_sha256,
        context="expected independent-reviewer attestation digest",
    )
    live_sha = _file_digest(path)
    if live_sha != expected_sha:
        raise ValueError(
            "independent-reviewer attestation differs from the external "
            "expected digest"
        )
    payload = _load_json(path)
    if set(payload) != _INDEPENDENT_REVIEWER_ATTESTATION_FIELDS:
        raise ValueError(
            "independent-reviewer attestation fields differ from the fixed "
            "schema"
        )
    if (
        payload.get("schema_version")
        != INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA
    ):
        raise ValueError(
            "independent-reviewer attestation schema_version differs"
        )

    expected_bindings = {
        "sample_artifact_digest": sample["artifact_digest"],
        "assignment_set_digest": sample["assignment_set_digest"],
        "primary_reviewer_id": sample["primary_reviewer_id"],
        "secondary_reviewer_id": sample["secondary_reviewer_id"],
        "primary_labels_sha256": primary["labels_sha256"],
        "secondary_labels_sha256": secondary["labels_sha256"],
    }
    for field in (
        "sample_artifact_digest",
        "assignment_set_digest",
        "primary_labels_sha256",
        "secondary_labels_sha256",
    ):
        _require_sha256(
            payload.get(field),
            context=f"independent-reviewer attestation {field}",
        )
    for field in ("primary_reviewer_id", "secondary_reviewer_id"):
        normalized = normalize_reviewer_id(
            payload.get(field),
            context=f"attestation {field}",
        )
        if normalized != payload.get(field):
            raise ValueError(
                f"independent-reviewer attestation {field} is not normalized"
            )
    for field, expected in expected_bindings.items():
        if payload.get(field) != expected:
            raise ValueError(
                "independent-reviewer attestation binding differs: "
                f"{field}"
            )
    if (
        payload.get("distinct_humans_attested") is not True
        or payload.get(
            "secondary_blinded_to_primary_until_completion"
        )
        is not True
    ):
        raise ValueError(
            "independent-reviewer process declarations must both be true"
        )
    if sample["primary_reviewer_id"] == sample["secondary_reviewer_id"]:
        raise ValueError(
            "independent-reviewer attestation binds identical reviewer IDs"
        )
    attestor_id = payload.get("attestor_id")
    if (
        not isinstance(attestor_id, str)
        or not attestor_id
        or attestor_id.strip() != attestor_id
        or "\x00" in attestor_id
    ):
        raise ValueError(
            "independent-reviewer attestation attestor_id is invalid"
        )
    timestamp = payload.get("timestamp")
    if (
        not isinstance(timestamp, str)
        or not timestamp
        or timestamp.strip() != timestamp
    ):
        raise ValueError(
            "independent-reviewer attestation timestamp is invalid"
        )
    try:
        parsed_timestamp = datetime.fromisoformat(
            timestamp[:-1] + "+00:00"
            if timestamp.endswith("Z")
            else timestamp
        )
    except ValueError as error:
        raise ValueError(
            "independent-reviewer attestation timestamp must be ISO-8601"
        ) from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError(
            "independent-reviewer attestation timestamp must include a "
            "timezone"
        )
    if _file_digest(path) != live_sha:
        raise RuntimeError(
            "independent-reviewer attestation changed while being read"
        )
    return {
        "schema_version": INDEPENDENT_REVIEWER_ATTESTATION_SCHEMA,
        "status": "verified",
        "required": bool(sample["secondary_rows"]),
        "path": path,
        "sha256": live_sha,
        "expected_sha256": expected_sha,
        "external_process_attestation_verified": True,
        "independent_humans_attested": True,
        "secondary_blinded_to_primary_until_completion_attested": True,
        "cryptographic_reviewer_identity_verified": False,
        "attestor_id": attestor_id,
        "timestamp": timestamp,
        "bindings": expected_bindings,
    }


def _outcome(verdict: str | None) -> str:
    if verdict is None:
        return "missing"
    if verdict in POSITIVE_VERDICTS:
        return "positive"
    if verdict == UNCERTAIN_VERDICT:
        return "uncertain"
    if verdict in NEGATIVE_VERDICTS:
        return "negative"
    raise AssertionError(f"unvalidated verdict={verdict!r}")


def _binary_sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )


def _ci95(estimate: float, variance: float) -> dict[str, float]:
    if variance < 0.0 and variance > -1e-15:
        variance = 0.0
    if variance < 0.0 or not math.isfinite(variance):
        raise ValueError("estimated variance is invalid")
    standard_error = math.sqrt(variance)
    return {
        "level": 0.95,
        "method": "normal_finite_population_wald_clipped",
        "lower": max(0.0, estimate - Z_95 * standard_error),
        "upper": min(1.0, estimate + Z_95 * standard_error),
        "standard_error": standard_error,
        "inference_role": "diagnostic_only",
    }


def _log_combination(total: int, selected: int) -> float:
    if selected < 0 or selected > total:
        return -math.inf
    selected = min(selected, total - selected)
    return (
        math.lgamma(total + 1)
        - math.lgamma(selected + 1)
        - math.lgamma(total - selected + 1)
    )


def _hypergeometric_tail_probability(
    *,
    population: int,
    successes: int,
    sample: int,
    observed: int,
    tail: str,
) -> float:
    """Return a numerically stable finite-population hypergeometric tail."""

    if (
        isinstance(population, bool)
        or not isinstance(population, int)
        or population < 1
        or isinstance(successes, bool)
        or not isinstance(successes, int)
        or successes < 0
        or successes > population
        or isinstance(sample, bool)
        or not isinstance(sample, int)
        or sample < 1
        or sample > population
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 0
        or observed > sample
    ):
        raise ValueError("hypergeometric arguments are invalid")
    if tail not in {"lower", "upper"}:
        raise ValueError("hypergeometric tail must be lower or upper")
    support_lower = max(0, sample - (population - successes))
    support_upper = min(sample, successes)
    if tail == "lower":
        first = max(observed, support_lower)
        last = support_upper
    else:
        first = support_lower
        last = min(observed, support_upper)
    if first > last:
        return 0.0
    denominator = _log_combination(population, sample)
    log_probabilities = [
        (
            _log_combination(successes, value)
            + _log_combination(population - successes, sample - value)
            - denominator
        )
        for value in range(first, last + 1)
    ]
    maximum = max(log_probabilities)
    probability = math.exp(maximum) * math.fsum(
        math.exp(value - maximum) for value in log_probabilities
    )
    return min(1.0, max(0.0, probability))


def _hypergeometric_population_bound(
    *,
    population: int,
    sample: int,
    observed: int,
    tail: str,
    tail_alpha: float,
) -> int:
    """Invert an exact hypergeometric tail for the population success total."""

    if (
        isinstance(population, bool)
        or not isinstance(population, int)
        or population < 1
        or isinstance(sample, bool)
        or not isinstance(sample, int)
        or sample < 1
        or sample > population
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 0
        or observed > sample
    ):
        raise ValueError("finite-population bound arguments are invalid")
    alpha = _require_finite(
        tail_alpha,
        context="finite-population tail alpha",
        minimum=0.0,
        maximum=1.0,
    )
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("finite-population tail alpha must be in (0,1)")
    if tail not in {"lower", "upper"}:
        raise ValueError("finite-population bound tail must be lower or upper")
    if sample == population:
        return observed

    feasible_lower = observed
    feasible_upper = population - sample + observed
    low = feasible_lower
    high = feasible_upper
    if tail == "lower":
        # P_K(X >= observed) is monotone non-decreasing in K.
        while low < high:
            middle = (low + high) // 2
            probability = _hypergeometric_tail_probability(
                population=population,
                successes=middle,
                sample=sample,
                observed=observed,
                tail="lower",
            )
            if probability >= alpha:
                high = middle
            else:
                low = middle + 1
        return low

    # P_K(X <= observed) is monotone non-increasing in K.
    while low < high:
        middle = (low + high + 1) // 2
        probability = _hypergeometric_tail_probability(
            population=population,
            successes=middle,
            sample=sample,
            observed=observed,
            tail="upper",
        )
        if probability >= alpha:
            low = middle
        else:
            high = middle - 1
    return low


def _finite_population_exact_completion_bound(
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    unresolved_value: int,
    tail: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = policy if policy is not None else _DEFAULT_POLICY
    interval_policy = active_policy["finite_population_interval"]
    familywise_alpha = float(interval_policy["familywise_alpha"])
    if unresolved_value not in {0, 1}:
        raise ValueError("unresolved completion value must be zero or one")
    if tail not in {"lower", "upper"}:
        raise ValueError("completion bound tail must be lower or upper")
    total_population = sum(
        int(rows[0]["population"]) for rows in strata.values()
    )
    if total_population <= 0:
        raise ValueError("probability cohort has no population")
    noncensus = sum(
        int(rows[0]["sample"]) < int(rows[0]["population"])
        for rows in strata.values()
    )
    per_stratum_tail_alpha = (
        familywise_alpha / 2.0 / noncensus
        if noncensus
        else None
    )
    bounded_total = 0
    stratum_bounds: list[dict[str, Any]] = []
    for sid, rows in sorted(strata.items()):
        population = int(rows[0]["population"])
        sample = int(rows[0]["sample"])
        if len(rows) != sample:
            raise ValueError(f"stratum sample is incomplete: {sid}")
        observed = sum(
            (
                1
                if row["outcome"] == "positive"
                else 0
                if row["outcome"] == "negative"
                else unresolved_value
            )
            for row in rows
        )
        census = sample == population
        if census:
            bounded_successes = observed
            boundary_tail_probability = 1.0
        else:
            assert per_stratum_tail_alpha is not None
            bounded_successes = _hypergeometric_population_bound(
                population=population,
                sample=sample,
                observed=observed,
                tail=tail,
                tail_alpha=per_stratum_tail_alpha,
            )
            boundary_tail_probability = _hypergeometric_tail_probability(
                population=population,
                successes=bounded_successes,
                sample=sample,
                observed=observed,
                tail=tail,
            )
        bounded_total += bounded_successes
        stratum_bounds.append(
            {
                "stratum_id": sid,
                "N": population,
                "n": sample,
                "observed_completed_successes": observed,
                "census": census,
                "tail": tail,
                "tail_alpha": (
                    None if census else per_stratum_tail_alpha
                ),
                "bounded_population_successes": bounded_successes,
                "bounded_population_rate":
                    bounded_successes / population,
                "boundary_tail_probability": boundary_tail_probability,
            }
        )
    return {
        "method":
            "bonferroni_hypergeometric_finite_population_exact_inversion",
        "confidence_level": 1.0 - familywise_alpha,
        "interval_convention": interval_policy["interval_convention"],
        "familywise_alpha": familywise_alpha,
        "familywise_alpha_per_tail":
            familywise_alpha / 2.0,
        "noncensus_strata": noncensus,
        "per_noncensus_stratum_tail_alpha": per_stratum_tail_alpha,
        "census_strata": len(strata) - noncensus,
        "tail": tail,
        "unresolved_completion_value": unresolved_value,
        "population_N": total_population,
        "bounded_population_successes": bounded_total,
        "bound": bounded_total / total_population,
        "strata": stratum_bounds,
    }


def _stratified_mean(
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    unresolved_value: float,
) -> dict[str, Any]:
    total_population = sum(
        int(rows[0]["population"]) for rows in strata.values()
    )
    if total_population <= 0:
        raise ValueError("probability cohort has no population")
    estimate = 0.0
    variance = 0.0
    stratum_values: dict[str, float] = {}
    for sid, rows in sorted(strata.items()):
        population = int(rows[0]["population"])
        assigned = int(rows[0]["sample"])
        if len(rows) != assigned:
            raise ValueError(f"stratum sample is incomplete: {sid}")
        values = [
            (
                1.0
                if row["outcome"] == "positive"
                else 0.0
                if row["outcome"] == "negative"
                else unresolved_value
            )
            for row in rows
        ]
        mean = sum(values) / assigned
        stratum_values[sid] = mean
        population_share = population / total_population
        estimate += population_share * mean
        if assigned < population:
            if assigned < 2:
                raise ValueError(
                    f"non-census stratum cannot estimate variance: {sid}"
                )
            sampling_fraction = assigned / population
            variance += (
                population_share**2
                * (1.0 - sampling_fraction)
                * _binary_sample_variance(values)
                / assigned
            )
    return {
        "estimator": "stratified_srs_mean_hajek_equivalent",
        "estimate": estimate,
        "finite_population_variance": variance,
        "ci95": _ci95(estimate, variance),
        "stratum_means": stratum_values,
    }


def _available_case_hajek(
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    for rows in strata.values():
        weight = float(rows[0]["weight"])
        for row in rows:
            if row["outcome"] in {"positive", "negative"}:
                denominator += weight
                numerator += weight * (
                    1.0 if row["outcome"] == "positive" else 0.0
                )
    if denominator == 0.0:
        return {
            "status": "INSUFFICIENT",
            "reason": "no_conclusive_human_outcomes",
            "estimator": "available_case_hajek_ratio",
            "estimate": None,
            "finite_population_linearization_variance": None,
            "ci95": None,
        }
    estimate = numerator / denominator
    total_variance = 0.0
    for sid, rows in sorted(strata.items()):
        population = int(rows[0]["population"])
        assigned = int(rows[0]["sample"])
        residuals = [
            (
                (1.0 if row["outcome"] == "positive" else 0.0)
                - estimate
                if row["outcome"] in {"positive", "negative"}
                else 0.0
            )
            for row in rows
        ]
        if assigned < population:
            if assigned < 2:
                raise ValueError(
                    f"non-census stratum cannot estimate variance: {sid}"
                )
            total_variance += (
                population**2
                * (1.0 - assigned / population)
                * _binary_sample_variance(residuals)
                / assigned
            )
    variance = total_variance / denominator**2
    return {
        "status": "ESTIMATED",
        "estimand_scope": "among_conclusive_responses_only",
        "estimator": "available_case_hajek_ratio_linearized",
        "estimate": estimate,
        "estimated_conclusive_population_total": denominator,
        "finite_population_linearization_variance": variance,
        "ci95": _ci95(estimate, variance),
        "warning": (
            "This point estimate is conditional on conclusive response. "
            "The all-assigned lower/upper bounds are required for gates."
        ),
    }


def _estimate_cohort(
    records: Sequence[Mapping[str, Any]],
    *,
    cohort: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = policy if policy is not None else _DEFAULT_POLICY
    if cohort not in PROBABILITY_COHORTS:
        raise ValueError(f"unsupported probability cohort={cohort!r}")
    if not records:
        raise ValueError(f"probability cohort is empty: {cohort}")
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("cohort") != cohort:
            raise ValueError("cohort record differs")
        sid = _require_sha256(
            record.get("stratum_id"),
            context="estimate stratum_id",
        )
        population = _require_int(
            record.get("population"),
            context="estimate stratum population",
            minimum=1,
        )
        sample = _require_int(
            record.get("sample"),
            context="estimate stratum sample",
            minimum=1,
        )
        weight = _require_finite(
            record.get("weight"),
            context="estimate weight",
            minimum=1.0,
        )
        if sample > population or not math.isclose(
            weight,
            population / sample,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("estimate design weight differs")
        if record.get("outcome") not in {
            "positive",
            "negative",
            "uncertain",
            "missing",
        }:
            raise ValueError("estimate outcome differs")
        strata[sid].append(record)
    for sid, rows in strata.items():
        config = {
            (
                int(row["population"]),
                int(row["sample"]),
                float(row["weight"]),
            )
            for row in rows
        }
        if len(config) != 1 or len(rows) != int(rows[0]["sample"]):
            raise ValueError(f"estimate stratum configuration differs: {sid}")

    lower_completion = float(
        active_policy["finite_population_interval"][
            "lower_completion_unresolved_value"
        ]
    )
    upper_completion = float(
        active_policy["finite_population_interval"][
            "upper_completion_unresolved_value"
        ]
    )
    lower = _stratified_mean(
        strata,
        unresolved_value=lower_completion,
    )
    upper = _stratified_mean(
        strata,
        unresolved_value=upper_completion,
    )
    exact_lower = _finite_population_exact_completion_bound(
        strata,
        unresolved_value=int(
            active_policy["finite_population_interval"][
                "lower_completion_unresolved_value"
            ]
        ),
        tail="lower",
        policy=active_policy,
    )
    exact_upper = _finite_population_exact_completion_bound(
        strata,
        unresolved_value=int(
            active_policy["finite_population_interval"][
                "upper_completion_unresolved_value"
            ]
        ),
        tail="upper",
        policy=active_policy,
    )
    available = _available_case_hajek(strata)
    counts = Counter(str(record["outcome"]) for record in records)
    assigned = len(records)
    conclusive = counts["positive"] + counts["negative"]
    unresolved = counts["uncertain"] + counts["missing"]
    total_population = sum(
        int(rows[0]["population"]) for rows in strata.values()
    )
    design_weighted_unresolved_fraction = sum(
        int(rows[0]["population"])
        * sum(
            row["outcome"] in {"uncertain", "missing"}
            for row in rows
        )
        / int(rows[0]["sample"])
        for rows in strata.values()
    ) / total_population
    stratum_rows: list[dict[str, Any]] = []
    for sid, rows in sorted(strata.items()):
        outcomes = Counter(str(row["outcome"]) for row in rows)
        n = int(rows[0]["sample"])
        population = int(rows[0]["population"])
        stratum_rows.append(
            {
                "stratum_id": sid,
                "N": population,
                "n": n,
                "n_over_N": n / population,
                "design_weight": float(rows[0]["weight"]),
                "outcomes": {
                    key: int(outcomes.get(key, 0))
                    for key in (
                        "positive",
                        "negative",
                        "uncertain",
                        "missing",
                    )
                },
                "conclusive_n": int(
                    outcomes["positive"] + outcomes["negative"]
                ),
                "lower_bound_stratum_mean": lower["stratum_means"][sid],
                "upper_bound_stratum_mean": upper["stratum_means"][sid],
            }
        )
    return {
        "cohort": cohort,
        "estimand": COHORT_ESTIMANDS[cohort],
        "sampling_mode": "probability",
        "population_N": total_population,
        "assigned_n": assigned,
        "outcomes": {
            key: int(counts.get(key, 0))
            for key in ("positive", "negative", "uncertain", "missing")
        },
        "conclusive_n": conclusive,
        "uncertain_fraction": counts["uncertain"] / assigned,
        "missing_fraction": counts["missing"] / assigned,
        "unresolved_n": unresolved,
        "raw_unresolved_fraction": unresolved / assigned,
        "design_weighted_unresolved_population_fraction":
            design_weighted_unresolved_fraction,
        "unresolved_fraction": design_weighted_unresolved_fraction,
        "strata": stratum_rows,
        "conclusive_available_case": available,
        "all_assigned_identification_bounds": {
            "unresolved_policy": {
                "lower": "uncertain_and_missing_assigned_event_0",
                "upper": "uncertain_and_missing_assigned_event_1",
            },
            "point_identification_interval": [
                lower["estimate"],
                upper["estimate"],
            ],
            "conservative_95_interval": [
                exact_lower["bound"],
                exact_upper["bound"],
            ],
            "gate_interval_method":
                "finite_population_exact_hypergeometric_bonferroni",
            "lower_completion_exact_lcb": exact_lower,
            "upper_completion_exact_ucb": exact_upper,
            "lower": lower,
            "upper": upper,
            "wald_fpc_diagnostic_only": True,
        },
    }


def _cohen_kappa(
    left: Sequence[str],
    right: Sequence[str],
) -> dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("kappa vectors differ in length")
    if not left:
        return {
            "value": None,
            "observed_agreement": None,
            "agreement_count": 0,
            "expected_agreement": None,
            "n": 0,
            "status": "INSUFFICIENT",
            "reason": "no_paired_reviews",
        }
    agreement_count = sum(
        a == b for a, b in zip(left, right, strict=True)
    )
    observed = agreement_count / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = sorted(set(left_counts) | set(right_counts))
    expected = sum(
        (left_counts[category] / len(left))
        * (right_counts[category] / len(right))
        for category in categories
    )
    if math.isclose(expected, 1.0, abs_tol=1e-15):
        return {
            "value": None,
            "observed_agreement": observed,
            "agreement_count": agreement_count,
            "expected_agreement": expected,
            "n": len(left),
            "status": "INSUFFICIENT",
            "reason": "kappa_undefined_zero_expected_disagreement",
        }
    return {
        "value": (observed - expected) / (1.0 - expected),
        "observed_agreement": observed,
        "agreement_count": agreement_count,
        "expected_agreement": expected,
        "n": len(left),
        "status": "ESTIMATED",
        "categories": categories,
    }


def _wilson_interval(
    successes: int,
    total: int,
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = policy if policy is not None else _DEFAULT_POLICY
    wilson_policy = active_policy["wilson_interval"]
    z_value = float(wilson_policy["z"])
    confidence_level = float(wilson_policy["confidence_level"])
    method = str(wilson_policy["method"])
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or successes < 0
        or total < 0
        or successes > total
    ):
        raise ValueError("Wilson interval counts are invalid")
    if total == 0:
        return {
            "status": "INSUFFICIENT",
            "method": method,
            "level": confidence_level,
            "successes": successes,
            "n": total,
            "lower": None,
            "upper": None,
            "reason": "no_both_rated_pairs",
        }
    estimate = successes / total
    z_squared = z_value**2
    denominator = 1.0 + z_squared / total
    centre = (estimate + z_squared / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return {
        "status": "ESTIMATED",
        "method": method,
        "level": confidence_level,
        "successes": successes,
        "n": total,
        "estimate": estimate,
        "lower": max(0.0, centre - half_width),
        "upper": min(1.0, centre + half_width),
    }


def _linear_percentile(
    sorted_values: Sequence[float],
    probability: float,
) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0,1]")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return (
        float(sorted_values[lower_index]) * (1.0 - fraction)
        + float(sorted_values[upper_index]) * fraction
    )


def _bootstrap_cohen_kappa(
    pairs: Sequence[Mapping[str, str]],
    *,
    seed: int | None = None,
    draws: int | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = policy if policy is not None else _DEFAULT_POLICY
    bootstrap_policy = active_policy["kappa_bootstrap"]
    if seed is None:
        seed = int(bootstrap_policy["seed"])
    if draws is None:
        draws = int(bootstrap_policy["draws"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("kappa bootstrap seed is invalid")
    if (
        isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws < 5_000
    ):
        raise ValueError("kappa bootstrap requires at least 5000 draws")
    left = [str(pair["left"]) for pair in pairs]
    right = [str(pair["right"]) for pair in pairs]
    point = _cohen_kappa(left, right)
    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in pairs:
        cohort = pair.get("cohort")
        if not isinstance(cohort, str) or not cohort:
            raise ValueError("kappa bootstrap pair lacks cohort")
        strata[cohort].append((str(pair["left"]), str(pair["right"])))
    provenance = {
        "method": bootstrap_policy["method"],
        "confidence_level": bootstrap_policy["confidence_level"],
        "interval": bootstrap_policy["interval"],
        "quantile_method": bootstrap_policy["quantile_method"],
        "seed": seed,
        "draws": draws,
        "resampling_unit": bootstrap_policy["resampling_unit"],
        "resampling": bootstrap_policy["resampling"],
        "strata": {
            cohort: len(values) for cohort, values in sorted(strata.items())
        },
        "undefined_draw_policy": bootstrap_policy[
            "undefined_draw_policy"
        ],
    }
    if point["value"] is None:
        return {
            **provenance,
            "status": "INSUFFICIENT",
            "reason": "point_kappa_is_undefined",
            "point_estimate": None,
            "undefined_draws": None,
            "ci95": None,
        }

    generator = random.Random(seed)
    values: list[float] = []
    undefined = 0
    ordered_strata = [
        strata[cohort] for cohort in sorted(strata)
    ]
    for _draw in range(draws):
        sampled_left: list[str] = []
        sampled_right: list[str] = []
        for stratum in ordered_strata:
            for _index in range(len(stratum)):
                sampled = stratum[generator.randrange(len(stratum))]
                sampled_left.append(sampled[0])
                sampled_right.append(sampled[1])
        bootstrap = _cohen_kappa(sampled_left, sampled_right)
        value = bootstrap["value"]
        if value is None:
            undefined += 1
            values.append(-1.0)
        else:
            values.append(float(value))
    values.sort()
    return {
        **provenance,
        "status": "ESTIMATED",
        "point_estimate": float(point["value"]),
        "undefined_draws": undefined,
        "ci95": {
            "lower": _linear_percentile(
                values,
                float(bootstrap_policy["lower_quantile"]),
            ),
            "upper": _linear_percentile(
                values,
                float(bootstrap_policy["upper_quantile"]),
            ),
        },
    }


def _double_review_report(
    *,
    assigned_iids: Sequence[str],
    primary_reviews: Mapping[str, Mapping[str, Any]],
    secondary_reviews: Mapping[str, Mapping[str, Any]],
    cohort_by_iid: Mapping[str, str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = policy if policy is not None else _DEFAULT_POLICY
    if len(set(assigned_iids)) != len(assigned_iids):
        raise ValueError("double-review assignments duplicate an IID")
    cohorts = (
        dict(cohort_by_iid)
        if cohort_by_iid is not None
        else {iid: "all" for iid in assigned_iids}
    )
    if set(cohorts) != set(assigned_iids):
        raise ValueError("double-review cohort mapping differs from assignments")
    if any(
        not isinstance(cohort, str) or not cohort
        for cohort in cohorts.values()
    ):
        raise ValueError("double-review cohort mapping contains invalid values")
    primary_verdicts: list[str] = []
    secondary_verdicts: list[str] = []
    conclusive_primary: list[str] = []
    conclusive_secondary: list[str] = []
    primary_binary: list[str] = []
    secondary_binary: list[str] = []
    rated_pairs: list[dict[str, str]] = []
    disagreements: list[dict[str, str]] = []
    missing_primary: list[str] = []
    missing_secondary: list[str] = []
    uncertain_pairs: list[str] = []
    same_assigned_reviewer_id: list[str] = []
    for iid in assigned_iids:
        primary = primary_reviews.get(iid)
        secondary = secondary_reviews.get(iid)
        if primary is None:
            missing_primary.append(iid)
        if secondary is None:
            missing_secondary.append(iid)
        if primary is None or secondary is None:
            continue
        left = str(primary["verdict"])
        right = str(secondary["verdict"])
        primary_verdicts.append(left)
        secondary_verdicts.append(right)
        rated_pairs.append(
            {
                "iid": iid,
                "left": left,
                "right": right,
                "cohort": str(cohorts[iid]),
            }
        )
        if str(primary["reviewer"]) == str(secondary["reviewer"]):
            same_assigned_reviewer_id.append(iid)
        if left != right:
            disagreements.append(
                {
                    "iid": iid,
                    "primary_verdict": left,
                    "secondary_verdict": right,
                }
            )
        if left == UNCERTAIN_VERDICT or right == UNCERTAIN_VERDICT:
            uncertain_pairs.append(iid)
            continue
        conclusive_primary.append(left)
        conclusive_secondary.append(right)
        left_binary = "positive" if left in POSITIVE_VERDICTS else "negative"
        right_binary = (
            "positive" if right in POSITIVE_VERDICTS else "negative"
        )
        primary_binary.append(left_binary)
        secondary_binary.append(right_binary)
    exact = _cohen_kappa(primary_verdicts, secondary_verdicts)
    conclusive_exact = _cohen_kappa(
        conclusive_primary,
        conclusive_secondary,
    )
    binary = _cohen_kappa(primary_binary, secondary_binary)
    agreement_interval = _wilson_interval(
        int(exact["agreement_count"]),
        int(exact["n"]),
        policy=active_policy,
    )
    kappa_bootstrap = _bootstrap_cohen_kappa(
        rated_pairs,
        policy=active_policy,
    )
    unresolved_pairs = len(assigned_iids) - len(conclusive_primary)
    return {
        "assigned_n": len(assigned_iids),
        "both_rated_n": len(primary_verdicts),
        "both_conclusive_n": len(conclusive_primary),
        "missing_primary_n": len(missing_primary),
        "missing_secondary_n": len(missing_secondary),
        "uncertain_pair_n": len(uncertain_pairs),
        "unresolved_pair_n": unresolved_pairs,
        "unresolved_pair_fraction": (
            unresolved_pairs / len(assigned_iids)
            if assigned_iids
            else 1.0
        ),
        "same_assigned_reviewer_id_n":
            len(same_assigned_reviewer_id),
        "same_assigned_reviewer_id_iids":
            same_assigned_reviewer_id,
        "cryptographic_reviewer_identity_verified": False,
        "independent_humans_attested": False,
        "exact_verdict": exact,
        "exact_verdict_raw_agreement_wilson_95": agreement_interval,
        "exact_verdict_cohen_kappa_bootstrap_95": kappa_bootstrap,
        "conclusive_only_exact_verdict_diagnostic": conclusive_exact,
        "conclusive_only_binary_outcome_diagnostic": binary,
        "binary_outcome": binary,
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "automatic_adjudication_performed": False,
        "adjudication_result": None,
        "note": (
            "Disagreements remain unresolved; this report never selects one "
            "reviewer's label or creates an adjudicated label."
        ),
    }


def _check(
    *,
    passed: bool,
    value: Any,
    comparator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "comparator": comparator,
        "threshold": threshold,
    }


def _build_gate(
    estimates: Mapping[str, Mapping[str, Any]],
    double: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    policy_digest: str | None = None,
    implementation_bundle_sha256: str,
    formal_evidence_eligible: bool = True,
    independent_reviewer_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    implementation_bundle_sha256 = _require_sha256(
        implementation_bundle_sha256,
        context="gate implementation bundle digest",
    )
    active_policy = (
        dict(policy) if policy is not None else policy_payload()
    )
    active_policy_digest = (
        policy_digest
        if policy_digest is not None
        else _object_digest(active_policy)
    )
    if _object_digest(active_policy) != active_policy_digest:
        raise ValueError("gate policy payload/digest differ")
    population_policy = active_policy["population_gate"]
    double_policy = active_policy["double_review_gate"]
    bootstrap_policy = active_policy["kappa_bootstrap"]
    min_conclusive = int(
        population_policy["min_conclusive_per_probability_cohort"]
    )
    max_unresolved = float(
        population_policy["max_design_weighted_unresolved_fraction"]
    )
    min_precision = float(
        population_policy["min_pseudo_positive_precision_95_lcb"]
    )
    max_fnr = float(
        population_policy[
            "max_pseudo_negative_false_negative_rate_95_ucb"
        ]
    )
    min_double_conclusive = int(
        double_policy["min_conclusive_pairs"]
    )
    max_double_unresolved = float(
        double_policy["max_unresolved_pair_fraction"]
    )
    min_raw_agreement = float(
        double_policy["min_exact_raw_agreement_wilson_95_lcb"]
    )
    min_kappa = float(
        double_policy[
            "min_exact_cohen_kappa_bootstrap_95_lcb"
        ]
    )
    thresholds = {
        "schema_version": "motive-r7-human-audit-gate-thresholds-v4",
        "min_conclusive_per_probability_cohort":
            min_conclusive,
        "max_unresolved_fraction_per_probability_cohort":
            max_unresolved,
        "unresolved_definition":
            "design_weighted_human_uncertain_plus_missing_primary_label",
        "min_pseudo_positive_precision_95_lcb":
            min_precision,
        "max_pseudo_negative_false_negative_rate_95_ucb":
            max_fnr,
        "precision_lcb_source":
            "all_assigned_lower_completion_exact_hypergeometric_lcb",
        "false_negative_ucb_source":
            "all_assigned_upper_completion_exact_hypergeometric_ucb",
        "min_double_conclusive": min_double_conclusive,
        "max_double_unresolved_fraction":
            max_double_unresolved,
        "double_unresolved_definition":
            "missing_either_review_or_uncertain_either_review",
        "min_double_exact_verdict_raw_agreement_wilson_95_lcb":
            min_raw_agreement,
        "min_double_exact_verdict_cohen_kappa_bootstrap_95_lcb":
            min_kappa,
        "kappa_bootstrap_seed": bootstrap_policy["seed"],
        "kappa_bootstrap_draws": bootstrap_policy["draws"],
        "same_assigned_reviewer_id_pairs_allowed":
            double_policy[
                "same_assigned_reviewer_id_pairs_allowed"
            ],
        "confidence_level": active_policy["finite_population_interval"][
            "confidence_level"
        ],
        "automatic_adjudication": False,
    }
    positive = estimates["pseudo_positive"]
    negative = estimates["pseudo_negative"]
    precision_lcb = positive["all_assigned_identification_bounds"][
        "lower_completion_exact_lcb"
    ]["bound"]
    fnr_ucb = negative["all_assigned_identification_bounds"][
        "upper_completion_exact_ucb"
    ]["bound"]
    exact = double["exact_verdict"]
    agreement_interval = double[
        "exact_verdict_raw_agreement_wilson_95"
    ]
    kappa_interval = double[
        "exact_verdict_cohen_kappa_bootstrap_95"
    ]
    agreement_lcb = agreement_interval.get("lower")
    kappa_ci = kappa_interval.get("ci95")
    kappa_lcb = (
        kappa_ci.get("lower")
        if isinstance(kappa_ci, Mapping)
        else None
    )
    process_attestation_verified = bool(
        independent_reviewer_attestation is not None
        and independent_reviewer_attestation.get(
            "external_process_attestation_verified"
        )
        is True
    )
    independent_humans_attested = bool(
        independent_reviewer_attestation is not None
        and independent_reviewer_attestation.get(
            "independent_humans_attested"
        )
        is True
    )

    prerequisite_checks = {
        "formal_media_and_source_evidence_bound": _check(
            passed=formal_evidence_eligible,
            value=formal_evidence_eligible,
            comparator="==",
            threshold=True,
        ),
        "pseudo_positive_conclusive": _check(
            passed=positive["conclusive_n"] >= min_conclusive,
            value=positive["conclusive_n"],
            comparator=">=",
            threshold=min_conclusive,
        ),
        "pseudo_negative_conclusive": _check(
            passed=negative["conclusive_n"] >= min_conclusive,
            value=negative["conclusive_n"],
            comparator=">=",
            threshold=min_conclusive,
        ),
        "pseudo_positive_unresolved_fraction": _check(
            passed=positive["unresolved_fraction"]
            <= max_unresolved,
            value=positive["unresolved_fraction"],
            comparator="<=",
            threshold=max_unresolved,
        ),
        "pseudo_negative_unresolved_fraction": _check(
            passed=negative["unresolved_fraction"]
            <= max_unresolved,
            value=negative["unresolved_fraction"],
            comparator="<=",
            threshold=max_unresolved,
        ),
        "double_conclusive": _check(
            passed=double["both_conclusive_n"] >= min_double_conclusive,
            value=double["both_conclusive_n"],
            comparator=">=",
            threshold=min_double_conclusive,
        ),
        "double_unresolved_fraction": _check(
            passed=double["unresolved_pair_fraction"]
            <= max_double_unresolved,
            value=double["unresolved_pair_fraction"],
            comparator="<=",
            threshold=max_double_unresolved,
        ),
        "double_assigned_reviewer_ids_distinct": _check(
            passed=double["same_assigned_reviewer_id_n"] == 0,
            value=double["same_assigned_reviewer_id_n"],
            comparator="==",
            threshold=0,
        ),
        "external_independent_reviewer_attestation": _check(
            passed=(
                process_attestation_verified
                and independent_humans_attested
            ),
            value={
                "external_process_attestation_verified":
                    process_attestation_verified,
                "independent_humans_attested":
                    independent_humans_attested,
                "cryptographic_reviewer_identity_verified": False,
            },
            comparator="==",
            threshold={
                "external_process_attestation_verified": True,
                "independent_humans_attested": True,
                "cryptographic_reviewer_identity_verified": False,
            },
        ),
        "double_kappa_defined": _check(
            passed=exact["value"] is not None
            and kappa_interval.get("status") == "ESTIMATED",
            value={
                "point": exact["status"],
                "bootstrap": kappa_interval.get("status"),
            },
            comparator="==",
            threshold={
                "point": "ESTIMATED",
                "bootstrap": "ESTIMATED",
            },
        ),
    }
    evidence_checks = {
        "pseudo_positive_precision_95_lcb": _check(
            passed=precision_lcb >= min_precision,
            value=precision_lcb,
            comparator=">=",
            threshold=min_precision,
        ),
        "pseudo_negative_false_negative_rate_95_ucb": _check(
            passed=fnr_ucb <= max_fnr,
            value=fnr_ucb,
            comparator="<=",
            threshold=max_fnr,
        ),
        "double_exact_verdict_raw_agreement_wilson_95_lcb": _check(
            passed=agreement_lcb is not None
            and agreement_lcb >= min_raw_agreement,
            value=agreement_lcb,
            comparator=">=",
            threshold=min_raw_agreement,
        ),
        "double_exact_verdict_cohen_kappa_bootstrap_95_lcb": _check(
            passed=kappa_lcb is not None
            and kappa_lcb >= min_kappa,
            value=kappa_lcb,
            comparator=">=",
            threshold=min_kappa,
        ),
    }
    prerequisites_pass = all(
        bool(value["passed"]) for value in prerequisite_checks.values()
    )
    if not prerequisites_pass:
        status = "INSUFFICIENT"
        reason = (
            "minimum_evidence_or_external_reviewer_attestation_not_satisfied"
        )
    elif all(bool(value["passed"]) for value in evidence_checks.values()):
        status = "PASS"
        reason = "all_pre_registered_thresholds_satisfied"
    else:
        status = "FAIL"
        reason = "one_or_more_evidence_thresholds_failed"
    return {
        "status": status,
        "reason": reason,
        "threshold_provenance": {
            "thresholds": thresholds,
            "thresholds_sha256": _object_digest(thresholds),
            "policy": active_policy,
            "policy_sha256": active_policy_digest,
            "declared_in_implementation_bundle_sha256":
                implementation_bundle_sha256,
        },
        "prerequisite_checks": prerequisite_checks,
        "evidence_checks": evidence_checks,
        "next_stage_eligible": status == "PASS",
        "external_independent_reviewer_attestation_required": True,
        "external_process_attestation_verified":
            process_attestation_verified,
        "independent_humans_attested": independent_humans_attested,
        "cryptographic_reviewer_identity_verified": False,
        "label_scope": "rate_audit_only",
        "direct_training_supervision_allowed": False,
        "training_authorized": False,
        "training_note": (
            "Passing this audit gate is necessary but does not by itself "
            "authorize representation or generation training."
        ),
    }


def _implementation_bundle() -> dict[str, Any]:
    bundle = implementation_bundle_payload()
    if bundle.get("bundle_sha256") != implementation_bundle_sha256():
        raise RuntimeError("current implementation bundle changed while read")
    return bundle


def build_human_audit_report(
    *,
    sample_dir: Path,
    expected_sample_artifact_digest: str,
    expected_implementation_bundle_digest: str,
    expected_source_artifact_digest: str,
    expected_source_input_sha256: str,
    primary_merged: Path,
    secondary_merged: Path,
    primary_labels_path: Path,
    expected_primary_labels_sha256: str,
    output_path: Path,
    secondary_labels_path: Path | None = None,
    expected_secondary_labels_sha256: str | None = None,
    independent_reviewer_attestation: Path | None = None,
    expected_independent_reviewer_attestation_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    sample = _validate_sample_directory(
        sample_dir,
        expected_sample_artifact_digest=expected_sample_artifact_digest,
        expected_implementation_bundle_digest=(
            expected_implementation_bundle_digest
        ),
        expected_source_artifact_digest=expected_source_artifact_digest,
        expected_source_input_sha256=expected_source_input_sha256,
    )
    target_input_paths = {
        sample["root"] / SAMPLED_MANIFEST_NAME,
        sample["root"] / SECONDARY_MANIFEST_NAME,
    }
    expanded_output = output_path.expanduser()
    if expanded_output.is_symlink():
        raise ValueError("output path must not be a symlink")
    target = expanded_output.resolve(strict=False)
    if target in target_input_paths:
        raise ValueError("output path collides with an input artifact")
    if resume:
        if not target.is_file():
            raise FileNotFoundError(
                "--resume is verification-only and requires the report"
            )
    elif target.exists():
        raise FileExistsError(target)

    primary = _validate_merge(
        slot="primary",
        merged_path=primary_merged,
        manifest_path=sample["root"] / SAMPLED_MANIFEST_NAME,
        manifest_rows=sample["manifest_rows"],
        templates=sample["primary_templates"],
        externally_attested_labels_path=primary_labels_path,
        expected_labels_sha256=expected_primary_labels_sha256,
    )
    secondary = _validate_merge(
        slot="secondary",
        merged_path=secondary_merged,
        manifest_path=sample["root"] / SECONDARY_MANIFEST_NAME,
        manifest_rows=sample["secondary_rows"],
        templates=sample["secondary_templates"],
        externally_attested_labels_path=secondary_labels_path,
        expected_labels_sha256=expected_secondary_labels_sha256,
    )
    reviewer_attestation = _validate_independent_reviewer_attestation(
        attestation_path=independent_reviewer_attestation,
        expected_attestation_sha256=(
            expected_independent_reviewer_attestation_sha256
        ),
        sample=sample,
        primary=primary,
        secondary=secondary,
    )

    probability_records: dict[str, list[dict[str, Any]]] = {
        cohort: [] for cohort in PROBABILITY_COHORTS
    }
    excluded_counts: Counter[tuple[str, str]] = Counter()
    for row in sample["manifest_rows"]:
        iid = str(row["iid"])
        sampling = row["r7_human_audit_sampling"]
        cohort = str(sampling["cohort"])
        mode = str(sampling["sampling_mode"])
        review = primary["reviews"].get(iid)
        verdict = str(review["verdict"]) if review is not None else None
        if cohort in PROBABILITY_COHORTS and mode == "probability":
            probability_records[cohort].append(
                {
                    "iid": iid,
                    "cohort": cohort,
                    "stratum_id": sampling["stratum_id"],
                    "population": sampling["stratum_population"],
                    "sample": sampling["stratum_sample_size"],
                    "weight": sampling["design_weight"],
                    "outcome": _outcome(verdict),
                }
            )
        else:
            excluded_counts[(cohort, mode)] += 1
    estimates = {
        cohort: _estimate_cohort(
            probability_records[cohort],
            cohort=cohort,
            policy=sample["policy"],
        )
        for cohort in PROBABILITY_COHORTS
    }

    double_iids = [str(row["iid"]) for row in sample["secondary_rows"]]
    double_cohort_by_iid = {
        str(row["iid"]): str(
            row["r7_human_audit_sampling"]["cohort"]
        )
        for row in sample["secondary_rows"]
    }
    double = _double_review_report(
        assigned_iids=double_iids,
        primary_reviews=primary["reviews"],
        secondary_reviews=secondary["reviews"],
        cohort_by_iid=double_cohort_by_iid,
        policy=sample["policy"],
    )
    implementation = _implementation_bundle()
    implementation_files = {
        str(entry["logical_name"]): str(entry["sha256"])
        for entry in implementation["files"]
    }
    implementation_sha = implementation_files["report"]
    gate = _build_gate(
        estimates,
        double,
        policy=sample["policy"],
        policy_digest=sample["policy_sha256"],
        implementation_bundle_sha256=implementation["bundle_sha256"],
        formal_evidence_eligible=sample["media_bytes_bound"],
        independent_reviewer_attestation=reviewer_attestation,
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "report_version": REPORT_VERSION,
        "status": "complete",
        "implementation_sha256": implementation_sha,
        "implementation_bundle": implementation,
        "policy": sample["policy"],
        "policy_sha256": sample["policy_sha256"],
        "inputs": {
            "sample": {
                "directory": str(sample["root"]),
                "artifact_digest": sample["artifact_digest"],
                "expected_artifact_digest":
                    sample["expected_artifact_digest"],
                "external_anchor_verified": True,
                "implementation_bundle_sha256":
                    sample["implementation_bundle_sha256"],
                "source_artifact_digest":
                    sample["source_artifact_digest"],
                "source_input_sha256":
                    sample["source_input_sha256"],
                "summary_sha256": sample["digests"][SUMMARY_NAME],
                "done_sha256": sample["digests"][DONE_NAME],
                "sampled_manifest_sha256":
                    sample["digests"][SAMPLED_MANIFEST_NAME],
                "sampling_ledger_sha256":
                    sample["digests"][SAMPLING_LEDGER_NAME],
                "primary_blind_template_sha256":
                    sample["digests"][PRIMARY_REVIEW_NAME],
                "secondary_blind_template_sha256":
                    sample["digests"][SECONDARY_REVIEW_NAME],
                "media_bytes_bound": sample["media_bytes_bound"],
                "media_binding_set_digest":
                    sample["media_binding_set_digest"],
                "assignment_set_digest":
                    sample["assignment_set_digest"],
            },
            "primary_merge": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in primary.items()
                if key != "reviews"
            },
            "secondary_merge": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in secondary.items()
                if key != "reviews"
            },
            "independent_reviewer_process_attestation": {
                key: (
                    str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in reviewer_attestation.items()
            },
        },
        "reviewer_evidence": {
            "evidence_level": "external_hash_attestation",
            "primary_label_bytes_external_hash_verified":
                primary["external_hash_attestation_verified"],
            "secondary_label_bytes_external_hash_verified":
                secondary["external_hash_attestation_verified"],
            "distinct_assigned_reviewer_ids": (
                sample["primary_reviewer_id"]
                != sample["secondary_reviewer_id"]
                if sample["secondary_rows"]
                else None
            ),
            "cryptographic_reviewer_identity_verified": False,
            "external_process_attestation_verified":
                reviewer_attestation[
                    "external_process_attestation_verified"
                ],
            "independent_humans_attested":
                reviewer_attestation["independent_humans_attested"],
            "secondary_blinded_to_primary_until_completion_attested":
                reviewer_attestation[
                    "secondary_blinded_to_primary_until_completion_attested"
                ],
            "external_independent_reviewer_attestation_required":
                bool(sample["secondary_rows"]),
            "reviewer_strings_are_signatures": False,
        },
        "threat_model": {
            "name":
                "controlled_immutable_storage_without_active_concurrent_path_swap",
            "active_concurrent_writer_resistant": False,
            "ordinary_post_commit_byte_mutation_detected": True,
            "media_descriptor_pre_post_fstat_checked": True,
            "limitation": (
                "Path containment and repeated hashing assume no malicious "
                "writer concurrently swaps ancestor paths during the audit."
            ),
        },
        "label_semantics": {
            "event": "human_positive_action_verdict",
            "merged_review_schema": R7_RATE_AUDIT_REVIEW_SCHEMA,
            "positive_verdicts": sorted(POSITIVE_VERDICTS),
            "negative_verdicts": sorted(NEGATIVE_VERDICTS),
            "uncertain_verdict": UNCERTAIN_VERDICT,
            "missing_policy": "retained_as_unresolved",
            "uncertain_policy": "retained_as_unresolved",
        },
        "population_estimates": estimates,
        "purposive_rows_excluded_from_population_inference": {
            "rows": sum(excluded_counts.values()),
            "by_cohort_and_mode": [
                {
                    "cohort": cohort,
                    "sampling_mode": mode,
                    "rows": count,
                }
                for (cohort, mode), count in sorted(excluded_counts.items())
            ],
            "population_inference_allowed": False,
        },
        "inter_reviewer_reliability": double,
        "recommended_gate": gate,
        "semantics": {
            "finite_population_design":
                "stratified_simple_random_sampling_without_replacement",
            "point_estimator":
                "available_case_hajek_ratio_with_fpc_linearization",
            "bound_estimator":
                "bonferroni_hypergeometric_finite_population_exact_inversion",
            "wald_fpc_interval_role": "diagnostic_only",
            "gate_population_interval":
                "two_sided_simultaneous_finite_population_exact_95",
            "gate_agreement_interval": "wilson_score_two_sided_95",
            "gate_kappa_interval":
                "paired_iid_cohort_stratified_percentile_bootstrap_95",
            "purposive_rows_used_for_population_inference": 0,
            "uncertain_or_missing_silently_dropped": False,
            "automatic_adjudication": False,
            "split_assigned": False,
            "label_scope": "rate_audit_only",
            "direct_training_supervision_allowed": False,
            "training_authorized": False,
        },
    }
    payload = _pretty_bytes(report)
    if resume:
        if target.read_bytes() != payload:
            raise ValueError("resume report differs")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        published = False
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)
            published = True
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        if not published:
            raise RuntimeError("report publication failed")
    returned = dict(report)
    returned["resume_verified"] = bool(resume)
    return returned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the strict design-based R7 human-audit report"
    )
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-sample-artifact-digest",
        required=True,
        help=(
            "Externally recorded 64-hex artifact digest from the completed "
            "sample; it must not be inferred from the sample directory"
        ),
    )
    parser.add_argument(
        "--expected-implementation-bundle-digest",
        required=True,
        help="externally recorded current implementation-bundle SHA-256",
    )
    parser.add_argument(
        "--expected-source-artifact-digest",
        required=True,
        help="externally recorded source manifest-v2 artifact digest",
    )
    parser.add_argument(
        "--expected-source-input-sha256",
        required=True,
        help="externally recorded fused-input SHA-256 for the source manifest",
    )
    parser.add_argument("--primary-merged", type=Path, required=True)
    parser.add_argument("--secondary-merged", type=Path, required=True)
    parser.add_argument("--primary-labels", type=Path, required=True)
    parser.add_argument(
        "--expected-primary-labels-sha256",
        required=True,
    )
    parser.add_argument("--secondary-labels", type=Path)
    parser.add_argument("--expected-secondary-labels-sha256")
    parser.add_argument(
        "--independent-reviewer-attestation",
        type=Path,
        help=(
            "optional externally anchored fixed-schema declaration that "
            "distinct humans reviewed independently and the secondary "
            "reviewer remained blinded until completion"
        ),
    )
    parser.add_argument(
        "--expected-independent-reviewer-attestation-sha256",
        help=(
            "external SHA-256 for --independent-reviewer-attestation; "
            "required whenever that file is supplied"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify the exact existing report; never repair or overwrite",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_human_audit_report(
        sample_dir=args.sample_dir,
        expected_sample_artifact_digest=(
            args.expected_sample_artifact_digest
        ),
        expected_implementation_bundle_digest=(
            args.expected_implementation_bundle_digest
        ),
        expected_source_artifact_digest=(
            args.expected_source_artifact_digest
        ),
        expected_source_input_sha256=args.expected_source_input_sha256,
        primary_merged=args.primary_merged,
        secondary_merged=args.secondary_merged,
        primary_labels_path=args.primary_labels,
        expected_primary_labels_sha256=(
            args.expected_primary_labels_sha256
        ),
        secondary_labels_path=args.secondary_labels,
        expected_secondary_labels_sha256=(
            args.expected_secondary_labels_sha256
        ),
        independent_reviewer_attestation=(
            args.independent_reviewer_attestation
        ),
        expected_independent_reviewer_attestation_sha256=(
            args.expected_independent_reviewer_attestation_sha256
        ),
        output_path=args.output,
        resume=args.resume,
    )
    print(
        _canonical_json(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "gate": report["recommended_gate"]["status"],
                "resume_verified": report["resume_verified"],
                "output": str(args.output.expanduser().resolve(strict=False)),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
