"""Strict retrieval audit for the R10B Bernini controlled pilot.

The evaluator consumes one finalized controlled-pilot commit and one or more
validated Bernini feature artifacts.  Feature artifacts are tagged by the
instruction variant used during extraction:

``canonical``
    The frozen canonical instruction in the controlled-pilot manifest.
``original``
    The original instruction associated with the source edit pair.
``cross_family``
    The other action family's canonical instruction with source and target
    held fixed.  This is a text-label leakage diagnostic, not a negative pair.

Only ``self_motion__factorial_did`` is evaluated.  CountSketch coordinates are
comparable between prompt variants only when every measurement and parameter
contract is identical, and they are never compared across projection seeds.
Cross-seed aggregation is restricted to scalar metrics and ranking agreement.

This module is deliberately fail closed.  A balanced Qwen-audited control set,
both action families, prompt controls, and appearance/morphology leakage
controls are prerequisites for even a development signal.  Regardless of the
numbers, this audit never promotes a representation or authorizes rendering,
generation, or training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import r10b_bernini_tangent_extract as bernini
from .r10b_bernini_pilot_manifest import (
    CANONICAL_PROMPTS,
    FINAL_DONE_NAME,
    FINAL_DONE_SCHEMA,
    FINAL_MANIFEST_NAME,
    FINAL_QUOTAS,
    FINAL_SHORTFALL_NAME,
    FINAL_SUMMARY_NAME,
    FINAL_SUMMARY_SCHEMA,
    MAX_FINAL_ROWS,
    SHORTFALL_SCHEMA,
)
from .r10b_tangent_core import (
    R10BTangentError,
    canonical_json,
    file_digest,
    object_digest,
    validate_smoke_rows,
)


AUDIT_SCHEMA = "motive-r10b-bernini-controlled-retrieval-audit-v1"
DONE_SCHEMA = "motive-r10b-bernini-controlled-retrieval-audit-done-v1"
AUDIT_NAME = "retrieval_audit.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (AUDIT_NAME, DONE_NAME)
PRIMARY_FEATURE = "self_motion__factorial_did"
ARTIFACT_TAGS = ("canonical", "original", "cross_family")
EXPECTED_FAMILIES = tuple(sorted(CANONICAL_PROMPTS))
PROMPT_FIELD_BY_TAG = {
    "canonical": "canonical_prompt",
    "original": "original_prompt",
    "cross_family": "cross_family_shuffle_prompt",
}

# Frozen before looking at controlled-pilot feature values.  These are
# development thresholds only; passing them cannot promote a representation.
DEVELOPMENT_THRESHOLDS = {
    "positive_per_family_min": 4,
    "static_global_min": 4,
    "camera_global_min": 4,
    "effect_global_min": 4,
    "macro_recall_at_1_min": 0.75,
    "macro_recall_at_3_min": 0.95,
    "macro_same_family_pair_auroc_min": 0.75,
    "macro_similarity_margin_min": 0.08,
    "positive_vs_control_auroc_min": 0.65,
    "positive_vs_control_margin_min": 0.03,
    "paraphrase_same_target_cosine_min": 0.75,
    "cross_prompt_same_target_cosine_min": 0.75,
    "cross_prompt_actual_family_accuracy_min": 0.75,
    "cross_prompt_actual_family_margin_min": 0.0,
    "seed_rank_agreement_min": 0.70,
    "morphology_readout_excess_over_chance_max": 0.15,
}

_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class R10BBerniniRetrievalAuditError(ValueError):
    """A retrieval input, binding, metric, or immutable commit is invalid."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise R10BBerniniRetrievalAuditError(
            f"{field} must be one lowercase SHA-256"
        )
    return value


def _read_json_object(
    path: Path,
    *,
    field: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise R10BBerniniRetrievalAuditError(
            f"{field} must be one regular non-symlink file: {path}"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise R10BBerniniRetrievalAuditError(
            f"{field} is not strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise R10BBerniniRetrievalAuditError(
            f"{field} must contain one JSON object"
        )
    return value, raw


def _read_jsonl(
    path: Path,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], bytes]:
    if path.is_symlink() or not path.is_file():
        raise R10BBerniniRetrievalAuditError(
            f"{field} must be one regular non-symlink file: {path}"
        )
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise R10BBerniniRetrievalAuditError(
            f"{field} is not UTF-8"
        ) from error
    if not text:
        if allow_empty:
            return [], raw
        raise R10BBerniniRetrievalAuditError(
            f"{field} must be non-empty canonical JSONL with a final newline"
        )
    if not text.endswith("\n"):
        raise R10BBerniniRetrievalAuditError(
            f"{field} must be non-empty canonical JSONL with a final newline"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise R10BBerniniRetrievalAuditError(
                f"{field}:{line_number} is blank"
            )
        try:
            value = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise R10BBerniniRetrievalAuditError(
                f"{field}:{line_number} is not strict JSON"
            ) from error
        if not isinstance(value, dict):
            raise R10BBerniniRetrievalAuditError(
                f"{field}:{line_number} must contain one JSON object"
            )
        if line != canonical_json(value):
            raise R10BBerniniRetrievalAuditError(
                f"{field}:{line_number} is not canonical JSON"
            )
        rows.append(value)
    if not rows:
        raise R10BBerniniRetrievalAuditError(f"{field} contains no rows")
    return rows, raw


def _require_false_authorization(
    value: Any,
    *,
    field: str,
) -> None:
    if value != _AUTHORIZATION:
        raise R10BBerniniRetrievalAuditError(
            f"{field} authorization differs"
        )


def _pilot_commit(
    pilot_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root = Path(pilot_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise R10BBerniniRetrievalAuditError(
            f"pilot_dir must be one non-symlink directory: {root}"
        )
    observed = sorted(path.name for path in root.iterdir())
    expected = sorted(
        (
            FINAL_MANIFEST_NAME,
            FINAL_SHORTFALL_NAME,
            FINAL_SUMMARY_NAME,
            FINAL_DONE_NAME,
        )
    )
    if observed != expected:
        raise R10BBerniniRetrievalAuditError(
            f"controlled-pilot closure differs: {observed}"
        )

    rows, manifest_raw = _read_jsonl(
        root / FINAL_MANIFEST_NAME,
        field="controlled-pilot manifest",
        allow_empty=True,
    )
    shortfalls, shortfall_raw = _read_json_object(
        root / FINAL_SHORTFALL_NAME,
        field="controlled-pilot shortfalls",
    )
    summary, summary_raw = _read_json_object(
        root / FINAL_SUMMARY_NAME,
        field="controlled-pilot summary",
    )
    done, done_raw = _read_json_object(
        root / FINAL_DONE_NAME,
        field="controlled-pilot done",
    )
    files = {
        FINAL_MANIFEST_NAME: _sha256_bytes(manifest_raw),
        FINAL_SHORTFALL_NAME: _sha256_bytes(shortfall_raw),
        FINAL_SUMMARY_NAME: _sha256_bytes(summary_raw),
    }
    if (
        done.get("schema_version") != FINAL_DONE_SCHEMA
        or done.get("files") != files
        or done.get("rows") != len(rows)
    ):
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot done binding differs"
        )
    if summary.get("schema_version") != FINAL_SUMMARY_SCHEMA:
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot summary schema differs"
        )
    if shortfalls.get("schema_version") != SHORTFALL_SCHEMA:
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot shortfall schema differs"
        )
    if rows:
        try:
            validate_smoke_rows(rows)
        except R10BTangentError as error:
            raise R10BBerniniRetrievalAuditError(str(error)) from error

    iids = [str(row["iid"]) for row in rows]
    components = [str(row["component_id"]) for row in rows]
    quota_counts = dict(
        sorted(Counter(str(row.get("quota_cell")) for row in rows).items())
    )
    if (
        len(set(iids)) != len(rows)
        or len(set(components)) != len(rows)
        or summary.get("rows") != len(rows)
        or summary.get("unique_iids") != len(rows)
        or summary.get("unique_components") != len(rows)
        or summary.get("component_disjoint") is not True
        or summary.get("quota_targets") != FINAL_QUOTAS
        or summary.get("quota_selected") != quota_counts
        or summary.get("outputs", {})
        .get(FINAL_MANIFEST_NAME, {})
        .get("sha256")
        != files[FINAL_MANIFEST_NAME]
        or summary.get("outputs", {})
        .get(FINAL_SHORTFALL_NAME, {})
        .get("sha256")
        != files[FINAL_SHORTFALL_NAME]
    ):
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot row/summary binding differs"
        )
    if (
        summary.get("shortfalls") != shortfalls.get("shortfalls")
        or summary.get("balanced_pilot_ready")
        is not shortfalls.get("balanced_pilot_ready")
        or done.get("balanced_pilot_ready")
        is not summary.get("balanced_pilot_ready")
    ):
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot balanced/shortfall binding differs"
        )
    balanced_expected = (
        not summary.get("shortfalls")
        and len(rows) == MAX_FINAL_ROWS
        and quota_counts == FINAL_QUOTAS
    )
    if summary.get("balanced_pilot_ready") is not balanced_expected:
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot balanced flag is not derived from exact quotas"
        )
    if (
        summary.get("video_bytes_copied") is not False
        or summary.get("controls_fabricated") is not False
        or summary.get("human_labels") is not False
        or shortfalls.get("no_control_rows_fabricated") is not True
        or shortfalls.get("row_reuse_allowed") is not False
        or shortfalls.get("component_reuse_allowed") is not False
    ):
        raise R10BBerniniRetrievalAuditError(
            "controlled-pilot media/control provenance differs"
        )
    _require_false_authorization(
        summary.get("authorization"),
        field="controlled-pilot summary",
    )
    _require_false_authorization(
        done.get("authorization"),
        field="controlled-pilot done",
    )
    for index, row in enumerate(rows):
        if row.get("pilot_rank") != index + 1:
            raise R10BBerniniRetrievalAuditError(
                "controlled-pilot rank/order differs"
            )
        _require_false_authorization(
            row.get("authorization"),
            field=f"controlled-pilot row {index}",
        )
        for gate in (
            "formal_evidence",
            "representation_promoted",
            "renderer_probe_authorized",
            "generation_authorized",
            "training_authorized",
        ):
            if row.get(gate) is not False:
                raise R10BBerniniRetrievalAuditError(
                    f"controlled-pilot row false gate differs: {gate}"
                )

    all_files = {**files, FINAL_DONE_NAME: _sha256_bytes(done_raw)}
    binding = {
        "path": str(root.resolve()),
        "files": all_files,
        "commit_digest": object_digest(all_files),
        "manifest_sha256": files[FINAL_MANIFEST_NAME],
        "rows": len(rows),
        "balanced_pilot_ready": balanced_expected,
        "qwen_audit": copy.deepcopy(summary.get("qwen_audit")),
    }
    return rows, summary, binding


def validate_controlled_pilot_commit(
    pilot_dir: str | Path,
) -> dict[str, Any]:
    """Validate a balanced or shortfall-only controlled-pilot commit."""

    _rows, summary, binding = _pilot_commit(pilot_dir)
    return {
        "status": "VALID",
        "pilot_dir": binding["path"],
        "commit_digest": binding["commit_digest"],
        "rows": binding["rows"],
        "balanced_pilot_ready": binding["balanced_pilot_ready"],
        "experiment_role": summary.get("experiment_role"),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "training_authorized": False,
    }


def _expected_prompt(row: Mapping[str, Any], tag: str) -> str:
    field = PROMPT_FIELD_BY_TAG[tag]
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise R10BBerniniRetrievalAuditError(
            f"iid={row.get('iid')} has no bound {tag} prompt"
        )
    return value


def _effective_prompt_digest(value: str) -> str:
    """Reproduce Bernini cleaning, with an ASCII-only dependency fallback."""

    try:
        import ftfy  # type: ignore

        cleaned = ftfy.fix_text(str(value))
    except ImportError:
        if not str(value).isascii():
            raise R10BBerniniRetrievalAuditError(
                "ftfy is required to bind a non-ASCII prompt"
            )
        cleaned = str(value)
    cleaned = re.sub(
        r"\s+",
        " ",
        html.unescape(html.unescape(cleaned)),
    ).strip()
    effective = bernini.V2V_SYSTEM_PROMPT + cleaned
    return _sha256_bytes(effective.encode("utf-8"))


def _variant_manifest(
    *,
    tag: str,
    summary: Mapping[str, Any],
    pilot_rows: Sequence[Mapping[str, Any]],
    pilot_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = summary.get("data")
    if not isinstance(data, Mapping):
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact data record is missing"
        )
    manifest_value = data.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact manifest path is missing"
        )
    path = Path(manifest_value).expanduser()
    rows, raw = _read_jsonl(
        path,
        field=f"{tag} extraction manifest",
    )
    digest = _sha256_bytes(raw)
    if data.get("manifest_sha256") != digest:
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact manifest digest differs"
        )
    if len(rows) != len(pilot_rows):
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact manifest row count differs"
        )
    for index, (observed, pilot) in enumerate(zip(rows, pilot_rows)):
        expected = copy.deepcopy(dict(pilot))
        expected["prompt"] = _expected_prompt(pilot, tag)
        if observed != expected:
            raise R10BBerniniRetrievalAuditError(
                f"{tag} manifest row {index} differs outside its bound prompt"
            )
    if tag == "canonical" and digest != pilot_manifest_sha256:
        raise R10BBerniniRetrievalAuditError(
            "canonical artifact is not bound to the finalized pilot manifest"
        )
    return rows, {
        "path": str(path.resolve(strict=True)),
        "sha256": digest,
        "rows": len(rows),
        "source_binding_sha256": object_digest(
            [
                {key: value for key, value in row.items() if key != "prompt"}
                for row in rows
            ]
        ),
    }


def _artifact_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    model = summary.get("model", {})
    measurement = summary.get("measurement", {})
    data = summary.get("data", {})
    runtime = summary.get("runtime", {})
    official_source = summary.get("official_bernini_source", {})
    if not all(
        isinstance(value, Mapping)
        for value in (model, measurement, data, runtime, official_source)
    ):
        raise R10BBerniniRetrievalAuditError(
            "Bernini artifact contract sections are missing"
        )
    return {
        "extract_schema": summary.get("schema_version"),
        "model": {
            "id": model.get("id"),
            "huggingface_repo": model.get("huggingface_repo"),
            "huggingface_revision": model.get("huggingface_revision"),
            "checkpoint_tree_sha256": model.get(
                "checkpoint_manifest", {}
            ).get("tree_sha256"),
            "selected_weight_sha256_before": model.get(
                "selected_weight_sha256_before"
            ),
            "selected_weight_sha256_after": model.get(
                "selected_weight_sha256_after"
            ),
            "selected_weights_unchanged": model.get(
                "selected_weights_unchanged"
            ),
        },
        "parameter_manifest": summary.get("parameter_manifest"),
        "parameter_manifest_sha256": summary.get(
            "parameter_manifest_sha256"
        ),
        # Exact equality binds projection, noise, sigma, source conditioning,
        # tokenizer, objective, and resize policy in one closed contract.
        "measurement": measurement,
        "track_cache_sha256": data.get("track_cache_sha256"),
        "rows": data.get("rows"),
        "families": data.get("families"),
        "unique_components": data.get("unique_components"),
        "runtime_geometry": {
            key: runtime.get(key)
            for key in (
                "dtype",
                "width",
                "height",
                "num_frames",
            )
        },
        "implementation": summary.get("implementation"),
        "official_bernini_source": official_source,
        "source_tree_sha256": summary.get("source_tree_sha256"),
    }


def _false_validation_gates(
    value: Mapping[str, Any],
    *,
    field: str,
) -> None:
    for gate in (
        "representation_gate_passed",
        "renderer_probe_authorized",
        "editor_training_authorized",
    ):
        if value.get(gate) is not False:
            raise R10BBerniniRetrievalAuditError(
                f"{field} false gate differs: {gate}"
            )


def _load_artifact(
    *,
    tag: str,
    artifact_dir: str | Path,
    pilot_rows: Sequence[Mapping[str, Any]],
    pilot_manifest_sha256: str,
) -> dict[str, Any]:
    root = Path(artifact_dir).expanduser()
    validation_before = bernini.validate_published_extract(root)
    if not isinstance(validation_before, Mapping):
        raise R10BBerniniRetrievalAuditError(
            f"{tag} backend validator returned no record"
        )
    _false_validation_gates(
        validation_before,
        field=f"{tag} backend validation",
    )
    summary, _summary_raw = _read_json_object(
        root / bernini.SUMMARY_NAME,
        field=f"{tag} artifact summary",
    )
    output_rows, _rows_raw = _read_jsonl(
        root / bernini.ROWS_NAME,
        field=f"{tag} artifact rows",
    )
    variant_rows, variant_binding = _variant_manifest(
        tag=tag,
        summary=summary,
        pilot_rows=pilot_rows,
        pilot_manifest_sha256=pilot_manifest_sha256,
    )
    seeds = summary.get("measurement", {}).get("projection_seeds")
    dimension = summary.get("measurement", {}).get(
        "projection_dimension_per_role"
    )
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise R10BBerniniRetrievalAuditError(
            f"{tag} projection contract differs"
        )
    ids = [str(row["iid"]) for row in pilot_rows]
    if (
        len(output_rows) != len(ids)
        or [row.get("iid") for row in output_rows] != ids
    ):
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact output row order differs"
        )
    vectors: dict[int, np.ndarray] = {}
    with np.load(root / bernini.FEATURES_NAME, allow_pickle=False) as archive:
        archive_ids = np.asarray(archive["ids"]).astype(str).tolist()
        if archive_ids != ids:
            raise R10BBerniniRetrievalAuditError(
                f"{tag} feature IID order differs"
            )
        for seed in seeds:
            name = f"{PRIMARY_FEATURE}__p{seed}"
            if name not in archive.files:
                raise R10BBerniniRetrievalAuditError(
                    f"{tag} primary feature is missing: {name}"
                )
            values = np.asarray(archive[name], dtype=np.float64)
            if (
                values.shape != (len(ids), dimension)
                or not np.isfinite(values).all()
            ):
                raise R10BBerniniRetrievalAuditError(
                    f"{tag} primary feature shape/finite contract differs: {name}"
                )
            norms = np.linalg.norm(values, axis=1)
            if np.any(np.abs(norms - 1.0) > 1e-3):
                raise R10BBerniniRetrievalAuditError(
                    f"{tag} primary feature is not normalized: {name}"
                )
            vectors[int(seed)] = values / norms[:, None]

    for index, (output_row, manifest_row) in enumerate(
        zip(output_rows, variant_rows)
    ):
        if (
            output_row.get("family") != manifest_row.get("family")
            or output_row.get("component_id") != manifest_row.get("component_id")
            or output_row.get("source_split") != manifest_row.get("source_split")
            or output_row.get("case_index") != index
            or output_row.get("projection_seed_coordinates_comparable")
            is not False
        ):
            raise R10BBerniniRetrievalAuditError(
                f"{tag} artifact output row {index} identity binding differs"
            )
        for gate in (
            "formal_evidence",
            "representation_gate_passed",
            "renderer_probe_authorized",
            "editor_training_authorized",
        ):
            if output_row.get(gate) is not False:
                raise R10BBerniniRetrievalAuditError(
                    f"{tag} row false gate differs: {gate}"
                )
        conditioning = output_row.get("prompt_conditioning", {})
        prompt = str(manifest_row["prompt"])
        noop = str(manifest_row["noop_prompt"])
        expected_digests = {
            "raw_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "raw_noop_prompt_sha256": _sha256_bytes(noop.encode("utf-8")),
            "effective_prompt_sha256": _effective_prompt_digest(prompt),
            "effective_noop_prompt_sha256": _effective_prompt_digest(noop),
        }
        for key, expected in expected_digests.items():
            if conditioning.get(key) != expected:
                raise R10BBerniniRetrievalAuditError(
                    f"{tag} row {index} prompt digest differs: {key}"
                )

    validation_after = bernini.validate_published_extract(root)
    if validation_after != validation_before:
        raise R10BBerniniRetrievalAuditError(
            f"{tag} artifact validation changed during read-only evaluation"
        )
    artifact_digest = _require_sha256(
        validation_before.get("artifact_digest"),
        field=f"{tag} artifact digest",
    )
    return {
        "tag": tag,
        "path": str(root.resolve(strict=True)),
        "validation": dict(validation_before),
        "artifact_digest": artifact_digest,
        "summary": summary,
        "contract": _artifact_contract(summary),
        "variant_manifest": variant_binding,
        "vectors": vectors,
        "projection_seeds": tuple(int(seed) for seed in seeds),
        "projection_dimension": int(dimension),
    }


def _scalar_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"available": False, "count": 0}
    if not np.isfinite(array).all():
        raise R10BBerniniRetrievalAuditError(
            "non-finite scalar metric was produced"
        )
    return {
        "available": True,
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            if left > right:
                wins += 1.0
            elif left == right:
                wins += 0.5
    return float(wins / (len(positive) * len(negative)))


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise R10BBerniniRetrievalAuditError("non-finite mean metric")
    return result


def _row_role(row: Mapping[str, Any]) -> str:
    value = row.get("pilot_role")
    if value not in {"positive", "wrong", "static", "camera", "effect"}:
        raise R10BBerniniRetrievalAuditError(
            f"iid={row.get('iid')} has unsupported pilot_role={value!r}"
        )
    return str(value)


def _support(rows: Sequence[Mapping[str, Any]], balanced: bool) -> dict[str, Any]:
    positive_by_family = Counter(
        str(row["family"])
        for row in rows
        if _row_role(row) == "positive"
    )
    role_counts = Counter(_row_role(row) for row in rows)
    checks = {
        "balanced_pilot_commit": bool(balanced),
        "exact_expected_families": (
            set(positive_by_family) == set(EXPECTED_FAMILIES)
        ),
        "positive_per_family": all(
            positive_by_family[family]
            >= DEVELOPMENT_THRESHOLDS["positive_per_family_min"]
            for family in EXPECTED_FAMILIES
        ),
        "wrong_prompt_counterfactual_is_separate_artifact": True,
        "static_global": (
            role_counts["static"]
            >= DEVELOPMENT_THRESHOLDS["static_global_min"]
        ),
        "camera_global": (
            role_counts["camera"]
            >= DEVELOPMENT_THRESHOLDS["camera_global_min"]
        ),
        "effect_global": (
            role_counts["effect"]
            >= DEVELOPMENT_THRESHOLDS["effect_global_min"]
        ),
    }
    return {
        "positive_by_family": dict(sorted(positive_by_family.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _ranked_candidates(
    similarity: np.ndarray,
    *,
    query_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    query_family = str(rows[query_index]["family"])
    order = sorted(
        (index for index in range(len(rows)) if index != query_index),
        key=lambda index: (
            -float(similarity[query_index, index]),
            str(rows[index]["iid"]),
        ),
    )
    result = []
    for rank, index in enumerate(order, 1):
        role = _row_role(rows[index])
        family = str(rows[index]["family"])
        if role == "positive" and family == query_family:
            relation = "same_family_positive"
        elif role == "positive":
            relation = "other_family_positive"
        else:
            relation = f"control_{role}"
        result.append(
            {
                "rank": rank,
                "iid": str(rows[index]["iid"]),
                "component_id": str(rows[index]["component_id"]),
                "family": family,
                "pilot_role": role,
                "relation": relation,
                "similarity": float(similarity[query_index, index]),
            }
        )
    return result


def _retrieval_for_seed(
    vectors: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    positives = [
        index for index, row in enumerate(rows) if _row_role(row) == "positive"
    ]
    by_family: dict[str, Any] = {}
    query_records: list[dict[str, Any]] = []
    per_family_queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in positives:
        family = str(rows[query]["family"])
        positive_candidates = [
            index for index in positives if index != query
        ]
        ranked_positive = sorted(
            positive_candidates,
            key=lambda index: (
                -float(similarity[query, index]),
                str(rows[index]["iid"]),
            ),
        )
        same_scores = [
            float(similarity[query, index])
            for index in positive_candidates
            if str(rows[index]["family"]) == family
        ]
        other_scores = [
            float(similarity[query, index])
            for index in positive_candidates
            if str(rows[index]["family"]) != family
        ]
        hit_at_1 = float(
            bool(ranked_positive)
            and str(rows[ranked_positive[0]]["family"]) == family
        )
        hit_at_3 = float(
            any(
                str(rows[index]["family"]) == family
                for index in ranked_positive[:3]
            )
        )
        pair_auroc = _auroc(same_scores, other_scores)
        margin = (
            float(np.mean(same_scores) - np.mean(other_scores))
            if same_scores and other_scores
            else None
        )
        record = {
            "iid": str(rows[query]["iid"]),
            "component_id": str(rows[query]["component_id"]),
            "family": family,
            "recall_at_1": hit_at_1,
            "recall_at_3": hit_at_3,
            "same_family_pair_auroc": pair_auroc,
            "similarity_margin": margin,
            "same_family_similarity": _scalar_summary(same_scores),
            "other_family_similarity": _scalar_summary(other_scores),
            "ranking": _ranked_candidates(
                similarity,
                query_index=query,
                rows=rows,
            ),
        }
        query_records.append(record)
        per_family_queries[family].append(record)

    for family in EXPECTED_FAMILIES:
        records = per_family_queries.get(family, [])
        by_family[family] = {
            "queries": len(records),
            "recall_at_1": _mean_or_none(
                [float(record["recall_at_1"]) for record in records]
            ),
            "recall_at_3": _mean_or_none(
                [float(record["recall_at_3"]) for record in records]
            ),
            "same_family_pair_auroc": _mean_or_none(
                [
                    float(record["same_family_pair_auroc"])
                    for record in records
                    if record["same_family_pair_auroc"] is not None
                ]
            ),
            "similarity_margin": _mean_or_none(
                [
                    float(record["similarity_margin"])
                    for record in records
                    if record["similarity_margin"] is not None
                ]
            ),
        }
    macro = {}
    for metric in (
        "recall_at_1",
        "recall_at_3",
        "same_family_pair_auroc",
        "similarity_margin",
    ):
        values = [
            float(by_family[family][metric])
            for family in EXPECTED_FAMILIES
            if by_family[family][metric] is not None
        ]
        macro[metric] = _mean_or_none(values)

    unique_same: list[float] = []
    unique_other: list[float] = []
    for left_position, left in enumerate(positives):
        for right in positives[left_position + 1 :]:
            score = float(similarity[left, right])
            if rows[left]["family"] == rows[right]["family"]:
                unique_same.append(score)
            else:
                unique_other.append(score)
    global_pair_auroc = _auroc(unique_same, unique_other)
    return {
        "by_family": by_family,
        "macro": macro,
        "unique_positive_pairs": {
            "same_family": len(unique_same),
            "other_family": len(unique_other),
            "same_family_vs_other_auroc": global_pair_auroc,
            "same_family_similarity": _scalar_summary(unique_same),
            "other_family_similarity": _scalar_summary(unique_other),
        },
        "queries": query_records,
        "_similarity": similarity,
    }


def _control_diagnostics(
    similarity: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive_indices = [
        index for index, row in enumerate(rows) if _row_role(row) == "positive"
    ]
    by_role: dict[str, Any] = {}
    for role in ("wrong", "static", "camera", "effect"):
        by_family: dict[str, Any] = {}
        for family in EXPECTED_FAMILIES:
            queries = [
                index
                for index in positive_indices
                if str(rows[index]["family"]) == family
            ]
            controls = [
                index
                for index, row in enumerate(rows)
                if _row_role(row) == role
                and (
                    role in {"camera", "effect"}
                    or str(row["family"]) == family
                )
            ]
            query_aurocs: list[float] = []
            query_margins: list[float] = []
            positive_scores_all: list[float] = []
            control_scores_all: list[float] = []
            for query in queries:
                peers = [
                    index
                    for index in positive_indices
                    if index != query and str(rows[index]["family"]) == family
                ]
                positive_scores = [
                    float(similarity[query, index]) for index in peers
                ]
                control_scores = [
                    float(similarity[query, index]) for index in controls
                ]
                auc = _auroc(positive_scores, control_scores)
                if auc is not None:
                    query_aurocs.append(auc)
                if positive_scores and control_scores:
                    query_margins.append(
                        float(
                            np.mean(positive_scores)
                            - np.mean(control_scores)
                        )
                    )
                positive_scores_all.extend(positive_scores)
                control_scores_all.extend(control_scores)
            by_family[family] = {
                "queries": len(queries),
                "controls": len(controls),
                "positive_vs_control_auroc": _mean_or_none(query_aurocs),
                "similarity_margin": _mean_or_none(query_margins),
                "positive_similarity": _scalar_summary(positive_scores_all),
                "control_similarity": _scalar_summary(control_scores_all),
            }
        macro_auc = _mean_or_none(
            [
                float(by_family[family]["positive_vs_control_auroc"])
                for family in EXPECTED_FAMILIES
                if by_family[family]["positive_vs_control_auroc"] is not None
            ]
        )
        macro_margin = _mean_or_none(
            [
                float(by_family[family]["similarity_margin"])
                for family in EXPECTED_FAMILIES
                if by_family[family]["similarity_margin"] is not None
            ]
        )
        by_role[role] = {
            "by_family": by_family,
            "macro_positive_vs_control_auroc": macro_auc,
            "macro_similarity_margin": macro_margin,
        }
    return {"by_control_role": by_role}


def _same_row_stability(
    canonical: np.ndarray,
    variant: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    by_family: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        if _row_role(row) != "positive":
            continue
        cosine = float(np.clip(canonical[index] @ variant[index], -1.0, 1.0))
        family = str(row["family"])
        by_family[family].append(cosine)
        records.append(
            {
                "iid": str(row["iid"]),
                "family": family,
                "same_target_cosine": cosine,
            }
        )
    family_means = {
        family: _mean_or_none(by_family.get(family, []))
        for family in EXPECTED_FAMILIES
    }
    return {
        "by_family_mean": family_means,
        "macro_mean": _mean_or_none(
            [
                float(value)
                for value in family_means.values()
                if value is not None
            ]
        ),
        "all_positive_rows": _scalar_summary(
            record["same_target_cosine"] for record in records
        ),
        "rows": records,
    }


def _cross_prompt_leakage(
    canonical: np.ndarray,
    cross: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positive_indices = [
        index for index, row in enumerate(rows) if _row_role(row) == "positive"
    ]
    for query in positive_indices:
        row = rows[query]
        actual = str(row["family"])
        shuffled = str(row.get("cross_family_shuffle_family"))
        actual_peers = [
            index
            for index in positive_indices
            if index != query and str(rows[index]["family"]) == actual
        ]
        shuffled_peers = [
            index
            for index in positive_indices
            if str(rows[index]["family"]) == shuffled
        ]
        if not actual_peers or not shuffled_peers:
            actual_mean = shuffled_mean = margin = None
            classified_actual = False
        else:
            actual_mean = float(
                np.mean([cross[query] @ canonical[index] for index in actual_peers])
            )
            shuffled_mean = float(
                np.mean(
                    [cross[query] @ canonical[index] for index in shuffled_peers]
                )
            )
            margin = actual_mean - shuffled_mean
            classified_actual = margin > 0.0
        same_target = float(
            np.clip(cross[query] @ canonical[query], -1.0, 1.0)
        )
        record = {
            "iid": str(row["iid"]),
            "actual_motion_family": actual,
            "shuffled_prompt_family": shuffled,
            "same_target_cosine": same_target,
            "similarity_to_actual_motion_family": actual_mean,
            "similarity_to_shuffled_prompt_family": shuffled_mean,
            "actual_minus_shuffled_margin": margin,
            "text_label_leakage_score": (
                -margin if margin is not None else None
            ),
            "classified_as_actual_motion_family": classified_actual,
        }
        records.append(record)
        by_family[actual].append(record)

    family_metrics: dict[str, Any] = {}
    for family in EXPECTED_FAMILIES:
        family_rows = by_family.get(family, [])
        family_metrics[family] = {
            "queries": len(family_rows),
            "same_target_cosine": _mean_or_none(
                [float(row["same_target_cosine"]) for row in family_rows]
            ),
            "actual_family_accuracy": _mean_or_none(
                [
                    float(bool(row["classified_as_actual_motion_family"]))
                    for row in family_rows
                ]
            ),
            "actual_minus_shuffled_margin": _mean_or_none(
                [
                    float(row["actual_minus_shuffled_margin"])
                    for row in family_rows
                    if row["actual_minus_shuffled_margin"] is not None
                ]
            ),
        }
    macro = {}
    for metric in (
        "same_target_cosine",
        "actual_family_accuracy",
        "actual_minus_shuffled_margin",
    ):
        macro[metric] = _mean_or_none(
            [
                float(family_metrics[family][metric])
                for family in EXPECTED_FAMILIES
                if family_metrics[family][metric] is not None
            ]
        )
    return {
        "definition": (
            "Cross-family instruction is applied to the identical source/target "
            "pair. Positive margin means the feature remains closer to actual "
            "target motion than to the shuffled text label."
        ),
        "by_family": family_metrics,
        "macro": macro,
        "rows": records,
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _seed_rank_agreement(
    similarities: Mapping[int, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive_queries = [
        index for index, row in enumerate(rows) if _row_role(row) == "positive"
    ]
    pairs = []
    correlations: list[float] = []
    for left_seed, right_seed in itertools.combinations(sorted(similarities), 2):
        per_query = []
        for query in positive_queries:
            candidates = [
                index for index in range(len(rows)) if index != query
            ]
            correlation = _spearman(
                [
                    float(similarities[left_seed][query, index])
                    for index in candidates
                ],
                [
                    float(similarities[right_seed][query, index])
                    for index in candidates
                ],
            )
            per_query.append(
                {
                    "iid": str(rows[query]["iid"]),
                    "rank_spearman": correlation,
                }
            )
            if correlation is not None:
                correlations.append(correlation)
        pair_values = [
            float(record["rank_spearman"])
            for record in per_query
            if record["rank_spearman"] is not None
        ]
        pairs.append(
            {
                "left_projection_seed": left_seed,
                "right_projection_seed": right_seed,
                "mean_query_rank_spearman": _mean_or_none(pair_values),
                "queries": per_query,
            }
        )
    return {
        "projection_seed_pairs": pairs,
        "all_query_pair_correlations": _scalar_summary(correlations),
        "mean": _mean_or_none(correlations),
        "policy": {
            "projected_coordinates_compared_across_seeds": False,
            "cross_seed_vector_dot_products_computed": False,
            "cross_seed_vector_cosines_computed": False,
            "cross_seed_feature_averaging_performed": False,
            "only_scalar_metrics_and_candidate_rankings_compared": True,
        },
    }


def _morphology_readout(
    vectors: Mapping[int, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    indices = [
        index for index, row in enumerate(rows) if _row_role(row) == "positive"
    ]
    labels: list[str] = []
    families: list[str] = []
    missing = []
    for index in indices:
        audit = rows[index].get("qwen_audit_binding", {}).get("audit", {})
        label = audit.get("subject_morphology")
        if not isinstance(label, str) or not label:
            missing.append(str(rows[index]["iid"]))
            label = ""
        labels.append(label)
        families.append(str(rows[index]["family"]))
    counts = Counter(label for label in labels if label)
    label_families: dict[str, set[str]] = defaultdict(set)
    for label, family in zip(labels, families):
        if label:
            label_families[label].add(family)
    reasons = []
    if missing:
        reasons.append("missing_qwen_subject_morphology")
    supported_labels = [label for label, count in counts.items() if count >= 2]
    if len(supported_labels) < 2:
        reasons.append("fewer_than_two_morphology_labels_with_two_examples")
    if any(len(label_families[label]) < 2 for label in supported_labels):
        reasons.append("morphology_is_confounded_with_motion_family")
    sufficient = not reasons
    per_seed: dict[str, Any] = {}
    passed = sufficient
    if sufficient:
        usable = [
            position
            for position, label in enumerate(labels)
            if label in supported_labels
        ]
        chance = 1.0 / len(supported_labels)
        for seed, all_vectors in sorted(vectors.items()):
            predictions: list[tuple[str, str]] = []
            for position in usable:
                source_index = indices[position]
                candidates = [
                    other for other in usable if other != position
                ]
                nearest = max(
                    candidates,
                    key=lambda other: (
                        float(
                            all_vectors[source_index]
                            @ all_vectors[indices[other]]
                        ),
                        str(rows[indices[other]]["iid"]),
                    ),
                )
                predictions.append((labels[position], labels[nearest]))
            recalls = []
            for label in supported_labels:
                relevant = [pair for pair in predictions if pair[0] == label]
                recalls.append(
                    sum(actual == predicted for actual, predicted in relevant)
                    / len(relevant)
                )
            balanced_accuracy = float(np.mean(recalls))
            excess = balanced_accuracy - chance
            seed_pass = (
                excess
                <= DEVELOPMENT_THRESHOLDS[
                    "morphology_readout_excess_over_chance_max"
                ]
            )
            passed = passed and seed_pass
            per_seed[str(seed)] = {
                "balanced_leave_one_out_1nn_accuracy": balanced_accuracy,
                "balanced_chance": chance,
                "excess_over_chance": excess,
                "readout_failed_as_required": seed_pass,
            }
    return {
        "available": sufficient,
        "sufficient_for_gate": sufficient,
        "insufficiency_reasons": sorted(set(reasons)),
        "label_counts": dict(sorted(counts.items())),
        "motion_families_by_label": {
            label: sorted(values)
            for label, values in sorted(label_families.items())
        },
        "per_projection_seed": per_seed,
        "passed": bool(passed),
    }


def _appearance_readout_placeholder(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # ``identity_appearance_change`` describes an edit failure, not a stable
    # source-content identity/category.  Treating it as an appearance label
    # would make a misleadingly easy "readout".
    has_stable_label = all(
        isinstance(row.get("source_appearance_class"), str)
        and bool(row.get("source_appearance_class"))
        for row in rows
        if _row_role(row) == "positive"
    )
    return {
        "available": False,
        "sufficient_for_gate": False,
        "passed": False,
        "stable_source_appearance_label_present": has_stable_label,
        "reason": (
            "No preregistered stable source-appearance/content label crosses "
            "both motion families; Qwen identity_appearance_change is an "
            "outcome-quality flag and is not reused as a content label."
        ),
    }


def _threshold_checks(
    *,
    per_seed: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Any],
    paraphrase: Mapping[str, Any],
    cross_prompt: Mapping[str, Any],
    rank_agreement: Mapping[str, Any],
    morphology: Mapping[str, Any],
    appearance: Mapping[str, Any],
) -> dict[str, Any]:
    seed_checks: dict[str, Any] = {}
    for seed, values in sorted(per_seed.items(), key=lambda pair: int(pair[0])):
        macro = values["retrieval"]["macro"]
        controls = values["controls"]["by_control_role"]
        retrieval_checks = {
            "macro_recall_at_1": (
                macro["recall_at_1"] is not None
                and macro["recall_at_1"]
                >= DEVELOPMENT_THRESHOLDS["macro_recall_at_1_min"]
            ),
            "macro_recall_at_3": (
                macro["recall_at_3"] is not None
                and macro["recall_at_3"]
                >= DEVELOPMENT_THRESHOLDS["macro_recall_at_3_min"]
            ),
            "macro_same_family_pair_auroc": (
                macro["same_family_pair_auroc"] is not None
                and macro["same_family_pair_auroc"]
                >= DEVELOPMENT_THRESHOLDS[
                    "macro_same_family_pair_auroc_min"
                ]
            ),
            "macro_similarity_margin": (
                macro["similarity_margin"] is not None
                and macro["similarity_margin"]
                >= DEVELOPMENT_THRESHOLDS["macro_similarity_margin_min"]
            ),
        }
        control_checks = {}
        for role in ("static", "camera", "effect"):
            record = controls[role]
            control_checks[role] = (
                record["macro_positive_vs_control_auroc"] is not None
                and record["macro_positive_vs_control_auroc"]
                >= DEVELOPMENT_THRESHOLDS[
                    "positive_vs_control_auroc_min"
                ]
                and record["macro_similarity_margin"] is not None
                and record["macro_similarity_margin"]
                >= DEVELOPMENT_THRESHOLDS[
                    "positive_vs_control_margin_min"
                ]
            )
        seed_checks[seed] = {
            "retrieval": retrieval_checks,
            "controls": control_checks,
            "passed": all(retrieval_checks.values())
            and all(control_checks.values()),
        }

    paraphrase_checks = {}
    if paraphrase.get("available") is True:
        for seed, record in paraphrase["per_projection_seed"].items():
            paraphrase_checks[seed] = (
                record["macro_mean"] is not None
                and record["macro_mean"]
                >= DEVELOPMENT_THRESHOLDS[
                    "paraphrase_same_target_cosine_min"
                ]
            )
    paraphrase_pass = bool(paraphrase_checks) and all(
        paraphrase_checks.values()
    )

    cross_checks = {}
    if cross_prompt.get("available") is True:
        for seed, record in cross_prompt["per_projection_seed"].items():
            macro = record["macro"]
            cross_checks[seed] = {
                "same_target_cosine": (
                    macro["same_target_cosine"] is not None
                    and macro["same_target_cosine"]
                    >= DEVELOPMENT_THRESHOLDS[
                        "cross_prompt_same_target_cosine_min"
                    ]
                ),
                "actual_family_accuracy": (
                    macro["actual_family_accuracy"] is not None
                    and macro["actual_family_accuracy"]
                    >= DEVELOPMENT_THRESHOLDS[
                        "cross_prompt_actual_family_accuracy_min"
                    ]
                ),
                "actual_family_margin": (
                    macro["actual_minus_shuffled_margin"] is not None
                    and macro["actual_minus_shuffled_margin"]
                    >= DEVELOPMENT_THRESHOLDS[
                        "cross_prompt_actual_family_margin_min"
                    ]
                ),
            }
    cross_pass = bool(cross_checks) and all(
        all(record.values()) for record in cross_checks.values()
    )
    rank_value = rank_agreement.get("mean")
    rank_pass = (
        rank_value is not None
        and rank_value
        >= DEVELOPMENT_THRESHOLDS["seed_rank_agreement_min"]
    )
    checks = {
        "balanced_support": support.get("passed") is True,
        "every_projection_seed": bool(seed_checks)
        and all(record["passed"] for record in seed_checks.values()),
        "canonical_original_paraphrase": paraphrase_pass,
        "cross_prompt_text_leakage": cross_pass,
        "projection_seed_rank_agreement": rank_pass,
        "morphology_readout_failed_as_required": (
            morphology.get("sufficient_for_gate") is True
            and morphology.get("passed") is True
        ),
        "appearance_readout_failed_as_required": (
            appearance.get("sufficient_for_gate") is True
            and appearance.get("passed") is True
        ),
    }
    return {
        "by_projection_seed": seed_checks,
        "paraphrase_by_projection_seed": paraphrase_checks,
        "cross_prompt_by_projection_seed": cross_checks,
        "global": checks,
        "failed": sorted(key for key, value in checks.items() if not value),
        "passed": all(checks.values()),
    }


def evaluate_retrieval(
    *,
    pilot_dir: str | Path,
    artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Evaluate validated feature artifacts without reading any video bytes."""

    if set(artifacts) - set(ARTIFACT_TAGS):
        raise R10BBerniniRetrievalAuditError(
            f"unsupported artifact tags: {sorted(set(artifacts) - set(ARTIFACT_TAGS))}"
        )
    if "canonical" not in artifacts:
        raise R10BBerniniRetrievalAuditError(
            "canonical Bernini feature artifact is required"
        )
    pilot_rows, pilot_summary, pilot_binding = _pilot_commit(pilot_dir)
    loaded = {
        tag: _load_artifact(
            tag=tag,
            artifact_dir=artifacts[tag],
            pilot_rows=pilot_rows,
            pilot_manifest_sha256=pilot_binding["manifest_sha256"],
        )
        for tag in ARTIFACT_TAGS
        if tag in artifacts
    }
    canonical = loaded["canonical"]
    reference_contract = canonical["contract"]
    reference_source_binding = canonical["variant_manifest"][
        "source_binding_sha256"
    ]
    for tag, artifact in loaded.items():
        if artifact["contract"] != reference_contract:
            raise R10BBerniniRetrievalAuditError(
                f"{tag} artifact measurement/checkpoint contract differs"
            )
        if (
            artifact["projection_seeds"]
            != canonical["projection_seeds"]
            or artifact["projection_dimension"]
            != canonical["projection_dimension"]
        ):
            raise R10BBerniniRetrievalAuditError(
                f"{tag} projection geometry differs"
            )
        if (
            artifact["variant_manifest"]["source_binding_sha256"]
            != reference_source_binding
        ):
            raise R10BBerniniRetrievalAuditError(
                f"{tag} source/target/component binding differs"
            )

    support = _support(
        pilot_rows,
        bool(pilot_summary["balanced_pilot_ready"]),
    )
    per_seed: dict[str, Any] = {}
    similarities: dict[int, np.ndarray] = {}
    for seed in canonical["projection_seeds"]:
        retrieval = _retrieval_for_seed(
            canonical["vectors"][seed],
            pilot_rows,
        )
        similarity = retrieval.pop("_similarity")
        similarities[seed] = similarity
        per_seed[str(seed)] = {
            "retrieval": retrieval,
            "controls": _control_diagnostics(similarity, pilot_rows),
        }

    paraphrase: dict[str, Any] = {
        "available": "original" in loaded,
        "required_for_development_signal": True,
        "per_projection_seed": {},
    }
    if "original" in loaded:
        paraphrase["per_projection_seed"] = {
            str(seed): _same_row_stability(
                canonical["vectors"][seed],
                loaded["original"]["vectors"][seed],
                pilot_rows,
            )
            for seed in canonical["projection_seeds"]
        }
    else:
        paraphrase["insufficiency_reason"] = (
            "original prompt artifact was not supplied"
        )

    cross_prompt: dict[str, Any] = {
        "available": "cross_family" in loaded,
        "required_for_development_signal": True,
        "per_projection_seed": {},
    }
    if "cross_family" in loaded:
        cross_prompt["per_projection_seed"] = {
            str(seed): _cross_prompt_leakage(
                canonical["vectors"][seed],
                loaded["cross_family"]["vectors"][seed],
                pilot_rows,
            )
            for seed in canonical["projection_seeds"]
        }
    else:
        cross_prompt["insufficiency_reason"] = (
            "cross-family fixed-target prompt artifact was not supplied"
        )

    rank_agreement = _seed_rank_agreement(similarities, pilot_rows)
    morphology = _morphology_readout(
        canonical["vectors"],
        pilot_rows,
    )
    appearance = _appearance_readout_placeholder(pilot_rows)
    checks = _threshold_checks(
        per_seed=per_seed,
        support=support,
        paraphrase=paraphrase,
        cross_prompt=cross_prompt,
        rank_agreement=rank_agreement,
        morphology=morphology,
        appearance=appearance,
    )
    development_signal = checks["passed"]
    artifact_bindings = []
    for tag in ARTIFACT_TAGS:
        if tag not in loaded:
            continue
        artifact = loaded[tag]
        artifact_bindings.append(
            {
                "tag": tag,
                "path": artifact["path"],
                "artifact_digest": artifact["artifact_digest"],
                "backend_validation": artifact["validation"],
                "variant_manifest": artifact["variant_manifest"],
                "contract_sha256": object_digest(artifact["contract"]),
                "primary_feature_names": [
                    f"{PRIMARY_FEATURE}__p{seed}"
                    for seed in artifact["projection_seeds"]
                ],
            }
        )

    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "artifact_kind": "immutable_read_only_retrieval_audit",
        "scope": (
            "controlled Bernini tangent retrieval development audit only; "
            "not representation promotion, rendering, generation, or training"
        ),
        "primary_feature": PRIMARY_FEATURE,
        "pilot": pilot_binding,
        "feature_artifacts": artifact_bindings,
        "shared_artifact_contract_sha256": object_digest(reference_contract),
        "projection": {
            "seeds": list(canonical["projection_seeds"]),
            "dimension_per_seed": canonical["projection_dimension"],
            "coordinates_compared_only_within_same_seed": True,
            "cross_seed_vector_comparison": False,
        },
        "support": support,
        "development_thresholds": dict(DEVELOPMENT_THRESHOLDS),
        "per_projection_seed": per_seed,
        "paraphrase_stability": paraphrase,
        "cross_prompt_text_leakage": cross_prompt,
        "projection_seed_rank_agreement": rank_agreement,
        "leakage_readouts": {
            "morphology": morphology,
            "appearance": appearance,
        },
        "development_checks": checks,
        "decision": {
            "classification": (
                "development_signal_requires_sigma_noise_dimension_holdout"
                if development_signal
                else "no_development_signal"
            ),
            "development_signal_requires_sigma_noise_dimension_holdout": (
                development_signal
            ),
            "remaining_mandatory_holdouts": [
                "multiple scheduler sigma values",
                "iid spatiotemporal versus temporal-broadcast noise",
                "higher CountSketch dimensions",
                "independent content/action holdout",
                "appearance and morphology readouts with deconfounded labels",
            ],
        },
        "limitations": {
            "qwen_labels_are_human_labels": False,
            "development_signal_is_formal_evidence": False,
            "single_sigma_noise_dimension_can_promote": False,
            "missing_balanced_controls_can_promote": False,
            "prompt_variant_coordinates_compared_only_after_exact_contract_binding": True,
        },
        "media_io": {
            "video_files_read": 0,
            "video_files_copied": 0,
            "video_files_rendered": 0,
            "feature_arrays_read_only": True,
        },
        "authorization": dict(_AUTHORIZATION),
        "formal_evidence": False,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "generation_authorized": False,
        "editor_training_authorized": False,
        "training_authorized": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_retrieval_audit(
    *,
    pilot_dir: str | Path,
    artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate and atomically publish a new immutable JSON-only commit."""

    result = evaluate_retrieval(pilot_dir=pilot_dir, artifacts=artifacts)
    output = Path(output_dir)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.work.", dir=parent)
    )
    try:
        audit_bytes = _pretty_bytes(result)
        audit_path = work / AUDIT_NAME
        with audit_path.open("xb") as handle:
            handle.write(audit_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "files": {
                AUDIT_NAME: {
                    "sha256": _sha256_bytes(audit_bytes),
                    "bytes": len(audit_bytes),
                }
            },
            "artifact_digest": object_digest(
                {
                    AUDIT_NAME: {
                        "sha256": _sha256_bytes(audit_bytes),
                        "bytes": len(audit_bytes),
                    }
                }
            ),
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        }
        done_bytes = _pretty_bytes(done)
        with (work / DONE_NAME).open("xb") as handle:
            handle.write(done_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(work)
        if output.exists():
            raise FileExistsError(output)
        os.replace(work, output)
        work = None
        for path in output.iterdir():
            path.chmod(0o444)
        output.chmod(0o555)
        _fsync_directory(parent)
    finally:
        if work is not None and work.exists():
            shutil.rmtree(work)
    return result


def validate_published_retrieval_audit(
    output_dir: str | Path,
    *,
    revalidate_sources: bool = True,
) -> dict[str, Any]:
    """Validate immutable closure, gates, and optionally every source commit."""

    output = Path(output_dir).expanduser()
    if output.is_symlink() or not output.is_dir():
        raise R10BBerniniRetrievalAuditError(
            f"retrieval audit directory is missing: {output}"
        )
    observed = sorted(path.name for path in output.iterdir())
    if observed != sorted(OUTPUT_NAMES):
        raise R10BBerniniRetrievalAuditError(
            f"retrieval audit closure differs: {observed}"
        )
    audit, audit_raw = _read_json_object(
        output / AUDIT_NAME,
        field="published retrieval audit",
    )
    done, _done_raw = _read_json_object(
        output / DONE_NAME,
        field="published retrieval done",
    )
    payload = {
        "sha256": _sha256_bytes(audit_raw),
        "bytes": len(audit_raw),
    }
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("files") != {AUDIT_NAME: payload}
        or done.get("artifact_digest")
        != object_digest({AUDIT_NAME: payload})
    ):
        raise R10BBerniniRetrievalAuditError(
            "retrieval audit done binding differs"
        )
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("status") != "complete"
        or audit.get("artifact_kind")
        != "immutable_read_only_retrieval_audit"
        or audit.get("primary_feature") != PRIMARY_FEATURE
    ):
        raise R10BBerniniRetrievalAuditError(
            "retrieval audit schema/primary feature differs"
        )
    _require_false_authorization(
        audit.get("authorization"),
        field="retrieval audit",
    )
    for field in (
        "formal_evidence",
        "representation_gate_passed",
        "renderer_probe_authorized",
        "generation_authorized",
        "editor_training_authorized",
        "training_authorized",
    ):
        if audit.get(field) is not False:
            raise R10BBerniniRetrievalAuditError(
                f"retrieval audit false gate differs: {field}"
            )
    _false_validation_gates(done, field="retrieval audit done")
    if (
        audit.get("media_io")
        != {
            "video_files_read": 0,
            "video_files_copied": 0,
            "video_files_rendered": 0,
            "feature_arrays_read_only": True,
        }
        or audit.get("projection", {}).get(
            "cross_seed_vector_comparison"
        )
        is not False
        or audit.get("projection_seed_rank_agreement", {})
        .get("policy", {})
        .get("cross_seed_vector_dot_products_computed")
        is not False
    ):
        raise R10BBerniniRetrievalAuditError(
            "retrieval audit media/seed safety policy differs"
        )
    decision = audit.get("decision", {})
    development = decision.get(
        "development_signal_requires_sigma_noise_dimension_holdout"
    )
    if not isinstance(development, bool) or decision.get("classification") != (
        "development_signal_requires_sigma_noise_dimension_holdout"
        if development
        else "no_development_signal"
    ):
        raise R10BBerniniRetrievalAuditError(
            "retrieval audit development classification differs"
        )

    if revalidate_sources:
        pilot_rows, _pilot_summary, pilot_binding = _pilot_commit(
            audit.get("pilot", {}).get("path", "")
        )
        if pilot_binding != audit.get("pilot"):
            raise R10BBerniniRetrievalAuditError(
                "retrieval audit pilot source changed"
            )
        expected_ids = [str(row["iid"]) for row in pilot_rows]
        for artifact in audit.get("feature_artifacts", []):
            if artifact.get("tag") not in ARTIFACT_TAGS:
                raise R10BBerniniRetrievalAuditError(
                    "retrieval audit artifact tag differs"
                )
            validation = bernini.validate_published_extract(
                artifact.get("path", "")
            )
            if validation != artifact.get("backend_validation"):
                raise R10BBerniniRetrievalAuditError(
                    "retrieval audit feature source changed"
                )
            if validation.get("rows") != len(expected_ids):
                raise R10BBerniniRetrievalAuditError(
                    "retrieval audit feature row count changed"
                )
    return {
        "status": "VALID",
        "output_dir": str(output.resolve()),
        "artifact_digest": done["artifact_digest"],
        "pilot_rows": audit.get("pilot", {}).get("rows"),
        "feature_artifacts": len(audit.get("feature_artifacts", [])),
        "development_signal_requires_sigma_noise_dimension_holdout": (
            development
        ),
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }


def _parse_artifacts(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise R10BBerniniRetrievalAuditError(
                "--artifact must use TAG=PATH"
            )
        tag, path = value.split("=", 1)
        if tag not in ARTIFACT_TAGS or tag in result or not path:
            raise R10BBerniniRetrievalAuditError(
                f"invalid/duplicate artifact binding: {value}"
            )
        result[tag] = Path(path)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit R10B Bernini controlled-pilot retrieval."
    )
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="TAG=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        result = validate_published_retrieval_audit(args.output_dir)
    else:
        if args.pilot_dir is None:
            raise R10BBerniniRetrievalAuditError(
                "--pilot-dir is required unless --validate-only is used"
            )
        artifacts = _parse_artifacts(args.artifact)
        publish_retrieval_audit(
            pilot_dir=args.pilot_dir,
            artifacts=artifacts,
            output_dir=args.output_dir,
        )
        result = validate_published_retrieval_audit(args.output_dir)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_TAGS",
    "AUDIT_SCHEMA",
    "DEVELOPMENT_THRESHOLDS",
    "PRIMARY_FEATURE",
    "R10BBerniniRetrievalAuditError",
    "evaluate_retrieval",
    "publish_retrieval_audit",
    "validate_controlled_pilot_commit",
    "validate_published_retrieval_audit",
]
