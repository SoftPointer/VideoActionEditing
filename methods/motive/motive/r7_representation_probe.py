"""Leakage-resistant R7-A observed-event representation diagnostics.

This module evaluates whether the *observed target event* extracted by the
R7 temporal teacher carries action-family information.  It is intentionally
not a generation or editing experiment.  Three frozen representations are
compared on one common cohort:

``target_actor_teacher``
    The target-side 224-D actor/event teacher embedding (primary).

``target_pooled_dino``
    Mean-pooled target DINO CLS features (appearance baseline).

``target_camera_trajectory``
    Flattened target camera trajectory (nuisance baseline).

Evaluation uses a train-only nearest-neighbour bank.  A query can never
retrieve its own IID, logical pair, or visual component, and source features
are never admitted to the reference bank.  The evaluator reports micro and
macro-family R@1/R@5, per-family metrics, visual-component bootstrap
confidence intervals, coverage, and a train-label permutation null.

Formal scope is deliberately fail-closed.  Pseudo or incompletely
provenanced labels always yield ``INSUFFICIENT``.  ``FORMAL_EVALUABLE`` is
possible only when every retained train/evaluation label has explicit human
annotation provenance and the supplied R7 visual split has a passing audit
with fresh evaluation assignments.  Even then this module never declares a
pass and never authorizes generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r7_preflight_extract import validate_final
from .r7_visual_split import (
    R7_FRESHNESS_POLICY_VERSION,
    R7_VISUAL_SPLIT_SCHEMA,
)


R7_REPRESENTATION_PROBE_SCHEMA = "motive-r7-representation-probe-v1"
R7_REPRESENTATION_SUMMARY_SCHEMA = (
    "motive-r7-representation-probe-summary-v1"
)
R7_REPRESENTATION_QUERY_SCHEMA = (
    "motive-r7-representation-probe-query-v1"
)
R7_REPRESENTATION_DONE_SCHEMA = "motive-r7-representation-probe-done-v1"
R7_RETRIEVAL_PROTOCOL = (
    "train-target-only-cosine-nn-exclude-iid-pair-component-v1"
)
R7_BOOTSTRAP_PROTOCOL = "evaluation-visual-component-cluster-bootstrap-v1"
R7_NULL_PROTOCOL = "train-bank-label-permutation-v1"

SUMMARY_NAME = "summary.json"
PER_QUERY_NAME = "per_query.jsonl"
DONE_NAME = "done.json"
VALID_SPLITS = frozenset({"train", "validation", "test"})
MODALITIES = (
    "target_actor_teacher",
    "target_pooled_dino",
    "target_camera_trajectory",
)
DEFAULT_SEED = 260108831


class RepresentationProbeError(ValueError):
    """The probe contract is invalid or scientifically unevaluable."""


@dataclass(frozen=True)
class RepresentationProbeConfig:
    """Fixed diagnostic protocol and minimum support requirements."""

    eval_splits: tuple[str, ...] = ("test",)
    minimum_total_train_references: int = 20
    minimum_train_references_per_family: int = 5
    minimum_train_components_per_family: int = 3
    minimum_eval_queries_per_family: int = 2
    minimum_eval_components: int = 2
    bootstrap_repetitions: int = 1000
    permutation_repetitions: int = 1000
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if (
            not self.eval_splits
            or len(set(self.eval_splits)) != len(self.eval_splits)
            or any(split not in {"validation", "test"} for split in self.eval_splits)
        ):
            raise RepresentationProbeError(
                "eval_splits must be a unique non-empty subset of "
                "{'validation','test'}"
            )
        for name in (
            "minimum_total_train_references",
            "minimum_train_references_per_family",
            "minimum_train_components_per_family",
            "minimum_eval_queries_per_family",
            "minimum_eval_components",
            "bootstrap_repetitions",
            "permutation_repetitions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RepresentationProbeError(
                    f"{name} must be a positive integer"
                )
        if self.minimum_total_train_references < 5:
            raise RepresentationProbeError(
                "minimum_total_train_references must be at least 5 for R@5"
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed >= 2**32
        ):
            raise RepresentationProbeError(
                "seed must be an integer in [0, 2**32)"
            )


@dataclass(frozen=True)
class _Assignment:
    iid: str
    split: str
    component_id: str
    evaluation_fresh: bool


@dataclass(frozen=True)
class _Label:
    iid: str
    pair_id: str
    family: str
    provenance_kind: str
    formal_human: bool


@dataclass(frozen=True)
class _Example:
    iid: str
    pair_id: str
    family: str
    provenance_kind: str
    formal_human: bool
    split: str
    component_id: str
    evaluation_fresh: bool
    features: Mapping[str, np.ndarray]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _visual_split_digest(value: Any) -> str:
    """Match r7_visual_split's canonical digest byte-for-byte."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
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
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepresentationProbeError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RepresentationProbeError(
                    f"{path}:{line_number} is blank"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RepresentationProbeError(
                    f"{path}:{line_number} is not a JSON object"
                )
            rows.append(value)
    if not rows:
        raise RepresentationProbeError(f"{path} contains no rows")
    return rows


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RepresentationProbeError(f"{name} must be a trimmed non-empty string")
    if "\x00" in value:
        raise RepresentationProbeError(f"{name} must not contain NUL")
    return value


def _parse_visual_split(
    path: Path,
) -> tuple[dict[str, _Assignment], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    payload = _load_json(resolved)
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise RepresentationProbeError(
            "visual split must contain a non-empty assignments list"
        )
    assignments: dict[str, _Assignment] = {}
    component_splits: dict[str, str] = {}
    for index, raw in enumerate(raw_assignments):
        if not isinstance(raw, Mapping):
            raise RepresentationProbeError(
                f"visual split assignment {index} is not an object"
            )
        iid = _nonempty_string(raw.get("iid"), name=f"assignment[{index}].iid")
        split = _nonempty_string(
            raw.get("split"), name=f"assignment[{index}].split"
        )
        if split not in VALID_SPLITS:
            raise RepresentationProbeError(
                f"assignment[{index}].split is unsupported: {split}"
            )
        component = _nonempty_string(
            raw.get("component_id"),
            name=f"assignment[{index}].component_id",
        )
        fresh = raw.get("evaluation_fresh")
        if type(fresh) is not bool:
            raise RepresentationProbeError(
                f"assignment[{index}].evaluation_fresh must be boolean"
            )
        if iid in assignments:
            raise RepresentationProbeError(
                f"visual split duplicates iid={iid}"
            )
        previous = component_splits.setdefault(component, split)
        if previous != split:
            raise RepresentationProbeError(
                f"visual component {component} crosses {previous}/{split}"
            )
        assignments[iid] = _Assignment(
            iid=iid,
            split=split,
            component_id=component,
            evaluation_fresh=fresh,
        )
    audit = payload.get("audit")
    provenance = payload.get("provenance")
    audit_passed = isinstance(audit, Mapping) and audit.get("passed") is True
    components = payload.get("components")
    edges = payload.get("edges")
    provenance_digest_valid = False
    artifact_digests_valid = False
    if isinstance(provenance, Mapping):
        provenance_without_digest = dict(provenance)
        recorded_provenance_digest = provenance_without_digest.pop(
            "provenance_digest", None
        )
        provenance_digest_valid = (
            isinstance(recorded_provenance_digest, str)
            and recorded_provenance_digest
            == _visual_split_digest(provenance_without_digest)
        )
        artifact_digests_valid = (
            isinstance(components, list)
            and isinstance(edges, list)
            and provenance.get("assignments_digest")
            == _visual_split_digest(raw_assignments)
            and provenance.get("components_digest")
            == _visual_split_digest(components)
            and provenance.get("edges_digest") == _visual_split_digest(edges)
            and isinstance(audit, Mapping)
            and provenance.get("audit_digest")
            == _visual_split_digest(dict(audit))
        )
    audit_structure_valid = (
        isinstance(audit, Mapping)
        and audit.get("samples") == len(assignments)
        and audit.get("cross_split_component_ids") == []
        and audit.get("cross_split_relation_edges") == []
        and audit.get("assignment_component_mismatches") == []
        and audit.get("stable_split_mismatches") == []
        and audit.get("seen_component_evaluation_iids") == []
    )
    provenance_valid = (
        isinstance(provenance, Mapping)
        and provenance.get("schema_version") == R7_VISUAL_SPLIT_SCHEMA
        and provenance.get("freshness_policy_version")
        == R7_FRESHNESS_POLICY_VERSION
        and provenance_digest_valid
        and artifact_digests_valid
        and audit_structure_valid
    )
    metadata = {
        "path": str(resolved),
        "sha256": _file_digest(resolved),
        "assignment_count": len(assignments),
        "component_count": len(component_splits),
        "audit_passed": audit_passed,
        "audit_structure_valid": audit_structure_valid,
        "artifact_digests_valid": artifact_digests_valid,
        "provenance_digest_valid": provenance_digest_valid,
        "formal_provenance_valid": provenance_valid,
        "formal_fresh_split_attested": bool(audit_passed and provenance_valid),
    }
    return assignments, metadata


def _parse_label_provenance(
    value: Any,
    *,
    iid: str,
) -> tuple[str, bool]:
    if not isinstance(value, Mapping):
        raise RepresentationProbeError(
            f"label iid={iid} provenance must be an explicit object"
        )
    kind = _nonempty_string(
        value.get("kind"), name=f"label iid={iid} provenance.kind"
    ).lower()
    if kind == "human":
        annotation_id = value.get("annotation_id")
        annotator_id = value.get("annotator_id")
        formal = (
            isinstance(annotation_id, str)
            and bool(annotation_id.strip())
            and annotation_id.strip() == annotation_id
            and isinstance(annotator_id, str)
            and bool(annotator_id.strip())
            and annotator_id.strip() == annotator_id
        )
        return kind, formal
    if kind == "pseudo":
        method = value.get("method")
        if (
            not isinstance(method, str)
            or not method.strip()
            or method.strip() != method
        ):
            raise RepresentationProbeError(
                f"pseudo label iid={iid} must record provenance.method"
            )
        return kind, False
    raise RepresentationProbeError(
        f"label iid={iid} provenance.kind must be human or pseudo"
    )


def _parse_labels(
    path: Path,
) -> tuple[dict[str, _Label], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    raw_rows = _load_jsonl(resolved)
    labels: dict[str, _Label] = {}
    seen_iids: set[str] = set()
    incomplete_rows = 0
    for index, raw in enumerate(raw_rows):
        iid = _nonempty_string(raw.get("iid"), name=f"label[{index}].iid")
        if iid in seen_iids:
            raise RepresentationProbeError(f"label manifest duplicates iid={iid}")
        seen_iids.add(iid)
        family_value = raw.get("action_signature")
        provenance_value = raw.get("label_provenance")
        # Missing family/provenance means "not explicitly labelled", not an
        # invitation to fall back to an inspected preflight pseudo-label.
        if family_value is None or provenance_value is None:
            incomplete_rows += 1
            continue
        family = _nonempty_string(
            family_value, name=f"label iid={iid} action_signature"
        )
        kind, formal_human = _parse_label_provenance(
            provenance_value,
            iid=iid,
        )
        pair_value = raw.get("pair_id", iid)
        pair_id = _nonempty_string(
            pair_value, name=f"label iid={iid} pair_id"
        )
        labels[iid] = _Label(
            iid=iid,
            pair_id=pair_id,
            family=family,
            provenance_kind=kind,
            formal_human=formal_human,
        )
    metadata = {
        "path": str(resolved),
        "sha256": _file_digest(resolved),
        "rows": len(raw_rows),
        "explicit_label_rows": len(labels),
        "incomplete_label_rows": incomplete_rows,
    }
    return labels, metadata


def _normalise_rows(matrix: np.ndarray, *, allow_zero: bool) -> np.ndarray:
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 1 or not np.isfinite(raw).all():
        raise RepresentationProbeError(
            "representation matrix must be finite and two-dimensional"
        )
    norms = np.linalg.norm(raw, axis=1)
    if not allow_zero and bool((norms <= 1e-12).any()):
        raise RepresentationProbeError(
            "a required representation contains a zero vector"
        )
    output = np.zeros_like(raw)
    nonzero = norms > 1e-12
    output[nonzero] = raw[nonzero] / norms[nonzero, None]
    return output.astype(np.float32)


def _extract_feature_rows(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    teacher = np.asarray(arrays["target_teacher_embedding"])[indices]
    dino_frames = np.asarray(arrays["target_dino_cls"])[indices]
    if dino_frames.ndim != 3:
        raise RepresentationProbeError(
            "target_dino_cls must have shape [N,frames,dim]"
        )
    pooled_dino = np.mean(dino_frames.astype(np.float64), axis=1)
    camera = np.asarray(arrays["target_camera_trajectory"])[indices]
    if camera.ndim != 3:
        raise RepresentationProbeError(
            "target_camera_trajectory must have shape [N,phase,4]"
        )
    return {
        "target_actor_teacher": _normalise_rows(
            teacher.reshape(len(indices), -1), allow_zero=False
        ),
        "target_pooled_dino": _normalise_rows(
            pooled_dino, allow_zero=False
        ),
        # A stationary camera is a meaningful nuisance condition.  Its zero
        # vector is retained and has zero cosine similarity to every bank row.
        "target_camera_trajectory": _normalise_rows(
            camera.reshape(len(indices), -1), allow_zero=True
        ),
    }


def _load_preflight_examples(
    final_directories: Sequence[Path],
    assignments: Mapping[str, _Assignment],
    labels: Mapping[str, _Label],
) -> tuple[list[_Example], dict[str, Any], list[dict[str, Any]]]:
    if not final_directories:
        raise RepresentationProbeError(
            "at least one preflight final directory is required"
        )
    examples: list[_Example] = []
    seen: set[str] = set()
    provenance: list[dict[str, Any]] = []
    counters = {
        "preflight_rows": 0,
        "filter_order": [
            "visual_split_assignment",
            "explicit_action_signature_and_label_provenance",
            "target_base_valid",
            "target_dino_valid_for_shared_comparison_cohort",
        ],
        "missing_split_assignment": 0,
        "missing_explicit_label": 0,
        "target_base_invalid": 0,
        "target_dino_invalid": 0,
        "common_cohort": 0,
    }
    for raw_directory in final_directories:
        directory = raw_directory.expanduser().resolve(strict=True)
        validated = validate_final(directory)
        rows = validated["rows"]
        arrays = validated["arrays"]
        counters["preflight_rows"] += len(rows)
        provenance.append(
            {
                "directory": str(directory),
                "done_sha256": _file_digest(directory / "done.json"),
                "manifest_sha256": _file_digest(directory / "manifest.jsonl"),
                "archive_sha256": _file_digest(directory / "features.npz"),
                "rows": len(rows),
            }
        )
        retained_indices: list[int] = []
        retained_meta: list[tuple[str, _Assignment, _Label]] = []
        base = np.asarray(arrays["target_base_valid"], dtype=bool)
        dino = np.asarray(arrays["target_dino_valid"], dtype=bool)
        for index, row in enumerate(rows):
            iid = _nonempty_string(
                row.get("iid"), name=f"preflight row {index} iid"
            )
            if iid in seen:
                raise RepresentationProbeError(
                    f"preflight inputs duplicate iid={iid}"
                )
            seen.add(iid)
            assignment = assignments.get(iid)
            if assignment is None:
                counters["missing_split_assignment"] += 1
                continue
            label = labels.get(iid)
            if label is None:
                counters["missing_explicit_label"] += 1
                continue
            if not bool(base[index]):
                counters["target_base_invalid"] += 1
                continue
            if not bool(dino[index]):
                counters["target_dino_invalid"] += 1
                continue
            retained_indices.append(index)
            retained_meta.append((iid, assignment, label))
        if retained_indices:
            indices = np.asarray(retained_indices, dtype=np.int64)
            feature_rows = _extract_feature_rows(arrays, indices)
            for local_index, (iid, assignment, label) in enumerate(retained_meta):
                examples.append(
                    _Example(
                        iid=iid,
                        pair_id=label.pair_id,
                        family=label.family,
                        provenance_kind=label.provenance_kind,
                        formal_human=label.formal_human,
                        split=assignment.split,
                        component_id=assignment.component_id,
                        evaluation_fresh=assignment.evaluation_fresh,
                        features={
                            name: feature_rows[name][local_index]
                            for name in MODALITIES
                        },
                    )
                )
    counters["common_cohort"] = len(examples)
    if not examples:
        raise RepresentationProbeError(
            "no rows have split assignment, explicit label, target "
            "base_valid, and target DINO validity"
        )
    return examples, counters, provenance


def _counts(values: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return dict(sorted(output.items()))


def _validate_support(
    train: Sequence[_Example],
    queries: Sequence[_Example],
    *,
    config: RepresentationProbeConfig,
) -> dict[str, Any]:
    if len(train) < config.minimum_total_train_references:
        raise RepresentationProbeError(
            "insufficient train references: "
            f"{len(train)} < {config.minimum_total_train_references}"
        )
    if not queries:
        raise RepresentationProbeError(
            "selected evaluation splits contain no eligible queries"
        )
    eval_components = {example.component_id for example in queries}
    if len(eval_components) < config.minimum_eval_components:
        raise RepresentationProbeError(
            "insufficient evaluation visual components: "
            f"{len(eval_components)} < {config.minimum_eval_components}"
        )
    train_counts = _counts(example.family for example in train)
    eval_counts = _counts(example.family for example in queries)
    train_components: dict[str, set[str]] = {}
    for example in train:
        train_components.setdefault(example.family, set()).add(
            example.component_id
        )
    failures: list[str] = []
    for family, query_count in eval_counts.items():
        reference_count = train_counts.get(family, 0)
        component_count = len(train_components.get(family, set()))
        if reference_count < config.minimum_train_references_per_family:
            failures.append(
                f"{family}:train_refs={reference_count}"
            )
        if component_count < config.minimum_train_components_per_family:
            failures.append(
                f"{family}:train_components={component_count}"
            )
        if query_count < config.minimum_eval_queries_per_family:
            failures.append(
                f"{family}:eval_queries={query_count}"
            )
    independent_reference_counts: dict[str, int] = {}
    independent_family_reference_counts: dict[str, int] = {}
    independent_family_component_counts: dict[str, int] = {}
    for query in queries:
        independent = [
            reference
            for reference in train
            if (
                reference.iid != query.iid
                and reference.pair_id != query.pair_id
                and reference.component_id != query.component_id
            )
        ]
        independent_family = [
            reference
            for reference in independent
            if reference.family == query.family
        ]
        independent_reference_counts[query.iid] = len(independent)
        independent_family_reference_counts[query.iid] = len(
            independent_family
        )
        independent_family_component_counts[query.iid] = len(
            {reference.component_id for reference in independent_family}
        )
        if len(independent) < 5:
            failures.append(
                f"{query.iid}:independent_train_refs={len(independent)}"
            )
        if (
            len(independent_family)
            < config.minimum_train_references_per_family
        ):
            failures.append(
                f"{query.iid}:independent_family_train_refs="
                f"{len(independent_family)}"
            )
        if (
            independent_family_component_counts[query.iid]
            < config.minimum_train_components_per_family
        ):
            failures.append(
                f"{query.iid}:independent_family_train_components="
                f"{independent_family_component_counts[query.iid]}"
            )
    if failures:
        raise RepresentationProbeError(
            "insufficient family support: " + ", ".join(failures)
        )
    return {
        "train_references": len(train),
        "train_components": len(
            {example.component_id for example in train}
        ),
        "train_family_counts": train_counts,
        "train_family_component_counts": {
            family: len(components)
            for family, components in sorted(train_components.items())
        },
        "evaluation_queries": len(queries),
        "evaluation_components": len(eval_components),
        "evaluation_family_counts": eval_counts,
        "minimum_independent_train_references_per_query": min(
            independent_reference_counts.values()
        ),
        "minimum_independent_family_references_per_query": min(
            independent_family_reference_counts.values()
        ),
        "minimum_independent_family_components_per_query": min(
            independent_family_component_counts.values()
        ),
    }


def _metric_summary(
    families: np.ndarray,
    hit1: np.ndarray,
    hit5: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    family_values = np.asarray(families, dtype=object)
    first = np.asarray(hit1, dtype=np.float64)
    fifth = np.asarray(hit5, dtype=np.float64)
    weight = (
        np.ones(len(family_values), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        family_values.ndim != 1
        or first.shape != family_values.shape
        or fifth.shape != family_values.shape
        or weight.shape != family_values.shape
        or not np.isfinite(weight).all()
        or bool((weight < 0).any())
        or float(np.sum(weight)) <= 0.0
    ):
        raise RepresentationProbeError("invalid retrieval metric arrays")
    total = float(np.sum(weight))
    per_family: dict[str, dict[str, Any]] = {}
    for family in sorted(set(str(value) for value in family_values.tolist())):
        mask = family_values == family
        family_weight = float(np.sum(weight[mask]))
        if family_weight <= 0.0:
            continue
        per_family[family] = {
            "queries": int(np.sum(mask)),
            "bootstrap_weight": family_weight,
            "r_at_1": float(np.sum(first[mask] * weight[mask]) / family_weight),
            "r_at_5": float(np.sum(fifth[mask] * weight[mask]) / family_weight),
        }
    return {
        "micro": {
            "queries": len(family_values),
            "bootstrap_weight": total,
            "r_at_1": float(np.sum(first * weight) / total),
            "r_at_5": float(np.sum(fifth * weight) / total),
        },
        "macro_family": {
            "families": len(per_family),
            "r_at_1": float(
                np.mean([value["r_at_1"] for value in per_family.values()])
            ),
            "r_at_5": float(
                np.mean([value["r_at_5"] for value in per_family.values()])
            ),
        },
        "per_family": per_family,
    }


def _cluster_bootstrap(
    families: np.ndarray,
    components: np.ndarray,
    hit1: np.ndarray,
    hit5: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    component_values = sorted(
        set(str(value) for value in np.asarray(components, dtype=object).tolist())
    )
    if not component_values:
        raise RepresentationProbeError("component bootstrap has no components")
    component_to_indices = {
        component: np.flatnonzero(components == component)
        for component in component_values
    }
    rng = np.random.default_rng(seed)
    series = {
        "micro_r_at_1": np.empty(repetitions, dtype=np.float64),
        "micro_r_at_5": np.empty(repetitions, dtype=np.float64),
        "macro_family_r_at_1": np.empty(repetitions, dtype=np.float64),
        "macro_family_r_at_5": np.empty(repetitions, dtype=np.float64),
    }
    unique_families = sorted(
        set(str(value) for value in np.asarray(families, dtype=object).tolist())
    )
    family_series = {
        family: {
            "r_at_1": np.full(repetitions, np.nan, dtype=np.float64),
            "r_at_5": np.full(repetitions, np.nan, dtype=np.float64),
        }
        for family in unique_families
    }
    for repetition in range(repetitions):
        sampled = rng.integers(
            0, len(component_values), size=len(component_values)
        )
        multiplicity = np.bincount(
            sampled, minlength=len(component_values)
        ).astype(np.float64)
        weights = np.zeros(len(families), dtype=np.float64)
        for index, component in enumerate(component_values):
            weights[component_to_indices[component]] = multiplicity[index]
        metric = _metric_summary(families, hit1, hit5, weights)
        series["micro_r_at_1"][repetition] = metric["micro"]["r_at_1"]
        series["micro_r_at_5"][repetition] = metric["micro"]["r_at_5"]
        series["macro_family_r_at_1"][repetition] = metric[
            "macro_family"
        ]["r_at_1"]
        series["macro_family_r_at_5"][repetition] = metric[
            "macro_family"
        ]["r_at_5"]
        for family, family_metric in metric["per_family"].items():
            family_series[family]["r_at_1"][repetition] = family_metric[
                "r_at_1"
            ]
            family_series[family]["r_at_5"][repetition] = family_metric[
                "r_at_5"
            ]
    return {
        "protocol": R7_BOOTSTRAP_PROTOCOL,
        "confidence": 0.95,
        "repetitions": repetitions,
        "seed": seed,
        "components": len(component_values),
        "intervals": {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in series.items()
        },
        "per_family_intervals": {
            family: {
                metric: {
                    "lower": float(
                        np.nanquantile(values, 0.025)
                    ),
                    "upper": float(
                        np.nanquantile(values, 0.975)
                    ),
                    "valid_repetitions": int(np.isfinite(values).sum()),
                }
                for metric, values in metrics.items()
            }
            for family, metrics in family_series.items()
        },
    }


def _rank_references(
    query: _Example,
    references: Sequence[_Example],
    *,
    modality: str,
) -> tuple[np.ndarray, dict[str, int]]:
    allowed: list[int] = []
    excluded = {"same_iid": 0, "same_pair": 0, "same_component": 0}
    for index, reference in enumerate(references):
        reasons: list[str] = []
        if reference.iid == query.iid:
            reasons.append("same_iid")
        if reference.pair_id == query.pair_id:
            reasons.append("same_pair")
        if reference.component_id == query.component_id:
            reasons.append("same_component")
        if reasons:
            for reason in reasons:
                excluded[reason] += 1
            continue
        allowed.append(index)
    if not allowed:
        raise RepresentationProbeError(
            f"query iid={query.iid} has no independent train references"
        )
    allowed_array = np.asarray(allowed, dtype=np.int64)
    bank = np.stack(
        [references[index].features[modality] for index in allowed],
        axis=0,
    ).astype(np.float64)
    query_vector = np.asarray(query.features[modality], dtype=np.float64)
    similarities = bank @ query_vector
    # The reference list is IID-sorted.  Stable sorting therefore makes ties
    # deterministic without allowing input order to influence retrieval.
    local_order = np.argsort(-similarities, kind="mergesort")
    return allowed_array[local_order], excluded


def _permutation_null(
    families: np.ndarray,
    top_indices: np.ndarray,
    bank_families: np.ndarray,
    observed: Mapping[str, Any],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = {
        "micro_r_at_1": np.empty(repetitions, dtype=np.float64),
        "micro_r_at_5": np.empty(repetitions, dtype=np.float64),
        "macro_family_r_at_1": np.empty(repetitions, dtype=np.float64),
        "macro_family_r_at_5": np.empty(repetitions, dtype=np.float64),
    }
    for repetition in range(repetitions):
        permuted = bank_families[rng.permutation(len(bank_families))]
        retrieved = permuted[top_indices]
        hit1 = retrieved[:, 0] == families
        hit5 = np.any(retrieved == families[:, None], axis=1)
        metric = _metric_summary(families, hit1, hit5)
        values["micro_r_at_1"][repetition] = metric["micro"]["r_at_1"]
        values["micro_r_at_5"][repetition] = metric["micro"]["r_at_5"]
        values["macro_family_r_at_1"][repetition] = metric[
            "macro_family"
        ]["r_at_1"]
        values["macro_family_r_at_5"][repetition] = metric[
            "macro_family"
        ]["r_at_5"]
    observed_values = {
        "micro_r_at_1": float(observed["micro"]["r_at_1"]),
        "micro_r_at_5": float(observed["micro"]["r_at_5"]),
        "macro_family_r_at_1": float(
            observed["macro_family"]["r_at_1"]
        ),
        "macro_family_r_at_5": float(
            observed["macro_family"]["r_at_5"]
        ),
    }
    return {
        "protocol": R7_NULL_PROTOCOL,
        "repetitions": repetitions,
        "seed": seed,
        "statistics": {
            name: {
                "observed": observed_values[name],
                "null_mean": float(np.mean(series)),
                "null_interval_95": {
                    "lower": float(np.quantile(series, 0.025)),
                    "upper": float(np.quantile(series, 0.975)),
                },
                "one_sided_p_value": float(
                    (
                        1
                        + np.sum(series >= observed_values[name])
                    )
                    / (repetitions + 1)
                ),
            }
            for name, series in values.items()
        },
    }


def evaluate_retrieval(
    train: Sequence[_Example],
    queries: Sequence[_Example],
    *,
    config: RepresentationProbeConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all three representations on a shared train/query cohort."""

    config.validate()
    sorted_train = sorted(train, key=lambda example: example.iid)
    sorted_queries = sorted(queries, key=lambda example: example.iid)
    families = np.asarray(
        [example.family for example in sorted_queries], dtype=object
    )
    components = np.asarray(
        [example.component_id for example in sorted_queries], dtype=object
    )
    bank_families = np.asarray(
        [example.family for example in sorted_train], dtype=object
    )
    per_query = [
        {
            "schema_version": R7_REPRESENTATION_QUERY_SCHEMA,
            "iid": query.iid,
            "pair_id": query.pair_id,
            "family": query.family,
            "label_provenance_kind": query.provenance_kind,
            "split": query.split,
            "component_id": query.component_id,
            "modalities": {},
        }
        for query in sorted_queries
    ]
    modality_metrics: dict[str, Any] = {}
    total_exclusions = {
        modality: {"same_iid": 0, "same_pair": 0, "same_component": 0}
        for modality in MODALITIES
    }
    top_width = min(5, len(sorted_train))
    if top_width < 1:
        raise RepresentationProbeError("retrieval bank is empty")
    for modality_index, modality in enumerate(MODALITIES):
        hit1 = np.zeros(len(sorted_queries), dtype=bool)
        hit5 = np.zeros(len(sorted_queries), dtype=bool)
        top_indices = np.zeros(
            (len(sorted_queries), top_width), dtype=np.int64
        )
        for query_index, query in enumerate(sorted_queries):
            ranking, excluded = _rank_references(
                query, sorted_train, modality=modality
            )
            if len(ranking) < top_width:
                raise RepresentationProbeError(
                    f"query iid={query.iid} has only {len(ranking)} "
                    f"independent references; need {top_width}"
                )
            top = ranking[:top_width]
            top_indices[query_index] = top
            retrieved = [sorted_train[index] for index in top]
            hit1[query_index] = retrieved[0].family == query.family
            hit5[query_index] = any(
                reference.family == query.family for reference in retrieved
            )
            for reason, count in excluded.items():
                total_exclusions[modality][reason] += count
            per_query[query_index]["modalities"][modality] = {
                "top_reference_iids": [
                    reference.iid for reference in retrieved
                ],
                "top_reference_pair_ids": [
                    reference.pair_id for reference in retrieved
                ],
                "top_reference_components": [
                    reference.component_id for reference in retrieved
                ],
                "top_reference_families": [
                    reference.family for reference in retrieved
                ],
                "correct_at_1": bool(hit1[query_index]),
                "correct_at_5": bool(hit5[query_index]),
            }
        observed = _metric_summary(families, hit1, hit5)
        observed["component_bootstrap_95"] = _cluster_bootstrap(
            families,
            components,
            hit1,
            hit5,
            repetitions=config.bootstrap_repetitions,
            seed=config.seed + 1009 * (modality_index + 1),
        )
        observed["label_permutation_null"] = _permutation_null(
            families,
            top_indices,
            bank_families,
            observed,
            repetitions=config.permutation_repetitions,
            seed=config.seed + 2003 * (modality_index + 1),
        )
        modality_metrics[modality] = observed
    return modality_metrics, per_query, {
        "protocol": R7_RETRIEVAL_PROTOCOL,
        "reference_feature_role": "target_only",
        "source_features_in_bank": False,
        "same_pair_source_target_reference_possible": False,
        "exclusions": total_exclusions,
    }


def _decision(
    examples: Sequence[_Example],
    queries: Sequence[_Example],
    split_metadata: Mapping[str, Any],
    config: RepresentationProbeConfig,
) -> dict[str, Any]:
    reasons: list[str] = []
    if config.eval_splits != ("test",):
        reasons.append(
            "formal evaluation requires the held-out test split alone"
        )
    if not bool(split_metadata["formal_fresh_split_attested"]):
        reasons.append("visual split lacks a passing fresh R7 audit/provenance")
    if not all(query.evaluation_fresh for query in queries):
        reasons.append("one or more evaluation assignments are not fresh")
    if not all(example.formal_human for example in examples):
        reasons.append(
            "retained train/evaluation labels are pseudo or lack explicit "
            "human annotation_id+annotator_id provenance"
        )
    pair_splits: dict[str, set[str]] = {}
    for example in examples:
        pair_splits.setdefault(example.pair_id, set()).add(example.split)
    cross_split_pairs = sorted(
        pair_id
        for pair_id, splits in pair_splits.items()
        if "train" in splits and bool(set(splits) & {"validation", "test"})
    )
    if cross_split_pairs:
        reasons.append("logical pair IDs cross train/evaluation splits")
    formal = not reasons
    return {
        "formal_status": "FORMAL_EVALUABLE" if formal else "INSUFFICIENT",
        "formal_evaluable": formal,
        "formal_reasons": (
            [
                "human-provenanced labels and fresh audited visual split; "
                "metrics still require external review and a prespecified gate"
            ]
            if formal
            else reasons
        ),
        "formal_probe_passed": None,
        "production_decision": False,
        "generation_authorized": False,
        "cross_split_pair_ids": cross_split_pairs,
    }


def _build_contract(
    *,
    preflight_provenance: Sequence[Mapping[str, Any]],
    split_metadata: Mapping[str, Any],
    label_metadata: Mapping[str, Any],
    config: RepresentationProbeConfig,
) -> dict[str, Any]:
    return {
        "schema_version": R7_REPRESENTATION_PROBE_SCHEMA,
        "preflight_finals": [dict(value) for value in preflight_provenance],
        "visual_split": dict(split_metadata),
        "labels": dict(label_metadata),
        "config": asdict(config),
        "protocol": {
            "retrieval": R7_RETRIEVAL_PROTOCOL,
            "bootstrap": R7_BOOTSTRAP_PROTOCOL,
            "null": R7_NULL_PROTOCOL,
            "cohort": (
                "explicit-label+split-assignment+target-base-valid+"
                "target-dino-valid-v1"
            ),
            "primary": "target_actor_teacher",
            "baselines": [
                "target_pooled_dino",
                "target_camera_trajectory",
            ],
        },
    }


def _output_paths(directory: Path) -> dict[str, Path]:
    return {
        "summary": directory / SUMMARY_NAME,
        "per_query": directory / PER_QUERY_NAME,
        "done": directory / DONE_NAME,
    }


def validate_probe_output(
    output_dir: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate a committed probe, including leakage invariants."""

    directory = output_dir.expanduser().resolve(strict=True)
    paths = _output_paths(directory)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != R7_REPRESENTATION_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("production_decision") is not False
        or done.get("generation_authorized") is not False
    ):
        raise RepresentationProbeError("invalid representation probe done marker")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "summary",
        "per_query",
    }:
        raise RepresentationProbeError(
            "representation probe artifact registry differs"
        )
    for name, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != paths[name].name
            or record.get("sha256") != _file_digest(paths[name])
        ):
            raise RepresentationProbeError(
                f"representation probe {name} digest mismatch"
            )
    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != R7_REPRESENTATION_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("formal_probe_passed") is not None
        or summary.get("production_decision") is not False
        or summary.get("generation_authorized") is not False
    ):
        raise RepresentationProbeError("invalid representation probe summary")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise RepresentationProbeError("probe summary contract is missing")
    digest = _object_digest(dict(contract))
    if (
        summary.get("contract_sha256") != digest
        or done.get("contract_sha256") != digest
    ):
        raise RepresentationProbeError("probe contract digest mismatch")
    if (
        expected_contract is not None
        and _object_digest(dict(expected_contract)) != digest
    ):
        raise RepresentationProbeError(
            "committed probe contract differs from requested resume contract"
        )
    rows = _load_jsonl(paths["per_query"])
    if (
        len(rows) != summary.get("support", {}).get("evaluation_queries")
        or len(rows) != done.get("queries")
    ):
        raise RepresentationProbeError("probe query count differs")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(MODALITIES):
        raise RepresentationProbeError("probe metric registry differs")
    replay: dict[str, dict[str, list[bool]]] = {
        modality: {"hit1": [], "hit5": []} for modality in MODALITIES
    }
    replay_families: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != R7_REPRESENTATION_QUERY_SCHEMA:
            raise RepresentationProbeError("per-query schema differs")
        iid = _nonempty_string(row.get("iid"), name="per-query iid")
        if iid in seen:
            raise RepresentationProbeError(f"per-query duplicates iid={iid}")
        seen.add(iid)
        pair_id = _nonempty_string(
            row.get("pair_id"), name=f"per-query {iid} pair_id"
        )
        component = _nonempty_string(
            row.get("component_id"),
            name=f"per-query {iid} component_id",
        )
        family = _nonempty_string(
            row.get("family"), name=f"per-query {iid} family"
        )
        replay_families.append(family)
        modalities = row.get("modalities")
        if not isinstance(modalities, Mapping) or set(modalities) != set(
            MODALITIES
        ):
            raise RepresentationProbeError(
                f"per-query {iid} modality registry differs"
            )
        for modality, result in modalities.items():
            if not isinstance(result, Mapping):
                raise RepresentationProbeError(
                    f"per-query {iid}/{modality} result is malformed"
                )
            ref_iids = result.get("top_reference_iids")
            ref_pairs = result.get("top_reference_pair_ids")
            ref_components = result.get("top_reference_components")
            ref_families = result.get("top_reference_families")
            if (
                not isinstance(ref_iids, list)
                or not isinstance(ref_pairs, list)
                or not isinstance(ref_components, list)
                or not isinstance(ref_families, list)
                or not (
                    len(ref_iids)
                    == len(ref_pairs)
                    == len(ref_components)
                    == len(ref_families)
                    == 5
                )
                or iid in ref_iids
                or pair_id in ref_pairs
                or component in ref_components
            ):
                raise RepresentationProbeError(
                    f"per-query {iid}/{modality} leaks IID/pair/component"
                )
            correct1 = ref_families[0] == family
            correct5 = family in ref_families
            if (
                type(result.get("correct_at_1")) is not bool
                or type(result.get("correct_at_5")) is not bool
                or result.get("correct_at_1") != correct1
                or result.get("correct_at_5") != correct5
            ):
                raise RepresentationProbeError(
                    f"per-query {iid}/{modality} correctness differs"
                )
            replay[modality]["hit1"].append(correct1)
            replay[modality]["hit5"].append(correct5)
    replay_family_array = np.asarray(replay_families, dtype=object)
    for modality in MODALITIES:
        rebuilt = _metric_summary(
            replay_family_array,
            np.asarray(replay[modality]["hit1"], dtype=bool),
            np.asarray(replay[modality]["hit5"], dtype=bool),
        )
        committed = metrics[modality]
        if (
            not isinstance(committed, Mapping)
            or committed.get("micro") != rebuilt["micro"]
            or committed.get("macro_family") != rebuilt["macro_family"]
            or committed.get("per_family") != rebuilt["per_family"]
            or not isinstance(
                committed.get("component_bootstrap_95"), Mapping
            )
            or committed["component_bootstrap_95"].get("protocol")
            != R7_BOOTSTRAP_PROTOCOL
            or not isinstance(
                committed.get("label_permutation_null"), Mapping
            )
            or committed["label_permutation_null"].get("protocol")
            != R7_NULL_PROTOCOL
        ):
            raise RepresentationProbeError(
                f"probe {modality} metrics do not replay from queries"
            )
    decision = summary.get("decision")
    if not isinstance(decision, Mapping):
        raise RepresentationProbeError("probe decision is missing")
    if (
        decision.get("formal_status") == "FORMAL_EVALUABLE"
        and decision.get("formal_evaluable") is not True
    ):
        raise RepresentationProbeError("formal decision fields disagree")
    if (
        summary.get("formal_status") != decision.get("formal_status")
        or done.get("formal_status") != summary.get("formal_status")
        or decision.get("formal_probe_passed") is not None
        or decision.get("production_decision") is not False
        or decision.get("generation_authorized") is not False
    ):
        raise RepresentationProbeError("probe decision scope fields disagree")
    return {"done": done, "summary": summary, "per_query": rows}


def run_representation_probe(
    *,
    preflight_final_directories: Sequence[Path],
    visual_split: Path,
    labels: Path,
    output_dir: Path,
    config: RepresentationProbeConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run, atomically commit, or strictly resume the R7-A probe."""

    cfg = config or RepresentationProbeConfig()
    cfg.validate()
    assignments, split_metadata = _parse_visual_split(visual_split)
    parsed_labels, label_metadata = _parse_labels(labels)
    examples, coverage, preflight_provenance = _load_preflight_examples(
        preflight_final_directories,
        assignments,
        parsed_labels,
    )
    contract = _build_contract(
        preflight_provenance=preflight_provenance,
        split_metadata=split_metadata,
        label_metadata=label_metadata,
        config=cfg,
    )
    directory = output_dir.expanduser()
    paths = _output_paths(directory)
    if paths["done"].exists():
        if not resume:
            raise FileExistsError(paths["done"])
        return validate_probe_output(
            directory,
            expected_contract=contract,
        )["done"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"partial representation probe cannot be resumed: {directory}"
        )
    train = [example for example in examples if example.split == "train"]
    queries = [
        example
        for example in examples
        if example.split in set(cfg.eval_splits)
    ]
    support = _validate_support(train, queries, config=cfg)
    metrics, per_query, leakage = evaluate_retrieval(
        train,
        queries,
        config=cfg,
    )
    decision = _decision(
        [*train, *queries],
        queries,
        split_metadata,
        cfg,
    )
    zero_camera = sum(
        float(np.linalg.norm(example.features["target_camera_trajectory"]))
        <= 1e-12
        for example in examples
    )
    coverage.update(
        {
            "split_counts_common_cohort": _counts(
                example.split for example in examples
            ),
            "label_provenance_counts_common_cohort": _counts(
                example.provenance_kind for example in examples
            ),
            "family_counts_common_cohort": _counts(
                example.family for example in examples
            ),
            "zero_camera_trajectory_rows": int(zero_camera),
            "comparison_modalities_share_exact_cohort": True,
        }
    )
    summary = {
        "schema_version": R7_REPRESENTATION_SUMMARY_SCHEMA,
        "status": "complete",
        "contract": contract,
        "contract_sha256": _object_digest(contract),
        "coverage": coverage,
        "support": support,
        "metrics": metrics,
        "leakage_control": leakage,
        "decision": decision,
        "formal_status": decision["formal_status"],
        "formal_probe_passed": None,
        "production_decision": False,
        "generation_authorized": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(paths["per_query"], per_query)
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R7_REPRESENTATION_DONE_SCHEMA,
        "status": "complete",
        "queries": len(per_query),
        "formal_status": decision["formal_status"],
        "formal_probe_passed": None,
        "production_decision": False,
        "generation_authorized": False,
        "contract_sha256": summary["contract_sha256"],
        "artifacts": {
            "summary": {
                "filename": SUMMARY_NAME,
                "sha256": _file_digest(paths["summary"]),
            },
            "per_query": {
                "filename": PER_QUERY_NAME,
                "sha256": _file_digest(paths["per_query"]),
            },
        },
    }
    _atomic_json(paths["done"], done)
    validate_probe_output(directory, expected_contract=contract)
    return done


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R7-A target observed-event nearest-neighbour representation probe"
        )
    )
    parser.add_argument(
        "--preflight-final",
        type=Path,
        action="append",
        required=True,
        dest="preflight_finals",
    )
    parser.add_argument("--visual-split", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-split",
        action="append",
        choices=("validation", "test"),
        dest="eval_splits",
    )
    parser.add_argument("--minimum-total-train-references", type=int, default=20)
    parser.add_argument(
        "--minimum-train-references-per-family", type=int, default=5
    )
    parser.add_argument(
        "--minimum-train-components-per-family", type=int, default=3
    )
    parser.add_argument(
        "--minimum-eval-queries-per-family", type=int, default=2
    )
    parser.add_argument("--minimum-eval-components", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--permutation-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = RepresentationProbeConfig(
        eval_splits=tuple(args.eval_splits or ("test",)),
        minimum_total_train_references=args.minimum_total_train_references,
        minimum_train_references_per_family=(
            args.minimum_train_references_per_family
        ),
        minimum_train_components_per_family=(
            args.minimum_train_components_per_family
        ),
        minimum_eval_queries_per_family=args.minimum_eval_queries_per_family,
        minimum_eval_components=args.minimum_eval_components,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        seed=args.seed,
    )
    result = run_representation_probe(
        preflight_final_directories=args.preflight_finals,
        visual_split=args.visual_split,
        labels=args.labels,
        output_dir=args.output_dir,
        config=config,
        resume=args.resume,
    )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
