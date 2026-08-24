"""Train and evaluate the R6 semantic/independent-reference diagnostic.

The R5 endpoint archive is used only as a source/teacher/evaluation bundle.
Predictor inputs are limited to query source features, a frozen
instruction-only semantic embedding, and (when available) motion from a
different-IID/content/subject train-positive reference selected without the
query target.  Failed outcomes update a parameter-disjoint compatibility head
only; they never become no-op labels or magnitude suppressors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r5_gate import cross_content_retrieval
from .r6_gate import (
    R6_QUERY_SCHEMA,
    R6PilotThresholds,
    evaluate_r6_gate,
    invalid_r6_gate_summary,
)
from .r6_semantic_features import validate_artifact as validate_semantic_artifact
from .source_aware_repr import FactorizedR5Targets
from .source_aware_repr_r6 import (
    R6_OBSERVED_ACTION_SEMANTIC_SCHEMA,
    IndependentReferencePairs,
    R6EndpointBatch,
    R6FeatureTransform,
    R6MotionFeatures,
    R6ObservedActionSemanticBank,
    R6SemanticProvenance,
    SourceAwareFactorizedR6,
    build_semantic_train_bank_reference_pairs,
    pair_compatibility_loss,
    positive_factorized_r6_loss,
)
from .train_source_aware_repr import derive_labels, load_feature_bundle


R6_TRAIN_SCHEMA = "motive-r6-training-v1"
R6_SEMANTIC_ENCODERS = frozenset({"clip", "umt5"})
R6_ARMS = (
    "semantic_only",
    "independent_ref",
    "wrong_ref",
    "matched_random",
    "centroid",
    "source_shuffle",
    "semantic_shuffle",
    "exact_target_oracle",
)
R6_GATE_ARMS = frozenset(R6_ARMS) - {"exact_target_oracle"}
_HEX_DIGITS = frozenset("0123456789abcdef")


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


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _validated_sha256(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_ready(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
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
                handle.write(_canonical_json(_json_ready(dict(row))) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _string_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"semantic archive is missing {name}")
    values = np.asarray(arrays[name])
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"semantic archive {name} must be a string vector")
    return values.astype(str)


def _semantic_matrix(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"semantic archive is missing {name}")
    values = np.asarray(arrays[name], dtype=np.float32)
    if values.ndim != 2 or len(values) != rows or values.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if bool((np.abs(norms - 1.0) > 5e-3).any()):
        raise ValueError(f"{name} is not L2-normalized per row")
    return values


def _encoder_field(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata and metadata[name] not in {None, ""}:
            return metadata[name]
    raise ValueError(f"semantic encoder metadata lacks one of {names}")


def load_semantic_bundle(
    *,
    semantic_archive: Path,
    encoder: str,
    expected_iids: Sequence[str],
    expected_input_manifest_sha256: str,
    production: bool,
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    R6SemanticProvenance,
    R6ObservedActionSemanticBank,
    dict[str, Any],
]:
    """Load one encoder independently and bind it to the R5 manifest."""

    if encoder not in R6_SEMANTIC_ENCODERS:
        raise ValueError(f"unsupported semantic encoder {encoder!r}")
    archive_path = semantic_archive.expanduser().resolve(strict=True)
    if archive_path.name != "semantic_features.npz":
        raise ValueError(
            "R6 trainer requires the canonical semantic_features.npz artifact"
        )
    validated = validate_semantic_artifact(archive_path.parent)
    if validated.get("synthetic_test_artifact") is not False:
        raise ValueError(
            "synthetic semantic artifact cannot enter R6 training"
        )
    digest_before = _file_digest(archive_path)
    arrays = _load_npz(archive_path)
    if _file_digest(archive_path) != digest_before:
        raise RuntimeError("semantic archive changed while it was read")
    iids = _string_array(arrays, "iids")
    expected = tuple(str(value) for value in expected_iids)
    if tuple(iids.tolist()) != expected:
        raise ValueError(
            "semantic archive IID order differs from the R5 endpoint bundle"
        )
    rows = len(iids)
    prompt = _semantic_matrix(
        arrays,
        f"{encoder}_prompt",
        rows=rows,
    )
    observed = _semantic_matrix(
        arrays,
        f"{encoder}_observed_target",
        rows=rows,
    )
    prompt_digests = _string_array(arrays, "prompt_text_sha256")
    observed_digests = _string_array(
        arrays,
        "observed_target_text_sha256",
    )
    if len(prompt_digests) != rows or len(observed_digests) != rows:
        raise ValueError("semantic input-digest vector length mismatch")
    for name, values in (
        ("prompt_text_sha256", prompt_digests),
        ("observed_target_text_sha256", observed_digests),
    ):
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values.tolist()
        ):
            raise ValueError(f"{name} contains a non-SHA256 value")
    if "metadata_json" not in arrays or arrays["metadata_json"].ndim != 0:
        raise ValueError("semantic archive metadata_json must be scalar")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("semantic metadata_json is not an object")
    if metadata != validated["metadata"]:
        raise ValueError(
            "validated semantic sidecar differs from archive metadata_json"
        )
    input_manifest = metadata.get("input_manifest")
    if not isinstance(input_manifest, Mapping):
        raise ValueError("semantic metadata lacks input_manifest")
    if str(input_manifest.get("sha256")) != expected_input_manifest_sha256:
        raise ValueError(
            "semantic artifact is not bound to the supplied R5 manifest"
        )
    encoders = metadata.get("encoders")
    if not isinstance(encoders, Mapping) or not isinstance(
        encoders.get(encoder), Mapping
    ):
        raise ValueError(f"semantic metadata lacks encoders.{encoder}")
    encoder_metadata = encoders[encoder]
    dimension = int(
        _encoder_field(encoder_metadata, "dim", "embedding_dim")
    )
    if prompt.shape[1] != dimension or observed.shape[1] != dimension:
        raise ValueError("semantic encoder dimension metadata mismatch")
    encoder_id = str(
        _encoder_field(
            encoder_metadata,
            "id",
            "encoder_id",
            "resolved_path",
        )
    )
    revision = str(
        _encoder_field(
            encoder_metadata,
            "revision",
            "encoder_revision",
        )
    )
    weights_sha256 = str(
        _encoder_field(encoder_metadata, "weights_sha256")
    ).lower()
    tokenizer_sha256 = str(
        _encoder_field(encoder_metadata, "tokenizer_sha256")
    ).lower()
    pooling = str(_encoder_field(encoder_metadata, "pooling"))
    dtype = str(
        _encoder_field(encoder_metadata, "output_dtype", "dtype")
    )
    prompt_template_version = str(
        _encoder_field(encoder_metadata, "prompt_template_version")
    )
    if prompt_template_version != "raw-manifest-text-no-template-v1":
        raise ValueError(
            "semantic prompt_template_version is not the pre-registered "
            "raw-manifest-text-no-template-v1 contract"
        )
    query_provenance = R6SemanticProvenance(
        encoder_id=encoder_id,
        encoder_revision=revision,
        weights_sha256=weights_sha256,
        tokenizer_sha256=tokenizer_sha256,
        prompt_template_version=prompt_template_version,
        pooling=pooling,
        embedding_dim=dimension,
        dtype=dtype,
        normalization="l2_per_row",
        source_field="instruction",
        frozen_encoder=True,
        target_derived_input=False,
        label_derived_input=False,
    )
    query_provenance.validate(production=production, usage="query_prompt")
    observed_provenance = R6SemanticProvenance(
        encoder_id=encoder_id,
        encoder_revision=revision,
        weights_sha256=weights_sha256,
        tokenizer_sha256=tokenizer_sha256,
        prompt_template_version=prompt_template_version,
        pooling=pooling,
        embedding_dim=dimension,
        dtype=dtype,
        normalization="l2_per_row",
        source_field="observed_target_action",
        frozen_encoder=True,
        target_derived_input=True,
        label_derived_input=False,
        schema_version=R6_OBSERVED_ACTION_SEMANTIC_SCHEMA,
    )
    bank = R6ObservedActionSemanticBank.create(
        iids=expected,
        embeddings=observed,
        input_digests=observed_digests.tolist(),
        provenance=observed_provenance,
        production=production,
    )
    provenance = {
        "archive": str(archive_path),
        "archive_sha256": digest_before,
        "metadata": metadata,
        "encoder": encoder,
        "query_provenance": asdict(query_provenance),
        "query_provenance_digest": query_provenance.digest(),
        "observed_bank_digest": bank.digest(),
        "observed_bank_provenance_digest": observed_provenance.digest(),
        "prompt_input_digest_list_sha256": _object_digest(
            prompt_digests.tolist()
        ),
        "observed_input_digest_list_sha256": _object_digest(
            observed_digests.tolist()
        ),
    }
    return (
        prompt,
        tuple(prompt_digests.tolist()),
        query_provenance,
        bank,
        provenance,
    )


def _subject_clusters(
    rows: Sequence[Mapping[str, Any]],
    content_groups: Sequence[str],
) -> tuple[tuple[str, ...], str]:
    explicit = [
        str(
            row.get("subject_cluster_id")
            or row.get("source_subject_cluster_id")
            or ""
        ).strip()
        for row in rows
    ]
    if all(explicit):
        return tuple(explicit), "manifest-explicit"
    # Conservative diagnostic fallback: it never relaxes the already-enforced
    # content exclusion, but cannot claim true cross-subject generalization.
    return tuple(str(value) for value in content_groups), (
        "content_group_conservative_surrogate"
    )


def make_r6_batch(
    *,
    r5_batch: Any,
    manifest_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    semantic_embeddings: np.ndarray,
    semantic_input_digests: Sequence[str],
    semantic_provenance: R6SemanticProvenance,
    production: bool,
) -> tuple[R6EndpointBatch, str]:
    roles = tuple(
        "positive_delta"
        if label["label_role"] == "positive_delta"
        else "failed_outcome_compatibility"
        for label in labels
    )
    if any(
        label["label_role"] not in {"positive_delta", "negative_audit"}
        for label in labels
    ):
        raise ValueError("R6 input contains unlabeled/excluded rows")
    compatibility = np.asarray(
        [1.0 if role == "positive_delta" else 0.0 for role in roles],
        dtype=np.float32,
    )
    subjects, subject_source = _subject_clusters(
        manifest_rows,
        r5_batch.content_group_ids,
    )
    batch = R6EndpointBatch.create(
        iids=r5_batch.iids,
        source_actor=r5_batch.source_actor,
        source_camera=r5_batch.source_camera,
        target_actor=r5_batch.target_actor,
        target_camera=r5_batch.target_camera,
        semantic_embeddings=semantic_embeddings,
        semantic_input_digests=semantic_input_digests,
        splits=r5_batch.splits,
        content_group_ids=r5_batch.content_group_ids,
        subject_cluster_ids=subjects,
        action_families=tuple(
            str(label["action_family"]) for label in labels
        ),
        label_roles=roles,
        compatibility_targets=compatibility,
        split_versions=r5_batch.split_versions,
        semantic_provenance=semantic_provenance,
        perceptual_hashes=r5_batch.perceptual_hashes,
        require_visual_clusters=production,
        maximum_cross_split_hamming_fraction=(
            r5_batch.maximum_cross_split_hamming_fraction or 0.10
        ),
    )
    return batch, subject_source


def _pair_ledger_rows(
    pairs: IndependentReferencePairs,
    batch: R6EndpointBatch,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, (
        query_raw,
        reference_raw,
        rank_raw,
        score_raw,
        candidate_count_raw,
        input_digest,
    ) in enumerate(
        zip(
            pairs.query_indices,
            pairs.reference_indices,
            pairs.reference_ranks,
            pairs.pair_scores,
            pairs.candidate_counts,
            pairs.pair_input_digests,
        )
    ):
        query = int(query_raw)
        reference = int(reference_raw)
        rows.append(
            {
                "schema_version": "motive-r6-reference-pair-ledger-v1",
                "pair_index": position,
                "query_index": query,
                "query_iid": batch.iids[query],
                "query_split": batch.splits[query],
                "query_content_group_id": batch.content_group_ids[query],
                "query_subject_cluster_id": batch.subject_cluster_ids[query],
                "reference_index": reference,
                "reference_iid": batch.iids[reference],
                "reference_split": batch.splits[reference],
                "reference_content_group_id": batch.content_group_ids[reference],
                "reference_subject_cluster_id": batch.subject_cluster_ids[reference],
                "reference_rank": int(rank_raw),
                "semantic_cosine": float(score_raw),
                "eligible_candidate_count": int(candidate_count_raw),
                "pair_input_digest": str(input_digest),
                "selector_name": pairs.selector_name,
                "gate_eligible": bool(pairs.gate_eligible),
            }
        )
    return rows


def _reference_audit(
    pairs: IndependentReferencePairs,
    batch: R6EndpointBatch,
) -> dict[str, Any]:
    scores = np.asarray(pairs.pair_scores, dtype=np.float64)

    def scoped_load(*, split: str, rank: int | None) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for query_raw, reference_raw, rank_raw in zip(
            pairs.query_indices,
            pairs.reference_indices,
            pairs.reference_ranks,
        ):
            query = int(query_raw)
            if (
                batch.splits[query] != split
                or batch.label_roles[query] != "positive_delta"
                or (rank is not None and int(rank_raw) != rank)
            ):
                continue
            iid = batch.iids[int(reference_raw)]
            counts[iid] = counts.get(iid, 0) + 1
        total = sum(counts.values())
        maximum = max(counts.values(), default=0)
        return {
            "scope": (
                f"{split}-positive-"
                + ("all-ranks" if rank is None else f"rank-{rank}")
            ),
            "pair_count": total,
            "unique_reference_count": len(counts),
            "maximum_reference_load": maximum,
            "maximum_reference_load_fraction": (
                float(maximum) / float(total) if total else 0.0
            ),
            "per_reference_iid": dict(sorted(counts.items())),
        }

    return {
        "schema_version": "motive-r6-reference-audit-v1",
        "selector_name": pairs.selector_name,
        "pairing_version": pairs.pairing_version,
        "pair_digest": pairs.digest(),
        "gate_eligible": bool(pairs.gate_eligible),
        "similarity_threshold": pairs.similarity_threshold,
        "threshold_quantile": pairs.threshold_quantile,
        "threshold_fit_iid_digest": pairs.threshold_fit_iid_digest,
        "threshold_fit_digest": pairs.threshold_fit_digest,
        "coverage": {
            split: pairs.coverage(batch, split=split, positive_only=True)
            for split in ("train", "validation", "test")
        },
        "reference_load_global_descriptive": pairs.reference_load(batch),
        "reference_load_test_positive_rank0": scoped_load(
            split="test",
            rank=0,
        ),
        "reference_load_test_positive_all_ranks": scoped_load(
            split="test",
            rank=None,
        ),
        "pair_score": {
            "count": int(len(scores)),
            "minimum": float(np.min(scores)) if len(scores) else None,
            "q10": float(np.quantile(scores, 0.10)) if len(scores) else None,
            "median": float(np.median(scores)) if len(scores) else None,
            "q90": float(np.quantile(scores, 0.90)) if len(scores) else None,
            "maximum": float(np.max(scores)) if len(scores) else None,
        },
        "unpaired_count": len(pairs.unpaired_iids),
        "undercovered_count": len(pairs.undercovered_iids),
        "unpaired_iids": list(pairs.unpaired_iids),
        "undercovered_iids": list(pairs.undercovered_iids),
        "no_fallback_selector": True,
        "oracle_family_fallback_used": False,
    }


def _primary_reference_arrays(
    *,
    pairs: IndependentReferencePairs,
    batch: R6EndpointBatch,
    observed_motion: R6MotionFeatures,
    reference_rank: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(batch.iids)
    actor = np.zeros(
        (rows, observed_motion.actor_input.shape[1]),
        dtype=np.float32,
    )
    camera = np.zeros(
        (rows, observed_motion.camera_input.shape[1]),
        dtype=np.float32,
    )
    mask = np.zeros(rows, dtype=np.float32)
    reference_index = np.full(rows, -1, dtype=np.int64)
    score = np.full(rows, np.nan, dtype=np.float64)
    for query_raw, reference_raw, rank_raw, score_raw in zip(
        pairs.query_indices,
        pairs.reference_indices,
        pairs.reference_ranks,
        pairs.pair_scores,
    ):
        query = int(query_raw)
        reference = int(reference_raw)
        if int(rank_raw) != int(reference_rank):
            continue
        actor[query] = observed_motion.actor_input[reference]
        camera[query] = observed_motion.camera_input[reference]
        mask[query] = 1.0
        reference_index[query] = reference
        score[query] = float(score_raw)
    return actor, camera, mask, reference_index, score


def _wrong_reference_arrays(
    *,
    batch: R6EndpointBatch,
    observed_action_bank: R6ObservedActionSemanticBank,
    observed_motion: R6MotionFeatures,
    primary_reference_index: np.ndarray,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the least-aligned legal train reference as a specificity control."""

    rows = len(batch.iids)
    actor = np.zeros(
        (rows, observed_motion.actor_input.shape[1]),
        dtype=np.float32,
    )
    camera = np.zeros(
        (rows, observed_motion.camera_input.shape[1]),
        dtype=np.float32,
    )
    mask = np.zeros(rows, dtype=np.float32)
    chosen = np.full(rows, -1, dtype=np.int64)
    scores = np.full(rows, np.nan, dtype=np.float64)
    bank_by_iid = observed_action_bank.index_by_iid()
    train = batch.positive_indices("train")
    for query in range(rows):
        candidates: list[tuple[float, bytes, int]] = []
        for raw_reference in train:
            reference = int(raw_reference)
            if (
                reference == query
                or reference == int(primary_reference_index[query])
                or batch.iids[reference] == batch.iids[query]
                or batch.content_group_ids[reference]
                == batch.content_group_ids[query]
                or batch.subject_cluster_ids[reference]
                == batch.subject_cluster_ids[query]
            ):
                continue
            bank_index = bank_by_iid[batch.iids[reference]]
            similarity = float(
                np.dot(
                    batch.semantic_embeddings[query].astype(np.float64),
                    observed_action_bank.embeddings[bank_index].astype(np.float64),
                )
            )
            tie = hashlib.sha256(
                (
                    f"{int(data_seed)}\0wrong\0{batch.iids[query]}\0"
                    f"{batch.iids[reference]}"
                ).encode("utf-8")
            ).digest()
            candidates.append((similarity, tie, reference))
        if not candidates:
            continue
        similarity, _, reference = min(
            candidates,
            key=lambda item: (item[0], item[1], batch.iids[item[2]]),
        )
        actor[query] = observed_motion.actor_input[reference]
        camera[query] = observed_motion.camera_input[reference]
        mask[query] = 1.0
        chosen[query] = reference
        scores[query] = similarity
    return actor, camera, mask, chosen, scores


def _semantic_shuffle_indices(
    batch: R6EndpointBatch,
    *,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(batch.iids), dtype=np.int64)
    valid = np.zeros(len(indices), dtype=bool)
    for split in ("train", "validation", "test"):
        selected = np.flatnonzero(np.asarray(batch.splits) == split)
        if len(selected) < 2:
            continue
        ordered = sorted(
            (int(index) for index in selected),
            key=lambda index: (
                hashlib.sha256(
                    (
                        f"{int(data_seed)}\0semantic-shuffle\0"
                        f"{batch.iids[index]}"
                    ).encode("utf-8")
                ).digest(),
                batch.iids[index],
            ),
        )
        rotated = ordered[1:] + ordered[:1]
        for query, source in zip(ordered, rotated):
            indices[query] = source
            valid[query] = query != source
    return indices, valid


def _positive_mismatch_indices(
    batch: R6EndpointBatch,
    *,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic within-split other-positive motion diagnostic."""

    indices = np.arange(len(batch.iids), dtype=np.int64)
    valid = np.zeros(len(indices), dtype=bool)
    roles = np.asarray(batch.label_roles)
    splits = np.asarray(batch.splits)
    for split in ("train", "validation", "test"):
        selected = np.flatnonzero(
            (splits == split) & (roles == "positive_delta")
        )
        if len(selected) < 2:
            continue
        ordered = sorted(
            (int(index) for index in selected),
            key=lambda index: (
                hashlib.sha256(
                    (
                        f"{int(data_seed)}\0compat-mismatch\0"
                        f"{batch.iids[index]}"
                    ).encode("utf-8")
                ).digest(),
                batch.iids[index],
            ),
        )
        rotated = ordered[1:] + ordered[:1]
        for query, motion_source in zip(ordered, rotated):
            indices[query] = motion_source
            valid[query] = query != motion_source
    return indices, valid


def _reindex_motion(
    motion: R6MotionFeatures,
    indices: np.ndarray,
) -> R6MotionFeatures:
    selected = np.asarray(indices, dtype=np.int64)
    return R6MotionFeatures(
        actor_direction=motion.actor_direction[selected],
        actor_log_magnitude=motion.actor_log_magnitude[selected],
        camera_direction=motion.camera_direction[selected],
        camera_log_magnitude=motion.camera_log_magnitude[selected],
    )


def _global_centroid(
    targets: FactorizedR5Targets,
    positive_train: np.ndarray,
    *,
    rows: int,
) -> FactorizedR5Targets:
    train = np.asarray(positive_train, dtype=np.int64)

    def direction(values: np.ndarray) -> np.ndarray:
        center = np.mean(values[train].astype(np.float64), axis=0)
        norm = float(np.linalg.norm(center))
        if norm <= 1e-12:
            center = np.zeros(values.shape[1], dtype=np.float64)
        else:
            center /= norm
        return np.repeat(center[None, :], rows, axis=0).astype(np.float32)

    def magnitude(values: np.ndarray) -> np.ndarray:
        center = float(np.mean(values[train]))
        return np.full(rows, center, dtype=np.float32)

    return FactorizedR5Targets(
        actor_direction=direction(targets.actor_direction),
        actor_log_magnitude=magnitude(targets.actor_log_magnitude),
        camera_direction=direction(targets.camera_direction),
        camera_log_magnitude=magnitude(targets.camera_log_magnitude),
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for R6 training") from error
    return torch


def _prediction_to_targets(prediction: Mapping[str, Any]) -> FactorizedR5Targets:
    return FactorizedR5Targets(
        actor_direction=prediction["actor_direction"].detach().float().cpu().numpy(),
        actor_log_magnitude=prediction["actor_log_magnitude"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1),
        camera_direction=prediction["camera_direction"]
        .detach()
        .float()
        .cpu()
        .numpy(),
        camera_log_magnitude=prediction["camera_log_magnitude"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1),
    )


def _predict(
    model: SourceAwareFactorizedR6,
    *,
    source_actor: np.ndarray,
    source_camera: np.ndarray,
    semantics: np.ndarray,
    reference_actor: np.ndarray | None,
    reference_camera: np.ndarray | None,
    reference_mask: np.ndarray | None,
    device: Any,
    batch_size: int,
) -> FactorizedR5Targets:
    torch = _torch()
    parts: list[FactorizedR5Targets] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(source_actor), int(batch_size)):
            stop = min(start + int(batch_size), len(source_actor))

            def tensor(values: np.ndarray) -> Any:
                return torch.as_tensor(
                    values[start:stop],
                    dtype=torch.float32,
                    device=device,
                )

            kwargs: dict[str, Any] = {
                "source_actor": tensor(source_actor),
                "source_camera": tensor(source_camera),
                "semantic_features": tensor(semantics),
            }
            if reference_actor is not None:
                assert reference_camera is not None
                assert reference_mask is not None
                kwargs.update(
                    {
                        "reference_actor_motion": tensor(reference_actor),
                        "reference_camera_motion": tensor(reference_camera),
                        "reference_mask": tensor(reference_mask),
                    }
                )
            parts.append(_prediction_to_targets(model(**kwargs)))
    if not parts:
        raise ValueError("cannot predict an empty R6 batch")
    return FactorizedR5Targets(
        actor_direction=np.concatenate(
            [part.actor_direction for part in parts], axis=0
        ),
        actor_log_magnitude=np.concatenate(
            [part.actor_log_magnitude for part in parts], axis=0
        ),
        camera_direction=np.concatenate(
            [part.camera_direction for part in parts], axis=0
        ),
        camera_log_magnitude=np.concatenate(
            [part.camera_log_magnitude for part in parts], axis=0
        ),
    )


def _compatibility_scores(
    model: SourceAwareFactorizedR6,
    *,
    semantics: np.ndarray,
    motion: R6MotionFeatures,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    torch = _torch()
    pieces: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(semantics), int(batch_size)):
            stop = min(start + int(batch_size), len(semantics))

            def tensor(values: np.ndarray) -> Any:
                return torch.as_tensor(
                    values[start:stop],
                    dtype=torch.float32,
                    device=device,
                )

            logits = model.score_compatibility(
                semantic_features=tensor(semantics),
                candidate_actor_motion=tensor(motion.actor_input),
                candidate_camera_motion=tensor(motion.camera_input),
            )
            pieces.append(
                torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
            )
    return np.concatenate(pieces).astype(np.float32)


def _pair_lookup(
    pairs: IndependentReferencePairs,
) -> dict[int, list[int]]:
    lookup: dict[int, list[tuple[int, int]]] = {}
    for query_raw, reference_raw, rank_raw in zip(
        pairs.query_indices,
        pairs.reference_indices,
        pairs.reference_ranks,
    ):
        lookup.setdefault(int(query_raw), []).append(
            (int(rank_raw), int(reference_raw))
        )
    return {
        query: [reference for _, reference in sorted(references)]
        for query, references in lookup.items()
    }


def _target_tensors(
    targets: FactorizedR5Targets,
    indices: np.ndarray,
    *,
    device: Any,
) -> dict[str, Any]:
    torch = _torch()
    selected = np.asarray(indices, dtype=np.int64)

    def tensor(values: np.ndarray) -> Any:
        return torch.as_tensor(
            values[selected],
            dtype=torch.float32,
            device=device,
        )

    return {
        "actor_direction": tensor(targets.actor_direction),
        "actor_log_magnitude": tensor(
            targets.actor_log_magnitude.reshape(-1, 1)
        ),
        "camera_direction": tensor(targets.camera_direction),
        "camera_log_magnitude": tensor(
            targets.camera_log_magnitude.reshape(-1, 1)
        ),
    }


def _registered_train_mismatch_map(
    batch: R6EndpointBatch,
    *,
    data_seed: int,
) -> dict[int, int]:
    """One-to-one fixed train prompt→other-IID observed-motion mapping."""

    train = [int(index) for index in batch.indices("train")]
    if len(train) < 2:
        raise ValueError("train mismatch map requires at least two IIDs")
    ordered = sorted(
        train,
        key=lambda index: (
            hashlib.sha256(
                (
                    f"{int(data_seed)}\0train-mismatch\0"
                    f"{batch.iids[index]}"
                ).encode("utf-8")
            ).digest(),
            batch.iids[index],
        ),
    )
    rotated = ordered[1:] + ordered[:1]
    result = dict(zip(ordered, rotated))
    if any(
        query == motion
        or batch.iids[query] == batch.iids[motion]
        for query, motion in result.items()
    ):
        raise RuntimeError("registered train mismatch map contains self-IID")
    return result


def _train_one_seed(
    *,
    batch: R6EndpointBatch,
    transform: R6FeatureTransform,
    pairs: IndependentReferencePairs,
    data_seed: int,
    model_seed: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    reference_dropout: float,
    magnitude_weight: float,
    compatibility_weight: float,
    device: Any,
    log_every: int,
) -> tuple[SourceAwareFactorizedR6, list[dict[str, Any]]]:
    """Train predictor and compatibility branches with disjoint optimizers."""

    torch = _torch()
    random.seed(int(model_seed))
    np.random.seed(int(model_seed) % (2**32 - 1))
    torch.manual_seed(int(model_seed))
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.manual_seed_all(int(model_seed))
    source_actor, source_camera, semantics = transform.source_inputs(batch)
    observed = transform.observed_motion(batch)
    targets = observed.as_targets()
    condition_dim = int(observed.actor_direction.shape[1])
    if observed.camera_direction.shape[1] != condition_dim:
        raise ValueError("R6 actor/camera motion condition dimensions differ")
    model = SourceAwareFactorizedR6(
        actor_source_dim=source_actor.shape[1],
        camera_source_dim=source_camera.shape[1],
        semantic_dim=semantics.shape[1],
        condition_dim=condition_dim,
        hidden_dim=int(hidden_dim),
    ).to(device)
    predictor_parameters = list(model.predictor_parameters())
    compatibility_parameters = list(model.compatibility_parameters())
    predictor_ids = {id(parameter) for parameter in predictor_parameters}
    compatibility_ids = {id(parameter) for parameter in compatibility_parameters}
    if not predictor_ids or not compatibility_ids or predictor_ids & compatibility_ids:
        raise RuntimeError("R6 predictor/compatibility parameter audit failed")
    predictor_optimizer = torch.optim.AdamW(
        predictor_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    compatibility_optimizer = torch.optim.AdamW(
        compatibility_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    positive_train = batch.positive_indices("train")
    compatibility_train = batch.indices("train")
    if len(positive_train) < 2 or len(compatibility_train) < 2:
        raise ValueError("R6 training needs at least two train positives/rows")
    compatibility_positive_count = int(
        np.count_nonzero(
            batch.compatibility_targets[compatibility_train] == 1.0
        )
    )
    compatibility_observed_negative_count = int(
        len(compatibility_train) - compatibility_positive_count
    )
    if compatibility_positive_count < 1:
        raise ValueError("R6 compatibility train set has no positive row")
    compatibility_registered_mismatch_count = int(
        len(compatibility_train)
    )
    compatibility_positive_weight = float(
        (
            compatibility_observed_negative_count
            + compatibility_registered_mismatch_count
        )
        / compatibility_positive_count
    )
    mismatch_map = _registered_train_mismatch_map(
        batch,
        data_seed=int(data_seed),
    )
    lookup = _pair_lookup(pairs)
    rng = np.random.default_rng(int(model_seed))
    logs: list[dict[str, Any]] = []
    model.train()
    for step in range(1, int(steps) + 1):
        positive_indices = rng.choice(
            positive_train,
            size=int(batch_size),
            replace=len(positive_train) < int(batch_size),
        ).astype(np.int64)
        reference_actor = np.zeros(
            (len(positive_indices), condition_dim + 1),
            dtype=np.float32,
        )
        reference_camera = np.zeros_like(reference_actor)
        reference_mask = np.zeros(len(positive_indices), dtype=np.float32)
        reference_available = 0
        for position, raw_query in enumerate(positive_indices):
            query = int(raw_query)
            candidates = lookup.get(query, [])
            if not candidates:
                continue
            reference = int(candidates[int(rng.integers(0, len(candidates)))])
            reference_actor[position] = observed.actor_input[reference]
            reference_camera[position] = observed.camera_input[reference]
            reference_mask[position] = 1.0
            reference_available += 1
        if reference_dropout > 0.0:
            keep = (
                rng.random(len(reference_mask)) >= float(reference_dropout)
            ).astype(np.float32)
            reference_mask *= keep

        def tensor(values: np.ndarray) -> Any:
            return torch.as_tensor(
                values,
                dtype=torch.float32,
                device=device,
            )

        predictor_optimizer.zero_grad(set_to_none=True)
        prediction = model(
            source_actor=tensor(source_actor[positive_indices]),
            source_camera=tensor(source_camera[positive_indices]),
            semantic_features=tensor(semantics[positive_indices]),
            reference_actor_motion=tensor(reference_actor),
            reference_camera_motion=tensor(reference_camera),
            reference_mask=tensor(reference_mask),
        )
        delta_losses = positive_factorized_r6_loss(
            prediction,
            _target_tensors(targets, positive_indices, device=device),
            positive_mask=torch.ones(
                len(positive_indices),
                dtype=torch.bool,
                device=device,
            ),
            magnitude_weight=float(magnitude_weight),
        )
        delta_loss = delta_losses["delta_loss"]
        delta_loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor_parameters, max_norm=5.0)
        predictor_optimizer.step()

        compatibility_indices = rng.choice(
            compatibility_train,
            size=int(batch_size),
            replace=len(compatibility_train) < int(batch_size),
        ).astype(np.int64)
        # The wrong target is a pre-registered one-to-one train derangement;
        # replacement sampling can never accidentally create a self-pair.
        wrong_indices = np.asarray(
            [mismatch_map[int(index)] for index in compatibility_indices],
            dtype=np.int64,
        )
        compatibility_semantics = np.concatenate(
            (
                semantics[compatibility_indices],
                semantics[compatibility_indices],
            ),
            axis=0,
        )
        compatibility_actor = np.concatenate(
            (
                observed.actor_input[compatibility_indices],
                observed.actor_input[wrong_indices],
            ),
            axis=0,
        )
        compatibility_camera = np.concatenate(
            (
                observed.camera_input[compatibility_indices],
                observed.camera_input[wrong_indices],
            ),
            axis=0,
        )
        compatibility_targets = np.concatenate(
            (
                batch.compatibility_targets[compatibility_indices],
                np.zeros(len(compatibility_indices), dtype=np.float32),
            )
        )
        compatibility_optimizer.zero_grad(set_to_none=True)
        compatibility_logits = model.score_compatibility(
            semantic_features=tensor(compatibility_semantics),
            candidate_actor_motion=tensor(compatibility_actor),
            candidate_camera_motion=tensor(compatibility_camera),
        )
        compatibility_loss = pair_compatibility_loss(
            compatibility_logits,
            tensor(compatibility_targets),
            positive_weight=compatibility_positive_weight,
        )
        weighted_compatibility_loss = (
            float(compatibility_weight) * compatibility_loss
        )
        weighted_compatibility_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            compatibility_parameters,
            max_norm=5.0,
        )
        compatibility_optimizer.step()

        if (
            step == 1
            or step == int(steps)
            or step % int(log_every) == 0
        ):
            logs.append(
                {
                    "step": step,
                    "delta_loss": float(delta_loss.detach().cpu()),
                    "actor_direction_loss": float(
                        delta_losses["actor_direction"].detach().cpu()
                    ),
                    "actor_magnitude_loss": float(
                        delta_losses["actor_magnitude"].detach().cpu()
                    ),
                    "camera_direction_loss": float(
                        delta_losses["camera_direction"].detach().cpu()
                    ),
                    "camera_magnitude_loss": float(
                        delta_losses["camera_magnitude"].detach().cpu()
                    ),
                    "compatibility_loss": float(
                        compatibility_loss.detach().cpu()
                    ),
                    "positive_predictor_rows": int(len(positive_indices)),
                    "failed_predictor_rows": 0,
                    "reference_available_before_dropout": int(
                        reference_available
                    ),
                    "reference_used_after_dropout": int(
                        np.count_nonzero(reference_mask)
                    ),
                    "compatibility_observed_rows": int(
                        len(compatibility_indices)
                    ),
                    "compatibility_registered_wrong_rows": int(
                        len(compatibility_indices)
                    ),
                    "compatibility_wrong_pair_contract": (
                        "fixed-one-to-one-train-derangement"
                    ),
                    "compatibility_positive_weight": (
                        compatibility_positive_weight
                    ),
                    "compatibility_weight_fit_scope": "train-only-fixed",
                }
            )
    return model, logs


def _new_random_model(
    trained: SourceAwareFactorizedR6,
    *,
    model_seed: int,
    device: Any,
) -> SourceAwareFactorizedR6:
    torch = _torch()
    torch.manual_seed(int(model_seed))
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.manual_seed_all(int(model_seed))
    return SourceAwareFactorizedR6(
        actor_source_dim=trained.actor_source_dim,
        camera_source_dim=trained.camera_source_dim,
        semantic_dim=trained.semantic_dim,
        condition_dim=trained.condition_dim,
        hidden_dim=trained.hidden_dim,
    ).to(device)


def _build_arm_predictions(
    *,
    model: SourceAwareFactorizedR6,
    random_model: SourceAwareFactorizedR6,
    source_actor: np.ndarray,
    source_camera: np.ndarray,
    semantics: np.ndarray,
    observed_motion: R6MotionFeatures,
    centroid: FactorizedR5Targets,
    independent_actor: np.ndarray,
    independent_camera: np.ndarray,
    independent_mask: np.ndarray,
    wrong_actor: np.ndarray,
    wrong_camera: np.ndarray,
    wrong_mask: np.ndarray,
    semantic_shuffle_indices: np.ndarray,
    semantic_shuffle_valid: np.ndarray,
    source_shuffle_indices: np.ndarray,
    source_shuffle_valid: np.ndarray,
    device: Any,
    eval_batch_size: int,
) -> dict[str, tuple[FactorizedR5Targets, np.ndarray]]:
    rows = len(source_actor)
    ones = np.ones(rows, dtype=bool)
    return {
        "semantic_only": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics,
                reference_actor=None,
                reference_camera=None,
                reference_mask=None,
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
        "independent_ref": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics,
                reference_actor=independent_actor,
                reference_camera=independent_camera,
                reference_mask=independent_mask,
                device=device,
                batch_size=eval_batch_size,
            ),
            independent_mask.astype(bool),
        ),
        "wrong_ref": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics,
                reference_actor=wrong_actor,
                reference_camera=wrong_camera,
                reference_mask=wrong_mask,
                device=device,
                batch_size=eval_batch_size,
            ),
            wrong_mask.astype(bool),
        ),
        "matched_random": (
            _predict(
                random_model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics,
                reference_actor=None,
                reference_camera=None,
                reference_mask=None,
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
        "centroid": (centroid, ones),
        "source_shuffle": (
            _predict(
                model,
                source_actor=source_actor[source_shuffle_indices],
                source_camera=source_camera[source_shuffle_indices],
                semantics=semantics,
                reference_actor=None,
                reference_camera=None,
                reference_mask=None,
                device=device,
                batch_size=eval_batch_size,
            ),
            source_shuffle_valid,
        ),
        "semantic_shuffle": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics[semantic_shuffle_indices],
                reference_actor=None,
                reference_camera=None,
                reference_mask=None,
                device=device,
                batch_size=eval_batch_size,
            ),
            semantic_shuffle_valid,
        ),
        "exact_target_oracle": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                semantics=semantics,
                reference_actor=observed_motion.actor_input,
                reference_camera=observed_motion.camera_input,
                reference_mask=np.ones(rows, dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
    }


def _per_query_rows(
    *,
    predictions: Mapping[str, tuple[FactorizedR5Targets, np.ndarray]],
    targets: FactorizedR5Targets,
    batch: R6EndpointBatch,
    data_seed: int,
    model_seed: int,
    compatibility_scores: np.ndarray,
    synthetic_mismatch_scores: np.ndarray,
    synthetic_mismatch_valid: np.ndarray,
    alternate_prediction: FactorizedR5Targets,
    alternate_reference_mask: np.ndarray,
    alternate_reference_index: np.ndarray,
    alternate_reference_score: np.ndarray,
    independent_reference_index: np.ndarray,
    independent_reference_score: np.ndarray,
    wrong_reference_index: np.ndarray,
    wrong_reference_score: np.ndarray,
    active_threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    roles = np.asarray(batch.label_roles)
    active = targets.actor_log_magnitude > float(active_threshold)
    for arm in R6_ARMS:
        prediction, control_valid_raw = predictions[arm]
        control_valid = np.asarray(control_valid_raw, dtype=bool)
        retrieval_by_index: dict[int, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            selected = np.flatnonzero(np.asarray(batch.splits) == split)
            if not len(selected):
                continue
            retrieval = cross_content_retrieval(
                predicted_actor_direction=prediction.actor_direction[selected],
                target_actor_direction=targets.actor_direction[selected],
                action_families=[
                    batch.action_families[int(index)] for index in selected
                ],
                content_group_ids=[
                    batch.content_group_ids[int(index)] for index in selected
                ],
                iids=[batch.iids[int(index)] for index in selected],
                valid_mask=(
                    roles[selected] == "positive_delta"
                ),
                active_mask=active[selected],
            )
            for position, index in enumerate(selected):
                if not control_valid[int(index)]:
                    retrieval[position] = {
                        **retrieval[position],
                        "retrieval_valid": False,
                        "actor_cross_content_ap": None,
                        "actor_cross_content_r1": None,
                        "actor_cross_content_r5": None,
                        "fixed_gallery_query_control_invalid": True,
                    }
            retrieval_by_index.update(
                {
                    int(index): value
                    for index, value in zip(selected, retrieval)
                }
            )
        for index, iid in enumerate(batch.iids):
            positive = batch.label_roles[index] == "positive_delta"
            metric_valid = bool(
                positive and active[index] and control_valid[index]
            )
            actor_cosine = (
                float(
                    np.dot(
                        prediction.actor_direction[index].astype(np.float64),
                        targets.actor_direction[index].astype(np.float64),
                    )
                )
                if metric_valid
                else None
            )
            camera_active = (
                targets.camera_log_magnitude[index]
                > float(active_threshold)
            )
            camera_cosine = (
                float(
                    np.dot(
                        prediction.camera_direction[index].astype(np.float64),
                        targets.camera_direction[index].astype(np.float64),
                    )
                )
                if positive and camera_active and control_valid[index]
                else None
            )
            retrieval = retrieval_by_index.get(
                index,
                {
                    "retrieval_valid": False,
                    "actor_cross_content_ap": None,
                    "actor_cross_content_r1": None,
                    "actor_cross_content_r5": None,
                },
            )
            row = {
                "schema_version": R6_QUERY_SCHEMA,
                "arm": arm,
                "model_seed": int(model_seed),
                "data_seed": int(data_seed),
                "iid": iid,
                "split": batch.splits[index],
                "content_group_id": batch.content_group_ids[index],
                "subject_cluster_id": batch.subject_cluster_ids[index],
                "action_family": batch.action_families[index],
                "label_role": batch.label_roles[index],
                "compatibility_target": float(
                    batch.compatibility_targets[index]
                ),
                "compatibility_probability": float(
                    compatibility_scores[index]
                ),
                "synthetic_mismatched_positive_probability": (
                    float(synthetic_mismatch_scores[index])
                    if synthetic_mismatch_valid[index]
                    else None
                ),
                "synthetic_mismatch_is_real_failed_outcome": False,
                "control_valid": bool(control_valid[index]),
                "target_actor_active": bool(active[index]),
                "actor_cosine": actor_cosine,
                "actor_log_magnitude_mae": (
                    float(
                        abs(
                            prediction.actor_log_magnitude[index]
                            - targets.actor_log_magnitude[index]
                        )
                    )
                    if positive and control_valid[index]
                    else None
                ),
                "camera_cosine": camera_cosine,
                "camera_log_magnitude_mae": (
                    float(
                        abs(
                            prediction.camera_log_magnitude[index]
                            - targets.camera_log_magnitude[index]
                        )
                    )
                    if positive and control_valid[index]
                    else None
                ),
                "retrieval_valid": bool(retrieval["retrieval_valid"]),
                "actor_cross_content_ap": retrieval[
                    "actor_cross_content_ap"
                ],
                "actor_cross_content_r1": retrieval[
                    "actor_cross_content_r1"
                ],
                "actor_cross_content_r5": retrieval[
                    "actor_cross_content_r5"
                ],
                "oracle_diagnostic": arm == "exact_target_oracle",
                "gate_eligible": arm in R6_GATE_ARMS,
                "query_target_used_as_predictor_input": (
                    arm == "exact_target_oracle"
                ),
                "failed_outcome_used_as_noop": False,
                "compatibility_scales_conditioning_tokens": False,
            }
            if arm == "independent_ref":
                reference = int(independent_reference_index[index])
                alternate_reference = int(
                    alternate_reference_index[index]
                )
                alternate_available = bool(
                    control_valid[index]
                    and alternate_reference_mask[index]
                )
                row.update(
                    {
                        "reference_iid": (
                            batch.iids[reference] if reference >= 0 else None
                        ),
                        "reference_semantic_cosine": (
                            float(independent_reference_score[index])
                            if reference >= 0
                            else None
                        ),
                        "alternate_reference_iid": (
                            batch.iids[alternate_reference]
                            if alternate_reference >= 0
                            else None
                        ),
                        "alternate_reference_semantic_cosine": (
                            float(alternate_reference_score[index])
                            if alternate_reference >= 0
                            else None
                        ),
                        "alternate_reference_available": (
                            alternate_available
                        ),
                        "alternate_reference_prediction_cosine": (
                            float(
                                np.dot(
                                    prediction.actor_direction[
                                        index
                                    ].astype(np.float64),
                                    alternate_prediction.actor_direction[
                                        index
                                    ].astype(np.float64),
                                )
                            )
                            if alternate_available
                            else None
                        ),
                    }
                )
            elif arm == "wrong_ref":
                reference = int(wrong_reference_index[index])
                row.update(
                    {
                        "reference_iid": (
                            batch.iids[reference] if reference >= 0 else None
                        ),
                        "reference_semantic_cosine": (
                            float(wrong_reference_score[index])
                            if reference >= 0
                            else None
                        ),
                    }
                )
            output.append(row)
    return output


def _atomic_torch_save(path: Path, value: Any) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in R6_ARMS:
        result[arm] = {}
        for split in ("train", "validation", "test"):
            selected = [
                row
                for row in rows
                if row["arm"] == arm
                and row["split"] == split
                and row["label_role"] == "positive_delta"
                and row["control_valid"]
            ]

            def mean(name: str) -> float | None:
                values = [
                    float(row[name])
                    for row in selected
                    if row.get(name) is not None
                ]
                return float(np.mean(values)) if values else None

            result[arm][split] = {
                "positive_rows": len(selected),
                "actor_cosine": mean("actor_cosine"),
                "actor_log_magnitude_mae": mean(
                    "actor_log_magnitude_mae"
                ),
                "camera_cosine": mean("camera_cosine"),
                "camera_log_magnitude_mae": mean(
                    "camera_log_magnitude_mae"
                ),
                "actor_cross_content_mAP": mean(
                    "actor_cross_content_ap"
                ),
            }
    return result


def _parse_model_seeds(value: str) -> list[int]:
    try:
        seeds = [
            int(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "model seeds must be comma-separated integers"
        ) from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "model seeds must be non-empty and unique"
        )
    return seeds


def train_and_evaluate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser()
    output_paths = {
        "ledger": output_dir / "reference_pairs.jsonl",
        "reference_audit": output_dir / "reference_audit.json",
        "contract": output_dir / "contract.json",
        "transform": output_dir / "transform.json",
        "per_query": output_dir / "per_query.jsonl",
        "gate": output_dir / "gate_summary.json",
        "summary": output_dir / "summary.json",
        "done": output_dir / "done.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "R6 output already contains committed paths; use a fresh "
            f"directory: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = args.source_snapshot.expanduser().resolve(strict=True)
    source_files_manifest = source_snapshot / "SOURCE_FILES.jsonl"
    if not source_files_manifest.is_file():
        raise FileNotFoundError(source_files_manifest)
    source_tree_sha256 = _validated_sha256(
        args.source_tree_sha256,
        name="--source-tree-sha256",
    )
    source_files_sha256 = _validated_sha256(
        args.source_files_sha256,
        name="--source-files-sha256",
    )
    if _file_digest(source_files_manifest) != source_files_sha256:
        raise ValueError(
            "SOURCE_FILES.jsonl bytes disagree with --source-files-sha256"
        )
    expected_code_root = (
        source_snapshot / "methods" / "motive"
    ).resolve(strict=True)
    actual_module_path = Path(__file__).resolve()
    if expected_code_root not in actual_module_path.parents:
        raise RuntimeError(
            "R6 trainer was imported outside the bound source snapshot: "
            f"{actual_module_path}"
        )
    model_seeds = _parse_model_seeds(args.model_seeds)
    r5_batch, manifest_rows, feature_provenance = load_feature_bundle(
        feature_archive=args.features,
        manifest_path=args.manifest,
        data_seed=int(args.data_seed),
    )
    labels = derive_labels(manifest_rows, label_mode=args.label_mode)
    production_split = (
        feature_provenance["split_version"]
        == "source-visual-cluster-v1"
    )
    (
        semantic_embeddings,
        semantic_input_digests,
        semantic_provenance,
        observed_action_bank,
        semantic_bundle_provenance,
    ) = load_semantic_bundle(
        semantic_archive=args.semantic_features,
        encoder=args.encoder,
        expected_iids=r5_batch.iids,
        expected_input_manifest_sha256=feature_provenance[
            "manifest_sha256"
        ],
        production=production_split,
    )
    batch, subject_cluster_source = make_r6_batch(
        r5_batch=r5_batch,
        manifest_rows=manifest_rows,
        labels=labels,
        semantic_embeddings=semantic_embeddings,
        semantic_input_digests=semantic_input_digests,
        semantic_provenance=semantic_provenance,
        production=production_split,
    )
    positive_train = batch.positive_indices("train")
    positive_validation = batch.positive_indices("validation")
    positive_test = batch.positive_indices("test")
    if len(positive_train) < 2:
        raise ValueError("R6 requires at least two positive train rows")
    if not len(positive_validation) or not len(positive_test):
        raise ValueError("R6 requires positive validation and test rows")
    pairs = build_semantic_train_bank_reference_pairs(
        batch,
        observed_action_bank,
        data_seed=int(args.data_seed),
        references_per_query=int(args.references_per_query),
        threshold_quantile=0.10,
        include_failed_outcomes=True,
        require_complete=bool(args.require_complete_references),
    )
    ledger_rows = _pair_ledger_rows(pairs, batch)
    for ledger_row in ledger_rows:
        ledger_row["subject_cluster_source"] = subject_cluster_source
        ledger_row["subject_cluster_verified"] = (
            subject_cluster_source == "manifest-explicit"
        )
    _atomic_jsonl(output_paths["ledger"], ledger_rows)
    ledger_sha256 = _file_digest(output_paths["ledger"])
    reference_audit = _reference_audit(pairs, batch)
    reference_audit["pair_ledger"] = str(output_paths["ledger"])
    reference_audit["pair_ledger_sha256"] = ledger_sha256
    _atomic_json(output_paths["reference_audit"], reference_audit)
    test_coverage = reference_audit["coverage"]["test"]
    print(
        "[r6-train] pretraining reference audit "
        f"encoder={args.encoder} pairs={len(ledger_rows)} "
        f"test_any={test_coverage['any_reference_fraction']:.3f} "
        f"test_full={test_coverage['full_reference_fraction']:.3f} "
        f"unpaired={reference_audit['unpaired_count']} "
        f"threshold={pairs.similarity_threshold:.6f}",
        flush=True,
    )

    transform = R6FeatureTransform.fit(
        batch,
        condition_dim=int(args.condition_dim),
        semantic_condition_dim=int(args.semantic_condition_dim),
    )
    source_actor, source_camera, transformed_semantics = (
        transform.source_inputs(batch)
    )
    observed_motion = transform.observed_motion(batch)
    targets = observed_motion.as_targets()
    centroid = _global_centroid(
        targets,
        positive_train,
        rows=len(batch.iids),
    )
    (
        independent_actor,
        independent_camera,
        independent_mask,
        independent_reference_index,
        independent_reference_score,
    ) = _primary_reference_arrays(
        pairs=pairs,
        batch=batch,
        observed_motion=observed_motion,
    )
    (
        alternate_actor,
        alternate_camera,
        alternate_mask,
        alternate_reference_index,
        alternate_reference_score,
    ) = _primary_reference_arrays(
        pairs=pairs,
        batch=batch,
        observed_motion=observed_motion,
        reference_rank=1,
    )
    (
        wrong_actor,
        wrong_camera,
        wrong_mask,
        wrong_reference_index,
        wrong_reference_score,
    ) = _wrong_reference_arrays(
        batch=batch,
        observed_action_bank=observed_action_bank,
        observed_motion=observed_motion,
        primary_reference_index=independent_reference_index,
        data_seed=int(args.data_seed),
    )
    semantic_shuffle_indices, semantic_shuffle_valid = (
        _semantic_shuffle_indices(
            batch,
            data_seed=int(args.data_seed),
        )
    )
    source_shuffle_indices, source_shuffle_valid = (
        _semantic_shuffle_indices(
            batch,
            data_seed=int(args.data_seed) + 97_003,
        )
    )
    synthetic_mismatch_indices, synthetic_mismatch_valid = (
        _positive_mismatch_indices(
            batch,
            data_seed=int(args.data_seed),
        )
    )
    reference_load = reference_audit[
        "reference_load_test_positive_rank0"
    ]
    compatibility_train_indices = batch.indices("train")
    compatibility_train_positive_count = int(
        np.count_nonzero(
            batch.compatibility_targets[compatibility_train_indices] == 1.0
        )
    )
    compatibility_train_observed_negative_count = int(
        len(compatibility_train_indices)
        - compatibility_train_positive_count
    )
    compatibility_train_mismatch_count = int(
        len(compatibility_train_indices)
    )
    compatibility_positive_weight = float(
        (
            compatibility_train_observed_negative_count
            + compatibility_train_mismatch_count
        )
        / compatibility_train_positive_count
    )
    torch = _torch()
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested but is unavailable: {requested_device}"
        )
    device = torch.device(requested_device)
    torch.use_deterministic_algorithms(True)
    runtime_provenance = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "requested_device": requested_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    contract = {
        "schema_version": R6_TRAIN_SCHEMA,
        "data_seed": int(args.data_seed),
        "model_seeds": model_seeds,
        "encoder": args.encoder,
        "source_snapshot": {
            "path": str(source_snapshot),
            "tree_sha256": source_tree_sha256,
            "source_files_manifest": str(source_files_manifest),
            "source_files_sha256": source_files_sha256,
            "trainer_module_path": str(actual_module_path),
        },
        "runtime": runtime_provenance,
        "dataset": {
            "label_mode": args.label_mode,
            "production_eligible": False,
            "split_version": feature_provenance["split_version"],
            "subject_cluster_source": subject_cluster_source,
            "action_family_source_verified": False,
            "action_family_usage": "descriptive-retrieval-and-oracle-only",
            "row_count": len(batch.iids),
            "iid_set_digest": _object_digest(sorted(batch.iids)),
            "positive_count": int(
                np.count_nonzero(
                    np.asarray(batch.label_roles) == "positive_delta"
                )
            ),
            "failed_outcome_count": int(
                np.count_nonzero(
                    np.asarray(batch.label_roles)
                    == "failed_outcome_compatibility"
                )
            ),
            "positive_train_count": len(positive_train),
            "positive_validation_count": len(positive_validation),
            "positive_test_count": len(positive_test),
            "split_counts": dict(Counter(batch.splits)),
            "split_role_counts": dict(
                Counter(
                    f"{split}:{role}"
                    for split, role in zip(
                        batch.splits,
                        batch.label_roles,
                    )
                )
            ),
        },
        "feature_artifact": {
            "archive": feature_provenance["feature_archive"],
            "archive_sha256": feature_provenance[
                "feature_archive_sha256"
            ],
            "manifest": feature_provenance["manifest"],
            "manifest_sha256": feature_provenance["manifest_sha256"],
        },
        "semantic_artifact": {
            **semantic_bundle_provenance["query_provenance"],
            "provenance_digest": semantic_bundle_provenance[
                "query_provenance_digest"
            ],
            "archive": semantic_bundle_provenance["archive"],
            "archive_sha256": semantic_bundle_provenance[
                "archive_sha256"
            ],
            "source_field": "instruction",
            "frozen_encoder": True,
            "target_derived_input": False,
            "label_derived_input": False,
        },
        "reference_selector": {
            "selector_kind": (
                "prompt-to-observed-action-semantic-train-bank"
            ),
            "pairing_version": pairs.pairing_version,
            "candidate_bank_split": "train",
            "candidate_bank_label_role": "positive_delta",
            "threshold_fit_split": "train",
            "threshold_fit_role": "positive_delta",
            "threshold_origin": "train-positive-self-alignment-q10",
            "threshold_quantile": 0.10,
            "similarity_threshold": float(pairs.similarity_threshold),
            "query_target_used": False,
            "different_iid_enforced": True,
            "different_content_group_enforced": True,
            "different_subject_cluster_enforced": (
                subject_cluster_source == "manifest-explicit"
            ),
            "subject_cluster_source": subject_cluster_source,
            "subject_cluster_verified": (
                subject_cluster_source == "manifest-explicit"
            ),
            "oracle_action_family_used": False,
            "gate_eligible": True,
            "pair_digest": pairs.digest(),
            "pair_ledger": str(output_paths["ledger"]),
            "pair_ledger_sha256": ledger_sha256,
            "reference_bank_provenance_digest": (
                semantic_bundle_provenance[
                    "observed_bank_provenance_digest"
                ]
            ),
            "test_positive_coverage": test_coverage,
            "reference_load": {
                **reference_load,
                "maximum_reference_fraction": reference_load[
                    "maximum_reference_load_fraction"
                ],
            },
            "reference_load_all_k_descriptive": reference_audit[
                "reference_load_test_positive_all_ranks"
            ],
            "reference_load_global_descriptive": reference_audit[
                "reference_load_global_descriptive"
            ],
            "no_fallback_selector": True,
            "coverage_shortfall_is_explicit": True,
        },
        "input_transform_fit_split": "train",
        "delta_transform_fit_split": "train",
        "delta_transform_fit_role": "positive_delta",
        "transform": {
            "schema_version": transform.schema_version,
            "digest": transform.digest(),
            "fit_input_train_iid_digest": getattr(
                transform,
                "fit_input_train_iid_digest",
            ),
            "fit_delta_positive_train_iid_digest": getattr(
                transform,
                "fit_delta_positive_train_iid_digest",
            ),
            "input_train_count": len(batch.indices("train")),
            "delta_positive_train_count": len(positive_train),
            "semantic_projection_dim": int(
                args.semantic_condition_dim
            ),
            "motion_condition_dim": int(args.condition_dim),
        },
        "query_target_is_predictor_input": False,
        "failed_outcomes_update_delta_predictor": False,
        "compatibility_scales_conditioning_tokens": False,
        "failed_outcome_semantics": {
            "compatibility_target": 0,
            "used_as_noop": False,
            "used_as_activity_label": False,
            "used_to_scale_generation_token": False,
        },
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "semantic_condition_dim": int(args.semantic_condition_dim),
            "motion_condition_dim": int(args.condition_dim),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "reference_dropout": float(args.reference_dropout),
            "magnitude_weight": float(args.magnitude_weight),
            "compatibility_weight": float(args.compatibility_weight),
            "predictor_optimizer_scope": "predictor_parameters_only",
            "compatibility_optimizer_scope": (
                "compatibility_parameters_only"
            ),
            "failed_rows_in_delta_loss": 0,
            "in_batch_wrong_motion_compatibility": False,
            "registered_train_wrong_motion_compatibility": True,
            "registered_train_wrong_motion_contract": (
                "fixed-one-to-one-train-derangement"
            ),
            "registered_train_wrong_motion_map_digest": _object_digest(
                {
                    batch.iids[query]: batch.iids[motion]
                    for query, motion in sorted(
                        _registered_train_mismatch_map(
                            batch,
                            data_seed=int(args.data_seed),
                        ).items()
                    )
                }
            ),
            "compatibility_class_weight": {
                "fit_scope": "train-only-fixed",
                "fit_iid_digest": _object_digest(
                    sorted(
                        batch.iids[int(index)]
                        for index in compatibility_train_indices
                    )
                ),
                "observed_positive_count": (
                    compatibility_train_positive_count
                ),
                "observed_failed_count": (
                    compatibility_train_observed_negative_count
                ),
                "registered_one_to_one_mismatch_count": (
                    compatibility_train_mismatch_count
                ),
                "positive_weight": compatibility_positive_weight,
            },
        },
        "evaluation": {
            "active_log_magnitude_threshold": float(
                args.active_threshold
            ),
            "active_threshold_origin": "fixed-pre-registered-1e-4",
            "active_threshold_fit_scope": "none",
            "split_positive_active_counts": {
                split: {
                    "active": int(
                        np.count_nonzero(
                            (
                                targets.actor_log_magnitude[
                                    batch.indices(split)
                                ]
                                > float(args.active_threshold)
                            )
                            & (
                                np.asarray(batch.label_roles)[
                                    batch.indices(split)
                                ]
                                == "positive_delta"
                            )
                        )
                    ),
                    "total_positive": int(
                        np.count_nonzero(
                            np.asarray(batch.label_roles)[
                                batch.indices(split)
                            ]
                            == "positive_delta"
                        )
                    ),
                }
                for split in ("train", "validation", "test")
            },
            "family_retrieval_gate_eligible": False,
            "family_retrieval_reason": (
                "auto/Qwen families are descriptive only"
            ),
        },
        "arms": {
            "semantic_only": "trained predictor, no reference motion",
            "independent_ref": (
                "primary prompt-to-train-observed-action semantic selector"
            ),
            "wrong_ref": (
                "least-aligned legal train-positive reference control"
            ),
            "matched_random": (
                "untrained identical architecture, source+semantic only "
                "(input-matched to semantic_only)"
            ),
            "centroid": "global positive-train motion centroid",
            "source_shuffle": (
                "trained semantic-only predictor with deterministic "
                "within-split rotated source actor/camera and query semantics"
            ),
            "semantic_shuffle": (
                "trained predictor with within-split shuffled semantics"
            ),
            "exact_target_oracle": (
                "query target motion as reference; oracle diagnostic only, "
                "excluded from gate"
            ),
        },
        "formal_gate_enabled": False,
        "formal_gate_status": "INSUFFICIENT",
        "deferred_controls": {
            "reference_only": {
                "implemented": False,
                "unsupported_claim": "semantic/reference disentanglement",
            },
            "unfiltered_reference": {
                "implemented": False,
                "unsupported_claim": "compatibility-filter utility",
            },
            "note": (
                "These controls are explicitly deferred; no R6 pilot result "
                "may be interpreted as evidence for the unsupported claims."
            ),
        },
        "claim_scope": {
            "representation_pilot": True,
            "action_editing": False,
            "i2v": False,
            "generation_authorized": False,
            "generator_ready_tokens": False,
            "motion_token_export_authorized": False,
            "encoder_comparison_is_pre_registered_independent": True,
            "test_set_encoder_selection_forbidden": True,
        },
    }
    _atomic_json(output_paths["contract"], contract)
    _atomic_json(output_paths["transform"], asdict(transform))

    all_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    for model_seed in model_seeds:
        seed_dir = output_dir / "seeds" / f"seed-{model_seed}"
        model, logs = _train_one_seed(
            batch=batch,
            transform=transform,
            pairs=pairs,
            data_seed=int(args.data_seed),
            model_seed=int(model_seed),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            hidden_dim=int(args.hidden_dim),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            reference_dropout=float(args.reference_dropout),
            magnitude_weight=float(args.magnitude_weight),
            compatibility_weight=float(args.compatibility_weight),
            device=device,
            log_every=int(args.log_every),
        )
        random_seed = int(model_seed) + 1_000_003
        random_model = _new_random_model(
            model,
            model_seed=random_seed,
            device=device,
        )
        checkpoint = seed_dir / "model.pt"
        random_checkpoint = seed_dir / "matched_random.pt"
        _atomic_torch_save(
            checkpoint,
            {
                "schema_version": R6_TRAIN_SCHEMA,
                "encoder": args.encoder,
                "model_seed": int(model_seed),
                "data_seed": int(args.data_seed),
                "architecture": {
                    "actor_source_dim": model.actor_source_dim,
                    "camera_source_dim": model.camera_source_dim,
                    "semantic_dim": model.semantic_dim,
                    "condition_dim": model.condition_dim,
                    "hidden_dim": model.hidden_dim,
                },
                "state_dict": model.state_dict(),
                "transform": _json_ready(asdict(transform)),
                "transform_digest": transform.digest(),
                "pair_digest": pairs.digest(),
                "semantic_archive_sha256": (
                    semantic_bundle_provenance["archive_sha256"]
                ),
                "feature_archive_sha256": feature_provenance[
                    "feature_archive_sha256"
                ],
                "manifest_sha256": feature_provenance[
                    "manifest_sha256"
                ],
                "source_tree_sha256": source_tree_sha256,
                "source_files_sha256": source_files_sha256,
                "runtime_digest": _object_digest(runtime_provenance),
                "failed_outcomes_update_delta_predictor": False,
                "compatibility_scales_conditioning_tokens": False,
            },
        )
        _atomic_torch_save(
            random_checkpoint,
            {
                "schema_version": R6_TRAIN_SCHEMA,
                "arm": "matched_random",
                "model_seed": random_seed,
                "paired_trained_model_seed": int(model_seed),
                "state_dict": random_model.state_dict(),
                "transform_digest": transform.digest(),
                "pair_digest": pairs.digest(),
            },
        )
        _atomic_jsonl(seed_dir / "train_log.jsonl", logs)
        predictions = _build_arm_predictions(
            model=model,
            random_model=random_model,
            source_actor=source_actor,
            source_camera=source_camera,
            semantics=transformed_semantics,
            observed_motion=observed_motion,
            centroid=centroid,
            independent_actor=independent_actor,
            independent_camera=independent_camera,
            independent_mask=independent_mask,
            wrong_actor=wrong_actor,
            wrong_camera=wrong_camera,
            wrong_mask=wrong_mask,
            semantic_shuffle_indices=semantic_shuffle_indices,
            semantic_shuffle_valid=semantic_shuffle_valid,
            source_shuffle_indices=source_shuffle_indices,
            source_shuffle_valid=source_shuffle_valid,
            device=device,
            eval_batch_size=int(args.eval_batch_size),
        )
        alternate_prediction = _predict(
            model,
            source_actor=source_actor,
            source_camera=source_camera,
            semantics=transformed_semantics,
            reference_actor=alternate_actor,
            reference_camera=alternate_camera,
            reference_mask=alternate_mask,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        compatibility = _compatibility_scores(
            model,
            semantics=transformed_semantics,
            motion=observed_motion,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        synthetic_mismatch = _compatibility_scores(
            model,
            semantics=transformed_semantics,
            motion=_reindex_motion(
                observed_motion,
                synthetic_mismatch_indices,
            ),
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        seed_rows = _per_query_rows(
            predictions=predictions,
            targets=targets,
            batch=batch,
            data_seed=int(args.data_seed),
            model_seed=int(model_seed),
            compatibility_scores=compatibility,
            synthetic_mismatch_scores=synthetic_mismatch,
            synthetic_mismatch_valid=synthetic_mismatch_valid,
            alternate_prediction=alternate_prediction,
            alternate_reference_mask=alternate_mask,
            alternate_reference_index=alternate_reference_index,
            alternate_reference_score=alternate_reference_score,
            independent_reference_index=independent_reference_index,
            independent_reference_score=independent_reference_score,
            wrong_reference_index=wrong_reference_index,
            wrong_reference_score=wrong_reference_score,
            active_threshold=float(args.active_threshold),
        )
        metrics = {
            "schema_version": R6_TRAIN_SCHEMA,
            "encoder": args.encoder,
            "model_seed": int(model_seed),
            "data_seed": int(args.data_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_digest(checkpoint),
            "matched_random_checkpoint": str(random_checkpoint),
            "matched_random_checkpoint_sha256": _file_digest(
                random_checkpoint
            ),
            "train_log": str(seed_dir / "train_log.jsonl"),
            "train_log_sha256": _file_digest(
                seed_dir / "train_log.jsonl"
            ),
            "train_final": logs[-1],
            "arms": _aggregate_seed_rows(seed_rows),
        }
        _atomic_json(seed_dir / "metrics.json", metrics)
        per_seed[str(model_seed)] = {
            **metrics,
            "metrics": str(seed_dir / "metrics.json"),
            "metrics_sha256": _file_digest(seed_dir / "metrics.json"),
        }
        all_rows.extend(seed_rows)
        print(
            f"[r6-train] encoder={args.encoder} seed={model_seed} "
            f"delta_loss={logs[-1]['delta_loss']:.6f} "
            f"compat_loss={logs[-1]['compatibility_loss']:.6f} "
            f"rows={len(seed_rows)}",
            flush=True,
        )
    all_rows.sort(
        key=lambda row: (
            int(row["model_seed"]),
            str(row["arm"]),
            str(row["split"]),
            str(row["iid"]),
        )
    )
    _atomic_jsonl(output_paths["per_query"], all_rows)
    gate = evaluate_r6_gate(
        rows=all_rows,
        contract=contract,
        thresholds=R6PilotThresholds(
            bootstrap_samples=int(args.bootstrap_samples)
        ),
        random_seed=int(args.data_seed),
        verified_pair_ledger_sha256=ledger_sha256,
    )
    _atomic_json(output_paths["gate"], gate)
    summary = {
        "schema_version": R6_TRAIN_SCHEMA,
        "status": "complete",
        "encoder": args.encoder,
        "formal_gate_status": gate["status"],
        "pilot_diagnostic_status": gate["pilot_diagnostic"]["status"],
        "production_decision": False,
        "generation_authorized": False,
        "source_tree_sha256": source_tree_sha256,
        "source_files_sha256": source_files_sha256,
        "runtime": runtime_provenance,
        "contract": str(output_paths["contract"]),
        "contract_sha256": _file_digest(output_paths["contract"]),
        "transform": str(output_paths["transform"]),
        "transform_sha256": _file_digest(output_paths["transform"]),
        "reference_audit": str(output_paths["reference_audit"]),
        "reference_audit_sha256": _file_digest(
            output_paths["reference_audit"]
        ),
        "pair_ledger": str(output_paths["ledger"]),
        "pair_ledger_sha256": ledger_sha256,
        "per_query": str(output_paths["per_query"]),
        "per_query_sha256": _file_digest(output_paths["per_query"]),
        "per_query_rows": len(all_rows),
        "gate": str(output_paths["gate"]),
        "gate_sha256": _file_digest(output_paths["gate"]),
        "per_seed": per_seed,
        "per_seed_artifacts_digest": _object_digest(per_seed),
    }
    _atomic_json(output_paths["summary"], summary)
    done = {
        "schema_version": R6_TRAIN_SCHEMA,
        "status": "complete",
        "encoder": args.encoder,
        "formal_gate_status": gate["status"],
        "pilot_diagnostic_status": gate["pilot_diagnostic"]["status"],
        "production_decision": False,
        "generation_authorized": False,
        "source_tree_sha256": source_tree_sha256,
        "source_files_sha256": source_files_sha256,
        "runtime_digest": _object_digest(runtime_provenance),
        "contract_sha256": summary["contract_sha256"],
        "transform_sha256": summary["transform_sha256"],
        "reference_audit_sha256": summary["reference_audit_sha256"],
        "pair_ledger_sha256": summary["pair_ledger_sha256"],
        "per_query_sha256": summary["per_query_sha256"],
        "gate_sha256": summary["gate_sha256"],
        "summary_sha256": _file_digest(output_paths["summary"]),
        "per_seed_artifacts_digest": summary[
            "per_seed_artifacts_digest"
        ],
        "model_seeds": model_seeds,
    }
    _atomic_json(output_paths["done"], done)
    print(
        f"[r6-train] complete encoder={args.encoder} "
        f"formal={gate['status']} "
        f"pilot={gate['pilot_diagnostic']['status']} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-features", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--source-files-sha256", required=True)
    parser.add_argument(
        "--encoder",
        choices=sorted(R6_SEMANTIC_ENCODERS),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-mode",
        choices=("human", "strict_legacy_qwen"),
        required=True,
    )
    parser.add_argument("--data-seed", type=int, default=260108828)
    parser.add_argument(
        "--model-seeds",
        default="2026,2027,2028,2029,2030",
    )
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--semantic-condition-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--references-per-query", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument("--magnitude-weight", type=float, default=0.25)
    parser.add_argument("--compatibility-weight", type=float, default=1.0)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument(
        "--require-complete-references",
        action="store_true",
        help=(
            "Abort on any pairing coverage shortfall. Without this flag, "
            "shortfalls remain explicit/invalid for affected controls and "
            "are never replaced by an oracle family fallback."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "steps",
        "batch_size",
        "eval_batch_size",
        "hidden_dim",
        "semantic_condition_dim",
        "condition_dim",
        "references_per_query",
        "log_every",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.bootstrap_samples) < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    for name in (
        "learning_rate",
        "magnitude_weight",
        "compatibility_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and positive"
            )
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("--reference-dropout must be in [0,1]")
    if (
        not math.isfinite(args.active_threshold)
        or abs(float(args.active_threshold) - 1e-4) > 1e-12
    ):
        raise ValueError(
            "--active-threshold is pre-registered and must equal 1e-4"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    try:
        return train_and_evaluate(args)
    except Exception as error:
        output_dir = args.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        invalid_path = output_dir / "gate_summary.json"
        if not invalid_path.exists():
            _atomic_json(invalid_path, invalid_r6_gate_summary(error))
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
