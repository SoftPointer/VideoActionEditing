"""Cosine retrieval and multi-query voting for motion representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    indices: np.ndarray
    votes: np.ndarray
    mean_scores: np.ndarray
    score_matrix: np.ndarray
    thresholds: np.ndarray


def _as_feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N, D] or [D]")
    return matrix


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = _as_feature_matrix(values, "values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        np.maximum(norms, eps),
        out=np.zeros_like(matrix),
        where=norms > eps,
    )


def cosine_score_matrix(
    features: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Return scores with shape ``[num_queries, num_features]``."""

    feature_matrix = l2_normalize_rows(features)
    query_matrix = l2_normalize_rows(queries)
    if feature_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError("features and queries must share descriptor dimension")
    return query_matrix @ feature_matrix.T


def rank_by_query(
    features: np.ndarray,
    query: np.ndarray,
    *,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_matrix = _as_feature_matrix(features, "features")
    query_matrix = _as_feature_matrix(query, "query")
    if len(query_matrix) != 1:
        raise ValueError("rank_by_query requires exactly one query")
    if float(np.linalg.norm(query_matrix[0])) <= 1e-8:
        raise ValueError("query fingerprint is zero/invalid")
    valid = np.linalg.norm(feature_matrix, axis=1) > 1e-8
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        raise ValueError("all candidate fingerprints are zero/invalid")
    scores = cosine_score_matrix(feature_matrix, query_matrix)[0]
    local_order = np.argsort(-scores[valid_indices], kind="stable")
    order = valid_indices[local_order]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > len(order):
            raise ValueError(
                f"top_k={top_k} exceeds {len(order)} valid fingerprints"
            )
        order = order[:top_k]
    return order, scores[order]


def majority_vote_select(
    features: np.ndarray,
    queries: np.ndarray,
    *,
    vote_percentile: float = 90.0,
    top_k: int | None = None,
    top_fraction: float | None = None,
) -> SelectionResult:
    """Motive-style percentile voting with deterministic tie-breaking.

    Ties in vote count are resolved by mean cosine score and then original
    dataset order.  This is important because the paper does not specify its
    tie policy.
    """

    if not 0.0 < vote_percentile < 100.0:
        raise ValueError("vote_percentile must be in (0, 100)")
    if (top_k is None) == (top_fraction is None):
        raise ValueError("provide exactly one of top_k or top_fraction")

    feature_matrix = _as_feature_matrix(features, "features")
    query_matrix = _as_feature_matrix(queries, "queries")
    feature_valid = np.linalg.norm(feature_matrix, axis=1) > 1e-8
    query_valid = np.linalg.norm(query_matrix, axis=1) > 1e-8
    if not np.all(query_valid):
        invalid = np.flatnonzero(~query_valid).tolist()
        raise ValueError(f"zero/invalid query fingerprints at rows {invalid}")
    valid_indices = np.flatnonzero(feature_valid)
    if not len(valid_indices):
        raise ValueError("all candidate fingerprints are zero/invalid")

    score_matrix = cosine_score_matrix(feature_matrix, query_matrix)
    sample_count = score_matrix.shape[1]
    valid_count = len(valid_indices)
    if top_fraction is not None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        top_k = max(1, int(np.ceil(valid_count * top_fraction)))
    assert top_k is not None
    if top_k <= 0 or top_k > valid_count:
        raise ValueError(f"top_k must be in [1, {valid_count}] valid samples")

    # Percentile values are ambiguous under ties: with identical scores,
    # ``score > threshold`` gives every sample zero votes. Implement the
    # intended percentile budget as a stable per-query top-rank vote.
    vote_count = max(
        1,
        int(np.ceil(valid_count * (100.0 - vote_percentile) / 100.0)),
    )
    votes = np.zeros(sample_count, dtype=np.int32)
    thresholds = np.empty(len(query_matrix), dtype=np.float32)
    for query_index, query_scores in enumerate(score_matrix):
        local_order = np.argsort(
            -query_scores[valid_indices],
            kind="stable",
        )
        voted = valid_indices[local_order[:vote_count]]
        votes[voted] += 1
        thresholds[query_index] = float(query_scores[voted[-1]])
    mean_scores = np.mean(score_matrix, axis=0)
    mean_scores = np.where(feature_valid, mean_scores, -np.inf)
    original_indices = np.arange(sample_count)
    # np.lexsort uses the final key as primary.
    order = np.lexsort((original_indices, -mean_scores, -votes))
    return SelectionResult(
        indices=order[:top_k],
        votes=votes,
        mean_scores=mean_scores,
        score_matrix=score_matrix,
        thresholds=thresholds,
    )
