"""R6 independent-reference, semantic action representation primitives.

R6 closes the central leakage path in R5-lite: a query target endpoint is
never accepted as a predictor input.  The delta predictor receives only

* the query source state,
* a frozen instruction-only semantic embedding, and
* motion from a different-content training reference retrieved by frozen
  instruction/action semantics.

The prompt-motion compatibility classifier is deliberately parameter
separate from the delta predictor.  It scores an *observed* candidate motion
(a generated target delta during data filtering, or a reference delta during
reference selection).  Failed-output negatives can train this classifier but
cannot update the positive-only delta predictor and cannot suppress or rescale
text-only conditioning tokens.  A no-change/suppression head is outside this
module because it requires human intent labels that the current pilot lacks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .source_aware_repr import (
    R5_CONTENT_SPLIT_VERSION,
    DeltaWhitening,
    FactorizedR5Targets,
    Standardizer,
    _TorchModuleBase,
    _nn,
    _torch,
    audit_content_disjoint_splits,
)


R6_SCHEMA_VERSION = "motive-independent-reference-semantic-action-repr-v1"
R6_PAIRING_VERSION = "train-bank-semantic-retrieval-different-content-subject-v1"
R6_ORACLE_FAMILY_PAIRING_VERSION = (
    "oracle-family-train-bank-different-content-subject-v1"
)
R6_SEMANTIC_SCHEMA = "motive-frozen-instruction-semantic-embedding-v1"
R6_OBSERVED_ACTION_SEMANTIC_SCHEMA = (
    "motive-frozen-observed-target-action-semantic-embedding-v1"
)
R6_LABEL_ROLES = frozenset(
    {"positive_delta", "failed_outcome_compatibility"}
)
R6_MOTION_TOKEN_ROLES = ("actor_delta", "camera_delta")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_matrix(
    values: Any,
    *,
    name: str,
    rows: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D] with D >= 1")
    if rows is not None and len(matrix) != rows:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _l2_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _finite_matrix(values, name=name)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError(f"{name} contains a zero semantic vector")
    return (matrix / norms.astype(np.float32)).astype(np.float32)


@dataclass(frozen=True)
class R6SemanticProvenance:
    """Immutable provenance for one instruction-only semantic artifact."""

    encoder_id: str
    encoder_revision: str
    weights_sha256: str
    tokenizer_sha256: str
    prompt_template_version: str
    pooling: str
    embedding_dim: int
    dtype: str
    normalization: str = "l2_per_row"
    source_field: str = "instruction"
    frozen_encoder: bool = True
    target_derived_input: bool = False
    label_derived_input: bool = False
    schema_version: str = R6_SEMANTIC_SCHEMA

    def validate(
        self,
        *,
        production: bool,
        usage: str = "query_prompt",
    ) -> None:
        text_fields = (
            "encoder_id",
            "encoder_revision",
            "prompt_template_version",
            "pooling",
            "dtype",
        )
        for name in text_fields:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"semantic provenance {name} is empty")
        if usage not in {"query_prompt", "reference_action"}:
            raise ValueError(f"unsupported semantic provenance usage={usage!r}")
        expected_schema = (
            R6_SEMANTIC_SCHEMA
            if usage == "query_prompt"
            else R6_OBSERVED_ACTION_SEMANTIC_SCHEMA
        )
        if self.schema_version != expected_schema:
            raise ValueError(
                f"unsupported {usage} semantic provenance schema"
            )
        if (
            isinstance(self.embedding_dim, bool)
            or not isinstance(self.embedding_dim, int)
            or self.embedding_dim < 1
        ):
            raise ValueError("semantic embedding_dim must be positive")
        for name in ("weights_sha256", "tokenizer_sha256"):
            value = str(getattr(self, name)).lower()
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"semantic provenance {name} is not SHA-256")
        if self.normalization != "l2_per_row":
            raise ValueError("R6 semantic artifacts must use l2_per_row")
        if self.frozen_encoder is not True:
            raise ValueError("R6 semantic encoder must be frozen")
        if self.label_derived_input is not False:
            raise ValueError("label-derived semantic input is forbidden")
        if usage == "query_prompt":
            if self.source_field != "instruction":
                raise ValueError(
                    "R6 query semantic input must come only from instruction"
                )
            if self.target_derived_input is not False:
                raise ValueError(
                    "target-derived query semantic input is forbidden"
                )
        else:
            if self.source_field != "observed_target_action":
                raise ValueError(
                    "R6 reference semantic input must be observed_target_action"
                )
            if self.target_derived_input is not True:
                raise ValueError(
                    "reference-action semantics must declare target derivation"
                )
        if production and _IMMUTABLE_REVISION_RE.fullmatch(
            self.encoder_revision.lower()
        ) is None:
            raise ValueError(
                "production semantic encoder_revision must be an immutable "
                "hex commit/revision"
            )

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


@dataclass(frozen=True)
class R6ObservedActionSemanticBank:
    """Candidate-side observed-action text semantics for reference retrieval.

    This artifact is kept outside :class:`R6EndpointBatch` so target-derived
    observed-action semantics cannot be passed to the delta predictor by
    accident.
    """

    iids: tuple[str, ...]
    embeddings: np.ndarray
    input_digests: tuple[str, ...]
    provenance: R6SemanticProvenance

    @classmethod
    def create(
        cls,
        *,
        iids: Sequence[str],
        embeddings: Any,
        input_digests: Sequence[str],
        provenance: R6SemanticProvenance,
        production: bool = True,
    ) -> "R6ObservedActionSemanticBank":
        iid_values = tuple(str(value).strip() for value in iids)
        if not iid_values or any(not value for value in iid_values):
            raise ValueError("observed-action semantic iids must be non-empty")
        if len(set(iid_values)) != len(iid_values):
            raise ValueError("observed-action semantic iids must be unique")
        matrix = _l2_rows(
            _finite_matrix(
                embeddings,
                name="observed_action_semantic_embeddings",
                rows=len(iid_values),
            ),
            name="observed_action_semantic_embeddings",
        )
        digest_values = tuple(str(value).lower() for value in input_digests)
        if len(digest_values) != len(iid_values) or any(
            _SHA256_RE.fullmatch(value) is None for value in digest_values
        ):
            raise ValueError(
                "observed-action semantic input_digests must be SHA-256"
            )
        provenance.validate(
            production=production,
            usage="reference_action",
        )
        if matrix.shape[1] != provenance.embedding_dim:
            raise ValueError(
                "observed-action semantic dimension disagrees with provenance"
            )
        return cls(
            iids=iid_values,
            embeddings=matrix,
            input_digests=digest_values,
            provenance=provenance,
        )

    def digest(self) -> str:
        return _canonical_digest(
            {
                "iids": list(self.iids),
                "input_digests": list(self.input_digests),
                "provenance": asdict(self.provenance),
                "embedding_shape": list(self.embeddings.shape),
                "embedding_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        self.embeddings.astype("<f4", copy=False)
                    ).tobytes()
                ).hexdigest(),
            }
        )

    def index_by_iid(self) -> dict[str, int]:
        return {iid: index for index, iid in enumerate(self.iids)}


@dataclass(frozen=True)
class R6EndpointBatch:
    """Query endpoints and labels; targets are never predictor inputs."""

    iids: tuple[str, ...]
    source_actor: np.ndarray
    source_camera: np.ndarray
    target_actor: np.ndarray
    target_camera: np.ndarray
    semantic_embeddings: np.ndarray
    semantic_input_digests: tuple[str, ...]
    splits: tuple[str, ...]
    content_group_ids: tuple[str, ...]
    subject_cluster_ids: tuple[str, ...]
    action_families: tuple[str, ...]
    label_roles: tuple[str, ...]
    compatibility_targets: np.ndarray
    split_versions: tuple[str, ...]
    semantic_provenance: R6SemanticProvenance
    perceptual_hashes: tuple[str, ...] | None = None
    maximum_cross_split_hamming_fraction: float | None = None

    @classmethod
    def create(
        cls,
        *,
        iids: Sequence[str],
        source_actor: Any,
        source_camera: Any,
        target_actor: Any,
        target_camera: Any,
        semantic_embeddings: Any,
        semantic_input_digests: Sequence[str],
        splits: Sequence[str],
        content_group_ids: Sequence[str],
        subject_cluster_ids: Sequence[str],
        action_families: Sequence[str],
        label_roles: Sequence[str],
        compatibility_targets: Sequence[float | int | bool],
        split_versions: Sequence[str],
        semantic_provenance: R6SemanticProvenance,
        perceptual_hashes: Sequence[str] | None = None,
        require_visual_clusters: bool = True,
        maximum_cross_split_hamming_fraction: float = 0.10,
    ) -> "R6EndpointBatch":
        iid_values = tuple(str(value).strip() for value in iids)
        if not iid_values or any(not value for value in iid_values):
            raise ValueError("R6 iids must be non-empty")
        if len(set(iid_values)) != len(iid_values):
            raise ValueError("R6 iids must be unique")
        rows = len(iid_values)
        matrices = {
            "source_actor": _finite_matrix(
                source_actor, name="source_actor", rows=rows
            ),
            "source_camera": _finite_matrix(
                source_camera, name="source_camera", rows=rows
            ),
            "target_actor": _finite_matrix(
                target_actor, name="target_actor", rows=rows
            ),
            "target_camera": _finite_matrix(
                target_camera, name="target_camera", rows=rows
            ),
            "semantic_embeddings": _l2_rows(
                _finite_matrix(
                    semantic_embeddings,
                    name="semantic_embeddings",
                    rows=rows,
                ),
                name="semantic_embeddings",
            ),
        }
        if matrices["source_actor"].shape != matrices["target_actor"].shape:
            raise ValueError("source/target actor dimensions differ")
        if matrices["source_camera"].shape != matrices["target_camera"].shape:
            raise ValueError("source/target camera dimensions differ")
        semantic_provenance.validate(production=require_visual_clusters)
        if (
            matrices["semantic_embeddings"].shape[1]
            != semantic_provenance.embedding_dim
        ):
            raise ValueError(
                "semantic embedding dimension disagrees with provenance"
            )
        semantic_digest_values = tuple(
            str(value).lower() for value in semantic_input_digests
        )
        if len(semantic_digest_values) != rows or any(
            _SHA256_RE.fullmatch(value) is None
            for value in semantic_digest_values
        ):
            raise ValueError(
                "semantic_input_digests must contain one SHA-256 per IID"
            )

        metadata = {
            "splits": tuple(str(value) for value in splits),
            "content_group_ids": tuple(
                str(value).strip() for value in content_group_ids
            ),
            "subject_cluster_ids": tuple(
                str(value).strip() for value in subject_cluster_ids
            ),
            "action_families": tuple(
                str(value).strip().lower() for value in action_families
            ),
            "label_roles": tuple(str(value).strip() for value in label_roles),
            "split_versions": tuple(str(value) for value in split_versions),
        }
        for name, values in metadata.items():
            if len(values) != rows:
                raise ValueError(f"{name} length mismatch")
            if any(not value for value in values):
                raise ValueError(f"{name} contains an empty value")
        invalid_roles = sorted(set(metadata["label_roles"]) - R6_LABEL_ROLES)
        if invalid_roles:
            raise ValueError(f"invalid R6 label roles: {invalid_roles}")
        compatibility = np.asarray(compatibility_targets, dtype=np.float32)
        if compatibility.shape != (rows,) or not np.isfinite(compatibility).all():
            raise ValueError("compatibility_targets must have finite shape [N]")
        if bool(((compatibility != 0.0) & (compatibility != 1.0)).any()):
            raise ValueError("compatibility_targets must be binary")
        positive = np.asarray(metadata["label_roles"]) == "positive_delta"
        if bool((compatibility[positive] != 1.0).any()):
            raise ValueError("positive_delta rows must be prompt-motion compatible")
        hash_values = (
            None
            if perceptual_hashes is None
            else tuple(str(value).strip().lower() for value in perceptual_hashes)
        )
        if hash_values is not None and len(hash_values) != rows:
            raise ValueError("perceptual_hashes length mismatch")

        audit_content_disjoint_splits(
            splits=metadata["splits"],
            content_group_ids=metadata["content_group_ids"],
            split_versions=metadata["split_versions"],
            perceptual_hashes=hash_values,
            maximum_cross_split_hamming_fraction=(
                maximum_cross_split_hamming_fraction
            ),
            require_visual_clusters=require_visual_clusters,
        )
        # A content cluster is not necessarily a subject cluster.  Audit both.
        audit_content_disjoint_splits(
            splits=metadata["splits"],
            content_group_ids=metadata["subject_cluster_ids"],
            split_versions=metadata["split_versions"],
            perceptual_hashes=hash_values,
            maximum_cross_split_hamming_fraction=(
                maximum_cross_split_hamming_fraction
            ),
            require_visual_clusters=require_visual_clusters,
        )
        return cls(
            iids=iid_values,
            source_actor=matrices["source_actor"],
            source_camera=matrices["source_camera"],
            target_actor=matrices["target_actor"],
            target_camera=matrices["target_camera"],
            semantic_embeddings=matrices["semantic_embeddings"],
            semantic_input_digests=semantic_digest_values,
            splits=metadata["splits"],
            content_group_ids=metadata["content_group_ids"],
            subject_cluster_ids=metadata["subject_cluster_ids"],
            action_families=metadata["action_families"],
            label_roles=metadata["label_roles"],
            compatibility_targets=compatibility,
            split_versions=metadata["split_versions"],
            semantic_provenance=semantic_provenance,
            perceptual_hashes=hash_values,
            maximum_cross_split_hamming_fraction=(
                float(maximum_cross_split_hamming_fraction)
                if hash_values is not None
                else None
            ),
        )

    def indices(self, split: str) -> np.ndarray:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split {split!r}")
        return np.flatnonzero(np.asarray(self.splits) == split).astype(np.int64)

    def positive_indices(self, split: str | None = None) -> np.ndarray:
        mask = np.asarray(self.label_roles) == "positive_delta"
        if split is not None:
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"invalid split {split!r}")
            mask &= np.asarray(self.splits) == split
        return np.flatnonzero(mask).astype(np.int64)


@dataclass(frozen=True)
class IndependentReferencePairs:
    """Immutable query-to-train-reference table."""

    query_indices: np.ndarray
    reference_indices: np.ndarray
    reference_ranks: np.ndarray
    pair_scores: np.ndarray
    candidate_counts: np.ndarray
    pair_input_digests: tuple[str, ...]
    data_seed: int
    references_per_query: int
    unpaired_iids: tuple[str, ...]
    undercovered_iids: tuple[str, ...]
    query_iid_digest: str
    reference_bank_iid_digest: str
    selector_name: str
    selector_input_fields: tuple[str, ...]
    similarity_threshold: float | None
    threshold_quantile: float | None
    threshold_fit_iid_digest: str | None
    threshold_fit_digest: str | None
    query_semantic_provenance_digest: str | None
    reference_semantic_bank_digest: str | None
    gate_eligible: bool
    pairing_version: str = R6_PAIRING_VERSION

    def validate(self, batch: R6EndpointBatch) -> None:
        query = np.asarray(self.query_indices, dtype=np.int64)
        reference = np.asarray(self.reference_indices, dtype=np.int64)
        ranks = np.asarray(self.reference_ranks, dtype=np.int64)
        scores = np.asarray(self.pair_scores, dtype=np.float64)
        candidate_counts = np.asarray(self.candidate_counts, dtype=np.int64)
        if (
            query.ndim != 1
            or reference.shape != query.shape
            or ranks.shape != query.shape
            or scores.shape != query.shape
            or candidate_counts.shape != query.shape
        ):
            raise ValueError("R6 reference table arrays must share shape [P]")
        if self.pairing_version not in {
            R6_PAIRING_VERSION,
            R6_ORACLE_FAMILY_PAIRING_VERSION,
        }:
            raise ValueError("unsupported R6 pairing version")
        if not np.isfinite(scores).all():
            raise ValueError("R6 reference pair scores must be finite")
        if len(self.pair_input_digests) != len(query) or any(
            _SHA256_RE.fullmatch(str(value)) is None
            for value in self.pair_input_digests
        ):
            raise ValueError("R6 pair input digests are invalid")
        if bool((candidate_counts < 1).any()):
            raise ValueError("R6 pair candidate_counts must be positive")
        if (
            isinstance(self.references_per_query, bool)
            or self.references_per_query < 1
        ):
            raise ValueError("references_per_query must be positive")
        if len(query):
            if int(query.min()) < 0 or int(reference.min()) < 0:
                raise ValueError("R6 pair index is negative")
            if int(query.max()) >= len(batch.iids) or int(reference.max()) >= len(
                batch.iids
            ):
                raise ValueError("R6 pair index is out of range")
        seen: set[tuple[int, int]] = set()
        for position, (q_raw, r_raw, rank_raw) in enumerate(
            zip(query, reference, ranks)
        ):
            q = int(q_raw)
            r = int(r_raw)
            rank = int(rank_raw)
            if (q, r) in seen:
                raise ValueError("duplicate R6 query/reference pair")
            seen.add((q, r))
            if batch.iids[q] == batch.iids[r] or q == r:
                raise ValueError("query cannot use itself as a reference")
            if batch.splits[r] != "train":
                raise ValueError("R6 reference bank must contain train rows only")
            if batch.label_roles[r] != "positive_delta":
                raise ValueError("R6 references must be positive_delta rows")
            if batch.content_group_ids[q] == batch.content_group_ids[r]:
                raise ValueError("R6 reference reuses query content group")
            if batch.subject_cluster_ids[q] == batch.subject_cluster_ids[r]:
                raise ValueError("R6 reference reuses query subject cluster")
            if rank < 0 or rank >= self.references_per_query:
                raise ValueError("R6 reference rank is out of range")
            if int(candidate_counts[position]) <= rank:
                raise ValueError("R6 candidate count is inconsistent with rank")
            if self.pairing_version == R6_ORACLE_FAMILY_PAIRING_VERSION:
                if batch.action_families[q] != batch.action_families[r]:
                    raise ValueError("oracle R6 reference action family mismatch")
        expected_query_digest = _canonical_digest(list(batch.iids))
        train_reference_iids = sorted(
            batch.iids[int(index)] for index in batch.positive_indices("train")
        )
        if self.query_iid_digest != expected_query_digest:
            raise ValueError("R6 query IID digest mismatch")
        if self.reference_bank_iid_digest != _canonical_digest(
            train_reference_iids
        ):
            raise ValueError("R6 reference-bank IID digest mismatch")
        if self.pairing_version == R6_PAIRING_VERSION:
            expected_fields = (
                "query.instruction_semantic_embedding",
                "candidate.observed_target_action_semantic_embedding",
                "query.iid",
                "query.content_group_id",
                "query.subject_cluster_id",
                "candidate.iid",
                "candidate.content_group_id",
                "candidate.subject_cluster_id",
                "candidate.split",
                "candidate.label_role",
            )
            if self.selector_name != "semantic_cosine_train_bank":
                raise ValueError("invalid gate-eligible R6 selector name")
            if self.selector_input_fields != expected_fields:
                raise ValueError("R6 selector input field contract changed")
            if self.gate_eligible is not True:
                raise ValueError("semantic R6 selector must be gate eligible")
            if (
                self.similarity_threshold is None
                or not math.isfinite(float(self.similarity_threshold))
                or self.threshold_quantile is None
                or not 0.0 < float(self.threshold_quantile) < 1.0
            ):
                raise ValueError("R6 semantic selector threshold is invalid")
            if bool((scores < float(self.similarity_threshold) - 1e-7).any()):
                raise ValueError("R6 semantic pair is below fitted threshold")
            for name in (
                "threshold_fit_iid_digest",
                "threshold_fit_digest",
                "query_semantic_provenance_digest",
                "reference_semantic_bank_digest",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                    raise ValueError(f"R6 semantic selector {name} is invalid")
        else:
            if self.selector_name != "oracle_action_family":
                raise ValueError("oracle R6 selector name is invalid")
            if self.gate_eligible is not False:
                raise ValueError("oracle-family selector cannot be gate eligible")

    def coverage(
        self,
        batch: R6EndpointBatch,
        *,
        split: str = "test",
        positive_only: bool = True,
    ) -> dict[str, Any]:
        eligible = (
            batch.positive_indices(split)
            if positive_only
            else batch.indices(split)
        )
        counts = {
            int(index): 0
            for index in eligible
        }
        for raw_query in np.asarray(self.query_indices, dtype=np.int64):
            query = int(raw_query)
            if query in counts:
                counts[query] += 1
        full = sum(
            count >= self.references_per_query for count in counts.values()
        )
        any_reference = sum(count >= 1 for count in counts.values())
        total = len(counts)
        return {
            "split": split,
            "positive_only": bool(positive_only),
            "eligible_queries": total,
            "queries_with_any_reference": any_reference,
            "queries_with_full_reference_count": full,
            "any_reference_fraction": (
                float(any_reference) / float(total) if total else 0.0
            ),
            "full_reference_fraction": (
                float(full) / float(total) if total else 0.0
            ),
        }

    def reference_load(self, batch: R6EndpointBatch) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for raw_reference in np.asarray(
            self.reference_indices,
            dtype=np.int64,
        ):
            iid = batch.iids[int(raw_reference)]
            counts[iid] = counts.get(iid, 0) + 1
        total = int(sum(counts.values()))
        maximum = max(counts.values(), default=0)
        probabilities = (
            np.asarray(list(counts.values()), dtype=np.float64) / float(total)
            if total
            else np.zeros(0, dtype=np.float64)
        )
        entropy = (
            float(-np.sum(probabilities * np.log(probabilities)))
            if len(probabilities)
            else 0.0
        )
        return {
            "pair_count": total,
            "unique_reference_count": len(counts),
            "maximum_reference_load": int(maximum),
            "maximum_reference_load_fraction": (
                float(maximum) / float(total) if total else 0.0
            ),
            "normalized_load_entropy": (
                entropy / math.log(len(counts))
                if len(counts) > 1
                else 0.0
            ),
            "per_reference_iid": dict(sorted(counts.items())),
        }

    def digest(self) -> str:
        return _canonical_digest(
            {
                "query_indices": np.asarray(
                    self.query_indices, dtype=np.int64
                ).tolist(),
                "reference_indices": np.asarray(
                    self.reference_indices, dtype=np.int64
                ).tolist(),
                "reference_ranks": np.asarray(
                    self.reference_ranks, dtype=np.int64
                ).tolist(),
                "pair_scores": np.asarray(
                    self.pair_scores, dtype=np.float64
                ).tolist(),
                "candidate_counts": np.asarray(
                    self.candidate_counts, dtype=np.int64
                ).tolist(),
                "pair_input_digests": list(self.pair_input_digests),
                "data_seed": int(self.data_seed),
                "references_per_query": int(self.references_per_query),
                "unpaired_iids": list(self.unpaired_iids),
                "undercovered_iids": list(self.undercovered_iids),
                "query_iid_digest": self.query_iid_digest,
                "reference_bank_iid_digest": self.reference_bank_iid_digest,
                "selector_name": self.selector_name,
                "selector_input_fields": list(self.selector_input_fields),
                "similarity_threshold": self.similarity_threshold,
                "threshold_quantile": self.threshold_quantile,
                "threshold_fit_iid_digest": self.threshold_fit_iid_digest,
                "threshold_fit_digest": self.threshold_fit_digest,
                "query_semantic_provenance_digest": (
                    self.query_semantic_provenance_digest
                ),
                "reference_semantic_bank_digest": (
                    self.reference_semantic_bank_digest
                ),
                "gate_eligible": self.gate_eligible,
                "pairing_version": self.pairing_version,
            }
        )


def _validate_reference_request(references_per_query: int) -> None:
    if (
        isinstance(references_per_query, bool)
        or not isinstance(references_per_query, int)
        or references_per_query < 1
    ):
        raise ValueError("references_per_query must be a positive integer")


def _query_indices(
    batch: R6EndpointBatch,
    *,
    include_failed_outcomes: bool,
) -> list[int]:
    return [
        index
        for index, role in enumerate(batch.label_roles)
        if role == "positive_delta" or include_failed_outcomes
    ]


def _semantic_spaces_match(
    query: R6SemanticProvenance,
    reference: R6SemanticProvenance,
) -> bool:
    names = (
        "encoder_id",
        "encoder_revision",
        "weights_sha256",
        "tokenizer_sha256",
        "pooling",
        "embedding_dim",
        "normalization",
    )
    return all(getattr(query, name) == getattr(reference, name) for name in names)


def build_semantic_train_bank_reference_pairs(
    batch: R6EndpointBatch,
    observed_action_bank: R6ObservedActionSemanticBank,
    *,
    data_seed: int,
    references_per_query: int = 3,
    threshold_quantile: float = 0.10,
    include_failed_outcomes: bool = True,
    require_complete: bool = False,
) -> IndependentReferencePairs:
    """Retrieve train references without query-target or family leakage.

    Similarity is between the query's instruction-only semantic embedding and
    each candidate train positive's observed-target-action text embedding.
    The acceptance threshold is fitted once from train-positive self
    prompt/action similarities, then frozen for validation and test.
    """

    _validate_reference_request(references_per_query)
    if not 0.0 < float(threshold_quantile) < 1.0:
        raise ValueError("threshold_quantile must be in (0,1)")
    observed_action_bank.provenance.validate(
        production=(
            len(set(batch.split_versions)) == 1
            and batch.split_versions[0] == R5_CONTENT_SPLIT_VERSION
        ),
        usage="reference_action",
    )
    if not _semantic_spaces_match(
        batch.semantic_provenance,
        observed_action_bank.provenance,
    ):
        raise ValueError(
            "query-prompt and observed-action embeddings do not share one "
            "frozen semantic space"
        )
    bank_by_iid = observed_action_bank.index_by_iid()
    train_bank = batch.positive_indices("train")
    missing = [
        batch.iids[int(index)]
        for index in train_bank
        if batch.iids[int(index)] not in bank_by_iid
    ]
    if missing:
        raise ValueError(
            "observed-action semantic bank misses train positives: "
            f"{missing[:5]}"
        )
    fit_records: list[dict[str, Any]] = []
    self_scores: list[float] = []
    for raw_index in train_bank:
        index = int(raw_index)
        bank_index = bank_by_iid[batch.iids[index]]
        score = float(
            np.dot(
                batch.semantic_embeddings[index].astype(np.float64),
                observed_action_bank.embeddings[bank_index].astype(np.float64),
            )
        )
        self_scores.append(score)
        fit_records.append(
            {
                "iid": batch.iids[index],
                "query_instruction_digest": (
                    batch.semantic_input_digests[index]
                ),
                "observed_action_digest": (
                    observed_action_bank.input_digests[bank_index]
                ),
                "cosine": score,
            }
        )
    if len(self_scores) < 2:
        raise ValueError(
            "semantic reference threshold needs two train positives"
        )
    threshold = float(np.quantile(self_scores, float(threshold_quantile)))
    if not math.isfinite(threshold):
        raise ValueError("semantic reference threshold is non-finite")

    queries = _query_indices(
        batch,
        include_failed_outcomes=include_failed_outcomes,
    )
    query_indices: list[int] = []
    reference_indices: list[int] = []
    reference_ranks: list[int] = []
    pair_scores: list[float] = []
    candidate_counts: list[int] = []
    pair_input_digests: list[str] = []
    unpaired: list[str] = []
    undercovered: list[str] = []
    for query in queries:
        candidates: list[tuple[int, float, bytes]] = []
        for raw_reference in train_bank:
            reference = int(raw_reference)
            if (
                reference == query
                or batch.iids[reference] == batch.iids[query]
                or batch.content_group_ids[reference]
                == batch.content_group_ids[query]
                or batch.subject_cluster_ids[reference]
                == batch.subject_cluster_ids[query]
            ):
                continue
            bank_index = bank_by_iid[batch.iids[reference]]
            score = float(
                np.dot(
                    batch.semantic_embeddings[query].astype(np.float64),
                    observed_action_bank.embeddings[bank_index].astype(
                        np.float64
                    ),
                )
            )
            if score + 1e-7 < threshold:
                continue
            tie_break = hashlib.sha256(
                (
                    f"{int(data_seed)}\0{batch.iids[query]}\0"
                    f"{batch.iids[reference]}"
                ).encode("utf-8")
            ).digest()
            candidates.append((reference, score, tie_break))
        candidates.sort(
            key=lambda item: (
                -item[1],
                item[2],
                batch.iids[item[0]],
            )
        )
        selected = candidates[:references_per_query]
        if not selected:
            unpaired.append(batch.iids[query])
        elif len(selected) < references_per_query:
            undercovered.append(batch.iids[query])
        for rank, (reference, score, _) in enumerate(selected):
            bank_index = bank_by_iid[batch.iids[reference]]
            query_indices.append(query)
            reference_indices.append(reference)
            reference_ranks.append(rank)
            pair_scores.append(score)
            candidate_counts.append(len(candidates))
            pair_input_digests.append(
                _canonical_digest(
                    {
                        "selector": "semantic_cosine_train_bank",
                        "query_iid": batch.iids[query],
                        "query_instruction_digest": (
                            batch.semantic_input_digests[query]
                        ),
                        "candidate_iid": batch.iids[reference],
                        "candidate_observed_action_digest": (
                            observed_action_bank.input_digests[bank_index]
                        ),
                        "query_content_group_id": (
                            batch.content_group_ids[query]
                        ),
                        "candidate_content_group_id": (
                            batch.content_group_ids[reference]
                        ),
                        "query_subject_cluster_id": (
                            batch.subject_cluster_ids[query]
                        ),
                        "candidate_subject_cluster_id": (
                            batch.subject_cluster_ids[reference]
                        ),
                    }
                )
            )
    if require_complete and (unpaired or undercovered):
        raise ValueError(
            "R6 semantic reference coverage is incomplete: "
            f"unpaired={len(unpaired)} undercovered={len(undercovered)}"
        )
    table = IndependentReferencePairs(
        query_indices=np.asarray(query_indices, dtype=np.int64),
        reference_indices=np.asarray(reference_indices, dtype=np.int64),
        reference_ranks=np.asarray(reference_ranks, dtype=np.int64),
        pair_scores=np.asarray(pair_scores, dtype=np.float64),
        candidate_counts=np.asarray(candidate_counts, dtype=np.int64),
        pair_input_digests=tuple(pair_input_digests),
        data_seed=int(data_seed),
        references_per_query=int(references_per_query),
        unpaired_iids=tuple(sorted(unpaired)),
        undercovered_iids=tuple(sorted(undercovered)),
        query_iid_digest=_canonical_digest(list(batch.iids)),
        reference_bank_iid_digest=_canonical_digest(
            sorted(batch.iids[int(index)] for index in train_bank)
        ),
        selector_name="semantic_cosine_train_bank",
        selector_input_fields=(
            "query.instruction_semantic_embedding",
            "candidate.observed_target_action_semantic_embedding",
            "query.iid",
            "query.content_group_id",
            "query.subject_cluster_id",
            "candidate.iid",
            "candidate.content_group_id",
            "candidate.subject_cluster_id",
            "candidate.split",
            "candidate.label_role",
        ),
        similarity_threshold=threshold,
        threshold_quantile=float(threshold_quantile),
        threshold_fit_iid_digest=_canonical_digest(
            sorted(record["iid"] for record in fit_records)
        ),
        threshold_fit_digest=_canonical_digest(fit_records),
        query_semantic_provenance_digest=(
            batch.semantic_provenance.digest()
        ),
        reference_semantic_bank_digest=observed_action_bank.digest(),
        gate_eligible=True,
    )
    table.validate(batch)
    return table


def build_oracle_family_reference_pairs(
    batch: R6EndpointBatch,
    *,
    data_seed: int,
    references_per_query: int = 3,
    include_failed_outcomes: bool = True,
    require_complete: bool = False,
) -> IndependentReferencePairs:
    """Build an optimistic action-family oracle for diagnostics only.

    Family labels are too sparse/noisy to be a primary selector.  This table
    is structurally marked ``gate_eligible=false``.
    """

    _validate_reference_request(references_per_query)
    train_bank = batch.positive_indices("train")
    queries = _query_indices(
        batch,
        include_failed_outcomes=include_failed_outcomes,
    )
    query_indices: list[int] = []
    reference_indices: list[int] = []
    reference_ranks: list[int] = []
    pair_scores: list[float] = []
    candidate_counts: list[int] = []
    pair_input_digests: list[str] = []
    unpaired: list[str] = []
    undercovered: list[str] = []
    for query in queries:
        candidates = [
            int(reference)
            for reference in train_bank
            if int(reference) != query
            and batch.iids[int(reference)] != batch.iids[query]
            and batch.action_families[int(reference)]
            == batch.action_families[query]
            and batch.content_group_ids[int(reference)]
            != batch.content_group_ids[query]
            and batch.subject_cluster_ids[int(reference)]
            != batch.subject_cluster_ids[query]
        ]
        candidates.sort(
            key=lambda reference: (
                hashlib.sha256(
                    (
                        f"{int(data_seed)}\0{batch.iids[query]}\0"
                        f"{batch.iids[reference]}"
                    ).encode("utf-8")
                ).digest(),
                batch.iids[reference],
            )
        )
        selected = candidates[:references_per_query]
        if not selected:
            unpaired.append(batch.iids[query])
        elif len(selected) < references_per_query:
            undercovered.append(batch.iids[query])
        for rank, reference in enumerate(selected):
            query_indices.append(query)
            reference_indices.append(reference)
            reference_ranks.append(rank)
            pair_scores.append(1.0)
            candidate_counts.append(len(candidates))
            pair_input_digests.append(
                _canonical_digest(
                    {
                        "selector": "oracle_action_family",
                        "query_iid": batch.iids[query],
                        "candidate_iid": batch.iids[reference],
                        "action_family": batch.action_families[query],
                    }
                )
            )
    if require_complete and (unpaired or undercovered):
        raise ValueError(
            "R6 reference coverage is incomplete: "
            f"unpaired={len(unpaired)} undercovered={len(undercovered)}"
        )
    table = IndependentReferencePairs(
        query_indices=np.asarray(query_indices, dtype=np.int64),
        reference_indices=np.asarray(reference_indices, dtype=np.int64),
        reference_ranks=np.asarray(reference_ranks, dtype=np.int64),
        pair_scores=np.asarray(pair_scores, dtype=np.float64),
        candidate_counts=np.asarray(candidate_counts, dtype=np.int64),
        pair_input_digests=tuple(pair_input_digests),
        data_seed=int(data_seed),
        references_per_query=int(references_per_query),
        unpaired_iids=tuple(sorted(unpaired)),
        undercovered_iids=tuple(sorted(undercovered)),
        query_iid_digest=_canonical_digest(list(batch.iids)),
        reference_bank_iid_digest=_canonical_digest(
            sorted(batch.iids[int(index)] for index in train_bank)
        ),
        selector_name="oracle_action_family",
        selector_input_fields=(
            "query.action_family",
            "candidate.action_family",
            "query.iid",
            "query.content_group_id",
            "query.subject_cluster_id",
            "candidate.iid",
            "candidate.content_group_id",
            "candidate.subject_cluster_id",
            "candidate.split",
            "candidate.label_role",
        ),
        similarity_threshold=None,
        threshold_quantile=None,
        threshold_fit_iid_digest=None,
        threshold_fit_digest=None,
        query_semantic_provenance_digest=None,
        reference_semantic_bank_digest=None,
        gate_eligible=False,
        pairing_version=R6_ORACLE_FAMILY_PAIRING_VERSION,
    )
    table.validate(batch)
    return table


@dataclass(frozen=True)
class R6MotionFeatures:
    actor_direction: np.ndarray
    actor_log_magnitude: np.ndarray
    camera_direction: np.ndarray
    camera_log_magnitude: np.ndarray

    @property
    def actor_input(self) -> np.ndarray:
        return np.concatenate(
            (
                self.actor_direction,
                self.actor_log_magnitude.reshape(-1, 1),
            ),
            axis=1,
        ).astype(np.float32)

    @property
    def camera_input(self) -> np.ndarray:
        return np.concatenate(
            (
                self.camera_direction,
                self.camera_log_magnitude.reshape(-1, 1),
            ),
            axis=1,
        ).astype(np.float32)

    def as_targets(self) -> FactorizedR5Targets:
        return FactorizedR5Targets(
            actor_direction=self.actor_direction,
            actor_log_magnitude=self.actor_log_magnitude,
            camera_direction=self.camera_direction,
            camera_log_magnitude=self.camera_log_magnitude,
        )


@dataclass(frozen=True)
class TrainOnlySemanticProjection:
    """PCA/SVD projection fitted only on train instruction rows."""

    mean: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    input_dim: int
    output_dim: int
    stable_rank: int
    l2_normalize_output: bool = True

    @classmethod
    def fit(
        cls,
        values: Any,
        *,
        output_dim: int = 64,
        minimum_relative_variance: float = 1e-6,
    ) -> "TrainOnlySemanticProjection":
        matrix = _finite_matrix(values, name="semantic projection values")
        if len(matrix) < 2:
            raise ValueError(
                "semantic projection requires at least two train rows"
            )
        if (
            isinstance(output_dim, bool)
            or not isinstance(output_dim, int)
            or output_dim < 1
        ):
            raise ValueError("semantic projection output_dim must be positive")
        mean = np.mean(matrix.astype(np.float64), axis=0)
        centered = matrix.astype(np.float64) - mean
        _, singular_values, right = np.linalg.svd(
            centered,
            full_matrices=False,
        )
        variance = singular_values**2 / max(len(matrix) - 1, 1)
        maximum = float(variance[0]) if len(variance) else 0.0
        if maximum <= 1e-12:
            raise ValueError("semantic train rows have no stable variance")
        stable_rank = int(
            np.count_nonzero(
                variance >= maximum * float(minimum_relative_variance)
            )
        )
        rank = max(1, min(output_dim, stable_rank, right.shape[0]))
        return cls(
            mean=tuple(float(value) for value in mean),
            components=tuple(
                tuple(float(value) for value in row)
                for row in right[:rank].astype(np.float32)
            ),
            input_dim=int(matrix.shape[1]),
            output_dim=int(output_dim),
            stable_rank=int(rank),
        )

    def transform(self, values: Any) -> np.ndarray:
        matrix = _finite_matrix(values, name="semantic projection input")
        if matrix.shape[1] != self.input_dim:
            raise ValueError("semantic projection input dimension changed")
        mean = np.asarray(self.mean, dtype=np.float32)
        components = np.asarray(self.components, dtype=np.float32)
        projected = (matrix - mean) @ components.T
        if projected.shape[1] < self.output_dim:
            projected = np.pad(
                projected,
                ((0, 0), (0, self.output_dim - projected.shape[1])),
            )
        projected = projected.astype(np.float32)
        if self.l2_normalize_output:
            norms = np.linalg.norm(
                projected.astype(np.float64),
                axis=1,
                keepdims=True,
            )
            projected = np.divide(
                projected,
                np.maximum(norms.astype(np.float32), 1e-12),
                out=np.zeros_like(projected),
                where=norms > 1e-12,
            )
        return projected.astype(np.float32)


@dataclass(frozen=True)
class R6FeatureTransform:
    """Preprocessing with an explicit train-only fit boundary."""

    actor_source: Standardizer
    camera_source: Standardizer
    actor_delta: DeltaWhitening
    camera_delta: DeltaWhitening
    semantic_projection: TrainOnlySemanticProjection
    fit_input_train_iid_digest: str
    fit_delta_positive_train_iid_digest: str
    semantic_provenance_digest: str
    schema_version: str = R6_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        batch: R6EndpointBatch,
        *,
        condition_dim: int = 16,
        semantic_condition_dim: int = 64,
    ) -> "R6FeatureTransform":
        train = batch.indices("train")
        positive_train = batch.positive_indices("train")
        if len(train) < 2:
            raise ValueError("R6 input transforms require two train rows")
        if len(positive_train) < 2:
            raise ValueError(
                "R6 delta transform requires at least two positive train rows"
            )
        return cls(
            actor_source=Standardizer.fit(batch.source_actor[train]),
            camera_source=Standardizer.fit(batch.source_camera[train]),
            actor_delta=DeltaWhitening.fit(
                batch.target_actor[positive_train]
                - batch.source_actor[positive_train],
                output_dim=int(condition_dim),
            ),
            camera_delta=DeltaWhitening.fit(
                batch.target_camera[positive_train]
                - batch.source_camera[positive_train],
                output_dim=int(condition_dim),
            ),
            semantic_projection=TrainOnlySemanticProjection.fit(
                batch.semantic_embeddings[train],
                output_dim=int(semantic_condition_dim),
            ),
            fit_input_train_iid_digest=_canonical_digest(
                sorted(batch.iids[int(index)] for index in train)
            ),
            fit_delta_positive_train_iid_digest=_canonical_digest(
                sorted(batch.iids[int(index)] for index in positive_train)
            ),
            semantic_provenance_digest=batch.semantic_provenance.digest(),
        )

    def source_inputs(
        self,
        batch: R6EndpointBatch,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            batch.semantic_embeddings.shape[1]
            != self.semantic_projection.input_dim
        ):
            raise ValueError("R6 semantic dimension changed after transform fit")
        if (
            batch.semantic_provenance.digest()
            != self.semantic_provenance_digest
        ):
            raise ValueError("R6 semantic provenance changed after transform fit")
        return (
            self.actor_source.transform(batch.source_actor),
            self.camera_source.transform(batch.source_camera),
            self.semantic_projection.transform(
                batch.semantic_embeddings,
            ),
        )

    def motion_from_deltas(
        self,
        *,
        actor_delta: Any,
        camera_delta: Any,
    ) -> R6MotionFeatures:
        actor_direction, actor_magnitude = self.actor_delta.transform(
            _finite_matrix(actor_delta, name="actor_delta")
        )
        camera_direction, camera_magnitude = self.camera_delta.transform(
            _finite_matrix(camera_delta, name="camera_delta")
        )
        if len(actor_direction) != len(camera_direction):
            raise ValueError("actor/camera motion row count differs")
        return R6MotionFeatures(
            actor_direction=actor_direction,
            actor_log_magnitude=actor_magnitude,
            camera_direction=camera_direction,
            camera_log_magnitude=camera_magnitude,
        )

    def observed_motion(self, batch: R6EndpointBatch) -> R6MotionFeatures:
        """Motion used as teacher/compatibility evidence, never predictor input."""

        return self.motion_from_deltas(
            actor_delta=batch.target_actor - batch.source_actor,
            camera_delta=batch.target_camera - batch.source_camera,
        )

    def reference_motion(
        self,
        batch: R6EndpointBatch,
        pairs: IndependentReferencePairs,
    ) -> R6MotionFeatures:
        pairs.validate(batch)
        reference = np.asarray(pairs.reference_indices, dtype=np.int64)
        return self.motion_from_deltas(
            actor_delta=(
                batch.target_actor[reference] - batch.source_actor[reference]
            ),
            camera_delta=(
                batch.target_camera[reference] - batch.source_camera[reference]
            ),
        )

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


class _R6PredictorBranch(_TorchModuleBase):
    """One positive-only delta branch."""

    def __init__(
        self,
        *,
        source_dim: int,
        semantic_dim: int,
        motion_dim: int,
        condition_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        nn = _nn()

        def tower(width: int) -> Any:
            return nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )

        self.source = tower(source_dim)
        self.semantic = tower(semantic_dim)
        self.reference_motion = tower(motion_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.direction = nn.Linear(hidden_dim, condition_dim)
        self.log_magnitude = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        source: Any,
        semantic: Any,
        reference_motion: Any,
        semantic_mask: Any,
        reference_mask: Any,
    ) -> tuple[Any, Any]:
        torch = _torch()
        source_hidden = self.source(source)
        semantic_hidden = self.semantic(semantic) * semantic_mask
        reference_hidden = self.reference_motion(reference_motion) * reference_mask
        hidden = self.fusion(
            torch.cat(
                (
                    source_hidden,
                    semantic_hidden,
                    reference_hidden,
                    semantic_mask,
                    reference_mask,
                ),
                dim=-1,
            )
        )
        direction = torch.nn.functional.normalize(
            self.direction(hidden).float(),
            dim=-1,
        ).to(hidden.dtype)
        magnitude = torch.nn.functional.softplus(
            self.log_magnitude(hidden).float()
        ).to(hidden.dtype)
        return direction, magnitude


class _R6CompatibilityHead(_TorchModuleBase):
    """Parameter-separate prompt/observed-motion compatibility head."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        actor_motion_dim: int,
        camera_motion_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        nn = _nn()

        def tower(width: int) -> Any:
            return nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            )

        self.semantic = tower(semantic_dim)
        self.actor_motion = tower(actor_motion_dim)
        self.camera_motion = tower(camera_motion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        semantic: Any,
        actor_motion: Any,
        camera_motion: Any,
    ) -> Any:
        torch = _torch()
        hidden = torch.cat(
            (
                self.semantic(semantic),
                self.actor_motion(actor_motion),
                self.camera_motion(camera_motion),
            ),
            dim=-1,
        )
        return self.classifier(hidden)


class SourceAwareFactorizedR6(_TorchModuleBase):
    """R6 predictor plus a structurally separate compatibility classifier."""

    def __init__(
        self,
        *,
        actor_source_dim: int,
        camera_source_dim: int,
        semantic_dim: int,
        condition_dim: int = 16,
        hidden_dim: int = 128,
        compatibility_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        dimensions = (
            actor_source_dim,
            camera_source_dim,
            semantic_dim,
            condition_dim,
            hidden_dim,
        )
        if any(
            isinstance(value, bool) or int(value) < 1
            for value in dimensions
        ):
            raise ValueError("all R6 dimensions must be positive")
        compatibility_width = (
            int(hidden_dim)
            if compatibility_hidden_dim is None
            else int(compatibility_hidden_dim)
        )
        if compatibility_width < 1:
            raise ValueError("compatibility_hidden_dim must be positive")
        self.actor_source_dim = int(actor_source_dim)
        self.camera_source_dim = int(camera_source_dim)
        self.semantic_dim = int(semantic_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        motion_dim = self.condition_dim + 1
        self.actor = _R6PredictorBranch(
            source_dim=self.actor_source_dim,
            semantic_dim=self.semantic_dim,
            motion_dim=motion_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )
        self.camera = _R6PredictorBranch(
            source_dim=self.camera_source_dim,
            semantic_dim=self.semantic_dim,
            motion_dim=motion_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )
        self.compatibility = _R6CompatibilityHead(
            semantic_dim=self.semantic_dim,
            actor_motion_dim=motion_dim,
            camera_motion_dim=motion_dim,
            hidden_dim=compatibility_width,
        )

    @staticmethod
    def _tensor(name: str, value: Any, *, width: int) -> None:
        torch = _torch()
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [B,{width}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")

    @staticmethod
    def _mask(reference: Any, value: Any | None, *, name: str) -> Any:
        torch = _torch()
        rows = reference.shape[0]
        if value is None:
            return reference.new_ones((rows, 1))
        if not torch.is_tensor(value) or value.shape not in {(rows,), (rows, 1)}:
            raise ValueError(f"{name} must have shape [B] or [B,1]")
        mask = value.reshape(rows, 1).to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if (
            not bool(torch.isfinite(mask).all())
            or bool(((mask < 0.0) | (mask > 1.0)).any())
        ):
            raise ValueError(f"{name} must contain values in [0,1]")
        return mask

    def forward(
        self,
        *,
        source_actor: Any,
        source_camera: Any,
        semantic_features: Any,
        reference_actor_motion: Any | None = None,
        reference_camera_motion: Any | None = None,
        semantic_mask: Any | None = None,
        reference_mask: Any | None = None,
    ) -> dict[str, Any]:
        """Predict a query delta without accepting a query target argument."""

        torch = _torch()
        self._tensor(
            "source_actor",
            source_actor,
            width=self.actor_source_dim,
        )
        self._tensor(
            "source_camera",
            source_camera,
            width=self.camera_source_dim,
        )
        self._tensor(
            "semantic_features",
            semantic_features,
            width=self.semantic_dim,
        )
        rows = source_actor.shape[0]
        if source_camera.shape[0] != rows or semantic_features.shape[0] != rows:
            raise ValueError("R6 predictor batch dimensions differ")
        has_reference = reference_actor_motion is not None
        if has_reference != (reference_camera_motion is not None):
            raise ValueError(
                "actor/camera reference motion must be supplied together"
            )
        if not has_reference and reference_mask is not None:
            raise ValueError("reference_mask requires reference motion")
        motion_width = self.condition_dim + 1
        if reference_actor_motion is None:
            reference_actor_motion = source_actor.new_zeros(
                (rows, motion_width)
            )
            reference_camera_motion = source_actor.new_zeros(
                (rows, motion_width)
            )
            reference_availability = source_actor.new_zeros((rows, 1))
        else:
            self._tensor(
                "reference_actor_motion",
                reference_actor_motion,
                width=motion_width,
            )
            self._tensor(
                "reference_camera_motion",
                reference_camera_motion,
                width=motion_width,
            )
            if (
                reference_actor_motion.shape[0] != rows
                or reference_camera_motion.shape[0] != rows
            ):
                raise ValueError("R6 reference motion batch dimensions differ")
            reference_availability = self._mask(
                source_actor,
                reference_mask,
                name="reference_mask",
            )
        semantic_availability = self._mask(
            source_actor,
            semantic_mask,
            name="semantic_mask",
        )
        actor_direction, actor_magnitude = self.actor(
            source_actor,
            semantic_features,
            reference_actor_motion,
            semantic_availability,
            reference_availability,
        )
        camera_direction, camera_magnitude = self.camera(
            source_camera,
            semantic_features,
            reference_camera_motion,
            semantic_availability,
            reference_availability,
        )
        actor_scale = torch.expm1(actor_magnitude.float()).to(
            actor_direction.dtype
        )
        camera_scale = torch.expm1(camera_magnitude.float()).to(
            camera_direction.dtype
        )
        # There are deliberately no source-token or learned role-token
        # parameters. Every value below is directly supervised through the
        # positive direction/magnitude objective. Compatibility never appears
        # in this computation.
        motion_tokens = torch.stack(
            (
                actor_direction * actor_scale,
                camera_direction * camera_scale,
            ),
            dim=1,
        )
        return {
            "actor_direction": actor_direction,
            "actor_log_magnitude": actor_magnitude,
            "camera_direction": camera_direction,
            "camera_log_magnitude": camera_magnitude,
            "motion_conditioning_tokens": motion_tokens,
            "motion_token_roles": R6_MOTION_TOKEN_ROLES,
            "generation_token_export_authorized": False,
            "semantic_mask": semantic_availability,
            "reference_mask": reference_availability,
        }

    def score_compatibility(
        self,
        *,
        semantic_features: Any,
        candidate_actor_motion: Any,
        candidate_camera_motion: Any,
    ) -> Any:
        """Score prompt/observed-motion alignment; never gate delta tokens."""

        self._tensor(
            "semantic_features",
            semantic_features,
            width=self.semantic_dim,
        )
        self._tensor(
            "candidate_actor_motion",
            candidate_actor_motion,
            width=self.condition_dim + 1,
        )
        self._tensor(
            "candidate_camera_motion",
            candidate_camera_motion,
            width=self.condition_dim + 1,
        )
        rows = semantic_features.shape[0]
        if (
            candidate_actor_motion.shape[0] != rows
            or candidate_camera_motion.shape[0] != rows
        ):
            raise ValueError("R6 compatibility batch dimensions differ")
        return self.compatibility(
            semantic_features,
            candidate_actor_motion,
            candidate_camera_motion,
        )

    def predictor_parameters(self) -> Any:
        """Parameters allowed to receive positive delta loss gradients."""

        for module in (self.actor, self.camera):
            yield from module.parameters()

    def compatibility_parameters(self) -> Any:
        """Parameters allowed to receive compatibility loss gradients."""

        yield from self.compatibility.parameters()


def positive_factorized_r6_loss(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    positive_mask: Any,
    direction_weight: float = 1.0,
    magnitude_weight: float = 0.25,
    active_log_magnitude_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Direction/magnitude loss whose gradient is zero for failed outcomes."""

    torch = _torch()
    if direction_weight < 0.0 or magnitude_weight < 0.0:
        raise ValueError("R6 delta loss weights must be non-negative")
    if not torch.is_tensor(positive_mask):
        raise ValueError("positive_mask must be a tensor")
    mask = positive_mask.reshape(-1).bool()
    losses: dict[str, Any] = {}
    total = None
    for factor in ("actor", "camera"):
        predicted_direction = prediction[f"{factor}_direction"].float()
        target_direction = target[f"{factor}_direction"].float()
        predicted_magnitude = prediction[f"{factor}_log_magnitude"].float().reshape(
            -1
        )
        target_magnitude = target[f"{factor}_log_magnitude"].float().reshape(-1)
        if predicted_direction.shape != target_direction.shape:
            raise ValueError(f"{factor} direction shape mismatch")
        if (
            len(predicted_direction) != len(predicted_magnitude)
            or mask.shape != predicted_magnitude.shape
        ):
            raise ValueError(f"{factor} delta mask/magnitude shape mismatch")
        active = mask & (
            target_magnitude > float(active_log_magnitude_threshold)
        )
        cosine = 1.0 - torch.sum(
            predicted_direction * target_direction,
            dim=-1,
        )
        direction_loss = (
            cosine[active].mean()
            if bool(active.any())
            else cosine.new_zeros(())
        )
        magnitude_loss = (
            torch.nn.functional.smooth_l1_loss(
                predicted_magnitude[mask],
                target_magnitude[mask],
            )
            if bool(mask.any())
            else predicted_magnitude.new_zeros(())
        )
        factor_loss = (
            float(direction_weight) * direction_loss
            + float(magnitude_weight) * magnitude_loss
        )
        losses[f"{factor}_direction"] = direction_loss
        losses[f"{factor}_magnitude"] = magnitude_loss
        losses[f"{factor}_positive_count"] = mask.sum()
        losses[f"{factor}_active_count"] = active.sum()
        total = factor_loss if total is None else total + factor_loss
    losses["delta_loss"] = total
    return losses


def pair_compatibility_loss(
    logits: Any,
    targets: Any,
    *,
    positive_weight: float | None = None,
) -> Any:
    """Binary text/observed-motion alignment loss."""

    torch = _torch()
    if not torch.is_tensor(logits) or not torch.is_tensor(targets):
        raise ValueError("compatibility logits/targets must be tensors")
    scores = logits.float().reshape(-1)
    labels = targets.float().reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("compatibility logits/targets shape mismatch")
    if (
        not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(labels).all())
        or bool(((labels != 0.0) & (labels != 1.0)).any())
    ):
        raise ValueError("compatibility logits/targets are invalid")
    weight = None
    if positive_weight is not None:
        value = float(positive_weight)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("positive_weight must be finite and positive")
        weight = scores.new_tensor(value)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        scores,
        labels,
        pos_weight=weight,
    )


def r6_training_loss(
    *,
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    positive_mask: Any,
    compatibility_logits: Any,
    compatibility_targets: Any,
    compatibility_weight: float = 1.0,
    compatibility_positive_weight: float | None = None,
    direction_weight: float = 1.0,
    magnitude_weight: float = 0.25,
) -> dict[str, Any]:
    """Joint scalar with disjoint model branches and explicit loss masks."""

    if compatibility_weight < 0.0:
        raise ValueError("compatibility_weight must be non-negative")
    delta = positive_factorized_r6_loss(
        prediction,
        target,
        positive_mask=positive_mask,
        direction_weight=direction_weight,
        magnitude_weight=magnitude_weight,
    )
    compatibility = pair_compatibility_loss(
        compatibility_logits,
        compatibility_targets,
        positive_weight=compatibility_positive_weight,
    )
    return {
        **delta,
        "compatibility_loss": compatibility,
        "loss": delta["delta_loss"]
        + float(compatibility_weight) * compatibility,
    }
