"""Train a lightweight prompt action encoder from geometry-delta teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .action_repr import (
    ACTION_REPR_SCHEMA,
    PROMPT_HASH_VERSION,
    PromptActionEncoder,
    TeacherTransform,
    build_raw_action_teacher,
    prompt_hash_features,
    save_action_checkpoint,
)
from .archive import (
    assert_archives_compatible,
    build_feature_metadata,
    load_feature_archive,
    save_feature_archive,
)

UPSTREAM_PROVENANCE_VERSION = "motive-action-upstream-bytes-v2"
HUMAN_REVIEW_SCHEMA = "motive-action-human-review-v1"
EXPERIMENT_PSEUDO_LABEL_SCHEMA = "motive-experiment-pseudo-label-v1"
SUPPORTED_EXPERIMENT_PSEUDO_POLICIES = {
    "legacy-qwen-original-valid-action-v1",
}
HUMAN_APPROVED_VERDICTS = {"valid_action", "valid_suppression"}
HUMAN_REJECTED_VERDICTS = {
    "endpoint_only",
    "appearance_only",
    "camera_motion",
    "background_motion",
    "static",
    "instruction_mismatch",
    "artifact",
    "uncertain",
}
CONTENT_SPLIT_VERSIONS = {
    "source-sampled-phash-v1",
    "source-visual-cluster-v1",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class _DescriptorReference:
    feature: np.ndarray
    archive_name: str
    archive_sha256: str
    feature_index: int
    input_digest: str


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validated_row_input_digest(
    row: dict[str, Any],
    *,
    context: str,
) -> str:
    digest = row.get("input_digest")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{context} has missing/invalid input_digest")
    required = (
        "iid",
        "prompt",
        "src_video",
        "tgt_video",
        "source_caption",
        "edited_caption",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{context} cannot validate input_digest; missing {missing}")
    rebuilt = _canonical_digest({key: row[key] for key in required})
    if rebuilt != digest:
        raise ValueError(
            f"{context} input_digest does not match its prompt/video/caption fields"
        )
    return digest


def _load_descriptors(
    feature_dir: Path,
) -> tuple[
    dict[str, _DescriptorReference],
    list[dict[str, Any]],
    dict[str, Any],
]:
    descriptors: dict[str, _DescriptorReference] = {}
    archive_inventory: list[dict[str, Any]] = []
    reference_metadata: dict[str, Any] | None = None
    paths = sorted(feature_dir.glob("shard-*.npz"))
    if not paths:
        raise RuntimeError(f"no shard-*.npz archives found in {feature_dir}")
    for path in paths:
        json_path = path.with_suffix(".jsonl")
        done_path = path.with_suffix(".done.json")
        if not json_path.is_file() or not done_path.is_file():
            raise ValueError(
                f"{path} is not a committed cascade shard: expected adjacent "
                f"{json_path.name} and {done_path.name}"
            )
        archive_sha256_before = _file_digest(path)
        features, ids, metadata = load_feature_archive(path)
        archive_sha256 = _file_digest(path)
        if archive_sha256 != archive_sha256_before:
            raise RuntimeError(f"{path} changed while it was being loaded")
        if metadata["feature_kind"] != "geometry_action_delta":
            raise ValueError(f"unexpected feature kind in {path}")
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            assert_archives_compatible(reference_metadata, metadata)

        stage_rows = list(_iter_jsonl(json_path))
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            raise ValueError(f"{done_path} is not a JSON object")
        if marker.get("archive_sha256") != archive_sha256:
            raise ValueError(f"{path} archive checksum disagrees with {done_path}")
        json_sha256 = _file_digest(json_path)
        if marker.get("json_sha256") != json_sha256:
            raise ValueError(f"{json_path} checksum disagrees with {done_path}")
        if int(marker.get("rows", -1)) != len(stage_rows):
            raise ValueError(f"{done_path} row count disagrees with {json_path}")
        shard_input_digest = _canonical_digest(
            [
                (row.get("iid"), row.get("input_digest"))
                for row in stage_rows
            ]
        )
        if marker.get("input_digest") != shard_input_digest:
            raise ValueError(f"{done_path} input digest disagrees with {json_path}")

        referenced_indices: set[int] = set()
        for row_number, row in enumerate(stage_rows, start=1):
            auto_feature = row.get("auto_feature")
            if not isinstance(auto_feature, dict):
                raise ValueError(
                    f"{json_path}:{row_number} has no auto_feature object"
                )
            if "feature_index" not in auto_feature:
                continue
            iid = str(row.get("iid"))
            input_digest = _validated_row_input_digest(
                row,
                context=f"{json_path}:{row_number}",
            )
            feature_index = auto_feature["feature_index"]
            if (
                isinstance(feature_index, bool)
                or not isinstance(feature_index, int)
                or not 0 <= feature_index < len(ids)
            ):
                raise ValueError(
                    f"{json_path}:{row_number} has invalid feature_index"
                )
            if feature_index in referenced_indices:
                raise ValueError(
                    f"{json_path}:{row_number} duplicates feature_index="
                    f"{feature_index}"
                )
            if str(ids[feature_index]) != iid:
                raise ValueError(
                    f"{json_path}:{row_number} iid does not match "
                    f"{path.name}[{feature_index}]"
                )
            if iid in descriptors:
                raise ValueError(f"duplicate descriptor iid={iid}")
            referenced_indices.add(feature_index)
            descriptors[iid] = _DescriptorReference(
                feature=np.asarray(features[feature_index], dtype=np.float32),
                archive_name=path.name,
                archive_sha256=archive_sha256,
                feature_index=feature_index,
                input_digest=input_digest,
            )
        expected_indices = set(range(len(ids)))
        if referenced_indices != expected_indices:
            missing = sorted(expected_indices - referenced_indices)
            raise ValueError(
                f"{path} contains features not referenced by {json_path}: "
                f"indices={missing[:10]}"
            )
        if int(marker.get("successful", -1)) != len(ids):
            raise ValueError(f"{done_path} successful count disagrees with {path}")
        archive_inventory.append(
            {
                "archive_name": path.name,
                "archive_sha256": archive_sha256,
                "archive_bytes": path.stat().st_size,
                "json_name": json_path.name,
                "json_sha256": json_sha256,
                "done_name": done_path.name,
                "done_sha256": _file_digest(done_path),
                "compatibility_digest": metadata["compatibility_digest"],
                "feature_count": int(len(ids)),
            }
        )
    assert reference_metadata is not None
    return descriptors, archive_inventory, reference_metadata


def _validate_manifest_reference(
    row: dict[str, Any],
    reference: _DescriptorReference,
    *,
    context: str,
) -> None:
    input_digest = _validated_row_input_digest(row, context=context)
    if input_digest != reference.input_digest:
        raise ValueError(
            f"{context} input_digest disagrees with committed feature shard"
        )
    auto_feature = row.get("auto_feature")
    if not isinstance(auto_feature, dict):
        raise ValueError(f"{context} has no auto_feature object")
    if auto_feature.get("feature_index") != reference.feature_index:
        raise ValueError(
            f"{context} feature_index disagrees with "
            f"{reference.archive_name}"
        )
    archive_name = auto_feature.get("feature_archive")
    if archive_name is not None and Path(str(archive_name)).name != reference.archive_name:
        raise ValueError(
            f"{context} feature_archive disagrees with committed feature shard"
        )
    archive_sha256 = auto_feature.get("feature_archive_sha256")
    if archive_sha256 is not None and archive_sha256 != reference.archive_sha256:
        raise ValueError(
            f"{context} feature_archive_sha256 disagrees with committed archive"
        )


def _scalar_features(row: dict[str, Any]) -> np.ndarray:
    feature = row["auto_feature"]
    source = feature["source_metrics"]
    target = feature["target_metrics"]
    source_actor = feature["source_actor_features"]
    target_actor = feature["target_actor_features"]
    source_p90 = float(source["residual_speed_p90"])
    target_p90 = float(target["residual_speed_p90"])
    return np.asarray(
        [
            np.log1p(100.0 * source_p90),
            np.log1p(100.0 * target_p90),
            float(source["active_pixel_fraction"]),
            float(target["active_pixel_fraction"]),
            float(source["active_frame_fraction"]),
            float(target["active_frame_fraction"]),
            float(source_actor["actor_likeness"]),
            float(target_actor["actor_likeness"]),
            float(source_actor["temporal_coverage"]),
            float(target_actor["temporal_coverage"]),
            float(feature["descriptor_delta_norm"]),
            float(
                np.clip(
                    np.log((target_p90 + 1e-5) / (source_p90 + 1e-5)),
                    -4.0,
                    4.0,
                )
            ),
        ],
        dtype=np.float32,
    )


def _signature_base(row: dict[str, Any], iid: str) -> str:
    review = row.get("human_review")
    has_human_review = review is not None
    human_value = (
        str(review.get("action_signature") or "").strip().lower()
        if isinstance(review, dict)
        else ""
    )
    final = row.get("final_triage", {})
    final_value = str(final.get("action_signature") or "").strip().lower()
    pseudo = row.get("experiment_pseudo_label")
    pseudo_value = (
        str(pseudo.get("action_signature") or "").strip().lower()
        if isinstance(pseudo, dict)
        else ""
    )
    families = [
        str(item).strip().lower()
        for item in row.get("auto_rule", {}).get("action_families", [])
        if str(item).strip()
    ]

    # Human corrections are authoritative.  Qwen/fused evidence is more
    # specific than the deliberately high-recall lexical families, so it is
    # the next fallback.  Never silently replace an explicit human "unknown"
    # with an automatic rule label: keep it iid-unique instead.
    if has_human_review:
        normalized = re.sub(r"[^a-z0-9]+", "_", human_value).strip("_")
        base = normalized if normalized and normalized != "unknown" else f"unknown:{iid}"
    elif pseudo_value:
        normalized = re.sub(r"[^a-z0-9]+", "_", pseudo_value).strip("_")
        base = normalized if normalized else f"unknown:{iid}"
    elif final_value and final_value != "unknown":
        base = re.sub(r"[^a-z0-9]+", "_", final_value).strip("_")
    elif families:
        base = "+".join(dict.fromkeys(families))
    else:
        base = f"unknown:{iid}"
    return base


def _validate_experiment_pseudo_label(
    row: dict[str, Any],
    *,
    context: str,
) -> str | None:
    pseudo = row.get("experiment_pseudo_label")
    if pseudo is None:
        return None
    if not isinstance(pseudo, dict):
        raise ValueError(f"{context} experiment_pseudo_label must be an object")
    required = {
        "schema_version",
        "policy",
        "action_signature",
        "source_manifest_sha256",
        "observation_digest",
        "result_object_digest",
        "legacy_result_digest_missing",
        "human_approved",
        "production_eligible",
    }
    if set(pseudo) != required:
        raise ValueError(
            f"{context} experiment_pseudo_label keys differ: "
            f"{sorted(set(pseudo) ^ required)}"
        )
    if pseudo["schema_version"] != EXPERIMENT_PSEUDO_LABEL_SCHEMA:
        raise ValueError(f"{context} has unsupported experiment pseudo schema")
    if pseudo["policy"] not in SUPPORTED_EXPERIMENT_PSEUDO_POLICIES:
        raise ValueError(f"{context} has unsupported experiment pseudo policy")
    for key in (
        "source_manifest_sha256",
        "observation_digest",
        "result_object_digest",
    ):
        if (
            not isinstance(pseudo[key], str)
            or _SHA256_RE.fullmatch(pseudo[key]) is None
        ):
            raise ValueError(f"{context} experiment pseudo {key} is invalid")
    if pseudo["human_approved"] is not False:
        raise ValueError(f"{context} experiment pseudo label claims human approval")
    if pseudo["production_eligible"] is not False:
        raise ValueError(
            f"{context} experiment pseudo label claims production eligibility"
        )
    if not isinstance(pseudo["legacy_result_digest_missing"], bool):
        raise ValueError(
            f"{context} legacy_result_digest_missing must be boolean"
        )
    action_signature = pseudo["action_signature"]
    if not isinstance(action_signature, str) or not action_signature.strip():
        raise ValueError(
            f"{context} experiment pseudo action_signature must be non-empty"
        )
    return str(pseudo["policy"])


def _signature(row: dict[str, Any], iid: str) -> str:
    prompt = str(row.get("prompt") or "").lower()
    base = _signature_base(row, iid)

    modifiers: list[str] = []
    cue_groups = (
        ("dir_left", r"\b(?:left|leftward)\b"),
        ("dir_right", r"\b(?:right|rightward)\b"),
        ("dir_up", r"\b(?:upward|upwards|up)\b"),
        ("dir_down", r"\b(?:downward|downwards|down)\b"),
        ("dir_forward", r"\bforward\b"),
        ("dir_backward", r"\bbackward\b"),
        ("dir_toward_camera", r"\btoward(?:s)? (?:the )?camera\b"),
        ("dir_away_camera", r"\baway from (?:the )?camera\b"),
        ("speed_faster", r"\b(?:faster|quickly|speed up)\b"),
        ("speed_slower", r"\b(?:slower|slowly|slow down)\b"),
        ("phase_start", r"\b(?:start|begin)(?:s|ning)?\b"),
        ("phase_stop", r"\b(?:stop|cease|freeze)(?:s|ping|ped)?\b"),
        ("phase_repeat", r"\b(?:repeat|repeatedly|continuously)\b"),
    )
    for name, pattern in cue_groups:
        if re.search(pattern, prompt):
            modifiers.append(name)
    return "|".join((base, *modifiers))


def _decision(row: dict[str, Any]) -> str:
    if "final_triage" in row:
        return str(row["final_triage"].get("decision", "review"))
    return str(row.get("auto_decision", {}).get("decision", "review"))


def _human_review_verdict(row: dict[str, Any], *, context: str) -> str | None:
    review = row.get("human_review")
    if review is None:
        return None
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
    if verdict not in HUMAN_APPROVED_VERDICTS | HUMAN_REJECTED_VERDICTS:
        raise ValueError(f"{context} has invalid human_review verdict")
    reviewer = review["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"{context} human_review reviewer must be non-empty")
    label_source_sha256 = review["label_source_sha256"]
    if (
        not isinstance(label_source_sha256, str)
        or _SHA256_RE.fullmatch(label_source_sha256) is None
    ):
        raise ValueError(
            f"{context} human_review label_source_sha256 is invalid"
        )
    return str(verdict)


def _confidence(row: dict[str, Any]) -> float:
    review = row.get("human_review")
    if isinstance(review, dict):
        # Once a person has accepted a row, automation scores must not
        # down-weight hard examples.  Optional reviewer confidence only
        # expresses label certainty, never the cascade's prior belief.
        review_confidence = str(
            review.get("review_confidence") or ""
        ).strip().lower()
        return {
            "high": 1.0,
            "medium": 0.8,
            "low": 0.6,
            "uncertain": 0.5,
        }.get(review_confidence, 1.0)
    final = row.get("final_triage") or row.get("auto_decision") or {}
    value = final.get("heuristic_score")
    if value is None:
        return 0.5
    return float(np.clip(float(value), 0.1, 1.0))


def _family_capped_indices(
    rows: Sequence[dict[str, Any]],
    *,
    max_per_action_family: int,
    seed: int,
) -> list[int]:
    """Cap each split independently so held-out frequency cannot alter training."""

    if max_per_action_family <= 0:
        return list(range(len(rows)))
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        iid = str(row["iid"])
        family = _signature_base(row, iid)
        if family.startswith("unknown:"):
            family = "unknown"
        split = str(row.get("split") or "unspecified")
        priority = int(
            hashlib.sha256(f"{seed}\0{iid}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        grouped.setdefault((split, family), []).append((priority, index))
    return sorted(
        index
        for values in grouped.values()
        for _priority, index in sorted(values)[:max_per_action_family]
    )


def _train_only_signature_weights(
    labels: Any,
    train_indices: np.ndarray,
) -> Any:
    """Inverse-sqrt weights whose statistics are fitted on train only."""

    import torch

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=labels.device)
    if not len(indices):
        raise ValueError("train_indices must be non-empty")
    if int(indices.min()) < 0 or int(indices.max()) >= len(labels):
        raise ValueError("train_indices are out of bounds")
    num_labels = int(labels.max().item()) + 1 if len(labels) else 0
    counts = torch.bincount(labels[indices], minlength=num_labels).float()
    # Held-out-only signatures are assigned a neutral value, but these entries
    # are never consumed by the training loop.
    return counts[labels].clamp_min(1.0).rsqrt()


def _multi_positive_loss(
    student: Any,
    teacher: Any,
    labels: Any,
    *,
    temperature: float,
    anchor_weights: Any | None = None,
) -> Any:
    import torch

    logits = student.float() @ teacher.float().t() / temperature
    positive = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positive = positive | eye
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    forward = -torch.logsumexp(
        log_probability.masked_fill(~positive, float("-inf")),
        dim=1,
    )
    reverse_logits = logits.t()
    reverse_log_probability = reverse_logits - torch.logsumexp(
        reverse_logits,
        dim=1,
        keepdim=True,
    )
    reverse = -torch.logsumexp(
        reverse_log_probability.masked_fill(~positive.t(), float("-inf")),
        dim=1,
    )
    if anchor_weights is None:
        return 0.5 * (forward.mean() + reverse.mean())
    weights = anchor_weights.float()
    weights = weights / weights.sum().clamp_min(1e-8)
    return 0.5 * (
        torch.sum(forward * weights)
        + torch.sum(reverse * weights)
    )


def _variance_covariance_loss(values: Any) -> Any:
    import torch

    if len(values) < 2:
        return values.new_zeros(())
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(0.25 - standard_deviation).mean()
    covariance = centered.t() @ centered / max(len(values) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return variance + 0.02 * off_diagonal.pow(2).mean()


def _retrieval_metrics(
    student: np.ndarray,
    teacher: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    if not len(student):
        return {
            "count": 0,
            "mean_cosine": 0.0,
            "diagonal_recall_at_1": 0.0,
            "signature_recall_at_1": 0.0,
        }
    scores = student @ teacher.T
    nearest = np.argmax(scores, axis=1)
    result = {
        "count": int(len(student)),
        "mean_cosine": float(np.mean(np.sum(student * teacher, axis=1))),
        "diagonal_recall_at_1": float(
            np.mean(nearest == np.arange(len(scores)))
        ),
    }
    result["signature_recall_at_1"] = (
        float(np.mean(labels[nearest] == labels))
        if labels is not None
        else result["diagonal_recall_at_1"]
    )
    return result


def _signature_centroid_metrics(
    *,
    train_teacher: np.ndarray,
    train_labels: np.ndarray,
    evaluation_teacher: np.ndarray,
    evaluation_labels: np.ndarray,
) -> dict[str, float]:
    """Train-fitted signature lookup baseline; never fits on held-out vectors."""

    if not len(evaluation_teacher):
        return {
            "count": 0,
            "mean_cosine": 0.0,
            "seen_signature_fraction": 0.0,
        }
    if not len(train_teacher):
        raise ValueError("signature centroid baseline requires training samples")
    global_centroid = np.mean(train_teacher, axis=0)
    global_centroid /= max(float(np.linalg.norm(global_centroid)), 1e-8)
    centroids: dict[int, np.ndarray] = {}
    for label in np.unique(train_labels):
        centroid = np.mean(train_teacher[train_labels == label], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        centroids[int(label)] = centroid
    predictions = np.stack(
        [centroids.get(int(label), global_centroid) for label in evaluation_labels]
    )
    cosine = np.sum(predictions * evaluation_teacher, axis=1)
    seen = np.asarray(
        [int(label) in centroids for label in evaluation_labels],
        dtype=np.float32,
    )
    return {
        "count": int(len(evaluation_teacher)),
        "mean_cosine": float(np.mean(cosine)),
        "seen_signature_fraction": float(np.mean(seen)),
    }


def _within_signature_shuffled_teacher_metrics(
    student: np.ndarray,
    teacher: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    """Pairing shortcut baseline from deterministic within-signature rotations."""

    if not len(student):
        return {
            "count": 0,
            "shuffled_count": 0,
            "mean_cosine": 0.0,
            "paired_minus_shuffled_mean_cosine": 0.0,
        }
    rng = np.random.default_rng(seed)
    shuffled = np.arange(len(labels), dtype=np.int64)
    shuffled_count = 0
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            continue
        offset = int(rng.integers(1, len(indices)))
        shuffled[indices] = np.roll(indices, offset)
        shuffled_count += len(indices)
    shuffled_cosine = np.sum(student * teacher[shuffled], axis=1)
    paired_cosine = np.sum(student * teacher, axis=1)
    return {
        "count": int(len(student)),
        "shuffled_count": int(shuffled_count),
        "mean_cosine": float(np.mean(shuffled_cosine)),
        "paired_minus_shuffled_mean_cosine": float(
            np.mean(paired_cosine) - np.mean(shuffled_cosine)
        ),
    }


def _split_metrics(
    *,
    student: np.ndarray,
    teacher: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    train_teacher: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    split_student = student[indices]
    split_teacher = teacher[indices]
    split_labels = labels[indices]
    model_metrics = _retrieval_metrics(
        split_student,
        split_teacher,
        split_labels,
    )
    centroid = _signature_centroid_metrics(
        train_teacher=train_teacher,
        train_labels=train_labels,
        evaluation_teacher=split_teacher,
        evaluation_labels=split_labels,
    )
    shuffled = _within_signature_shuffled_teacher_metrics(
        split_student,
        split_teacher,
        split_labels,
        seed=seed,
    )
    model_metrics["minus_signature_centroid_mean_cosine"] = float(
        model_metrics["mean_cosine"] - centroid["mean_cosine"]
    )
    return model_metrics, {
        "signature_centroid_train_only": centroid,
        "within_signature_shuffled_teacher": shuffled,
    }


def train(args: argparse.Namespace) -> int:
    import torch

    max_per_action_family = int(
        getattr(args, "max_per_action_family", 2000)
    )
    if max_per_action_family < 0:
        raise ValueError("--max-per-action-family must be non-negative")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least two")
    if args.action_dim <= 0 or args.text_feature_dim <= 0:
        raise ValueError("representation dimensions must be positive")
    if args.camera_dims <= 0:
        raise ValueError("--camera-dims must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.epochs <= 0 or args.log_every <= 0:
        raise ValueError("--epochs and --log-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    descriptors, archive_inventory, descriptor_metadata = _load_descriptors(
        args.feature_dir.expanduser()
    )
    allowed_decisions = set(args.decisions)
    allow_unreviewed = bool(
        getattr(args, "allow_unreviewed_pseudo_labels", False)
    )
    allow_non_content_splits = bool(
        getattr(args, "allow_non_content_splits", False)
    )
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    scalars: list[np.ndarray] = []
    references: list[_DescriptorReference] = []
    seen_manifest_ids: set[str] = set()
    review_counts: Counter[str] = Counter()
    pseudo_policy_counts: Counter[str] = Counter()
    for line_number, row in enumerate(
        _iter_jsonl(args.manifest.expanduser()),
        start=1,
    ):
        iid = str(row["iid"])
        if _decision(row) not in allowed_decisions:
            continue
        review_verdict = _human_review_verdict(
            row,
            context=f"{args.manifest}:{line_number}",
        )
        pseudo_policy = _validate_experiment_pseudo_label(
            row,
            context=f"{args.manifest}:{line_number}",
        )
        if review_verdict in HUMAN_REJECTED_VERDICTS:
            review_counts[f"human_reject:{review_verdict}"] += 1
            continue
        if review_verdict in HUMAN_APPROVED_VERDICTS:
            review_counts[f"human_approve:{review_verdict}"] += 1
        elif allow_unreviewed:
            if pseudo_policy is not None:
                pseudo_policy_counts[pseudo_policy] += 1
            review_counts["unreviewed_pseudo_label"] += 1
        else:
            review_counts["pending_human_review"] += 1
            continue
        split_version = str(
            (row.get("split_provenance") or {}).get("version", "")
        )
        if (
            not allow_non_content_splits
            and split_version not in CONTENT_SPLIT_VERSIONS
        ):
            raise ValueError(
                f"{args.manifest}:{line_number} uses split provenance "
                f"{split_version or '<missing>'!r}; action representation "
                "training requires a content-derived source split in "
                f"{sorted(CONTENT_SPLIT_VERSIONS)}. "
                "--allow-non-content-splits is baseline-only."
            )
        if iid not in descriptors:
            raise ValueError(
                f"{args.manifest}:{line_number} selects iid={iid}, but no "
                "committed feature archive contains it"
            )
        if iid in seen_manifest_ids:
            raise ValueError(f"duplicate manifest iid={iid}")
        reference = descriptors[iid]
        _validate_manifest_reference(
            row,
            reference,
            context=f"{args.manifest}:{line_number}",
        )
        seen_manifest_ids.add(iid)
        rows.append(row)
        features.append(reference.feature)
        scalars.append(_scalar_features(row))
        references.append(reference)
    eligible_samples = len(rows)
    if max_per_action_family > 0:
        selected_indices = _family_capped_indices(
            rows,
            max_per_action_family=max_per_action_family,
            seed=args.seed,
        )
        rows = [rows[index] for index in selected_indices]
        features = [features[index] for index in selected_indices]
        scalars = [scalars[index] for index in selected_indices]
        references = [references[index] for index in selected_indices]
    if len(rows) < 4:
        raise RuntimeError(
            "need at least four eligible samples with committed descriptors; "
            f"review_counts={dict(sorted(review_counts.items()))}. By default "
            f"human_review schema {HUMAN_REVIEW_SCHEMA!r} with verdict "
            f"{sorted(HUMAN_APPROVED_VERDICTS)} is required. "
            "--allow-unreviewed-pseudo-labels is baseline-only."
        )

    descriptor_matrix = np.stack(features)
    scalar_matrix = np.stack(scalars)
    raw_teacher = build_raw_action_teacher(
        descriptor_matrix,
        scalar_matrix,
        camera_dims=args.camera_dims,
    )
    train_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.get("split") == "train"],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.get("split") == "validation"],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.get("split") == "test"],
        dtype=np.int64,
    )
    if len(train_indices) < 2:
        raise RuntimeError("need at least two training-split samples")
    transform = TeacherTransform.fit(
        raw_teacher[train_indices],
        output_dim=args.action_dim,
        camera_dims_excluded=args.camera_dims,
    )
    teacher = transform.transform(raw_teacher)
    prompts = [str(row["prompt"]) for row in rows]
    prompt_features = prompt_hash_features(
        prompts,
        feature_dim=args.text_feature_dim,
    ).cpu()
    signature_values = [_signature(row, str(row["iid"])) for row in rows]
    signature_vocab = {
        value: index for index, value in enumerate(sorted(set(signature_values)))
    }
    labels = torch.tensor(
        [signature_vocab[value] for value in signature_values],
        dtype=torch.long,
    )
    confidence = torch.tensor(
        [_confidence(row) for row in rows],
        dtype=torch.float32,
    )
    sample_weights = _train_only_signature_weights(labels, train_indices)
    teacher_tensor = torch.from_numpy(teacher)

    device = torch.device(args.device)
    model = PromptActionEncoder(
        input_dim=args.text_feature_dim,
        action_dim=args.action_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    history: list[float] = []
    for epoch in range(args.epochs):
        order = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        epoch_losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            if len(indices) < 2:
                continue
            batch_features = prompt_features[indices].to(device)
            batch_teacher = teacher_tensor[indices].to(device)
            batch_labels = labels[indices].to(device)
            batch_weights = (
                sample_weights[indices] * confidence[indices]
            ).to(device)
            student = model(batch_features)
            cosine = 1.0 - torch.sum(student.float() * batch_teacher.float(), dim=1)
            alignment = torch.sum(cosine * batch_weights) / (
                batch_weights.sum().clamp_min(1e-8)
            )
            contrastive = _multi_positive_loss(
                student,
                batch_teacher,
                batch_labels,
                temperature=args.temperature,
                anchor_weights=batch_weights,
            )
            regularization = _variance_covariance_loss(student)
            loss = (
                alignment
                + args.contrastive_weight * contrastive
                + args.regularization_weight * regularization
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(mean_loss)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(
                f"[motive-action-repr] epoch={epoch + 1}/{args.epochs} "
                f"loss={mean_loss:.6f}",
                flush=True,
            )

    model.eval()
    with torch.inference_mode():
        student_all = model(prompt_features.to(device)).float().cpu().numpy()
    label_array = labels.numpy()
    train_teacher = teacher[train_indices]
    train_labels = label_array[train_indices]
    train_metrics, train_baselines = _split_metrics(
        student=student_all,
        teacher=teacher,
        labels=label_array,
        indices=train_indices,
        train_teacher=train_teacher,
        train_labels=train_labels,
        seed=args.seed,
    )
    validation_metrics, validation_baselines = _split_metrics(
        student=student_all,
        teacher=teacher,
        labels=label_array,
        indices=validation_indices,
        train_teacher=train_teacher,
        train_labels=train_labels,
        seed=args.seed + 1,
    )
    test_metrics, test_baselines = _split_metrics(
        student=student_all,
        teacher=teacher,
        labels=label_array,
        indices=test_indices,
        train_teacher=train_teacher,
        train_labels=train_labels,
        seed=args.seed + 2,
    )
    manifest_sha256 = _file_digest(args.manifest.expanduser())
    selected_references = sorted(
        (
            {
                "iid": str(row["iid"]),
                "input_digest": reference.input_digest,
                "archive_name": reference.archive_name,
                "archive_sha256": reference.archive_sha256,
                "feature_index": reference.feature_index,
            }
            for row, reference in zip(rows, references)
        ),
        key=lambda item: item["iid"],
    )
    selected_reference_digest = _canonical_digest(selected_references)
    feature_archive_digest = _canonical_digest(archive_inventory)
    upstream_payload = {
        "provenance_version": UPSTREAM_PROVENANCE_VERSION,
        "feature_archive_digest": feature_archive_digest,
        "manifest_sha256": manifest_sha256,
        "selected_reference_digest": selected_reference_digest,
    }
    upstream_digest = _canonical_digest(upstream_payload)
    metrics = {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "loss_history": history,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "shortcut_baselines": {
            "train": train_baselines,
            "validation": validation_baselines,
            "test": test_baselines,
        },
        "samples": len(rows),
        "eligible_samples_before_family_cap": eligible_samples,
        "max_per_action_family": max_per_action_family,
        "family_cap_scope": "split_then_action_family",
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "validation_is_holdout": bool(len(validation_indices)),
        "test_is_holdout": bool(len(test_indices)),
        "sample_weight_fit_split": "train",
        "signatures": len(signature_vocab),
        "human_review_schema": HUMAN_REVIEW_SCHEMA,
        "review_counts": dict(sorted(review_counts.items())),
        "experiment_pseudo_policy_counts": dict(
            sorted(pseudo_policy_counts.items())
        ),
        "allow_unreviewed_pseudo_labels": allow_unreviewed,
        "label_policy": (
            "human_approved_or_unreviewed_pseudo_baseline"
            if allow_unreviewed
            else "human_approved_only"
        ),
        "content_split_versions": sorted(CONTENT_SPLIT_VERSIONS),
        "allow_non_content_splits": allow_non_content_splits,
    }
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "prompt_action_encoder.pt"
    checkpoint_metadata = save_action_checkpoint(
        checkpoint_path,
        model=model,
        teacher_transform=transform,
        provenance={
            "upstream_digest": upstream_digest,
            "upstream_provenance_version": UPSTREAM_PROVENANCE_VERSION,
            "upstream_payload": upstream_payload,
            "feature_archives": archive_inventory,
            "manifest_sha256": manifest_sha256,
            "selected_reference_digest": selected_reference_digest,
            "descriptor_compatibility_digest": descriptor_metadata[
                "compatibility_digest"
            ],
            "manifest": str(args.manifest),
            "split_seed": args.seed,
            "decisions": sorted(allowed_decisions),
            "human_review_schema": HUMAN_REVIEW_SCHEMA,
            "allow_unreviewed_pseudo_labels": allow_unreviewed,
            "experiment_pseudo_policies": sorted(pseudo_policy_counts),
            "label_policy": metrics["label_policy"],
            "content_split_versions": sorted(CONTENT_SPLIT_VERSIONS),
            "allow_non_content_splits": allow_non_content_splits,
            "prompt_hash_version": PROMPT_HASH_VERSION,
        },
        metrics=metrics,
    )
    embedding_metadata = build_feature_metadata(
        feature_kind="action_text_representation",
        dimension=args.action_dim,
        provenance={
            "schema_version": ACTION_REPR_SCHEMA,
            "checkpoint_state_digest": checkpoint_metadata["state_digest"],
            "upstream_digest": upstream_digest,
            "prompt_hash_version": PROMPT_HASH_VERSION,
        },
    )
    save_feature_archive(
        output_dir / "action_embeddings.npz",
        features=student_all,
        ids=np.asarray([str(row["iid"]) for row in rows]),
        metadata=embedding_metadata,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "training_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row, signature, reference in zip(
            rows,
            signature_values,
            references,
        ):
            handle.write(
                json.dumps(
                    {
                        "iid": row["iid"],
                        "split": row.get("split"),
                        "prompt": row["prompt"],
                        "action_signature": signature,
                        "decision": _decision(row),
                        "human_review_verdict": _human_review_verdict(
                            row,
                            context=f"training row iid={row['iid']}",
                        ),
                        "experiment_pseudo_policy": (
                            row.get("experiment_pseudo_label") or {}
                        ).get("policy"),
                        "confidence": _confidence(row),
                        "input_digest": reference.input_digest,
                        "feature_archive": reference.archive_name,
                        "feature_archive_sha256": reference.archive_sha256,
                        "feature_index": reference.feature_index,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"[motive-action-repr] samples={len(rows)} "
        f"val_r1={validation_metrics['diagonal_recall_at_1']:.4f} "
        f"test_r1={test_metrics['diagonal_recall_at_1']:.4f} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill motion geometry into a prompt action encoder."
    )
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--decisions",
        nargs="+",
        default=["auto_keep", "review", "auto_reject"],
        help=(
            "Automatic triage strata admitted before applying the authoritative "
            "human-review gate. Production defaults to every stratum so a "
            "human-approved hard case is not excluded by an earlier heuristic."
        ),
    )
    parser.add_argument(
        "--allow-unreviewed-pseudo-labels",
        action="store_true",
        help=(
            "Baseline-only opt-in: train on automatic pseudo-labels that lack "
            f"{HUMAN_REVIEW_SCHEMA} approval. Human-rejected rows remain excluded."
        ),
    )
    parser.add_argument(
        "--allow-non-content-splits",
        action="store_true",
        help=(
            "Baseline-only opt-in for legacy caption/path splits. Production "
            "training requires a content-derived source split."
        ),
    )
    parser.add_argument(
        "--max-per-action-family",
        type=int,
        default=2000,
        help="Deterministic cap preventing dominant families from swamping training.",
    )
    parser.add_argument("--camera-dims", type=int, default=8)
    parser.add_argument("--text-feature-dim", type=int, default=512)
    parser.add_argument("--action-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--regularization-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=260108828)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_per_action_family < 0:
        raise ValueError("--max-per-action-family must be non-negative")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
