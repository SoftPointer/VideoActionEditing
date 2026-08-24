"""Train and evaluate the R5-lite source-aware representation.

Inputs are the committed ``source_aware_features`` ``final.npz`` and
``manifest.jsonl``.  The implementation is intentionally fail-closed:

* only human-approved labels or strict, original/unrepaired Qwen evidence are
  accepted;
* pseudo labels are always diagnostic and ``production_eligible=false``;
* clean negatives never enter the delta loss;
* every transform, centroid and activation threshold is fitted on positive
  train rows only;
* ``data_seed`` controls the immutable split/control construction while
  ``model_seed`` controls initialization and optimizer sampling.

The trained model is evaluated as reference-aware ``full`` and ``text_only``.
``pairshuffle`` replaces the target reference, while ``source_shuffle`` and
``prompt_shuffle`` intervene on the trained full model. ``matched_random`` is
the identical untrained full architecture and ``centroid`` is train-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .qwen_filter import (
    _object_digest as _qwen_object_digest,
    _validate_observation,
    _validate_visual,
)
from .r5_gate import (
    R5GateThresholds,
    R5_PRODUCTION_SPLIT_VERSION,
    R5_QUERY_SCHEMA,
    cross_content_retrieval,
    evaluate_r5_gate,
    invalid_gate_summary,
    summarize_arm_rows,
)
from .source_aware_repr import (
    FactorizedCentroidControl,
    FactorizedR5Targets,
    R5EndpointBatch,
    R5FeatureTransform,
    SourceAwareFactorizedR5,
    audit_content_disjoint_splits,
    factorized_r5_loss,
    make_matched_random_control,
    prompt_shuffled_indices,
    source_shuffled_indices,
)


R5_TRAIN_SCHEMA = "motive-r5-lite-training-v1"
R5_HUMAN_REVIEW_SCHEMA = "motive-action-human-review-v1"
R5_PILOT_SCHEMA = "motive-r5-pseudo-pilot-v1"
R5_PILOT_PROFILE = "strict-legacy-qwen-original-v1"
LABEL_MODES = frozenset({"human", "strict_legacy_qwen"})
POSITIVE_VERDICTS = frozenset({"valid_action", "valid_suppression"})
NEGATIVE_AUDIT_VERDICTS = frozenset(
    {"static", "instruction_mismatch", "endpoint_only"}
)
HUMAN_VERDICTS = frozenset(
    {
        "valid_action",
        "valid_suppression",
        "endpoint_only",
        "appearance_only",
        "camera_motion",
        "background_motion",
        "static",
        "instruction_mismatch",
        "artifact",
        "uncertain",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FAMILY_TOKEN_RE = re.compile(r"[a-z0-9]+")


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
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _string_array(
    arrays: Mapping[str, np.ndarray],
    singular: str,
    plural: str | None = None,
) -> np.ndarray:
    if singular in arrays:
        values = np.asarray(arrays[singular]).astype(str)
    elif plural and plural in arrays:
        values = np.asarray(arrays[plural]).astype(str)
    else:
        raise ValueError(f"feature archive is missing {singular}")
    if values.ndim != 1:
        raise ValueError(f"{singular} must have shape [N]")
    if plural and plural in arrays:
        alias = np.asarray(arrays[plural]).astype(str)
        if not np.array_equal(values, alias):
            raise ValueError(f"{singular}/{plural} aliases differ")
    return values


def _matrix(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"feature archive is missing {name}")
    values = np.asarray(arrays[name], dtype=np.float32)
    if values.ndim != 2 or len(values) != rows or values.shape[1] < 1:
        raise ValueError(f"{name} must have finite shape [N,D]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def load_feature_bundle(
    *,
    feature_archive: Path,
    manifest_path: Path,
    data_seed: int,
) -> tuple[R5EndpointBatch, list[dict[str, Any]], dict[str, Any]]:
    """Load and re-audit a committed endpoint archive without assuming production."""

    feature_archive = feature_archive.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    archive_digest_before = _file_digest(feature_archive)
    manifest_digest_before = _file_digest(manifest_path)
    arrays = _load_npz(feature_archive)
    rows = _load_jsonl(manifest_path)
    if (
        _file_digest(feature_archive) != archive_digest_before
        or _file_digest(manifest_path) != manifest_digest_before
    ):
        raise RuntimeError("R5 feature input changed while it was read")
    if not rows:
        raise ValueError("R5 feature manifest is empty")

    iids = _string_array(arrays, "iids")
    count = len(iids)
    if count != len(rows) or len(set(iids.tolist())) != count:
        raise ValueError("feature archive/manifest row count or iid uniqueness failed")
    split = _string_array(arrays, "split", "splits")
    groups = _string_array(
        arrays,
        "content_group_id",
        "content_group_ids",
    )
    split_versions = _string_array(
        arrays,
        "split_version",
        "split_versions",
    )
    signatures = _string_array(
        arrays,
        "action_signatures",
    )
    perceptual_hashes = _string_array(
        arrays,
        "source_perceptual_hash",
    )
    for name, values in (
        ("split", split),
        ("content_group_id", groups),
        ("split_version", split_versions),
        ("action_signatures", signatures),
        ("source_perceptual_hash", perceptual_hashes),
    ):
        if len(values) != count:
            raise ValueError(f"{name} row count mismatch")

    if "metadata_json" not in arrays or arrays["metadata_json"].ndim != 0:
        raise ValueError("feature archive metadata_json must be scalar")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("feature archive metadata_json is not an object")
    cluster_config = metadata.get("cluster_config")
    if not isinstance(cluster_config, dict):
        raise ValueError("feature metadata lacks cluster_config")
    archive_data_seed = cluster_config.get("data_seed")
    if (
        isinstance(archive_data_seed, bool)
        or not isinstance(archive_data_seed, int)
        or archive_data_seed != int(data_seed)
    ):
        raise ValueError(
            "requested data_seed disagrees with the committed content split: "
            f"{data_seed} != {archive_data_seed}"
        )

    for index, row in enumerate(rows):
        context = f"{manifest_path}:{index + 1}"
        if int(row.get("feature_index", -1)) != index:
            raise ValueError(f"{context} feature_index is not contiguous")
        checks = (
            ("iid", iids[index]),
            ("split", split[index]),
            ("content_group_id", groups[index]),
            ("split_version", split_versions[index]),
            ("source_perceptual_hash", perceptual_hashes[index]),
        )
        for name, expected in checks:
            if str(row.get(name)) != str(expected):
                raise ValueError(f"{context} {name} disagrees with final.npz")

    versions = sorted(set(split_versions.tolist()))
    if len(versions) != 1:
        raise ValueError(f"feature archive mixes split versions: {versions}")
    split_version = versions[0]
    maximum_hamming = float(
        cluster_config.get("maximum_hamming_fraction", 0.10)
    )
    split_audit = audit_content_disjoint_splits(
        splits=split.tolist(),
        content_group_ids=groups.tolist(),
        split_versions=split_versions.tolist(),
        perceptual_hashes=perceptual_hashes.tolist(),
        maximum_cross_split_hamming_fraction=maximum_hamming,
        require_visual_clusters=(
            split_version == R5_PRODUCTION_SPLIT_VERSION
        ),
    )

    # Adjacent commit markers are mandatory for the canonical final artifact.
    done_path = feature_archive.parent / "done.json"
    summary_path = feature_archive.parent / "summary.json"
    if feature_archive.name == "final.npz" and manifest_path.name == "manifest.jsonl":
        if not done_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                "canonical final.npz requires adjacent done.json and summary.json"
            )
        done = _load_json(done_path)
        summary = _load_json(summary_path)
        digest_checks = {
            "archive_sha256": archive_digest_before,
            "manifest_sha256": manifest_digest_before,
            "summary_sha256": _file_digest(summary_path),
        }
        for key, actual in digest_checks.items():
            if done.get(key) != actual:
                raise ValueError(f"feature done marker {key} mismatch")
        if summary.get("archive_sha256") != archive_digest_before:
            raise ValueError("feature summary archive digest mismatch")
        if summary.get("manifest_sha256") != manifest_digest_before:
            raise ValueError("feature summary manifest digest mismatch")
    else:
        done = {}
        summary = {}

    source_actor = _matrix(arrays, "source_actor", rows=count)
    source_camera = _matrix(arrays, "source_camera", rows=count)
    target_actor = _matrix(arrays, "target_actor", rows=count)
    target_camera = _matrix(arrays, "target_camera", rows=count)
    instruction = _matrix(arrays, "instruction_features", rows=count)
    if source_actor.shape != target_actor.shape:
        raise ValueError("source/target actor endpoint shapes differ")
    if source_camera.shape != target_camera.shape:
        raise ValueError("source/target camera endpoint shapes differ")

    batch = R5EndpointBatch.create(
        iids=tuple(iids.tolist()),
        source_actor=source_actor,
        source_camera=source_camera,
        target_actor=target_actor,
        target_camera=target_camera,
        instruction_features=instruction,
        splits=tuple(split.tolist()),
        content_group_ids=tuple(groups.tolist()),
        action_signatures=tuple(signatures.tolist()),
        split_versions=tuple(split_versions.tolist()),
        perceptual_hashes=tuple(perceptual_hashes.tolist()),
        require_visual_clusters=(
            split_version == R5_PRODUCTION_SPLIT_VERSION
        ),
        maximum_cross_split_hamming_fraction=maximum_hamming,
    )
    provenance = {
        "feature_archive": str(feature_archive),
        "feature_archive_sha256": archive_digest_before,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest_before,
        "metadata": metadata,
        "done": done,
        "summary": summary,
        "content_split_audit": split_audit.to_dict(),
        "split_version": split_version,
        "source_perceptual_hashes": perceptual_hashes.tolist(),
    }
    return batch, rows, provenance


def _validated_human_review(
    row: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any] | None:
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
    if review["schema_version"] != R5_HUMAN_REVIEW_SCHEMA:
        raise ValueError(f"{context} has unsupported human_review schema")
    verdict = review["verdict"]
    if verdict not in HUMAN_VERDICTS:
        raise ValueError(f"{context} has invalid human_review verdict")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError(f"{context} human reviewer is empty")
    if (
        not isinstance(review["label_source_sha256"], str)
        or _SHA256_RE.fullmatch(review["label_source_sha256"]) is None
    ):
        raise ValueError(f"{context} human label_source_sha256 is invalid")
    review_input_digest = review.get("input_digest")
    if (
        review_input_digest is not None
        and review_input_digest != row.get("input_digest")
    ):
        raise ValueError(f"{context} human review input_digest mismatch")
    return review


def _validated_original_qwen(
    row: Mapping[str, Any],
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = row.get("qwen_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{context} lacks qwen_evidence")
    visual = evidence.get("visual")
    if not isinstance(visual, Mapping):
        raise ValueError(f"{context} lacks qwen_evidence.visual")
    if visual.get("status") != "ok":
        raise ValueError(f"{context} Qwen status is not ok")
    if visual.get("iid") != row.get("iid"):
        raise ValueError(f"{context} Qwen iid mismatch")
    if visual.get("input_digest") != row.get("input_digest"):
        raise ValueError(f"{context} Qwen input_digest mismatch")
    if visual.get("observation_validated_from") != "original":
        raise ValueError(f"{context} Qwen observation is not original")
    if visual.get("result_validated_from") != "original":
        raise ValueError(f"{context} Qwen result is not original")
    if visual.get("observation_repairs"):
        raise ValueError(f"{context} Qwen observation has repairs")
    if visual.get("alignment_repairs"):
        raise ValueError(f"{context} Qwen result has repairs")
    observation = visual.get("observation")
    result = visual.get("result")
    if not isinstance(observation, dict) or not isinstance(result, dict):
        raise ValueError(f"{context} Qwen structured objects are missing")
    _validate_observation(observation)
    _validate_visual(result, observation=observation)
    observation_digest = _qwen_object_digest(observation)
    if visual.get("observation_digest") != observation_digest:
        raise ValueError(f"{context} Qwen observation digest mismatch")
    result_digest = visual.get("result_digest")
    if (
        result_digest is not None
        and result_digest != _qwen_object_digest(result)
    ):
        raise ValueError(f"{context} Qwen result digest mismatch")
    return observation, result


def _negative_type(verdict: str) -> str:
    return "endpoint" if verdict == "endpoint_only" else verdict


def _action_family(
    row: Mapping[str, Any],
    *,
    action_signature: str,
    review: Mapping[str, Any] | None,
) -> str:
    # Human semantic fields are authoritative.  Automatic/top-level families
    # must not override a reviewer's event type in a production run.
    sources = (
        (
            review.get("action_family"),
            review.get("event_type"),
        )
        if review is not None
        else (
            (
                row.get("r5_pilot_label", {}).get("action_family")
                if isinstance(row.get("r5_pilot_label"), Mapping)
                else None
            ),
            row.get("action_family"),
        )
    )
    for source in sources:
        if isinstance(source, str) and source.strip():
            return source.strip().lower()
    tokens = _FAMILY_TOKEN_RE.findall(action_signature.lower())
    if not tokens:
        return "unknown"
    # A stable canonical fallback is explicit in the contract.  Formal data
    # should supply reviewer action_family/event_type instead.
    return "_".join(tokens[:2])


def derive_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_mode: str,
) -> list[dict[str, Any]]:
    """Re-derive every usable label from its authoritative evidence."""

    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode={label_mode!r}")
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = f"manifest row {index + 1} iid={row.get('iid')}"
        if label_mode == "human":
            review = _validated_human_review(row, context=context)
            if review is None:
                labels.append(
                    {
                        "label_role": "unlabeled",
                        "label_type": "unlabeled",
                        "action_signature": str(
                            row.get("action_signature") or "unknown"
                        ),
                        "action_family": "unknown",
                        "production_eligible": False,
                        "label_provenance": "no_human_review",
                    }
                )
                continue
            verdict = str(review["verdict"])
            if verdict in POSITIVE_VERDICTS:
                signature = str(review.get("action_signature") or "").strip()
                if not signature or signature.lower() in {"unknown", "unclear"}:
                    raise ValueError(
                        f"{context} human positive requires action_signature"
                    )
                role = "positive_delta"
                label_type = "positive"
            elif verdict in NEGATIVE_AUDIT_VERDICTS:
                signature = f"negative:{_negative_type(verdict)}"
                role = "negative_audit"
                label_type = _negative_type(verdict)
            else:
                signature = str(
                    review.get("action_signature")
                    or row.get("action_signature")
                    or "unknown"
                )
                role = "excluded"
                label_type = verdict
            labels.append(
                {
                    "label_role": role,
                    "label_type": label_type,
                    "action_signature": signature,
                    "action_family": _action_family(
                        row,
                        action_signature=signature,
                        review=review,
                    ),
                    "production_eligible": True,
                    "label_provenance": "human_review",
                }
            )
            continue

        declared_role = str(row.get("label_role") or "")
        pilot_label = row.get("r5_pilot_label")
        if not isinstance(pilot_label, Mapping):
            raise ValueError(f"{context} lacks r5_pilot_label provenance")
        required_pilot_fields = {
            "schema_version",
            "profile",
            "class",
            "negative_type",
            "action_signature",
            "human_approved",
            "production_eligible",
            "legacy_result_digest_missing",
            "source_manifest_sha256",
            "observation_digest",
            "result_object_digest",
        }
        if set(pilot_label) != required_pilot_fields:
            raise ValueError(
                f"{context} r5_pilot_label keys differ: "
                f"{sorted(set(pilot_label) ^ required_pilot_fields)}"
            )
        if (
            pilot_label["schema_version"] != R5_PILOT_SCHEMA
            or pilot_label["profile"] != R5_PILOT_PROFILE
        ):
            raise ValueError(f"{context} has unsupported R5 pilot provenance")
        if (
            pilot_label["human_approved"] is not False
            or pilot_label["production_eligible"] is not False
        ):
            raise ValueError(f"{context} pseudo label claims formal eligibility")
        for digest_name in (
            "source_manifest_sha256",
            "observation_digest",
            "result_object_digest",
        ):
            if (
                not isinstance(pilot_label[digest_name], str)
                or _SHA256_RE.fullmatch(pilot_label[digest_name]) is None
            ):
                raise ValueError(
                    f"{context} pilot {digest_name} is invalid"
                )
        if not isinstance(pilot_label["legacy_result_digest_missing"], bool):
            raise ValueError(
                f"{context} legacy_result_digest_missing must be boolean"
            )
        visual_record = (
            (row.get("qwen_evidence") or {}).get("visual")
            if isinstance(row.get("qwen_evidence"), Mapping)
            else None
        )
        actual_missing_result_digest = (
            not isinstance(visual_record, Mapping)
            or "result_digest" not in visual_record
        )
        if (
            pilot_label["legacy_result_digest_missing"]
            != actual_missing_result_digest
        ):
            raise ValueError(
                f"{context} legacy result-digest declaration mismatch"
            )
        if not declared_role and isinstance(pilot_label, Mapping):
            declared_class = str(pilot_label.get("class") or "")
            declared_role = (
                "positive_delta"
                if declared_class == "positive"
                else "negative_audit"
                if declared_class == "negative"
                else "unlabeled"
            )
        if declared_role not in {"positive_delta", "negative_audit"}:
            labels.append(
                {
                    "label_role": "unlabeled",
                    "label_type": "unlabeled",
                    "action_signature": str(
                        row.get("action_signature") or "unknown"
                    ),
                    "action_family": "unknown",
                    "production_eligible": False,
                    "label_provenance": "strict_qwen_not_selected",
                }
            )
            continue
        observation, result = _validated_original_qwen(row, context=context)
        if pilot_label["observation_digest"] != _qwen_object_digest(observation):
            raise ValueError(f"{context} pilot observation digest mismatch")
        if pilot_label["result_object_digest"] != _qwen_object_digest(result):
            raise ValueError(f"{context} pilot result digest mismatch")
        if (
            observation.get("camera_dominance") != "low"
            or observation.get("background_dominance") != "low"
            or observation.get("artifact_level") != "low"
            or observation.get("preservation_quality") != "acceptable"
        ):
            raise ValueError(f"{context} does not satisfy strict clean quality")
        verdict = str(result["verdict"])
        if declared_role == "positive_delta":
            if verdict not in POSITIVE_VERDICTS:
                raise ValueError(
                    f"{context} pseudo positive has verdict={verdict!r}"
                )
            signature = str(result.get("action_signature") or "").strip()
            if not signature or signature.lower() in {"unknown", "unclear"}:
                raise ValueError(
                    f"{context} pseudo positive requires action_signature"
                )
            if observation.get("target_actor_motion") != "clear":
                raise ValueError(
                    f"{context} pseudo positive target motion is not clear"
                )
            if pilot_label["class"] != "positive":
                raise ValueError(f"{context} pilot class/role mismatch")
            if str(pilot_label["action_signature"]) != signature:
                raise ValueError(f"{context} pilot action signature mismatch")
            if pilot_label["negative_type"] is not None:
                raise ValueError(f"{context} positive has negative_type")
            label_type = "positive"
        else:
            if verdict not in NEGATIVE_AUDIT_VERDICTS:
                raise ValueError(
                    f"{context} pseudo negative has verdict={verdict!r}"
                )
            label_type = _negative_type(verdict)
            declared_type = str(row.get("label_type") or "")
            if declared_type and declared_type not in {
                label_type,
                verdict,
            }:
                raise ValueError(
                    f"{context} pseudo negative type disagrees with Qwen"
                )
            if pilot_label["class"] != "negative":
                raise ValueError(f"{context} pilot class/role mismatch")
            if str(pilot_label["negative_type"]) != verdict:
                raise ValueError(f"{context} pilot negative_type mismatch")
            if str(pilot_label["action_signature"]) != f"negative:{verdict}":
                raise ValueError(f"{context} pilot negative signature mismatch")
            signature = f"negative:{label_type}"
        labels.append(
            {
                "label_role": declared_role,
                "label_type": label_type,
                "action_signature": signature,
                "action_family": _action_family(
                    row,
                    action_signature=signature,
                    review=None,
                ),
                "production_eligible": False,
                "label_provenance": "strict_legacy_qwen_original",
            }
        )
    if label_mode == "strict_legacy_qwen":
        source_digests = {
            str(row["r5_pilot_label"]["source_manifest_sha256"])
            for row, label in zip(rows, labels)
            if label["label_role"] in {"positive_delta", "negative_audit"}
        }
        if len(source_digests) != 1:
            raise ValueError(
                "strict R5 pilot mixes source_manifest_sha256 provenance"
            )
    return labels


def _subset_batch(batch: R5EndpointBatch, indices: np.ndarray) -> R5EndpointBatch:
    selected = np.asarray(indices, dtype=np.int64)
    return R5EndpointBatch(
        iids=tuple(batch.iids[int(index)] for index in selected),
        source_actor=batch.source_actor[selected],
        source_camera=batch.source_camera[selected],
        target_actor=batch.target_actor[selected],
        target_camera=batch.target_camera[selected],
        instruction_features=batch.instruction_features[selected],
        splits=tuple(batch.splits[int(index)] for index in selected),
        content_group_ids=tuple(
            batch.content_group_ids[int(index)] for index in selected
        ),
        action_signatures=tuple(
            batch.action_signatures[int(index)] for index in selected
        ),
        split_versions=tuple(
            batch.split_versions[int(index)] for index in selected
        ),
        perceptual_hashes=(
            None
            if batch.perceptual_hashes is None
            else tuple(batch.perceptual_hashes[int(index)] for index in selected)
        ),
        maximum_cross_split_hamming_fraction=(
            batch.maximum_cross_split_hamming_fraction
        ),
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to train R5-lite; install motive[action-repr]"
        ) from error
    return torch


def _factorized_from_prediction(prediction: Mapping[str, Any]) -> FactorizedR5Targets:
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
    model: SourceAwareFactorizedR5,
    *,
    source_actor: np.ndarray,
    source_camera: np.ndarray,
    instruction: np.ndarray,
    reference_actor: np.ndarray | None,
    reference_camera: np.ndarray | None,
    reference_mask: np.ndarray | None,
    device: Any,
    batch_size: int,
) -> FactorizedR5Targets:
    torch = _torch()
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    parts: list[FactorizedR5Targets] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(source_actor), batch_size):
            stop = min(start + batch_size, len(source_actor))

            def tensor(values: np.ndarray) -> Any:
                return torch.as_tensor(
                    values[start:stop],
                    dtype=torch.float32,
                    device=device,
                )

            kwargs: dict[str, Any] = {
                "source_actor": tensor(source_actor),
                "source_camera": tensor(source_camera),
                "instruction_features": tensor(instruction),
            }
            if reference_actor is not None:
                assert reference_camera is not None
                kwargs["reference_actor"] = tensor(reference_actor)
                kwargs["reference_camera"] = tensor(reference_camera)
                if reference_mask is not None:
                    kwargs["reference_mask"] = tensor(
                        np.asarray(reference_mask, dtype=np.float32).reshape(-1, 1)
                    )
            parts.append(_factorized_from_prediction(model(**kwargs)))
    return FactorizedR5Targets(
        actor_direction=np.concatenate(
            [part.actor_direction for part in parts],
            axis=0,
        ),
        actor_log_magnitude=np.concatenate(
            [part.actor_log_magnitude for part in parts],
            axis=0,
        ),
        camera_direction=np.concatenate(
            [part.camera_direction for part in parts],
            axis=0,
        ),
        camera_log_magnitude=np.concatenate(
            [part.camera_log_magnitude for part in parts],
            axis=0,
        ),
    )


def _train_one_seed(
    *,
    batch: R5EndpointBatch,
    positive_train_indices: np.ndarray,
    transform: R5FeatureTransform,
    model_seed: int,
    hidden_dim: int,
    condition_dim: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    reference_dropout: float,
    magnitude_weight: float,
    device: Any,
    log_every: int,
) -> tuple[SourceAwareFactorizedR5, list[dict[str, Any]]]:
    torch = _torch()
    if not 0.0 <= reference_dropout <= 1.0:
        raise ValueError("reference_dropout must be in [0,1]")
    torch.manual_seed(int(model_seed))
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.manual_seed_all(int(model_seed))
    model = SourceAwareFactorizedR5(
        actor_state_dim=batch.source_actor.shape[1],
        camera_state_dim=batch.source_camera.shape[1],
        instruction_dim=batch.instruction_features.shape[1],
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    source_actor, source_camera, instruction = transform.source_inputs(batch)
    reference_actor, reference_camera = transform.reference_inputs(batch)
    targets = transform.targets(batch)
    train = np.asarray(positive_train_indices, dtype=np.int64)
    if len(train) < 2:
        raise ValueError("R5 training needs at least two positive train rows")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(model_seed))
    logs: list[dict[str, Any]] = []
    model.train()
    for step in range(1, steps + 1):
        selected_position = torch.randint(
            0,
            len(train),
            (batch_size,),
            generator=generator,
        ).numpy()
        selected = train[selected_position]
        reference_mask = (
            torch.rand(
                (batch_size, 1),
                generator=generator,
                dtype=torch.float32,
            )
            >= float(reference_dropout)
        ).float()

        def tensor(values: np.ndarray) -> Any:
            return torch.as_tensor(
                values[selected],
                dtype=torch.float32,
                device=device,
            )

        prediction = model(
            source_actor=tensor(source_actor),
            source_camera=tensor(source_camera),
            instruction_features=tensor(instruction),
            reference_actor=tensor(reference_actor),
            reference_camera=tensor(reference_camera),
            reference_mask=reference_mask.to(device),
        )
        target = {
            "actor_direction": tensor(targets.actor_direction),
            "actor_log_magnitude": tensor(
                targets.actor_log_magnitude.reshape(-1, 1)
            ),
            "camera_direction": tensor(targets.camera_direction),
            "camera_log_magnitude": tensor(
                targets.camera_log_magnitude.reshape(-1, 1)
            ),
        }
        losses = factorized_r5_loss(
            prediction,
            target,
            magnitude_weight=magnitude_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            logs.append(
                {
                    "step": step,
                    "model_seed": int(model_seed),
                    "loss": float(losses["loss"].detach().cpu()),
                    "actor_direction_loss": float(
                        losses["actor_direction"].detach().cpu()
                    ),
                    "actor_magnitude_loss": float(
                        losses["actor_magnitude"].detach().cpu()
                    ),
                    "camera_direction_loss": float(
                        losses["camera_direction"].detach().cpu()
                    ),
                    "camera_magnitude_loss": float(
                        losses["camera_magnitude"].detach().cpu()
                    ),
                    "reference_fraction": float(reference_mask.mean()),
                    "gradient_norm": float(
                        torch.as_tensor(gradient_norm).detach().cpu()
                    ),
                }
            )
    return model, logs


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


def _row_metrics(
    *,
    prediction: FactorizedR5Targets,
    target: FactorizedR5Targets,
    batch: R5EndpointBatch,
    labels: Sequence[Mapping[str, Any]],
    arm: str,
    data_seed: int,
    model_seed: int,
    valid_mask: np.ndarray,
    active_threshold: float,
) -> list[dict[str, Any]]:
    rows = len(batch.iids)
    if valid_mask.shape != (rows,):
        raise ValueError(f"{arm} valid mask shape mismatch")
    actor_cosine = np.sum(
        prediction.actor_direction * target.actor_direction,
        axis=1,
    )
    camera_cosine = np.sum(
        prediction.camera_direction * target.camera_direction,
        axis=1,
    )
    actor_prediction_magnitude = np.asarray(
        prediction.actor_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    camera_prediction_magnitude = np.asarray(
        prediction.camera_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    actor_target_magnitude = np.asarray(
        target.actor_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    camera_target_magnitude = np.asarray(
        target.camera_log_magnitude,
        dtype=np.float32,
    ).reshape(-1)
    output: list[dict[str, Any]] = []
    for index in range(rows):
        label = labels[index]
        positive = label["label_role"] == "positive_delta"
        actor_active = bool(
            positive and actor_target_magnitude[index] > active_threshold
        )
        camera_active = bool(
            positive and camera_target_magnitude[index] > active_threshold
        )
        control_valid = bool(valid_mask[index])
        output.append(
            {
                "schema_version": R5_QUERY_SCHEMA,
                "iid": batch.iids[index],
                "split": batch.splits[index],
                "content_group_id": batch.content_group_ids[index],
                "split_version": batch.split_versions[index],
                "action_signature": label["action_signature"],
                "action_family": label["action_family"],
                "label_role": label["label_role"],
                "label_type": label["label_type"],
                "label_provenance": label["label_provenance"],
                "production_eligible": bool(label["production_eligible"]),
                "arm": arm,
                "data_seed": int(data_seed),
                "model_seed": int(model_seed),
                "control_valid": control_valid,
                "actor_target_active": actor_active,
                "actor_direction_cosine": (
                    float(actor_cosine[index])
                    if positive and actor_active and control_valid
                    else None
                ),
                "actor_predicted_log_magnitude": float(
                    actor_prediction_magnitude[index]
                ),
                "actor_target_log_magnitude": (
                    float(actor_target_magnitude[index]) if positive else None
                ),
                "actor_log_magnitude_absolute_error": (
                    float(
                        abs(
                            actor_prediction_magnitude[index]
                            - actor_target_magnitude[index]
                        )
                    )
                    if positive and control_valid
                    else None
                ),
                "camera_target_active": camera_active,
                "camera_direction_cosine": (
                    float(camera_cosine[index])
                    if positive and camera_active and control_valid
                    else None
                ),
                "camera_predicted_log_magnitude": float(
                    camera_prediction_magnitude[index]
                ),
                "camera_target_log_magnitude": (
                    float(camera_target_magnitude[index]) if positive else None
                ),
                "camera_log_magnitude_absolute_error": (
                    float(
                        abs(
                            camera_prediction_magnitude[index]
                            - camera_target_magnitude[index]
                        )
                    )
                    if positive and control_valid
                    else None
                ),
                "retrieval_valid": False,
                "retrieval_candidates": 0,
                "retrieval_positives": 0,
                "actor_cross_content_ap": None,
                "actor_cross_content_r1": None,
                "actor_cross_content_r5": None,
            }
        )

    for split in ("train", "validation", "test"):
        selected = np.asarray(
            [
                index
                for index in range(rows)
                if batch.splits[index] == split
                and labels[index]["label_role"] == "positive_delta"
            ],
            dtype=np.int64,
        )
        if not len(selected):
            continue
        retrieval = cross_content_retrieval(
            predicted_actor_direction=prediction.actor_direction[selected],
            target_actor_direction=target.actor_direction[selected],
            action_families=[
                str(labels[int(index)]["action_family"]) for index in selected
            ],
            content_group_ids=[
                batch.content_group_ids[int(index)] for index in selected
            ],
            iids=[batch.iids[int(index)] for index in selected],
            valid_mask=valid_mask[selected],
            active_mask=actor_target_magnitude[selected] > active_threshold,
        )
        for local_index, global_index in enumerate(selected):
            output[int(global_index)].update(retrieval[local_index])
    return output


def _seed_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    activation_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    arms = sorted({str(row["arm"]) for row in rows})
    for arm in arms:
        result[arm] = {}
        for split in ("train", "validation", "test"):
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["split"] == split
            ]
            result[arm][split] = summarize_arm_rows(
                selected,
                activation_threshold=activation_threshold,
            )
    return result


def _parse_model_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "model seeds must be comma-separated integers"
        ) from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("model seeds must be non-empty and unique")
    return seeds


def _dataset_contract(
    *,
    batch: R5EndpointBatch,
    labels: Sequence[Mapping[str, Any]],
    label_mode: str,
    split_version: str,
    human_label_source_verified: bool,
) -> dict[str, Any]:
    positive = [
        index
        for index, label in enumerate(labels)
        if label["label_role"] == "positive_delta"
    ]
    negative = [
        index
        for index, label in enumerate(labels)
        if label["label_role"] == "negative_audit"
    ]
    positive_groups = {
        batch.content_group_ids[index] for index in positive
    }
    test_groups = {
        batch.content_group_ids[index]
        for index in positive
        if batch.splits[index] == "test"
    }
    families = {
        str(labels[index]["action_family"])
        for index in positive
        if str(labels[index]["action_family"]) != "unknown"
    }
    return {
        "label_mode": label_mode,
        "production_eligible": bool(
            label_mode == "human"
            and split_version == R5_PRODUCTION_SPLIT_VERSION
            and human_label_source_verified
        ),
        "human_label_source_verified": bool(human_label_source_verified),
        "split_version": split_version,
        "positive_count": len(positive),
        "positive_group_count": len(positive_groups),
        "action_family_count": len(families),
        "test_positive_group_count": len(test_groups),
        "negative_audit_count": len(negative),
        "split_positive_counts": dict(
            Counter(batch.splits[index] for index in positive)
        ),
        "negative_type_counts": dict(
            Counter(str(labels[index]["label_type"]) for index in negative)
        ),
        "label_provenance_counts": dict(
            Counter(str(label["label_provenance"]) for label in labels)
        ),
    }


def _build_arm_predictions(
    *,
    model: SourceAwareFactorizedR5,
    random_model: SourceAwareFactorizedR5,
    transform: R5FeatureTransform,
    centroid: FactorizedCentroidControl,
    batch: R5EndpointBatch,
    labels: Sequence[Mapping[str, Any]],
    data_seed: int,
    device: Any,
    eval_batch_size: int,
) -> dict[str, tuple[FactorizedR5Targets, np.ndarray]]:
    source_actor, source_camera, instruction = transform.source_inputs(batch)
    reference_actor, reference_camera = transform.reference_inputs(batch)
    signatures = [str(label["action_signature"]) for label in labels]
    source_indices, source_valid = source_shuffled_indices(
        splits=batch.splits,
        action_signatures=signatures,
        content_group_ids=batch.content_group_ids,
        data_seed=int(data_seed),
    )
    prompt_indices, prompt_valid = prompt_shuffled_indices(
        splits=batch.splits,
        action_signatures=signatures,
        content_group_ids=batch.content_group_ids,
        data_seed=int(data_seed) + 1,
    )
    pair_indices, pair_valid = prompt_shuffled_indices(
        splits=batch.splits,
        action_signatures=signatures,
        content_group_ids=batch.content_group_ids,
        data_seed=int(data_seed) + 2,
    )
    ones = np.ones(len(batch.iids), dtype=bool)
    return {
        "full": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                instruction=instruction,
                reference_actor=reference_actor,
                reference_camera=reference_camera,
                reference_mask=np.ones(len(batch.iids), dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
        "text_only": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                instruction=instruction,
                reference_actor=None,
                reference_camera=None,
                reference_mask=None,
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
        "pairshuffle": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                instruction=instruction,
                reference_actor=reference_actor[pair_indices],
                reference_camera=reference_camera[pair_indices],
                reference_mask=np.ones(len(batch.iids), dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            pair_valid,
        ),
        "matched_random": (
            _predict(
                random_model,
                source_actor=source_actor,
                source_camera=source_camera,
                instruction=instruction,
                reference_actor=reference_actor,
                reference_camera=reference_camera,
                reference_mask=np.ones(len(batch.iids), dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            ones,
        ),
        "centroid": (
            centroid.predict(signatures),
            ones,
        ),
        "source_shuffle": (
            _predict(
                model,
                source_actor=source_actor[source_indices],
                source_camera=source_camera[source_indices],
                instruction=instruction,
                reference_actor=reference_actor,
                reference_camera=reference_camera,
                reference_mask=np.ones(len(batch.iids), dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            source_valid,
        ),
        "prompt_shuffle": (
            _predict(
                model,
                source_actor=source_actor,
                source_camera=source_camera,
                instruction=instruction[prompt_indices],
                reference_actor=reference_actor,
                reference_camera=reference_camera,
                reference_mask=np.ones(len(batch.iids), dtype=np.float32),
                device=device,
                batch_size=eval_batch_size,
            ),
            prompt_valid,
        ),
    }


def train_and_evaluate(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser()
    output_paths = {
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
            "R5 output already contains committed paths; use a fresh directory: "
            f"{existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = _parse_model_seeds(args.model_seeds)
    batch, manifest_rows, provenance = load_feature_bundle(
        feature_archive=args.features,
        manifest_path=args.manifest,
        data_seed=int(args.data_seed),
    )
    labels = derive_labels(manifest_rows, label_mode=args.label_mode)
    human_label_source: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "verified": False,
    }
    if args.label_mode == "human" and args.human_label_source is not None:
        label_path = args.human_label_source.expanduser().resolve(strict=True)
        label_digest = _file_digest(label_path)
        declared_digests = {
            str(row["human_review"]["label_source_sha256"])
            for row, label in zip(manifest_rows, labels)
            if label["label_provenance"] == "human_review"
        }
        if declared_digests != {label_digest}:
            raise ValueError(
                "human review label_source_sha256 is not bound to "
                f"--human-label-source: {sorted(declared_digests)}"
            )
        human_label_source = {
            "path": str(label_path),
            "sha256": label_digest,
            "verified": True,
        }
    positive_indices = np.asarray(
        [
            index
            for index, label in enumerate(labels)
            if label["label_role"] == "positive_delta"
        ],
        dtype=np.int64,
    )
    positive_train = np.asarray(
        [
            index
            for index in positive_indices
            if batch.splits[int(index)] == "train"
        ],
        dtype=np.int64,
    )
    positive_validation = [
        index
        for index in positive_indices
        if batch.splits[int(index)] == "validation"
    ]
    positive_test = [
        index
        for index in positive_indices
        if batch.splits[int(index)] == "test"
    ]
    if len(positive_train) < 2:
        raise ValueError("R5-lite requires at least two positive train rows")
    if not positive_validation or not positive_test:
        raise ValueError("R5-lite requires positive validation and test rows")

    label_signatures = tuple(
        str(label["action_signature"]) for label in labels
    )
    labeled_batch = R5EndpointBatch(
        iids=batch.iids,
        source_actor=batch.source_actor,
        source_camera=batch.source_camera,
        target_actor=batch.target_actor,
        target_camera=batch.target_camera,
        instruction_features=batch.instruction_features,
        splits=batch.splits,
        content_group_ids=batch.content_group_ids,
        action_signatures=label_signatures,
        split_versions=batch.split_versions,
        perceptual_hashes=batch.perceptual_hashes,
        maximum_cross_split_hamming_fraction=(
            batch.maximum_cross_split_hamming_fraction
        ),
    )
    positive_batch = _subset_batch(labeled_batch, positive_indices)
    transform = R5FeatureTransform.fit(
        positive_batch,
        condition_dim=int(args.condition_dim),
    )
    targets = transform.targets(labeled_batch)
    centroid = FactorizedCentroidControl.fit(
        targets=targets,
        action_signatures=label_signatures,
        train_indices=positive_train,
    )
    train_actor_magnitudes = targets.actor_log_magnitude[positive_train]
    active_train_magnitudes = train_actor_magnitudes[
        train_actor_magnitudes > float(args.active_threshold)
    ]
    if args.activation_threshold is None:
        if not len(active_train_magnitudes):
            raise ValueError(
                "positive train set has no active actor magnitude for threshold fit"
            )
        activation_threshold = max(
            float(args.active_threshold),
            0.5 * float(np.quantile(active_train_magnitudes, 0.1)),
        )
        activation_threshold_origin = "train-positive-q10-times-0.5"
    else:
        activation_threshold = float(args.activation_threshold)
        if not math.isfinite(activation_threshold) or activation_threshold < 0.0:
            raise ValueError("--activation-threshold must be finite and non-negative")
        activation_threshold_origin = "explicit-cli"

    split_version = provenance["split_version"]
    dataset = _dataset_contract(
        batch=labeled_batch,
        labels=labels,
        label_mode=args.label_mode,
        split_version=split_version,
        human_label_source_verified=bool(
            human_label_source["verified"]
        ),
    )
    contract = {
        "schema_version": R5_TRAIN_SCHEMA,
        "data_seed": int(args.data_seed),
        "model_seeds": model_seeds,
        "dataset": dataset,
        "feature_provenance": {
            key: provenance[key]
            for key in (
                "feature_archive",
                "feature_archive_sha256",
                "manifest",
                "manifest_sha256",
                "content_split_audit",
                "split_version",
            )
        },
        "human_label_source": human_label_source,
        "transform": {
            "schema_version": transform.schema_version,
            "digest": transform.digest(),
            "fit_scope": "positive_train_only",
            "train_iid_digest": transform.train_iid_digest,
        },
        "centroid": {
            "fit_scope": "positive_train_only",
            "train_samples": centroid.train_samples,
        },
        "activation_threshold": {
            "value": activation_threshold,
            "origin": activation_threshold_origin,
            "fit_scope": "positive_train_only",
        },
        "arms": {
            "full": "trained model with paired target-reference endpoints",
            "text_only": "same trained model with reference mask zero",
            "pairshuffle": "full model with cross-content/different-family reference",
            "matched_random": "untrained identical architecture, paired reference",
            "centroid": "train-only action-signature centroid",
            "source_shuffle": "full-model source intervention within signature/split",
            "prompt_shuffle": "full-model prompt intervention across signatures",
        },
        "formal_auxiliary_checks_complete": False,
        "formal_auxiliary_checks": {
            "direction_probe": False,
            "speed_probe": False,
            "phase_probe": False,
            "camera_leakage": False,
            "stability": False,
            "pair_specificity": False,
        },
        "formal_auxiliary_note": (
            "R5-lite reports factor metrics and interventions, but the "
            "pre-registered labeled direction/speed/phase probes, camera "
            "leakage, stability and pair-specificity audits are not yet "
            "complete; production PASS is therefore disabled."
        ),
        "claim_scope": {
            "representation": "reference-conditioned R5-lite diagnostic",
            "action_editing": False,
            "i2v": False,
            "note": (
                "The full arm receives exact target-reference endpoints; "
                "passing this representation gate would not by itself prove "
                "text-only action editing or I2V control."
            ),
        },
        "training": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "condition_dim": int(args.condition_dim),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "reference_dropout": float(args.reference_dropout),
            "magnitude_weight": float(args.magnitude_weight),
            "active_threshold": float(args.active_threshold),
            "negative_rows_in_loss": 0,
        },
    }
    _atomic_json(output_paths["contract"], contract)
    _atomic_json(output_paths["transform"], asdict(transform))

    torch = _torch()
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested_device}")
    device = torch.device(requested_device)
    all_query_rows: list[dict[str, Any]] = []
    per_seed_summaries: dict[str, Any] = {}
    for model_seed in model_seeds:
        seed_dir = output_dir / "seeds" / f"seed-{model_seed}"
        model, logs = _train_one_seed(
            batch=labeled_batch,
            positive_train_indices=positive_train,
            transform=transform,
            model_seed=model_seed,
            hidden_dim=int(args.hidden_dim),
            condition_dim=int(args.condition_dim),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            reference_dropout=float(args.reference_dropout),
            magnitude_weight=float(args.magnitude_weight),
            device=device,
            log_every=int(args.log_every),
        )
        random_seed = int(model_seed) + 1_000_003
        random_model = make_matched_random_control(
            model,
            model_seed=random_seed,
        ).to(device)
        checkpoint = seed_dir / "model.pt"
        random_checkpoint = seed_dir / "matched_random.pt"
        _atomic_torch_save(
            checkpoint,
            {
                "schema_version": R5_TRAIN_SCHEMA,
                "arm": "trained",
                "model_seed": int(model_seed),
                "data_seed": int(args.data_seed),
                "architecture": {
                    "actor_state_dim": model.actor_state_dim,
                    "camera_state_dim": model.camera_state_dim,
                    "instruction_dim": model.instruction_dim,
                    "condition_dim": model.condition_dim,
                    "hidden_dim": model.hidden_dim,
                },
                "state_dict": model.state_dict(),
                "transform": asdict(transform),
                "transform_digest": transform.digest(),
                "feature_archive_sha256": provenance["feature_archive_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
            },
        )
        _atomic_torch_save(
            random_checkpoint,
            {
                "schema_version": R5_TRAIN_SCHEMA,
                "arm": "matched_random",
                "model_seed": random_seed,
                "paired_trained_model_seed": int(model_seed),
                "state_dict": random_model.state_dict(),
                "transform_digest": transform.digest(),
            },
        )
        _atomic_jsonl(seed_dir / "train_log.jsonl", logs)
        arm_predictions = _build_arm_predictions(
            model=model,
            random_model=random_model,
            transform=transform,
            centroid=centroid,
            batch=labeled_batch,
            labels=labels,
            data_seed=int(args.data_seed),
            device=device,
            eval_batch_size=int(args.eval_batch_size),
        )
        seed_rows: list[dict[str, Any]] = []
        for arm in (
            "full",
            "text_only",
            "pairshuffle",
            "matched_random",
            "centroid",
            "source_shuffle",
            "prompt_shuffle",
        ):
            prediction, valid = arm_predictions[arm]
            seed_rows.extend(
                _row_metrics(
                    prediction=prediction,
                    target=targets,
                    batch=labeled_batch,
                    labels=labels,
                    arm=arm,
                    data_seed=int(args.data_seed),
                    model_seed=int(model_seed),
                    valid_mask=np.asarray(valid, dtype=bool),
                    active_threshold=float(args.active_threshold),
                )
            )
        metrics = {
            "schema_version": R5_TRAIN_SCHEMA,
            "model_seed": int(model_seed),
            "data_seed": int(args.data_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_digest(checkpoint),
            "matched_random_checkpoint": str(random_checkpoint),
            "matched_random_checkpoint_sha256": _file_digest(random_checkpoint),
            "train_final": logs[-1],
            "arms": _seed_metrics(
                seed_rows,
                activation_threshold=activation_threshold,
            ),
        }
        _atomic_json(seed_dir / "metrics.json", metrics)
        per_seed_summaries[str(model_seed)] = metrics
        all_query_rows.extend(seed_rows)
        print(
            f"[r5-train] seed={model_seed} loss={logs[-1]['loss']:.6f} "
            f"rows={len(seed_rows)}",
            flush=True,
        )

    all_query_rows.sort(
        key=lambda row: (
            int(row["model_seed"]),
            str(row["arm"]),
            str(row["split"]),
            str(row["iid"]),
        )
    )
    _atomic_jsonl(output_paths["per_query"], all_query_rows)
    thresholds = R5GateThresholds(
        bootstrap_samples=int(args.bootstrap_samples),
        signflip_samples=int(args.signflip_samples),
    )
    gate = evaluate_r5_gate(
        rows=all_query_rows,
        contract=contract,
        thresholds=thresholds,
    )
    _atomic_json(output_paths["gate"], gate)
    summary = {
        "schema_version": R5_TRAIN_SCHEMA,
        "status": "complete",
        "gate_status": gate["status"],
        "production_decision": gate["production_decision"],
        "contract": str(output_paths["contract"]),
        "contract_sha256": _file_digest(output_paths["contract"]),
        "per_query": str(output_paths["per_query"]),
        "per_query_sha256": _file_digest(output_paths["per_query"]),
        "per_query_rows": len(all_query_rows),
        "gate": str(output_paths["gate"]),
        "gate_sha256": _file_digest(output_paths["gate"]),
        "per_seed": per_seed_summaries,
    }
    _atomic_json(output_paths["summary"], summary)
    done = {
        "schema_version": R5_TRAIN_SCHEMA,
        "status": "complete",
        "gate_status": gate["status"],
        "contract_sha256": summary["contract_sha256"],
        "per_query_sha256": summary["per_query_sha256"],
        "gate_sha256": summary["gate_sha256"],
        "summary_sha256": _file_digest(output_paths["summary"]),
        "model_seeds": model_seeds,
    }
    _atomic_json(output_paths["done"], done)
    print(
        f"[r5-train] complete gate={gate['status']} output={output_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--label-mode",
        choices=sorted(LABEL_MODES),
        required=True,
    )
    parser.add_argument(
        "--human-label-source",
        type=Path,
        help=(
            "Original human label JSONL whose SHA256 must match every merged "
            "human_review.label_source_sha256; required for production eligibility"
        ),
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
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument("--magnitude-weight", type=float, default=0.25)
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--activation-threshold", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--signflip-samples", type=int, default=50_000)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.label_mode != "human"
        and args.human_label_source is not None
    ):
        raise ValueError(
            "--human-label-source is only valid with --label-mode human"
        )
    integer_positive = (
        "steps",
        "batch_size",
        "eval_batch_size",
        "hidden_dim",
        "condition_dim",
        "log_every",
    )
    for name in integer_positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.bootstrap_samples) < 0 or int(args.signflip_samples) < 0:
        raise ValueError("resampling counts must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("--reference-dropout must be in [0,1]")
    if not math.isfinite(args.magnitude_weight) or args.magnitude_weight < 0.0:
        raise ValueError("--magnitude-weight must be finite and non-negative")
    if not math.isfinite(args.active_threshold) or args.active_threshold < 0.0:
        raise ValueError("--active-threshold must be finite and non-negative")


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
            _atomic_json(invalid_path, invalid_gate_summary(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
