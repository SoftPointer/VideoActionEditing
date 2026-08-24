#!/usr/bin/env python3
"""Audit and summarize the two-seed Bernini FITQ engineering scan.

This program consumes only the read-only phase/head sufficient statistics
emitted by :mod:`infer_fitq_official_runtime_scan`.  It verifies every receipt
and artifact before computing pre-registered diagnostic summaries.  The A0
scan has one proposal family, two seeds, global phase means, and only two hard
negatives.  Consequently this analyzer can shortlist sites for A1, but it is
structurally unable to authorize FITQ, an optimizer, or a scientific claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import tempfile
import os
from typing import Any, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import internal_temporal_quotient as fitq  # noqa: E402
import internal_temporal_quotient_observer as observer  # noqa: E402
import infer_fitq_official_runtime_scan as runtime  # noqa: E402


SCHEMA_VERSION = "bernini-fitq-engineering-a0-analysis-v1"
EXPECTED_SIGMAS = (0.80, 0.60, 0.35, 0.15)
EXPECTED_RUNTIME_FP32_SIGMAS = tuple(
    struct.unpack("<f", struct.pack("<f", value))[0]
    for value in EXPECTED_SIGMAS
)
EXPECTED_LAMBDAS = (1.0, 0.5, 0.0)
EXPECTED_BRANCHES = (
    "frozen_t2v_action",
    "frozen_t2v_hard_negative[0]",
    "frozen_t2v_hard_negative[1]",
    "frozen_identity_noop_correct",
    "frozen_identity_noop_wrong_source[0]",
    "frozen_identity_action_correct",
    "frozen_identity_action_wrong_source[0]",
)
DUPLICATE_BRANCH = "frozen_t2v_action_duplicate"
EXPECTED_MEAN_SHAPE = (121, 1, 21, 1, 1536)
EXPECTED_COUNT_SHAPE = (21, 12)
EXPECTED_COUNT_VALUE = 930.0
MIN_ACTION_TO_DUPLICATE_ENERGY_RATIO = 10.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FITQEngineeringAnalysisError(RuntimeError):
    """Raised before incomplete or mutable A0 evidence is accepted."""


def _reject_constant(value: str) -> None:
    raise FITQEngineeringAnalysisError(f"non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FITQEngineeringAnalysisError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise FITQEngineeringAnalysisError(
            "analysis payload is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: str | Path, *, label: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise FITQEngineeringAnalysisError(f"{label} path must be text")
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise FITQEngineeringAnalysisError(f"{label} must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise FITQEngineeringAnalysisError(f"{label} does not exist") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise FITQEngineeringAnalysisError(f"{label} must be a plain file")
    return resolved


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path]:
    resolved = _plain_file(path, label=label)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FITQEngineeringAnalysisError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise FITQEngineeringAnalysisError(f"{label} root must be an object")
    return value, resolved


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FITQEngineeringAnalysisError(f"{label} must be a lowercase SHA-256")
    return value


def _normalize_runtime_sigma(value: Any) -> float:
    """Map only the exact FP32 runtime coordinates to registered labels."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FITQEngineeringAnalysisError("statistics sigma coordinate differs")
    numeric = float(value)
    for registered, runtime_fp32 in zip(
        EXPECTED_SIGMAS, EXPECTED_RUNTIME_FP32_SIGMAS
    ):
        if numeric == runtime_fp32:
            return registered
    raise FITQEngineeringAnalysisError(
        "statistics sigma is not an exact registered FP32 coordinate"
    )


def validate_runtime_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fail-closed fields needed by the A0 analyzer."""

    if not isinstance(value, Mapping):
        raise FITQEngineeringAnalysisError("runtime receipt must be an object")
    declared = _require_sha256(value.get("receipt_digest"), label="receipt_digest")
    unsigned = dict(value)
    unsigned.pop("receipt_digest", None)
    if object_sha256(unsigned) != declared:
        raise FITQEngineeringAnalysisError("runtime receipt digest differs")
    if value.get("schema_version") != runtime.RECEIPT_SCHEMA:
        raise FITQEngineeringAnalysisError("runtime receipt schema differs")
    required_false = (
        "training_authorized",
        "fitq_stage1_authorized",
        "scientific_claim_authorized",
    )
    if any(value.get(field) is not False for field in required_false):
        raise FITQEngineeringAnalysisError("runtime receipt over-authorizes A0")
    if value.get("optimizer_update") != "null":
        raise FITQEngineeringAnalysisError("A0 optimizer update must be null")
    training = value.get("training")
    if not isinstance(training, Mapping) or any(
        training.get(field) is not expected
        for field, expected in (
            ("forward_only", True),
            ("backward_performed", False),
            ("optimizer_present", False),
            ("checkpoint_saved", False),
            ("model_weights_written", False),
        )
    ):
        raise FITQEngineeringAnalysisError("runtime receipt training closure differs")
    evidence = value.get("fitq_observation")
    if not isinstance(evidence, Mapping):
        raise FITQEngineeringAnalysisError("FITQ observation evidence is absent")
    if (
        evidence.get("statistics_artifact_count") != runtime.EXPECTED_OBSERVED_FORWARDS
        or evidence.get("context_count") != runtime.EXPECTED_OBSERVED_FORWARDS
        or evidence.get("tokenwise_localization_available") is not False
        or evidence.get("fitq_go_authorized") is not False
        or evidence.get("proposal_bank_status") != "insufficient_bank"
    ):
        raise FITQEngineeringAnalysisError("A0 observation scope differs")
    artifacts = evidence.get("statistics_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 85:
        raise FITQEngineeringAnalysisError("A0 artifact list is incomplete")
    return dict(value)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - deployment dependent
        raise FITQEngineeringAnalysisError("A0 analysis requires PyTorch") from error
    return torch


def _load_statistics_payload(
    record: Mapping[str, Any], *, statistics_directory: Path
) -> dict[str, Any]:
    torch = _require_torch()
    path = _plain_file(record.get("path"), label="statistics artifact")
    if path.parent != statistics_directory:
        raise FITQEngineeringAnalysisError("statistics artifact escaped its directory")
    expected_sha = _require_sha256(record.get("sha256"), label="artifact SHA-256")
    if file_sha256(path) != expected_sha:
        raise FITQEngineeringAnalysisError("statistics artifact SHA-256 differs")
    if record.get("size_bytes") != path.stat().st_size:
        raise FITQEngineeringAnalysisError("statistics artifact size differs")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise FITQEngineeringAnalysisError(
            "statistics artifact failed safe tensor-only loading"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "site_order",
        "context",
        "phase_feature_mean_fp32",
        "phase_feature_second_moment_fp32",
        "global_count_fp32_shared_by_all_sites",
    }:
        raise FITQEngineeringAnalysisError("statistics payload fields differ")
    if payload.get("schema_version") != runtime.STATISTICS_SCHEMA:
        raise FITQEngineeringAnalysisError("statistics payload schema differs")
    sites = tuple(payload.get("site_order", ()))
    if sites != observer.expected_site_order():
        raise FITQEngineeringAnalysisError("statistics site order differs")
    if payload.get("context") != record.get("context"):
        raise FITQEngineeringAnalysisError("artifact context differs from receipt")
    mean = payload.get("phase_feature_mean_fp32")
    second = payload.get("phase_feature_second_moment_fp32")
    count = payload.get("global_count_fp32_shared_by_all_sites")
    if any(not isinstance(item, torch.Tensor) for item in (mean, second, count)):
        raise FITQEngineeringAnalysisError("statistics payload lacks tensors")
    if any(item.dtype != torch.float32 for item in (mean, second, count)):
        raise FITQEngineeringAnalysisError("statistics payload dtype differs")
    if tuple(mean.shape) != EXPECTED_MEAN_SHAPE or tuple(second.shape) != EXPECTED_MEAN_SHAPE:
        raise FITQEngineeringAnalysisError("phase feature geometry differs")
    if tuple(count.shape) != EXPECTED_COUNT_SHAPE:
        raise FITQEngineeringAnalysisError("phase/head count geometry differs")
    if any(item.requires_grad or item.grad_fn is not None for item in (mean, second, count)):
        raise FITQEngineeringAnalysisError("statistics payload retained autograd")
    if not all(bool(torch.isfinite(item).all().item()) for item in (mean, second, count)):
        raise FITQEngineeringAnalysisError("statistics payload is non-finite")
    if not bool(torch.equal(count, torch.full_like(count, EXPECTED_COUNT_VALUE))):
        raise FITQEngineeringAnalysisError("global phase/head token counts differ")
    # E[x^2] >= E[x]^2 up to a small FP32 accumulation tolerance.
    violation = mean.square() - second
    tolerance = 2.0e-4 * torch.maximum(second.abs(), torch.ones_like(second))
    if bool((violation > tolerance).any().item()):
        raise FITQEngineeringAnalysisError("second moment is inconsistent with mean")
    return payload


def load_group(receipt_path: str | Path) -> dict[str, Any]:
    value, resolved_receipt = _load_json(receipt_path, label="FITQ runtime receipt")
    receipt = validate_runtime_receipt(value)
    evidence = receipt["fitq_observation"]
    directory_text = evidence.get("statistics_directory")
    if not isinstance(directory_text, str):
        raise FITQEngineeringAnalysisError("statistics directory is absent")
    directory = Path(directory_text).expanduser()
    if directory.is_symlink():
        raise FITQEngineeringAnalysisError("statistics directory must not be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir() or directory != resolved_receipt.parent / "statistics":
        raise FITQEngineeringAnalysisError("statistics directory binding differs")

    payloads: dict[tuple[str, float, float], dict[str, Any]] = {}
    for ordinal, record in enumerate(evidence["statistics_artifacts"]):
        if not isinstance(record, Mapping) or record.get("ordinal") != ordinal:
            raise FITQEngineeringAnalysisError("statistics ordinal closure differs")
        payload = _load_statistics_payload(record, statistics_directory=directory)
        context = payload["context"]
        if not isinstance(context, Mapping):
            raise FITQEngineeringAnalysisError("statistics context must be an object")
        branch = context.get("branch")
        sigma = context.get("sigma")
        lambda_value = context.get("lambda")
        if (
            not isinstance(branch, str)
            or isinstance(lambda_value, bool)
            or not isinstance(lambda_value, (int, float))
        ):
            raise FITQEngineeringAnalysisError("statistics context coordinate differs")
        normalized_sigma = _normalize_runtime_sigma(sigma)
        key = (branch, normalized_sigma, float(lambda_value))
        if key in payloads:
            raise FITQEngineeringAnalysisError("duplicate statistics context")
        payloads[key] = payload

    expected = {
        (branch, sigma, lambda_value)
        for sigma in EXPECTED_SIGMAS
        for lambda_value in EXPECTED_LAMBDAS
        for branch in EXPECTED_BRANCHES
    }
    duplicate_keys = [key for key in payloads if key[0] == DUPLICATE_BRANCH]
    if set(payloads) - set(duplicate_keys) != expected:
        raise FITQEngineeringAnalysisError("statistics grid is not exact S4xL3xB7")
    if duplicate_keys != [(DUPLICATE_BRANCH, 0.8, 1.0)]:
        raise FITQEngineeringAnalysisError("action duplicate coordinate differs")
    return {
        "receipt_path": str(resolved_receipt),
        "receipt_file_sha256": file_sha256(resolved_receipt),
        "receipt_digest": receipt["receipt_digest"],
        "payloads": payloads,
    }


def _temporal_vector(payload: Mapping[str, Any], site_index: int) -> Any:
    mean = payload["phase_feature_mean_fp32"][site_index]
    bundle = fitq.build_temporal_bundle(mean)
    return fitq.weight_temporal_direct_sum(bundle.features).reshape(-1).double()


def _residual(group: Mapping[str, Any], branch_a: str, branch_b: str, sigma: float, lambda_value: float, site_index: int) -> Any:
    payloads = group["payloads"]
    return _temporal_vector(payloads[(branch_a, sigma, lambda_value)], site_index) - _temporal_vector(
        payloads[(branch_b, sigma, lambda_value)], site_index
    )


def _norm(value: Any) -> float:
    return float(value.square().sum().sqrt().item())


def _cosine(left: Any, right: Any) -> Optional[float]:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    result = float((left @ right).item() / (left_norm * right_norm))
    if not math.isfinite(result):
        raise FITQEngineeringAnalysisError("diagnostic cosine is non-finite")
    return max(-1.0, min(1.0, result))


def _finite_values(values: Iterable[Optional[float]]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def _summary(values: Iterable[Optional[float]]) -> dict[str, Any]:
    kept = sorted(_finite_values(values))
    if not kept:
        return {"count": 0, "min": None, "q10": None, "median": None, "max": None}
    q10_index = int(math.floor(0.10 * (len(kept) - 1)))
    return {
        "count": len(kept),
        "min": kept[0],
        "q10": kept[q10_index],
        "median": float(statistics.median(kept)),
        "max": kept[-1],
    }


def _site_diagnostics(groups: Sequence[Mapping[str, Any]], site_index: int) -> dict[str, Any]:
    action_noop_by_group: list[dict[tuple[float, float], Any]] = []
    action_incomplete_by_group: list[dict[tuple[float, float], Any]] = []
    within_hard_negative_cosines: list[Optional[float]] = []
    cross_mode_correct_cosines: list[Optional[float]] = []
    cross_mode_wrong_cosines: list[Optional[float]] = []
    correct_minus_wrong: list[Optional[float]] = []
    action_energies: list[float] = []
    duplicate_energies: list[float] = []
    lambda_continuity: list[Optional[float]] = []

    for group in groups:
        noop_map: dict[tuple[float, float], Any] = {}
        incomplete_map: dict[tuple[float, float], Any] = {}
        for sigma in EXPECTED_SIGMAS:
            for lambda_value in EXPECTED_LAMBDAS:
                action_noop = _residual(
                    group,
                    "frozen_t2v_action",
                    "frozen_t2v_hard_negative[0]",
                    sigma,
                    lambda_value,
                    site_index,
                )
                action_incomplete = _residual(
                    group,
                    "frozen_t2v_action",
                    "frozen_t2v_hard_negative[1]",
                    sigma,
                    lambda_value,
                    site_index,
                )
                correct = _residual(
                    group,
                    "frozen_identity_action_correct",
                    "frozen_identity_noop_correct",
                    sigma,
                    lambda_value,
                    site_index,
                )
                wrong = _residual(
                    group,
                    "frozen_identity_action_wrong_source[0]",
                    "frozen_identity_noop_wrong_source[0]",
                    sigma,
                    lambda_value,
                    site_index,
                )
                noop_map[(sigma, lambda_value)] = action_noop
                incomplete_map[(sigma, lambda_value)] = action_incomplete
                action_energies.append(_norm(action_noop))
                within_hard_negative_cosines.append(
                    _cosine(action_noop, action_incomplete)
                )
                correct_cosine = _cosine(action_noop, correct)
                wrong_cosine = _cosine(action_noop, wrong)
                cross_mode_correct_cosines.append(correct_cosine)
                cross_mode_wrong_cosines.append(wrong_cosine)
                if correct_cosine is not None and wrong_cosine is not None:
                    correct_minus_wrong.append(correct_cosine - wrong_cosine)
                else:
                    correct_minus_wrong.append(None)
            lambda_continuity.extend(
                _cosine(noop_map[(sigma, left)], noop_map[(sigma, right)])
                for left, right in ((1.0, 0.5), (0.5, 0.0))
            )
        action_noop_by_group.append(noop_map)
        action_incomplete_by_group.append(incomplete_map)
        original = _temporal_vector(
            group["payloads"][("frozen_t2v_action", 0.8, 1.0)], site_index
        )
        duplicate = _temporal_vector(
            group["payloads"][(DUPLICATE_BRANCH, 0.8, 1.0)], site_index
        )
        duplicate_energies.append(_norm(original - duplicate))

    seed_action_noop_cosines = [
        _cosine(action_noop_by_group[0][key], action_noop_by_group[1][key])
        for key in sorted(action_noop_by_group[0])
    ]
    seed_action_incomplete_cosines = [
        _cosine(action_incomplete_by_group[0][key], action_incomplete_by_group[1][key])
        for key in sorted(action_incomplete_by_group[0])
    ]
    summaries = {
        "action_noop_energy": _summary(action_energies),
        "duplicate_null_energy": _summary(duplicate_energies),
        "action_noop_vs_action_incomplete_cosine": _summary(
            within_hard_negative_cosines
        ),
        "cross_seed_action_noop_cosine": _summary(seed_action_noop_cosines),
        "cross_seed_action_incomplete_cosine": _summary(
            seed_action_incomplete_cosines
        ),
        "lambda_adjacent_action_noop_cosine": _summary(lambda_continuity),
        "cross_mode_correct_source_cosine": _summary(cross_mode_correct_cosines),
        "cross_mode_wrong_source_cosine": _summary(cross_mode_wrong_cosines),
        "correct_minus_wrong_source_cosine": _summary(correct_minus_wrong),
    }
    action_q10 = summaries["action_noop_energy"]["q10"]
    duplicate_max = summaries["duplicate_null_energy"]["max"]
    if action_q10 is None or duplicate_max is None:
        energy_ratio = None
        energy_gate_pass = False
    elif duplicate_max == 0.0:
        energy_ratio = None
        energy_gate_pass = action_q10 > 0.0
    else:
        energy_ratio = action_q10 / duplicate_max
        energy_gate_pass = energy_ratio >= MIN_ACTION_TO_DUPLICATE_ENERGY_RATIO
    score_components = (
        summaries["action_noop_vs_action_incomplete_cosine"]["q10"],
        summaries["cross_seed_action_noop_cosine"]["q10"],
        summaries["cross_seed_action_incomplete_cosine"]["q10"],
        summaries["lambda_adjacent_action_noop_cosine"]["q10"],
        summaries["cross_mode_correct_source_cosine"]["q10"],
        # A generic text/action code that aligns equally well to an unrelated
        # source is not a useful MV2V write-path hypothesis.  Keep the
        # correct-minus-wrong-source margin inside the conservative minimum
        # used for shortlist ranking rather than merely reporting it later.
        summaries["correct_minus_wrong_source_cosine"]["q10"],
    )
    score = (
        min(score_components)
        if energy_gate_pass and all(value is not None for value in score_components)
        else None
    )
    return {
        "site": observer.expected_site_order()[site_index],
        "engineering_shortlist_score": score,
        "score_is_minimum_of_preregistered_q10_components": True,
        "energy_gate": {
            "minimum_action_q10_to_duplicate_max_ratio": (
                MIN_ACTION_TO_DUPLICATE_ENERGY_RATIO
            ),
            "action_q10_to_duplicate_max_ratio": energy_ratio,
            "duplicate_null_max_is_exact_zero": duplicate_max == 0.0,
            "passed": energy_gate_pass,
        },
        "summaries": summaries,
    }


def _adjacent_bands(site_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_site = {row["site"]: row for row in site_rows}
    rows: list[dict[str, Any]] = []
    for suffix in ("input", "attn1", "attn2", "output"):
        for index in range(29):
            left = by_site[f"block.{index:02d}.{suffix}"]
            right = by_site[f"block.{index + 1:02d}.{suffix}"]
            scores = (
                left["engineering_shortlist_score"],
                right["engineering_shortlist_score"],
            )
            rows.append(
                {
                    "site_type": suffix,
                    "layers": [index, index + 1],
                    "sites": [left["site"], right["site"]],
                    "band_score": min(scores) if all(value is not None for value in scores) else None,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["band_score"] is None,
            -(row["band_score"] if row["band_score"] is not None else -math.inf),
            row["site_type"],
            row["layers"],
        ),
    )


def analyze(receipt_paths: Sequence[str | Path]) -> dict[str, Any]:
    if len(receipt_paths) != 2:
        raise FITQEngineeringAnalysisError("A0 requires exactly two SP4 seed receipts")
    groups = [load_group(path) for path in receipt_paths]
    if groups[0]["receipt_digest"] == groups[1]["receipt_digest"]:
        raise FITQEngineeringAnalysisError("A0 seed receipts must be distinct")
    site_rows = [
        _site_diagnostics(groups, site_index)
        for site_index in range(len(observer.expected_site_order()))
    ]
    ranked_sites = sorted(
        site_rows,
        key=lambda row: (
            row["engineering_shortlist_score"] is None,
            -(
                row["engineering_shortlist_score"]
                if row["engineering_shortlist_score"] is not None
                else -math.inf
            ),
            row["site"],
        ),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "frozen-internal-temporal-quotient-engineering-a0",
        "analysis_implementation": {
            "analyzer_source_sha256": file_sha256(Path(__file__).resolve()),
            "fitq_core_source_sha256": file_sha256(
                Path(fitq.__file__).resolve()
            ),
            "observer_source_sha256": file_sha256(
                Path(observer.__file__).resolve()
            ),
            "runtime_source_sha256": file_sha256(Path(runtime.__file__).resolve()),
            "energy_gate_threshold_preregistered_before_analysis": True,
        },
        "input_receipts": [
            {
                "path": group["receipt_path"],
                "file_sha256": group["receipt_file_sha256"],
                "receipt_digest": group["receipt_digest"],
            }
            for group in groups
        ],
        "grid": {
            "seed_replicates": 2,
            "sigmas": list(EXPECTED_SIGMAS),
            "accepted_exact_runtime_fp32_sigmas": list(
                EXPECTED_RUNTIME_FP32_SIGMAS
            ),
            "lambdas": list(EXPECTED_LAMBDAS),
            "branches": list(EXPECTED_BRANCHES),
            "hook_sites": len(observer.expected_site_order()),
            "phase_descriptor": "global_phase_head_mean",
            "tokenwise_locality_available": False,
        },
        "preregistered_engineering_site_ranking": ranked_sites,
        "preregistered_adjacent_band_ranking": _adjacent_bands(site_rows),
        "interpretation": {
            "engineering_artifact_closure_passed": True,
            "proposal_bank_size": 1,
            "hard_negative_count": 2,
            "scientific_negative_contract_satisfied": False,
            "scientific_nuisance_contract_satisfied": False,
            "scientific_signed_spatial_sketch_available": False,
            "event_verified_discovery_count": None,
            "event_verified_confirmation_count": None,
            "scientific_fitq_outcome": "not_evaluated_insufficient_bank",
            "shortlist_is_hypothesis_generation_only": True,
            "fitq_go_authorized": False,
            "stage1_training_authorized": False,
            "scientific_claim_authorized": False,
            "optimizer_update": "null",
        },
    }
    result["analysis_digest"] = object_sha256(result)
    return result


def _new_output(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise FITQEngineeringAnalysisError("output must be absolute and non-root")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise FITQEngineeringAnalysisError("output parent must be a plain directory")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise FITQEngineeringAnalysisError("refusing to overwrite analysis output")
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-a-receipt", required=True)
    parser.add_argument("--seed-b-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = _new_output(args.output)
    result = analyze((args.seed_a_receipt, args.seed_b_receipt))
    _write_json(output, result)
    print(json.dumps({"output": str(output), "analysis_digest": result["analysis_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
