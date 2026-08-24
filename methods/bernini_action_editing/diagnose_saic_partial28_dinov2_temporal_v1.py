#!/usr/bin/env python3
"""Diagnostic-only frozen-DINO measurements for an incomplete SAIC T2V bank.

The program authenticates exactly 28 completed v2 top-up generation receipts,
decodes every bound MP4 as exact81/25fps RGB, and measures same-video temporal
appearance stability with a sealed, local DINOv2 evaluator.  Eight independent
workers own a deterministic modulo partition and each load one frozen model on
one GPU.  Results are raw technical/proxy evidence only: this program cannot
verify identity, an event, scientific success, or authorize selection/training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generate_saic_pure_t2v_event_bank_topup_v2 as topup_generate  # noqa: E402


SCHEMA_VERSION = "bernini-saic-partial28-dinov2-temporal-diagnostic-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
ATTEMPT_BASENAME = topup_generate.ATTEMPT_RECEIPT_BASENAME
EXPECTED_ATTEMPT_COUNT = 28
EXPECTED_WORLD_SIZE = 8
EVAL_FRAME_INDICES = tuple(range(0, 81, 5))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

AUTHORITY_CLOSURE = {
    "diagnostic_only": True,
    "raw_proxy_evidence_only": True,
    "identity_authority": False,
    "identity_preservation_verified": False,
    "event_authority": False,
    "event_verified": False,
    "scientific_claim_authorized": False,
    "selection_authorized": False,
    "ranking_authorized": False,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
}


class Partial28DINOError(RuntimeError):
    """Raised before unauthenticated or over-claimed evidence is emitted."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Partial28DINOError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise Partial28DINOError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise Partial28DINOError(f"{label} must be an absolute plain directory")
    return path.resolve(strict=True)


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise Partial28DINOError(f"{label} field closure differs")
    return value


def _strict_json(path: str | Path, *, expected_sha256: str | None, label: str) -> tuple[dict[str, Any], str]:
    source = _plain_file(path, label=label)
    before = source.stat()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(expected_sha256, label=f"{label} expected hash"):
        raise Partial28DINOError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> None:
        raise Partial28DINOError(f"{label} contains {token}")

    def reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Partial28DINOError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant,
            object_pairs_hook=reject_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Partial28DINOError(f"{label} is invalid JSON") from error
    after = source.stat()
    if type(value) is not dict or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or file_sha256(source) != digest:
        raise Partial28DINOError(f"{label} changed while reading")
    return value, digest


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.parent.is_symlink() or not path.parent.is_dir():
        raise Partial28DINOError("receipt parent must be one existing absolute directory")
    raw = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise Partial28DINOError(f"refusing to overwrite {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return hashlib.sha256(raw).hexdigest()


def _verify_self(expected_sha256: str) -> str:
    actual = file_sha256(Path(__file__).resolve())
    if actual != _sha256(expected_sha256, label="diagnostic source SHA-256"):
        raise Partial28DINOError("diagnostic source SHA-256 differs")
    return actual


def _stable_bound_file(path_value: Any, hash_value: Any, *, label: str, inside: Path | None = None) -> tuple[Path, str]:
    path = _plain_file(path_value, label=label)
    if inside is not None:
        try:
            path.relative_to(inside)
        except ValueError as error:
            raise Partial28DINOError(f"{label} escaped its attempt directory") from error
    expected = _sha256(hash_value, label=f"{label} hash")
    before = path.stat()
    actual = file_sha256(path)
    after = path.stat()
    if actual != expected or (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise Partial28DINOError(f"{label} hash or identity differs")
    return path, actual


def validate_attempt_receipt(path_value: str | Path, *, expected_root_spec_sha256: str) -> dict[str, Any]:
    path = _plain_file(path_value, label="top-up generation receipt")
    receipt, receipt_sha = _strict_json(path, expected_sha256=None, label="top-up generation receipt")
    _closed(receipt, topup_generate._ATTEMPT_FIELDS, label="top-up generation receipt")
    unsigned = dict(receipt)
    declared = _sha256(unsigned.pop("receipt_digest", None), label="generation receipt digest")
    if declared != object_sha256(unsigned):
        raise Partial28DINOError("generation receipt embedded digest differs")
    candidate = receipt.get("candidate")
    artifacts = receipt.get("artifacts")
    artifact_authority = receipt.get("artifact_authority")
    if not isinstance(candidate, Mapping) or not isinstance(artifacts, Mapping) or not isinstance(artifact_authority, Mapping):
        raise Partial28DINOError("generation receipt nested structure differs")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or _SAFE_ID.fullmatch(candidate_id) is None:
        raise Partial28DINOError("candidate ID differs")
    expected_root = _sha256(expected_root_spec_sha256, label="root spec SHA-256")
    if (
        receipt.get("schema_version") != topup_generate.SCHEMA_VERSION
        or receipt.get("bank_id") != topup_generate.contract.BANK_ID
        or receipt.get("top_up_only") is not True
        or receipt.get("root_spec_raw_sha256") != expected_root
        or receipt.get("sampling_contract") != topup_generate.contract.SAMPLING_CONTRACT
        or receipt.get("semantic_input_closure") != topup_generate.contract.SEMANTIC_INPUT_CLOSURE
        or receipt.get("geometry_proxy_contract") != topup_generate.contract.GEOMETRY_PROXY_CONTRACT
        or receipt.get("artifact_authority") != topup_generate.contract.ARTIFACT_AUTHORITY
        or receipt.get("event_audit_status") != "pending_detached_full81_review"
        or receipt.get("event_verified") is not False
        or receipt.get("identity_preservation_verified") is not False
        or receipt.get("seed_selection_authorized") is not False
        or receipt.get("training_target_authorized") is not False
        or receipt.get("optimizer_or_parameter_update_authorized") is not False
        or candidate.get("event_verified") is not False
        or candidate.get("identity_preservation_verified") is not False
        or candidate.get("seed_selection_authorized") is not False
        or candidate.get("training_target_authorized") is not False
        or candidate.get("optimizer_authorized") is not False
    ):
        raise Partial28DINOError("generation receipt authority/spec binding differs")
    attempt_root = path.parent.resolve(strict=True)
    mp4 = artifacts.get("mp4")
    if not isinstance(mp4, Mapping) or mp4.get("frame_count") != 81 or mp4.get("fps") != 25:
        raise Partial28DINOError("generation receipt MP4 metadata is not exact81/25fps")
    mp4_path, mp4_sha = _stable_bound_file(
        mp4.get("path"), mp4.get("sha256"), label="candidate MP4", inside=attempt_root,
    )
    native_path, native_sha = _stable_bound_file(
        receipt.get("native_receipt_path"), receipt.get("native_receipt_sha256"),
        label="native generation receipt", inside=attempt_root,
    )
    envelope_path, envelope_sha = _stable_bound_file(
        receipt.get("candidate_envelope_path"), receipt.get("candidate_envelope_sha256"),
        label="candidate envelope",
    )
    return {
        "candidate_id": candidate_id,
        "ordinal": candidate.get("ordinal"),
        "iid": candidate.get("iid"),
        "row_id": candidate.get("row_id"),
        "actor_family": candidate.get("actor_family"),
        "analysis_split": candidate.get("analysis_split"),
        "branch": candidate.get("branch"),
        "seed": candidate.get("seed"),
        "receipt_path": str(path),
        "receipt_sha256": receipt_sha,
        "receipt_digest": declared,
        "native_receipt_path": str(native_path),
        "native_receipt_sha256": native_sha,
        "native_receipt_digest": _sha256(receipt.get("native_receipt_digest"), label="native receipt digest"),
        "candidate_envelope_path": str(envelope_path),
        "candidate_envelope_sha256": envelope_sha,
        "mp4_path": str(mp4_path),
        "mp4_sha256": mp4_sha,
        "declared_frame_count": 81,
        "declared_fps": 25,
        "upstream_event_verified": False,
        "upstream_identity_preservation_verified": False,
        "upstream_selection_authorized": False,
        "upstream_training_target_authorized": False,
    }


def build_manifest(args: argparse.Namespace) -> int:
    source_sha = _verify_self(args.expected_source_sha256)
    attempts_root = _plain_directory(args.attempts_root, label="attempts root")
    output_root = Path(args.output_root)
    if not output_root.is_absolute() or output_root == Path("/") or output_root.exists() or output_root.is_symlink():
        raise Partial28DINOError("output root must be fresh, absolute, and non-root")
    paths = sorted(attempts_root.rglob(ATTEMPT_BASENAME), key=lambda item: item.as_posix())
    if len(paths) != EXPECTED_ATTEMPT_COUNT:
        raise Partial28DINOError(
            f"partial bank must contain exactly {EXPECTED_ATTEMPT_COUNT} receipts, found {len(paths)}"
        )
    rows = [
        validate_attempt_receipt(path, expected_root_spec_sha256=args.expected_root_spec_sha256)
        for path in paths
    ]
    candidate_ids = [row["candidate_id"] for row in rows]
    if len(set(candidate_ids)) != EXPECTED_ATTEMPT_COUNT:
        raise Partial28DINOError("partial bank candidate IDs are not unique")
    rows.sort(key=lambda row: row["candidate_id"])
    output_root.mkdir(mode=0o700)
    unsigned = {
        "schema_version": INPUT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "attempts_root": str(attempts_root),
        "root_spec_raw_sha256": _sha256(args.expected_root_spec_sha256, label="root spec SHA-256"),
        "attempt_count": EXPECTED_ATTEMPT_COUNT,
        "world_size": EXPECTED_WORLD_SIZE,
        "partition_rule": "candidate_order_index_modulo_world_size",
        "selected_frame_indices": list(EVAL_FRAME_INDICES),
        "attempts": rows,
        "authority": dict(AUTHORITY_CLOSURE),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(output_root / "input-manifest.json", receipt)
    return 0


def load_input_manifest(path: str | Path, *, expected_sha256: str, expected_source_sha256: str) -> tuple[dict[str, Any], str]:
    value, raw_sha = _strict_json(path, expected_sha256=expected_sha256, label="input manifest")
    fields = {
        "schema_version", "diagnostic_source_sha256", "attempts_root",
        "root_spec_raw_sha256", "attempt_count", "world_size", "partition_rule",
        "selected_frame_indices", "attempts", "authority", "receipt_digest",
    }
    _closed(value, fields, label="input manifest")
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest"), label="input manifest digest")
    attempts = value.get("attempts")
    if (
        value.get("schema_version") != INPUT_SCHEMA
        or value.get("diagnostic_source_sha256") != expected_source_sha256
        or value.get("attempt_count") != EXPECTED_ATTEMPT_COUNT
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("partition_rule") != "candidate_order_index_modulo_world_size"
        or value.get("selected_frame_indices") != list(EVAL_FRAME_INDICES)
        or value.get("authority") != AUTHORITY_CLOSURE
        or declared != object_sha256(unsigned)
        or not isinstance(attempts, list)
        or len(attempts) != EXPECTED_ATTEMPT_COUNT
        or len({row.get("candidate_id") for row in attempts if isinstance(row, Mapping)}) != EXPECTED_ATTEMPT_COUNT
    ):
        raise Partial28DINOError("input manifest contract differs")
    return value, raw_sha


def _load_exact_module(name: str, path_value: str | Path, expected_sha256: str) -> Any:
    path = _plain_file(path_value, label=f"{name} source")
    if file_sha256(path) != _sha256(expected_sha256, label=f"{name} source hash"):
        raise Partial28DINOError(f"{name} source hash differs")
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise Partial28DINOError(f"cannot construct {name} module spec")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def _load_evaluator(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    visual_contract = _load_exact_module(
        "pair_v5_source_bound_preservation_evaluator_v1",
        args.visual_contract_source,
        args.expected_visual_contract_sha256,
    )
    visual_scorer = _load_exact_module(
        "partial28_exact_visual_scorer_v1",
        args.visual_scorer_source,
        args.expected_visual_scorer_sha256,
    )
    if file_sha256(args.visual_scorer_source) != _sha256(args.expected_visual_scorer_sha256, label="visual scorer source hash"):
        raise Partial28DINOError("visual scorer source hash differs")
    raw_spec, spec_sha = _strict_json(
        args.evaluator_spec, expected_sha256=args.expected_evaluator_spec_sha256,
        label="sealed visual evaluator spec",
    )
    try:
        spec = visual_contract.validate_evaluator_spec(raw_spec)
    except Exception as error:
        raise Partial28DINOError(f"sealed visual evaluator spec failed validation: {error}") from error
    if (
        spec.get("implementation_sha256") != args.expected_visual_scorer_sha256
        or spec.get("contract_sha256") != args.expected_visual_contract_sha256
    ):
        raise Partial28DINOError("visual evaluator spec/source binding differs")
    try:
        checkpoint = visual_scorer.verify_checkpoint_content(
            args.visual_checkpoint, args.visual_checkpoint_manifest,
            evaluator_spec=spec,
        )
    except Exception as error:
        raise Partial28DINOError(f"sealed visual checkpoint failed validation: {error}") from error
    versions = visual_scorer.runtime_versions()
    if versions != spec["runtime_versions"]:
        raise Partial28DINOError("visual evaluator runtime versions differ")
    processor = checkpoint.pop("processor")
    checkpoint["evaluator_spec_sha256"] = spec_sha
    checkpoint["runtime_versions"] = versions
    checkpoint["identity_authority"] = False
    checkpoint["scientific_claim_authorized"] = False
    return {
        "processor": processor, "spec": spec, "scorer": visual_scorer,
        "contract": visual_contract,
    }, checkpoint


def _configure_device() -> Any:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Partial28DINOError("each worker must see exactly one ROCm/CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Partial28DINOError(f"{label} is non-finite")
    return result


def temporal_proxy(global_feature: Any, dense_feature: Any) -> dict[str, Any]:
    """Return same-video DINO cosine traces without an acceptance threshold."""

    import torch

    count = len(EVAL_FRAME_INDICES)
    if list(global_feature.shape[:1]) != [count] or list(dense_feature.shape[:1]) != [count]:
        raise Partial28DINOError("temporal feature geometry differs")
    mapped_global_adjacent = (((global_feature[1:] * global_feature[:-1]).sum(dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    mapped_global_frame0 = (((global_feature[1:] * global_feature[0:1]).sum(dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    dense_cosine = (dense_feature[1:] * dense_feature[:-1]).sum(dim=-1)
    mapped_dense_adjacent = ((dense_cosine + 1.0) * 0.5).clamp(0.0, 1.0).median(dim=-1).values
    if not all(bool(torch.isfinite(item).all().item()) for item in (
        mapped_global_adjacent, mapped_global_frame0, mapped_dense_adjacent,
    )):
        raise Partial28DINOError("temporal proxy contains a non-finite value")

    def trace(value: Any) -> list[float]:
        return [_finite(item, label="temporal proxy") for item in value.tolist()]

    return {
        "measurement_label": "frozen_dinov2_same_video_identity_appearance_temporal_proxy_only",
        "selected_frame_indices": list(EVAL_FRAME_INDICES),
        "global_adjacent_mapped_cosine": trace(mapped_global_adjacent),
        "global_adjacent_mean": _finite(mapped_global_adjacent.mean().item(), label="global adjacent mean"),
        "global_adjacent_minimum": _finite(mapped_global_adjacent.min().item(), label="global adjacent minimum"),
        "global_frame0_to_later_mapped_cosine": trace(mapped_global_frame0),
        "global_frame0_to_later_mean": _finite(mapped_global_frame0.mean().item(), label="global frame0 mean"),
        "global_frame0_to_endpoint": _finite(mapped_global_frame0[-1].item(), label="global endpoint"),
        "dense_adjacent_token_median_mapped_cosine": trace(mapped_dense_adjacent),
        "dense_adjacent_mean": _finite(mapped_dense_adjacent.mean().item(), label="dense adjacent mean"),
        "dense_adjacent_minimum": _finite(mapped_dense_adjacent.min().item(), label="dense adjacent minimum"),
        "thresholds": None,
        "identity_authority": False,
        "event_authority": False,
        "scientific_claim_authorized": False,
        "selection_authorized": False,
    }


def _measure_attempt(row: Mapping[str, Any], *, visual_scorer: Any, processor: Any, model: Any, device: Any) -> dict[str, Any]:
    # Re-hash both the generation receipt and candidate immediately before decode.
    if file_sha256(row["receipt_path"]) != row["receipt_sha256"]:
        raise Partial28DINOError("generation receipt changed after input sealing")
    frames, decode = visual_scorer.decode_exact81_rgb(
        row["mp4_path"], expected_sha256=row["mp4_sha256"]
    )
    _, normalized = visual_scorer.preprocess_selected_rgb(frames, processor)
    global_feature, dense_feature, features = visual_scorer.extract_features(
        model, normalized, device=device, num_register_tokens=0,
    )
    return {
        "candidate_id": row["candidate_id"],
        "candidate_binding": dict(row),
        "decode": decode,
        "features": features,
        "temporal_proxy": temporal_proxy(global_feature, dense_feature),
        "authority": dict(AUTHORITY_CLOSURE),
    }


def _rank(value: Any, *, world_size: int) -> int:
    if type(value) is not int or value < 0 or value >= world_size:
        raise Partial28DINOError("rank is outside the fixed world")
    return value


def partition_indices(count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if count != EXPECTED_ATTEMPT_COUNT or world_size != EXPECTED_WORLD_SIZE:
        raise Partial28DINOError("partial28 partition geometry differs")
    _rank(rank, world_size=world_size)
    return tuple(index for index in range(count) if index % world_size == rank)


def _worker_common(args: argparse.Namespace) -> tuple[str, dict[str, Any], str, Mapping[str, Any], dict[str, Any], Any]:
    source_sha = _verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest, expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    evaluator, checkpoint_evidence = _load_evaluator(args)
    device = _configure_device()
    try:
        model, loading_counts = evaluator["scorer"].load_frozen_model(
            checkpoint_evidence, device=device
        )
    except Exception as error:
        raise Partial28DINOError(f"cannot load frozen DINO model: {error}") from error
    checkpoint_evidence["root"] = str(checkpoint_evidence["root"])
    checkpoint_evidence["loading_counts"] = loading_counts
    checkpoint_evidence["frozen_eval"] = True
    checkpoint_evidence["trainable_parameter_tensors"] = 0
    return source_sha, manifest, manifest_sha, evaluator, checkpoint_evidence, (model, device)


def preflight(args: argparse.Namespace) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned = _worker_common(args)
    model, device = owned
    rank = _rank(args.rank, world_size=EXPECTED_WORLD_SIZE)
    indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
    result = _measure_attempt(
        manifest["attempts"][indices[0]], visual_scorer=evaluator["scorer"],
        processor=evaluator["processor"], model=model, device=device,
    )
    unsigned = {
        "schema_version": PREFLIGHT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "rank": rank,
        "world_size": EXPECTED_WORLD_SIZE,
        "one_candidate_only": True,
        "candidate_result": result,
        "visual_evaluator": checkpoint,
        "authority": dict(AUTHORITY_CLOSURE),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(_plain_directory(args.output_root, label="output root") / f"preflight-rank-{rank:02d}.json", receipt)
    return 0


def worker(args: argparse.Namespace) -> int:
    source_sha, manifest, manifest_sha, evaluator, checkpoint, owned = _worker_common(args)
    model, device = owned
    rank = _rank(args.rank, world_size=args.world_size)
    if args.world_size != EXPECTED_WORLD_SIZE:
        raise Partial28DINOError("worker world size must be exactly eight")
    indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, args.world_size)
    results = [
        _measure_attempt(
            manifest["attempts"][index], visual_scorer=evaluator["scorer"],
            processor=evaluator["processor"], model=model, device=device,
        )
        for index in indices
    ]
    unsigned = {
        "schema_version": SHARD_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "rank": rank,
        "world_size": args.world_size,
        "partition_indices": list(indices),
        "candidate_count": len(results),
        "candidate_results": results,
        "visual_evaluator": checkpoint,
        "authority": dict(AUTHORITY_CLOSURE),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(
        _plain_directory(args.output_root, label="output root")
        / f"shard-{rank:02d}-of-{args.world_size:02d}.json",
        receipt,
    )
    return 0


def _validate_shard(value: Mapping[str, Any], *, rank: int, source_sha: str, manifest_sha: str) -> Mapping[str, Any]:
    fields = {
        "schema_version", "diagnostic_source_sha256", "input_manifest_sha256",
        "rank", "world_size", "partition_indices", "candidate_count",
        "candidate_results", "visual_evaluator", "authority", "receipt_digest",
    }
    _closed(value, fields, label=f"shard {rank}")
    unsigned = dict(value)
    declared = _sha256(unsigned.pop("receipt_digest"), label=f"shard {rank} digest")
    indices = partition_indices(EXPECTED_ATTEMPT_COUNT, rank, EXPECTED_WORLD_SIZE)
    results = value.get("candidate_results")
    if (
        value.get("schema_version") != SHARD_SCHEMA
        or value.get("diagnostic_source_sha256") != source_sha
        or value.get("input_manifest_sha256") != manifest_sha
        or value.get("rank") != rank
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("partition_indices") != list(indices)
        or not isinstance(results, list)
        or value.get("candidate_count") != len(indices)
        or len(results) != len(indices)
        or value.get("authority") != AUTHORITY_CLOSURE
        or declared != object_sha256(unsigned)
    ):
        raise Partial28DINOError(f"shard {rank} contract differs")
    return value


def aggregate(args: argparse.Namespace) -> int:
    source_sha = _verify_self(args.expected_source_sha256)
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest, expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    output_root = _plain_directory(args.output_root, label="output root")
    shards: list[dict[str, Any]] = []
    by_index: dict[int, Mapping[str, Any]] = {}
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-{EXPECTED_WORLD_SIZE:02d}.json"
        value, raw_sha = _strict_json(path, expected_sha256=None, label=f"shard {rank}")
        checked = _validate_shard(value, rank=rank, source_sha=source_sha, manifest_sha=manifest_sha)
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": checked["receipt_digest"],
        })
        for index, result in zip(checked["partition_indices"], checked["candidate_results"]):
            if index in by_index:
                raise Partial28DINOError("shard partition overlaps")
            by_index[index] = result
    if set(by_index) != set(range(EXPECTED_ATTEMPT_COUNT)):
        raise Partial28DINOError("shards do not cover exact partial28")
    ordered = [by_index[index] for index in range(EXPECTED_ATTEMPT_COUNT)]
    expected_ids = [row["candidate_id"] for row in manifest["attempts"]]
    if [row.get("candidate_id") for row in ordered] != expected_ids:
        raise Partial28DINOError("aggregate candidate order differs")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "candidate_count": EXPECTED_ATTEMPT_COUNT,
        "coverage": "exactly_once_complete_partial28",
        "candidate_order": expected_ids,
        "shards": shards,
        "candidate_results": ordered,
        "interpretation": {
            "measurement": "same-video frozen-DINO temporal appearance stability proxy plus exact81 decode evidence",
            "no_source_comparison": True,
            "no_event_measurement": True,
            "no_threshold_or_ranking": True,
            "cannot_verify_identity_or_action_editing_success": True,
        },
        "authority": dict(AUTHORITY_CLOSURE),
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_create_only(output_root / "aggregate-receipt.json", receipt)
    return 0


def _add_visual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--visual-checkpoint", required=True)
    parser.add_argument("--visual-checkpoint-manifest", required=True)
    parser.add_argument("--evaluator-spec", required=True)
    parser.add_argument("--expected-evaluator-spec-sha256", required=True)
    parser.add_argument("--visual-scorer-source", required=True)
    parser.add_argument("--expected-visual-scorer-sha256", required=True)
    parser.add_argument("--visual-contract-source", required=True)
    parser.add_argument("--expected-visual-contract-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rank", required=True, type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--attempts-root", required=True)
    build.add_argument("--expected-root-spec-sha256", required=True)
    build.add_argument("--expected-source-sha256", required=True)
    build.add_argument("--output-root", required=True)
    check = commands.add_parser("preflight")
    _add_visual_arguments(check)
    run = commands.add_parser("worker")
    _add_visual_arguments(run)
    run.add_argument("--world-size", required=True, type=int)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--input-manifest", required=True)
    combine.add_argument("--expected-input-manifest-sha256", required=True)
    combine.add_argument("--expected-source-sha256", required=True)
    combine.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        return build_manifest(args)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "worker":
        return worker(args)
    if args.command == "aggregate":
        return aggregate(args)
    raise Partial28DINOError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
