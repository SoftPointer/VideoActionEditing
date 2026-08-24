#!/usr/bin/env python3
"""Exact8 registered-source frozen-DINO pair-hardness matrix diagnostic.

This source-only operational diagnostic decodes the eight sealed exact81 source
videos at the registered 17 frame indices.  Candidate/proposal media are never
opened.  A frozen candidate-bearing receipt is parsed only after the new matrix
is sealed, solely for whole-receipt integrity, source-feature hashes, and pair
bindings; its candidate-metric fields are never queried or used.  This defines
no threshold and cannot rank, select, train, authorize formal evidence, or
support a scientific claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
SOURCE_VALIDATOR_NAME = "build_saic_reversible_source_set_v1.py"
SOURCE_VALIDATOR_SHA256 = "0cf012adf25dd1afffb33d1e0c918630a574c9075e9aa293914e04890c71cf5b"
PINNED_LOCAL_SOURCE_CLOSURE = {SOURCE_VALIDATOR_NAME: SOURCE_VALIDATOR_SHA256}
SCHEMA_VERSION = "bernini-saic-registered-source8-dinov2-pair-matrix-raw-v1"
INPUT_SCHEMA = f"{SCHEMA_VERSION}-input"
PREFLIGHT_SCHEMA = f"{SCHEMA_VERSION}-preflight"
SHARD_SCHEMA = f"{SCHEMA_VERSION}-shard"
AGGREGATE_SCHEMA = f"{SCHEMA_VERSION}-aggregate"
EXPECTED_WORLD_SIZE = 8
EXPECTED_SOURCE_COUNT = 8
EXPECTED_MATRIX_CELL_COUNT = 64
EXPECTED_SAME_ACTOR_CELL_COUNT = 32
EXPECTED_CROSS_ACTOR_CELL_COUNT = 32
EXPECTED_DIAGONAL_COUNT = 8
EXPECTED_REGISTERED_ALL3_DIRECTED_PAIR_COUNT = 24
EXPECTED_EXECUTED_ALL3_DIRECTED_PAIR_COUNT = 21
EXPECTED_MISSING_CORRECT_SOURCE_IID = "6ea45d35943742bb"
EVAL_FRAME_INDICES = tuple(range(0, 81, 5))
EXPECTED_SOURCE_MANIFEST_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/runs/"
    "t2v-events-topup-r6-umaskfix-72f3a40-r1/sealed-saic-source-manifest.json"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "899b5a1dd66fc0bf6d4d0192fb6157f4afe691c50633246dddcaa1db2c2a98a9"
)
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "9c2a3d6841951ea0ed050dc230630a1176460e25a979ec199eab575ad22f3c6f"
)
EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256 = (
    "257d3aafaaee126ff2c1a061413d01bd0457676eb5d1ee027671221a5a794218"
)
EXPECTED_EVALUATOR_SPEC_SHA256 = "6b18b9bc10589325ee2c09af339ef43a3eff507bcc754a2a6984cb70f0afd736"
EXPECTED_VISUAL_SCORER_SHA256 = "9e86ee8128841f624db92b99914235a37fee4d7b92aeda2e62104ab57e531b39"
EXPECTED_VISUAL_CONTRACT_SHA256 = "183eaafaebef426f888aa3abe91632a884f827d39ae16db576d57da401a8533a"
EXPECTED_EXPERIMENT_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_pair_v5_source_bound_preservation_v1_20260808"
)
EXPECTED_CHECKPOINT_ROOT = f"{EXPECTED_EXPERIMENT_ROOT}/vendor/dinov2-base-f9e44c8"
EXPECTED_CHECKPOINT_MANIFEST_PATH = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/dinov2-base-f9e44c8.sha256"
EXPECTED_EVALUATOR_SPEC_PATH = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/pair_v5_source_bound_preservation_evaluator_7c4c837_v1.json"
EXPECTED_VISUAL_SOURCE_ROOT = f"{EXPECTED_EXPERIMENT_ROOT}/inputs/source-preservation-7c4c837-minimal/methods/bernini_action_editing"
EXPECTED_VISUAL_SCORER_PATH = f"{EXPECTED_VISUAL_SOURCE_ROOT}/score_pair_v5_source_bound_preservation_v1.py"
EXPECTED_VISUAL_CONTRACT_PATH = f"{EXPECTED_VISUAL_SOURCE_ROOT}/pair_v5_source_bound_preservation_evaluator_v1.py"
EXPECTED_LEGACY_ALL3_AGGREGATE_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/diagnostics/"
    "allocation-134936-sourcebound47-all3-dinov2-full-847f3cf3-r1/"
    "aggregate-receipt.json"
)
LEGACY_ALL3_AGGREGATE_SCHEMA = (
    "bernini-saic-partial47-source-bound-dinov2-"
    "same-actor-all-three-negatives-raw-v1-aggregate"
)
LEGACY_ALL3_SOURCE_SHA256 = "847f3cf31553cb8a73d276278026dbdd898ed4bd094fd973978ce1f143be50dd"
LEGACY_ALL3_CANDIDATE_COUNT = 47
EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256 = (
    "3f5169d45b603ac6c10da12d6736b7878c30a6654b8aeedd335a8548865b7beb"
)
EXPECTED_RUNTIME_VERSIONS = {
    "python_version": "3.12.13",
    "torch_version": "2.7.1+rocm6.3",
    "torch_hip_version": "6.3.42131-fa1d09cbd",
    "transformers_version": "4.53.2",
    "safetensors_version": "0.8.0rc0",
    "av_version": "13.1.0",
    "numpy_version": "1.26.4",
    "pillow_version": "11.3.0",
}
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "b61f251411f0d8f6a617b67d0b903c333d16c77fb6b3f49507225884d4aed0ea"
)
EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256 = (
    "d8217ce3a86de051a4affd701c965befd12584cce51902c9f266fff952ebd18a"
)
EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256 = (
    "b5ef31a8754b854ce64dcf49a79949e22ff9219a7db5d2dfd5fec1ed0602fb6a"
)
EXPECTED_FEATURE_GEOMETRY = {
    "selected_frame_count": 17,
    "dense_grid_height": 16,
    "dense_grid_width": 16,
    "feature_dimension": 768,
}
SYMMETRY_ABS_TOLERANCE = 1.0e-6

AUTHORITY = {
    "operational_diagnostic_authority": False,
    "absolute_preservation_authority": False,
    "identity_authority": False,
    "event_authority": False,
    "multi_negative_proxy_authority": False,
    "formal_retained_source_fd_authority": False,
    "formal_evidence_authority": False,
    "scientific_claim_authorized": False,
    "ranking_authorized": False,
    "selection_authorized": False,
    "candidate_selection_allowed": False,
    "training_allowed": False,
    "training_target_authorized": False,
    "optimizer_or_parameter_update_authorized": False,
}
LIMITATION = {
    "source_only_operational_diagnostic": True,
    "candidate_or_proposal_media_consulted": False,
    "candidate_metric_fields_queried": False,
    "candidate_metric_values_used": False,
    "legacy_candidate_bearing_bytes_permitted_only_after_matrix_validation": True,
    "thresholds": None,
    "source_features_held_in_cpu_memory_until_worker_exit": True,
    "source_files_retained_open_until_worker_exit": False,
    "formal_retained_source_fd_closure_satisfied": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
_MODEL_FIELDS = {
    "adapter_id", "architecture_id", "checkpoint_manifest_sha256",
    "checkpoint_config_sha256", "preprocessor_config_sha256",
    "checkpoint_file_count", "num_register_tokens", "image_size",
    "patch_size", "preprocessor_golden_input_sha256",
    "preprocessor_golden_output_sha256", "preprocessor_golden_output_shape",
}
_MODEL_EVIDENCE_FIELDS = {
    "adapter_id", "architecture_id", "checkpoint_manifest_sha256",
    "checkpoint_config_sha256", "preprocessor_config_sha256",
    "checkpoint_file_count", "verified_entries_digest",
    "preprocessor_golden_input_sha256", "preprocessor_golden_output_sha256",
    "preprocessor_golden_output_shape", "every_checkpoint_file_verified",
    "all_parameters_frozen", "trainable_parameter_tensors",
    "parameter_tensor_count", "parameter_element_count",
    "parameter_metadata_digest", "missing_key_count", "unexpected_key_count",
    "mismatched_key_count", "loading_error_count", "runtime_versions",
}
_VISUAL_EVIDENCE_FIELDS = {
    "checkpoint_root", "checkpoint_manifest_path", "evaluator_spec_path",
    "visual_scorer_path", "visual_contract_path", "evaluator_spec_sha256",
    "visual_scorer_sha256", "visual_contract_sha256",
    "checkpoint_manifest_raw_sha256", "model_evidence",
    "model_evidence_sha256", "candidate_or_proposal_media_consulted",
    "candidate_metric_fields_queried", "candidate_metric_values_used",
    "identity_authority", "scientific_claim_authorized",
}


class Source8MatrixError(RuntimeError):
    """Raised before unauthenticated or over-claimed evidence is emitted."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Source8MatrixError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Source8MatrixError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(
    value: str | Path, *, label: str, expected_path: str | None = None,
) -> Path:
    path = Path(value)
    if expected_path is not None and str(path) != expected_path:
        raise Source8MatrixError(f"{label} lexical path differs")
    if (
        not path.is_absolute() or not path.is_file() or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Source8MatrixError(f"{label} must be an absolute plain file")
    return path


def _plain_directory(
    value: str | Path, *, label: str, expected_path: str | None = None,
) -> Path:
    path = Path(value)
    if expected_path is not None and str(path) != expected_path:
        raise Source8MatrixError(f"{label} lexical path differs")
    if (
        not path.is_absolute() or not path.is_dir() or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise Source8MatrixError(f"{label} must be an absolute plain directory")
    return path


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Source8MatrixError(f"{label} field closure differs")
    return value


def _strict_json(
    path_value: str | Path,
    *,
    expected_sha256: str | None,
    label: str,
    expected_path: str | None = None,
) -> tuple[dict[str, Any], str]:
    path = _plain_file(path_value, label=label, expected_path=expected_path)
    before = path.stat()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, label=f"{label} expected SHA-256"
    ):
        raise Source8MatrixError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> Any:
        raise Source8MatrixError(f"{label} contains non-finite token {token}")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise Source8MatrixError(f"{label} contains duplicate key {key!r}")
            result[key] = child
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Source8MatrixError(f"{label} is invalid JSON") from error
    after = path.stat()
    if (
        type(value) is not dict
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(path) != digest
    ):
        raise Source8MatrixError(f"{label} changed while reading")
    return value, digest


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if (
        not path.is_absolute() or path.exists() or path.is_symlink()
        or not path.parent.is_dir() or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise Source8MatrixError("receipt path must be fresh in one plain directory")
    raw = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
    except FileExistsError as error:
        raise Source8MatrixError(f"refusing to overwrite {path}") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        leaf_stat = path.lstat()
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_stat.st_nlink != 1
            or (descriptor_stat.st_dev, descriptor_stat.st_ino)
            != (leaf_stat.st_dev, leaf_stat.st_ino)
        ):
            raise Source8MatrixError("create-only receipt reservation differs")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    final = path.lstat()
    if (
        not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(final.st_mode) != 0o400
        or final.st_nlink != 1
        or (final.st_dev, final.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
        or file_sha256(path) != hashlib.sha256(raw).hexdigest()
    ):
        raise Source8MatrixError("create-only receipt finalization differs")
    return hashlib.sha256(raw).hexdigest()


def _rank(value: Any, *, world_size: int) -> int:
    if type(value) is not int or value < 0 or value >= world_size:
        raise Source8MatrixError("rank is outside the fixed world")
    return value


class _IndependentCore:
    EVAL_FRAME_INDICES = EVAL_FRAME_INDICES
    file_sha256 = staticmethod(file_sha256)
    object_sha256 = staticmethod(object_sha256)
    canonical_json_bytes = staticmethod(canonical_json_bytes)
    _sha256 = staticmethod(_sha256)
    _plain_file = staticmethod(_plain_file)
    _plain_directory = staticmethod(_plain_directory)
    _closed = staticmethod(_closed)
    _strict_json = staticmethod(_strict_json)
    _write_create_only = staticmethod(_write_create_only)
    _rank = staticmethod(_rank)

    @staticmethod
    def _load_evaluator(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return _load_evaluator(args)

    @staticmethod
    def _configure_device() -> Any:
        return _configure_device()

    @staticmethod
    def _add_visual_arguments(parser: Any) -> None:
        _add_visual_arguments(parser)


core = _IndependentCore()


def _verify_self(expected_sha256: str) -> str:
    actual = core.file_sha256(Path(__file__).resolve())
    if actual != core._sha256(expected_sha256, label="diagnostic source SHA-256"):
        raise Source8MatrixError("diagnostic source SHA-256 differs")
    return actual


def _finite(value: Any, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise Source8MatrixError(f"{label} is not a strict scalar")
    result = float(value)
    if not math.isfinite(result):
        raise Source8MatrixError(f"{label} is non-finite")
    return result


def _manifest_design(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    order = list(sources)
    if len(order) != EXPECTED_SOURCE_COUNT or len(set(order)) != EXPECTED_SOURCE_COUNT:
        raise Source8MatrixError("registered source manifest is not exact8 ordered")
    cells = []
    for row_ordinal, row_iid in enumerate(order):
        row = sources[row_iid]
        for column_ordinal, column_iid in enumerate(order):
            column = sources[column_iid]
            same_actor = row["actor_family"] == column["actor_family"]
            cells.append({
                "matrix_ordinal": row_ordinal * EXPECTED_SOURCE_COUNT + column_ordinal,
                "row_ordinal": row_ordinal,
                "column_ordinal": column_ordinal,
                "row_source_iid": row_iid,
                "row_source_video_sha256": row["source_video_sha256"],
                "column_source_iid": column_iid,
                "column_source_video_sha256": column["source_video_sha256"],
                "relationship": "same_actor" if same_actor else "cross_actor",
                "diagonal": row_ordinal == column_ordinal,
                "registered_all3_directed_pair": same_actor and row_ordinal != column_ordinal,
            })
    if (
        len(cells) != EXPECTED_MATRIX_CELL_COUNT
        or sum(row["relationship"] == "same_actor" for row in cells)
        != EXPECTED_SAME_ACTOR_CELL_COUNT
        or sum(row["relationship"] == "cross_actor" for row in cells)
        != EXPECTED_CROSS_ACTOR_CELL_COUNT
        or sum(row["diagonal"] for row in cells) != EXPECTED_DIAGONAL_COUNT
        or sum(row["registered_all3_directed_pair"] for row in cells)
        != EXPECTED_REGISTERED_ALL3_DIRECTED_PAIR_COUNT
    ):
        raise Source8MatrixError("registered exact8 pair universe differs")
    return {
        "registration_policy": "sealed_source_manifest_row_major_exact8x8_v1",
        "source_manifest_order": order,
        "matrix_shape": [8, 8],
        "matrix_cell_count": EXPECTED_MATRIX_CELL_COUNT,
        "same_actor_cell_count_including_diagonal": EXPECTED_SAME_ACTOR_CELL_COUNT,
        "cross_actor_cell_count": EXPECTED_CROSS_ACTOR_CELL_COUNT,
        "diagonal_cell_count": EXPECTED_DIAGONAL_COUNT,
        "same_actor_off_diagonal_directed_pair_count": EXPECTED_REGISTERED_ALL3_DIRECTED_PAIR_COUNT,
        "cells": cells,
        "candidate_or_proposal_media_or_metrics_consulted_during_registration": False,
    }


def _load_local_source_validator() -> Any:
    path = _plain_file(
        METHOD_ROOT / SOURCE_VALIDATOR_NAME,
        label="source-manifest validator source",
        expected_path=str(METHOD_ROOT / SOURCE_VALIDATOR_NAME),
    )
    if file_sha256(path) != SOURCE_VALIDATOR_SHA256:
        raise Source8MatrixError("source-manifest validator source SHA-256 differs")
    before = path.stat()
    name = "_registered_source8_frozen_source_validator_v1"
    if name in sys.modules:
        raise Source8MatrixError("source-manifest validator module was preloaded")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Source8MatrixError("cannot construct source-manifest validator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    after = path.stat()
    if (
        getattr(module, "__file__", None) != str(path)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(path) != SOURCE_VALIDATOR_SHA256
    ):
        sys.modules.pop(name, None)
        raise Source8MatrixError("source-manifest validator __file__ differs")
    return module


def _source_closure(
    source_manifest_path: str | Path,
    expected_raw_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if expected_raw_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise Source8MatrixError("caller source-manifest raw SHA-256 differs")
    manifest, raw_sha = _strict_json(
        source_manifest_path,
        expected_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
        expected_path=EXPECTED_SOURCE_MANIFEST_PATH,
        label="sealed source manifest",
    )
    validator = _load_local_source_validator()
    try:
        summary = dict(validator.validate_manifest(manifest, verify_bound_files=True))
    except Exception as error:
        raise Source8MatrixError(
            f"sealed source manifest failed frozen validation: {error}"
        ) from error
    finally:
        sys.modules.pop("_registered_source8_frozen_source_validator_v1", None)
    validator_summary_sha = object_sha256(summary)
    if (
        summary.get("manifest_content_sha256")
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or summary.get("row_count") != EXPECTED_SOURCE_COUNT
        or summary.get("bound_files_verified") is not True
        or validator_summary_sha != EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256
    ):
        raise Source8MatrixError("sealed source content/validator closure differs")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SOURCE_COUNT:
        raise Source8MatrixError("sealed source manifest is not exact8")
    sources: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Source8MatrixError("sealed source row is not an object")
        iid = row.get("iid")
        if not isinstance(iid, str) or iid in sources:
            raise Source8MatrixError("sealed source IID closure differs")
        path = _plain_file(row.get("source_video"), label=f"registered source {iid}")
        declared_sha = _sha256(
            row.get("source_video_sha256"), label=f"registered source {iid} SHA-256"
        )
        if file_sha256(path) != declared_sha:
            raise Source8MatrixError("registered source file SHA-256 differs")
        sources[iid] = {
            "ordinal": ordinal,
            "iid": iid,
            "row_id": row.get("row_id"),
            "analysis_split": row.get("analysis_split"),
            "actor_family": row.get("actor_family"),
            "actor_group_id": row.get("actor_group_id"),
            "scene_group_id": row.get("scene_group_id"),
            "source_video": str(path),
            "source_video_sha256": declared_sha,
        }
    evidence = {
        "path": EXPECTED_SOURCE_MANIFEST_PATH,
        "raw_sha256": raw_sha,
        "content_sha256": EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256,
        "validator_source_path": str(METHOD_ROOT / SOURCE_VALIDATOR_NAME),
        "validator_source_sha256": SOURCE_VALIDATOR_SHA256,
        "validator_module_file_verified": True,
        "validator_summary_sha256": validator_summary_sha,
        "bound_files_verified": True,
    }
    return sources, evidence, _manifest_design(sources)


def build_manifest(args: Any) -> int:
    source_sha = _verify_self(args.expected_source_sha256)
    if (
        args.expected_source_manifest_sha256 != EXPECTED_SOURCE_MANIFEST_SHA256
        or args.expected_source_manifest_content_sha256
        != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256
        or args.expected_source_validator_summary_sha256
        != EXPECTED_SOURCE_VALIDATOR_SUMMARY_SHA256
    ):
        raise Source8MatrixError("caller source-manifest closure pins differ")
    output_root = Path(args.output_root)
    if (
        not output_root.is_absolute()
        or output_root == Path("/")
        or output_root.exists()
        or output_root.is_symlink()
        or not output_root.parent.is_dir()
        or output_root.parent.is_symlink()
        or output_root.parent.resolve(strict=True) != output_root.parent
    ):
        raise Source8MatrixError("output root must be fresh, absolute, and non-root")
    sources, evidence, design = _source_closure(
        args.source_manifest, args.expected_source_manifest_sha256
    )
    output_root.mkdir(mode=0o700)
    _plain_directory(output_root, label="fresh output root")
    unsigned = {
        "schema_version": INPUT_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "source_manifest": evidence,
        "source_manifest_order": list(sources),
        "sources": [dict(sources[iid]) for iid in sources],
        "selected_frame_indices": list(core.EVAL_FRAME_INDICES),
        "matrix_registration": design,
        "legacy_r4_all3_aggregate_was_not_opened_during_registration": True,
        "pinned_local_source_closure": dict(PINNED_LOCAL_SOURCE_CLOSURE),
        "pinned_local_source_closure_sha256": core.object_sha256(PINNED_LOCAL_SOURCE_CLOSURE),
        "limitation": dict(LIMITATION),
        "authority": dict(AUTHORITY),
    }
    core._write_create_only(
        output_root / "input-manifest.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def load_input_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = core._strict_json(
        path, expected_sha256=expected_sha256, label="source8 matrix input manifest"
    )
    core._closed(value, {
        "schema_version", "diagnostic_source_sha256", "source_manifest",
        "source_manifest_order", "sources", "selected_frame_indices",
        "matrix_registration", "limitation", "authority", "receipt_digest",
        "pinned_local_source_closure", "pinned_local_source_closure_sha256",
        "legacy_r4_all3_aggregate_was_not_opened_during_registration",
    }, label="source8 matrix input manifest")
    unsigned = dict(value)
    declared = core._sha256(unsigned.pop("receipt_digest", None), label="input digest")
    sources, evidence, design = _source_closure(
        value.get("source_manifest", {}).get("path"),
        EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    if (
        value.get("schema_version") != INPUT_SCHEMA
        or value.get("diagnostic_source_sha256") != expected_source_sha256
        or value.get("source_manifest") != evidence
        or value.get("source_manifest_order") != list(sources)
        or value.get("sources") != [dict(sources[iid]) for iid in sources]
        or value.get("selected_frame_indices") != list(core.EVAL_FRAME_INDICES)
        or value.get("matrix_registration") != design
        or value.get("legacy_r4_all3_aggregate_was_not_opened_during_registration") is not True
        or value.get("pinned_local_source_closure") != PINNED_LOCAL_SOURCE_CLOSURE
        or value.get("pinned_local_source_closure_sha256")
        != core.object_sha256(PINNED_LOCAL_SOURCE_CLOSURE)
        or value.get("limitation") != LIMITATION
        or value.get("authority") != AUTHORITY
        or declared != core.object_sha256(unsigned)
    ):
        raise Source8MatrixError("source8 matrix input manifest contract differs")
    return value, raw_sha


def _load_exact_module(
    name: str,
    path_value: str | Path,
    *,
    expected_path: str,
    expected_sha256: str,
) -> Any:
    path = _plain_file(path_value, label=f"{name} source", expected_path=expected_path)
    if file_sha256(path) != expected_sha256:
        raise Source8MatrixError(f"{name} source SHA-256 differs")
    before = path.stat()
    if name in sys.modules:
        raise Source8MatrixError(f"{name} was preloaded before exact-path verification")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Source8MatrixError(f"cannot construct {name} module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    after = path.stat()
    if (
        getattr(module, "__file__", None) != str(path)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(path) != expected_sha256
    ):
        sys.modules.pop(name, None)
        raise Source8MatrixError(f"{name} module __file__ differs")
    return module


def _load_evaluator(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        str(args.visual_checkpoint) != EXPECTED_CHECKPOINT_ROOT
        or str(args.visual_checkpoint_manifest) != EXPECTED_CHECKPOINT_MANIFEST_PATH
        or str(args.evaluator_spec) != EXPECTED_EVALUATOR_SPEC_PATH
        or str(args.visual_scorer_source) != EXPECTED_VISUAL_SCORER_PATH
        or str(args.visual_contract_source) != EXPECTED_VISUAL_CONTRACT_PATH
        or args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
    ):
        raise Source8MatrixError("frozen evaluator lexical path/SHA arguments differ")
    _plain_directory(
        args.visual_checkpoint,
        label="visual checkpoint root",
        expected_path=EXPECTED_CHECKPOINT_ROOT,
    )
    checkpoint_manifest = _plain_file(
        args.visual_checkpoint_manifest,
        label="visual checkpoint content manifest",
        expected_path=EXPECTED_CHECKPOINT_MANIFEST_PATH,
    )
    if file_sha256(checkpoint_manifest) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
        raise Source8MatrixError("visual checkpoint content-manifest raw SHA-256 differs")
    contract_name = "pair_v5_source_bound_preservation_evaluator_v1"
    scorer_name = "registered_source8_visual_scorer_v1"
    original_sys_path = list(sys.path)
    try:
        contract = _load_exact_module(
            contract_name,
            args.visual_contract_source,
            expected_path=EXPECTED_VISUAL_CONTRACT_PATH,
            expected_sha256=EXPECTED_VISUAL_CONTRACT_SHA256,
        )
        scorer = _load_exact_module(
            scorer_name,
            args.visual_scorer_source,
            expected_path=EXPECTED_VISUAL_SCORER_PATH,
            expected_sha256=EXPECTED_VISUAL_SCORER_SHA256,
        )
    except Exception:
        sys.modules.pop(scorer_name, None)
        sys.modules.pop(contract_name, None)
        raise
    finally:
        sys.path[:] = original_sys_path
    if (
        getattr(contract, "__file__", None) != EXPECTED_VISUAL_CONTRACT_PATH
        or getattr(scorer, "__file__", None) != EXPECTED_VISUAL_SCORER_PATH
        or getattr(scorer, "contract", None) is not contract
    ):
        raise Source8MatrixError("visual scorer/contract import closure differs")
    raw_spec, spec_sha = _strict_json(
        args.evaluator_spec,
        expected_sha256=EXPECTED_EVALUATOR_SPEC_SHA256,
        expected_path=EXPECTED_EVALUATOR_SPEC_PATH,
        label="sealed visual evaluator spec",
    )
    try:
        spec = contract.validate_evaluator_spec(raw_spec)
    except Exception as error:
        raise Source8MatrixError(
            f"sealed visual evaluator spec failed validation: {error}"
        ) from error
    if (
        spec.get("implementation_sha256") != EXPECTED_VISUAL_SCORER_SHA256
        or spec.get("contract_sha256") != EXPECTED_VISUAL_CONTRACT_SHA256
    ):
        raise Source8MatrixError("visual evaluator spec/source binding differs")
    try:
        checkpoint = scorer.verify_checkpoint_content(
            args.visual_checkpoint,
            args.visual_checkpoint_manifest,
            evaluator_spec=spec,
        )
    except Exception as error:
        raise Source8MatrixError(f"sealed visual checkpoint failed validation: {error}") from error
    versions = scorer.runtime_versions()
    if versions != EXPECTED_RUNTIME_VERSIONS or versions != spec.get("runtime_versions"):
        raise Source8MatrixError("visual evaluator runtime versions differ")
    processor = checkpoint.pop("processor", None)
    if processor is None:
        raise Source8MatrixError("verified checkpoint omitted official processor")
    return {
        "processor": processor,
        "spec": spec,
        "scorer": scorer,
        "contract": contract,
        "spec_raw_sha256": spec_sha,
    }, checkpoint


def _configure_device() -> Any:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Source8MatrixError("each worker must see exactly one ROCm/CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def _add_visual_arguments(parser: Any) -> None:
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


def _checkpoint_manifest_evidence() -> dict[str, Any]:
    path = _plain_file(
        EXPECTED_CHECKPOINT_MANIFEST_PATH,
        label="visual checkpoint content manifest",
        expected_path=EXPECTED_CHECKPOINT_MANIFEST_PATH,
    )
    before = path.stat()
    if file_sha256(path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
        raise Source8MatrixError("visual checkpoint content-manifest raw SHA-256 differs")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise Source8MatrixError("cannot parse visual checkpoint content manifest") from error
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        match = _CHECKPOINT_MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise Source8MatrixError("checkpoint manifest line is not canonical")
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise Source8MatrixError("checkpoint manifest path escapes root")
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in {"", "."})
        ).as_posix()
        if not normalized or normalized in seen:
            raise Source8MatrixError("checkpoint manifest path is empty or duplicate")
        seen.add(normalized)
        entries.append({"path": normalized, "sha256": digest})
    entries.sort(key=lambda row: row["path"])
    after = path.stat()
    if (
        not entries
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or file_sha256(path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256
    ):
        raise Source8MatrixError("checkpoint manifest is empty or changed while reading")
    return {
        "checkpoint_file_count": len(entries),
        "verified_entries_digest": object_sha256(entries),
    }


def _frozen_pins_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise Source8MatrixError("sealed evaluator spec is absent")
    model = _closed(spec.get("model"), _MODEL_FIELDS, label="sealed evaluator model")
    runtime = _closed(
        spec.get("runtime_versions"), set(EXPECTED_RUNTIME_VERSIONS),
        label="sealed evaluator runtime",
    )
    checkpoint_manifest = _checkpoint_manifest_evidence()
    for field in (
        "checkpoint_manifest_sha256", "checkpoint_config_sha256",
        "preprocessor_config_sha256", "preprocessor_golden_input_sha256",
        "preprocessor_golden_output_sha256",
    ):
        _sha256(model.get(field), label=f"sealed evaluator model {field}")
    if (
        spec.get("implementation_sha256") != EXPECTED_VISUAL_SCORER_SHA256
        or spec.get("contract_sha256") != EXPECTED_VISUAL_CONTRACT_SHA256
        or dict(runtime) != EXPECTED_RUNTIME_VERSIONS
        or model.get("checkpoint_manifest_sha256")
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or model.get("checkpoint_file_count")
        != checkpoint_manifest["checkpoint_file_count"]
        or model.get("preprocessor_golden_input_sha256")
        != EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256
        or model.get("preprocessor_golden_output_sha256")
        != EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256
        or model.get("num_register_tokens") != 0
        or model.get("image_size") != 518
        or model.get("patch_size") != 14
        or model.get("preprocessor_golden_output_shape") != [1, 3, 224, 224]
        or not isinstance(model.get("adapter_id"), str)
        or not model["adapter_id"]
        or not isinstance(model.get("architecture_id"), str)
        or not model["architecture_id"]
    ):
        raise Source8MatrixError("frozen runtime/model/golden closure differs")
    return {
        "checkpoint_root": EXPECTED_CHECKPOINT_ROOT,
        "checkpoint_manifest_path": EXPECTED_CHECKPOINT_MANIFEST_PATH,
        "evaluator_spec_path": EXPECTED_EVALUATOR_SPEC_PATH,
        "visual_scorer_path": EXPECTED_VISUAL_SCORER_PATH,
        "visual_contract_path": EXPECTED_VISUAL_CONTRACT_PATH,
        "evaluator_spec_sha256": EXPECTED_EVALUATOR_SPEC_SHA256,
        "visual_scorer_sha256": EXPECTED_VISUAL_SCORER_SHA256,
        "visual_contract_sha256": EXPECTED_VISUAL_CONTRACT_SHA256,
        "runtime_versions": dict(EXPECTED_RUNTIME_VERSIONS),
        "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_verified_entries_digest": checkpoint_manifest[
            "verified_entries_digest"
        ],
        "model_adapter_id": model["adapter_id"],
        "model_architecture_id": model["architecture_id"],
        "checkpoint_config_sha256": model["checkpoint_config_sha256"],
        "preprocessor_config_sha256": model["preprocessor_config_sha256"],
        "checkpoint_file_count": model["checkpoint_file_count"],
        "num_register_tokens": 0,
        "model_image_size": 518,
        "patch_size": 14,
        "preprocessor_golden_input_sha256": EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256,
        "preprocessor_golden_output_sha256": EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256,
        "preprocessor_golden_output_shape": [1, 3, 224, 224],
        "selected_frame_indices": list(EVAL_FRAME_INDICES),
        "feature_geometry": dict(EXPECTED_FEATURE_GEOMETRY),
    }


def _frozen_pins_from_sealed_spec() -> dict[str, Any]:
    spec, _ = _strict_json(
        EXPECTED_EVALUATOR_SPEC_PATH,
        expected_sha256=EXPECTED_EVALUATOR_SPEC_SHA256,
        expected_path=EXPECTED_EVALUATOR_SPEC_PATH,
        label="sealed visual evaluator spec",
    )
    return _frozen_pins_from_spec(spec)


def _validate_evaluator_pins(args: Any, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    if (
        str(args.visual_checkpoint) != EXPECTED_CHECKPOINT_ROOT
        or str(args.visual_checkpoint_manifest) != EXPECTED_CHECKPOINT_MANIFEST_PATH
        or str(args.evaluator_spec) != EXPECTED_EVALUATOR_SPEC_PATH
        or str(args.visual_scorer_source) != EXPECTED_VISUAL_SCORER_PATH
        or str(args.visual_contract_source) != EXPECTED_VISUAL_CONTRACT_PATH
        or args.expected_evaluator_spec_sha256 != EXPECTED_EVALUATOR_SPEC_SHA256
        or args.expected_visual_scorer_sha256 != EXPECTED_VISUAL_SCORER_SHA256
        or args.expected_visual_contract_sha256 != EXPECTED_VISUAL_CONTRACT_SHA256
    ):
        raise Source8MatrixError("frozen DINO evaluator identity differs")
    return _frozen_pins_from_spec(evaluator["spec"])


def _features(
    binding: Mapping[str, Any],
    *,
    scorer: Any,
    processor: Any,
    model: Any,
    device: Any,
    evaluator_spec: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    frames, raw_decode = scorer.decode_exact81_rgb(
        binding["source_video"], expected_sha256=binding["source_video_sha256"]
    )
    _, normalized = scorer.preprocess_selected_rgb(frames, processor)
    if raw_decode.get("preprocessed_tensor_sha256") != "0" * 64:
        raise Source8MatrixError("frozen scorer preprocess placeholder differs")
    decode = dict(raw_decode)
    decode["preprocessed_tensor_sha256"] = scorer.tensor_sha256(normalized)
    if decode["preprocessed_tensor_sha256"] == "0" * 64:
        raise Source8MatrixError("preprocessed tensor SHA-256 is an unfilled placeholder")
    model_spec = evaluator_spec["model"]
    global_feature, dense_feature, features = scorer.extract_features(
        model,
        normalized,
        device=device,
        num_register_tokens=model_spec["num_register_tokens"],
        evaluation_image_size=model_spec["preprocessor_golden_output_shape"][-1],
        patch_size=model_spec["patch_size"],
    )
    if (
        list(global_feature.shape) != [17, 768]
        or list(dense_feature.shape) != [17, 256, 768]
        or getattr(getattr(global_feature, "device", None), "type", None) != "cpu"
        or getattr(getattr(dense_feature, "device", None), "type", None) != "cpu"
        or str(global_feature.dtype) != "torch.float32"
        or str(dense_feature.dtype) != "torch.float32"
    ):
        raise Source8MatrixError("exact CPU source feature tensor geometry differs")
    expected_features = {
        "global_feature_sha256", "dense_feature_sha256",
        *EXPECTED_FEATURE_GEOMETRY,
    }
    _closed(features, expected_features, label="registered source feature evidence")
    if {
        key: features.get(key) for key in EXPECTED_FEATURE_GEOMETRY
    } != EXPECTED_FEATURE_GEOMETRY:
        raise Source8MatrixError("registered source feature evidence geometry differs")
    _sha256(features.get("global_feature_sha256"), label="global feature SHA-256")
    _sha256(features.get("dense_feature_sha256"), label="dense feature SHA-256")
    return global_feature, dense_feature, {"decode": decode, "features": dict(features)}


def _validate_visual_evidence(
    value: Any, *, frozen_pins: Mapping[str, Any]
) -> dict[str, Any]:
    row = _closed(value, _VISUAL_EVIDENCE_FIELDS, label="visual evaluator evidence")
    model = _closed(
        row.get("model_evidence"), _MODEL_EVIDENCE_FIELDS,
        label="visual evaluator model evidence",
    )
    runtime = _closed(
        model.get("runtime_versions"), set(EXPECTED_RUNTIME_VERSIONS),
        label="visual evaluator model runtime",
    )
    for field in (
        "checkpoint_manifest_sha256", "checkpoint_config_sha256",
        "preprocessor_config_sha256", "verified_entries_digest",
        "preprocessor_golden_input_sha256", "preprocessor_golden_output_sha256",
        "parameter_metadata_digest",
    ):
        _sha256(model.get(field), label=f"visual evaluator {field}")
    for field in (
        "checkpoint_file_count", "trainable_parameter_tensors",
        "parameter_tensor_count", "parameter_element_count", "missing_key_count",
        "unexpected_key_count", "mismatched_key_count", "loading_error_count",
    ):
        if type(model.get(field)) is not int or model[field] < 0:
            raise Source8MatrixError(f"visual evaluator {field} is not a strict count")
    if (
        row.get("checkpoint_root") != EXPECTED_CHECKPOINT_ROOT
        or row.get("checkpoint_manifest_path") != EXPECTED_CHECKPOINT_MANIFEST_PATH
        or row.get("evaluator_spec_path") != EXPECTED_EVALUATOR_SPEC_PATH
        or row.get("visual_scorer_path") != EXPECTED_VISUAL_SCORER_PATH
        or row.get("visual_contract_path") != EXPECTED_VISUAL_CONTRACT_PATH
        or row.get("evaluator_spec_sha256") != EXPECTED_EVALUATOR_SPEC_SHA256
        or row.get("visual_scorer_sha256") != EXPECTED_VISUAL_SCORER_SHA256
        or row.get("visual_contract_sha256") != EXPECTED_VISUAL_CONTRACT_SHA256
        or row.get("checkpoint_manifest_raw_sha256")
        != EXPECTED_CHECKPOINT_MANIFEST_SHA256
        or row.get("model_evidence_sha256") != object_sha256(model)
        or row.get("candidate_or_proposal_media_consulted") is not False
        or row.get("candidate_metric_fields_queried") is not False
        or row.get("candidate_metric_values_used") is not False
        or row.get("identity_authority") is not False
        or row.get("scientific_claim_authorized") is not False
        or model.get("adapter_id") != frozen_pins["model_adapter_id"]
        or model.get("architecture_id") != frozen_pins["model_architecture_id"]
        or model.get("checkpoint_manifest_sha256")
        != frozen_pins["checkpoint_manifest_sha256"]
        or model.get("checkpoint_config_sha256")
        != frozen_pins["checkpoint_config_sha256"]
        or model.get("preprocessor_config_sha256")
        != frozen_pins["preprocessor_config_sha256"]
        or model.get("checkpoint_file_count") != frozen_pins["checkpoint_file_count"]
        or model.get("verified_entries_digest")
        != frozen_pins["checkpoint_verified_entries_digest"]
        or model.get("preprocessor_golden_input_sha256")
        != EXPECTED_PREPROCESSOR_GOLDEN_INPUT_SHA256
        or model.get("preprocessor_golden_output_sha256")
        != EXPECTED_PREPROCESSOR_GOLDEN_OUTPUT_SHA256
        or model.get("preprocessor_golden_output_shape") != [1, 3, 224, 224]
        or model.get("every_checkpoint_file_verified") is not True
        or model.get("all_parameters_frozen") is not True
        or model.get("trainable_parameter_tensors") != 0
        or model.get("parameter_tensor_count", 0) <= 0
        or model.get("parameter_element_count", 0) <= 0
        or any(model.get(field) != 0 for field in (
            "missing_key_count", "unexpected_key_count", "mismatched_key_count",
            "loading_error_count",
        ))
        or dict(runtime) != EXPECTED_RUNTIME_VERSIONS
    ):
        raise Source8MatrixError("visual evaluator evidence/spec/path closure differs")
    return {**dict(row), "model_evidence": {**dict(model), "runtime_versions": dict(runtime)}}


def _build_visual_evidence(
    *,
    evaluator: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model: Any,
    loading_counts: Mapping[str, int],
    frozen_pins: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        observed = evaluator["scorer"].model_evidence(
            checkpoint,
            evaluator_spec=evaluator["spec"],
            model=model,
            loading_counts=loading_counts,
        )
        checked = evaluator["contract"]._validate_model_evidence(
            observed, spec=evaluator["spec"]
        )
    except Exception as error:
        raise Source8MatrixError(f"visual evaluator model evidence differs: {error}") from error
    if observed != checked:
        raise Source8MatrixError("visual model evidence canonical projection differs")
    evidence = {
        "checkpoint_root": EXPECTED_CHECKPOINT_ROOT,
        "checkpoint_manifest_path": EXPECTED_CHECKPOINT_MANIFEST_PATH,
        "evaluator_spec_path": EXPECTED_EVALUATOR_SPEC_PATH,
        "visual_scorer_path": EXPECTED_VISUAL_SCORER_PATH,
        "visual_contract_path": EXPECTED_VISUAL_CONTRACT_PATH,
        "evaluator_spec_sha256": EXPECTED_EVALUATOR_SPEC_SHA256,
        "visual_scorer_sha256": EXPECTED_VISUAL_SCORER_SHA256,
        "visual_contract_sha256": EXPECTED_VISUAL_CONTRACT_SHA256,
        "checkpoint_manifest_raw_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "model_evidence": dict(checked),
        "model_evidence_sha256": object_sha256(checked),
        "candidate_or_proposal_media_consulted": False,
        "candidate_metric_fields_queried": False,
        "candidate_metric_values_used": False,
        "identity_authority": False,
        "scientific_claim_authorized": False,
    }
    return _validate_visual_evidence(evidence, frozen_pins=frozen_pins)


def _feature_cache(
    manifest: Mapping[str, Any],
    *,
    evaluator: Mapping[str, Any],
    model: Any,
    device: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    entries = []
    spec = evaluator["spec"]
    for ordinal, binding in enumerate(manifest["sources"]):
        global_feature, dense_feature, observed = _features(
            binding,
            scorer=evaluator["scorer"],
            processor=evaluator["processor"],
            model=model,
            device=device,
            evaluator_spec=spec,
        )
        features = observed["features"]
        geometry = {
            key: features.get(key) for key in EXPECTED_FEATURE_GEOMETRY
        }
        if geometry != EXPECTED_FEATURE_GEOMETRY:
            raise Source8MatrixError("registered source feature geometry differs")
        entry = {
            "ordinal": ordinal,
            "iid": binding["iid"],
            "actor_family": binding["actor_family"],
            "source_video": binding["source_video"],
            "source_video_sha256": binding["source_video_sha256"],
            "global_feature_sha256": features["global_feature_sha256"],
            "dense_feature_sha256": features["dense_feature_sha256"],
            "decode": observed["decode"],
            "feature_geometry": geometry,
        }
        cache[binding["iid"]] = {
            "global": global_feature, "dense": dense_feature, "entry": entry,
        }
        entries.append(entry)
    feature_hash_map = [{
        "iid": row["iid"],
        "source_video_sha256": row["source_video_sha256"],
        "global_feature_sha256": row["global_feature_sha256"],
        "dense_feature_sha256": row["dense_feature_sha256"],
    } for row in entries]
    summary = {
        "cache_scope": "one_rank_process_exact8",
        "source_count": len(entries),
        "source_manifest_order": list(manifest["source_manifest_order"]),
        "all_exact8_sources_warmed_before_pair_computation": True,
        "source_features_held_in_cpu_memory_until_worker_exit": True,
        "source_files_retained_open_until_worker_exit": False,
        "entries": entries,
        "entries_sha256": core.object_sha256(entries),
        "feature_hash_map": feature_hash_map,
        "feature_hash_map_sha256": core.object_sha256(feature_hash_map),
    }
    if len(cache) != EXPECTED_SOURCE_COUNT:
        raise Source8MatrixError("source cache is not exact8")
    return cache, summary


def _similarity(left: Any, right: Any, *, dense: bool) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise Source8MatrixError("pair feature geometry differs")
    mapped = (((left * right).sum(dim=-1) + 1.0) * 0.5).clamp(0.0, 1.0)
    value = mapped.reshape(-1).median() if dense else mapped.mean()
    return _finite(value.item(), label="mapped cosine")


def pair_row(
    rank: int,
    manifest: Mapping[str, Any],
    cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sources = manifest["sources"]
    row_binding = sources[rank]
    row_features = cache[row_binding["iid"]]
    cells = []
    for column_ordinal, column_binding in enumerate(sources):
        column_features = cache[column_binding["iid"]]
        global_value = _similarity(
            row_features["global"], column_features["global"], dense=False
        )
        dense_value = _similarity(
            row_features["dense"], column_features["dense"], dense=True
        )
        diagonal = rank == column_ordinal
        if diagonal:
            if (
                not math.isclose(global_value, 1.0, rel_tol=0.0, abs_tol=SYMMETRY_ABS_TOLERANCE)
                or not math.isclose(dense_value, 1.0, rel_tol=0.0, abs_tol=SYMMETRY_ABS_TOLERANCE)
            ):
                raise Source8MatrixError("normalized source self differs from one")
            global_value = dense_value = 1.0
        same_actor = row_binding["actor_family"] == column_binding["actor_family"]
        cells.append({
            "matrix_ordinal": rank * EXPECTED_SOURCE_COUNT + column_ordinal,
            "row_ordinal": rank,
            "column_ordinal": column_ordinal,
            "row_source_iid": row_binding["iid"],
            "row_source_video_sha256": row_binding["source_video_sha256"],
            "column_source_iid": column_binding["iid"],
            "column_source_video_sha256": column_binding["source_video_sha256"],
            "relationship": "same_actor" if same_actor else "cross_actor",
            "diagonal": diagonal,
            "registered_all3_directed_pair": same_actor and not diagonal,
            "global_mean_mapped_cosine": global_value,
            "dense_median_mapped_cosine": dense_value,
            "thresholds": None,
            "authority": dict(AUTHORITY),
        })
    return {
        "row_ordinal": rank,
        "row_source_iid": row_binding["iid"],
        "row_source_video_sha256": row_binding["source_video_sha256"],
        "cell_count": EXPECTED_SOURCE_COUNT,
        "cells": cells,
        "candidate_or_proposal_media_or_metrics_consulted": False,
        "authority": dict(AUTHORITY),
    }


def _worker_common(args: Any) -> tuple[Any, ...]:
    source_sha = _verify_self(args.expected_source_sha256)
    output_root = core._plain_directory(args.output_root, label="output root")
    if str(args.input_manifest) != str(output_root / "input-manifest.json"):
        raise Source8MatrixError("worker input-manifest lexical path differs")
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    evaluator, checkpoint = core._load_evaluator(args)
    frozen_pins = _validate_evaluator_pins(args, evaluator)
    device = core._configure_device()
    model, loading_counts = evaluator["scorer"].load_frozen_model(
        checkpoint, device=device
    )
    visual_evidence = _build_visual_evidence(
        evaluator=evaluator,
        checkpoint=checkpoint,
        model=model,
        loading_counts=loading_counts,
        frozen_pins=frozen_pins,
    )
    cache, cache_summary = _feature_cache(
        manifest, evaluator=evaluator, model=model, device=device
    )
    _validate_cache(cache_summary, manifest=manifest)
    return (
        source_sha, manifest, manifest_sha, visual_evidence, frozen_pins,
        cache, cache_summary,
    )


def _write_rank(args: Any, *, preflight: bool) -> int:
    (
        source_sha, manifest, manifest_sha, visual_evidence, frozen_pins,
        cache, cache_summary,
    ) = _worker_common(args)
    rank = core._rank(args.rank, world_size=EXPECTED_WORLD_SIZE)
    if not preflight and args.world_size != EXPECTED_WORLD_SIZE:
        raise Source8MatrixError("worker world size must be exactly eight")
    row = pair_row(rank, manifest, cache)
    _validate_row(row, rank=rank, manifest=manifest)
    unsigned = {
        "schema_version": PREFLIGHT_SCHEMA if preflight else SHARD_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "rank": rank,
        "world_size": EXPECTED_WORLD_SIZE,
        "exact8_source_feature_cache": cache_summary,
        "pair_matrix_row": row,
        "visual_evaluator": visual_evidence,
        "visual_evaluator_projection_sha256": core.object_sha256(visual_evidence),
        "frozen_runtime_and_golden_pins": frozen_pins,
        "limitation": dict(LIMITATION),
        "authority": dict(AUTHORITY),
    }
    name = f"preflight-rank-{rank:02d}.json" if preflight else f"shard-{rank:02d}-of-08.json"
    core._write_create_only(
        core._plain_directory(args.output_root, label="output root") / name,
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def preflight(args: Any) -> int:
    return _write_rank(args, preflight=True)


def worker(args: Any) -> int:
    return _write_rank(args, preflight=False)


def _validate_cache(
    summary: Any,
    *,
    manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(summary, Mapping):
        raise Source8MatrixError("rank cache summary is absent")
    entries = summary.get("entries")
    feature_map = summary.get("feature_hash_map")
    core._closed(summary, {
        "cache_scope", "source_count", "source_manifest_order",
        "all_exact8_sources_warmed_before_pair_computation",
        "source_features_held_in_cpu_memory_until_worker_exit",
        "source_files_retained_open_until_worker_exit", "entries",
        "entries_sha256", "feature_hash_map", "feature_hash_map_sha256",
    }, label="rank exact8 source cache")
    if (
        summary.get("cache_scope") != "one_rank_process_exact8"
        or summary.get("source_count") != EXPECTED_SOURCE_COUNT
        or summary.get("source_manifest_order") != manifest["source_manifest_order"]
        or summary.get("all_exact8_sources_warmed_before_pair_computation") is not True
        or summary.get("source_features_held_in_cpu_memory_until_worker_exit") is not True
        or summary.get("source_files_retained_open_until_worker_exit") is not False
        or not isinstance(entries, list) or len(entries) != EXPECTED_SOURCE_COUNT
        or summary.get("entries_sha256") != core.object_sha256(entries)
        or not isinstance(feature_map, list) or len(feature_map) != EXPECTED_SOURCE_COUNT
        or summary.get("feature_hash_map_sha256") != core.object_sha256(feature_map)
    ):
        raise Source8MatrixError("rank exact8 cache contract differs")
    for ordinal, (entry, binding, hashes) in enumerate(
        zip(entries, manifest["sources"], feature_map)
    ):
        core._closed(entry, {
            "ordinal", "iid", "actor_family", "source_video",
            "source_video_sha256",
            "global_feature_sha256", "dense_feature_sha256", "decode",
            "feature_geometry",
        }, label=f"rank cached source {ordinal}")
        core._closed(hashes, {
            "iid", "source_video_sha256", "global_feature_sha256",
            "dense_feature_sha256",
        }, label=f"rank source feature hash {ordinal}")
        decode = entry.get("decode")
        core._closed(decode, {
            "artifact_sha256", "decoded_rgb_sha256", "frame_count",
            "fps_numerator", "fps_denominator", "time_base_numerator",
            "time_base_denominator", "pts_step", "pts_sha256", "width",
            "height", "selected_frame_indices", "selected_rgb_sha256",
            "preprocessed_tensor_sha256",
        }, label=f"rank cached source decode {ordinal}")
        global_feature_sha = core._sha256(
            entry.get("global_feature_sha256"),
            label=f"rank cached source {ordinal} global feature SHA-256",
        )
        dense_feature_sha = core._sha256(
            entry.get("dense_feature_sha256"),
            label=f"rank cached source {ordinal} dense feature SHA-256",
        )
        for field in (
            "artifact_sha256", "decoded_rgb_sha256", "pts_sha256",
            "selected_rgb_sha256", "preprocessed_tensor_sha256",
        ):
            digest = core._sha256(
                decode.get(field), label=f"rank cached source {ordinal} {field}"
            )
            if digest == "0" * 64:
                raise Source8MatrixError(
                    f"rank cached source {ordinal} {field} is an unfilled placeholder"
                )
        if (
            entry.get("ordinal") != ordinal
            or entry.get("iid") != binding["iid"]
            or entry.get("actor_family") != binding["actor_family"]
            or entry.get("source_video") != binding["source_video"]
            or entry.get("source_video_sha256") != binding["source_video_sha256"]
            or entry.get("feature_geometry") != EXPECTED_FEATURE_GEOMETRY
            or decode.get("artifact_sha256") != binding["source_video_sha256"]
            or decode.get("frame_count") != 81
            or decode.get("fps_numerator") != 25
            or decode.get("fps_denominator") != 1
            or decode.get("selected_frame_indices") != list(core.EVAL_FRAME_INDICES)
            or decode.get("preprocessed_tensor_sha256") == "0" * 64
            or any(
                type(decode.get(field)) is not int or decode[field] <= 0
                for field in (
                    "time_base_numerator", "time_base_denominator", "pts_step",
                    "width", "height",
                )
            )
            or decode.get("time_base_numerator") * decode.get("pts_step") * 25
            != decode.get("time_base_denominator")
            or hashes != {
                "iid": binding["iid"],
                "source_video_sha256": binding["source_video_sha256"],
                "global_feature_sha256": global_feature_sha,
                "dense_feature_sha256": dense_feature_sha,
            }
        ):
            raise Source8MatrixError("cached source IID/SHA/ordinal binding differs")
    return feature_map


def _validate_row(
    row: Any,
    *,
    rank: int,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    binding = manifest["sources"][rank]
    if isinstance(row, Mapping):
        core._closed(row, {
            "row_ordinal", "row_source_iid", "row_source_video_sha256",
            "cell_count", "cells",
            "candidate_or_proposal_media_or_metrics_consulted", "authority",
        }, label=f"matrix row {rank}")
    if (
        not isinstance(row, Mapping)
        or row.get("row_ordinal") != rank
        or row.get("row_source_iid") != binding["iid"]
        or row.get("row_source_video_sha256") != binding["source_video_sha256"]
        or row.get("cell_count") != EXPECTED_SOURCE_COUNT
        or row.get("candidate_or_proposal_media_or_metrics_consulted") is not False
        or row.get("authority") != AUTHORITY
        or not isinstance(row.get("cells"), list)
        or len(row["cells"]) != EXPECTED_SOURCE_COUNT
    ):
        raise Source8MatrixError("matrix row contract differs")
    design = manifest["matrix_registration"]["cells"]
    for column, cell in enumerate(row["cells"]):
        expected = design[rank * EXPECTED_SOURCE_COUNT + column]
        core._closed(cell, set(expected) | {
            "global_mean_mapped_cosine", "dense_median_mapped_cosine",
            "thresholds", "authority",
        }, label=f"matrix cell {rank},{column}")
        projection = {key: cell.get(key) for key in expected}
        global_value = _finite(cell.get("global_mean_mapped_cosine"), label="global cell")
        dense_value = _finite(cell.get("dense_median_mapped_cosine"), label="dense cell")
        if (
            projection != expected
            or cell.get("thresholds") is not None
            or cell.get("authority") != AUTHORITY
            or not 0.0 <= global_value <= 1.0
            or not 0.0 <= dense_value <= 1.0
            or (expected["diagonal"] and (global_value != 1.0 or dense_value != 1.0))
        ):
            raise Source8MatrixError("matrix cell IID/SHA/ordinal/value closure differs")
    return row["cells"]


def _legacy_regression(
    path_value: str | Path,
    expected_sha256: str,
    *,
    feature_map: Sequence[Mapping[str, Any]],
    registered_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    if expected_sha256 != EXPECTED_LEGACY_ALL3_AGGREGATE_SHA256:
        raise Source8MatrixError("caller r4 all-three aggregate pin differs")
    value, raw_sha = core._strict_json(
        path_value, expected_sha256=expected_sha256,
        expected_path=EXPECTED_LEGACY_ALL3_AGGREGATE_PATH,
        label="frozen r4 all-three aggregate",
    )
    unsigned = dict(value)
    declared = core._sha256(unsigned.pop("receipt_digest", None), label="legacy digest")
    cache = value.get("cross_rank_source_feature_cache_consistency", {})
    old_feature_map = cache.get("per_source_feature_hashes")
    executed_rows = value.get("executed_directed_source_pairs")
    if not isinstance(executed_rows, list):
        raise Source8MatrixError("frozen r4 all-three executed-pair rows are absent")
    executed_bindings: list[dict[str, str]] = []
    for ordinal, row in enumerate(executed_rows):
        checked = _closed(
            row,
            {"correct_source_iid", "negative_source_iid"},
            label=f"frozen r4 executed source pair {ordinal}",
        )
        correct_iid = checked.get("correct_source_iid")
        negative_iid = checked.get("negative_source_iid")
        if not isinstance(correct_iid, str) or not isinstance(negative_iid, str):
            raise Source8MatrixError("frozen r4 executed source-pair IID differs")
        executed_bindings.append({
            "correct_source_iid": correct_iid,
            "negative_source_iid": negative_iid,
        })
    executed = {
        (row.get("correct_source_iid"), row.get("negative_source_iid"))
        for row in executed_bindings
    }
    if (
        value.get("schema_version") != LEGACY_ALL3_AGGREGATE_SCHEMA
        or value.get("diagnostic_source_sha256") != LEGACY_ALL3_SOURCE_SHA256
        or value.get("world_size") != EXPECTED_WORLD_SIZE
        or value.get("candidate_count") != LEGACY_ALL3_CANDIDATE_COUNT
        or value.get("executed_directed_source_pair_count")
        != EXPECTED_EXECUTED_ALL3_DIRECTED_PAIR_COUNT
        or value.get("executed_directed_source_pairs_sha256")
        != core.object_sha256(executed_bindings)
        or old_feature_map != list(feature_map)
        or len(executed_bindings) != EXPECTED_EXECUTED_ALL3_DIRECTED_PAIR_COUNT
        or len(executed) != EXPECTED_EXECUTED_ALL3_DIRECTED_PAIR_COUNT
        or not executed < registered_pairs
        or declared != core.object_sha256(unsigned)
    ):
        raise Source8MatrixError("frozen r4 all-three provenance regression differs")
    new_pairs = registered_pairs - executed
    if (
        len(new_pairs) != 3
        or {left for left, _ in new_pairs} != {EXPECTED_MISSING_CORRECT_SOURCE_IID}
    ):
        raise Source8MatrixError("new missing-correct source-pair closure differs")
    return {
        "path": str(core._plain_file(
            path_value,
            label="frozen r4 all-three aggregate",
            expected_path=EXPECTED_LEGACY_ALL3_AGGREGATE_PATH,
        )),
        "raw_sha256": raw_sha,
        "receipt_digest": declared,
        "matrix_was_registered_computed_and_sharded_before_legacy_aggregate_open": True,
        "legacy_aggregate_bytes_parsed": True,
        "legacy_candidate_bearing_bytes_included_only_in_whole_receipt_integrity_check": True,
        "candidate_metric_fields_queried": False,
        "candidate_metric_values_used": False,
        "exact_source_feature_hash_map_match": True,
        "executed_same_actor_directed_pair_binding_match_count": len(executed),
        "new_missing_correct_directed_pair_count": len(new_pairs),
        "new_missing_correct_source_iid": EXPECTED_MISSING_CORRECT_SOURCE_IID,
        "new_pairs": [
            {"row_source_iid": left, "column_source_iid": right}
            for left, right in sorted(new_pairs)
        ],
        "executed_pairs": [
            {"row_source_iid": left, "column_source_iid": right}
            for left, right in sorted(executed)
        ],
    }


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(_finite(value, label="descriptive matrix value") for value in values)
    if not ordered:
        raise Source8MatrixError("descriptive matrix set is empty")
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) * 0.5
    )
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "maximum": ordered[-1],
        "descriptive_only": True,
        "thresholds": None,
        "ranking_authorized": False,
        "scientific_claim_authorized": False,
    }


def aggregate(args: Any) -> int:
    source_sha = _verify_self(args.expected_source_sha256)
    output_root = core._plain_directory(args.output_root, label="output root")
    if str(args.input_manifest) != str(output_root / "input-manifest.json"):
        raise Source8MatrixError("aggregate input-manifest lexical path differs")
    manifest, manifest_sha = load_input_manifest(
        args.input_manifest,
        expected_sha256=args.expected_input_manifest_sha256,
        expected_source_sha256=source_sha,
    )
    expected_frozen_pins = _frozen_pins_from_sealed_spec()
    rows, shards, reference_feature_map = [], [], None
    reference_visual_evaluator = None
    reference_cache_entries = None
    rank_feature_receipts = []
    rank_cache_entry_receipts = []
    rank_visual_evaluator_receipts = []
    # The old candidate-bearing aggregate path is intentionally not touched in
    # this shard loop.  It is opened only after all eight new rows pass closure.
    for rank in range(EXPECTED_WORLD_SIZE):
        path = output_root / f"shard-{rank:02d}-of-08.json"
        value, raw_sha = core._strict_json(path, expected_sha256=None, label=f"shard {rank}")
        unsigned = dict(value)
        declared = core._sha256(unsigned.pop("receipt_digest", None), label="shard digest")
        core._closed(value, {
            "schema_version", "diagnostic_source_sha256", "input_manifest_sha256",
            "rank", "world_size", "exact8_source_feature_cache",
            "pair_matrix_row", "visual_evaluator",
            "visual_evaluator_projection_sha256",
            "frozen_runtime_and_golden_pins", "limitation", "authority",
            "receipt_digest",
        }, label=f"shard {rank}")
        if (
            value.get("schema_version") != SHARD_SCHEMA
            or value.get("diagnostic_source_sha256") != source_sha
            or value.get("input_manifest_sha256") != manifest_sha
            or value.get("rank") != rank
            or value.get("world_size") != EXPECTED_WORLD_SIZE
            or value.get("frozen_runtime_and_golden_pins") != expected_frozen_pins
            or value.get("limitation") != LIMITATION
            or value.get("authority") != AUTHORITY
            or declared != core.object_sha256(unsigned)
        ):
            raise Source8MatrixError(f"shard {rank} envelope differs")
        visual_evaluator = _validate_visual_evidence(
            value.get("visual_evaluator"), frozen_pins=expected_frozen_pins
        )
        visual_projection_sha = core._sha256(
            value.get("visual_evaluator_projection_sha256"),
            label=f"shard {rank} visual evaluator projection SHA-256",
        )
        if visual_projection_sha != core.object_sha256(visual_evaluator):
            raise Source8MatrixError("shard visual evaluator projection digest differs")
        if reference_visual_evaluator is None:
            reference_visual_evaluator = visual_evaluator
        elif visual_evaluator != reference_visual_evaluator:
            raise Source8MatrixError("visual evaluator evidence differs across ranks")
        rank_visual_evaluator_receipts.append({
            "rank": rank,
            "visual_evaluator_projection_sha256": visual_projection_sha,
        })
        feature_map = _validate_cache(value.get("exact8_source_feature_cache"), manifest=manifest)
        cache_entries = value["exact8_source_feature_cache"]["entries"]
        if reference_feature_map is None:
            reference_feature_map = feature_map
            reference_cache_entries = cache_entries
        elif feature_map != reference_feature_map:
            raise Source8MatrixError("exact8 source feature hashes differ across ranks")
        elif cache_entries != reference_cache_entries:
            raise Source8MatrixError("exact8 source decode/geometry entries differ across ranks")
        cells = _validate_row(value.get("pair_matrix_row"), rank=rank, manifest=manifest)
        rows.append(value["pair_matrix_row"])
        rank_feature_receipts.append({
            "rank": rank, "feature_hash_map_sha256": core.object_sha256(feature_map)
        })
        rank_cache_entry_receipts.append({
            "rank": rank,
            "decode_geometry_entries_sha256": core.object_sha256(cache_entries),
        })
        shards.append({
            "rank": rank, "path": str(path.resolve(strict=True)),
            "sha256": raw_sha, "receipt_digest": declared,
        })
    cells = [cell for row in rows for cell in row["cells"]]
    if [cell["matrix_ordinal"] for cell in cells] != list(range(EXPECTED_MATRIX_CELL_COUNT)):
        raise Source8MatrixError("aggregate matrix row-major order differs")
    same_actor = [cell for cell in cells if cell["relationship"] == "same_actor"]
    cross_actor = [cell for cell in cells if cell["relationship"] == "cross_actor"]
    diagonal = [cell for cell in cells if cell["diagonal"]]
    registered = {
        (cell["row_source_iid"], cell["column_source_iid"])
        for cell in cells if cell["registered_all3_directed_pair"]
    }
    if (
        len(cells) != EXPECTED_MATRIX_CELL_COUNT
        or len(same_actor) != EXPECTED_SAME_ACTOR_CELL_COUNT
        or len(cross_actor) != EXPECTED_CROSS_ACTOR_CELL_COUNT
        or len(diagonal) != EXPECTED_DIAGONAL_COUNT
        or len(registered) != EXPECTED_REGISTERED_ALL3_DIRECTED_PAIR_COUNT
    ):
        raise Source8MatrixError("aggregate matrix counts differ")
    by_coordinate = {
        (cell["row_ordinal"], cell["column_ordinal"]): cell for cell in cells
    }
    global_differences, dense_differences = [], []
    same_actor_unordered, cross_actor_unordered = [], []
    for left in range(EXPECTED_SOURCE_COUNT):
        for right in range(left + 1, EXPECTED_SOURCE_COUNT):
            forward, reverse = by_coordinate[left, right], by_coordinate[right, left]
            global_difference = abs(
                forward["global_mean_mapped_cosine"] - reverse["global_mean_mapped_cosine"]
            )
            dense_difference = abs(
                forward["dense_median_mapped_cosine"] - reverse["dense_median_mapped_cosine"]
            )
            if (
                global_difference > SYMMETRY_ABS_TOLERANCE
                or dense_difference > SYMMETRY_ABS_TOLERANCE
            ):
                raise Source8MatrixError("independent directed pair symmetry differs")
            global_differences.append(global_difference)
            dense_differences.append(dense_difference)
            target = (
                same_actor_unordered
                if forward["relationship"] == "same_actor"
                else cross_actor_unordered
            )
            target.append(forward)
    if len(same_actor_unordered) != 12 or len(cross_actor_unordered) != 16:
        raise Source8MatrixError("unordered actor pair counts differ")
    # Only now, after the new matrix is independently registered, computed, sealed,
    # and validated, open the old candidate-bearing aggregate for provenance-only
    # feature-hash and executed-pair binding regression.
    legacy = _legacy_regression(
        args.legacy_all3_aggregate,
        args.expected_legacy_all3_aggregate_sha256,
        feature_map=reference_feature_map,
        registered_pairs=registered,
    )
    executed_pairs = {
        (row["row_source_iid"], row["column_source_iid"])
        for row in legacy["executed_pairs"]
    }
    registered_pair_execution_policy = [{
        "row_source_iid": left,
        "column_source_iid": right,
        "registered_in_all3_pair_universe": True,
        "executed_by_r4_candidate_bank": (left, right) in executed_pairs,
        "new_source_only_information_for_missing_correct_iid":
            left == EXPECTED_MISSING_CORRECT_SOURCE_IID,
        "candidate_metric_fields_queried": False,
        "candidate_metric_values_used": False,
    } for left, right in sorted(registered)]
    if (
        len(registered_pair_execution_policy)
        != EXPECTED_REGISTERED_ALL3_DIRECTED_PAIR_COUNT
        or sum(row["executed_by_r4_candidate_bank"] for row in registered_pair_execution_policy)
        != EXPECTED_EXECUTED_ALL3_DIRECTED_PAIR_COUNT
        or sum(
            row["new_source_only_information_for_missing_correct_iid"]
            for row in registered_pair_execution_policy
        ) != 3
    ):
        raise Source8MatrixError("registered/executed pair policy annotation differs")
    unsigned = {
        "schema_version": AGGREGATE_SCHEMA,
        "diagnostic_source_sha256": source_sha,
        "input_manifest_sha256": manifest_sha,
        "world_size": EXPECTED_WORLD_SIZE,
        "matrix_shape": [8, 8],
        "matrix_cell_count": len(cells),
        "same_actor_cell_count_including_diagonal": len(same_actor),
        "cross_actor_cell_count": len(cross_actor),
        "diagonal_cell_count": len(diagonal),
        "registered_all3_directed_pair_count": len(registered),
        "registered_all3_directed_pair_execution_policy": registered_pair_execution_policy,
        "source_manifest_order": list(manifest["source_manifest_order"]),
        "pinned_local_source_closure": dict(PINNED_LOCAL_SOURCE_CLOSURE),
        "pinned_local_source_closure_sha256": core.object_sha256(PINNED_LOCAL_SOURCE_CLOSURE),
        "shards": shards,
        "frozen_runtime_and_golden_pins": expected_frozen_pins,
        "visual_evaluator_evidence_projection": reference_visual_evaluator,
        "visual_evaluator_evidence_projection_sha256": core.object_sha256(
            reference_visual_evaluator
        ),
        "per_rank_visual_evaluator_projection_receipts":
            rank_visual_evaluator_receipts,
        "all8_visual_evaluator_projections_identical": True,
        "cross_rank_source_feature_cache_consistency": {
            "source_count": EXPECTED_SOURCE_COUNT,
            "rank_count": EXPECTED_WORLD_SIZE,
            "per_source_feature_hashes": reference_feature_map,
            "per_source_decode_geometry_entries": reference_cache_entries,
            "per_source_decode_geometry_entries_sha256": core.object_sha256(reference_cache_entries),
            "per_rank_feature_hash_map_receipts": rank_feature_receipts,
            "per_rank_decode_geometry_entries_receipts": rank_cache_entry_receipts,
            "all_exact8_source_feature_hashes_identical_across_all8_ranks": True,
            "all_exact8_source_decode_geometry_entries_identical_across_all8_ranks": True,
        },
        "symmetry_recomputation": {
            "unordered_off_diagonal_pair_count": 28,
            "absolute_tolerance": SYMMETRY_ABS_TOLERANCE,
            "global_max_absolute_difference": max(global_differences),
            "dense_max_absolute_difference": max(dense_differences),
            "all_global_and_dense_pairs_symmetric_within_tolerance": True,
        },
        "descriptive_pair_hardness": {
            "same_actor_off_diagonal_unordered_pair_count": 12,
            "cross_actor_unordered_pair_count": 16,
            "same_actor_global_mean_mapped_cosine": _descriptive([
                row["global_mean_mapped_cosine"] for row in same_actor_unordered
            ]),
            "same_actor_dense_median_mapped_cosine": _descriptive([
                row["dense_median_mapped_cosine"] for row in same_actor_unordered
            ]),
            "cross_actor_global_mean_mapped_cosine": _descriptive([
                row["global_mean_mapped_cosine"] for row in cross_actor_unordered
            ]),
            "cross_actor_dense_median_mapped_cosine": _descriptive([
                row["dense_median_mapped_cosine"] for row in cross_actor_unordered
            ]),
            "descriptive_only": True,
            "thresholds": None,
            "ranking_authorized": False,
            "scientific_claim_authorized": False,
        },
        "legacy_r4_all3_provenance_regression": legacy,
        "ordered_matrix_rows": rows,
        "interpretation": {
            "measurement": "raw exact8 registered-source frozen-DINO pair-hardness matrix",
            "global_metric": "mean_aligned_selected17_mapped_cosine",
            "dense_metric": "median_selected17_by_256_aligned_token_mapped_cosine",
            "all_eight_source_self_values_are_exactly_one": True,
            "candidate_or_proposal_media_consulted": False,
            "legacy_candidate_bearing_aggregate_bytes_parsed_after_matrix_validation": True,
            "candidate_metric_fields_queried": False,
            "candidate_metric_values_used": False,
            "thresholds": None,
            "no_ranking_selection_training_formal_or_scientific_authority": True,
        },
        "limitation": dict(LIMITATION),
        "authority": dict(AUTHORITY),
    }
    core._write_create_only(
        output_root / "aggregate-receipt.json",
        {**unsigned, "receipt_digest": core.object_sha256(unsigned)},
    )
    return 0


def _visual_args(parser: Any) -> None:
    core._add_visual_arguments(parser)


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--expected-source-manifest-sha256", required=True)
    build.add_argument("--expected-source-manifest-content-sha256", required=True)
    build.add_argument("--expected-source-validator-summary-sha256", required=True)
    build.add_argument("--expected-source-sha256", required=True)
    build.add_argument("--output-root", required=True)
    check = commands.add_parser("preflight")
    _visual_args(check)
    run = commands.add_parser("worker")
    _visual_args(run)
    run.add_argument("--world-size", required=True, type=int)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--input-manifest", required=True)
    combine.add_argument("--expected-input-manifest-sha256", required=True)
    combine.add_argument("--expected-source-sha256", required=True)
    combine.add_argument("--legacy-all3-aggregate", required=True)
    combine.add_argument("--expected-legacy-all3-aggregate-sha256", required=True)
    combine.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    return {
        "build-manifest": build_manifest,
        "preflight": preflight,
        "worker": worker,
        "aggregate": aggregate,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
