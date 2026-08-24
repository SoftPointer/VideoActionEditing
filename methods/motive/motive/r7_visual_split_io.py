"""Strict committed-artifact glue for the R7 joint visual split.

The numerical graph construction lives in :mod:`motive.r7_visual_split`.
This module is deliberately limited to provenance-preserving I/O:

* validate a complete eight-rank R7 preflight output, including ``final``;
* prove that ``final`` is the exact row/array merge of those eight shards;
* require one identical frozen-DINO contract on every shard;
* construct source/target :class:`~motive.r7_visual_split.R7VisualPair`
  values from committed video SHA-256, sampled-frame indices, pHashes, and
  DINO CLS arrays;
* build and independently audit a whole-component split;
* commit hash-bound assignment/component/summary/done artifacts.

The repeatedly inspected R7-P0/R5 181-row pilot may be used to exercise this
I/O contract only.  A ledger-relative ``evaluation_fresh`` flag is not a
claim of formal benchmark freshness.  Fresh evaluation requires a separate
preflight run over independently selected, previously unseen data.  Every
summary emitted here records that limitation and never authorizes a fresh
evaluation or generation experiment by itself.

Resume is validation-only.  It never fills in, repairs, or overwrites a
partial output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r7_preflight_extract import (
    FINAL_WORLD_SIZE,
    R7_ROW_SCHEMA,
    rank_directory,
    validate_final,
    validate_shard,
)
from .r7_visual_split import (
    R7DinoProvenance,
    R7VisualAsset,
    R7VisualPair,
    R7VisualSplitConfig,
    R7VisualSplitResult,
    audit_r7_visual_split,
    build_r7_visual_split,
)


R7_VISUAL_SPLIT_IO_SCHEMA = "motive-r7-visual-split-io-v1"
R7_VISUAL_ASSIGNMENT_ROW_SCHEMA = "motive-r7-visual-assignment-row-v1"
R7_VISUAL_COMPONENT_ROW_SCHEMA = "motive-r7-visual-component-row-v1"
R7_VISUAL_SPLIT_SUMMARY_SCHEMA = "motive-r7-visual-split-summary-v1"
R7_VISUAL_SPLIT_DONE_SCHEMA = "motive-r7-visual-split-done-v1"
R7_VISUAL_SPLIT_IO_VERSION = (
    "complete-8-shard-final-crosscheck-atomic-v1"
)

ASSIGNMENTS_NAME = "assignments.jsonl"
COMPONENTS_NAME = "components.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
_EXPECTED_OUTPUT_NAMES = frozenset(
    {ASSIGNMENTS_NAME, COMPONENTS_NAME, SUMMARY_NAME, DONE_NAME}
)
_SIDES = ("source", "target")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: Any) -> Any:
    """Return the exact JSON-domain representation of ``value``."""

    return json.loads(_canonical_json(value))


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish one fully fsynced file without ever replacing a target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publish is atomic and, unlike os.replace, fails if a
        # concurrent process has created the destination.
        os.link(temporary, path)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    payload = "".join(
        _canonical_json(dict(row)) + "\n" for row in rows
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_canonical_jsonl(
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            if line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no rows")
    return rows


def _normalized_iid(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} IID must be a string")
    iid = value.strip()
    if not iid or iid != value or "\x00" in iid:
        raise ValueError(f"{context} IID is empty or not normalized")
    return iid


@dataclass(frozen=True)
class PriorIidLedger:
    path: Path
    file_sha256: str
    iids: tuple[str, ...]
    canonical_iids_sha256: str


def read_prior_iid_ledger(path: Path) -> PriorIidLedger:
    """Read a JSONL ledger/manifest containing an ``iid`` per row.

    A JSON string row is also accepted for a deliberately minimal IID
    ledger.  Blank rows, duplicate IIDs, and non-normalized values fail
    closed.  Exact bytes and the sorted canonical IID set are bound
    separately in output provenance.
    """

    ledger_path = path.expanduser().resolve(strict=True)
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    iids: list[str] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"{ledger_path}:{line_number} is blank"
                )
            value = json.loads(line)
            raw_iid = value.get("iid") if isinstance(value, dict) else value
            iid = _normalized_iid(
                raw_iid,
                context=f"{ledger_path}:{line_number}",
            )
            iids.append(iid)
    if len(set(iids)) != len(iids):
        raise ValueError("prior IID ledger contains duplicate IIDs")
    normalized = tuple(sorted(iids))
    return PriorIidLedger(
        path=ledger_path,
        file_sha256=_file_digest(ledger_path),
        iids=normalized,
        canonical_iids_sha256=_object_digest(list(normalized)),
    )


@dataclass(frozen=True)
class PreparedVisualInput:
    root: Path
    input_manifest: Path
    input_manifest_sha256: str
    pairs: tuple[R7VisualPair, ...]
    dino_provenance: R7DinoProvenance
    raw_dino_provenance: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _core_dino_provenance(
    raw: Mapping[str, Any],
) -> R7DinoProvenance:
    required = {
        "encoder_id",
        "encoder_revision",
        "weights_sha256",
        "frame_sampling_version",
        "preprocessing_version",
        "pooling",
        "embedding_dim",
        "dtype",
        "normalization",
        "frozen_encoder",
        "local_files_only",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"preflight DINO provenance is incomplete: missing={missing}"
        )
    if raw.get("normalization") != "l2-per-frame":
        raise ValueError(
            "preflight DINO normalization must be 'l2-per-frame'"
        )
    if raw.get("dtype") != "float32":
        raise ValueError("preflight DINO dtype must be float32")
    if raw.get("frozen_encoder") is not True:
        raise ValueError("preflight DINO encoder is not frozen")
    if raw.get("local_files_only") is not True:
        raise ValueError("preflight DINO provenance is not local-only")
    embedding_dim = raw.get("embedding_dim")
    if isinstance(embedding_dim, bool) or not isinstance(
        embedding_dim, int
    ):
        raise ValueError("preflight DINO embedding_dim is not an integer")
    provenance = R7DinoProvenance(
        encoder_id=str(raw["encoder_id"]),
        encoder_revision=str(raw["encoder_revision"]),
        weights_sha256=str(raw["weights_sha256"]),
        frame_sampling_version=str(raw["frame_sampling_version"]),
        preprocessing_version=str(raw["preprocessing_version"]),
        pooling=str(raw["pooling"]),
        embedding_dim=embedding_dim,
        dtype="float32",
        # The archive is already normalized, and the visual-split core
        # independently normalizes at comparison time.  This value names the
        # core's comparison contract, not a mutation of the stored features.
        normalization="cosine_l2_at_split",
        frozen_encoder=True,
    )
    provenance.validate()
    return provenance


def _validate_complete_shards(
    root: Path,
    *,
    input_manifest: Path,
    final: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    shards_root = root / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(shards_root)
    expected_names = {
        rank_directory(root, rank, FINAL_WORLD_SIZE).name
        for rank in range(FINAL_WORLD_SIZE)
    }
    actual_names = {child.name for child in shards_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "R7 preflight shard directory set differs: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    shards: list[dict[str, Any]] = []
    raw_dino: dict[str, Any] | None = None
    common_contract: dict[str, Any] | None = None
    done_digests: list[str] = []
    for rank in range(FINAL_WORLD_SIZE):
        directory = rank_directory(root, rank, FINAL_WORLD_SIZE)
        shard = validate_shard(
            directory,
            input_manifest=input_manifest,
            rehash_videos=False,
        )
        contract = dict(shard["contract"])
        if (
            contract.get("rank") != rank
            or contract.get("world_size") != FINAL_WORLD_SIZE
        ):
            raise ValueError(
                f"R7 shard {rank} rank/world contract differs"
            )
        candidate_dino = contract.get("dino")
        if not isinstance(candidate_dino, Mapping):
            raise ValueError(
                f"R7 shard {rank} has no DINO provenance mapping"
            )
        candidate_dino_dict = dict(candidate_dino)
        if raw_dino is None:
            raw_dino = candidate_dino_dict
        elif candidate_dino_dict != raw_dino:
            raise ValueError(
                "eight R7 shard DINO provenance contracts differ"
            )

        candidate_common = dict(contract)
        candidate_common.pop("rank", None)
        candidate_common.pop("device", None)
        if common_contract is None:
            common_contract = candidate_common
        elif candidate_common != common_contract:
            raise ValueError(
                "eight R7 shard common extraction contracts differ"
            )
        done_digests.append(_file_digest(directory / "done.json"))
        shards.append(shard)

    assert raw_dino is not None and common_contract is not None
    final_summary = final["summary"]
    if final_summary.get("world_size") != FINAL_WORLD_SIZE:
        raise ValueError("R7 final world_size is not exactly eight")
    if final_summary.get("shard_done_sha256") != done_digests:
        raise ValueError(
            "R7 final shard-done registry differs from the eight shards"
        )

    final_rows = final["rows"]
    final_arrays = final["arrays"]
    for rank, shard in enumerate(shards):
        indices = [
            int(row["input_index"]) for row in shard["rows"]
        ]
        for local_index, input_index in enumerate(indices):
            expected_row = dict(shard["rows"][local_index])
            expected_row["merged_array_index"] = input_index
            if final_rows[input_index] != expected_row:
                raise ValueError(
                    f"R7 final row {input_index} differs from shard {rank}"
                )
        if set(shard["arrays"]) != set(final_arrays):
            raise ValueError(
                f"R7 final/shard {rank} array names differ"
            )
        for name, shard_array in shard["arrays"].items():
            merged_values = np.asarray(final_arrays[name])[indices]
            if not np.array_equal(
                merged_values,
                np.asarray(shard_array),
            ):
                raise ValueError(
                    f"R7 final array {name} differs from shard {rank}"
                )
    return shards, raw_dino, done_digests


def _side_asset(
    *,
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    row_index: int,
    side: str,
) -> R7VisualAsset:
    result = row.get(side)
    if not isinstance(result, Mapping):
        raise ValueError(f"row {row_index} has no {side} record")
    if result.get("dino_valid") is not True:
        raise ValueError(
            f"row {row_index} {side} lacks committed DINO evidence"
        )
    if not bool(np.asarray(arrays[f"{side}_dino_valid"])[row_index]):
        raise ValueError(
            f"row {row_index} {side} DINO validity array is false"
        )
    decode = result.get("decode")
    if not isinstance(decode, Mapping):
        raise ValueError(
            f"row {row_index} {side} lacks decode provenance"
        )
    frame_indices = decode.get("dino_source_frame_indices")
    if (
        not isinstance(frame_indices, list)
        or not frame_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in frame_indices
        )
    ):
        raise ValueError(
            f"row {row_index} {side} DINO source indices are invalid"
        )
    archive_hashes = [
        str(value)
        for value in np.asarray(
            arrays[f"{side}_perceptual_hashes"]
        )[row_index].tolist()
    ]
    recorded_hashes = decode.get("perceptual_hashes")
    if recorded_hashes != archive_hashes:
        raise ValueError(
            f"row {row_index} {side} pHash manifest/archive differ"
        )
    matrix = np.asarray(arrays[f"{side}_dino_cls"])[row_index]
    if len(frame_indices) != len(archive_hashes) or len(frame_indices) != len(
        matrix
    ):
        raise ValueError(
            f"row {row_index} {side} visual feature lengths differ"
        )
    video_sha256 = result.get("video_sha256")
    if not isinstance(video_sha256, str):
        raise ValueError(
            f"row {row_index} {side} lacks video SHA-256"
        )
    return R7VisualAsset.create(
        video_sha256=video_sha256,
        frame_indices=frame_indices,
        perceptual_hashes=archive_hashes,
        dino_embeddings=matrix,
    )


def load_preflight_visual_pairs(
    preflight_output_root: Path,
) -> PreparedVisualInput:
    """Validate one complete preflight output and construct visual pairs."""

    root = preflight_output_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    final_directory = root / "final"
    initial_final = validate_final(final_directory)
    initial_summary = initial_final["summary"]
    recorded_manifest = initial_summary.get("input_manifest")
    if not isinstance(recorded_manifest, str) or not recorded_manifest:
        raise ValueError("R7 final summary has no input_manifest")
    input_manifest = Path(recorded_manifest).expanduser().resolve(strict=True)
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    input_manifest_sha256 = _file_digest(input_manifest)
    if (
        initial_summary.get("input_manifest_sha256")
        != input_manifest_sha256
    ):
        raise ValueError("R7 final input manifest digest differs")
    final = validate_final(
        final_directory,
        input_manifest=input_manifest,
    )
    _shards, raw_dino, shard_done_digests = (
        _validate_complete_shards(
            root,
            input_manifest=input_manifest,
            final=final,
        )
    )
    dino_provenance = _core_dino_provenance(raw_dino)

    rows = final["rows"]
    arrays = final["arrays"]
    pairs: list[R7VisualPair] = []
    seen_iids: set[str] = set()
    for row_index, row in enumerate(rows):
        if (
            row.get("schema_version") != R7_ROW_SCHEMA
            or row.get("input_index") != row_index
            or row.get("merged_array_index") != row_index
            or row.get("world_size") != FINAL_WORLD_SIZE
            or row.get("shard_rank") != row_index % FINAL_WORLD_SIZE
        ):
            raise ValueError(
                f"R7 final row {row_index} metadata differs"
            )
        iid = _normalized_iid(
            row.get("iid"),
            context=f"R7 final row {row_index}",
        )
        if iid in seen_iids:
            raise ValueError(f"R7 final duplicates iid={iid}")
        seen_iids.add(iid)
        source = _side_asset(
            row=row,
            arrays=arrays,
            row_index=row_index,
            side="source",
        )
        target = _side_asset(
            row=row,
            arrays=arrays,
            row_index=row_index,
            side="target",
        )
        pairs.append(
            R7VisualPair.create(
                iid=iid,
                source=source,
                target=target,
            )
        )

    final_artifacts = {
        name: {
            "filename": name,
            "sha256": _file_digest(final_directory / name),
        }
        for name in (
            "features.npz",
            "manifest.jsonl",
            "summary.json",
            "done.json",
        )
    }
    provenance = {
        "schema_version": R7_VISUAL_SPLIT_IO_SCHEMA,
        "implementation_version": R7_VISUAL_SPLIT_IO_VERSION,
        "preflight_output_root": str(root),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_manifest_sha256,
        "final_artifacts": final_artifacts,
        "shard_done_sha256": shard_done_digests,
        "shard_count": FINAL_WORLD_SIZE,
        "raw_dino_provenance": dict(raw_dino),
        "raw_dino_provenance_sha256": _object_digest(dict(raw_dino)),
        "core_dino_provenance": asdict(dino_provenance),
        "core_dino_provenance_sha256": dino_provenance.digest(),
    }
    return PreparedVisualInput(
        root=root,
        input_manifest=input_manifest,
        input_manifest_sha256=input_manifest_sha256,
        pairs=tuple(pairs),
        dino_provenance=dino_provenance,
        raw_dino_provenance=dict(raw_dino),
        provenance=_json_value(provenance),
    )


def _build_and_audit(
    prepared: PreparedVisualInput,
    ledger: PriorIidLedger,
    config: R7VisualSplitConfig,
) -> R7VisualSplitResult:
    result = build_r7_visual_split(
        prepared.pairs,
        config=config,
        dino_provenance=prepared.dino_provenance,
        previously_seen_iids=ledger.iids,
    )
    independent_audit = audit_r7_visual_split(
        prepared.pairs,
        result.assignments,
        config=config,
        dino_provenance=prepared.dino_provenance,
        previously_seen_iids=ledger.iids,
    )
    if (
        not result.audit.passed
        or not independent_audit.passed
        or independent_audit.to_dict() != result.audit.to_dict()
    ):
        raise ValueError("R7 visual split independent audit differs")
    return result


def _assignment_rows(
    result: R7VisualSplitResult,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": R7_VISUAL_ASSIGNMENT_ROW_SCHEMA,
            **assignment.to_dict(),
        }
        for assignment in result.assignments
    ]


def _component_rows(
    result: R7VisualSplitResult,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": R7_VISUAL_COMPONENT_ROW_SCHEMA,
            **component.to_dict(),
        }
        for component in result.components
    ]


def _summary(
    *,
    prepared: PreparedVisualInput,
    ledger: PriorIidLedger,
    config: R7VisualSplitConfig,
    result: R7VisualSplitResult,
    assignments_sha256: str,
    components_sha256: str,
) -> dict[str, Any]:
    return _json_value(
        {
            "schema_version": R7_VISUAL_SPLIT_SUMMARY_SCHEMA,
            "status": "complete",
            "implementation_version": R7_VISUAL_SPLIT_IO_VERSION,
            "assignments": len(result.assignments),
            "components": len(result.components),
            "edges": len(result.edges),
            "config": asdict(config),
            "config_sha256": config.digest(),
            "preflight": dict(prepared.provenance),
            "prior_iid_ledger": {
                "path": str(ledger.path),
                "file_sha256": ledger.file_sha256,
                "iid_count": len(ledger.iids),
                "canonical_iids_sha256": ledger.canonical_iids_sha256,
            },
            "core_provenance": result.provenance.to_dict(),
            "audit": result.audit.to_dict(),
            "artifacts": {
                "assignments": {
                    "filename": ASSIGNMENTS_NAME,
                    "sha256": assignments_sha256,
                },
                "components": {
                    "filename": COMPONENTS_NAME,
                    "sha256": components_sha256,
                },
            },
            "interpretation": {
                "component_freshness_scope":
                    "relative_to_supplied_prior_iid_ledger_only",
                "p0_old_181_allowed_use":
                    "io_contract_test_only_not_fresh_evaluation",
                "fresh_evaluation_requires":
                    "new_independently_selected_unseen_data_and_new_preflight",
                "fresh_evaluation_authorized": False,
                "generation_authorized": False,
            },
        }
    )


def _done(
    *,
    result: R7VisualSplitResult,
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": R7_VISUAL_SPLIT_DONE_SCHEMA,
        "status": "complete",
        "assignments": len(result.assignments),
        "components": len(result.components),
        "audit_passed": result.audit.passed,
        "fresh_evaluation_authorized": False,
        "generation_authorized": False,
        "core_provenance_sha256": result.provenance.provenance_digest,
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
            for name, path in output_paths.items()
            if name != "done"
        },
    }


def _output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "assignments": output_directory / ASSIGNMENTS_NAME,
        "components": output_directory / COMPONENTS_NAME,
        "summary": output_directory / SUMMARY_NAME,
        "done": output_directory / DONE_NAME,
    }


def validate_visual_split_artifacts(
    *,
    preflight_output_root: Path,
    prior_iid_ledger: Path,
    output_dir: Path,
    config: R7VisualSplitConfig | None = None,
) -> dict[str, Any]:
    """Revalidate a complete output without writing or repairing anything."""

    split_config = config or R7VisualSplitConfig()
    split_config.validate()
    directory = output_dir.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    actual_names = {child.name for child in directory.iterdir()}
    if actual_names != _EXPECTED_OUTPUT_NAMES:
        raise ValueError(
            "R7 visual-split output file set differs: "
            f"missing={sorted(_EXPECTED_OUTPUT_NAMES - actual_names)}, "
            f"extra={sorted(actual_names - _EXPECTED_OUTPUT_NAMES)}"
        )
    paths = _output_paths(directory)
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != R7_VISUAL_SPLIT_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("audit_passed") is not True
        or done.get("fresh_evaluation_authorized") is not False
        or done.get("generation_authorized") is not False
    ):
        raise ValueError("invalid R7 visual-split done marker")
    registry = done.get("artifacts")
    if not isinstance(registry, Mapping) or set(registry) != {
        "assignments",
        "components",
        "summary",
    }:
        raise ValueError("R7 visual-split artifact registry differs")
    for name, record in registry.items():
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != paths[name].name
            or record.get("sha256") != _file_digest(paths[name])
        ):
            raise ValueError(
                f"R7 visual-split {name} digest differs"
            )

    assignments = _load_canonical_jsonl(paths["assignments"])
    components = _load_canonical_jsonl(paths["components"])
    summary = _load_json(paths["summary"])
    prepared = load_preflight_visual_pairs(preflight_output_root)
    ledger = read_prior_iid_ledger(prior_iid_ledger)
    result = _build_and_audit(prepared, ledger, split_config)
    expected_assignments = _json_value(_assignment_rows(result))
    expected_components = _json_value(_component_rows(result))
    if assignments != expected_assignments:
        raise ValueError("committed R7 visual assignments are not reproducible")
    if components != expected_components:
        raise ValueError("committed R7 visual components are not reproducible")
    expected_summary = _summary(
        prepared=prepared,
        ledger=ledger,
        config=split_config,
        result=result,
        assignments_sha256=_file_digest(paths["assignments"]),
        components_sha256=_file_digest(paths["components"]),
    )
    if summary != expected_summary:
        raise ValueError("committed R7 visual summary is not reproducible")
    expected_done = _done(result=result, output_paths=paths)
    if done != expected_done:
        raise ValueError("committed R7 visual done marker is not reproducible")
    return {
        "done": done,
        "summary": summary,
        "assignments": assignments,
        "components": components,
    }


def build_visual_split_artifacts(
    *,
    preflight_output_root: Path,
    prior_iid_ledger: Path,
    output_dir: Path,
    config: R7VisualSplitConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build a committed split, or strictly validate it when resuming."""

    split_config = config or R7VisualSplitConfig()
    split_config.validate()
    directory = output_dir.expanduser().resolve()
    if resume:
        if not (directory / DONE_NAME).is_file():
            raise FileNotFoundError(
                "resume is validation-only and requires a complete done.json"
            )
        return validate_visual_split_artifacts(
            preflight_output_root=preflight_output_root,
            prior_iid_ledger=prior_iid_ledger,
            output_dir=directory,
            config=split_config,
        )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite R7 visual-split output: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)

    prepared = load_preflight_visual_pairs(preflight_output_root)
    ledger = read_prior_iid_ledger(prior_iid_ledger)
    result = _build_and_audit(prepared, ledger, split_config)
    paths = _output_paths(directory)
    _atomic_jsonl(paths["assignments"], _assignment_rows(result))
    _atomic_jsonl(paths["components"], _component_rows(result))
    summary = _summary(
        prepared=prepared,
        ledger=ledger,
        config=split_config,
        result=result,
        assignments_sha256=_file_digest(paths["assignments"]),
        components_sha256=_file_digest(paths["components"]),
    )
    _atomic_json(paths["summary"], summary)
    done = _done(result=result, output_paths=paths)
    _atomic_json(paths["done"], done)
    return {
        "done": done,
        "summary": summary,
        "assignments": _assignment_rows(result),
        "components": _component_rows(result),
    }


def _positive_fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and in (0,1)")
    return parsed


def _nonnegative_fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("must be finite and in [0,1)")
    return parsed


def _positive_unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and in (0,1]")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preflight-output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prior-iid-ledger",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=260108828)
    parser.add_argument(
        "--train-fraction",
        type=_positive_fraction,
        default=0.8,
    )
    parser.add_argument(
        "--validation-fraction",
        type=_positive_fraction,
        default=0.1,
    )
    parser.add_argument(
        "--maximum-phash-hamming-fraction",
        type=_nonnegative_fraction,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-dino-cosine",
        type=_positive_unit_interval,
        default=0.95,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit/audit an R7 source+target visual split",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    _add_common_arguments(build)
    build.add_argument(
        "--resume",
        action="store_true",
        help="validate an existing complete artifact; never write",
    )
    validate = subparsers.add_parser("validate")
    _add_common_arguments(validate)
    return parser


def _config_from_args(args: argparse.Namespace) -> R7VisualSplitConfig:
    return R7VisualSplitConfig(
        data_seed=args.data_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        maximum_phash_hamming_fraction=(
            args.maximum_phash_hamming_fraction
        ),
        minimum_dino_cosine=args.minimum_dino_cosine,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "build":
        result = build_visual_split_artifacts(
            preflight_output_root=args.preflight_output_root,
            prior_iid_ledger=args.prior_iid_ledger,
            output_dir=args.output_dir,
            config=config,
            resume=args.resume,
        )
    else:
        result = validate_visual_split_artifacts(
            preflight_output_root=args.preflight_output_root,
            prior_iid_ledger=args.prior_iid_ledger,
            output_dir=args.output_dir,
            config=config,
        )
    print(_canonical_json(result["done"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENTS_NAME",
    "COMPONENTS_NAME",
    "DONE_NAME",
    "PreparedVisualInput",
    "PriorIidLedger",
    "R7_VISUAL_SPLIT_DONE_SCHEMA",
    "R7_VISUAL_SPLIT_IO_SCHEMA",
    "R7_VISUAL_SPLIT_SUMMARY_SCHEMA",
    "SUMMARY_NAME",
    "build_visual_split_artifacts",
    "load_preflight_visual_pairs",
    "main",
    "read_prior_iid_ledger",
    "validate_visual_split_artifacts",
]
