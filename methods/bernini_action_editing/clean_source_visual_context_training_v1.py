#!/usr/bin/env python3
"""Data, objective, and checkpoint contract for clean-source visual context.

The legacy full644 VAE container holds ``[source_posterior, synthetic_target]``
inside one Arrow list.  A one-time, explicitly acknowledged extraction boundary
parses that container and writes 88 physical files containing index zero only.
The Stage-B optimizer process opens only those files: it has byte-level zero
access to index one and reuses the real-source posterior as both source
condition and clean no-op target.  Index one is never decoded, hashed, sampled,
retained, or returned by this route.

The deterministic exploratory cohort is 64 train + 16 confirmation + 8
strict single-dynamic-actor heldout sources.  The qualified heldout canaries
are reserved before train/confirmation selection so rare eligible families
cannot be consumed by the optimizer split.  All 88 source-video hashes and
group IDs are disjoint.  The
upstream release remains ``training_use_forbidden=true``; a caller must also
provide the separate user exploratory acknowledgement before training.

This file provides an executable manifest builder/auditor and the create-only
0/20/40/60/80 checkpoint coordinator.  The separate registered WORLD8 runner
binds model loading and native RV2V/SP authentication.  Decoded checkpoint
inference and HTML review remain a separate, still-required integration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


SCHEMA_VERSION = "bernini-clean-source-visual-context-source-only-split-v3"
CHECKPOINT_SCHEMA_VERSION = "bernini-clean-source-visual-context-checkpoint-v1"
TRAINING_CONTRACT_SCHEMA_VERSION = (
    "bernini-clean-source-visual-context-training-contract-v1"
)
FULL644_ROWS = 644
FULL644_ACTION_FAMILIES = 28
FULL644_STRICT_SINGLE_ACTOR_ELIGIBLE_ROWS = 359
KNOWN_INELIGIBLE_LEGACY_HELDOUT_IIDS = (
    "2bae4cc4a86d4888",
    "034a3a35acf04eb1",
)
EXPECTED_STRICT_HELDOUT_REPLACEMENT_IIDS = (
    "4281f2d12d3e4419",
    "1b0e34725c7648c4",
)
SPLIT_COUNTS = {"train": 64, "confirmation": 16, "heldout": 8}
CHECKPOINT_STEPS = (0, 20, 40, 60, 80)
MAX_OPTIMIZER_STEPS = 80
SAVE_EVERY = 20
GRADIENT_ACCUMULATION_STEPS = 4
PHYSICAL_DP_SIZE = 2
EFFECTIVE_GLOBAL_BATCH = GRADIENT_ACCUMULATION_STEPS * PHYSICAL_DP_SIZE
CHECKPOINT_LOGICAL_RECORDS = tuple(
    step * EFFECTIVE_GLOBAL_BATCH for step in CHECKPOINT_STEPS
)
LATENT_PHASES = 21
POSTERIOR_CHANNELS = 32
LATENT_CHANNELS = 16
SPLIT_SEED = 20260814

PINNED_FULL644_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/vae_full_81f_4d41e4c"
)
PINNED_FULL644_SHARDS = PINNED_FULL644_ROOT / "shards"
PINNED_FULL644_SUMMARY = PINNED_FULL644_ROOT / "dataset_summary.json"
PINNED_FULL644_INDEX = PINNED_FULL644_ROOT / "dataset_index.jsonl"
PINNED_RAW_PARQUET = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_r13_action_81f_v1/data/"
    "raw_644_release_a6d6c39/raw.parquet"
)
PINNED_FULL644_SUMMARY_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
PINNED_FULL644_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
PINNED_RAW_PARQUET_SHA256 = (
    "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
)


class CleanSourceVisualTrainingError(RuntimeError):
    """Raised before a data leak, weak split, or missing checkpoint."""


def fail(message: str) -> NoReturn:
    raise CleanSourceVisualTrainingError(message)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise CleanSourceVisualTrainingError(
            "PyTorch is required for posterior decode/checkpoint operations"
        ) from error
    return torch


def _require_adapter_core() -> Any:
    try:
        import clean_source_visual_context_adapter_v1 as adapter_core
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise CleanSourceVisualTrainingError(
            "clean-source visual adapter core is unavailable"
        ) from error
    return adapter_core


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
        raise CleanSourceVisualTrainingError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} must be lowercase SHA-256")
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        fail(f"{label} must be non-empty text")
    return value


def _plain_absolute_file(path_value: Any, *, label: str) -> Path:
    if not isinstance(path_value, (str, Path)):
        fail(f"{label} path is invalid")
    requested = Path(path_value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink")
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise CleanSourceVisualTrainingError(f"{label} is unavailable: {error}") from error
    if path != requested or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanSourceVisualTrainingError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        fail(f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class SourceOnlySplitRow:
    iid: str
    split: str
    group_id: str
    action_family: str
    source_video_sha256: str
    strict_selection_gates_all_true: bool
    single_dynamic_actor: bool
    heldout_action_canary_eligible: bool
    source_posterior_path: str
    source_posterior_file_sha256: str
    legacy_shard_path: str
    legacy_shard_sha256: str
    source_posterior_blob_sha256: str

    def __post_init__(self) -> None:
        _required_text(self.iid, label="iid")
        if self.split not in SPLIT_COUNTS:
            fail("source-only row split is invalid")
        _required_text(self.group_id, label=f"{self.iid} group_id")
        _required_text(self.action_family, label=f"{self.iid} action_family")
        _sha(self.source_video_sha256, label=f"{self.iid} source_video_sha256")
        if (
            type(self.strict_selection_gates_all_true) is not bool
            or type(self.single_dynamic_actor) is not bool
            or type(self.heldout_action_canary_eligible) is not bool
            or self.heldout_action_canary_eligible
            != (
                self.strict_selection_gates_all_true
                and self.single_dynamic_actor
            )
        ):
            fail(f"{self.iid} heldout action-canary qualification differs")
        if self.split == "heldout" and not self.heldout_action_canary_eligible:
            fail(f"{self.iid} heldout row is not a strict single-actor canary")
        _sha(
            self.source_posterior_file_sha256,
            label=f"{self.iid} source_posterior_file_sha256",
        )
        _sha(self.legacy_shard_sha256, label=f"{self.iid} legacy_shard_sha256")
        _sha(
            self.source_posterior_blob_sha256,
            label=f"{self.iid} source_posterior_blob_sha256",
        )
        path = Path(self.source_posterior_path)
        legacy = Path(self.legacy_shard_path)
        if (
            not path.is_absolute()
            or path.name != f"{self.iid}.source-posterior-index0.pt"
            or self.source_posterior_file_sha256
            != self.source_posterior_blob_sha256
            or legacy != PINNED_FULL644_SHARDS / f"{self.iid}.parquet"
        ):
            fail(f"{self.iid} source-only materialization path differs")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceOnlySplitRow":
        expected = {
            "iid",
            "split",
            "group_id",
            "action_family",
            "source_video_sha256",
            "strict_selection_gates_all_true",
            "single_dynamic_actor",
            "heldout_action_canary_eligible",
            "source_posterior_path",
            "source_posterior_file_sha256",
            "legacy_shard_path",
            "legacy_shard_sha256",
            "source_posterior_blob_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            fail("source-only split row fields differ")
        return cls(**{name: value[name] for name in expected})

    def receipt(self) -> Mapping[str, Any]:
        return {
            "iid": self.iid,
            "split": self.split,
            "group_id": self.group_id,
            "action_family": self.action_family,
            "source_video_sha256": self.source_video_sha256,
            "strict_selection_gates_all_true": (
                self.strict_selection_gates_all_true
            ),
            "single_dynamic_actor": self.single_dynamic_actor,
            "heldout_action_canary_eligible": (
                self.heldout_action_canary_eligible
            ),
            "source_posterior_path": self.source_posterior_path,
            "source_posterior_file_sha256": self.source_posterior_file_sha256,
            "legacy_shard_path": self.legacy_shard_path,
            "legacy_shard_sha256": self.legacy_shard_sha256,
            "source_posterior_blob_sha256": self.source_posterior_blob_sha256,
        }


@dataclass(frozen=True)
class SourceOnlySplitManifest:
    rows: tuple[SourceOnlySplitRow, ...]
    manifest_digest: str
    source_dataset: Mapping[str, Any]
    source_only_materialization: Mapping[str, Any]

    def rows_for_split(self, split: str) -> tuple[SourceOnlySplitRow, ...]:
        if split not in SPLIT_COUNTS:
            fail("requested source-only split is invalid")
        return tuple(row for row in self.rows if row.split == split)

    def receipt(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_digest": self.manifest_digest,
            "source_dataset": dict(self.source_dataset),
            "source_only_materialization": dict(self.source_only_materialization),
            "split_counts": {
                split: len(self.rows_for_split(split)) for split in SPLIT_COUNTS
            },
            "selected_rows": len(self.rows),
            "selected_action_families": len({row.action_family for row in self.rows}),
            "heldout_basic_action_canaries": len(self.rows_for_split("heldout")),
            "heldout_strict_single_actor_all_true": all(
                row.heldout_action_canary_eligible
                for row in self.rows_for_split("heldout")
            ),
            "heldout_distinct_action_families": len(
                {
                    row.action_family
                    for row in self.rows_for_split("heldout")
                }
            ),
            "split_selection_order": ["heldout", "train", "confirmation"],
            "source_hash_disjoint": True,
            "group_id_disjoint": True,
            "source_posterior_only": True,
            "legacy_parquet_opened_by_stage_b": False,
            "synthetic_target_posterior_accessed_by_stage_b": False,
        }


def _expected_source_dataset_receipt() -> Mapping[str, Any]:
    return {
        "membership_rows": FULL644_ROWS,
        "unique_source_video_sha256": FULL644_ROWS,
        "unique_group_id": FULL644_ROWS,
        "action_family_count": FULL644_ACTION_FAMILIES,
        "strict_single_dynamic_actor_eligible_rows": (
            FULL644_STRICT_SINGLE_ACTOR_ELIGIBLE_ROWS
        ),
        "raw_parquet_path": str(PINNED_RAW_PARQUET),
        "raw_parquet_sha256": PINNED_RAW_PARQUET_SHA256,
        "vae_summary_path": str(PINNED_FULL644_SUMMARY),
        "vae_summary_sha256": PINNED_FULL644_SUMMARY_SHA256,
        "vae_index_path": str(PINNED_FULL644_INDEX),
        "vae_index_sha256": PINNED_FULL644_INDEX_SHA256,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "user_authorized_exploratory_training": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "legacy_container_has_synthetic_target_blob": True,
        "legacy_container_opened_by_stage_b_training": False,
        "synthetic_target_posterior_accessed_by_stage_b_training": False,
        "source_only_physical_extraction_required": True,
        "source_posterior_reused_as_source_and_noop_target": True,
    }


def validate_source_only_split_manifest_value(
    value: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> SourceOnlySplitManifest:
    if not isinstance(value, Mapping):
        fail("source-only manifest root must be an object")
    expected_fields = {
        "schema_version",
        "source_dataset",
        "source_only_materialization",
        "split_seed",
        "split_counts",
        "rows",
        "source_only_contract",
        "manifest_digest",
    }
    if set(value) != expected_fields or value.get("schema_version") != SCHEMA_VERSION:
        fail("source-only manifest schema or fields differ")
    candidate = dict(value)
    declared_digest = candidate.pop("manifest_digest", None)
    if _sha(declared_digest, label="manifest_digest") != object_sha256(candidate):
        fail("source-only manifest digest differs")
    if value.get("split_seed") != SPLIT_SEED or value.get("split_counts") != SPLIT_COUNTS:
        fail("source-only split seed/count contract differs")
    source_dataset = value.get("source_dataset")
    if not isinstance(source_dataset, Mapping) or dict(source_dataset) != dict(
        _expected_source_dataset_receipt()
    ):
        fail("pinned full644 source dataset identity differs")
    contract = value.get("source_only_contract")
    expected_contract = {
        "stage_b_physical_file_payload": "source_posterior_index_0_only",
        "stage_b_legacy_parquet_opened": False,
        "stage_b_posterior_index_1_bytes_read": False,
        "stage_b_posterior_index_1_synthetic_target_decoded": False,
        "stage_b_posterior_index_1_synthetic_target_hashed": False,
        "source_posterior_reused_as_source_condition": True,
        "source_posterior_reused_as_clean_noop_target": True,
        "synthetic_target_supervision": False,
        "feature_reward": False,
        "vlm_reward": False,
        "split_selection_order": ["heldout", "train", "confirmation"],
        "heldout_requires_strict_selection_gates_all_true": True,
        "heldout_requires_single_dynamic_actor": True,
        "heldout_distinct_action_families": 8,
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        fail("source-only posterior access contract differs")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        fail("source-only manifest rows must be a list")
    rows = tuple(SourceOnlySplitRow.from_mapping(row) for row in raw_rows)
    if len(rows) != sum(SPLIT_COUNTS.values()):
        fail("source-only manifest must contain exact 64/16/8 rows")
    for split, expected_count in SPLIT_COUNTS.items():
        if sum(row.split == split for row in rows) != expected_count:
            fail(f"source-only manifest {split} count differs")
    fields = {
        "iid": [row.iid for row in rows],
        "source hash": [row.source_video_sha256 for row in rows],
        "group ID": [row.group_id for row in rows],
        "source posterior file": [row.source_posterior_path for row in rows],
    }
    for label, items in fields.items():
        if len(set(items)) != len(items):
            fail(f"selected 64/16/8 {label}s are not disjoint")
    # The train split must not collapse to a two-scene probe.  Requiring at
    # least 16 families also ensures each heldout row is not the only diversity.
    if len({row.action_family for row in rows if row.split == "train"}) < 16:
        fail("64-row train split covers fewer than 16 action families")
    heldout_rows = tuple(row for row in rows if row.split == "heldout")
    if (
        len(heldout_rows) != 8
        or not all(row.heldout_action_canary_eligible for row in heldout_rows)
        or len({row.action_family for row in heldout_rows}) != 8
    ):
        fail("heldout split must be eight distinct strict single-actor canaries")
    materialization = value.get("source_only_materialization")
    if not isinstance(materialization, Mapping):
        fail("source-only physical materialization receipt differs")
    root = Path(str(materialization.get("root"))).expanduser()
    expected_paths = [Path(row.source_posterior_path) for row in rows]
    materialized_files = [
        {
            "iid": row.iid,
            "path": row.source_posterior_path,
            "sha256": row.source_posterior_file_sha256,
        }
        for row in rows
    ]
    if (
        not root.is_absolute()
        or root.is_symlink()
        or materialization.get("physical_file_count") != len(rows)
        or materialization.get("physical_files_digest")
        != object_sha256(materialized_files)
        or materialization.get("stage_b_opens_only_these_physical_files") is not True
        or materialization.get("legacy_parquet_opened_by_stage_b") is not False
        or materialization.get("legacy_list_container_parsed_during_extraction")
        is not True
        or materialization.get("synthetic_target_semantics_used_during_extraction")
        is not False
        or materialization.get("synthetic_target_decoded_during_extraction") is not False
        or materialization.get("synthetic_target_hashed_during_extraction") is not False
        or any(path.parent != root for path in expected_paths)
    ):
        fail("source-only physical materialization contract differs")
    if verify_files:
        resolved_root = root.resolve(strict=True)
        if resolved_root != root or not resolved_root.is_dir():
            fail("source-only physical materialization root differs")
        entries = list(resolved_root.iterdir())
        if any(not path.is_file() or path.is_symlink() for path in entries):
            fail("source-only materialization contains a non-plain entry")
        observed_names = {path.name for path in entries}
        if observed_names != {path.name for path in expected_paths}:
            fail("source-only physical materialization file set differs")
        for row in rows:
            source_file = _plain_absolute_file(
                row.source_posterior_path,
                label=f"{row.iid} source-only posterior",
            )
            if (
                source_file.parent != resolved_root
                or file_sha256(source_file) != row.source_posterior_file_sha256
            ):
                fail(f"{row.iid} source-only posterior identity differs")
    return SourceOnlySplitManifest(
        rows=rows,
        manifest_digest=str(declared_digest),
        source_dataset=dict(source_dataset),
        source_only_materialization=dict(materialization),
    )


def load_source_only_split_manifest(
    path_value: str | Path, *, verify_files: bool = True
) -> SourceOnlySplitManifest:
    path = _plain_absolute_file(path_value, label="source-only split manifest")
    return validate_source_only_split_manifest_value(
        _read_json(path), verify_files=verify_files
    )


def authorize_exploratory_training(
    manifest: SourceOnlySplitManifest,
    *,
    ack_upstream_training_use_forbidden: Any,
    ack_user_authorized_exploratory_training: Any,
) -> Mapping[str, Any]:
    source = manifest.source_dataset
    if (
        source.get("training_authorized") is not False
        or source.get("training_use_forbidden") is not True
        or source.get("user_authorized_exploratory_training") is not True
        or ack_upstream_training_use_forbidden is not True
        or ack_user_authorized_exploratory_training is not True
    ):
        fail(
            "exploratory training requires both upstream-forbidden and user-authorization acknowledgements"
        )
    return {
        "upstream_training_authorized": False,
        "upstream_training_use_forbidden": True,
        "upstream_training_use_forbidden_acknowledged": True,
        "user_authorized_exploratory_training": True,
        "user_authorization_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }


def _extract_source_blob_from_legacy_container(
    shard_path: Path, *, expected_iid: str
) -> bytes:
    """Extract index 0 at a one-time boundary outside Stage-B training.

    PyArrow materializes the legacy two-element list container, so this helper
    must never be called by the optimizer process.  It retains, decodes and
    hashes index 0 only; index 1 is not assigned semantic authority, decoded,
    hashed or written.  The resulting physical file contains index 0 alone.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise CleanSourceVisualTrainingError("pyarrow is required for full644") from error
    try:
        rows = pq.read_table(
            shard_path, columns=["iid", "video_vae_latents"]
        ).to_pylist()
    except Exception as error:
        raise CleanSourceVisualTrainingError(
            f"cannot read selected full644 shard {shard_path}: {error}"
        ) from error
    if len(rows) != 1 or rows[0].get("iid") != expected_iid:
        fail(f"selected shard row identity differs: {expected_iid}")
    posterior_list = rows[0].get("video_vae_latents")
    if not isinstance(posterior_list, list) or len(posterior_list) != 2:
        fail(f"legacy source/target posterior list differs: {expected_iid}")
    source_blob = posterior_list[0]
    if not isinstance(source_blob, (bytes, bytearray, memoryview)):
        fail(f"source posterior blob type differs: {expected_iid}")
    # The list container has already been parsed by PyArrow.  Do not falsely
    # call that byte-zero access.  The Stage-B loader below never opens this
    # Parquet file and therefore has the strict zero-access property.
    return bytes(source_blob)


def _write_create_only_blob(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink() or not payload:
        fail("source-only posterior output must be a fresh absolute file")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes on the local filesystem."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode_source_posterior_parameters(source_blob: bytes, *, iid: str) -> torch.Tensor:
    torch = _require_torch()
    buffer = io.BytesIO(source_blob)
    try:
        parameters = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch on AUH
        buffer.seek(0)
        parameters = torch.load(buffer, map_location="cpu")
    except Exception as error:
        raise CleanSourceVisualTrainingError(
            f"cannot decode source posterior for {iid}: {error}"
        ) from error
    if (
        not isinstance(parameters, torch.Tensor)
        or parameters.layout != torch.strided
        or parameters.device.type != "cpu"
        or parameters.ndim != 5
        or tuple(int(value) for value in parameters.shape[:3])
        != (1, POSTERIOR_CHANNELS, LATENT_PHASES)
        or int(parameters.shape[3]) <= 0
        or int(parameters.shape[4]) <= 0
        or int(parameters.shape[3]) % 2
        or int(parameters.shape[4]) % 2
        or not bool(torch.isfinite(parameters.float()).all().item())
    ):
        fail(f"source posterior parameters differ for {iid}")
    return parameters.float().contiguous()


@dataclass(frozen=True)
class SourceOnlyNoOpLatents:
    iid: str
    split: str
    source_condition: torch.Tensor
    clean_noop_target: torch.Tensor
    source_video_sha256: str
    source_posterior_blob_sha256: str

    def __post_init__(self) -> None:
        torch = _require_torch()
        if self.source_condition is not self.clean_noop_target:
            fail("source condition and no-op target must reuse one exact posterior object")
        tensor = self.source_condition
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or tensor.ndim != 5
            or tuple(int(value) for value in tensor.shape[:3])
            != (1, LATENT_CHANNELS, LATENT_PHASES)
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            fail("normalized source/no-op latent differs")

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "iid": self.iid,
            "split": self.split,
            "source_video_sha256": self.source_video_sha256,
            "source_posterior_blob_sha256": self.source_posterior_blob_sha256,
            "latent_shape": list(self.source_condition.shape),
            "same_tensor_object_for_source_and_noop_target": True,
            "posterior_index_decoded": 0,
            "synthetic_target_posterior_decoded": False,
            "synthetic_target_posterior_hashed": False,
        }
        return {**value, "digest": object_sha256(value)}


class PinnedPhysicalSourceOnlyPosteriorStore:
    """Read only physical index-0 blobs; legacy Parquet is unreachable here."""

    def __init__(
        self,
        manifest: SourceOnlySplitManifest,
        *,
        vae_latents_mean: torch.Tensor,
        vae_latents_std: torch.Tensor,
        verify_files_on_first_access: bool = True,
    ) -> None:
        torch = _require_torch()
        if not isinstance(manifest, SourceOnlySplitManifest):
            fail("source posterior store requires a validated manifest")
        expected_stats_shape = (1, LATENT_CHANNELS, 1, 1, 1)
        for value, label in (
            (vae_latents_mean, "VAE mean"),
            (vae_latents_std, "VAE std"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or tuple(value.shape) != expected_stats_shape
                or value.requires_grad
                or not bool(torch.isfinite(value).all().item())
            ):
                fail(f"{label} must be detached finite CPU FP32 [1,16,1,1,1]")
        if not bool((vae_latents_std > 0).all().item()):
            fail("VAE standard deviations must be positive")
        if not isinstance(verify_files_on_first_access, bool):
            fail("verify_files_on_first_access must be boolean")
        self.manifest = manifest
        self.mean = vae_latents_mean.contiguous()
        self.std = vae_latents_std.contiguous()
        self.verify_files_on_first_access = verify_files_on_first_access
        self._cache: dict[int, SourceOnlyNoOpLatents] = {}

    def __len__(self) -> int:
        return len(self.manifest.rows)

    def load(self, index: int) -> SourceOnlyNoOpLatents:
        torch = _require_torch()
        if isinstance(index, bool) or not isinstance(index, int):
            fail("source posterior index must be an integer")
        row = self.manifest.rows[index]
        if index in self._cache:
            return self._cache[index]
        source_file = _plain_absolute_file(
            row.source_posterior_path, label=f"{row.iid} source-only posterior"
        )
        if (
            self.verify_files_on_first_access
            and file_sha256(source_file) != row.source_posterior_file_sha256
        ):
            fail(f"{row.iid} source-only posterior changed before access")
        source_blob = source_file.read_bytes()
        if bytes_sha256(source_blob) != row.source_posterior_blob_sha256:
            fail(f"{row.iid} source posterior blob SHA differs")
        parameters = _decode_source_posterior_parameters(source_blob, iid=row.iid)
        # DiagonalGaussianDistribution.mode() is exactly the first C channels.
        # Keeping the operation explicit avoids any API that could consume the
        # synthetic posterior at list index one.
        source_mode = parameters[:, :LATENT_CHANNELS, :, :, :]
        normalized = ((source_mode - self.mean) / self.std).detach().contiguous()
        if not bool(torch.isfinite(normalized).all().item()):
            fail(f"{row.iid} normalized source posterior is non-finite")
        result = SourceOnlyNoOpLatents(
            iid=row.iid,
            split=row.split,
            source_condition=normalized,
            clean_noop_target=normalized,
            source_video_sha256=row.source_video_sha256,
            source_posterior_blob_sha256=row.source_posterior_blob_sha256,
        )
        self._cache[index] = result
        return result

    def preload(self, indices: Sequence[int]) -> Mapping[str, Any]:
        normalized = tuple(indices)
        if (
            not normalized
            or any(type(index) is not int or not 0 <= index < len(self) for index in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            fail("source-only preload indices differ")
        for index in normalized:
            self.load(index)
        value = {
            "preloaded_indices": list(normalized),
            "preloaded_rows": len(normalized),
            "legacy_parquet_opened": False,
            "synthetic_target_index1_bytes_read": False,
            "physical_index0_files_only": True,
        }
        return {**value, "digest": object_sha256(value)}


def _ranked_rows(
    rows: Iterable[Mapping[str, Any]], *, split_name: str
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (
                f"{SPLIT_SEED}:{split_name}:"
                f"{row['action_family']}:{row['source_video_sha256']}:{row['iid']}"
            ).encode("ascii")
        ).digest(),
    )


def _round_robin_family_selection(
    available: Sequence[Mapping[str, Any]],
    *,
    count: int,
    split_name: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in _ranked_rows(available, split_name=split_name):
        grouped.setdefault(str(row["action_family"]), []).append(row)
    families = sorted(
        grouped,
        key=lambda family: hashlib.sha256(
            f"{SPLIT_SEED}:{split_name}:{family}".encode("utf-8")
        ).digest(),
    )
    selected: list[Mapping[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for family in families:
            bucket = grouped[family]
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
                progressed = True
        if not progressed:
            fail(f"full644 cannot supply exact {split_name} split")
    used = {str(row["iid"]) for row in selected}
    remaining = [row for row in available if str(row["iid"]) not in used]
    return selected, remaining


def deterministic_source_hash_split(
    full_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """Reserve strict heldout canaries, then create train/confirmation.

    Heldout-first selection prevents the broad real-source train pool from
    consuming rare strict, single-dynamic-actor families.  All selections are
    deterministic family-round-robin functions of pinned source identities.
    """

    if len(full_rows) != FULL644_ROWS:
        fail("full644 normalized membership must contain exactly 644 rows")
    for field in ("iid", "group_id", "action_family", "source_video_sha256"):
        values = [row.get(field) for row in full_rows]
        if any(not isinstance(value, str) or not value for value in values):
            fail(f"full644 normalized rows lack {field}")
        if field != "action_family" and len(set(values)) != FULL644_ROWS:
            fail(f"full644 {field} values must be globally unique")
    if len({str(row["action_family"]) for row in full_rows}) != FULL644_ACTION_FAMILIES:
        fail("full644 action family count differs from 28")
    for row in full_rows:
        strict = row.get("strict_selection_gates_all_true")
        single_actor = row.get("single_dynamic_actor")
        eligible = row.get("heldout_action_canary_eligible")
        if (
            type(strict) is not bool
            or type(single_actor) is not bool
            or type(eligible) is not bool
            or eligible != (strict and single_actor)
        ):
            fail("full644 heldout action-canary qualification differs")

    qualified = [
        row for row in full_rows if row["heldout_action_canary_eligible"]
    ]
    heldout, _ = _round_robin_family_selection(
        qualified, count=SPLIT_COUNTS["heldout"], split_name="heldout"
    )
    if len({str(row["action_family"]) for row in heldout}) != 8:
        fail("full644 cannot supply eight distinct heldout action families")
    heldout_iids = {str(row["iid"]) for row in heldout}
    remaining = [row for row in full_rows if str(row["iid"]) not in heldout_iids]
    train, remaining = _round_robin_family_selection(
        remaining, count=SPLIT_COUNTS["train"], split_name="train"
    )
    confirmation, _ = _round_robin_family_selection(
        remaining,
        count=SPLIT_COUNTS["confirmation"],
        split_name="confirmation",
    )
    result = {
        "train": tuple(train),
        "confirmation": tuple(confirmation),
        "heldout": tuple(heldout),
    }
    return result


def _read_index_rows(path: Path) -> Mapping[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    fail(f"blank full644 index line {line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    fail(f"full644 index line {line_number} is not an object")
                iid = _required_text(value.get("iid"), label="full644 index iid")
                if iid in rows:
                    fail(f"duplicate full644 index IID: {iid}")
                rows[iid] = value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanSourceVisualTrainingError(f"cannot read full644 index: {error}") from error
    if len(rows) != FULL644_ROWS:
        fail("full644 index row count differs")
    return rows


def _action_canary_qualification(
    raw_row: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Parse and cross-check the two pinned preservation qualification gates."""

    strict = raw_row.get("strict_selection_gates_all_true")
    gates_text = raw_row.get("selection_gates_json")
    if type(strict) is not bool or not isinstance(gates_text, str):
        fail("full644 preservation qualification fields differ")
    try:
        gates = json.loads(gates_text)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CleanSourceVisualTrainingError(
            f"cannot parse full644 selection gates: {error}"
        ) from error
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(type(value) is not bool for value in gates.values())
        or type(gates.get("single_dynamic_actor")) is not bool
        or strict != all(gates.values())
    ):
        fail("full644 selection-gate truth binding differs")
    single_actor = gates["single_dynamic_actor"]
    return strict, single_actor, strict and single_actor


def build_source_only_split_manifest_value(
    materialization_root_value: str | Path,
) -> Mapping[str, Any]:
    """Extract 88 physical index-0 blobs and build the sealed split manifest."""

    materialization_root = Path(materialization_root_value).expanduser()
    if (
        not materialization_root.is_absolute()
        or materialization_root.exists()
        or materialization_root.is_symlink()
        or materialization_root == Path("/")
    ):
        fail("source-only materialization root must be a fresh absolute directory")
    materialization_parent = materialization_root.parent.resolve(strict=True)
    if (
        materialization_parent != materialization_root.parent
        or not materialization_parent.is_dir()
        or materialization_root.parent.is_symlink()
    ):
        fail("source-only materialization parent differs")
    materialization_root.mkdir(mode=0o700)

    for path, expected_sha, label in (
        (PINNED_RAW_PARQUET, PINNED_RAW_PARQUET_SHA256, "raw parquet"),
        (PINNED_FULL644_SUMMARY, PINNED_FULL644_SUMMARY_SHA256, "VAE summary"),
        (PINNED_FULL644_INDEX, PINNED_FULL644_INDEX_SHA256, "VAE index"),
    ):
        _plain_absolute_file(path, label=label)
        if file_sha256(path) != expected_sha:
            fail(f"pinned {label} SHA-256 differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise CleanSourceVisualTrainingError("pyarrow is required for split build") from error
    try:
        raw_rows = pq.read_table(
            PINNED_RAW_PARQUET,
            columns=[
                "iid",
                "group_id",
                "family",
                "source_video_sha256",
                "selection_gates_json",
                "strict_selection_gates_all_true",
            ],
        ).to_pylist()
    except Exception as error:
        raise CleanSourceVisualTrainingError(f"cannot read pinned raw parquet: {error}") from error
    normalized = []
    for row in raw_rows:
        strict, single_actor, eligible = _action_canary_qualification(row)
        normalized.append(
            {
                "iid": row.get("iid"),
                "group_id": row.get("group_id"),
                "action_family": row.get("family"),
                "source_video_sha256": row.get("source_video_sha256"),
                "strict_selection_gates_all_true": strict,
                "single_dynamic_actor": single_actor,
                "heldout_action_canary_eligible": eligible,
            }
        )
    if (
        sum(bool(row["heldout_action_canary_eligible"]) for row in normalized)
        != FULL644_STRICT_SINGLE_ACTOR_ELIGIBLE_ROWS
    ):
        fail("pinned full644 strict single-actor eligible count differs")
    splits = deterministic_source_hash_split(normalized)
    heldout_iids = {str(row["iid"]) for row in splits["heldout"]}
    if (
        heldout_iids.intersection(KNOWN_INELIGIBLE_LEGACY_HELDOUT_IIDS)
        or not set(EXPECTED_STRICT_HELDOUT_REPLACEMENT_IIDS).issubset(
            heldout_iids
        )
    ):
        fail("pinned reserve-first strict heldout replacement set differs")
    index = _read_index_rows(PINNED_FULL644_INDEX)
    selected_rows: list[Mapping[str, Any]] = []
    for split in SPLIT_COUNTS:
        for raw in splits[split]:
            iid = str(raw["iid"])
            indexed = index.get(iid)
            if indexed is None:
                fail(f"selected source lacks VAE index row: {iid}")
            shard = Path(str(indexed.get("parquet_path")))
            shard_sha = _sha(indexed.get("parquet_sha256"), label=f"{iid} shard SHA")
            if shard != PINNED_FULL644_SHARDS / f"{iid}.parquet":
                fail(f"selected source shard path differs: {iid}")
            _plain_absolute_file(shard, label=f"{iid} shard")
            if file_sha256(shard) != shard_sha:
                fail(f"selected source shard SHA differs: {iid}")
            source_blob = _extract_source_blob_from_legacy_container(
                shard, expected_iid=iid
            )
            source_path = (
                materialization_root / f"{iid}.source-posterior-index0.pt"
            )
            _write_create_only_blob(source_path, source_blob)
            source_sha = bytes_sha256(source_blob)
            selected_rows.append(
                SourceOnlySplitRow(
                    iid=iid,
                    split=split,
                    group_id=str(raw["group_id"]),
                    action_family=str(raw["action_family"]),
                    source_video_sha256=str(raw["source_video_sha256"]),
                    strict_selection_gates_all_true=bool(
                        raw["strict_selection_gates_all_true"]
                    ),
                    single_dynamic_actor=bool(raw["single_dynamic_actor"]),
                    heldout_action_canary_eligible=bool(
                        raw["heldout_action_canary_eligible"]
                    ),
                    source_posterior_path=str(source_path),
                    source_posterior_file_sha256=source_sha,
                    legacy_shard_path=str(shard),
                    legacy_shard_sha256=shard_sha,
                    source_posterior_blob_sha256=source_sha,
                ).receipt()
            )
    _fsync_directory(materialization_root)
    os.chmod(materialization_root, 0o555)
    materialized_files = [
        {
            "iid": row["iid"],
            "path": row["source_posterior_path"],
            "sha256": row["source_posterior_file_sha256"],
        }
        for row in selected_rows
    ]
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": dict(_expected_source_dataset_receipt()),
        "source_only_materialization": {
            "root": str(materialization_root),
            "physical_file_count": len(materialized_files),
            "physical_files_digest": object_sha256(materialized_files),
            "stage_b_opens_only_these_physical_files": True,
            "legacy_parquet_opened_by_stage_b": False,
            "legacy_list_container_parsed_during_extraction": True,
            "synthetic_target_semantics_used_during_extraction": False,
            "synthetic_target_decoded_during_extraction": False,
            "synthetic_target_hashed_during_extraction": False,
        },
        "split_seed": SPLIT_SEED,
        "split_counts": dict(SPLIT_COUNTS),
        "rows": selected_rows,
        "source_only_contract": {
            "stage_b_physical_file_payload": "source_posterior_index_0_only",
            "stage_b_legacy_parquet_opened": False,
            "stage_b_posterior_index_1_bytes_read": False,
            "stage_b_posterior_index_1_synthetic_target_decoded": False,
            "stage_b_posterior_index_1_synthetic_target_hashed": False,
            "source_posterior_reused_as_source_condition": True,
            "source_posterior_reused_as_clean_noop_target": True,
            "synthetic_target_supervision": False,
            "feature_reward": False,
            "vlm_reward": False,
            "split_selection_order": ["heldout", "train", "confirmation"],
            "heldout_requires_strict_selection_gates_all_true": True,
            "heldout_requires_single_dynamic_actor": True,
            "heldout_distinct_action_families": 8,
        },
    }
    value["manifest_digest"] = object_sha256(value)
    validate_source_only_split_manifest_value(value, verify_files=True)
    return value


def write_create_only_json(path_value: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("output JSON must be a new absolute non-symlink path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        fail("output JSON parent must be an existing directory")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


@dataclass
class CheckpointCadence:
    """State machine that forbids missing or post-hoc 20-step checkpoints."""

    observed_steps: list[int]

    def __init__(self) -> None:
        self.observed_steps = []

    @property
    def next_required_step(self) -> int:
        if len(self.observed_steps) >= len(CHECKPOINT_STEPS):
            fail("checkpoint cadence is already complete")
        return CHECKPOINT_STEPS[len(self.observed_steps)]

    def observe(self, step: Any) -> None:
        if isinstance(step, bool) or not isinstance(step, int):
            fail("checkpoint step must be an integer")
        expected = self.next_required_step
        if step != expected:
            fail(f"checkpoint step {step} arrived; exact next step is {expected}")
        self.observed_steps.append(step)

    def assert_complete(self) -> None:
        if tuple(self.observed_steps) != CHECKPOINT_STEPS:
            fail("training cannot finish without checkpoints 0/20/40/60/80")

    def receipt(self) -> Mapping[str, Any]:
        return {
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "save_every": SAVE_EVERY,
            "required_checkpoint_steps": list(CHECKPOINT_STEPS),
            "observed_checkpoint_steps": list(self.observed_steps),
            "required_logical_records": list(CHECKPOINT_LOGICAL_RECORDS),
            "observed_logical_records": [
                step * EFFECTIVE_GLOBAL_BATCH for step in self.observed_steps
            ],
            "complete": tuple(self.observed_steps) == CHECKPOINT_STEPS,
        }


def _trainable_state(
    handle: adapter_core.CleanSourceVisualContextHandle,
) -> Mapping[str, torch.Tensor]:
    adapter_core = _require_adapter_core()
    if not isinstance(handle, adapter_core.CleanSourceVisualContextHandle):
        fail("checkpoint requires a live visual-context adapter handle")
    if handle.restored or not handle.base_parameters_frozen() or not handle.native_structure_untouched():
        fail("checkpoint adapter/base scope is not closed")
    return handle.state_dict_for_save()


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    torch = _require_torch()
    projection = []
    for name, tensor in sorted(state.items()):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            fail(f"checkpoint tensor differs: {name}")
        projection.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
            }
        )
    return object_sha256(projection)


def save_visual_context_checkpoint(
    *,
    output_directory: str | Path,
    step: int,
    handle: adapter_core.CleanSourceVisualContextHandle,
    optimizer: torch.optim.Optimizer,
    manifest: SourceOnlySplitManifest,
    authorization_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create one non-overwriting adapter checkpoint at a registered step."""

    torch = _require_torch()
    if step not in CHECKPOINT_STEPS:
        fail("visual-context checkpoint step is outside 0/20/40/60/80")
    root = Path(output_directory).expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("checkpoint output directory must be absolute and non-symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    path = root / f"checkpoint_step_{step:08d}.pt"
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite checkpoint: {path}")
    state = _trainable_state(handle)
    state_digest = _state_digest(state)
    if step == 0:
        for name, tensor in state.items():
            if name.endswith(".output.weight") and bool(torch.count_nonzero(tensor).item()):
                fail("step-0 output projection is not exactly zero")
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "global_step": step,
        "logical_records_seen": step * EFFECTIVE_GLOBAL_BATCH,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "checkpoint_cadence": list(CHECKPOINT_STEPS),
        "manifest_digest": manifest.manifest_digest,
        "split_counts": dict(SPLIT_COUNTS),
        "authorization": dict(authorization_receipt),
        "adapter_receipt": dict(handle.receipt()),
        "adapter_parameter_digest": state_digest,
        "base_frozen": True,
        "native_kv_untouched": True,
        "source_posterior_only": True,
        "synthetic_target_posterior_accessed": False,
        "objective": "same_real_source_noop_flow_matching",
        "feature_or_vlm_reward": False,
    }
    payload = {
        "metadata": metadata,
        "adapter_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=root, delete=False
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        with Path(temporary_name).open("rb") as handle_file:
            os.fsync(handle_file.fileno())
        # Hard-link publication is create-only even under concurrent writers.
        os.link(temporary_name, path)
        _fsync_directory(root)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
            _fsync_directory(root)
    return {
        "step": step,
        "logical_records_seen": step * EFFECTIVE_GLOBAL_BATCH,
        "path": str(path),
        "file_sha256": file_sha256(path),
        "adapter_parameter_digest": state_digest,
    }


class VisualContextCheckpointCoordinator:
    """Training-loop helper: save step 0, then every exact 20 optimizer steps."""

    def __init__(
        self,
        *,
        output_directory: str | Path,
        handle: adapter_core.CleanSourceVisualContextHandle,
        optimizer: torch.optim.Optimizer,
        manifest: SourceOnlySplitManifest,
        authorization_receipt: Mapping[str, Any],
    ) -> None:
        self.output_directory = Path(output_directory)
        self.handle = handle
        self.optimizer = optimizer
        self.manifest = manifest
        self.authorization_receipt = dict(authorization_receipt)
        self.cadence = CheckpointCadence()
        self.records: list[Mapping[str, Any]] = []
        self.current_step = 0

    def start_before_first_optimizer_step(self) -> Mapping[str, Any]:
        if self.current_step != 0 or self.records:
            fail("step-0 checkpoint coordinator was already started")
        self.cadence.observe(0)
        record = save_visual_context_checkpoint(
            output_directory=self.output_directory,
            step=0,
            handle=self.handle,
            optimizer=self.optimizer,
            manifest=self.manifest,
            authorization_receipt=self.authorization_receipt,
        )
        self.records.append(record)
        return record

    def after_optimizer_step(self, completed_step: Any) -> Optional[Mapping[str, Any]]:
        if isinstance(completed_step, bool) or not isinstance(completed_step, int):
            fail("completed optimizer step must be an integer")
        if not self.records or completed_step != self.current_step + 1:
            fail("optimizer steps must be reported contiguously after step 0")
        if completed_step > MAX_OPTIMIZER_STEPS:
            fail("optimizer step exceeds registered 80-step pilot")
        self.current_step = completed_step
        if completed_step % SAVE_EVERY:
            return None
        self.cadence.observe(completed_step)
        record = save_visual_context_checkpoint(
            output_directory=self.output_directory,
            step=completed_step,
            handle=self.handle,
            optimizer=self.optimizer,
            manifest=self.manifest,
            authorization_receipt=self.authorization_receipt,
        )
        self.records.append(record)
        return record

    def finalize(self) -> Mapping[str, Any]:
        if self.current_step != MAX_OPTIMIZER_STEPS:
            fail("visual-context pilot must finish exactly 80 optimizer steps")
        self.cadence.assert_complete()
        value = {
            "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
            "complete": True,
            "optimizer_steps": self.current_step,
            "checkpoint_cadence": self.cadence.receipt(),
            "checkpoints": list(self.records),
            "manifest_digest": self.manifest.manifest_digest,
            "split_counts": dict(SPLIT_COUNTS),
            "source_only_noop_flow_matching": True,
            "synthetic_target_posterior_accessed": False,
            "decoded_quality_claim": False,
            "gpu_validated": False,
        }
        return {**value, "digest": object_sha256(value)}


def training_contract_receipt() -> Mapping[str, Any]:
    value = {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "dataset": "pinned_full644_source_posterior_only",
        "full_membership_rows": FULL644_ROWS,
        "split_counts": dict(SPLIT_COUNTS),
        "source_hash_and_group_id_disjoint": True,
        "train_minimum_not_two_sample_probe": True,
        "split_selection_order": ["heldout", "train", "confirmation"],
        "heldout_strict_single_dynamic_actor_canaries": 8,
        "heldout_distinct_action_families": 8,
        "objective": "same_real_source_noop_flow_matching",
        "physical_index0_only_materialization_required": True,
        "legacy_parquet_opened_by_stage_b": False,
        "clean_target_is_source_posterior_index_0": True,
        "source_condition_is_source_posterior_index_0": True,
        "synthetic_target_posterior_index_1_accessed": False,
        "checkpoints": list(CHECKPOINT_STEPS),
        "checkpoint_logical_records": list(CHECKPOINT_LOGICAL_RECORDS),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": EFFECTIVE_GLOBAL_BATCH,
        "base_frozen": True,
        "native_self_attention_kv_untouched": True,
        "native_text_cross_attention_untouched": True,
        "adapter_block_scope_status": "structural_candidate_not_causally_admitted",
        "stage_a_decoded_admission_required_before_optimizer": True,
        "optimizer_authorized_by_this_contract": False,
        "feature_reward": False,
        "vlm_reward": False,
        "full_world8_runner_implemented_here": False,
        "registered_world8_runner": (
            "train_clean_source_visual_context_stage_b_v1.py"
        ),
        "remaining_integration": [
            "obtain manual admission from complete decoded Stage-A middle-band evidence",
            "run the registered WORLD8 DP2xSP4 GPU preflight and exact80 trajectory",
            "decode one fixed sentinel manifest at every saved checkpoint and build HTML review",
        ],
    }
    return {**value, "digest": object_sha256(value)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--materialization-root", type=Path, required=True)
    audit = subparsers.add_parser("audit-manifest")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument(
        "--ack-upstream-training-use-forbidden", action="store_true"
    )
    audit.add_argument(
        "--ack-user-authorized-exploratory-training", action="store_true"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-manifest":
        value = build_source_only_split_manifest_value(args.materialization_root)
        path = write_create_only_json(args.output, value)
        print(
            json.dumps(
                {
                    "manifest": str(path),
                    "sha256": file_sha256(path),
                    "manifest_digest": value["manifest_digest"],
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = load_source_only_split_manifest(args.manifest, verify_files=True)
    authorization = authorize_exploratory_training(
        manifest,
        ack_upstream_training_use_forbidden=args.ack_upstream_training_use_forbidden,
        ack_user_authorized_exploratory_training=(
            args.ack_user_authorized_exploratory_training
        ),
    )
    print(
        json.dumps(
            {
                "manifest": manifest.receipt(),
                "authorization": authorization,
                "training_contract": training_contract_receipt(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_STEPS",
    "CHECKPOINT_LOGICAL_RECORDS",
    "CheckpointCadence",
    "CleanSourceVisualTrainingError",
    "FULL644_ACTION_FAMILIES",
    "FULL644_ROWS",
    "MAX_OPTIMIZER_STEPS",
    "PINNED_FULL644_INDEX_SHA256",
    "PINNED_FULL644_SUMMARY_SHA256",
    "PINNED_RAW_PARQUET_SHA256",
    "PinnedPhysicalSourceOnlyPosteriorStore",
    "SPLIT_COUNTS",
    "SourceOnlyNoOpLatents",
    "SourceOnlySplitManifest",
    "SourceOnlySplitRow",
    "VisualContextCheckpointCoordinator",
    "authorize_exploratory_training",
    "build_source_only_split_manifest_value",
    "deterministic_source_hash_split",
    "load_source_only_split_manifest",
    "save_visual_context_checkpoint",
    "training_contract_receipt",
    "validate_source_only_split_manifest_value",
]
