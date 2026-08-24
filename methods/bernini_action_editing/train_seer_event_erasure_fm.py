#!/usr/bin/env python3
"""Train SEER with exact same-state action/no-op clean fields and full-pair FM.

This is the parameter-updating version of the self-generated event-erasure
idea.  Each accepted frozen-Bernini action video is paired with a source made
only from its pre-event frames.  Source and target therefore share the same
generated identity/background coordinate, while the requested transition is
absent from the source.

Every optimizer step executes two text branches on the *identical* noisy
target query, sigma, timestep, rotary geometry, and clean source prefix:

* action text receives ordinary full-pair flow-matching supervision;
* fixed semantic no-op text reconstructs the event-erased source; and
* their clean prediction difference matches the source-to-target causal
  motion field.

The implementation deliberately reuses ``train_delta_lora.py``'s audited
shared-noisy batch construction and optimizer/checkpoint path.  It does not
reuse the rejected cross-identity CMSG teacher-delta gate.  Training completion
is engineering evidence only; method success requires decoded held-out review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import train_delta_lora as base  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_seer_event_erasure_smoke as owner_contract  # noqa: E402


METHOD_NAME = "self-generated-event-erasure-same-state-fm-motion-copy-v1"
RECEIPT_SCHEMA = "bernini-seer-same-state-fm-training-receipt-v1"
OPTIMIZER_SCHEMA = "bernini-seer-same-state-fm-optimizer-v1"
MANIFEST_SCHEMA = "bernini-seer-event-erasure-dataset-v1"
INCLUSION_POLICY = "strict_single_actor"
AUTHORITY = {
    "experimental_parameter_update_authorized": True,
    "self_generated_video_target_authorized_for_this_fresh_experiment": True,
    "training_completion_is_method_success": False,
    "heldout_decoded_review_required": True,
    "production_claim_authorized": False,
}
ROW_FIELDS = frozenset(
    {
        "iid",
        "source_iid",
        "source_video",
        "source_video_sha256",
        "target_video",
        "target_video_sha256",
        "shared_i0_path",
        "shared_i0_sha256",
        "index_map_sha256",
        "prefix_rgb_exact",
        "transition_indices_absent",
    }
)
RAW_FIELDS = frozenset(
    {
        "parquet_path",
        "parquet_sha256",
        "receipt_path",
        "receipt_sha256",
        "job_done_path",
        "job_done_sha256",
    }
)
VAE_FIELDS = frozenset(
    {
        "parquet_directory",
        "dataset_summary_path",
        "dataset_summary_sha256",
        "index_path",
        "index_sha256",
        "row_count",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class SeerTrainingError(base.DeltaTrainingError):
    """Fail-closed SEER training contract error."""


@dataclass(frozen=True)
class SeerBinding:
    owner_path: Path
    owner_sha256: str
    manifest_path: Path
    manifest_sha256: str
    row_count: int
    iids: tuple[str, ...]
    manifest: Mapping[str, Any]


_SEALED_BINDING: Optional[SeerBinding] = None
_base_build_parser = base.build_parser
_base_validate_cli = base.validate_cli
_base_immutable_contract = base._immutable_contract
_base_supervision_receipt = base._supervision_receipt
_base_build_receipt = base._build_receipt
_base_save_checkpoint = base._save_checkpoint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SeerTrainingError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerTrainingError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SeerTrainingError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise SeerTrainingError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SeerTrainingError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SeerTrainingError(f"{label} is unavailable: {error}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise SeerTrainingError(f"{label} must be a plain directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeerTrainingError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SeerTrainingError(f"{label} must contain one object")
    return value


def _verified_file_binding(value: Mapping[str, Any], stem: str, *, label: str) -> Path:
    path = _plain_file(value.get(f"{stem}_path", ""), label=label)
    expected = _sha(value.get(f"{stem}_sha256"), label=f"{label} hash")
    if file_sha256(path) != expected:
        raise SeerTrainingError(f"{label} SHA-256 differs")
    return path


def _optional_manifest_digest(manifest: Mapping[str, Any]) -> None:
    if "manifest_digest" not in manifest:
        return
    candidate = dict(manifest)
    declared = candidate.pop("manifest_digest")
    if declared != legacy.object_sha256(candidate):
        raise SeerTrainingError("SEER dataset manifest digest differs")


def validate_manifest(
    *,
    owner_path: Path,
    expected_owner_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    verify_media: bool = True,
) -> SeerBinding:
    """Validate owner authority and the derived same-coordinate dataset."""

    expected_owner_sha256 = _sha(
        expected_owner_sha256, label="expected owner-spec hash"
    )
    expected_manifest_sha256 = _sha(
        expected_manifest_sha256, label="expected SEER manifest hash"
    )
    if file_sha256(owner_path) != expected_owner_sha256:
        raise SeerTrainingError("SEER owner-spec SHA-256 differs")
    try:
        owner = owner_contract._load_owner_spec(owner_path, expected_owner_sha256)
    except owner_contract.SeerSmokeError as error:
        raise SeerTrainingError(str(error)) from error
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise SeerTrainingError("SEER dataset manifest SHA-256 differs")
    manifest = _read_json(manifest_path, label="SEER dataset manifest")
    _optional_manifest_digest(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SeerTrainingError("SEER dataset manifest schema differs")
    owner_binding = manifest.get("owner_spec")
    if (
        not isinstance(owner_binding, Mapping)
        or set(owner_binding) != {"path", "sha256"}
        or Path(str(owner_binding.get("path"))).resolve() != owner_path
        or owner_binding.get("sha256") != expected_owner_sha256
    ):
        raise SeerTrainingError("SEER manifest owner-spec binding differs")
    authority = manifest.get("authority")
    if authority != AUTHORITY:
        raise SeerTrainingError("SEER fresh-experiment authority differs")

    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SeerTrainingError("SEER manifest rows must be a non-empty list")
    owner_rows = {str(row["iid"]): row for row in owner["rows"]}
    manifest_rows: dict[str, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != ROW_FIELDS:
            raise SeerTrainingError(f"SEER manifest row {index} field closure differs")
        iid = raw_row.get("iid")
        if (
            not isinstance(iid, str)
            or _IID.fullmatch(iid) is None
            or iid in manifest_rows
        ):
            raise SeerTrainingError(f"SEER manifest row {index} IID differs")
        wanted = owner_rows.get(iid)
        if wanted is None:
            raise SeerTrainingError(f"SEER manifest IID is outside owner authority: {iid}")
        if (
            raw_row.get("source_iid") != wanted.get("source_iid")
            or raw_row.get("target_video") != wanted.get("target_video")
            or raw_row.get("target_video_sha256")
            != wanted.get("target_video_sha256")
            or raw_row.get("source_video_sha256")
            == raw_row.get("target_video_sha256")
            or raw_row.get("prefix_rgb_exact") is not True
            or raw_row.get("transition_indices_absent") is not True
        ):
            raise SeerTrainingError(f"SEER row event-erasure binding differs: {iid}")
        for field in (
            "source_video_sha256",
            "target_video_sha256",
            "shared_i0_sha256",
            "index_map_sha256",
        ):
            _sha(raw_row.get(field), label=f"{iid} {field}")
        if verify_media:
            for path_field, hash_field, label in (
                ("source_video", "source_video_sha256", "source video"),
                ("target_video", "target_video_sha256", "target video"),
                ("shared_i0_path", "shared_i0_sha256", "shared I0"),
            ):
                path = _plain_file(raw_row[path_field], label=f"{iid} {label}")
                if file_sha256(path) != raw_row[hash_field]:
                    raise SeerTrainingError(f"{iid} {label} hash differs")
        manifest_rows[iid] = raw_row
    if set(manifest_rows) != set(owner_rows):
        raise SeerTrainingError("SEER manifest does not close the owner IID set")

    raw = manifest.get("raw")
    if not isinstance(raw, Mapping) or set(raw) != RAW_FIELDS:
        raise SeerTrainingError("SEER raw dataset binding differs")
    for stem, label in (
        ("parquet", "raw parquet"),
        ("receipt", "raw receipt"),
        ("job_done", "raw job-done"),
    ):
        if verify_media:
            _verified_file_binding(raw, stem, label=label)
        else:
            _sha(raw.get(f"{stem}_sha256"), label=f"{label} hash")
    vae = manifest.get("vae")
    if not isinstance(vae, Mapping) or set(vae) != VAE_FIELDS:
        raise SeerTrainingError(
            "SEER training requires the materialized VAE binding"
        )
    if vae.get("row_count") != len(rows):
        raise SeerTrainingError("SEER VAE row count differs")
    for field in ("dataset_summary_sha256", "index_sha256"):
        _sha(vae.get(field), label=f"SEER VAE {field}")
    return SeerBinding(
        owner_path=owner_path,
        owner_sha256=expected_owner_sha256,
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha256,
        row_count=len(rows),
        iids=tuple(sorted(manifest_rows)),
        manifest=manifest,
    )


def _blob_sha256(value: Any) -> str:
    try:
        return hashlib.sha256(value).hexdigest()
    except (TypeError, BufferError) as error:
        raise SeerTrainingError("materialized VAE latent is not a byte blob") from error


def _validate_renderer_messages(raw_row: Mapping[str, Any], instruction: str) -> None:
    try:
        messages = json.loads(str(raw_row.get("inputs")))
    except json.JSONDecodeError as error:
        raise SeerTrainingError("SEER renderer messages are invalid") from error
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or not isinstance(messages[1], Mapping)
        or messages[1].get("type") != "text"
        or messages[1].get("text") != instruction
    ):
        raise SeerTrainingError("SEER renderer instruction differs from owner spec")


def validate_full_pair_routing(
    *,
    binding: SeerBinding,
    routing_jsonl: Path,
    expected_routing_sha256: str,
) -> None:
    """Require one explicit unit-weight full-pair route per manifest IID."""

    expected_routing_sha256 = _sha(
        expected_routing_sha256, label="expected routing JSONL hash"
    )
    if file_sha256(routing_jsonl) != expected_routing_sha256:
        raise SeerTrainingError("SEER routing JSONL SHA-256 differs")
    try:
        router = motion.ReviewRouter.load(routing_jsonl, default_tier="reject")
    except motion.MotionContractError as error:
        raise SeerTrainingError(str(error)) from error
    receipt = router.receipt()
    if (
        receipt.get("explicit_route_counts")
        != {"full_pair": binding.row_count, "motion_only": 0, "reject": 0}
        or any(
            router.route(iid).tier != "full_pair"
            or router.route(iid).full_target_weight != 1.0
            for iid in binding.iids
        )
    ):
        raise SeerTrainingError("SEER requires exact full_pair=1 routes for all rows")


def cross_bind_materialized_dataset(
    *,
    binding: SeerBinding,
    parquet_directory: Path,
    dataset_summary_path: Path,
    routing_jsonl: Path,
    expected_routing_sha256: str,
) -> None:
    """Bind raw, VAE, index, routing, and actual parquet row bytes."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SeerTrainingError(
            "pyarrow is required to cross-bind the SEER training dataset"
        ) from error
    manifest = binding.manifest
    rows = {str(row["iid"]): row for row in manifest["rows"]}
    owner = owner_contract._load_owner_spec(
        binding.owner_path, binding.owner_sha256
    )
    owner_rows = {str(row["iid"]): row for row in owner["rows"]}
    vae = manifest["vae"]
    if Path(str(vae["parquet_directory"])).resolve() != parquet_directory:
        raise SeerTrainingError("SEER manifest points to a different VAE shard directory")
    if Path(str(vae["dataset_summary_path"])).resolve() != dataset_summary_path:
        raise SeerTrainingError("SEER manifest points to a different dataset summary")
    if file_sha256(dataset_summary_path) != vae["dataset_summary_sha256"]:
        raise SeerTrainingError("SEER dataset summary SHA-256 differs")
    summary = _read_json(dataset_summary_path, label="SEER dataset summary")
    if (
        summary.get("expected_sample_count") != binding.row_count
        or summary.get("materialized_sample_count") != binding.row_count
        or summary.get("missing_sample_count") != 0
        or summary.get("complete") is not True
        or summary.get("experimental_inclusion_policy") != INCLUSION_POLICY
        or Path(str(summary.get("shards_directory"))).resolve()
        != parquet_directory
    ):
        raise SeerTrainingError("SEER dataset summary cohort differs")
    index_path = _plain_file(vae["index_path"], label="SEER VAE index")
    if (
        Path(str(summary.get("index_path"))).resolve() != index_path
        or summary.get("index_sha256") != vae["index_sha256"]
        or file_sha256(index_path) != vae["index_sha256"]
    ):
        raise SeerTrainingError("SEER VAE index binding differs")
    index_rows: list[dict[str, Any]] = []
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise SeerTrainingError(
                        f"blank SEER VAE index row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SeerTrainingError("SEER VAE index row must be an object")
                index_rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SeerTrainingError(f"cannot read SEER VAE index: {error}") from error
    if len(index_rows) != binding.row_count:
        raise SeerTrainingError("SEER VAE index row count differs")

    raw_path = _plain_file(manifest["raw"]["parquet_path"], label="SEER raw parquet")
    raw_values = pq.read_table(raw_path).to_pylist()
    if len(raw_values) != binding.row_count:
        raise SeerTrainingError("SEER raw parquet row count differs")
    raw_by_iid = {str(row.get("iid")): row for row in raw_values}
    if set(raw_by_iid) != set(rows) or len(raw_by_iid) != len(raw_values):
        raise SeerTrainingError("SEER raw parquet IID closure differs")
    for iid, raw_row in raw_by_iid.items():
        declared = raw_row.get("renderer_row_digest")
        if declared != legacy.object_sha256(
            {key: value for key, value in raw_row.items() if key != "renderer_row_digest"}
        ):
            raise SeerTrainingError(f"SEER raw renderer row digest differs: {iid}")

    indexed_iids: set[str] = set()
    for index_row in index_rows:
        iid = str(index_row.get("iid"))
        if iid not in rows or iid in indexed_iids:
            raise SeerTrainingError("SEER VAE index IID closure differs")
        shard = _plain_file(index_row.get("parquet_path", ""), label=f"{iid} VAE shard")
        if (
            shard.parent != parquet_directory
            or shard.name != f"{iid}.parquet"
            or file_sha256(shard) != index_row.get("parquet_sha256")
        ):
            raise SeerTrainingError(f"SEER VAE shard binding differs: {iid}")
        values = pq.read_table(shard).to_pylist()
        if len(values) != 1 or not isinstance(values[0], dict):
            raise SeerTrainingError(f"SEER VAE shard row count differs: {iid}")
        value = values[0]
        wanted = rows[iid]
        owner_row = owner_rows[iid]
        if (
            value.get("iid") != iid
            or value.get("source_video_path") != wanted["source_video"]
            or value.get("source_video_sha256") != wanted["source_video_sha256"]
            or value.get("target_video_path") != wanted["target_video"]
            or value.get("target_video_sha256") != wanted["target_video_sha256"]
            or value.get("shared_i0_path") != wanted["shared_i0_path"]
            or value.get("shared_i0_sha256") != wanted["shared_i0_sha256"]
            or value.get("raw_renderer_row_digest")
            != raw_by_iid[iid].get("renderer_row_digest")
        ):
            raise SeerTrainingError(f"SEER materialized row binding differs: {iid}")
        _validate_renderer_messages(value, str(owner_row["instruction"]))
        blobs = value.get("video_vae_latents")
        if not isinstance(blobs, list) or len(blobs) != 2:
            raise SeerTrainingError(f"SEER VAE pair blobs differ: {iid}")
        calculated = legacy.object_sha256(
            {
                key: item
                for key, item in value.items()
                if key not in ("video_vae_latents", "materialized_row_digest")
            }
            | {"video_vae_latents_sha256": [_blob_sha256(blob) for blob in blobs]}
        )
        if (
            value.get("materialized_row_digest") != calculated
            or index_row.get("materialized_row_digest") != calculated
        ):
            raise SeerTrainingError(f"SEER materialized row digest differs: {iid}")
        indexed_iids.add(iid)
    if indexed_iids != set(rows):
        raise SeerTrainingError("SEER VAE index does not cover all manifest rows")

    validate_full_pair_routing(
        binding=binding,
        routing_jsonl=routing_jsonl,
        expected_routing_sha256=expected_routing_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _base_build_parser()
    parser.description = (
        "Train SEER same-state action/no-op FM + motion + copy LoRA"
    )
    parser.add_argument("--seer-owner-spec", required=True)
    parser.add_argument("--expected-seer-owner-spec-sha256", required=True)
    parser.add_argument("--seer-dataset-manifest", required=True)
    parser.add_argument("--expected-seer-manifest-sha256", required=True)
    parser.add_argument("--fm-loss-weight", type=float, default=1.0)
    parser.set_defaults(
        unreviewed_tier="reject",
        learning_rate=1.0e-6,
        lora_scope="cross_q_out",
        branch_state_mode="shared_noisy_clean_field",
        minimum_training_sigma=0.1,
        inverse_sigma_weight_floor=0.25,
        motion_loss_weight=0.5,
        copy_loss_weight=0.5,
        boundary_gauge_loss_weight=0.0,
        motion_objective="causal_boundary_charbonnier",
        bridge_consistency_weight=0.0,
        high_noise_floor=1.0,
        high_noise_power=2.0,
        max_steps=160,
        save_every=40,
        seed=20260813,
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    _base_validate_cli(args)
    fixed = {
        "unreviewed_tier": "reject",
        "learning_rate": 1.0e-6,
        "lora_scope": "cross_q_out",
        "branch_state_mode": "shared_noisy_clean_field",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": 0.25,
        "motion_loss_weight": 0.5,
        "copy_loss_weight": 0.5,
        "boundary_gauge_loss_weight": 0.0,
        "motion_objective": "causal_boundary_charbonnier",
        "bridge_consistency_weight": 0.0,
        "high_noise_floor": 1.0,
        "high_noise_power": 2.0,
        "fm_loss_weight": 1.0,
    }
    for name, expected in fixed.items():
        actual = getattr(args, name)
        if actual != expected:
            raise SeerTrainingError(
                f"SEER v1 fixes {name}={expected!r}, got {actual!r}"
            )
    if args.routing_jsonl is None:
        raise SeerTrainingError("SEER requires an explicit full_pair routing JSONL")
    _sha(
        args.expected_routing_jsonl_sha256,
        label="expected routing JSONL hash",
    )
    if args.allow_incomplete_dataset:
        raise SeerTrainingError("SEER requires the complete manifest-bound dataset")


def _immutable_contract(**kwargs: Any) -> dict[str, Any]:
    if _SEALED_BINDING is None:
        raise SeerTrainingError("SEER manifest was not sealed before training")
    args = kwargs["args"]
    result = _base_immutable_contract(**kwargs)
    value = dict(result["value"])
    value.update(
        {
            "method": METHOD_NAME,
            "expected_seer_owner_spec_sha256": _SEALED_BINDING.owner_sha256,
            "expected_seer_manifest_sha256": _SEALED_BINDING.manifest_sha256,
            "seer_row_count": _SEALED_BINDING.row_count,
            "seer_iids_sha256": legacy.object_sha256(list(_SEALED_BINDING.iids)),
            "seer_authority": dict(AUTHORITY),
            "same_generated_video_coordinate": True,
            "event_erasure_source_excludes_transition_and_terminal": True,
            "full_pair_flow_matching_weight": float(args.fm_loss_weight),
            "same_state_causal_motion_weight": float(args.motion_loss_weight),
            "same_state_noop_copy_weight": float(args.copy_loss_weight),
            "rejected_cmsg_cross_identity_gate_reused": False,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        }
    )
    return {
        "value": value,
        "digest": legacy.object_sha256(value),
        # Direct duplicates are required by the standalone postflight verifier;
        # the same values are also included in the hashed value above.
        "expected_seer_manifest_sha256": _SEALED_BINDING.manifest_sha256,
        "expected_seer_owner_spec_sha256": _SEALED_BINDING.owner_sha256,
        "method_source_archive_sha256": args.method_source_archive_sha256,
    }


def _supervision_receipt(args: argparse.Namespace) -> dict[str, Any]:
    result = dict(_base_supervision_receipt(args))
    result.update(
        {
            "self_generated_target_supervision": True,
            "event_erased_source_supervision": True,
            "same_generated_identity_background_coordinate": True,
            "full_pair_flow_matching_enabled": True,
            "full_pair_flow_matching_weight": float(args.fm_loss_weight),
            "same_state_causal_motion_weight": float(args.motion_loss_weight),
            "same_state_noop_copy_weight": float(args.copy_loss_weight),
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        }
    )
    return result


def _build_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
    if _SEALED_BINDING is None:
        raise SeerTrainingError("SEER manifest was not sealed before receipt creation")
    receipt = _base_build_receipt(*args, **kwargs)
    adapter = receipt.get("adapter")
    metrics = receipt.get("last_metrics")
    if not isinstance(adapter, Mapping):
        raise SeerTrainingError("SEER receipt lacks adapter parameter identity")
    initial = adapter.get("initialization_digest")
    final = adapter.get("checkpoint_parameter_digest")
    if (
        not isinstance(initial, str)
        or not isinstance(final, str)
        or initial == final
    ):
        raise SeerTrainingError("SEER optimizer did not change LoRA parameter bytes")
    if (
        not isinstance(metrics, Mapping)
        or isinstance(metrics.get("preclip_gradient_norm"), bool)
        or not isinstance(metrics.get("preclip_gradient_norm"), (int, float))
        or not math.isfinite(float(metrics["preclip_gradient_norm"]))
        or float(metrics["preclip_gradient_norm"]) <= 0.0
    ):
        raise SeerTrainingError("SEER final gradient norm is not finite and positive")
    receipt["seer"] = {
        "owner_spec_path": str(_SEALED_BINDING.owner_path),
        "owner_spec_sha256": _SEALED_BINDING.owner_sha256,
        "dataset_manifest_path": str(_SEALED_BINDING.manifest_path),
        "dataset_manifest_sha256": _SEALED_BINDING.manifest_sha256,
        "row_count": _SEALED_BINDING.row_count,
        "self_generated_target_supervision": True,
        "event_erased_source_supervision": True,
        "training_completion_is_method_success": False,
        "heldout_decoded_review_required": True,
    }
    receipt["parameter_update_evidence"] = {
        "initial_trainable_parameter_digest": initial,
        "final_trainable_parameter_digest": final,
        "exact_parameter_bytes_changed": True,
        "final_preclip_gradient_norm": float(metrics["preclip_gradient_norm"]),
        "engineering_execution_success": True,
        "method_success_claimed": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    return receipt


def _install_specialization() -> None:
    base.METHOD_NAME = METHOD_NAME
    base.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    base.OPTIMIZER_SCHEMA = OPTIMIZER_SCHEMA
    base.build_parser = build_parser
    base.validate_cli = validate_cli
    base._immutable_contract = _immutable_contract
    base._supervision_receipt = _supervision_receipt
    base._build_receipt = _build_receipt
    # This assignment documents that the standard PEFT checkpoint writer is
    # retained; it writes adapter_config.json + adapter_model.safetensors.
    base._save_checkpoint = _base_save_checkpoint


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _SEALED_BINDING

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_cli(args)
    owner_path = _plain_file(args.seer_owner_spec, label="SEER owner spec")
    manifest_path = _plain_file(
        args.seer_dataset_manifest, label="SEER dataset manifest"
    )
    binding = validate_manifest(
        owner_path=owner_path,
        expected_owner_sha256=args.expected_seer_owner_spec_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha256=args.expected_seer_manifest_sha256,
        verify_media=True,
    )
    parquet_directory = _directory(
        args.preprocessed_parquet_dir, label="SEER preprocessed parquet directory"
    )
    summary_path = _plain_file(args.dataset_summary, label="SEER dataset summary")
    routing_path = _plain_file(args.routing_jsonl, label="SEER routing JSONL")
    cross_bind_materialized_dataset(
        binding=binding,
        parquet_directory=parquet_directory,
        dataset_summary_path=summary_path,
        routing_jsonl=routing_path,
        expected_routing_sha256=args.expected_routing_jsonl_sha256,
    )

    # ``train_lora`` originally binds a 644-row preview release.  SEER has its
    # own manifest authority, so derive all counts from that sealed manifest.
    legacy.EXPECTED_DATASET_ROWS = binding.row_count
    legacy.EXPECTED_STRICT_ROWS = binding.row_count
    legacy.EXPECTED_NON_STRICT_ROWS = 0
    legacy.EXPECTED_INCLUSION_POLICY = INCLUSION_POLICY
    _SEALED_BINDING = binding
    _install_specialization()
    return base.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SeerTrainingError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
