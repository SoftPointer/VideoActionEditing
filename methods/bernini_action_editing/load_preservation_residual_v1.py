#!/usr/bin/env python3
"""Strict resolver/loader for preservation-residual training bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import preservation_source_role_v1 as role
import train_preservation_residual_v1 as training


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PreservationResidualLoadError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class PreservationBundle:
    root: Path
    adapter: Path
    receipt_path: Path
    adapter_sha256: str
    receipt_sha256: str
    receipt_digest: str
    adapter_rank: int
    receipt: Mapping[str, Any]


def resolve_bundle(
    value: str | Path,
    *,
    expected_adapter_sha256: str,
    expected_receipt_sha256: str,
) -> PreservationBundle:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise PreservationResidualLoadError("training bundle must be absolute non-symlink")
    root = requested.resolve(strict=True)
    if root != requested or not stat.S_ISDIR(root.lstat().st_mode):
        raise PreservationResidualLoadError("training bundle directory differs")
    entries = {item.name: item for item in root.iterdir()}
    if set(entries) != {"adapter.safetensors", "optimizer.pt", "history.json", "receipt.json"}:
        raise PreservationResidualLoadError("training bundle file closure differs")
    if any(item.is_symlink() or not item.is_file() for item in entries.values()):
        raise PreservationResidualLoadError("training bundle contains a non-plain file")
    if not _SHA256.fullmatch(expected_adapter_sha256) or not _SHA256.fullmatch(expected_receipt_sha256):
        raise PreservationResidualLoadError("expected file SHA differs")
    adapter_sha = _sha(entries["adapter.safetensors"])
    receipt_sha = _sha(entries["receipt.json"])
    if adapter_sha != expected_adapter_sha256 or receipt_sha != expected_receipt_sha256:
        raise PreservationResidualLoadError("training bundle file SHA differs")
    receipt = json.loads(entries["receipt.json"].read_text(encoding="ascii"))
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or hashlib.sha256(_canonical(unsigned)).hexdigest() != declared:
        raise PreservationResidualLoadError("training receipt digest differs")
    objective = receipt.get("objective")
    optimizer_steps = receipt.get("optimizer_steps")
    required = {
        "schema_version": training.RUN_RECEIPT_SCHEMA,
        "method": training.METHOD_NAME,
        "complete": True,
        "mode": training.MODE,
        "registered_schedule_indices": list(training.REGISTERED_SCHEDULE_INDICES),
        "base_frozen": True,
        "frozen_base_action_prior_not_retrained": True,
    }
    if any(receipt.get(name) != expected for name, expected in required.items()):
        raise PreservationResidualLoadError("training receipt contract differs")
    if optimizer_steps not in training.LOADABLE_CHECKPOINT_STEPS:
        raise PreservationResidualLoadError("training checkpoint step differs")
    expected_prefix = list(training.REGISTERED_SCHEDULE_INDICES[:optimizer_steps])
    # Full exact40 bundles published before checkpoint-cadence support did not
    # carry this redundant prefix field.  Partial checkpoints must carry it.
    observed_prefix = receipt.get("training_schedule_indices")
    if observed_prefix is None and optimizer_steps == training.OPTIMIZER_STEPS:
        observed_prefix = expected_prefix
    if observed_prefix != expected_prefix:
        raise PreservationResidualLoadError("training checkpoint schedule prefix differs")
    if (
        not isinstance(objective, Mapping)
        or objective.get("name") != "single_preservation_residual_mse"
        or objective.get("action_reward") is not False
        or objective.get("feature_reward") is not False
        or objective.get("synthetic_target") is not False
    ):
        raise PreservationResidualLoadError("preservation objective receipt differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("adapter.safetensors") != adapter_sha:
        raise PreservationResidualLoadError("adapter artifact binding differs")
    rank = receipt.get("adapter_rank")
    if rank not in {training.LOW_CAPACITY_LORA_RANK, training.MAIN_LORA_RANK}:
        raise PreservationResidualLoadError("adapter rank differs")
    return PreservationBundle(
        root,
        entries["adapter.safetensors"],
        entries["receipt.json"],
        adapter_sha,
        receipt_sha,
        declared,
        int(rank),
        receipt,
    )


def strict_load(transformer: Any, bundle: PreservationBundle) -> tuple[Any, Mapping[str, Any]]:
    import torch
    from safetensors import safe_open

    handle = role.install_source_self_adapter(
        transformer,
        rank=bundle.adapter_rank,
        alpha=float(bundle.adapter_rank),
        block_indices=role.TRAINABLE_BLOCK_INDICES,
    )
    try:
        with safe_open(str(bundle.adapter), framework="pt", device="cpu") as opened:
            metadata = dict(opened.metadata() or {})
            keys = tuple(sorted(opened.keys()))
            tensors = {name: opened.get_tensor(name).contiguous() for name in keys}
        expected_metadata = {
            "schema_version": training.ADAPTER_FILE_SCHEMA,
            "role_adapter_schema_version": role.SCHEMA_VERSION,
            "block_indices_json": training.canonical_json_bytes(list(role.TRAINABLE_BLOCK_INDICES)).decode("ascii"),
            "projections_json": training.canonical_json_bytes(["attn1.to_q", "attn1.to_out.0"]).decode("ascii"),
            "target_row_only": "true",
            "role_embedding": "donor_reference_target",
            "lora_rank": str(bundle.adapter_rank),
            "lora_alpha_hex": float(bundle.adapter_rank).hex(),
            "exact40_schedule_sha256": training.EXPECTED_EXACT40_SCHEDULE_SHA256,
            "registered_schedule_indices_json": training.canonical_json_bytes(list(training.REGISTERED_SCHEDULE_INDICES)).decode("ascii"),
            "target_and_source_same_epsilon": "true",
            "forward_noising_only": "true",
            "inversion_claimed": "false",
            "matched_carrier_runtime_required": "true",
            "objective": "mse_adapted_minus_base_vs_target_minus_base",
            "action_reward_consumed": "false",
            "synthetic_target_consumed": "false",
        }
        if metadata != expected_metadata:
            raise PreservationResidualLoadError("adapter metadata differs")
        named = handle.trainable_named_parameters()
        parameter_map = dict(named)
        if keys != tuple(sorted(parameter_map)):
            raise PreservationResidualLoadError("adapter tensor closure differs")
        with torch.no_grad():
            for name in keys:
                tensor = tensors[name]
                parameter = parameter_map[name]
                if (
                    tensor.dtype != torch.float32
                    or tuple(tensor.shape) != tuple(parameter.shape)
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise PreservationResidualLoadError(f"adapter tensor differs: {name}")
                parameter.copy_(tensor.to(parameter.device))
        parameter_digest = __import__("source_self_runtime").trainable_parameters_digest(named)
        if parameter_digest != bundle.receipt.get("final_adapter_sha256"):
            raise PreservationResidualLoadError("loaded adapter parameter digest differs")
        for _, parameter in named:
            parameter.requires_grad_(False)
        return handle, {
            "schema_version": "bernini-preservation-residual-strict-load-v1",
            "adapter_file_sha256": bundle.adapter_sha256,
            "training_receipt_sha256": bundle.receipt_sha256,
            "training_receipt_digest": bundle.receipt_digest,
            "adapter_rank": bundle.adapter_rank,
            "parameter_digest": parameter_digest,
            "strict_tensor_and_metadata_closure": True,
            "all_adapter_parameters_frozen_for_inference": True,
        }
    except Exception:
        if not handle.restored:
            handle.restore()
        raise


__all__ = [
    "PreservationBundle",
    "PreservationResidualLoadError",
    "resolve_bundle",
    "strict_load",
]
