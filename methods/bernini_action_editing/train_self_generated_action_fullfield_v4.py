#!/usr/bin/env python3
"""Historical V4 self-generated-anchor trainer (permanently invalidated).

The objective produced post-phase0 black/colour-noise collapse and conflicts
with ``md/action_editing/20260817_box``.  Historical parsing helpers remain for
failure-audit reproducibility, but :func:`main` fails before argument parsing,
model construction, or optimizer construction.  Do not use this file as an
initialization or training entrypoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import self_generated_action_fullfield_v4 as objective
import train_lora as legacy
import train_self_generated_action_quotient_v1 as data


METHOD = objective.SCHEMA
RECEIPT_SCHEMA = "bernini-r-1p3b-self-generated-fullfield-lora-receipt-v4"
FULL644_MANIFEST_SCHEMA = "bernini-full644-self-generated-action-anchor-manifest-v1"
FULL644_ROW_COUNT = 644
FULL644_AUTHORIZATION = "user_explicit_self_generated_action_anchor_training_20260818"
ARMS = (
    "direct_anchor_sft",
    "fullfield_action_noop",
    "fullfield_action_noop_pcgrad_preserve",
    "source_carrier_sft",
)
LEARNING_RATES = {
    "direct_anchor_sft": 5.0e-4,
    "fullfield_action_noop": 3.0e-4,
    "fullfield_action_noop_pcgrad_preserve": 3.0e-4,
    "source_carrier_sft": 5.0e-4,
}
LORA_RANK = 256
LORA_ALPHA = 256
EXPECTED_TARGET_MODULES = 240
EXPECTED_TRAINABLE_PARAMETERS = 188_743_680
MEMORY_FRACTION_GATE = 0.50
PCGRAD_PRESERVATION_CAP = 0.25
SELECTIVE_CHECKPOINT_STRIDE = 4
SAVE_STEPS = (1, 5, 10, 20, 40, 80)
_BLOCK = re.compile(r"\.blocks\.(\d+)\.")


class FullFieldTrainingError(RuntimeError):
    pass


LEGACY_TRAINING_BLOCKED_STATUS = (
    "LEGACY_SELF_GENERATED_ANCHOR_OBJECTIVE_INVALIDATED_USE_EXACT160_BOX"
)


def fail(message: str) -> None:
    raise FullFieldTrainingError(message)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--source-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--arm", choices=ARMS, required=True)
    value.add_argument("--max-steps", type=int, default=40)
    value.add_argument("--micro-records", type=int, default=1)
    value.add_argument("--overfit-row", type=int, default=None)
    value.add_argument("--seed", type=int, default=20260820)
    value.add_argument("--max-grad-norm", type=float, default=100.0)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive-sha256", required=True)
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.micro_records <= 0 or args.micro_records > 8:
        fail("max-steps must be positive and micro-records must lie in [1,8]")
    if args.overfit_row is not None and args.overfit_row < 0:
        fail("overfit-row must be non-negative")
    if not math.isfinite(float(args.max_grad_norm)) or float(args.max_grad_norm) <= 0:
        fail("max-grad-norm must be finite and positive")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_manifest_sha256):
        fail("source manifest SHA-256 differs")
    if not re.fullmatch(r"[0-9a-f]{40}", args.method_source_revision):
        fail("method source revision must be a full SHA-1")
    if not re.fullmatch(r"[0-9a-f]{64}", args.method_source_archive_sha256):
        fail("method source archive SHA-256 differs")


def normalized_clean_to_posterior_blob(clean: Any, mean: Any, std: Any) -> bytes:
    import torch

    if clean.ndim != 5 or tuple(clean.shape[:3]) != (1, 16, 21):
        fail("normalized clean target geometry differs")
    raw_mean = clean.float() * std.unsqueeze(0) + mean.unsqueeze(0)
    posterior = torch.cat(
        (raw_mean, torch.full_like(raw_mean, -30.0)), dim=1
    ).contiguous()
    return data.tensor_blob(posterior)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_manifest(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the stable V1 manifest without inheriting its objective policy."""

    resolved = path.resolve(strict=True)
    if _file_sha256(resolved) != expected_sha256:
        fail("source manifest byte identity differs")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    stored = value.pop("manifest_digest", None)
    schema = value.get("schema_version")
    if schema not in (data.SOURCE_MANIFEST_SCHEMA, FULL644_MANIFEST_SCHEMA) or data.object_sha(value) != stored:
        fail("source manifest semantic digest differs")
    rows = value.get("rows")
    expected_rows = 4 if schema == data.SOURCE_MANIFEST_SCHEMA else FULL644_ROW_COUNT
    if not isinstance(rows, list) or len(rows) != expected_rows:
        fail(f"V4 source manifest must contain exactly {expected_rows} rows")
    if schema == FULL644_MANIFEST_SCHEMA:
        if (
            value.get("authorization_label") != FULL644_AUTHORIZATION
            or value.get("row_count") != FULL644_ROW_COUNT
            or value.get("optimizer_schedule") != "exact644_unique_rows_once"
            or value.get("source_anchor_role")
            != "identity_appearance_background_camera_and_non_target_preservation"
            or value.get("self_generated_action_anchor_role")
            != "dense_action_trajectory_supervision"
            or value.get("paired_ground_truth_claimed") is not False
            or value.get("qwen_or_other_verifier_controls_optimizer_admission") is not False
            or value.get("production_claim_forbidden") is not True
            or value.get("scientific_claim_authorized") is not False
        ):
            fail("full644 action-anchor authority differs")
        if [row.get("iid") for row in rows] != sorted(row.get("iid") for row in rows):
            fail("full644 row order differs")
        if len({row.get("iid") for row in rows}) != FULL644_ROW_COUNT:
            fail("full644 row identity uniqueness differs")
    for row in rows:
        if schema == FULL644_MANIFEST_SCHEMA:
            pair = row.get("posterior_pair")
            if (
                not isinstance(pair, dict)
                or pair.get("source_role_index") != 0
                or pair.get("action_anchor_role_index") != 1
                or not isinstance(row.get("noop_instruction"), str)
            ):
                fail("full644 posterior role closure differs")
            parquet = Path(pair["parquet_path"]).resolve(strict=True)
            if str(parquet) != pair["parquet_path"]:
                fail("full644 parquet path is not canonical")
        else:
            source = Path(row["source_posterior"]["path"]).resolve(strict=True)
            anchor = Path(row["action_anchor"]["latent_path"]).resolve(strict=True)
            if _file_sha256(source) != row["source_posterior"]["sha256"]:
                fail(f"source posterior hash differs: {row.get('iid')}")
            if _file_sha256(anchor) != row["action_anchor"]["latent_sha256"]:
                fail(f"action anchor hash differs: {row.get('iid')}")
            if set(row.get("teacher_captions", {})) != {
                "action", "noop", "camera_only", "appearance_only"
            }:
                fail("teacher caption role closure differs")
    return {**value, "manifest_digest": stored}, rows


def _row_latents(
    row: Mapping[str, Any], mean: Any, std: Any
) -> tuple[bytes, bytes, Any, Any]:
    if "posterior_pair" in row:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pair = row["posterior_pair"]
        parquet_path = Path(pair["parquet_path"])
        parquet_raw = parquet_path.read_bytes()
        if _file_sha256_bytes(parquet_raw) != pair["parquet_sha256"]:
            fail(f"full644 parquet SHA-256 differs: {row.get('iid')}")
        table = pq.read_table(
            pa.BufferReader(parquet_raw), columns=["iid", "video_vae_latents"]
        )
        if table.num_rows != 1:
            fail("full644 parquet row count differs")
        parquet_row = table.to_pylist()[0]
        latents = parquet_row.get("video_vae_latents")
        if parquet_row.get("iid") != row.get("iid") or not isinstance(latents, list) or len(latents) != 2:
            fail("full644 posterior row identity/roles differ")
        source_blob, anchor_blob = bytes(latents[0]), bytes(latents[1])
        if _file_sha256_bytes(source_blob) != pair["source_blob_sha256"]:
            fail("full644 source posterior SHA-256 differs")
        if _file_sha256_bytes(anchor_blob) != pair["action_anchor_blob_sha256"]:
            fail("full644 action anchor posterior SHA-256 differs")
        source_clean = data.source_clean_from_posterior(source_blob, mean, std)
        anchor_clean = data.source_clean_from_posterior(anchor_blob, mean, std)
    else:
        from safetensors.torch import load_file

        source_path = Path(row["source_posterior"]["path"])
        anchor_path = Path(row["action_anchor"]["latent_path"])
        source_blob = source_path.read_bytes()
        source_clean = data.source_clean_from_posterior(source_blob, mean, std)
        tensors = load_file(str(anchor_path), device="cpu")
        if set(tensors) != {"normalized_clean_latent"}:
            fail("action anchor safetensors key closure differs")
        anchor_clean = tensors["normalized_clean_latent"].float().contiguous()
        if anchor_clean.ndim != 5 or tuple(anchor_clean.shape[:3]) != (1, 16, 21):
            fail("action anchor full-field geometry differs")
        anchor_blob = normalized_clean_to_posterior_blob(anchor_clean, mean, std)
    return source_blob, anchor_blob, source_clean, anchor_clean


def _file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _velocity_target(batch: Mapping[str, Any], shape: Sequence[int]) -> Any:
    return data.patches_to_spatial(
        batch["target_velocity"], spatial_shape=shape
    ).float()


def install_selective_activation_checkpointing(model: Any) -> list[int]:
    """Checkpoint 8/30 blocks while preserving module/state-dict names."""

    import torch
    from torch.utils.checkpoint import checkpoint

    transformer = model.get_base_model().diff_dec.transformer
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or len(blocks) != 30:
        fail("selective checkpointing requires the exact 30-block Wan transformer")
    chosen = list(range(0, 30, SELECTIVE_CHECKPOINT_STRIDE))
    for index in chosen:
        block = blocks[index]
        original = block.forward

        def checkpointed_forward(
            *args: Any, _original: Any = original, **kwargs: Any
        ) -> Any:
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(
                _original, *args, use_reentrant=False, **kwargs
            )

        block.forward = checkpointed_forward
    return chosen


def _build_record(
    *,
    row: Mapping[str, Any],
    arm: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> dict[str, Any]:
    source_blob, anchor_blob, source_clean, anchor_clean = _row_latents(
        row, mean, std
    )
    if tuple(source_clean.shape) != tuple(anchor_clean.shape):
        fail("source/anchor full-field geometry differs")
    shape = tuple(int(item) for item in source_clean.shape)
    if arm == "direct_anchor_sft":
        action = transform(
            data.make_sample(
                instruction=row["instruction"],
                source_blob=source_blob,
                target_blob=anchor_blob,
            ),
            seed,
        )
        return {"action": action, "shape": shape, "anchor_clean": anchor_clean}
    if arm == "source_carrier_sft":
        carrier = objective.source_carrier_target(source_clean, anchor_clean)
        carrier_blob = normalized_clean_to_posterior_blob(carrier, mean, std)
        action = transform(
            data.make_sample(
                instruction=row["instruction"],
                source_blob=source_blob,
                target_blob=carrier_blob,
            ),
            seed,
        )
        return {
            "action": action,
            "shape": shape,
            "anchor_clean": anchor_clean,
            "carrier_clean": carrier,
        }
    action = transform(
        data.make_sample(
            instruction=row["instruction"],
            source_blob=source_blob,
            target_blob=source_blob,
        ),
        seed,
    )
    noop = transform(
        data.make_sample(
            instruction=(
                row["noop_instruction"]
                if "noop_instruction" in row
                else row["teacher_captions"]["noop"]
            ),
            source_blob=source_blob,
            target_blob=source_blob,
        ),
        seed,
    )
    action_state = data.paired_state(action, source_clean)
    noop_state = data.paired_state(noop, source_clean)
    if action_state[3] != noop_state[3] or abs(action_state[2] - noop_state[2]) > 1.0e-7:
        fail("source action/no-op branches do not share exact noise/timestep")
    return {
        "action": action,
        "noop": noop,
        "shape": shape,
        "anchor_clean": anchor_clean,
        "source_clean": source_clean,
        "source_noisy": action_state[0],
        "sigma": action_state[2],
    }


def _record_losses(renderer: Any, record: Mapping[str, Any], arm: str) -> tuple[Any, Optional[Any], dict[str, Any]]:
    import torch

    shape = record["shape"]
    if arm in ("direct_anchor_sft", "source_carrier_sft"):
        action_velocity = data.predicted_target_velocity(
            renderer, record["action"], spatial_shape=shape
        )
        target_velocity = _velocity_target(record["action"], shape)
        flow = torch.nn.functional.mse_loss(action_velocity, target_velocity)
        return flow, None, {
            "flow_matching": flow.detach(),
            "dense": flow.detach().new_zeros(()),
            "multiscale": flow.detach().new_zeros(()),
            "teacher_rms": flow.detach().new_zeros(()),
            "student_rms": flow.detach().new_zeros(()),
            "preservation": flow.detach().new_zeros(()),
        }
    # Build the sole retained graph explicitly, then evaluate no-op as a
    # stop-gradient counterfactual at the current adapter state.  No-op is not
    # a Frozen target or a trust radius.
    with torch.enable_grad():
        action_velocity = data.predicted_target_velocity(
            renderer, record["action"], spatial_shape=shape
        )
    if not bool(action_velocity.requires_grad):
        fail("full-field action velocity lost its LoRA gradient graph")
    with torch.no_grad():
        noop_velocity = data.predicted_target_velocity(
            renderer, record["noop"], spatial_shape=shape
        )
    action_clean = objective.predicted_clean(
        record["source_noisy"], action_velocity, record["sigma"]
    )
    noop_clean = objective.predicted_clean(
        record["source_noisy"], noop_velocity, record["sigma"]
    )
    student = objective.student_action_trajectory(action_clean, noop_clean)
    teacher = objective.anchor_action_trajectory(record["anchor_clean"].to(student.device))
    parts = objective.fullfield_action_loss(student, teacher)
    if not bool(parts.total.requires_grad):
        fail(
            "full-field loss lost its action graph: "
            f"velocity={action_velocity.requires_grad} "
            f"action_clean={action_clean.requires_grad} "
            f"student={student.requires_grad}"
        )
    preservation_diagnostic = torch.nn.functional.mse_loss(
        noop_velocity, _velocity_target(record["noop"], shape)
    )
    return parts.total, None, {
        "flow_matching": parts.total.detach().new_zeros(()),
        "dense": parts.dense.detach(),
        "multiscale": parts.multiscale.detach(),
        "teacher_rms": parts.teacher_rms.detach(),
        "student_rms": parts.student_rms.detach(),
        "preservation": preservation_diagnostic.detach(),
    }


def _preservation_loss(renderer: Any, record: Mapping[str, Any]) -> Any:
    """Run the source-noop graph only after the action graph has been freed."""

    import torch

    shape = record["shape"]
    noop_velocity = data.predicted_target_velocity(
        renderer, record["noop"], spatial_shape=shape
    )
    return torch.nn.functional.mse_loss(
        noop_velocity, _velocity_target(record["noop"], shape)
    )


def _set_gradients(named: Sequence[tuple[str, Any]], gradients: Sequence[Any]) -> None:
    if len(named) != len(gradients):
        fail("gradient list length differs from optimizer parameters")
    for (_, parameter), gradient in zip(named, gradients):
        if gradient is None:
            fail("one full30 trainable parameter lacks a gradient")
        parameter.grad = gradient.detach()


def _all_reduce_gradient_list(named: Sequence[tuple[str, Any]], gradients: Sequence[Any]) -> tuple[list[Any], float]:
    _set_gradients(named, gradients)
    norm = legacy.all_reduce_lora_gradients(named)
    return [parameter.grad.detach().clone() for _, parameter in named], norm


def _gradient_coverage(named: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    flags = []
    blocks: set[int] = set()
    for name, parameter in named:
        if parameter.grad is None:
            flags.append(False)
            continue
        active = bool(torch.linalg.vector_norm(parameter.grad.detach().float()).item() > 0.0)
        flags.append(active)
        if active:
            match = _BLOCK.search(name)
            if match is not None:
                blocks.add(int(match.group(1)))
    return {
        "active_tensor_count": sum(flags),
        "trainable_tensor_count": len(flags),
        "active_tensor_fraction": sum(flags) / len(flags),
        "active_blocks": sorted(blocks),
    }


def _memory_receipt(device: Any, micro_records: int) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    total = int(torch.cuda.get_device_properties(device).total_memory)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    local = {
        "rank": int(dist.get_rank()),
        "total_bytes": total,
        "max_allocated_bytes": allocated,
        "max_reserved_bytes": reserved,
        "reserved_fraction": reserved / total,
        "micro_records": int(micro_records),
    }
    gathered: list[Any] = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, local)
    minimum = min(float(item["reserved_fraction"]) for item in gathered)
    return {
        "per_rank": gathered,
        "minimum_reserved_fraction": minimum,
        "required_strictly_above": MEMORY_FRACTION_GATE,
        "passed": minimum > MEMORY_FRACTION_GATE,
        "true_training_tensors_only": True,
        "dummy_or_padding_allocations": False,
    }


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        fail("micro-record metric closure differs")
    return {
        key: sum(float(row[key].item()) for row in rows) / len(rows)
        for key in sorted(keys)
    }


def checkpoint_receipt(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    step: int,
    last_metrics: Mapping[str, float],
    grad_norm: float,
    memory: Mapping[str, Any],
    coverage: Mapping[str, Any],
    targets: Sequence[str],
    initial_digest: str,
    bernini_revision: str,
    veomni_revision: str,
    transformers_version: str,
    consumed_row_ids: Sequence[str],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "global_step": step,
        "max_steps": args.max_steps,
        "last_loss": float(last_metrics["total"]),
        "last_loss_components": dict(last_metrics),
        "last_preclip_gradient_norm": float(grad_norm),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "bernini_training_files_index_sha256": legacy.object_sha256(
            legacy.BERNINI_PINNED_FILE_HASHES
        ),
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "source_manifest_digest": manifest["manifest_digest"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "dataset_row_count": len(manifest["rows"]),
        "consumed_unique_row_count": len(set(consumed_row_ids)),
        "consumed_row_ids_sha256": legacy.object_sha256(list(consumed_row_ids)),
        "all_manifest_rows_consumed_exactly_once": (
            len(consumed_row_ids) == FULL644_ROW_COUNT
            and len(set(consumed_row_ids)) == FULL644_ROW_COUNT
            and args.micro_records * args.max_steps == FULL644_ROW_COUNT
        ),
        "initialization_seed": args.seed,
        "training_contract": {
            "method": METHOD,
            "arm": args.arm,
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": legacy.TASK_SOURCE_NAME,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
            "gradient_checkpointing": "selective_nonreentrant_stride4",
            "selective_checkpoint_blocks": list(
                range(0, 30, SELECTIVE_CHECKPOINT_STRIDE)
            ),
            "micro_records_retained_to_one_backward": args.micro_records,
            "full_field_shape": "[B,16,21,H,W]",
            "action_teacher": (
                "self_generated_action_anchor_dense_trajectory_minus_phase0"
            ),
            "source_anchor": (
                "identity_appearance_background_camera_and_non_target_preservation"
            ),
            "qwen_or_other_verifier_controls_optimizer_admission": False,
            "noop_counterfactual": "current_adapter_detached_full_clean_field",
            "frozen_rv2v_action_target": False,
            "frozen_relative_band_or_trust_radius": False,
            "pooled_or_32d_representation": False,
            "temporal_lags": list(objective.DEFAULT_LAGS),
            "phase0_action_teacher_exact_zero": True,
            "pcgrad_preservation_cap": (
                PCGRAD_PRESERVATION_CAP
                if args.arm == "fullfield_action_noop_pcgrad_preserve"
                else None
            ),
            "transformers_version": transformers_version,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": LEARNING_RATES[args.arm],
            "weight_decay": 0.0,
            "max_grad_norm": float(args.max_grad_norm),
        },
        "target_module_count": len(targets),
        "target_modules": list(targets),
        "target_modules_sha256": legacy.object_sha256(list(targets)),
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "same_sample_all_ranks": True,
            "explicit_gradient_all_reduce": True,
            "lora_initialization_digest": initial_digest,
        },
        "memory_gate": dict(memory),
        "gradient_coverage": dict(coverage),
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "experimental_training": True,
    }
    value["receipt_digest"] = legacy.object_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    fail(LEGACY_TRAINING_BLOCKED_STATUS)
    args = parser().parse_args(argv)
    validate_args(args)
    bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4:
        fail("V4 requires one exact SP4 four-rank worker")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    manifest, rows = load_source_manifest(
        Path(args.source_manifest), args.source_manifest_sha256
    )
    if args.overfit_row is not None and args.overfit_row >= len(rows):
        fail("overfit-row exceeds the source manifest")
    if manifest["schema_version"] == FULL644_MANIFEST_SCHEMA and (
        args.arm != "fullfield_action_noop_pcgrad_preserve"
        or args.overfit_row is not None
        or args.max_steps * args.micro_records != FULL644_ROW_COUNT
    ):
        fail(
            "full644 training requires the PCGrad preservation arm and exactly "
            "644 unique scheduled row exposures"
        )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    with data.serialized_model_load():
        base = BerniniRendererModel(config)
        base.requires_grad_(False)
        base.t5_text_encoder.eval()
        # Checkpoint only a fixed 8/30 block subset to leave enough headroom
        # for the largest real video geometry.  The optimizer still covers
        # every attention projection in all 30 blocks, and the post-Adam gate
        # rejects any resulting run whose real peak is not strictly >50%.
        targets = legacy.select_attention_projection_names(base)
        if len(targets) != EXPECTED_TARGET_MODULES:
            fail("full30 attention target count differs")
        model = get_peft_model(
            base,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
        )
        selective_checkpoint_blocks = install_selective_activation_checkpointing(
            model
        )
        if selective_checkpoint_blocks != list(range(0, 30, 4)):
            fail("selective checkpoint block schedule differs")
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    named = legacy.trainable_lora_parameters(model)
    trainable_count = sum(int(parameter.numel()) for _, parameter in named)
    if trainable_count != EXPECTED_TRAINABLE_PARAMETERS:
        fail(
            f"full30 rank256 trainable count differs: {trainable_count}"
        )
    initial_digest = legacy.synchronize_trainable_parameters(named, source_rank=0)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform = data.build_transform(
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=LEARNING_RATES[args.arm],
        weight_decay=0.0,
    )
    model.train()
    model.get_base_model().t5_text_encoder.eval()
    torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        fail(f"training output exists: {output}")

    initial_loss: Optional[float] = None
    memory: Optional[dict[str, Any]] = None
    coverage: dict[str, Any] = {
        "active_tensor_count": 0,
        "trainable_tensor_count": len(named),
        "active_tensor_fraction": 0.0,
        "active_blocks": [],
    }
    last_metrics: dict[str, float] = {}
    last_grad_norm = 0.0
    def row_storage_size(index: int) -> int:
        row = rows[index]
        path = (
            row["posterior_pair"]["parquet_path"]
            if "posterior_pair" in row
            else row["source_posterior"]["path"]
        )
        return Path(path).stat().st_size

    formal_row_order = sorted(
        range(len(rows)), key=row_storage_size, reverse=True
    )
    consumed_row_ids: list[str] = []
    for global_step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        action_losses = []
        records = []
        metric_rows = []
        row_indices = []
        for micro in range(args.micro_records):
            row_index = (
                args.overfit_row
                if args.overfit_row is not None
                else formal_row_order[
                    (global_step * args.micro_records + micro) % len(rows)
                ]
            )
            row_indices.append(row_index)
            consumed_row_ids.append(str(rows[row_index]["iid"]))
            # An overfit control must revisit the identical noisy states; a
            # changing-noise loss curve is not evidence that one example can
            # be fitted.  Formal runs continue to traverse fresh states.
            state_index = (
                micro
                if args.overfit_row is not None
                else global_step * args.micro_records + micro
            )
            seed = legacy.step_seed(args.seed, state_index, row_index)
            record = _build_record(
                row=rows[row_index],
                arm=args.arm,
                transform=transform,
                mean=mean,
                std=std,
                seed=seed,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                action_loss, preservation_loss, metrics = _record_losses(
                    model.get_base_model(), record, args.arm
                )
            action_losses.append(action_loss / float(args.micro_records))
            if preservation_loss is not None:
                fail("action pass unexpectedly retained a preservation graph")
            records.append(record)
            metric_rows.append(metrics)
        action_total = sum(action_losses)
        pcgrad_metrics: dict[str, Any] = {}
        if args.arm == "fullfield_action_noop_pcgrad_preserve":
            action_gradients = torch.autograd.grad(
                action_total, [parameter for _, parameter in named]
            )
            reduced_action, action_norm = _all_reduce_gradient_list(
                named, action_gradients
            )
            del action_gradients
            optimizer.zero_grad(set_to_none=True)
            # The action graph is gone before constructing the independent
            # source-noop graph.  This is still exact PCGrad over two losses;
            # it is not scalar loss mixing and does not weaken the action term.
            preservation_terms = []
            for record in records:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preservation_terms.append(
                        _preservation_loss(model.get_base_model(), record)
                        / float(args.micro_records)
                    )
            preservation_total = sum(preservation_terms)
            preservation_gradients = torch.autograd.grad(
                preservation_total, [parameter for _, parameter in named]
            )
            reduced_preservation, preservation_norm = _all_reduce_gradient_list(
                named, preservation_gradients
            )
            del preservation_gradients
            optimizer.zero_grad(set_to_none=True)
            combined, pcgrad_metrics = objective.project_and_cap_preservation_gradients(
                reduced_action,
                reduced_preservation,
                cap_ratio=PCGRAD_PRESERVATION_CAP,
            )
            _set_gradients(named, combined)
            last_grad_norm = math.sqrt(
                sum(
                    float(value.detach().double().square().sum().item())
                    for value in combined
                )
            )
            pcgrad_metrics.update(
                {
                    "all_reduced_action_norm": action_norm,
                    "all_reduced_preservation_norm": preservation_norm,
                }
            )
            total_for_log = action_total + preservation_total.detach()
        else:
            action_total.backward()
            last_grad_norm = legacy.all_reduce_lora_gradients(named)
            total_for_log = action_total

        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named], args.max_grad_norm
        )
        optimizer.step()
        step = global_step + 1
        # Measure only after AdamW has materialized its real moments.  This is
        # a training-capacity gate, not a pre-forward or dummy-allocation gate.
        if memory is None:
            memory = _memory_receipt(device, args.micro_records)
            if contract.rank == 0:
                print(json.dumps({"memory_gate": memory}, sort_keys=True), flush=True)
            if not bool(memory["passed"]):
                fail(
                    "real training peak reserved memory is not strictly above 50%; "
                    "rerun with a larger true micro-record count"
                )
            if contract.rank == 0:
                output.mkdir(parents=True)
            dist.barrier()
        if step >= 2:
            coverage = _gradient_coverage(named)
            if (
                float(coverage["active_tensor_fraction"]) < 0.95
                or coverage["active_blocks"] != list(range(30))
            ):
                fail("full30 gradient coverage gate failed after two updates")
        averaged = _mean_metrics(metric_rows)
        last_metrics = {
            "total": float(total_for_log.detach().item()),
            **averaged,
        }
        if initial_loss is None:
            initial_loss = last_metrics["total"]
        row_log: dict[str, Any] = {
            "step": step,
            "arm": args.arm,
            "row_indices": row_indices,
            "micro_records": args.micro_records,
            "preclip_grad_norm": last_grad_norm,
            "gradient_coverage": coverage,
            **last_metrics,
        }
        if pcgrad_metrics:
            row_log["pcgrad"] = {
                key: (
                    bool(value)
                    if isinstance(value, bool)
                    else (
                        float(value)
                        if isinstance(value, (int, float))
                        else float(value.detach().item())
                    )
                )
                for key, value in pcgrad_metrics.items()
            }
        if contract.rank == 0:
            print(json.dumps(row_log, sort_keys=True), flush=True)

        if args.overfit_row is not None and step == min(10, args.max_steps):
            assert initial_loss is not None
            if last_metrics["total"] >= 0.80 * initial_loss:
                fail(
                    "overfit positive control failed to reduce its full objective by 20%"
                )
        if step in SAVE_STEPS or step == args.max_steps:
            assert memory is not None
            receipt = checkpoint_receipt(
                args=args,
                manifest=manifest,
                step=step,
                last_metrics=last_metrics,
                grad_norm=last_grad_norm,
                memory=memory,
                coverage=coverage,
                targets=targets,
                initial_digest=initial_digest,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                transformers_version=transformers_version,
                consumed_row_ids=consumed_row_ids,
            )
            data.save_checkpoint(
                model=model,
                optimizer=optimizer,
                output=output,
                step=step,
                receipt=receipt,
                rank=contract.rank,
            )
    dist.barrier()
    if contract.rank == 0:
        (output / "TRAINING_COMPLETE").write_text(
            "training_complete=true\n"
            "frozen_relative_small_update=false\n"
            "memory_gate_passed=true\n",
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
