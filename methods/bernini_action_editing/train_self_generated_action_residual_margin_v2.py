#!/usr/bin/env python3
"""Train frozen-editor-relative action residual margins on source-only RV2V.

The detached T2V action anchor supplies only a post-head unit direction and
amplitude scale.  The trainable quantity is the LoRA-induced change relative
to the frozen RV2V action response at the exact same source/noise/timestep.
No RGB/latent/velocity target from the T2V video reaches the RV2V student.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_learning_v1 as action_core
import self_generated_action_residual_margin_v2 as residual
import train_lora as legacy
import train_self_generated_action_quotient_v1 as parent


METHOD = residual.SCHEMA
REPLICATION_SEED = 20260817


class ResidualTrainingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ResidualTrainingError(message)


def require_replication_seed(seed: int) -> None:
    if type(seed) is not int or seed != REPLICATION_SEED:
        fail(f"initialization seed must be {REPLICATION_SEED}")


def require_sha256(value: str | None, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        fail(f"{label} must be a lowercase SHA-256")
    return value


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_sha(path: Path, expected_sha256: str, label: str) -> str:
    observed = file_sha(path)
    if observed != expected_sha256:
        fail(f"{label} SHA-256 differs")
    return observed


def validate_teacher_cache_seed(cache: Mapping[str, Any], expected_seed: int) -> int:
    seed = cache.get("seed")
    if type(seed) is not int or seed != expected_seed:
        fail("teacher cache seed differs")
    return seed


def validate_teacher_cache_cells(
    cache: Mapping[str, Any], *, slots: int, expected_seed: int,
) -> tuple[list[Mapping[str, Any]], dict[tuple[int, int], Mapping[str, Any]]]:
    cells = cache.get("cells")
    if cache.get("slots") != slots or not isinstance(cells, list):
        fail("teacher cache row x slot grid differs")
    expected_keys = {
        (row_index, slot)
        for row_index in range(4)
        for slot in range(slots)
    }
    if len(cells) != len(expected_keys):
        fail("teacher cache row x slot grid is incomplete")
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            fail("teacher cache cell differs")
        row_index, slot, seed = cell.get("row_index"), cell.get("slot"), cell.get("seed")
        if any(type(value) is not int for value in (row_index, slot, seed)):
            fail("teacher cache row/slot/seed must be numeric")
        key = (row_index, slot)
        if key not in expected_keys or key in by_key:
            fail("teacher cache row x slot key differs")
        if seed != legacy.step_seed(expected_seed, slot, row_index):
            fail("teacher cache cell seed differs")
        by_key[key] = cell
    if set(by_key) != expected_keys:
        fail("teacher cache row x slot grid is incomplete")
    return cells, by_key


def validate_residual_cache_cells(cells: Sequence[Mapping[str, Any]]) -> None:
    for index, cell in enumerate(cells):
        amplitude = cell.get("teacher_amplitude")
        if (
            not isinstance(amplitude, (int, float))
            or isinstance(amplitude, bool)
            or not math.isfinite(float(amplitude))
            or float(amplitude) <= 0
        ):
            fail(f"teacher cache cell {index} has invalid teacher amplitude")


def checkpoint_receipt(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    step: int,
    loss: float,
    grad_norm: float,
    target_modules: Sequence[str],
    trainable_count: int,
    bernini_revision: str,
    veomni_revision: str,
    transformers_version: str,
    initial_digest: str,
    teacher_cache_seed: int,
    teacher_cache_sha256: str,
) -> dict[str, Any]:
    spec = residual.arm_spec(args.arm)
    receipt = {
        "schema_version": legacy.RECEIPT_SCHEMA,
        "global_step": step,
        "max_steps": args.max_steps,
        "last_loss": loss,
        "last_preclip_gradient_norm": grad_norm,
        "bernini_commit": bernini_revision,
        "bernini_training_files_index_sha256": legacy.object_sha256(
            legacy.BERNINI_PINNED_FILE_HASHES
        ),
        "veomni_commit": veomni_revision,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "initialization_seed": args.seed,
        "teacher_cache_seed": teacher_cache_seed,
        "checkpoint_tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        "training_contract": {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": legacy.TASK_SOURCE_NAME,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": 8,
            "lora_alpha": 8,
            "tokenizer_fix_mistral_regex": True,
            "transformers_version": transformers_version,
            "objective": METHOD,
            "arm": args.arm,
            "weights": {
                "margin_scale": spec.margin_scale,
                "perpendicular": spec.perpendicular_weight,
                "onset": spec.onset_weight,
                "onset_frames": spec.onset_frames,
                "nuisance": spec.nuisance_weight,
                "noop": spec.noop_weight,
            },
            "optimized_quantity": "Psi(v_lora_action)-Psi(v_frozen_action)",
            "action_constraint": "positive_teacher_direction_gain_margin",
            "orthogonal_adapter_change_penalized": spec.perpendicular_weight > 0,
            "rv2v_supervision_target": "source_video_only",
            "self_generated_anchor_role": "detached_post_head_action_phase_code_only",
            "historical_selected_target_reachable": False,
        },
        "source_manifest_digest": manifest["manifest_digest"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "teacher_cache_sha256": teacher_cache_sha256,
        "optimizer": {
            "type": "AdamW",
            "learning_rate": spec.learning_rate,
            "weight_decay": 0.0,
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "same_sample_all_ranks": True,
            "same_seed_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
            "lora_initialization_digest": initial_digest,
        },
        "target_module_count": len(target_modules),
        "target_modules_sha256": legacy.object_sha256(list(target_modules)),
        "trainable_parameter_count": trainable_count,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "experimental_training": True,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--source-manifest", required=True)
    value.add_argument("--source-manifest-sha256", required=True)
    value.add_argument("--cache", required=True)
    value.add_argument("--expected-cache-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--arm", choices=residual.ARM_NAMES, required=True)
    value.add_argument("--slots", type=int, default=4)
    value.add_argument("--max-steps", type=int, default=160)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive-sha256", required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    require_replication_seed(args.seed)
    if args.slots <= 0 or args.max_steps <= 0:
        fail("slots/max_steps must be positive")
    require_sha256(args.source_manifest_sha256, "source manifest SHA-256")
    require_sha256(args.expected_cache_sha256, "expected teacher cache SHA-256")
    cache_path = Path(args.cache).resolve(strict=True)
    teacher_cache_sha256 = validate_file_sha(
        cache_path, args.expected_cache_sha256, "teacher cache"
    )
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
    from bernini.training.data import NoiseScheduler
    from bernini.parallel import init_parallel_state

    contract = legacy.distributed_contract()
    if contract.world_size != 4:
        fail("residual-margin training requires SP4")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    source_manifest_path = Path(args.source_manifest).resolve(strict=True)
    validate_file_sha(
        source_manifest_path, args.source_manifest_sha256, "source manifest"
    )
    manifest, rows = parent.load_manifest(source_manifest_path)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    with parent.serialized_model_load():
        base = BerniniRendererModel(config)
        base.requires_grad_(False)
        base.t5_text_encoder.eval()
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        targets = legacy.select_attention_projection_names(base)
        model = get_peft_model(
            base,
            LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
        )
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
    named = legacy.trainable_lora_parameters(model)
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
    transform = parent.build_transform(
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
    )

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if (
        cache.get("schema_version") != parent.CACHE_SCHEMA
        or cache.get("manifest_digest") != manifest["manifest_digest"]
    ):
        fail("teacher cache identity differs")
    teacher_cache_seed = validate_teacher_cache_seed(cache, args.seed)
    cells, by_key = validate_teacher_cache_cells(
        cache, slots=args.slots, expected_seed=args.seed
    )
    validate_residual_cache_cells(cells)

    output = Path(args.output).resolve()
    if contract.rank == 0:
        if output.exists() or output.is_symlink():
            fail(f"training output exists: {output}")
        output.mkdir(parents=True)
    dist.barrier()

    spec = residual.arm_spec(args.arm)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named], lr=spec.learning_rate
    )
    model.train()
    model.get_base_model().t5_text_encoder.eval()
    last_loss = last_grad = 0.0
    for global_step in range(args.max_steps):
        row_index = global_step % 4
        slot = (global_step // 4) % args.slots
        row = rows[row_index]
        cell = by_key[(row_index, slot)]
        source_blob = Path(row["source_posterior"]["path"]).read_bytes()
        source_clean_cpu = parent.source_clean_from_posterior(source_blob, mean, std)
        shape = tuple(int(item) for item in source_clean_cpu.shape)
        action_batch = transform(
            parent.make_sample(
                instruction=row["instruction"],
                source_blob=source_blob,
                target_blob=source_blob,
            ),
            int(cell["seed"]),
        )
        x_sigma, _, sigma, state_digest = parent.paired_state(
            action_batch, source_clean_cpu
        )
        if (
            state_digest != cell["source_state_digest"]
            or abs(sigma - float(cell["sigma"])) > 1.0e-6
        ):
            fail("trainable source state differs from detached cache")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # The frozen reference is evaluated on the exact same packed
            # source/noise/timestep.  Computing it online costs one additional
            # forward but avoids changing or weakening the sealed V1 cache.
            with torch.no_grad(), model.disable_adapter():
                frozen_velocity = parent.predicted_target_velocity(
                    model, action_batch, spatial_shape=shape
                )
            velocity = parent.predicted_target_velocity(
                model, action_batch, spatial_shape=shape
            )
            source_noop_raw = cell["source_noop_raw"].to(device)
            student_raw = action_core.psiout_raw_v1(velocity) - source_noop_raw
            frozen_raw = (
                action_core.psiout_raw_v1(frozen_velocity) - source_noop_raw
            ).detach()
            teacher_unit = cell["teacher_unit"].to(device)
            teacher_amplitude = torch.tensor(
                [float(cell["teacher_amplitude"])],
                device=device,
                dtype=torch.float32,
            )
            margin = residual.residual_margin_loss(
                student_raw=student_raw,
                frozen_raw=frozen_raw,
                detached_teacher_unit=teacher_unit,
                detached_teacher_amplitude=teacher_amplitude,
                margin_scale=spec.margin_scale,
            )
            predicted_clean = x_sigma - float(sigma) * velocity
            onset_loss = residual.onset_preservation_loss(
                predicted_clean=predicted_clean,
                source_clean=source_clean_cpu.to(device),
                onset_frames=spec.onset_frames,
            )
            delta_raw = student_raw - frozen_raw
            nuisance_loss = parent.quotient.nuisance_coefficient_loss(
                delta_raw,
                cell["camera_unit"].to(device),
                cell["appearance_unit"].to(device),
            )
            noop_loss = torch.zeros((), device=device, dtype=torch.float32)
            if spec.noop_weight > 0:
                noop_batch = transform(
                    parent.make_sample(
                        instruction=parent.NOOP_INSTRUCTION,
                        source_blob=source_blob,
                        target_blob=source_blob,
                    ),
                    int(cell["seed"]),
                )
                noop_loss = model(**noop_batch, use_cache=False).diff_loss.float().mean()
            total = residual.weighted_total(
                spec=spec,
                action=margin.action.float(),
                perpendicular=margin.perpendicular.float(),
                onset=onset_loss.float(),
                nuisance=nuisance_loss.float(),
                noop=noop_loss.float(),
            )
        total.backward()
        last_grad = legacy.all_reduce_lora_gradients(named)
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named], 1.0)
        optimizer.step()
        step = global_step + 1
        last_loss = float(total.detach().item())
        if contract.rank == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "arm": args.arm,
                        "iid": row["iid"],
                        "slot": slot,
                        "total": last_loss,
                        "action_margin": float(margin.action.detach().item()),
                        "gain": float(margin.gain_mean.detach().item()),
                        "required_gain": float(margin.margin_mean.detach().item()),
                        "perpendicular": float(margin.perpendicular.detach().item()),
                        "delta_norm": float(margin.delta_norm_mean.detach().item()),
                        "onset": float(onset_loss.detach().item()),
                        "nuisance": float(nuisance_loss.detach().item()),
                        "noop": float(noop_loss.detach().item()),
                        "sigma": sigma,
                        "preclip_grad_norm": last_grad,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step in parent.SAVE_STEPS or step == args.max_steps:
            receipt = checkpoint_receipt(
                args=args,
                manifest=manifest,
                step=step,
                loss=last_loss,
                grad_norm=last_grad,
                target_modules=targets,
                trainable_count=sum(int(parameter.numel()) for _, parameter in named),
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                transformers_version=transformers_version,
                initial_digest=initial_digest,
                teacher_cache_seed=teacher_cache_seed,
                teacher_cache_sha256=teacher_cache_sha256,
            )
            parent.save_checkpoint(
                model=model,
                optimizer=optimizer,
                output=output,
                step=step,
                receipt=receipt,
                rank=contract.rank,
            )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
