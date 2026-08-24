#!/usr/bin/env python3
"""Train endpoint-consensus action reward arms on source-state RV2V.

The self-generated pure-T2V video is never an RGB, latent, or velocity target.
It supplies a detached endpoint action direction only.  The trainable state is
the source video, and every adapter response is measured against an online
frozen RV2V forward at the identical source/noise/sigma/timestep.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_learning_v1 as action_core
import self_generated_action_endpoint_consensus_v3 as endpoint
import train_lora as legacy
import train_self_generated_action_quotient_v1 as parent
import train_self_generated_action_residual_margin_v2 as v2


METHOD = endpoint.SCHEMA
REPLICATION_SEED = v2.REPLICATION_SEED


class EndpointTrainingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise EndpointTrainingError(message)


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
    authority: Mapping[tuple[int, int], endpoint.EndpointAuthority],
) -> dict[str, Any]:
    # Reuse the already inference-compatible receipt envelope, then replace
    # every objective-specific field and recompute its self digest.
    shadow = argparse.Namespace(**vars(args))
    shadow.arm = "margin_010_perp_100"
    receipt = v2.checkpoint_receipt(
        args=shadow,
        manifest=manifest,
        step=step,
        loss=loss,
        grad_norm=grad_norm,
        target_modules=target_modules,
        trainable_count=trainable_count,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        transformers_version=transformers_version,
        initial_digest=initial_digest,
        teacher_cache_seed=teacher_cache_seed,
        teacher_cache_sha256=teacher_cache_sha256,
    )
    spec = endpoint.arm_spec(args.arm)
    family_amplitudes = {
        "rows_0_1": authority[(0, 0)].robust_amplitude,
        "rows_2_3": authority[(2, 0)].robust_amplitude,
    }
    peer_cosines = [item.peer_consensus_cosine for item in authority.values()]
    contract = receipt["training_contract"]
    contract.update(
        {
            "objective": METHOD,
            "arm": args.arm,
            "weights": {
                "endpoint_lower_scale": spec.lower_scale,
                "endpoint_upper_scale": spec.upper_scale,
                "endpoint_perpendicular": spec.endpoint_perpendicular_weight,
                "full_functional_trust": spec.full_trust_weight,
                "nuisance": spec.nuisance_weight,
            },
            "optimized_quantity": (
                "endpoint(Psi(v_lora_action)-Psi(v_frozen_action))"
            ),
            "action_constraint": "two_sided_endpoint_gain_band",
            "teacher_representation": {
                "type": "late3_mean_minus_early3_mean",
                "mode": spec.teacher_mode,
                "family_consensus_rows": [[0, 1], [2, 3]],
                "amplitude": "per_family_median_endpoint_amplitude",
                "held_anchor_positive_admission_required": True,
            },
            "full_post_head_velocity_trust": spec.full_trust_weight > 0,
            "rv2v_supervision_target": "source_video_only",
            "self_generated_anchor_role": "detached_endpoint_action_code_only",
            "historical_selected_target_reachable": False,
        }
    )
    receipt["optimizer"]["learning_rate"] = spec.learning_rate
    receipt["endpoint_authority"] = {
        "cell_count": len(authority),
        "family_robust_amplitudes": family_amplitudes,
        "peer_consensus_cosine_min": min(peer_cosines),
        "peer_consensus_cosine_mean": sum(peer_cosines) / len(peer_cosines),
    }
    receipt.pop("receipt_digest", None)
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
    value.add_argument("--arm", choices=endpoint.ARM_NAMES, required=True)
    value.add_argument("--slots", type=int, default=4)
    value.add_argument("--max-steps", type=int, default=80)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--method-source-revision", required=True)
    value.add_argument("--method-source-archive-sha256", required=True)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    v2.require_replication_seed(args.seed)
    if args.slots != 4 or args.max_steps <= 0:
        fail("endpoint-consensus training requires four slots and positive max steps")
    v2.require_sha256(args.source_manifest_sha256, "source manifest SHA-256")
    v2.require_sha256(args.expected_cache_sha256, "expected teacher cache SHA-256")
    cache_path = Path(args.cache).resolve(strict=True)
    teacher_cache_sha256 = v2.validate_file_sha(
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
        fail("endpoint-consensus training requires SP4")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)
    source_manifest_path = Path(args.source_manifest).resolve(strict=True)
    v2.validate_file_sha(
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
    teacher_cache_seed = v2.validate_teacher_cache_seed(cache, args.seed)
    cells, by_key = v2.validate_teacher_cache_cells(
        cache, slots=args.slots, expected_seed=args.seed
    )
    v2.validate_residual_cache_cells(cells)
    authority = endpoint.build_endpoint_authority(cells)

    output = Path(args.output).resolve()
    if contract.rank == 0:
        if output.exists() or output.is_symlink():
            fail(f"training output exists: {output}")
        output.mkdir(parents=True)
    dist.barrier()

    spec = endpoint.arm_spec(args.arm)
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
        endpoint_teacher = authority[(row_index, slot)]
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
        _, _, sigma, state_digest = parent.paired_state(action_batch, source_clean_cpu)
        if (
            state_digest != cell["source_state_digest"]
            or abs(sigma - float(cell["sigma"])) > 1.0e-6
        ):
            fail("trainable source state differs from detached cache")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            teacher_unit_cpu = (
                endpoint_teacher.cell_unit
                if spec.teacher_mode == "cell"
                else endpoint_teacher.consensus_unit
            )
            teacher_unit = teacher_unit_cpu.to(device)
            teacher_amplitude = torch.tensor(
                [endpoint_teacher.robust_amplitude], device=device, dtype=torch.float32
            )
            band = endpoint.endpoint_band_loss(
                student_raw=student_raw,
                frozen_raw=frozen_raw,
                detached_teacher_unit=teacher_unit,
                detached_teacher_amplitude=teacher_amplitude,
                lower_scale=spec.lower_scale,
                upper_scale=spec.upper_scale,
            )
            full_trust = endpoint.full_functional_trust(
                student_velocity=velocity, frozen_velocity=frozen_velocity
            )
            delta_raw = student_raw - frozen_raw
            nuisance_loss = parent.quotient.nuisance_coefficient_loss(
                delta_raw,
                cell["camera_unit"].to(device),
                cell["appearance_unit"].to(device),
            )
            total = endpoint.weighted_total(
                spec=spec,
                action=band.action.float(),
                perpendicular=band.perpendicular.float(),
                full_trust=full_trust.float(),
                nuisance=nuisance_loss.float(),
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
                        "teacher_mode": spec.teacher_mode,
                        "iid": row["iid"],
                        "slot": slot,
                        "total": last_loss,
                        "action_band": float(band.action.detach().item()),
                        "gain": float(band.gain_mean.detach().item()),
                        "lower_gain": float(band.lower_mean.detach().item()),
                        "upper_gain": float(band.upper_mean.detach().item()),
                        "endpoint_perpendicular": float(band.perpendicular.detach().item()),
                        "endpoint_delta_norm": float(band.delta_norm_mean.detach().item()),
                        "full_trust": float(full_trust.detach().item()),
                        "nuisance": float(nuisance_loss.detach().item()),
                        "peer_consensus_cosine": endpoint_teacher.peer_consensus_cosine,
                        "cell_endpoint_amplitude": endpoint_teacher.cell_amplitude,
                        "robust_endpoint_amplitude": endpoint_teacher.robust_amplitude,
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
                authority=authority,
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
