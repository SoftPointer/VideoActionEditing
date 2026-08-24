#!/usr/bin/env python3
"""Score exact81 candidates by same-video temporal counterfactuals.

One SP4 invocation handles one sealed core4-v2 group.  For every candidate it
loads the candidate's own native predecode clean latent and its cell's exact
official Gaussian, constructs the seven preregistered temporal arms, and runs
frozen Bernini under exactly two prompts: the cell-fixed target-action caption
and the cell-fixed, scene-matched no-op caption.  Both prompts share one patched ``x_sigma`` object
and one native sigma/timestep object at each of three schedule coordinates.

This is a real GPU integration boundary, not a placeholder.  It directly uses
Bernini ``patch_vae_latent`` and ``diffusion.shared_step`` at the same insertion
point as the frozen d541801 scorer.  The d541801 scorer source is hash-pinned
and reused only for bank/checkpoint/prompt provenance helpers.  No T2V tensor,
media, or proposal is authorized as an RV2V condition, target, donor, or noise.
This executable performs no training and never authorizes an optimizer.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import temporal_counterfactual_contract_v1 as contract  # noqa: E402


SCORE_FILENAME = "temporal-counterfactual-action-score-v1.json"
GROUP_FILENAME = "temporal-counterfactual-action-score-{group_id}-v1.json"
PROMPT_ORDER = ("target_action", "noop")
MODEL_FORWARDS_PER_CANDIDATE = (
    len(contract.TRANSFORM_ORDER) * len(contract.NATIVE_SIGMA_COORDINATES) * len(PROMPT_ORDER)
)
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class TemporalCounterfactualScoringError(RuntimeError):
    """The frozen model, tensor, prompt, schedule, or output closure differs."""


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TemporalCounterfactualScoringError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise TemporalCounterfactualScoringError(f"{label} must be lowercase SHA-1")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TemporalCounterfactualScoringError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise TemporalCounterfactualScoringError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _frozen_d541801_runtime() -> Any:
    """Load only the formal v3 scorer bytes that produced job 131177."""

    source = _plain_file(
        METHOD_ROOT / "score_pair_v5_t2v_energy_bank_frozen_d541801.py",
        label="d541801 frozen scorer source",
    )
    if contract.file_sha256(source) != contract.REQUIRED_D541801_SCORER_SHA256:
        raise TemporalCounterfactualScoringError(
            "d541801 frozen scorer source SHA-256 differs; later v4 code is forbidden"
        )
    try:
        runtime = importlib.import_module(
            "score_pair_v5_t2v_energy_bank_frozen_d541801"
        )
    except Exception as error:
        raise TemporalCounterfactualScoringError(
            "d541801 frozen scorer runtime is unavailable"
        ) from error
    if (
        getattr(runtime, "SCORE_RECEIPT_SCHEMA", None)
        != "bernini-pair-v5-frozen-t2v-global-energy-score-v3"
        or getattr(runtime, "GROUP_RECEIPT_SCHEMA", None)
        != "bernini-pair-v5-frozen-t2v-global-energy-group-v3"
    ):
        raise TemporalCounterfactualScoringError(
            "loaded scorer is not the formal d541801/v3 authority"
        )
    return runtime


def validate_native_coordinate_runtime(frozen: Any) -> None:
    native = getattr(frozen, "native_schedule", None)
    if (
        native is None
        or getattr(native, "PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST", None)
        != contract.NATIVE_SCHEDULE_DIGEST
    ):
        raise TemporalCounterfactualScoringError("native schedule digest differs")
    for index, sigma, timestep in contract.NATIVE_SIGMA_COORDINATES:
        if (
            float(native.NATIVE_UNIPC40_SIGMAS[index]).hex() != float(sigma).hex()
            or int(native.NATIVE_UNIPC40_TIMESTEPS[index]) != timestep
        ):
            raise TemporalCounterfactualScoringError(
                f"native schedule coordinate {index} differs"
            )


def _encode_prompt_pair(
    renderer: Any,
    tokenizer: Any,
    *,
    action_prompt: str,
    noop_prompt: str,
    device: Any,
    frozen: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    import torch

    legacy = frozen.native_generation.legacy
    conditions: dict[str, Any] = {}
    prompts = {"target_action": action_prompt, "noop": noop_prompt}
    for name in PROMPT_ORDER:
        ids, mask = legacy._tokenize_training_prompt(tokenizer, prompts[name])
        with torch.inference_mode():
            condition = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        if (
            tuple(int(item) for item in condition.shape) != (1, 512, 4096)
            or condition.device != device
            or condition.requires_grad
            or condition.grad_fn is not None
            or not bool(torch.isfinite(condition).all().item())
        ):
            raise TemporalCounterfactualScoringError(
                f"{name} prompt condition closure differs"
            )
        conditions[name] = condition
    hashes = {
        name: frozen.tensor_sha256(value.float()) for name, value in conditions.items()
    }
    if (
        action_prompt == noop_prompt
        or torch.equal(conditions["target_action"], conditions["noop"])
        or len(set(hashes.values())) != 2
    ):
        raise TemporalCounterfactualScoringError("action/no-op prompt pair aliases")
    return conditions, hashes


def forward_native_prompt_pair(
    *,
    diffusion: Any,
    transformer: Any,
    x_sigma: Any,
    native_schedule_index: int,
    action_condition: Any,
    noop_condition: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run the real Bernini target-only pair on one shared patched state."""

    import torch
    import dclr_runtime_contract as runtime_contract
    import pair_v5_native_bridge as native_bridge

    try:
        native_bridge._validate_exact81_spatial(
            x_sigma, label="temporal-counterfactual x_sigma", detached_fp32=True
        )
    except native_bridge.PairV5NativeBridgeError as error:
        raise TemporalCounterfactualScoringError(str(error)) from error
    coordinate = next(
        (row for row in contract.NATIVE_SIGMA_COORDINATES if row[0] == native_schedule_index),
        None,
    )
    if coordinate is None:
        raise TemporalCounterfactualScoringError("model call sigma is not preregistered")
    if (
        not callable(getattr(diffusion, "shared_step", None))
        or not callable(getattr(transformer, "patch_vae_latent", None))
        or any(parameter.requires_grad for parameter in diffusion.parameters())
        or any(parameter.requires_grad for parameter in transformer.parameters())
    ):
        raise TemporalCounterfactualScoringError("Bernini scorer is not frozen")
    dtype = getattr(transformer, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TemporalCounterfactualScoringError("transformer dtype differs")
    expected_condition_shape = (1, 512, 4096)
    for name, condition in (
        ("action", action_condition),
        ("noop", noop_condition),
    ):
        if (
            not isinstance(condition, torch.Tensor)
            or tuple(int(item) for item in condition.shape) != expected_condition_shape
            or condition.device != x_sigma.device
            or condition.requires_grad
            or condition.grad_fn is not None
            or not bool(torch.isfinite(condition).all().item())
        ):
            raise TemporalCounterfactualScoringError(
                f"{name} condition is not frozen device-local [1,512,4096]"
            )
    if torch.equal(action_condition, noop_condition):
        raise TemporalCounterfactualScoringError("action/no-op conditions alias")

    with torch.inference_mode():
        patched = transformer.patch_vae_latent(
            x_sigma.to(dtype=dtype), source_id=native_bridge.T2V_TARGET_SOURCE_ID
        )
    if not isinstance(patched, (tuple, list)) or len(patched) != 2:
        raise TemporalCounterfactualScoringError("patch_vae_latent output differs")
    try:
        branch = runtime_contract.build_t2v_target_branch(
            patched[0],
            patched[1],
            target_source_id=native_bridge.T2V_TARGET_SOURCE_ID,
        )
    except runtime_contract.DCLRRuntimeContractError as error:
        raise TemporalCounterfactualScoringError(str(error)) from error
    timestep = torch.tensor(
        [float(coordinate[2])], dtype=torch.float32, device=x_sigma.device
    )
    expected_tokens = int(x_sigma.shape[2]) * (int(x_sigma.shape[3]) // 2) * (
        int(x_sigma.shape[4]) // 2
    )
    if branch.target_token_count != expected_tokens:
        raise TemporalCounterfactualScoringError("patched target geometry differs")

    tracked = {
        "noisy_latents": branch.noisy_latents,
        "rotary_embs": branch.rotary_embs,
        "native_timestep": timestep,
    }
    object_ids = {name: id(value) for name, value in tracked.items()}

    def snapshot() -> dict[str, str]:
        return {
            name: native_bridge._tensor_sha256(value)
            for name, value in tracked.items()
        }

    hashes_by_stage = {"before_action": snapshot()}
    results = []
    for prompt_name, condition in zip(PROMPT_ORDER, (action_condition, noop_condition)):
        with torch.inference_mode():
            prediction = diffusion.shared_step(
                model_id="transformer_1",
                noisy_latents=branch.noisy_latents,
                timesteps=timestep,
                cond_embeds=condition,
                rotary_embs=branch.rotary_embs,
                batch_vae_seqlen=list(branch.batch_vae_seqlen),
                batch_text_seqlen=[runtime_contract.PINNED_TEXT_TOKENS],
            )
        total = branch.total_token_count
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(int(item) for item in prediction.shape)
            != (1, total, runtime_contract.PINNED_PATCH_DIM)
            or prediction.device != x_sigma.device
            or prediction.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or prediction.requires_grad
            or prediction.grad_fn is not None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise TemporalCounterfactualScoringError(
                "frozen shared_step velocity closure differs"
            )
        packed_velocity = prediction[:, -branch.target_token_count :, :]
        try:
            spatial = native_bridge._unpack_spatial_velocity(
                packed_velocity, video_shape=tuple(int(item) for item in x_sigma.shape)
            )
        except native_bridge.PairV5NativeBridgeError as error:
            raise TemporalCounterfactualScoringError(str(error)) from error
        results.append(spatial.detach())
        stage = "after_action" if prompt_name == "target_action" else "after_noop"
        hashes_by_stage[stage] = snapshot()
        if any(
            id(tracked[name]) != object_ids[name]
            for name in tracked
        ):
            raise TemporalCounterfactualScoringError(
                "shared state/timestep/rotary object identity changed"
            )
    if any(
        len({hashes_by_stage[stage][name] for stage in hashes_by_stage}) != 1
        for name in tracked
    ):
        raise TemporalCounterfactualScoringError(
            "shared state/timestep/rotary tensor bytes changed across prompt pair"
        )
    proof = {
        "noisy_latents_sha256_by_stage": {
            stage: values["noisy_latents"] for stage, values in hashes_by_stage.items()
        },
        "rotary_embs_sha256_by_stage": {
            stage: values["rotary_embs"] for stage, values in hashes_by_stage.items()
        },
        "native_timestep_sha256_by_stage": {
            stage: values["native_timestep"] for stage, values in hashes_by_stage.items()
        },
        "same_noisy_latents_object_reused": True,
        "same_rotary_embs_object_reused": True,
        "same_native_timestep_object_reused": True,
        "post_call_tensor_bytes_unchanged": True,
    }
    return results[0], results[1], proof


def _energy(value: Any, target: Any) -> float:
    import torch

    result = (value.float() - target).square().flatten(start_dim=1).mean(dim=1)
    if (
        tuple(result.shape) != (1,)
        or result.dtype != torch.float32
        or not bool(torch.isfinite(result).all().item())
        or bool((result < 0.0).any().item())
    ):
        raise TemporalCounterfactualScoringError("denoising energy differs")
    scalar = float(result.item())
    if not math.isfinite(scalar) or scalar < 0.0:
        raise TemporalCounterfactualScoringError("denoising energy scalar differs")
    return scalar


def _prompt_binding(
    *,
    target_action_caption_sha256: str,
    target_noop_caption_sha256: str,
    action_prompt: str,
    noop_prompt: str,
    condition_hashes: Mapping[str, str],
    prompt_builder_contract_digest: str,
) -> dict[str, Any]:
    action_prompt_sha = hashlib.sha256(action_prompt.encode("utf-8")).hexdigest()
    noop_prompt_sha = hashlib.sha256(noop_prompt.encode("utf-8")).hexdigest()
    value = {
        "action_raw_caption_utf8_sha256": target_action_caption_sha256,
        "noop_raw_caption_utf8_sha256": target_noop_caption_sha256,
        "action_full_prompt_utf8_sha256": action_prompt_sha,
        "noop_full_prompt_utf8_sha256": noop_prompt_sha,
        "action_condition_tensor_sha256": condition_hashes["target_action"],
        "noop_condition_tensor_sha256": condition_hashes["noop"],
        "prompt_builder_contract_digest": prompt_builder_contract_digest,
    }
    value["prompt_pair_digest"] = contract.object_sha256(
        {
            "action_full_prompt_utf8_sha256": action_prompt_sha,
            "noop_full_prompt_utf8_sha256": noop_prompt_sha,
            "action_condition_tensor_sha256": condition_hashes["target_action"],
            "noop_condition_tensor_sha256": condition_hashes["noop"],
        }
    )
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TemporalCounterfactualScoringError(f"refusing to overwrite {path}")
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")
    os.chmod(path, 0o400)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--expected-scorer-source-sha256", required=True)
    parser.add_argument("--expected-contract-source-sha256", required=True)
    parser.add_argument("--ack-t2v-calibration-only-never-rv2v-input", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_root_spec_sha256",
        "expected_bank_receipt_sha256",
        "method_source_archive_sha256",
        "expected_scorer_source_sha256",
        "expected_contract_source_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        _sha1(getattr(args, name), label=name)
    if (
        args.expected_bernini_commit != contract.REQUIRED_BERNINI_REVISION
        or args.expected_veomni_commit != contract.REQUIRED_VEOMNI_REVISION
    ):
        raise TemporalCounterfactualScoringError(
            "Bernini/VeOmni revisions differ from the frozen authority"
        )
    if args.ack_t2v_calibration_only_never_rv2v_input is not True:
        raise TemporalCounterfactualScoringError(
            "T2V calibration-only acknowledgement is mandatory"
        )
    if contract.file_sha256(Path(__file__).resolve()) != args.expected_scorer_source_sha256:
        raise TemporalCounterfactualScoringError("temporal scorer source hash differs")
    if (
        contract.file_sha256(METHOD_ROOT / "temporal_counterfactual_contract_v1.py")
        != args.expected_contract_source_sha256
    ):
        raise TemporalCounterfactualScoringError("temporal contract source hash differs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    frozen = _frozen_d541801_runtime()
    validate_native_coordinate_runtime(frozen)
    try:
        spec, bank, bound_rows = frozen.load_group_bank(
            root_spec=args.root_spec,
            root_spec_sha256=args.expected_root_spec_sha256,
            bank_output_dir=args.bank_output_dir,
            bank_receipt=args.bank_receipt,
            bank_receipt_sha256=args.expected_bank_receipt_sha256,
            group_id=args.group_id,
        )
    except frozen.PairV5T2VEnergyScoringError as error:
        raise TemporalCounterfactualScoringError(str(error)) from error
    del spec
    output = Path(args.output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise TemporalCounterfactualScoringError(
            "output must be a fresh absolute directory under a plain parent"
        )

    native_generation = frozen.native_generation
    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise TemporalCounterfactualScoringError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise TemporalCounterfactualScoringError("pinned Bernini heads differ")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise TemporalCounterfactualScoringError("scorer requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            identity = native_generation.source_audit.validate_checkpoint_content(
                checkpoint, Path(args.checkpoint_content_manifest)
            )
            checkpoint_rows[0] = {"ok": True, "identity": identity}
        except Exception as error:
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise TemporalCounterfactualScoringError(
            f"rank-zero checkpoint audit failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])
    checkpoint_receipt_digest = frozen.object_sha256(checkpoint_identity)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    try:
        freeze_certificate = native_generation.source_audit.model_freeze_certificate(
            renderer
        )
        checkpoint_binding = frozen.checkpoint_content_binding(
            checkpoint_identity, freeze_certificate
        )
    except Exception as error:
        raise TemporalCounterfactualScoringError(str(error)) from error
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if (
        transformer is None
        or diffusion.transformer_2 is not None
        or any(parameter.requires_grad for parameter in renderer.parameters())
    ):
        raise TemporalCounterfactualScoringError("frozen transformer_1 closure differs")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    builder_contract = frozen.prompt_builder_contract()

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in bound_rows:
        by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
    if distributed.rank == 0:
        output.mkdir()
    dist.barrier()
    receipts: list[dict[str, Any]] = []
    model_binding = {
        "frozen_checkpoint_receipt_digest": checkpoint_receipt_digest,
        "checkpoint_content_manifest_sha256": checkpoint_binding["manifest_sha256"],
        "checkpoint_content_binding_digest": checkpoint_binding["binding_digest"],
        "d541801_scorer_source_revision": contract.REQUIRED_D541801_SCORER_REVISION,
        "d541801_scorer_source_sha256": contract.REQUIRED_D541801_SCORER_SHA256,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "native_schedule_digest": contract.NATIVE_SCHEDULE_DIGEST,
    }
    for cell_id, cell_rows in by_cell.items():
        if [row["candidate"]["semantic_branch"] for row in cell_rows] != list(
            contract.BRANCH_ORDER
        ):
            raise TemporalCounterfactualScoringError("cell branch order differs")
        target_rows = [
            row
            for row in cell_rows
            if row["candidate"]["semantic_branch"] == contract.ACTION_BRANCH
        ]
        if len(target_rows) != 1:
            raise TemporalCounterfactualScoringError("cell target-action row differs")
        noop_rows = [
            row
            for row in cell_rows
            if row["candidate"]["semantic_branch"] == "noop"
        ]
        if len(noop_rows) != 1:
            raise TemporalCounterfactualScoringError("cell no-op row differs")
        target_row = target_rows[0]
        target_candidate = target_row["candidate"]
        noop_candidate = noop_rows[0]["candidate"]
        action_prompt = native_generation.build_task_prompt(
            "t2v", target_candidate["full_t2v_caption"], prompt_cleaner=prompt_clean
        )
        noop_prompt = native_generation.build_task_prompt(
            "t2v", noop_candidate["full_t2v_caption"], prompt_cleaner=prompt_clean
        )
        conditions, condition_hashes = _encode_prompt_pair(
            renderer,
            tokenizer,
            action_prompt=action_prompt,
            noop_prompt=noop_prompt,
            device=device,
            frozen=frozen,
        )
        prompt_binding = _prompt_binding(
            target_action_caption_sha256=target_candidate[
                "full_t2v_caption_utf8_sha256"
            ],
            target_noop_caption_sha256=noop_candidate[
                "full_t2v_caption_utf8_sha256"
            ],
            action_prompt=action_prompt,
            noop_prompt=noop_prompt,
            condition_hashes=condition_hashes,
            prompt_builder_contract_digest=builder_contract["contract_digest"],
        )
        target_binding = {
            "target_action_candidate_id": target_candidate["candidate_id"],
            "target_noop_candidate_id": noop_candidate["candidate_id"],
            "calibration_group_id": cell_id,
            "target_action_caption_utf8_sha256": target_candidate[
                "full_t2v_caption_utf8_sha256"
            ],
            "target_noop_caption_utf8_sha256": noop_candidate[
                "full_t2v_caption_utf8_sha256"
            ],
        }

        first_gaussian = frozen._load_exact81_tensor(
            cell_rows[0]["artifacts"]["official_initial_gaussian"],
            key="official_initial_gaussian",
            label=f"{cell_id} official Gaussian",
        )
        first_identity = frozen.verify_native_tensor_value_identity(
            first_gaussian,
            cell_rows[0]["artifacts"]["official_initial_gaussian"],
            label=f"{cell_id} official Gaussian",
        )
        epsilon = first_gaussian.to(device=device).contiguous()
        epsilon_tensor_sha = frozen.tensor_sha256(epsilon)
        for row_index, row in enumerate(cell_rows):
            gaussian_artifact = row["artifacts"]["official_initial_gaussian"]
            candidate_gaussian = (
                first_gaussian
                if row_index == 0
                else frozen._load_exact81_tensor(
                    gaussian_artifact,
                    key="official_initial_gaussian",
                    label=f"{row['candidate']['candidate_id']} official Gaussian",
                )
            )
            observed_identity = frozen.verify_native_tensor_value_identity(
                candidate_gaussian,
                gaussian_artifact,
                label=f"{row['candidate']['candidate_id']} official Gaussian",
            )
            if observed_identity != first_identity or not torch.equal(
                candidate_gaussian, first_gaussian
            ):
                raise TemporalCounterfactualScoringError(
                    "same-cell official Gaussian tensor values differ"
                )
            clean_cpu = frozen._load_exact81_tensor(
                row["artifacts"]["predecode_clean_latent"],
                key="normalized_clean_latent",
                label=f"{row['candidate']['candidate_id']} clean latent",
            )
            clean = clean_cpu.to(device=device).contiguous()
            if clean.shape != epsilon.shape:
                raise TemporalCounterfactualScoringError(
                    "candidate/Gaussian geometry differs"
                )
            candidate_id = row["candidate"]["candidate_id"]
            energy_grid: dict[str, list[dict[str, Any]]] = {}
            for transform_name in contract.TRANSFORM_ORDER:
                transformed = contract.apply_temporal_transform_tensor(
                    clean, transform_name
                )
                transformed_sha = frozen.tensor_sha256(transformed)
                effective_epsilon = contract.fixed_official_gaussian_tensor(
                    epsilon, transform_name
                )
                effective_epsilon_sha = frozen.tensor_sha256(effective_epsilon)
                target = (
                    effective_epsilon - transformed
                ).float().contiguous().detach()
                target_sha = frozen.tensor_sha256(target)
                energy_rows = []
                for schedule_index, sigma_value, _timestep in contract.NATIVE_SIGMA_COORDINATES:
                    sigma = torch.tensor(
                        [sigma_value], dtype=torch.float32, device=device
                    )
                    x_sigma = (
                        transformed + sigma.reshape(1, 1, 1, 1, 1) * target
                    ).float().contiguous().detach()
                    action_velocity, noop_velocity, same_state_proof = forward_native_prompt_pair(
                        diffusion=diffusion,
                        transformer=transformer,
                        x_sigma=x_sigma,
                        native_schedule_index=schedule_index,
                        action_condition=conditions["target_action"],
                        noop_condition=conditions["noop"],
                    )
                    pair_receipt = contract.make_prompt_pair_receipt(
                        candidate_id=candidate_id,
                        transform_name=transform_name,
                        native_schedule_index=schedule_index,
                        transformed_clean_tensor_sha256=transformed_sha,
                        official_gaussian_tensor_sha256=epsilon_tensor_sha,
                        effective_gaussian_tensor_sha256=effective_epsilon_sha,
                        x_sigma_tensor_sha256=frozen.tensor_sha256(x_sigma),
                        velocity_target_tensor_sha256=target_sha,
                        action_velocity_tensor_sha256=frozen.tensor_sha256(
                            action_velocity.float()
                        ),
                        noop_velocity_tensor_sha256=frozen.tensor_sha256(
                            noop_velocity.float()
                        ),
                        action_full_prompt_sha256=prompt_binding[
                            "action_full_prompt_utf8_sha256"
                        ],
                        noop_full_prompt_sha256=prompt_binding[
                            "noop_full_prompt_utf8_sha256"
                        ],
                        action_condition_tensor_sha256=condition_hashes[
                            "target_action"
                        ],
                        noop_condition_tensor_sha256=condition_hashes["noop"],
                        frozen_model_receipt_digest=checkpoint_receipt_digest,
                        same_state_execution_proof=same_state_proof,
                    )
                    energy_rows.append(
                        {
                            "native_schedule_index": schedule_index,
                            "action_energy": _energy(action_velocity, target),
                            "noop_energy": _energy(noop_velocity, target),
                            "prompt_pair_receipt": pair_receipt,
                        }
                    )
                    del sigma, x_sigma, action_velocity, noop_velocity
                energy_grid[transform_name] = energy_rows
                del transformed, effective_epsilon, target

            generation_binding = {
                "candidate_envelope_sha256": row["candidate_envelope_sha256"],
                "generation_receipt_digest": row["generation_receipt_digest"],
                "generation_receipt_file_sha256": row[
                    "generation_receipt_file_sha256"
                ],
                "native_rollout_receipt_digest": row[
                    "native_rollout_receipt_digest"
                ],
                "native_rollout_receipt_file_sha256": row[
                    "native_rollout_receipt_file_sha256"
                ],
                "generated_mp4_sha256": row["artifacts"]["mp4"]["sha256"],
                "geometry_source_video_sha256": row["candidate"][
                    "geometry_source_video_sha256"
                ],
                "candidate_own_caption_utf8_sha256": row["candidate"][
                    "full_t2v_caption_utf8_sha256"
                ],
                "clean_latent_artifact_sha256": row["artifacts"][
                    "predecode_clean_latent"
                ]["sha256"],
                "clean_latent_tensor_sha256": frozen.tensor_sha256(clean),
                "official_gaussian_artifact_sha256": gaussian_artifact["sha256"],
                "official_gaussian_raw_value_sha256": gaussian_artifact[
                    "raw_value_sha256"
                ],
                "official_gaussian_content_sha256": gaussian_artifact[
                    "content_sha256"
                ],
                "official_gaussian_tensor_sha256": epsilon_tensor_sha,
            }
            candidate_identity = {
                name: row["candidate"][name]
                for name in (
                    "candidate_id",
                    "analysis_split",
                    "action_family_id",
                    "calibration_group_id",
                    "actor_group_id",
                    "scene_group_id",
                    "action_group_id",
                    "semantic_branch",
                )
            }
            receipt = contract.make_candidate_score_receipt(
                group_id=args.group_id,
                candidate_identity=candidate_identity,
                root_spec_raw_sha256=args.expected_root_spec_sha256,
                bank_receipt_digest=bank["receipt_digest"],
                bank_receipt_file_sha256=bank["file_sha256"],
                generation_binding=generation_binding,
                target_action_binding=target_binding,
                prompt_binding=prompt_binding,
                model_binding=model_binding,
                energy_by_transform=energy_grid,
            )
            contract.validate_candidate_score_receipt(receipt)
            try:
                freeze_after = native_generation.source_audit.model_freeze_certificate(
                    renderer
                )
            except Exception as error:
                raise TemporalCounterfactualScoringError(str(error)) from error
            if freeze_after != freeze_certificate or any(
                parameter.requires_grad for parameter in renderer.parameters()
            ):
                raise TemporalCounterfactualScoringError(
                    "frozen renderer changed during scoring"
                )
            digests: list[Any] = [None] * distributed.world_size
            dist.all_gather_object(digests, receipt["receipt_digest"])
            if len(set(digests)) != 1:
                raise TemporalCounterfactualScoringError("SP4 score receipts differ")
            if distributed.rank == 0:
                candidate_dir = output / candidate_id
                candidate_dir.mkdir()
                _write_create_only(candidate_dir / SCORE_FILENAME, receipt)
                receipts.append(receipt)
            del clean, clean_cpu, energy_grid, receipt
            if row_index != 0:
                del candidate_gaussian
        del first_gaussian, first_identity, epsilon, conditions

    if distributed.rank == 0:
        group_receipt = contract.make_group_receipt(
            group_id=args.group_id,
            candidate_receipts=receipts,
            root_spec_raw_sha256=args.expected_root_spec_sha256,
            bank_receipt_digest=bank["receipt_digest"],
            method_source_revision=args.method_source_revision,
            method_source_archive_sha256=args.method_source_archive_sha256,
            scorer_source_sha256=args.expected_scorer_source_sha256,
            contract_source_sha256=args.expected_contract_source_sha256,
        )
        contract.validate_group_receipt(group_receipt)
        _write_create_only(
            output / GROUP_FILENAME.format(group_id=args.group_id), group_receipt
        )
        os.chmod(output, 0o500)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROUP_FILENAME",
    "MODEL_FORWARDS_PER_CANDIDATE",
    "PROMPT_ORDER",
    "SCORE_FILENAME",
    "TemporalCounterfactualScoringError",
    "build_parser",
    "forward_native_prompt_pair",
    "main",
    "validate_native_coordinate_runtime",
]
