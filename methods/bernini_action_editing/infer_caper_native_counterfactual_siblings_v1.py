#!/usr/bin/env python3
"""Render one sealed CAPER same-source/same-seed four-arm sibling cell.

Each arm independently executes Bernini-R 1.3B's stock full-source-video
``v2v_apg`` endpoint for all 40 UniPC steps.  Arms share the exact source
latent, checkpoint, schedule, negative embedding, seed, and therefore the
equal-valued *official* sampler Gaussian.  They do not share a denoising
prefix, scheduler state, positive embedding, or post-initial latent.

The official Gaussian is observed, never supplied.  Target videos, custom
noise, first-frame anchors, references, masks, tracks, pose, flow, trajectory,
homotopy, tri-branch projection, optimization, and parameter updates are
outside this runner's call graph.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
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

import infer_caper_native_kseed_population_v1 as provenance  # noqa: E402


METHOD = "frozen-bernini-caper-native-same-seed-counterfactual-siblings-v1"
SCHEMA_VERSION = "bernini-caper-native-counterfactual-sibling-cell-receipt-v1"
ARM_RECEIPT_SCHEMA_VERSION = (
    "bernini-caper-native-counterfactual-sibling-arm-receipt-v1"
)
REGISTRY_SCHEMA_VERSION = (
    "bernini-caper-native-counterfactual-sibling-population-sit-v1"
)
CANONICAL_REGISTRY_RELATIVE = (
    "assets/caper_native_counterfactual_siblings_sit_v1.json"
)
CANONICAL_REGISTRY_SHA256 = (
    "09b1bdc13a91ded9fe853f9582499da310995c0bc4d48e50599f52cb0de154e0"
)

SOURCE_ID = "7b88a1ca1f804f41"
SEEDS = (2026081801, 2026081802, 2026081803, 2026081804)
CELL_ORDER = tuple(f"fit-{SOURCE_ID}-s{seed}" for seed in SEEDS)
ARM_ORDER = ("target", "noop", "incomplete", "phase-order-violation")
NATIVE_ARM = provenance.NATIVE_ARM
FRAME_COUNT = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
FPS = 25
ULYSSES_SIZE = 4
SCHEDULE_SHA256 = provenance.SCHEDULE_SHA256
BERNINI_OFFICIAL_COMMIT = provenance.BERNINI_OFFICIAL_COMMIT
VEOMNI_TESTED_COMMIT = provenance.VEOMNI_TESTED_COMMIT
CHECKPOINT_TREE_SHA256 = provenance.CHECKPOINT_TREE_SHA256

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CaperNativeSiblingError(RuntimeError):
    """Raised before an incomplete or non-native sibling cell is published."""


def _object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_registry() -> Mapping[str, Any]:
    path = METHOD_ROOT / CANONICAL_REGISTRY_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != CANONICAL_REGISTRY_SHA256
    ):
        raise CaperNativeSiblingError("sealed sibling registry bytes differ")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CaperNativeSiblingError("sealed sibling registry JSON differs") from error
    if not isinstance(value, Mapping):
        raise CaperNativeSiblingError("sibling registry root differs")
    return value


def _validate_native_contract(registry: Mapping[str, Any]) -> None:
    row = registry.get("native_v2v_contract")
    if (
        not isinstance(row, Mapping)
        or row.get("endpoint") != NATIVE_ARM
        or row.get("guidance_mode") != "v2v_apg"
        or row.get("positive_task") != "mv2v"
        or row.get("full_source_video_count") != 1
        or any(
            row.get(key) != 0
            for key in (
                "source_reference_count",
                "first_frame_anchor_count",
                "target_video_count",
                "mask_count",
                "track_count",
                "pose_count",
                "flow_count",
                "trajectory_count",
            )
        )
        or row.get("custom_initial_noise") is not False
        or row.get(
            "official_gaussian_shared_by_equal_value_not_external_injection"
        )
        is not True
        or row.get("official_gaussian_captured_by_read_only_observer") is not True
        or row.get("observer_injects_or_replaces_noise") is not False
        or row.get("shared_high_sigma_prefix") is not False
        or row.get("independent_complete_stock_trajectory_per_arm") is not True
        or row.get("frame_count") != FRAME_COUNT
        or row.get("fps") != FPS
        or row.get("latent_phases") != LATENT_PHASES
        or row.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or row.get("flow_shift_from_renderer_config") != 5.0
        or row.get("omega_text") != 4.0
        or row.get("eta") != 0.5
        or row.get("norm_threshold") != 50.0
        or row.get("momentum") != 0.0
        or row.get("exact40_shift5_schedule_sha256") != SCHEDULE_SHA256
        or row.get("frozen_model") is not True
        or row.get("training") is not False
        or row.get("optimizer") is not False
        or row.get("parameter_update") is not False
    ):
        raise CaperNativeSiblingError("stock full-source V2V contract differs")


def _validate_registry(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    if registry != _canonical_registry():
        raise CaperNativeSiblingError("registry differs from sealed authority")
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("method") != METHOD
        or registry.get("arm_order") != list(ARM_ORDER)
    ):
        raise CaperNativeSiblingError("registry root differs")

    population = registry.get("population_design")
    if (
        not isinstance(population, Mapping)
        or population.get("split") != "fit"
        or population.get("source_ids") != [SOURCE_ID]
        or population.get("seeds") != list(SEEDS)
        or population.get("cell_order") != list(CELL_ORDER)
        or population.get("k") != len(SEEDS)
        or population.get("expected_cell_count") != len(CELL_ORDER)
        or population.get("arms_per_cell") != len(ARM_ORDER)
        or population.get("expected_rollout_count") != len(CELL_ORDER) * len(ARM_ORDER)
        or population.get("cartesian_population_required") is not True
        or population.get("all_registered_cells_and_arms_must_materialize") is not True
        or population.get("seed_filtering_or_best_of_k_authorized") is not False
        or population.get("replacement_seed_authorized") is not False
        or population.get("partial_population_scientific_claim_authorized") is not False
        or population.get("lockbox_source_or_media_access_authorized") is not False
    ):
        raise CaperNativeSiblingError("K=4 public-fit population differs")
    _validate_native_contract(registry)

    semantics = registry.get("counterfactual_semantics")
    if (
        not isinstance(semantics, Mapping)
        or set(semantics).issuperset(ARM_ORDER) is False
        or semantics.get("literal_time_reversal_used") is not False
        or semantics.get("all_arms_share_source_initial_precondition") is not True
        or semantics.get("phase_order_violation_is_not_labeled_literal_reverse")
        is not True
        or "same_initial_standing_precondition"
        not in str(semantics.get("phase-order-violation"))
    ):
        raise CaperNativeSiblingError("counterfactual semantics differ")

    source = registry.get("source")
    if not isinstance(source, Mapping):
        raise CaperNativeSiblingError("source row differs")
    captions = source.get("captions")
    caption_hashes = source.get("caption_sha256")
    if (
        source.get("split") != "fit"
        or source.get("source_id") != SOURCE_ID
        or source.get("actor_kind") != "dog"
        or source.get("identity_id") != "grey-french-bulldog-black-harness"
        or source.get("scene_id") != "yellow-autumn-leaves-sunlit-park"
        or source.get("registered_turn_target") != "frame_right_toward_background"
        or not str(source.get("source_video", "")).endswith(
            f"/samples/{SOURCE_ID}/samples/{SOURCE_ID}/source_video.mp4"
        )
        or source.get("source_video_sha256")
        != "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"
        or source.get("source_hw") != [704, 736]
        or source.get("bucket_hw") != [480, 496]
        or source.get("latent_shape") != [1, 16, 21, 60, 62]
        or not isinstance(captions, Mapping)
        or tuple(captions) != ARM_ORDER
        or not isinstance(caption_hashes, Mapping)
        or tuple(caption_hashes) != ARM_ORDER
    ):
        raise CaperNativeSiblingError("source geometry/caption closure differs")
    for arm in ARM_ORDER:
        caption = captions[arm]
        if (
            not isinstance(caption, str)
            or _sha256_text(caption) != caption_hashes[arm]
            or "A locked-off camera shows the same single grey French bulldog" not in caption
            or "From frames 0 through 20" not in caption
            or "From frames 20 through 40" not in caption
            or "From frames 40 through 80" not in caption
            or "Preserve the source dog" not in caption
        ):
            raise CaperNativeSiblingError(f"{arm} complete source-scene caption differs")
    if (
        "source standing pose" not in captions["noop"]
        or "only partway" not in captions["incomplete"]
        or "first turns its head" not in captions["phase-order-violation"]
        or "rises back to a source-consistent standing pose"
        not in captions["phase-order-violation"]
    ):
        raise CaperNativeSiblingError("counterfactual arm meaning differs")
    return source


def _registry_cell(registry: Mapping[str, Any], *, cell_id: str) -> Mapping[str, Any]:
    source = _validate_registry(registry)
    if cell_id not in CELL_ORDER:
        raise CaperNativeSiblingError("cell is outside sealed public-fit K=4")
    for seed in SEEDS:
        if cell_id == f"fit-{SOURCE_ID}-s{seed}":
            return {
                **dict(source),
                "cell_id": cell_id,
                "seed": seed,
                "arm_order": list(ARM_ORDER),
                "selected_before_generation": True,
                "seed_filtering_or_best_of_k_authorized": False,
            }
    raise CaperNativeSiblingError("cell coordinates differ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--cell-id", choices=CELL_ORDER, required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--runtime-source-closure-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", default=BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=VEOMNI_TESTED_COMMIT)
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=CHECKPOINT_TREE_SHA256
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_registry_sha256",
        "runtime_source_archive_sha256",
        "runtime_source_closure_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise CaperNativeSiblingError(f"{name} differs")
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise CaperNativeSiblingError(f"{name} differs")
    if args.expected_registry_sha256 != CANONICAL_REGISTRY_SHA256:
        raise CaperNativeSiblingError("canonical sibling registry digest differs")
    if args.expected_bernini_commit != BERNINI_OFFICIAL_COMMIT:
        raise CaperNativeSiblingError("Bernini revision differs")
    if args.expected_veomni_commit != VEOMNI_TESTED_COMMIT:
        raise CaperNativeSiblingError("VeOmni revision differs")
    if args.expected_checkpoint_tree_sha256 != CHECKPOINT_TREE_SHA256:
        raise CaperNativeSiblingError("checkpoint tree differs")
    if args.cell_id not in CELL_ORDER:
        raise CaperNativeSiblingError("CLI cell differs")


def _live_exact40_schedule_receipt(scheduler: Any, *, base: Any) -> Mapping[str, Any]:
    """Read the completed stock scheduler state without wrapping its step call."""

    try:
        timesteps = [float(item) for item in scheduler.timesteps.detach().cpu().tolist()]
        sigmas = [float(item) for item in scheduler.sigmas.detach().cpu().tolist()]
    except Exception as error:
        raise CaperNativeSiblingError("cannot read live stock UniPC schedule") from error
    pinned = base.branch_base.pinned_exact40_schedule_receipt()
    expected_timesteps = list(base.branch_base.NATIVE_UNIPC40_TIMESTEPS)
    expected_sigmas = list(base.branch_base.NATIVE_UNIPC40_SIGMAS)
    if (
        len(timesteps) != NUM_INFERENCE_STEPS
        or [int(round(item)) for item in timesteps] != expected_timesteps
        or len(sigmas) != NUM_INFERENCE_STEPS + 1
        or sigmas[:NUM_INFERENCE_STEPS] != expected_sigmas
        or sigmas[-1] != 0.0
        or pinned.get("digest") != SCHEDULE_SHA256
    ):
        raise CaperNativeSiblingError("live stock exact40/shift5 schedule differs")
    step_index = getattr(scheduler, "step_index", None)
    if step_index is None:
        step_index = getattr(scheduler, "_step_index", None)
    if type(step_index) is not int or step_index != NUM_INFERENCE_STEPS:
        raise CaperNativeSiblingError("stock UniPC did not complete exactly 40 steps")
    live = {
        "scheduler": type(scheduler).__name__,
        "completed_step_index": step_index,
        "timesteps": [int(round(item)) for item in timesteps],
        "sigma_float64_hex": [float(item).hex() for item in sigmas[:-1]],
        "terminal_sigma_float64_hex": float(sigmas[-1]).hex(),
        "pinned_schedule_sha256": SCHEDULE_SHA256,
    }
    return {**live, "live_schedule_digest": _object_sha256(live)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    try:
        registry_path, registry = provenance._plain_json(args.registry, label="registry")
    except Exception as error:
        raise CaperNativeSiblingError(str(error)) from error
    if _file_sha256(registry_path) != args.expected_registry_sha256:
        raise CaperNativeSiblingError("registry file digest differs")
    cell = _registry_cell(registry, cell_id=args.cell_id)
    base = provenance._load_base()
    output_dir = base.native._resolve_fresh_output_dir(args.output_dir)

    source_requested = Path(str(cell["source_video"])).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise CaperNativeSiblingError("source path differs")
    source_path = source_requested.resolve(strict=True)
    if (
        source_path != source_requested
        or not source_path.is_file()
        or _file_sha256(source_path) != cell["source_video_sha256"]
    ):
        raise CaperNativeSiblingError("source bytes differ")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            base.native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = base.native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except Exception as error:
        raise CaperNativeSiblingError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise CaperNativeSiblingError("attention heads do not divide Ulysses4")
    inference_file_hashes = base.native.legacy.validate_inference_source_files(
        bernini_root
    )
    base.native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != base.native.legacy.MV2V_SYSTEM_PROMPT:
        raise CaperNativeSiblingError("runtime MV2V system prompt differs")
    if DEFAULT_NEG_PROMPT != base.native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise CaperNativeSiblingError("runtime renderer negative prompt differs")

    distributed = base.native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != ULYSSES_SIZE
        or distributed.ulysses_size != ULYSSES_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise CaperNativeSiblingError("runtime requires AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": base.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if (
        not isinstance(checkpoint_rows[0], Mapping)
        or checkpoint_rows[0].get("ok") is not True
    ):
        raise CaperNativeSiblingError(
            f"checkpoint validation failed: {checkpoint_rows[0]}"
        )
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = (
        base.source_audit.prepare_hashed_source_snapshot(source_path)
    )
    bucket_hw = tuple(int(item) for item in cell["bucket_hw"])
    latent_shape = tuple(int(item) for item in cell["latent_shape"])
    if (
        source_sha != cell["source_video_sha256"]
        or source_metadata.get("frame_count") != FRAME_COUNT
        or abs(float(source_metadata.get("reported_fps", 0.0)) - FPS) > 1.0e-6
        or tuple(source_metadata.get("source_input_hw", ()))
        != tuple(cell["source_hw"])
        or tuple(source_metadata.get("source_derived_bucket_hw", ())) != bucket_hw
    ):
        raise CaperNativeSiblingError("source exact81 geometry differs")

    captions = {arm: str(cell["captions"][arm]) for arm in ARM_ORDER}
    positive_prompts = {
        arm: base.build_mode_native_prompt(
            "source-mv2v", captions[arm], prompt_cleaner=prompt_clean
        )
        for arm in ARM_ORDER
    }
    if len(set(positive_prompts.values())) != len(ARM_ORDER):
        raise CaperNativeSiblingError("arm-native MV2V prompts alias")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        **base.native.legacy.tokenizer_load_kwargs(),
    )
    positive_tokens = {
        arm: base.native.legacy._tokenize_training_prompt(
            tokenizer, positive_prompts[arm]
        )
        for arm in ARM_ORDER
    }
    negative_ids, negative_mask = base.native.legacy._tokenize_renderer_negative(
        tokenizer, base.native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **base.native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    base.native.legacy.trainer.validate_renderer_config_mapping(
        config.to_dict(), checkpoint
    )
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise CaperNativeSiblingError("renderer exact40 shift5 UniPC differs")
    model = BerniniRendererModel(config)
    model.eval().requires_grad_(False)
    freeze_before = base._strong_model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    if distributed.rank == 0:
        source_pixels = source_tensor.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            source_latent = _vae_encode(vae, source_pixels).contiguous()
        del source_pixels
    else:
        source_latent = torch.empty(latent_shape, device=device, dtype=torch.float32)
    dist.broadcast(source_latent, src=0)
    if tuple(source_latent.shape) != latent_shape:
        raise CaperNativeSiblingError("full source latent geometry differs")
    source_identity_before = base.native._all_rank_tensor_identity(
        source_latent,
        label=f"{args.cell_id}_shared_full_source_video",
        world_size=ULYSSES_SIZE,
    )
    vae.to("cpu")
    del source_tensor
    torch.cuda.empty_cache()

    model.to(device)
    model.t5_text_encoder.to(device)
    positive_embeds: dict[str, Any] = {}
    with torch.inference_mode():
        for arm in ARM_ORDER:
            ids, mask = positive_tokens[arm]
            positive_embeds[arm] = model.encode_prompt(
                ids.to(device), mask.to(device)
            ).detach()
        uncond_embeds = model.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    expected_embed_shape = (1, 512, 4096)
    if any(
        tuple(value.shape) != expected_embed_shape
        for value in (*positive_embeds.values(), uncond_embeds)
    ):
        raise CaperNativeSiblingError("MV2V sibling embedding geometry differs")
    for left_index, left in enumerate(ARM_ORDER):
        for right in ARM_ORDER[left_index + 1 :]:
            if torch.equal(positive_embeds[left], positive_embeds[right]):
                raise CaperNativeSiblingError("distinct arm positive embeddings alias")
    positive_embedding_identities = {
        arm: base.native._all_rank_tensor_identity(
            positive_embeds[arm],
            label=f"{args.cell_id}_{arm}_positive_embedding",
            world_size=ULYSSES_SIZE,
        )
        for arm in ARM_ORDER
    }
    negative_embedding_identity = base.native._all_rank_tensor_identity(
        uncond_embeds,
        label=f"{args.cell_id}_shared_negative_embedding",
        world_size=ULYSSES_SIZE,
    )
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = base.sampler_contract.resolve_diffusion_core(model.diff_dec)
    wan_source_sha = base.sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
    )
    base.sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=5.0
    )
    if diffusion.transformer_2 is not None:
        raise CaperNativeSiblingError("siblings require Bernini-R 1.3B single DiT")
    pinned_schedule = base.branch_base.pinned_exact40_schedule_receipt()
    if pinned_schedule.get("digest") != SCHEDULE_SHA256:
        raise CaperNativeSiblingError("pinned exact40 schedule differs")

    common_conditions = base.conditions_for_arm(
        NATIVE_ARM, source_latent=source_latent
    )
    common_sampling = base.sampling_contract(NATIVE_ARM, seed=int(cell["seed"]))
    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    initial_noise: dict[str, Any] = {}
    initial_noise_identities: dict[str, Any] = {}
    live_schedules: dict[str, Any] = {}
    runtime_traces: dict[str, Any] = {}
    with torch.inference_mode():
        for arm in ARM_ORDER:
            sample_kwargs = {
                "prompt_embeds": positive_embeds[arm],
                "uncond_prompt_embeds": uncond_embeds,
                **common_conditions,
                "width": bucket_hw[1],
                "height": bucket_hw[0],
                "device": device,
                **common_sampling,
            }
            result, capture = base.native._sample_with_native_initial_noise_observer(
                sample_fn=lambda kwargs=sample_kwargs: diffusion.sample(**kwargs),
                wan_diffusion_module=wan_diffusion,
                expected_shape=latent_shape,
                expected_device=device,
                expected_seed=int(cell["seed"]),
            )
            if (
                not isinstance(result, torch.Tensor)
                or tuple(result.shape) != latent_shape
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not bool(torch.isfinite(result).all().item())
            ):
                raise CaperNativeSiblingError(f"{arm} stock sampler result differs")
            stored = result.detach().to(device="cpu").contiguous()
            generated[arm] = stored
            generated_identities[arm] = base.native._all_rank_tensor_identity(
                stored,
                label=f"{args.cell_id}_{arm}_clean_latent",
                world_size=ULYSSES_SIZE,
            )
            initial_noise[arm] = capture
            initial_noise_identities[arm] = base.native._all_rank_tensor_identity(
                capture.tensor,
                label=f"{args.cell_id}_{arm}_official_initial_gaussian",
                world_size=ULYSSES_SIZE,
            )
            live_schedules[arm] = dict(
                _live_exact40_schedule_receipt(diffusion.scheduler, base=base)
            )
            runtime_traces[arm] = {
                "native_endpoint": True,
                "denoiser_or_scheduler_field_hook_installed": False,
                "initial_noise_observer_installed": True,
                "initial_noise_observer_read_only": True,
                "initial_noise_observer_replaces_or_injects_noise": False,
                "guidance_mode": "v2v_apg",
                "positive_task": "mv2v",
                "expected_transformer_forwards_per_step_from_pinned_vendor": 2,
                "source_video_count": 1,
                "source_reference_count": 0,
                "target_video_count": 0,
                "mask_track_pose_flow_count": 0,
                "shared_prefix_steps": 0,
                "complete_independent_stock_steps": NUM_INFERENCE_STEPS,
                "vendor_source_sha256": wan_source_sha,
            }

    noise_hashes = {capture.raw_value_sha256 for capture in initial_noise.values()}
    live_schedule_digests = {
        row["live_schedule_digest"] for row in live_schedules.values()
    }
    if len(noise_hashes) != 1:
        raise CaperNativeSiblingError("same seed did not yield one official Gaussian value")
    if len(live_schedule_digests) != 1:
        raise CaperNativeSiblingError("arm live schedules differ")
    source_identity_after = base.native._all_rank_tensor_identity(
        source_latent,
        label=f"{args.cell_id}_shared_full_source_video",
        world_size=ULYSSES_SIZE,
    )
    if source_identity_after != source_identity_before:
        raise CaperNativeSiblingError("shared source latent changed across arms")
    trace_digest = _object_sha256(runtime_traces)
    trace_rows: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(trace_rows, trace_digest)
    if len(set(trace_rows)) != 1:
        raise CaperNativeSiblingError("stock sibling traces differ across SP4 ranks")

    freeze_after = base._strong_model_freeze_certificate(model)
    if freeze_after != freeze_before or any(p.requires_grad for p in model.parameters()):
        raise CaperNativeSiblingError("frozen model changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_rows[0] = {
                "ok": True,
                "identity": base.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_rows, src=0)
    if (
        not isinstance(after_rows[0], Mapping)
        or after_rows[0].get("identity") != checkpoint_identity
    ):
        raise CaperNativeSiblingError("checkpoint content changed")

    if distributed.rank == 0:
        output_dir.mkdir(parents=False, exist_ok=False)
        source_artifact = base.native._save_normalized_clean_latent_atomically(
            output_dir / "source.normalized-clean-latent.safetensors",
            source_latent,
            artifact_role="source_video_condition",
        )
        noise_artifacts = {
            arm: base.native._save_initial_noise_atomically(
                output_dir / f"{arm}.official-initial-gaussian.safetensors",
                initial_noise[arm],
                all_rank_identity=initial_noise_identities[arm],
            )
            for arm in ARM_ORDER
        }
        generated_for_decode = {
            arm: generated[arm].to(device=device).contiguous() for arm in ARM_ORDER
        }
        try:
            outputs = base.native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=bucket_hw,
                device=device,
                save_output_fn=save_output,
            )
        finally:
            generated_for_decode.clear()

        shared_gaussian_sha = next(iter(noise_hashes))
        arm_receipt_rows: list[dict[str, Any]] = []
        for arm_index, arm in enumerate(ARM_ORDER):
            arm_receipt: dict[str, Any] = {
                "schema_version": ARM_RECEIPT_SCHEMA_VERSION,
                "method": METHOD,
                "cell_id": args.cell_id,
                "arm": arm,
                "arm_index": arm_index,
                "source_id": SOURCE_ID,
                "seed": int(cell["seed"]),
                "caption": captions[arm],
                "caption_sha256": _sha256_text(captions[arm]),
                "full_mv2v_prompt_sha256": _sha256_text(positive_prompts[arm]),
                "positive_embedding_identity": positive_embedding_identities[arm],
                "shared_negative_prompt_sha256": _sha256_text(
                    base.native.legacy.DEFAULT_NEGATIVE_PROMPT
                ),
                "shared_negative_embedding_identity": negative_embedding_identity,
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "shared_source_identity": source_identity_before,
                "source_condition_artifact": source_artifact,
                "sampling": {
                    "guidance_mode": "v2v_apg",
                    "positive_task": "mv2v",
                    "frame_count": FRAME_COUNT,
                    "latent_phases": LATENT_PHASES,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "fps": FPS,
                    "seed": int(cell["seed"]),
                    "official_gaussian_raw_sha256": shared_gaussian_sha,
                    "official_gaussian_is_sampler_initial_noise": True,
                    "official_gaussian_captured_by_read_only_observer": True,
                    "observer_injects_or_replaces_noise": False,
                    "external_initial_noise_injection": False,
                    "shared_high_sigma_prefix": False,
                    "independent_complete_stock_trajectory": True,
                    "flow_shift": 5.0,
                    "omega_text": 4.0,
                    "eta": 0.5,
                    "norm_threshold": 50.0,
                    "momentum": 0.0,
                    "schedule_sha256": SCHEDULE_SHA256,
                    "live_schedule": live_schedules[arm],
                },
                "runtime_trace": runtime_traces[arm],
                "generated_identity": generated_identities[arm],
                "official_initial_gaussian": noise_artifacts[arm],
                "output": outputs[arm],
                "checkpoint": checkpoint_identity,
                "freeze_certificate": freeze_after,
                "training_performed": False,
                "optimizer_created": False,
                "parameter_update": False,
                "preference_admission_performed": False,
                "scientific_or_action_editing_claim_authorized": False,
            }
            arm_receipt["receipt_digest"] = _object_sha256(arm_receipt)
            arm_path = output_dir / f"{arm}.receipt.json"
            base.value_audit.write_receipt_atomically(arm_path, arm_receipt)
            arm_receipt_rows.append(
                {
                    "arm": arm,
                    "path": str(arm_path),
                    "file_sha256": _file_sha256(arm_path),
                    "receipt_digest": arm_receipt["receipt_digest"],
                }
            )

        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_id": args.cell_id,
            "source_id": SOURCE_ID,
            "seed": int(cell["seed"]),
            "arm_order": list(ARM_ORDER),
            "expected_arm_count": len(ARM_ORDER),
            "complete_arm_count": len(arm_receipt_rows),
            "all_four_sibling_arms_complete": True,
            "registry": {
                "path": str(registry_path),
                "sha256": CANONICAL_REGISTRY_SHA256,
                "complete_k4_cell_order": list(CELL_ORDER),
                "seed_filtering_or_best_of_k_authorized": False,
                "replacement_seed_authorized": False,
            },
            "input": {
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "target_video": False,
                "custom_initial_noise": False,
                "mask_track_pose_flow_trajectory": False,
                "first_frame_anchor": False,
                "source_reference": False,
            },
            "preprocessing": dict(source_metadata),
            "shared_contract": {
                "same_source_latent_all_arms": True,
                "source_identity_before": source_identity_before,
                "source_identity_after": source_identity_after,
                "same_checkpoint_all_arms": True,
                "same_negative_embedding_object_within_each_rank": True,
                "same_official_gaussian_value_all_arms": True,
                "official_gaussian_raw_sha256": shared_gaussian_sha,
                "same_exact40_schedule_all_arms": True,
                "live_schedule_digest": next(iter(live_schedule_digests)),
                "shared_high_sigma_prefix": False,
                "shared_prefix_steps": 0,
                "independent_complete_stock_trajectory_per_arm": True,
                "positive_embedding_shared_across_arms": False,
                "external_initial_noise_injection": False,
            },
            "source_condition_artifact": source_artifact,
            "arm_receipts": arm_receipt_rows,
            "runtime_trace_digest": trace_digest,
            "checkpoint": checkpoint_identity,
            "freeze_certificate": freeze_after,
            "source_revisions": {
                "bernini": bernini_revision,
                "veomni": veomni_revision,
                "wan_diffusion_sha256": wan_source_sha,
                "runtime_method": args.runtime_source_revision,
                "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                "runtime_source_closure_sha256": args.runtime_source_closure_sha256,
                "launcher_source_sha256": args.launcher_source_sha256,
                "inference_files": inference_file_hashes,
            },
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "training_performed": False,
            "optimizer_created": False,
            "parameter_update": False,
            "preference_admission_performed": False,
            "all_k4_cells_required_before_population_claim": True,
            "partial_population_scientific_claim_authorized": False,
            "scientific_or_action_editing_claim_authorized": False,
        }
        receipt["receipt_digest"] = _object_sha256(receipt)
        base.value_audit.write_receipt_atomically(output_dir / "receipt.json", receipt)
        print(base.native.legacy.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, generated, initial_noise, positive_embeds, uncond_embeds
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER",
    "ARM_RECEIPT_SCHEMA_VERSION",
    "CANONICAL_REGISTRY_RELATIVE",
    "CANONICAL_REGISTRY_SHA256",
    "CELL_ORDER",
    "CaperNativeSiblingError",
    "METHOD",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_ID",
    "_file_sha256",
    "_object_sha256",
    "_registry_cell",
    "_sha256_text",
    "_validate_cli",
    "_validate_registry",
    "build_parser",
    "main",
]
