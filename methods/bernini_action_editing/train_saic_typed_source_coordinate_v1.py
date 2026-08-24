#!/usr/bin/env python3
"""Train an exploratory typed local-cell operator with a SCAID target.

This is an executable *pilot*, not the complete Source-Anchored Inverse-Cycle
(SAIC) method and not evidence that action editing works.  It
keeps the PAIR-v6 SCAID manifest, authoritative evidence gate, source-
coordinate objective, and serial leaf-VJP runtime, but replaces the static
PAIR action LoRA with ``saic_typed_action_operator_v1``.

WORLD8 is fixed to DP2 x Ulysses-SP4.  DP arm zero owns the sealed French-
bulldog standing-to-sitting fit source and receives the exact 32-D arrow e0.
DP arm one owns the sealed human kneeling-to-standing fit source and receives
e1.  Every native RV2V branch binds its route from the branch's real target
mask, the live sequence-parallel group, and the device-local physical sigma.
The frozen T2V field never receives an action route.

This first pilot deliberately has no decoded event gate and no inverse-cycle
loss.  It accepts neither a generated proposal nor extra inference-time
localization input.  Indices 38/39 remain byte-path frozen-base anchors.  A
completed optimizer run is only a typed source-coordinate training artifact;
it is not a semantic-success claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dclr_runtime_contract as t2v_runtime  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v6_scaid_source_coordinate as scaid  # noqa: E402
import saic_typed_action_operator_v1 as typed_operator  # noqa: E402
import source_self_native_ref_contrastive_v3 as native  # noqa: E402
import source_self_runtime as distributed_runtime  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_pair_v5_action_preference as native_runtime  # noqa: E402
import train_pair_v5_t2v_guidance_distill as cagd_runtime  # noqa: E402
import train_pair_v6_scaid as baseline  # noqa: E402


METHOD_NAME = "bernini-exploratory-typed-local-cell-v1"
RUN_RECEIPT_SCHEMA = "bernini-saic-typed-source-coordinate-run-v1"
HISTORY_SCHEMA = "bernini-saic-typed-source-coordinate-history-v1"
WORLD_SIZE = 8
DP_SIZE = 2
SP_SIZE = 4
FRAME_COUNT = 81
FPS = 25.0
REFERENCE_INDICES = (0, 27, 53, 80)
EXACT40_STEPS = 40
DEFAULT_SEED = 20260809
DEFAULT_LR = 1.0e-6
DEFAULT_MAX_GRAD_NORM = 1.0
# These are source-video IIDs, not caller-selectable guidance candidate IDs.
FIT_SAMPLE_IDS = ("7b88a1ca1f804f41", "a35b590961d24694")
ARROW_STATE_TYPES = (
    ("dog_standing", "dog_sitting"),
    ("human_kneeling", "human_standing"),
)


class SAICTypedSourceCoordinateTrainingError(RuntimeError):
    """Raised before an ambiguous typed pilot update or publication."""


EventSpec = baseline.EventSpec
TrainingManifest = baseline.TrainingManifest
RuntimeEvent = baseline.RuntimeEvent
file_sha256 = baseline.file_sha256
object_sha256 = baseline.object_sha256
load_manifest = baseline.load_manifest
build_task_prompt_registry = baseline.build_task_prompt_registry
validate_native_condition_geometry = baseline.validate_native_condition_geometry
target_tail_equality_receipt = baseline.target_tail_equality_receipt
source_audit = baseline.source_audit


# This alias keeps the annotation readable without changing the runtime type.
SCAIDAuthorizationType = scaid.SCAIDAuthorization


@dataclass(frozen=True)
class TypedPilotPreflight:
    manifest: TrainingManifest
    authorizations: tuple[SCAIDAuthorizationType, SCAIDAuthorizationType]
    checkpoint_identity: Mapping[str, Any]


def exact40_schedule_index(step: int) -> int:
    if type(step) is not int or step < 0:
        raise SAICTypedSourceCoordinateTrainingError(
            "schedule step must be a nonnegative exact integer"
        )
    index = step % EXACT40_STEPS
    typed_operator.sigma_gate(index)
    return index


def expected_optimizer_updates(steps: int) -> int:
    if type(steps) is not int or steps <= 0 or steps % EXACT40_STEPS:
        raise SAICTypedSourceCoordinateTrainingError(
            "full training must contain complete exact40 cycles"
        )
    return steps // EXACT40_STEPS * len(
        typed_operator.HIGH_SIGMA_INDICES + typed_operator.MID_SIGMA_INDICES
    )


def noise_seed(*, seed: int, step: int, dp_arm: int) -> int:
    if type(seed) is not int or type(step) is not int or type(dp_arm) is not int:
        raise SAICTypedSourceCoordinateTrainingError("noise seed inputs differ")
    material = (
        f"{seed}\0saic-typed-source-coordinate-v1\0{step}\0{dp_arm}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def action_arrow_for_dp_arm(dp_arm: int) -> typed_operator.SAICArrowCode:
    """Return the preregistered DP0=e0 or DP1=e1 typed arrow."""

    if type(dp_arm) is not int or not 0 <= dp_arm < DP_SIZE:
        raise SAICTypedSourceCoordinateTrainingError("DP arm must be zero or one")
    values = [0.0] * typed_operator.ARROW_CODE_DIM
    values[dp_arm] = 1.0
    initial, terminal = ARROW_STATE_TYPES[dp_arm]
    return typed_operator.SAICArrowCode(initial, terminal, tuple(values))


def noop_arrow_for_dp_arm(dp_arm: int) -> typed_operator.SAICArrowCode:
    if type(dp_arm) is not int or not 0 <= dp_arm < DP_SIZE:
        raise SAICTypedSourceCoordinateTrainingError("DP arm must be zero or one")
    return typed_operator.SAICArrowCode.noop(ARROW_STATE_TYPES[dp_arm][0])


def _live_sp_coordinate(group: Any) -> tuple[int, int]:
    import torch

    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise SAICTypedSourceCoordinateTrainingError(
                "SP group supplied before distributed initialization"
            )
        return 0, 1
    try:
        return int(dist.get_rank(group=group)), int(dist.get_world_size(group=group))
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(
            f"cannot query live sequence-parallel group: {error}"
        ) from error


def actual_local_target_mask(native_branch: Any, sequence_parallel_group: Any) -> Any:
    """Slice the branch-owned suffix mask by the observed SP coordinate."""

    import torch

    try:
        target_mask = native_branch.target_mask
        total_tokens = native_branch.total_tokens
    except AttributeError as error:
        raise SAICTypedSourceCoordinateTrainingError(
            "native branch does not expose its target mask/geometry"
        ) from error
    if (
        type(target_mask) is not torch.Tensor
        or target_mask.dtype != torch.bool
        or target_mask.ndim != 1
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens <= 0
        or int(target_mask.numel()) != total_tokens
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "native branch target mask geometry differs"
        )
    rank, size = _live_sp_coordinate(sequence_parallel_group)
    if size not in typed_operator.ALLOWED_SP_SIZES or not 0 <= rank < size:
        raise SAICTypedSourceCoordinateTrainingError(
            "pilot supports only live SP1 tests or Ulysses-SP4"
        )
    local_length = math.ceil(total_tokens / size)
    padded = target_mask.detach().contiguous()
    if local_length * size > total_tokens:
        padded = torch.cat(
            (
                padded,
                torch.zeros(
                    local_length * size - total_tokens,
                    dtype=torch.bool,
                    device=padded.device,
                ),
            )
        )
    return padded[rank * local_length : (rank + 1) * local_length].contiguous()


def bind_native_action_route(
    *,
    handle: typed_operator.SAICTypedActionOperatorHandle,
    native_branch: Any,
    actual_sigma: Any,
    arrow: typed_operator.SAICArrowCode,
    parallel: Any,
) -> typed_operator.SAICTypedActionRoute:
    """Bind only from actual branch/SP/sigma runtime objects."""

    try:
        group = parallel.sp_group
    except AttributeError as error:
        raise SAICTypedSourceCoordinateTrainingError(
            "parallel state does not expose the live SP group"
        ) from error
    local_mask = actual_local_target_mask(native_branch, group)
    try:
        return handle.bind_runtime_route(
            native_branch=native_branch,
            actual_local_target_mask=local_mask,
            actual_sigma=actual_sigma,
            arrow=arrow,
            sequence_parallel_group=group,
        )
    except typed_operator.SAICTypedActionOperatorError as error:
        raise SAICTypedSourceCoordinateTrainingError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cagd-validator-evidence", required=True)
    parser.add_argument("--expected-cagd-validator-evidence-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-schedule-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-high-sigma-only", action="store_true")
    parser.add_argument(
        "--ack-typed-source-coordinate-pilot-no-semantic-success-claim",
        action="store_true",
    )
    return parser


def preflight(args: argparse.Namespace) -> TypedPilotPreflight:
    if (
        args.ack_typed_source_coordinate_pilot_no_semantic_success_claim
        is not True
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "typed pilot/no-success-claim acknowledgement is required"
        )
    if args.smoke_high_sigma_only:
        if args.max_schedule_steps != 1:
            raise SAICTypedSourceCoordinateTrainingError(
                "high-sigma smoke requires exactly one schedule cell"
            )
    else:
        expected_optimizer_updates(args.max_schedule_steps)
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise SAICTypedSourceCoordinateTrainingError(
                f"{name} must be finite and positive"
            )
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        baseline._sha(getattr(args, name), length=40, label=name)
    for name in (
        "expected_manifest_sha256",
        "expected_cagd_validator_evidence_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        baseline._sha(getattr(args, name), length=64, label=name)

    # Validate source trees before any evidence object can authorize a run.
    try:
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(
            f"official source-tree validation failed: {error}"
        ) from error
    manifest = load_manifest(args.manifest, args.expected_manifest_sha256)
    if (
        any(
            source_iid not in event.source_video.parts
            for source_iid, event in zip(FIT_SAMPLE_IDS, manifest.events)
        )
        or args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256
        or manifest.checkpoint_tree_sha256 != args.expected_checkpoint_tree_sha256
        or args.expected_checkpoint_content_manifest_sha256
        != source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "typed DP0 dog/DP1 human fit manifest or checkpoint identity differs"
        )
    try:
        checkpoint_identity = source_audit.validate_checkpoint_content(
            Path(args.checkpoint),
            Path(args.checkpoint_content_manifest),
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
        )
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(
            f"checkpoint content validation failed: {error}"
        ) from error
    authorizations = tuple(
        scaid.load_authoritative_v3_authorization(
            args.cagd_validator_evidence,
            expected_evidence_sha256=args.expected_cagd_validator_evidence_sha256,
            checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
            fit_candidate_id=event.fit_candidate_id,
        )
        for event in manifest.events
    )
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        rebuilt_t2v = tuple(
            build_task_prompt_registry(
                event.raw_caption_by_branch, prompt_cleaner=prompt_clean
            )[0]
            for event in manifest.events
        )
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(
            f"authoritative task-prompt reconstruction failed: {error}"
        ) from error
    if any(
        gate.action_family != event.action_family
        or gate.prompt_bank_sha256 != object_sha256(prompts)
        or event.sample_id != gate.fit_candidate_id
        or event.source_video != gate.geometry_source_video_path
        or event.source_video_sha256 != gate.geometry_source_video_sha256
        for gate, event, prompts in zip(
            authorizations, manifest.events, rebuilt_t2v
        )
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "SCAID evidence is not bound to the two typed fit events"
        )
    return TypedPilotPreflight(
        manifest=manifest,
        authorizations=(authorizations[0], authorizations[1]),
        checkpoint_identity=checkpoint_identity,
    )


class FrozenT2VCallback:
    """Frozen, unrouted T2V field used only by the inherited objective."""

    def __init__(
        self,
        diffusion: Any,
        transformer: Any,
        conditions: Mapping[str, Any],
        task_prompts: Mapping[str, str],
    ) -> None:
        self.diffusion = diffusion
        self.transformer = transformer
        self.conditions = conditions
        self.task_prompts = task_prompts
        self.query_id: Optional[int] = None
        self.branch: Any = None
        self.target_tail_sha256: Optional[str] = None

    def __call__(self, request: scaid.T2VFieldRequest) -> Any:
        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise SAICTypedSourceCoordinateTrainingError(
                "T2V callback prompt differs from the encoded task prompt"
            )
        query = request.coordinate
        if self.query_id != id(query):
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                patched = self.transformer.patch_vae_latent(
                    query.x_sigma.to(dtype=self.transformer.dtype), source_id=0
                )
            self.branch = t2v_runtime.build_t2v_target_branch(
                patched[0], patched[1], target_source_id=0
            )
            self.query_id = id(query)
            self.target_tail_sha256 = distributed_runtime.tensor_sha256(
                self.branch.noisy_latents.detach().float()
            )
        # No active SAIC route: pure T2V remains the exact frozen base.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = self.diffusion.shared_step(
                model_id="transformer_1",
                noisy_latents=self.branch.noisy_latents,
                timesteps=query.timestep,
                cond_embeds=self.conditions[request.branch],
                rotary_embs=self.branch.rotary_embs,
                batch_vae_seqlen=list(self.branch.batch_vae_seqlen),
                batch_text_seqlen=[512],
            )
        return native_bridge._unpack_spatial_velocity(
            packed, video_shape=query.x_sigma.shape
        )


class NativeTypedSCAIDCallback:
    """Native callback whose every model call is factory-route-bound."""

    def __init__(
        self,
        diffusion: Any,
        transformer: Any,
        handle: typed_operator.SAICTypedActionOperatorHandle,
        event: RuntimeEvent,
        task_prompts: Mapping[str, str],
        parallel: Any,
        action_arrow: typed_operator.SAICArrowCode,
        noop_arrow: typed_operator.SAICArrowCode,
    ) -> None:
        self.diffusion = diffusion
        self.transformer = transformer
        self.handle = handle
        self.event = event
        self.task_prompts = task_prompts
        self.parallel = parallel
        self.action_arrow = action_arrow
        self.noop_arrow = noop_arrow
        self.cache: dict[tuple[int, str], Any] = {}
        self.target_tail_sha256_by_source_role: dict[str, str] = {}
        self.route_receipts: list[Mapping[str, Any]] = []

    def _pack(self, request: scaid.NativeFieldRequest) -> Any:
        key = (id(request.coordinate), request.source_role)
        if key not in self.cache:
            if request.source_role == "wrong":
                video = self.event.wrong_source_latent
                refs = self.event.wrong_source_references
            else:
                video = self.event.source_latent
                refs = self.event.source_references
            self.cache[key] = native_runtime._build_pack(
                self.transformer, video, refs, request.coordinate.x_sigma
            )
            pack = self.cache[key]
            target_tails = {
                branch.name: branch.latents[:, branch.condition_tokens :, :]
                for branch in (pack.none, pack.video, pack.image, pack.video_image)
            }
            reference = target_tails["none"]
            if any(not reference.equal(value) for value in target_tails.values()):
                raise SAICTypedSourceCoordinateTrainingError(
                    "native none/V/I/VI target tails differ in content"
                )
            self.target_tail_sha256_by_source_role[request.source_role] = (
                distributed_runtime.tensor_sha256(reference.detach().float())
            )
        return self.cache[key]

    def _forward(
        self,
        branch: Any,
        *,
        coordinate: scaid.SourceCoordinate,
        text: Any,
        action_enabled: bool,
    ) -> Any:
        arrow = self.action_arrow if action_enabled else self.noop_arrow
        route = bind_native_action_route(
            handle=self.handle,
            native_branch=branch,
            actual_sigma=coordinate.sigma,
            arrow=arrow,
            parallel=self.parallel,
        )
        self.route_receipts.append(dict(route.receipt()))
        with self.handle.route(route):
            return native.forward_native_target_branch(
                self.diffusion,
                branch,
                timestep=coordinate.timestep,
                cond_embeds=text,
            )

    def __call__(self, request: scaid.NativeFieldRequest) -> Any:
        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise SAICTypedSourceCoordinateTrainingError(
                "native callback prompt differs from encoded RV2V task prompt"
            )
        pack = self._pack(request)
        coordinate = request.coordinate
        if request.phase == "frozen_native_reference_identity_control_dI":
            components = []
            for branch in (pack.video, pack.video_image):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    packed = self._forward(
                        branch,
                        coordinate=coordinate,
                        text=self.event.unconditional,
                        action_enabled=False,
                    )
                components.append(
                    native_bridge._unpack_spatial_velocity(
                        packed, video_shape=coordinate.x_sigma.shape
                    ).float()
                )
            return components[1] - components[0]
        condition = self.event.native_conditions[request.branch]
        rows = native_runtime._native_rows(
            pack, cond_embeds=condition, uncond_embeds=self.event.unconditional
        )
        if tuple(
            (name, float(coefficient))
            for name, _branch, _text, coefficient in rows
        ) != scaid.NATIVE_GUIDANCE_COMPONENTS:
            raise SAICTypedSourceCoordinateTrainingError(
                "native CFG order/coefficients differ"
            )
        components: dict[str, Any] = {}
        for name, branch, text, _coefficient in rows:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                packed = self._forward(
                    branch,
                    coordinate=coordinate,
                    text=text,
                    action_enabled=request.adapter_enabled,
                )
            components[name] = native_bridge._unpack_spatial_velocity(
                packed, video_shape=coordinate.x_sigma.shape
            )
        return scaid.aggregate_native_guidance_components(components)

    def replay_component(
        self, request: scaid.NativeFieldRequest, component_name: str
    ) -> Any:
        """Rebuild exactly one graph-bearing native CFG leaf for serial VJP."""

        import torch

        if request.prompt != self.task_prompts.get(request.branch):
            raise SAICTypedSourceCoordinateTrainingError(
                "native VJP replay prompt differs"
            )
        if request.phase != "native_student_component_serial_vjp_replay":
            raise SAICTypedSourceCoordinateTrainingError(
                "native VJP replay phase differs"
            )
        pack = self._pack(request)
        rows = native_runtime._native_rows(
            pack,
            cond_embeds=self.event.native_conditions[request.branch],
            uncond_embeds=self.event.unconditional,
        )
        if tuple(
            (name, float(coefficient))
            for name, _branch, _text, coefficient in rows
        ) != scaid.NATIVE_GUIDANCE_COMPONENTS:
            raise SAICTypedSourceCoordinateTrainingError(
                "native VJP CFG order/coefficients differ"
            )
        by_name = {
            name: (branch, text) for name, branch, text, _coefficient in rows
        }
        if component_name not in by_name:
            raise SAICTypedSourceCoordinateTrainingError(
                "native VJP component differs"
            )
        branch, text = by_name[component_name]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed = self._forward(
                branch,
                coordinate=request.coordinate,
                text=text,
                action_enabled=True,
            )
        return native_bridge._unpack_spatial_velocity(
            packed, video_shape=request.coordinate.x_sigma.shape
        )


def _fresh_epsilon(shape: Sequence[int], *, seed: int, device: Any) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        tuple(shape), generator=generator, dtype=torch.float32
    ).to(device).detach()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checked = preflight(args)
    manifest = checked.manifest
    gates = checked.authorizations
    checkpoint_identity = checked.checkpoint_identity
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_only": True,
                    "optimizer_authorized": True,
                    "topology": "WORLD8/DP2xSP4",
                    "frame_count": FRAME_COUNT,
                    "fit_source_iids": list(FIT_SAMPLE_IDS),
                    "arrow_receipts": [
                        dict(action_arrow_for_dp_arm(index).receipt())
                        for index in range(DP_SIZE)
                    ],
                    "event_gate_present": False,
                    "inverse_cycle_present": False,
                    "semantic_action_editing_success_claimed": False,
                    "checkpoint_content_receipt_digest": object_sha256(
                        checkpoint_identity
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    try:
        bernini_root, veomni_root, _, _ = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(str(error)) from error
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    contract = distributed_runtime.distributed_contract()
    device = distributed_runtime.initialise_distributed(contract)
    parallel = distributed_runtime.validate_parallel_state(
        contract, init_parallel_state(ulysses_size=SP_SIZE)
    )
    output, stage = distributed_runtime.prepare_output_transaction(
        args.output, contract.rank, parallel.world_group
    )
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval()
    diffusion = renderer.diff_dec
    transformer = renderer.diff_dec.transformer
    cagd_runtime._disable_gradient_checkpointing(renderer, transformer)
    handle = typed_operator.install_saic_typed_action_operator(transformer)
    trainable = handle.trainable_named_parameters()
    if not handle.base_parameters_frozen():
        raise SAICTypedSourceCoordinateTrainingError(
            "typed operator/base trainability closure differs"
        )

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    dp_arm = contract.arm_index
    if dp_arm not in (0, 1):
        raise SAICTypedSourceCoordinateTrainingError("runtime DP arm differs")
    spec = manifest.events[dp_arm]
    gate = gates[dp_arm]
    source, refs, source_preprocessing_receipt = baseline._encode_video(
        spec.source_video,
        spec.source_video_sha256,
        vae=vae,
        device=device,
        parallel=parallel,
    )
    wrong, wrong_refs, wrong_preprocessing_receipt = baseline._encode_video(
        spec.wrong_source_video,
        spec.wrong_source_video_sha256,
        vae=vae,
        device=device,
        parallel=parallel,
    )
    condition_geometry = validate_native_condition_geometry(
        source, refs, wrong, wrong_refs
    )
    del vae
    torch.cuda.empty_cache()
    renderer.to(device).eval()
    initial_digest = distributed_runtime.synchronize_initial_parameters(
        trainable, parallel.world_group
    )
    (
        t2v_conditions,
        native_conditions,
        unconditional,
        prompt_construction_receipt,
        t2v_task_prompts,
        rv2v_task_prompts,
    ) = baseline._encode_conditions(
        renderer,
        tokenizer,
        spec.raw_caption_by_branch,
        device=device,
        parallel=parallel,
    )
    del tokenizer
    event = RuntimeEvent(
        spec,
        gate,
        source,
        refs,
        wrong,
        wrong_refs,
        t2v_conditions,
        native_conditions,
        unconditional,
        prompt_construction_receipt,
    )
    action_arrow = action_arrow_for_dp_arm(dp_arm)
    noop_arrow = noop_arrow_for_dp_arm(dp_arm)
    scaid_config = scaid.SCAIDConfig()
    trainable_by_sigma = {
        stratum: handle.trainable_named_parameters_for_sigma(stratum)
        for stratum in ("high", "mid")
    }
    if (
        set(id(parameter) for _, parameter in trainable_by_sigma["high"])
        & set(id(parameter) for _, parameter in trainable_by_sigma["mid"])
        or set(id(parameter) for _, parameter in trainable)
        != {
            id(parameter)
            for stratum in ("high", "mid")
            for _, parameter in trainable_by_sigma[stratum]
        }
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "typed sigma optimizer partitions are not a disjoint exact cover"
        )
    optimizers = {
        stratum: torch.optim.AdamW(
            [parameter for _, parameter in trainable_by_sigma[stratum]],
            lr=args.learning_rate,
            weight_decay=0.0,
        )
        for stratum in ("high", "mid")
    }
    schedule_indices = (
        (typed_operator.HIGH_SIGMA_INDICES[0],)
        if args.smoke_high_sigma_only
        else tuple(
            exact40_schedule_index(step)
            for step in range(args.max_schedule_steps)
        )
    )
    history: list[Mapping[str, Any]] = []
    updates = 0
    for step, index in enumerate(schedule_indices):
        seed_value = noise_seed(seed=args.seed, step=step, dp_arm=dp_arm)
        epsilon = _fresh_epsilon(source.shape, seed=seed_value, device=device)
        baseline._broadcast_sp(epsilon, parallel=parallel)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        before = distributed_runtime.trainable_parameters_digest(trainable)
        t2v_callback = FrozenT2VCallback(
            diffusion, transformer, t2v_conditions, t2v_task_prompts
        )
        native_callback = NativeTypedSCAIDCallback(
            diffusion,
            transformer,
            handle,
            event,
            rv2v_task_prompts,
            parallel,
            action_arrow,
            noop_arrow,
        )
        cell = scaid.run_scaid_cell(
            source,
            epsilon,
            schedule_index=index,
            authoritative_evidence_path=gate.evidence_path,
            expected_authoritative_evidence_sha256=gate.evidence_file_sha256,
            fit_candidate_id=gate.fit_candidate_id,
            raw_caption_by_branch=spec.raw_caption_by_branch,
            expected_raw_caption_bank_sha256=spec.raw_caption_bank_sha256,
            checkpoint_tree_sha256=manifest.checkpoint_tree_sha256,
            frozen_t2v_callback=t2v_callback,
            native_callback=native_callback,
            config=scaid_config,
            leaf_vjp_mode=True,
        )
        if index in typed_operator.LOW_SIGMA_INDICES:
            if not cell.zero_update or any(
                parameter.grad is not None for _, parameter in trainable
            ):
                raise SAICTypedSourceCoordinateTrainingError(
                    "low sigma constructed an update or gradient"
                )
            after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"typed low anchor {step}"
            )
            if before != after:
                raise SAICTypedSourceCoordinateTrainingError(
                    "low sigma changed the typed operator"
                )
            record: Mapping[str, Any] = {
                "step": step,
                "index": index,
                "noise_seed": seed_value,
                "optimizer_step": False,
                "loss": None,
                "cell_receipt": dict(cell.receipt),
            }
        else:
            if cell.objective is None or not cell.optimizer_authorized:
                raise SAICTypedSourceCoordinateTrainingError(
                    "trainable SCAID cell was not authorized"
                )
            sigma_stratum = cell.coordinate.gate_name
            if sigma_stratum not in trainable_by_sigma:
                raise SAICTypedSourceCoordinateTrainingError(
                    "trainable cell did not bind one optimizer sigma partition"
                )
            active_trainable = trainable_by_sigma[sigma_stratum]
            inactive_stratum = "mid" if sigma_stratum == "high" else "high"
            inactive_before = distributed_runtime.trainable_parameters_digest(
                trainable_by_sigma[inactive_stratum]
            )
            target_tail_receipt = target_tail_equality_receipt(
                t2v_callback, native_callback
            )
            cell.objective.loss.backward()
            replay = scaid.replay_native_student_vjp(cell, native_callback)
            grad_norm = distributed_runtime.synchronize_gradients(
                active_trainable, parallel
            )
            clipped = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in active_trainable],
                args.max_grad_norm,
            )
            if not math.isfinite(float(clipped)):
                raise SAICTypedSourceCoordinateTrainingError(
                    "typed operator gradient is non-finite"
                )
            optimizers[sigma_stratum].step()
            inactive_after = distributed_runtime.trainable_parameters_digest(
                trainable_by_sigma[inactive_stratum]
            )
            if inactive_before != inactive_after:
                raise SAICTypedSourceCoordinateTrainingError(
                    "inactive sigma parameter partition changed"
                )
            updates += 1
            after = distributed_runtime.parameter_consensus(
                trainable, parallel.world_group, f"typed update {updates}"
            )
            record = {
                "step": step,
                "index": index,
                "noise_seed": seed_value,
                "optimizer_step": True,
                "optimizer_sigma_partition": sigma_stratum,
                "inactive_sigma_partition": inactive_stratum,
                "inactive_sigma_partition_unchanged": True,
                "loss": float(cell.objective.loss.detach().item()),
                "loss_components": {
                    "action_match": float(
                        cell.objective.action_match_loss.detach().item()
                    ),
                    "negative_parity": float(
                        cell.objective.negative_parity_loss.detach().item()
                    ),
                    "action_base_trust": float(
                        cell.objective.action_base_trust_loss.detach().item()
                    ),
                },
                "vjp_replay_max_abs": max(replay.values()),
                "preclip_gradient_norm": grad_norm,
                "parameter_digest": after,
                "target_tail_equality_receipt": target_tail_receipt,
                "route_call_count": len(native_callback.route_receipts),
                "action_arrow_digest": action_arrow.receipt()["digest"],
                "cell_receipt": dict(cell.receipt),
            }
        distributed_runtime.digest_consensus(
            object_sha256(record),
            group=parallel.sp_group,
            expected_count=SP_SIZE,
            label=f"typed SCAID step {step}",
        )
        gathered: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, record, group=parallel.world_group)
        if contract.rank == 0:
            history.append(
                {
                    "step": step,
                    "index": index,
                    "dp_records": [gathered[0], gathered[4]],
                }
            )
        del epsilon, t2v_callback, native_callback, cell
        torch.cuda.empty_cache()

    expected_updates = (
        1
        if args.smoke_high_sigma_only
        else expected_optimizer_updates(args.max_schedule_steps)
    )
    if updates != expected_updates:
        raise SAICTypedSourceCoordinateTrainingError(
            "typed exact40 optimizer count differs"
        )
    final_digest = distributed_runtime.parameter_consensus(
        trainable, parallel.world_group, "typed final"
    )
    if final_digest == initial_digest:
        raise SAICTypedSourceCoordinateTrainingError(
            "typed pilot did not change its operator"
        )
    manifest.assert_unchanged()
    try:
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
    except Exception as error:
        raise SAICTypedSourceCoordinateTrainingError(
            f"official source trees changed during training: {error}"
        ) from error
    if (
        file_sha256(Path(args.checkpoint_content_manifest))
        != args.expected_checkpoint_content_manifest_sha256
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "checkpoint manifest changed during training"
        )
    final_checkpoint_identity = source_audit.validate_checkpoint_content(
        Path(args.checkpoint),
        Path(args.checkpoint_content_manifest),
        expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
    )
    if object_sha256(final_checkpoint_identity) != object_sha256(
        checkpoint_identity
    ):
        raise SAICTypedSourceCoordinateTrainingError(
            "checkpoint content changed during training"
        )

    local_runtime_provenance = {
        "dp_arm": dp_arm,
        "sample_id": spec.sample_id,
        "action_family": spec.action_family,
        "source_video_sha256": spec.source_video_sha256,
        "wrong_source_video_sha256": spec.wrong_source_video_sha256,
        "native_condition_geometry": condition_geometry,
        "source_preprocessing_receipt": source_preprocessing_receipt,
        "wrong_source_preprocessing_receipt": wrong_source_preprocessing_receipt,
        "prompt_construction_receipt": prompt_construction_receipt,
        "action_arrow": dict(action_arrow.receipt()),
        "noop_arrow": dict(noop_arrow.receipt()),
    }
    gathered_runtime: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(
        gathered_runtime, local_runtime_provenance, group=parallel.world_group
    )
    dist.barrier(group=parallel.world_group)
    if contract.rank == 0:
        checkpoint_receipt = handle.save_checkpoint(stage / "operator.pt")
        history_path = stage / "history.json"
        distributed_runtime.atomic_json(
            history_path,
            {
                "schema_version": HISTORY_SCHEMA,
                "records": history,
                "optimizer_updates": updates,
            },
        )
        artifacts = {
            "operator.pt": file_sha256(stage / "operator.pt"),
            "history.json": file_sha256(history_path),
        }
        receipt_value = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "artifact_complete": True,
            "complete_saic_method": False,
            "run_kind": (
                "single_cell_high_sigma_smoke"
                if args.smoke_high_sigma_only
                else "typed_source_coordinate_exact40_pilot"
            ),
            "topology": "WORLD8/DP2xSP4",
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "schedule_indices": list(schedule_indices),
            "optimizer_updates": updates,
            "low_sigma_indices_exact_base": list(
                typed_operator.LOW_SIGMA_INDICES
            ),
            "training_config": {
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "max_grad_norm": args.max_grad_norm,
                "gradient_checkpointing_enabled": False,
                "leaf_vjp_mode": True,
                "serial_native_cfg_leaf_replay": True,
                "typed_operator_blocks": list(
                    typed_operator.ACTION_BLOCK_INDICES
                ),
                "typed_operator_rank": typed_operator.ACTION_OPERATOR_RANK,
                "arrow_dimension": typed_operator.ARROW_CODE_DIM,
                "sigma_parameter_partitions": ["high", "mid"],
                "optimizer_instances": 2,
                "optimizer_step_scope": "active_sigma_partition_only",
                "cross_sigma_optimizer_momentum": False,
            },
            "fit_source_iids": list(FIT_SAMPLE_IDS),
            "dp_runtime_provenance": [gathered_runtime[0], gathered_runtime[4]],
            "checkpoint": {
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_manifest_sha256": (
                    args.expected_checkpoint_content_manifest_sha256
                ),
                "content_receipt_digest": object_sha256(checkpoint_identity),
            },
            "method_source": {
                "revision": args.method_source_revision,
                "archive_sha256": args.method_source_archive_sha256,
            },
            "training_manifest": {
                "file_sha256": manifest.file_sha256,
                "manifest_digest": manifest.manifest_digest,
            },
            "authoritative_evidence": {
                "file_sha256": gates[0].evidence_file_sha256,
                "authorization_digests": [
                    gate_item.authorization_digest for gate_item in gates
                ],
            },
            "operator_checkpoint": {
                key: value
                for key, value in checkpoint_receipt.items()
                if key != "path"
            },
            "artifacts": artifacts,
            "initial_parameter_digest": initial_digest,
            "final_parameter_digest": final_digest,
            "event_gate_present": False,
            "inverse_cycle_present": False,
            "pure_t2v_generated_visual_consumed": False,
            "semantic_action_editing_success_claimed": False,
        }
        receipt = {
            **receipt_value,
            "receipt_digest": object_sha256(receipt_value),
        }
        distributed_runtime.atomic_json(stage / "receipt.json", receipt)
        os.replace(stage, output)
        distributed_runtime.fsync_directory(output.parent)
    dist.barrier(group=parallel.world_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARROW_STATE_TYPES",
    "FIT_SAMPLE_IDS",
    "SAICTypedSourceCoordinateTrainingError",
    "TypedPilotPreflight",
    "action_arrow_for_dp_arm",
    "actual_local_target_mask",
    "bind_native_action_route",
    "build_parser",
    "exact40_schedule_index",
    "expected_optimizer_updates",
    "noop_arrow_for_dp_arm",
    "preflight",
]
