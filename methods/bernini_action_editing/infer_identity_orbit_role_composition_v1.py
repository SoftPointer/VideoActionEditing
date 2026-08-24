#!/usr/bin/env python3
"""Held-out exact81 decode of Bernini's counterfactual identity-orbit adapter.

This is a read-only composition experiment, not another action-pair training
run.  A frozen Bernini T2V pass first creates one *fresh* ordered action donor
from the pre-registered source caption.  The donor is then supplied as native
``V`` evidence while four independently VAE-encoded RGB frames from the source
video are supplied as native ``I`` evidence.  Two native RV2V-4 rollouts use
the same prompt, official Gaussian seed, scheduler, guidance and conditions:

``base``
    The installed target-row adapter route is explicitly disabled.

``orbit-adapter``
    The strictly reloaded v5 target-row Q/O adapter is active on every one of
    the 40 native UniPC coordinates.

The T2V donor is generated inside this invocation and is never a target,
noise, pseudo-label or optimizer input.  This avoids retroactively changing
the sealed use contract of older calibration-bank artifacts.  No mask, flow,
pose, track, trajectory, source-rich noise or external target is accepted.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
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

import infer_native_identity_generation_canary as native  # noqa: E402
import infer_native_multivideo_motion_donor_oracle as output_runtime  # noqa: E402
import source_self_native_ref_contrastive_v3 as native_pack  # noqa: E402
import source_self_native_rv2v_guidance as native_guidance  # noqa: E402
import source_self_native_target_adapter as target_adapter  # noqa: E402
import source_self_runtime as source_runtime  # noqa: E402
import train_source_self_identity_orbit_v4 as orbit_trainer  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


SCHEMA_VERSION = "bernini-identity-orbit-heldout-role-composition-receipt-v1"
SPEC_SCHEMA = "bernini-identity-orbit-heldout-role-composition-core2-spec-v1"
METHOD = "bernini-identity-orbit-heldout-role-composition-v1"
FRAME_COUNT = 81
REFERENCE_INDICES = (0, 27, 53, 80)
LATENT_PHASES = 21
SP_SIZE = 4
ALLOWED_STEPS = (1, 40)
ARM_ORDER = ("base", "orbit-adapter")
EXPECTED_ADAPTER_SCHEMA = "bernini-native-target-row-qo-lora-checkpoint-v2"
EXPECTED_TRAINING_RECEIPT_SCHEMA = (
    "bernini-counterfactual-identity-orbit-training-receipt-v5"
)
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class RoleCompositionError(RuntimeError):
    """Raised before ambiguous role-composition evidence is published."""


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
        raise RoleCompositionError(f"value is not finite canonical ASCII JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _rank_invariant_route_payload(
    route_traces: Mapping[str, Any], *, expected_rank: int
) -> Mapping[str, Any]:
    """Validate rank-local route receipts and return their shared SP payload.

    ``NativeTargetRoute.receipt()`` deliberately records the local Ulysses
    rank, so hashing the raw trace and requiring equality across SP ranks is
    invalid.  Before removing that one legitimate rank-local field (and the
    digests derived from it), validate both the nested receipt digest and that
    every route names the caller's exact SP rank.  All other route evidence is
    retained for the cross-rank equality check.
    """

    if isinstance(expected_rank, bool) or not 0 <= int(expected_rank) < SP_SIZE:
        raise RoleCompositionError("expected route-audit SP rank lies outside WORLD4")
    normalized = json.loads(canonical_json_bytes(route_traces).decode("ascii"))
    if not isinstance(normalized, dict) or set(normalized) != set(ARM_ORDER):
        raise RoleCompositionError("route audit arms differ before rank normalization")
    for arm in ARM_ORDER:
        trace = normalized.get(arm)
        if not isinstance(trace, dict):
            raise RoleCompositionError(f"{arm} route trace is not a mapping")
        step_records = trace.get("step_records")
        if (
            not isinstance(step_records, list)
            or len(step_records) != trace.get("step_count")
            or trace.get("step_records_digest") != object_sha256(step_records)
        ):
            raise RoleCompositionError(f"{arm} rank-local step receipt digest differs")
        for step in step_records:
            routes = step.get("routes") if isinstance(step, dict) else None
            if not isinstance(routes, list) or len(routes) != 4:
                raise RoleCompositionError(f"{arm} rank-local step route closure differs")
            for route in routes:
                if not isinstance(route, dict):
                    raise RoleCompositionError(f"{arm} rank-local route is not a mapping")
                receipt_digest = route.pop("digest", None)
                if receipt_digest != object_sha256(route):
                    raise RoleCompositionError(f"{arm} rank-local route receipt digest differs")
                local_rank = route.pop("sequence_parallel_rank", None)
                if local_rank != int(expected_rank):
                    raise RoleCompositionError(
                        f"{arm} route receipt does not belong to expected SP rank"
                    )
                if route.get("sequence_parallel_size") != SP_SIZE:
                    raise RoleCompositionError(f"{arm} route receipt is not native SP4")
        # This digest was verified above and is rank-dependent only because it
        # covers the nested rank-local route receipts.
        trace.pop("step_records_digest")
    return normalized


def file_sha256(path: Path) -> str:
    return source_runtime.file_sha256(path)


def _sha(value: Any, *, length: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1 if length == 40 else _SHA256
    if pattern.fullmatch(text) is None:
        raise RoleCompositionError(f"{label} must be lowercase SHA-{1 if length == 40 else 256}")
    return text


def _plain_file(path: str | Path, *, label: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise RoleCompositionError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_file() or resolved.is_symlink():
        raise RoleCompositionError(f"{label} must be a canonical plain file")
    return resolved


def load_cell_spec(
    path: str | Path, *, expected_file_sha256: str, cell_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, str]:
    """Load one pre-registered held-out cell without quality-based selection."""

    spec_path = _plain_file(path, label="core2 spec")
    observed_sha = file_sha256(spec_path)
    if observed_sha != _sha(expected_file_sha256, length=64, label="spec file SHA-256"):
        raise RoleCompositionError("core2 spec file SHA-256 differs")
    try:
        root = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RoleCompositionError("core2 spec is not valid JSON") from error
    if not isinstance(root, dict) or root.get("schema_version") != SPEC_SCHEMA:
        raise RoleCompositionError("core2 spec schema differs")
    cells = root.get("cells")
    if not isinstance(cells, list) or len(cells) != 2:
        raise RoleCompositionError("core2 spec must contain exactly dog/human cells")
    ids = [row.get("cell_id") for row in cells if isinstance(row, Mapping)]
    if ids != ["dog-fit", "human-fit"] or len(set(ids)) != 2:
        raise RoleCompositionError("core2 cell order/identity differs")
    selected = next((row for row in cells if row.get("cell_id") == cell_id), None)
    if not isinstance(selected, Mapping):
        raise RoleCompositionError("requested held-out cell is absent")
    required = {
        "cell_id",
        "actor_kind",
        "source_video",
        "source_video_sha256",
        "action_caption",
        "action_caption_utf8_sha256",
        "donor_seed",
        "target_seed",
        "identity_orbit_training_iid_overlap",
        "donor_policy",
    }
    if set(selected) != required:
        raise RoleCompositionError("held-out cell field closure differs")
    if selected["actor_kind"] not in {"dog", "human"}:
        raise RoleCompositionError("held-out actor kind differs")
    if selected["identity_orbit_training_iid_overlap"] is not False:
        raise RoleCompositionError("held-out cell overlaps identity-orbit training IID")
    if selected["donor_policy"] != "fresh_frozen_t2v_in_same_invocation_condition_only":
        raise RoleCompositionError("donor policy permits an external or selected proposal")
    source = _plain_file(selected["source_video"], label=f"{cell_id} source video")
    if file_sha256(source) != _sha(
        selected["source_video_sha256"], length=64, label="source SHA-256"
    ):
        raise RoleCompositionError("held-out source SHA-256 differs")
    caption = selected["action_caption"]
    if not isinstance(caption, str) or not caption.strip() or "\x00" in caption:
        raise RoleCompositionError("action caption is empty or contains NUL")
    if hashlib.sha256(caption.encode("utf-8")).hexdigest() != _sha(
        selected["action_caption_utf8_sha256"], length=64, label="caption SHA-256"
    ):
        raise RoleCompositionError("action caption SHA-256 differs")
    for name in ("donor_seed", "target_seed"):
        value = selected[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            raise RoleCompositionError(f"{name} lies outside [0,2^63)")
    if selected["donor_seed"] == selected["target_seed"]:
        raise RoleCompositionError("donor and renderer Gaussian seeds must be independent")
    global_contract = root.get("contract")
    expected_contract = {
        "frame_count": FRAME_COUNT,
        "fps": 25,
        "latent_phases": LATENT_PHASES,
        "num_inference_steps": 40,
        "reference_indices": list(REFERENCE_INDICES),
        "topology": "two_isolated_world4_on_one_8gpu_node",
        "source_rich_noise_rho": 0.0,
        "external_donor_media": False,
        "target_video": False,
        "mask_flow_pose_track_trajectory": False,
        "cell_selection_uses_generated_quality": False,
    }
    if global_contract != expected_contract:
        raise RoleCompositionError("core2 global contract differs")
    return root, selected, spec_path, observed_sha


def expected_rv2v_layout(shape: Sequence[int]) -> Mapping[str, Any]:
    """Return the exact native one-video/four-ref visual token layout."""

    dims = tuple(int(item) for item in shape)
    if len(dims) != 5 or dims[:3] != (1, 16, LATENT_PHASES):
        raise RoleCompositionError("video latent must be exact81 [1,16,21,H,W]")
    height, width = dims[-2:]
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise RoleCompositionError("latent H/W must be positive and even")
    target = LATENT_PHASES * (height // 2) * (width // 2)
    reference = (height // 2) * (width // 2)
    shared = [target, 2 * target, 2 * target + 4 * reference, 2 * target + 4 * reference]
    return {
        "video_shape": list(dims),
        "reference_shape": [1, 16, 1, height, width],
        "target_patch_tokens": target,
        "reference_patch_tokens": reference,
        "shared_visual_token_lengths": shared,
        "branch_names": ["none", "V", "VI", "VI"],
        "condition_tokens": [value - target for value in shared],
        "patch_source_ids": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0],
        "native_rv2v4_reference_contract_digest": native_pack.native_rv2v4_reference_contract()[
            "digest"
        ],
    }


def validate_adapter_metadata(
    metadata: Mapping[str, str], *, adapter_contract_digest: str
) -> None:
    expected = {
        "schema_version": EXPECTED_ADAPTER_SCHEMA,
        "adapter_contract_digest": adapter_contract_digest,
        "block_indices_json": "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]",
        "rank": "8",
        "alpha_hex": float(8.0).hex(),
        "rho_hex": float(0.0).hex(),
        "native_guidance_digest": native_guidance.guidance_receipt()["digest"],
        "native_schedule_digest": native_pack.native_unipc40_schedule_receipt()["digest"],
        "native_rv2v4_reference_contract_digest": native_pack.native_rv2v4_reference_contract()[
            "digest"
        ],
        "reference_rgb_indices_json": "[0,27,53,80]",
        "gradient_checkpointing_enabled": "false",
        "adapter_activation_schedule": "all_40_native_unipc_forward_coordinates",
        "inference_requires_same_target_route_and_rho": "true",
    }
    if dict(metadata) != expected:
        raise RoleCompositionError("adapter safetensors metadata differs from v5 train contract")


def validate_adapter_publication(
    *,
    adapter_path: str | Path,
    receipt_path: str | Path,
    expected_adapter_sha256: str,
    expected_receipt_sha256: str,
) -> Mapping[str, Any]:
    adapter = _plain_file(adapter_path, label="published adapter")
    receipt_file = _plain_file(receipt_path, label="training receipt")
    adapter_sha = file_sha256(adapter)
    receipt_sha = file_sha256(receipt_file)
    if adapter_sha != _sha(expected_adapter_sha256, length=64, label="adapter SHA-256"):
        raise RoleCompositionError("published adapter SHA-256 differs")
    if receipt_sha != _sha(expected_receipt_sha256, length=64, label="training receipt SHA-256"):
        raise RoleCompositionError("training receipt SHA-256 differs")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RoleCompositionError("training receipt is not valid JSON") from error
    adapter_row = receipt.get("adapter", {})
    roundtrip = adapter_row.get("strict_real_file_roundtrip", {})
    model = receipt.get("model", {})
    if (
        receipt.get("schema_version") != EXPECTED_TRAINING_RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("mode") != "complete-cycle"
        or receipt.get("frame_count") != FRAME_COUNT
        or receipt.get("latent_phases") != LATENT_PHASES
        or receipt.get("next_heldout_role_composition_experiment_authorized") is not True
        or receipt.get("action_editing_claim_authorized") is not False
        or receipt.get("appearance_motion_role_learning_claim_only") is not True
        or receipt.get("objective", {}).get("adapter_activation_schedule_train_and_inference")
        != "all_40_native_unipc_forward_coordinates"
        or adapter_row.get("block_indices") != list(range(23))
        or adapter_row.get("rank") != 8
        or adapter_row.get("alpha") != 8.0
        or adapter_row.get("target_row_only") is not True
        or adapter_row.get("key_value_trainable") is not False
        or adapter_row.get("cross_attention_trainable") is not False
        or adapter_row.get("late_blocks_trainable") is not False
        or roundtrip.get("file_sha256") != adapter_sha
        or roundtrip.get("file_loaded_into_live_adapter") is not True
        or roundtrip.get("strict_key_shape_dtype_value_roundtrip") is not True
        or roundtrip.get("tensor_count") != 92
        or model.get("bernini_commit") != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or model.get("veomni_commit") != native.legacy.trainer.VEOMNI_TESTED_COMMIT
        or model.get("checkpoint_tree_sha256") != native.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise RoleCompositionError("published adapter receipt does not authorize this held-out decode")
    return {
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_sha,
        "training_receipt_path": str(receipt_file),
        "training_receipt_sha256": receipt_sha,
        "training_method_source_revision": receipt.get("method_source_revision"),
        "training_method_source_archive_sha256": receipt.get("method_source_archive_sha256"),
        "final_parameter_digest": adapter_row.get("final_parameter_digest"),
        "adapter_contract_digest": adapter_row.get("digest"),
        "metadata": dict(roundtrip.get("metadata", {})),
        "complete_cycle_steps": receipt.get("training", {}).get("optimizer_steps", 36),
        "heldout_decode_authorized": True,
        "semantic_action_claim_authorized": False,
    }


def strict_load_adapter(
    path: Path,
    handle: target_adapter.NativeTargetAdapterHandle,
    publication: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Strict file -> live wrappers load, followed by inference freezing."""

    import torch
    from safetensors import safe_open

    named = handle.trainable_named_parameters()
    expected = {name: parameter for name, parameter in named}
    contract = handle.receipt()
    if contract["digest"] != publication["adapter_contract_digest"]:
        raise RoleCompositionError("installed target-row adapter contract differs from training")
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        metadata = dict(opened.metadata() or {})
        keys = list(opened.keys())
        tensors = {key: opened.get_tensor(key).contiguous() for key in keys}
    validate_adapter_metadata(metadata, adapter_contract_digest=contract["digest"])
    if metadata != publication["metadata"] or set(keys) != set(expected) or len(keys) != 92:
        raise RoleCompositionError("adapter key/metadata closure differs")
    for name, parameter in expected.items():
        value = tensors[name]
        if (
            value.dtype != torch.float32
            or tuple(value.shape) != tuple(parameter.shape)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RoleCompositionError(f"adapter tensor differs: {name}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    digest = source_runtime.trainable_parameters_digest(named)
    if digest != publication["final_parameter_digest"]:
        raise RoleCompositionError("live adapter parameter digest differs from training receipt")
    for _, parameter in named:
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in handle.transformer.parameters()):
        raise RoleCompositionError("inference transformer remains trainable")
    return {
        "file_sha256": publication["adapter_sha256"],
        "tensor_count": len(named),
        "live_parameter_digest": digest,
        "contract": contract,
        "metadata": metadata,
        "strict_file_to_live_load": True,
        "all_parameters_frozen_after_load": True,
    }


@dataclass
class _ActiveAudit:
    completed_steps: int = 0
    patch_source_ids: list[float] = field(default_factory=list)
    shared_lengths: list[int] = field(default_factory=list)
    shared_visual_objects: list[Any] = field(default_factory=list)
    shared_rotary_objects: list[Any] = field(default_factory=list)
    route_records: list[Mapping[str, Any]] = field(default_factory=list)
    step_records: list[Mapping[str, Any]] = field(default_factory=list)


class RV2VTargetRouteAudit:
    """Reversibly audit and activate the exact target-row route per branch."""

    def __init__(
        self,
        diffusion: Any,
        *,
        handle: target_adapter.NativeTargetAdapterHandle,
        adapter_enabled: bool,
        layout: Mapping[str, Any],
        donor_condition: Any,
        image_references: Sequence[Any],
        prompt_embeds: Any,
        uncond_embeds: Any,
        expected_steps: int,
        target_seed: int,
        sp_rank: int,
    ) -> None:
        self.diffusion = sampler_contract.resolve_diffusion_core(diffusion)
        self.transformer = self.diffusion.transformer
        self.scheduler = self.diffusion.scheduler
        self.handle = handle
        self.adapter_enabled = bool(adapter_enabled)
        self.layout = dict(layout)
        self.donor_list = [donor_condition]
        self.reference_list = list(image_references)
        self.prompt_embeds = prompt_embeds
        self.uncond_embeds = uncond_embeds
        self.expected_steps = int(expected_steps)
        self.target_seed = int(target_seed)
        self.sp_rank = int(sp_rank)
        self._original_sample = self.diffusion.sample
        self._original_shared = self.diffusion.shared_step
        self._original_patch = self.transformer.patch_vae_latent
        self._original_scheduler = self.scheduler.step
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._active: Optional[_ActiveAudit] = None
        self.trace: dict[str, Any] = {}
        self.restored = False
        if (
            self.expected_steps not in ALLOWED_STEPS
            or len(self.reference_list) != 4
            or not 0 <= self.sp_rank < SP_SIZE
            or self.handle.transformer is not self.transformer
            or target_adapter.active_route() is not None
        ):
            raise RoleCompositionError("target-route audit construction differs")
        for owner, name in (
            (self.diffusion, "sample"),
            (self.diffusion, "shared_step"),
            (self.transformer, "patch_vae_latent"),
            (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise RoleCompositionError(f"refusing stacked runtime hook on {name}")

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had, previous))

    def install(self) -> None:
        if self._patches:
            raise RoleCompositionError("target-route audit already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler(*args, **kwargs)

        try:
            self._set_patch(self.transformer, "patch_vae_latent", patch_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had, previous = self._patches.pop()
            try:
                if had:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:  # pragma: no cover - catastrophic runtime failure.
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise RoleCompositionError("failed to restore target-route hooks") from errors[0]

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.trace:
            raise RoleCompositionError("target-route audit permits exactly one sample")
        values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        expected = native.native_sampling_contract(
            "rv2v", steps=self.expected_steps, seed=self.target_seed
        )
        for name, wanted in expected.items():
            observed = values.get(name)
            if isinstance(wanted, tuple):
                observed = tuple(observed)
            if observed != wanted:
                raise RoleCompositionError(f"native RV2V sample argument differs: {name}")
        if (
            values.get("multi_video_vae_latents") is not self.donor_list
            or values.get("multi_image_vae_latents") is not self.reference_list
            or values.get("image_vae_latents") is not None
            or values.get("prompt_embeds") is not self.prompt_embeds
            or values.get("uncond_prompt_embeds") is not self.uncond_embeds
        ):
            raise RoleCompositionError("native donor/ref/text object identity differs")
        state = _ActiveAudit()
        self._active = state
        try:
            result = self._original_sample(*args, **kwargs)
            if (
                state.completed_steps != self.expected_steps
                or state.patch_source_ids
                or state.shared_lengths
                or len(state.step_records) != self.expected_steps
            ):
                raise RoleCompositionError("native RV2V sample ended with incomplete route audit")
            routes = [
                route
                for step in state.step_records
                for route in step.get("routes", ())
            ]
            if (
                len(routes) != self.expected_steps * 4
                or any(route.get("enabled") is not self.adapter_enabled for route in routes)
                or any(
                    route.get("target_tokens") != self.layout["target_patch_tokens"]
                    for route in routes
                )
            ):
                raise RoleCompositionError("all-coordinate target-row route closure differs")
            self.trace = {
                "sample_calls": 1,
                "step_count": self.expected_steps,
                "guidance_forward_count": self.expected_steps * 4,
                "adapter_route_enabled": self.adapter_enabled,
                "adapter_active_on_every_native_coordinate": self.adapter_enabled,
                "adapter_explicitly_disabled_on_every_native_coordinate": not self.adapter_enabled,
                "branch_order": list(self.layout["branch_names"]),
                "shared_visual_token_lengths": list(self.layout["shared_visual_token_lengths"]),
                "target_patch_tokens": self.layout["target_patch_tokens"],
                "source_ids_per_step": list(self.layout["patch_source_ids"]),
                "step_records": list(state.step_records),
                "step_records_digest": object_sha256(state.step_records),
                "same_target_suffix_route_as_training": True,
                "gradient_checkpointing_enabled": False,
            }
            return result
        finally:
            self._active = None

    def _wrapped_patch(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleCompositionError("patch_vae_latent ran outside audited sample")
        values = sampler_contract._bind_call(self._original_patch, args, kwargs)
        expected = self.layout["patch_source_ids"]
        index = len(state.patch_source_ids)
        if index >= len(expected):
            raise RoleCompositionError("too many native patch calls")
        source_id = float(values.get("source_id"))
        if not math.isclose(source_id, float(expected[index]), rel_tol=0.0, abs_tol=0.0):
            raise RoleCompositionError("native patch source-id order differs")
        result = self._original_patch(*args, **kwargs)
        state.patch_source_ids.append(source_id)
        return result

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleCompositionError("shared_step ran outside audited sample")
        values = sampler_contract._bind_call(self._original_shared, args, kwargs)
        index = len(state.shared_lengths)
        expected_lengths = self.layout["shared_visual_token_lengths"]
        if index >= 4:
            raise RoleCompositionError("too many native guidance forwards")
        lengths = values.get("batch_vae_seqlen")
        if not isinstance(lengths, (list, tuple)) or tuple(int(item) for item in lengths) != (
            int(expected_lengths[index]),
        ):
            raise RoleCompositionError("native shared visual-token length/order differs")
        expected_prompts = (
            self.uncond_embeds,
            self.uncond_embeds,
            self.uncond_embeds,
            self.prompt_embeds,
        )
        visual = values.get("noisy_latents")
        rotary = values.get("rotary_embs")
        if (
            values.get("cond_embeds") is not expected_prompts[index]
            or values.get("model_id") != "transformer_1"
            or visual is None
            or rotary is None
        ):
            raise RoleCompositionError("native RV2V forward order/model differs")
        if index == 3 and (
            visual is not state.shared_visual_objects[2]
            or rotary is not state.shared_rotary_objects[2]
        ):
            raise RoleCompositionError("VI uncond/cond visual object differs")
        route = target_adapter.NativeTargetRoute(
            total_tokens=int(expected_lengths[index]),
            condition_tokens=int(self.layout["condition_tokens"][index]),
            sequence_parallel_rank=self.sp_rank,
            sequence_parallel_size=SP_SIZE,
            branch_name=self.layout["branch_names"][index],
            enabled=self.adapter_enabled,
        )
        with self.handle.route(route):
            result = self._original_shared(*args, **kwargs)
        if target_adapter.active_route() is not None:
            raise RoleCompositionError("target-row route leaked past one native forward")
        state.shared_lengths.append(int(expected_lengths[index]))
        state.shared_visual_objects.append(visual)
        state.shared_rotary_objects.append(rotary)
        state.route_records.append(dict(route.receipt()))
        return result

    def _wrapped_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise RoleCompositionError("scheduler.step ran outside audited sample")
        if (
            state.patch_source_ids != self.layout["patch_source_ids"]
            or state.shared_lengths != self.layout["shared_visual_token_lengths"]
            or len(state.route_records) != 4
        ):
            raise RoleCompositionError("scheduler arrived before complete routed guidance")
        result = self._original_scheduler(*args, **kwargs)
        state.step_records.append(
            {
                "step_index": state.completed_steps,
                "patch_source_ids": list(state.patch_source_ids),
                "routes": list(state.route_records),
                "adapter_enabled": self.adapter_enabled,
                "scheduler_original_return_forwarded": True,
            }
        )
        state.completed_steps += 1
        state.patch_source_ids.clear()
        state.shared_lengths.clear()
        state.shared_visual_objects.clear()
        state.shared_rotary_objects.clear()
        state.route_records.clear()
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", required=True)
    parser.add_argument("--expected-cell-spec-sha256", required=True)
    parser.add_argument("--cell-id", required=True, choices=("dog-fit", "human-fit"))
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-training-receipt", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-adapter-training-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-inference-steps", type=int, choices=ALLOWED_STEPS, required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument(
        "--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=native.legacy.trainer.CHECKPOINT_TREE_SHA256,
    )
    return parser


def _disable_gradient_checkpointing(renderer: Any, transformer: Any) -> Mapping[str, Any]:
    disable = getattr(renderer, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    enabled = any(
        bool(getattr(owner, name, False))
        for owner, name in (
            (renderer, "is_gradient_checkpointing"),
            (transformer, "gradient_checkpointing"),
            (transformer, "is_gradient_checkpointing"),
        )
    )
    if enabled:
        raise RoleCompositionError("gradient checkpointing remains enabled")
    return {
        "gradient_checkpointing_enabled": False,
        "route_lifetime": "one_native_shared_step_forward",
    }


def _validate_generated(value: Any, *, shape: Sequence[int], device: Any, label: str) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device != device
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or tuple(int(item) for item in value.shape) != tuple(shape)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise RoleCompositionError(f"{label} sampler return contract differs")
    return value


def _rank_consensus(value: str, *, world_size: int, label: str) -> str:
    import torch.distributed as dist

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    if any(item != value for item in gathered):
        raise RoleCompositionError(f"{label} differs across WORLD4 ranks")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    for name in (
        "runtime_source_revision",
        "expected_bernini_commit",
        "expected_veomni_commit",
    ):
        _sha(getattr(args, name), length=40, label=name)
    for name in (
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _sha(getattr(args, name), length=64, label=name)
    if (
        args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise RoleCompositionError("pinned model/source identity differs")
    root_spec, cell, spec_path, spec_sha = load_cell_spec(
        args.cell_spec,
        expected_file_sha256=args.expected_cell_spec_sha256,
        cell_id=args.cell_id,
    )
    output_dir = output_runtime._fresh_output_directory(args.output_dir)
    publication = validate_adapter_publication(
        adapter_path=args.adapter,
        receipt_path=args.adapter_training_receipt,
        expected_adapter_sha256=args.expected_adapter_sha256,
        expected_receipt_sha256=args.expected_adapter_training_receipt_sha256,
    )
    adapter_path = Path(publication["adapter_path"])
    training_receipt_path = Path(publication["training_receipt_path"])
    source_path = Path(cell["source_video"])

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise RoleCompositionError(str(error)) from error
    if int(transformer_config.get("num_attention_heads", 0)) % SP_SIZE:
        raise RoleCompositionError("Bernini heads do not divide Ulysses4")
    inference_file_hashes = native.legacy.validate_inference_source_files(bernini_root)
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

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

    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != SP_SIZE
        or distributed.ulysses_size != SP_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise RoleCompositionError("runtime requires one AUH WORLD4/Ulysses4 group")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=SP_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_manifest = _plain_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    validation: list[Any] = [None]
    if distributed.rank == 0:
        try:
            validation[0] = {
                "ok": True,
                "checkpoint": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            validation[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(validation, src=0)
    if not isinstance(validation[0], Mapping) or validation[0].get("ok") is not True:
        raise RoleCompositionError(f"checkpoint validation failed: {validation[0]}")
    checkpoint_identity = dict(validation[0]["checkpoint"])

    source_tensor, source_metadata, source_sha = native.source_audit.prepare_hashed_source_snapshot(
        source_path
    )
    if source_sha != cell["source_video_sha256"] or source_metadata.get("frame_count") != FRAME_COUNT:
        raise RoleCompositionError("source snapshot differs from held-out exact81 cell")
    bucket_hw = tuple(int(item) for item in source_metadata["source_derived_bucket_hw"])

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    donor_prompt = native.build_task_prompt(
        "t2v", cell["action_caption"], prompt_cleaner=prompt_clean
    )
    donor_ids, donor_mask = native.legacy._tokenize_training_prompt(tokenizer, donor_prompt)
    composition_ids, composition_mask = orbit_trainer._tokenize_positive(
        tokenizer, orbit_trainer.GENERIC_INSTRUCTION
    )
    negative_ids, negative_mask = orbit_trainer._tokenize_negative(
        tokenizer, orbit_trainer.DEFAULT_NEGATIVE_PROMPT
    )
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise RoleCompositionError("runtime Bernini negative prompt differs")

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise RoleCompositionError("renderer is not pinned UniPC shift5")
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    source_pixels = source_tensor.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        source_references = {
            index: _vae_encode(
                vae, source_pixels[:, :, index : index + 1].contiguous()
            ).contiguous()
            for index in REFERENCE_INDICES
        }
    reference_broadcasts = {
        str(index): native._broadcast_condition_from_rank_zero(
            value, label=f"source_reference_{index}", world_size=SP_SIZE
        )
        for index, value in source_references.items()
    }
    vae.to("cpu")
    del source_tensor, source_pixels
    torch.cuda.empty_cache()

    renderer.to(device)
    renderer.t5_text_encoder.to(device)
    with torch.inference_mode():
        donor_embeds = renderer.encode_prompt(donor_ids.to(device), donor_mask.to(device)).detach()
        composition_embeds = renderer.encode_prompt(
            composition_ids.to(device), composition_mask.to(device)
        ).detach()
        uncond_embeds = renderer.encode_prompt(
            negative_ids.to(device), negative_mask.to(device)
        ).detach()
    for value in (donor_embeds, composition_embeds, uncond_embeds):
        dist.broadcast(value, src=0)
    renderer.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(renderer.diff_dec)
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise RoleCompositionError("held-out decode requires only Bernini transformer_1")
    checkpointing = _disable_gradient_checkpointing(renderer, transformer)
    handle = target_adapter.install_native_target_adapter(
        transformer,
        rank=8,
        alpha=8.0,
        block_indices=target_adapter.DEFAULT_BLOCK_INDICES,
    )
    renderer.eval()
    loaded_adapter = strict_load_adapter(adapter_path, handle, publication)
    adapter_digest = _rank_consensus(
        loaded_adapter["live_parameter_digest"], world_size=SP_SIZE, label="loaded adapter"
    )

    latent_geometry = native._latent_geometry_receipt(
        bucket_hw=bucket_hw, z_dim=int(vae.config.z_dim)
    )
    video_shape = tuple(latent_geometry["video_latent_shape"])
    reference_shape = tuple(latent_geometry["reference_latent_shape"])
    layout = expected_rv2v_layout(video_shape)
    if (
        video_shape[:3] != (1, 16, LATENT_PHASES)
        or any(tuple(value.shape) != reference_shape for value in source_references.values())
        or list(REFERENCE_INDICES) != list(native.RV2V_REFERENCE_INDICES)
    ):
        raise RoleCompositionError("held-out source reference geometry differs")
    reference_identities = {
        str(index): native._all_rank_tensor_identity(
            value, label=f"source_reference_{index}", world_size=SP_SIZE
        )
        for index, value in source_references.items()
    }

    # Fresh, frozen, source-independent T2V proposal.  Adapter wrappers are
    # installed but have no active route, so they return their base projections.
    if target_adapter.active_route() is not None:
        raise RoleCompositionError("adapter route active before T2V donor generation")
    with torch.inference_mode():
        donor_result, donor_noise = native._sample_with_native_initial_noise_observer(
            sample_fn=lambda: diffusion.sample(
                prompt_embeds=donor_embeds,
                uncond_prompt_embeds=uncond_embeds,
                image_vae_latents=None,
                multi_video_vae_latents=None,
                multi_image_vae_latents=None,
                width=bucket_hw[1],
                height=bucket_hw[0],
                device=device,
                **native.native_sampling_contract(
                    "t2v", steps=args.num_inference_steps, seed=cell["donor_seed"]
                ),
            ),
            wan_diffusion_module=wan_diffusion,
            expected_shape=video_shape,
            expected_device=device,
            expected_seed=cell["donor_seed"],
        )
    donor_result = _validate_generated(
        donor_result, shape=video_shape, device=device, label="fresh T2V donor"
    )
    if target_adapter.active_route() is not None:
        raise RoleCompositionError("adapter route leaked into/out of T2V donor generation")
    donor_broadcast = native._broadcast_condition_from_rank_zero(
        donor_result, label="fresh_t2v_action_donor", world_size=SP_SIZE
    )
    donor_identity = native._all_rank_tensor_identity(
        donor_result, label="fresh_t2v_action_donor", world_size=SP_SIZE
    )

    generated: dict[str, Any] = {"t2v-donor": donor_result.detach().cpu().contiguous()}
    generated_identities: dict[str, Any] = {"t2v-donor": donor_identity}
    noises: dict[str, Any] = {"t2v-donor": donor_noise}
    noise_rank_identities: dict[str, Any] = {
        "t2v-donor": native._all_rank_tensor_identity(
            donor_noise.tensor, label="t2v_donor_initial_gaussian", world_size=SP_SIZE
        )
    }
    route_traces: dict[str, Any] = {}
    refs = [source_references[index] for index in REFERENCE_INDICES]
    for arm in ARM_ORDER:
        audit = RV2VTargetRouteAudit(
            diffusion,
            handle=handle,
            adapter_enabled=arm == "orbit-adapter",
            layout=layout,
            donor_condition=donor_result,
            image_references=refs,
            prompt_embeds=composition_embeds,
            uncond_embeds=uncond_embeds,
            expected_steps=args.num_inference_steps,
            target_seed=cell["target_seed"],
            sp_rank=distributed.local_rank,
        )
        audit.install()
        try:
            with torch.inference_mode():
                result, capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda a=audit: diffusion.sample(
                        prompt_embeds=composition_embeds,
                        uncond_prompt_embeds=uncond_embeds,
                        image_vae_latents=None,
                        multi_video_vae_latents=a.donor_list,
                        multi_image_vae_latents=a.reference_list,
                        width=bucket_hw[1],
                        height=bucket_hw[0],
                        device=device,
                        **native.native_sampling_contract(
                            "rv2v", steps=args.num_inference_steps, seed=cell["target_seed"]
                        ),
                    ),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=video_shape,
                    expected_device=device,
                    expected_seed=cell["target_seed"],
                )
        finally:
            audit.restore()
        if not audit.restored:
            raise RoleCompositionError("native target-route hooks did not restore")
        result = _validate_generated(result, shape=video_shape, device=device, label=arm)
        generated[arm] = result.detach().cpu().contiguous()
        generated_identities[arm] = native._all_rank_tensor_identity(
            generated[arm], label=f"generated_{arm}", world_size=SP_SIZE
        )
        noises[arm] = capture
        noise_rank_identities[arm] = native._all_rank_tensor_identity(
            capture.tensor, label=f"{arm}_initial_gaussian", world_size=SP_SIZE
        )
        route_traces[arm] = dict(audit.trace)

    if noises["base"].raw_value_sha256 != noises["orbit-adapter"].raw_value_sha256:
        raise RoleCompositionError("base/adapter did not start from byte-identical Gaussian")
    base_tensor = generated["base"].float()
    adapted_tensor = generated["orbit-adapter"].float()
    paired_delta = {
        "latent_rmse": float((adapted_tensor - base_tensor).square().mean().sqrt().item()),
        "latent_max_abs": float((adapted_tensor - base_tensor).abs().max().item()),
        "base_and_adapter_outputs_bitwise_equal": bool(torch.equal(base_tensor, adapted_tensor)),
    }
    if not all(math.isfinite(value) for key, value in paired_delta.items() if key != "base_and_adapter_outputs_bitwise_equal"):
        raise RoleCompositionError("paired output delta is non-finite")

    del donor_result, source_references, refs
    renderer.to("cpu")
    torch.cuda.empty_cache()

    after_adapter_sha = file_sha256(adapter_path)
    after_training_receipt_sha = file_sha256(training_receipt_path)
    if (
        after_adapter_sha != publication["adapter_sha256"]
        or after_training_receipt_sha != publication["training_receipt_sha256"]
    ):
        raise RoleCompositionError("published training artifacts changed during decode")
    after_checkpoint: list[Any] = [None]
    if distributed.rank == 0:
        try:
            after_checkpoint[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            after_checkpoint[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(after_checkpoint, src=0)
    if (
        not isinstance(after_checkpoint[0], Mapping)
        or after_checkpoint[0].get("identity") != checkpoint_identity
    ):
        raise RoleCompositionError("checkpoint content changed during held-out decode")

    rank_invariant_route_payload = _rank_invariant_route_payload(
        route_traces, expected_rank=distributed.local_rank
    )
    local_evidence = {
        "rank": distributed.rank,
        "adapter_digest": adapter_digest,
        "generated_digest": object_sha256(generated_identities),
        "rank_invariant_route_payload_digest": object_sha256(
            rank_invariant_route_payload
        ),
        "rank_local_route_receipt_digest": object_sha256(route_traces),
        "target_noise_sha256": noises["base"].raw_value_sha256,
    }
    gathered: list[Any] = [None] * SP_SIZE
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != list(range(4)):
        raise RoleCompositionError("WORLD4 rank evidence is incomplete")
    for name in (
        "adapter_digest",
        "generated_digest",
        "rank_invariant_route_payload_digest",
        "target_noise_sha256",
    ):
        if len({row.get(name) for row in gathered if isinstance(row, Mapping)}) != 1:
            raise RoleCompositionError(f"WORLD4 ranks disagree on {name}")
    if len(
        {
            row.get("rank_local_route_receipt_digest")
            for row in gathered
            if isinstance(row, Mapping)
        }
    ) != SP_SIZE:
        raise RoleCompositionError("WORLD4 rank-local route receipts are not one-to-one")

    if distributed.rank == 0:
        stage = output_runtime._output_staging_directory(output_dir)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                stage / f"{arm}.official-initial-gaussian.safetensors",
                noises[arm],
                all_rank_identity=noise_rank_identities[arm],
            )
            for arm in ("t2v-donor", *ARM_ORDER)
        }
        if args.num_inference_steps == 40:
            generated_device = {
                arm: value.to(device=device).contiguous() for arm, value in generated.items()
            }
            try:
                outputs = native._save_outputs(
                    output_dir=stage,
                    generated=generated_device,
                    vae=vae,
                    bucket_hw=bucket_hw,
                    device=device,
                    save_output_fn=save_output,
                )
            finally:
                generated_device.clear()
        else:
            outputs = {
                arm: {
                    "mp4": None,
                    "engineering_canary_only": True,
                    "normalized_clean_latent": native._save_normalized_clean_latent_atomically(
                        stage / f"{arm}.normalized-clean-latent.safetensors", value
                    ),
                }
                for arm, value in generated.items()
            }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "stage": "matched_exact40_qualitative_pilot"
            if args.num_inference_steps == 40
            else "one_step_engineering_canary",
            "runtime_source": {
                "revision": args.runtime_source_revision,
                "archive_sha256": args.runtime_source_archive_sha256,
                "launcher_sha256": args.launcher_source_sha256,
            },
            "cell_spec": {
                "path": str(spec_path),
                "file_sha256": spec_sha,
                "schema_version": root_spec["schema_version"],
                "cell": dict(cell),
                "selected_before_generation": True,
                "generated_quality_used_for_selection": False,
            },
            "model": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "checkpoint_content": checkpoint_identity,
                "checkpoint_unchanged": True,
                "inference_files": inference_file_hashes,
            },
            "adapter": {
                "publication": dict(publication),
                "loaded": dict(loaded_adapter),
                "file_unchanged": True,
                "training_receipt_unchanged": True,
                "loaded_on_all_four_ranks": True,
            },
            "topology": {
                "world_size": SP_SIZE,
                "ulysses_size": SP_SIZE,
                "local_group_role": cell["actor_kind"],
                "world4_rank_evidence": gathered,
            },
            "source": {
                "path": str(source_path),
                "sha256": source_sha,
                "metadata": source_metadata,
                "reference_indices": list(REFERENCE_INDICES),
                "reference_encoding": "four_independent_rgb_frame_to_wan_vae_calls",
                "references_from_full_video_latent_slice": False,
                "reference_broadcasts": reference_broadcasts,
                "reference_identities": reference_identities,
            },
            "fresh_t2v_donor": {
                "generated_inside_same_invocation": True,
                "frozen_base_model": True,
                "adapter_route_active": False,
                "caption_utf8_sha256": cell["action_caption_utf8_sha256"],
                "seed": cell["donor_seed"],
                "condition_only": True,
                "rv2v_target": False,
                "rv2v_noise": False,
                "pseudo_label": False,
                "optimizer_input": False,
                "external_media": False,
                "broadcast": donor_broadcast,
                "identity": donor_identity,
                "event_success_claim": False,
            },
            "composition": {
                "condition_contract": "V=fresh_ordered_T2V_donor; I=four_source_RGB_refs",
                "target_initialization": native.TARGET_INITIALIZATION,
                "rho": 0.0,
                "prompt": orbit_trainer.GENERIC_INSTRUCTION,
                "prompt_utf8_sha256": hashlib.sha256(
                    orbit_trainer.GENERIC_INSTRUCTION.encode("utf-8")
                ).hexdigest(),
                "prompt_matches_identity_orbit_training_text": True,
                "negative_prompt_matches_training": True,
                "target_seed": cell["target_seed"],
                "same_official_gaussian_base_and_adapter": True,
                "same_donor_refs_text_scheduler_guidance_base_and_adapter": True,
                "layout": layout,
                "gradient_checkpointing": checkpointing,
                "route_traces": route_traces,
                "paired_delta": paired_delta,
            },
            "sampling": {
                "frame_count": FRAME_COUNT,
                "latent_phases": LATENT_PHASES,
                "fps": 25,
                "num_inference_steps": args.num_inference_steps,
                "exact81": True,
                "native_unipc_shift5": True,
                "native_guidance": native_guidance.guidance_receipt(),
                "source_rich_noise": False,
            },
            "initial_noise_artifacts": noise_artifacts,
            "generated_identities": generated_identities,
            "outputs": outputs,
            "runtime_versions": {
                "torch": torch.__version__,
                "torch_hip": str(torch.version.hip),
                "diffusers": diffusers_version,
                "transformers": transformers_version,
            },
            "interpretation": {
                "training_performed": False,
                "optimizer": None,
                "backward": False,
                "model_or_adapter_written": False,
                "target_video": False,
                "mask": False,
                "flow": False,
                "pose": False,
                "track": False,
                "trajectory": False,
                "external_donor": False,
                "older_calibration_bank_consumed": False,
                "action_success_evaluated": False,
                "identity_preservation_evaluated": False,
                "quality_claim": False,
                "scientific_claim_authorized": False,
                "one_step_is_engineering_only": args.num_inference_steps == 1,
                "heldout_role_composition_decode_authorized_by_training_receipt": True,
            },
        }
        receipt = output_runtime._rebase_artifact_paths(
            receipt, old_root=stage, new_root=output_dir
        )
        receipt["receipt_digest"] = object_sha256(receipt)
        output_runtime._write_receipt(stage / "receipt.json", receipt)
        output_runtime._commit_output_transaction(staging=stage, final=output_dir)
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STEPS",
    "ARM_ORDER",
    "FRAME_COUNT",
    "METHOD",
    "REFERENCE_INDICES",
    "RV2VTargetRouteAudit",
    "RoleCompositionError",
    "SCHEMA_VERSION",
    "SPEC_SCHEMA",
    "expected_rv2v_layout",
    "load_cell_spec",
    "main",
    "strict_load_adapter",
    "validate_adapter_metadata",
    "validate_adapter_publication",
]
