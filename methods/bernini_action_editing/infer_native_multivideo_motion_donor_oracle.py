#!/usr/bin/env python3
"""Frozen exact-81 Bernini native multi-video motion-donor oracle.

This engineering runner asks one narrow causal question: does the *second*
native video condition in Bernini's official ``rv2v`` sampler contribute a
useful motion marginal while the first video condition preserves source
content?  It does not train, patch model arithmetic, inject a target, or turn a
donor into supervision.  Registered DMIQ T2V proposals are consumed only as
their FP32, pre-decode normalized clean latents.  Donor MP4 files are neither
opened nor decoded/re-encoded.

For ``[source, donor]`` and no image conditions, the pinned sampler implements
``V=[source]`` and ``VI=[source, donor]``.  Its official ``rv2v`` field is
therefore

    eps_none + w_vid (eps_V-eps_none)
             + w_img (eps_VI-eps_V)
             + w_txt (eps_VTI-eps_VI).

The installed condition audit is observational and reversible.  It verifies,
without replacing values, the per-step source-id order, visual-token lengths,
prompt branch order, and one untouched scheduler call.  Every arm, including
``O0``, uses the same read-only initial-Gaussian capture wrapper; ``O0`` alone
has no multi-video/shared-step/scheduler audit hooks.  ``Z0`` adds those hooks
to the same source-only call and must return a byte-identical native FP32
latent.  The order-swap arm is explicitly a joint
privileged-V/source-id/order diagnostic, not a pure semantic role swap.

``--num-inference-steps 1`` is an OOM/call-path engineering canary only.
``--num-inference-steps 40`` is the matched qualitative causal pilot.  Neither
stage authorizes an action-success, identity-preservation, or motion-transfer
claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dmiq_t2v_factorial_bank as factor_bank  # noqa: E402
import infer_fitq_owner_prompt_cross_query_micro as bank_contract  # noqa: E402
import infer_native_identity_generation_canary as native  # noqa: E402
import tri_branch_unipc as sampler_contract  # noqa: E402


METHOD = "frozen-bernini-native-multivideo-motion-donor-oracle"
SCHEMA_VERSION = "bernini-native-multivideo-motion-donor-oracle-receipt-v1"
FRAME_COUNT = 81
LATENT_SHAPE = (1, 16, 21, 62, 60)
PATCH_TOKENS = 19_530
FPS = 25
HEIGHT = 496
WIDTH = 480
TARGET_SEED = 20_260_810
ALLOWED_STEPS = (1, 40)
ULYSSES_SIZE = 4
CDF_DOG_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
DONOR_BRANCHES = ("full_action", "noop", "reverse_action", "wrong_actor")
PINNED_TRANSFORMER_WAN_SHA256 = (
    "9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223"
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class MotionDonorOracleError(RuntimeError):
    """Raised before ambiguous donor evidence is emitted."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    guidance_mode: str
    condition_roles: tuple[str, ...]
    donor_branch: Optional[str]
    observed: bool
    privileged_v_role: str
    diagnostic: str


ARM_SPECS = (
    ArmSpec(
        "O0", "rv2v", ("source_video",), None, False, "source_video",
        "source_only_without_condition_shared_step_scheduler_audit_hooks",
    ),
    ArmSpec(
        "Z0", "rv2v", ("source_video",), None, True, "source_video",
        "read_only_observer_zero_effect_control",
    ),
    ArmSpec(
        "D-action", "rv2v", ("source_video", "registered_full_action_donor"),
        "full_action", True, "source_video", "registered_action_donor",
    ),
    ArmSpec(
        "D-noop", "rv2v", ("source_video", "registered_noop_donor"),
        "noop", True, "source_video", "registered_noop_negative_control",
    ),
    ArmSpec(
        "D-reverse", "rv2v", ("source_video", "registered_reverse_action_donor"),
        "reverse_action", True, "source_video", "registered_reverse_negative_control",
    ),
    ArmSpec(
        "D-wrong-actor", "rv2v", ("source_video", "registered_wrong_actor_donor"),
        "wrong_actor", True, "source_video", "registered_wrong_actor_control",
    ),
    ArmSpec(
        "D-duplicate-source", "rv2v", ("source_video", "source_video_duplicate"),
        None, True, "source_video", "second_slot_without_external_donor",
    ),
    ArmSpec(
        "D-action-order-swap", "rv2v",
        ("registered_full_action_donor", "source_video"), "full_action", True,
        "registered_full_action_donor",
        "joint_privileged_v_source_id_and_order_diagnostic_not_pure_role_swap",
    ),
    ArmSpec(
        "A-source-v2v-apg", "v2v_apg", ("source_video",), None, True,
        "source_video", "cheap_old_base_sampler_anchor_not_a_donor_arm",
    ),
)
ARM_ORDER = tuple(spec.arm_id for spec in ARM_SPECS)


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
        raise MotionDonorOracleError(f"receipt is not canonical finite ASCII JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: Any, *, length: int, label: str) -> str:
    text = str(value)
    pattern = _SHA1 if length == 40 else _SHA256
    if pattern.fullmatch(text) is None:
        raise MotionDonorOracleError(f"{label} must be lowercase SHA-{1 if length == 40 else 256}")
    return text


def _load_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise MotionDonorOracleError(f"{label} must be an absolute non-symlink file")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_file() or resolved.is_symlink():
        raise MotionDonorOracleError(f"{label} must be a canonical plain file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MotionDonorOracleError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise MotionDonorOracleError(f"{label} root must be an object")
    return value, resolved, file_sha256(resolved)


def _canonical_root(path: str | Path, *, label: str) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise MotionDonorOracleError(f"{label} must be absolute and non-symlink")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_dir() or resolved.is_symlink():
        raise MotionDonorOracleError(f"{label} must be a canonical plain directory")
    return resolved


def _fresh_output_directory(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or requested == Path("/") or _SAFE_NAME.fullmatch(requested.name) is None:
        raise MotionDonorOracleError("output directory must be an absolute safe non-root path")
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise MotionDonorOracleError("output parent must be a plain directory")
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise MotionDonorOracleError("refusing to reuse output directory")
    return output


def _sealed_digest(value: Mapping[str, Any], *, field_name: str, label: str) -> str:
    declared = _require_sha(value.get(field_name), length=64, label=f"{label} embedded digest")
    unsigned = dict(value)
    unsigned.pop(field_name, None)
    if object_sha256(unsigned) != declared:
        raise MotionDonorOracleError(f"{label} embedded digest differs")
    return declared


def arm_plan() -> tuple[ArmSpec, ...]:
    """Return and self-check the fixed matched-arm order."""

    if tuple(spec.arm_id for spec in ARM_SPECS) != ARM_ORDER or len(set(ARM_ORDER)) != len(ARM_ORDER):
        raise MotionDonorOracleError("arm order/uniqueness changed")
    for spec in ARM_SPECS:
        if spec.guidance_mode not in {"rv2v", "v2v_apg"}:
            raise MotionDonorOracleError("unsupported guidance mode in arm plan")
        if spec.arm_id != "D-action-order-swap" and spec.condition_roles[0] != "source_video":
            raise MotionDonorOracleError("source-first closure changed")
        if spec.arm_id == "D-action-order-swap" and spec.privileged_v_role != spec.condition_roles[0]:
            raise MotionDonorOracleError("order-swap privileged-V contract changed")
        if spec.donor_branch is not None and spec.donor_branch not in DONOR_BRANCHES:
            raise MotionDonorOracleError("arm requests an unregistered donor branch")
    return ARM_SPECS


def bind_registered_donors(
    *,
    manifest: Mapping[str, Any],
    bank_receipt: Mapping[str, Any],
    execution_group: str,
) -> dict[str, Any]:
    """Bind one complete registered cell, retaining only required donor rows."""

    try:
        bound = bank_contract.validate_micro_bank_bindings(
            manifest, bank_receipt, execution_group=execution_group
        )
        bank_contract.validate_renderer_contract(bound["manifest"]["renderer_contract"])
    except Exception as error:
        raise MotionDonorOracleError(str(error)) from error
    rows = {
        row["manifest"]["semantic_branch"]: row
        for row in bound["prompt_rows"]
    }
    if any(branch not in rows for branch in DONOR_BRANCHES):
        raise MotionDonorOracleError("registered cell lacks required donor branches")
    if rows["full_action"]["manifest"].get("prompt") is None:
        raise MotionDonorOracleError("registered full-action prompt is absent")
    return {
        "manifest": bound["manifest"],
        "bank_receipt_digest": bound["bank_receipt_digest"],
        "cell": bound["cell"],
        "target_prompt": rows["full_action"]["manifest"]["prompt"],
        "target_prompt_sha256": rows["full_action"]["manifest"]["prompt_utf8_sha256"],
        "donor_rows": {branch: rows[branch] for branch in DONOR_BRANCHES},
    }


def _safe_tensor_load(path: Path) -> tuple[Any, dict[str, str]]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise MotionDonorOracleError("safetensors is required for registered clean donors") from error
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != ["normalized_clean_latent"]:
            raise MotionDonorOracleError("registered clean donor has an unexpected tensor key")
        metadata = dict(opened.metadata() or {})
        tensor = opened.get_tensor("normalized_clean_latent").contiguous()
    return tensor, metadata


def load_registered_clean_donor(
    *,
    row: Mapping[str, Any],
    bank_root: Path,
    tensor_loader: Callable[[Path], tuple[Any, dict[str, str]]] = _safe_tensor_load,
) -> tuple[Any, dict[str, Any]]:
    """Load one receipt-bound native pre-decode tensor; never inspect its MP4."""

    manifest_row = row.get("manifest")
    bank_row = row.get("bank")
    if not isinstance(manifest_row, Mapping) or not isinstance(bank_row, Mapping):
        raise MotionDonorOracleError("donor row lacks manifest/bank halves")
    expected_parent = bank_root / str(manifest_row.get("output_subdir"))
    if expected_parent.is_symlink():
        raise MotionDonorOracleError("registered donor entry directory is a symlink")
    expected_parent = expected_parent.resolve(strict=True)
    try:
        expected_parent.relative_to(bank_root)
    except ValueError as error:
        raise MotionDonorOracleError("registered donor entry escaped bank root") from error

    receipt, receipt_path, receipt_file_sha = _load_json(
        bank_row.get("native_receipt_path", ""), label="registered native donor receipt"
    )
    if receipt_path.parent != expected_parent:
        raise MotionDonorOracleError("native donor receipt escaped its registered entry")
    if receipt_file_sha != bank_row.get("native_receipt_file_sha256"):
        raise MotionDonorOracleError("native donor receipt file SHA-256 differs")
    receipt_digest = _sealed_digest(receipt, field_name="receipt_digest", label="native donor receipt")
    if (
        receipt_digest != bank_row.get("native_receipt_digest")
        or receipt.get("schema_version") != factor_bank.NATIVE_RECEIPT_SCHEMA
        or receipt.get("arms") != ["t2v"]
    ):
        raise MotionDonorOracleError("native donor receipt schema/digest/arm differs")
    interpretation = receipt.get("interpretation")
    if not isinstance(interpretation, Mapping) or interpretation.get("training_performed") is not False:
        raise MotionDonorOracleError("native donor is not a frozen T2V proposal")

    outputs = receipt.get("outputs")
    output = outputs.get("t2v") if isinstance(outputs, Mapping) else None
    clean = output.get("normalized_clean_latent") if isinstance(output, Mapping) else None
    if not isinstance(clean, Mapping):
        raise MotionDonorOracleError("native pre-decode clean donor is absent")
    clean_requested = Path(bank_row.get("clean_latent_path", "")).expanduser()
    if not clean_requested.is_absolute() or clean_requested.is_symlink():
        raise MotionDonorOracleError("registered clean donor path must be absolute and non-symlink")
    clean_path = clean_requested.resolve(strict=True)
    if clean_path != clean_requested or not clean_path.is_file() or clean_path.parent != expected_parent:
        raise MotionDonorOracleError("registered clean donor path is noncanonical or escaped")
    if Path(str(clean.get("path", ""))) != clean_path:
        raise MotionDonorOracleError("native receipt clean path differs from bank binding")
    clean_file_sha = file_sha256(clean_path)
    if clean_file_sha != bank_row.get("clean_latent_sha256") or clean_file_sha != clean.get("sha256"):
        raise MotionDonorOracleError("registered clean donor file SHA-256 differs")
    required_clean = {
        "tensor_key": "normalized_clean_latent",
        "shape": list(LATENT_SHAPE),
        "stored_dtype": "torch.float32",
        "coordinate": "bernini_normalized_clean_vae_latent",
        "artifact_role": "native_sampler_proposal",
        "native_sampler_before_vae_decode": True,
        "source_video_vae_encode_before_any_decode": False,
        "mp4_decode_reencode_used": False,
        "roundtrip_byte_exact_fp32": True,
    }
    if any(clean.get(name) != wanted for name, wanted in required_clean.items()):
        raise MotionDonorOracleError("native clean-donor artifact contract differs")

    # The loader is called only after every path/receipt/hash check above.  No
    # companion video path is opened anywhere in this runtime.
    tensor, metadata = tensor_loader(clean_path)
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise MotionDonorOracleError("PyTorch is required for donor tensor validation") from error
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tuple(int(item) for item in tensor.shape) != LATENT_SHAPE
        or tensor.requires_grad
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise MotionDonorOracleError("registered donor tensor dtype/shape/finite contract differs")
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "native_sampler_proposal",
        "source": "native_sampler_before_vae_decode",
    }
    if metadata != expected_metadata:
        raise MotionDonorOracleError("registered donor safetensors metadata differs")
    identity = native.value_audit.tensor_identity(tensor, label="registered_clean_donor")
    generated = receipt.get("generated_identities", {}).get("t2v", {})
    generated_identity = generated.get("identity") if isinstance(generated, Mapping) else None
    if (
        generated.get("all_rank_exact") is not True
        or not isinstance(generated_identity, Mapping)
        or generated_identity.get("raw_storage_sha256") != identity.get("raw_storage_sha256")
        or generated_identity.get("shape") != list(LATENT_SHAPE)
    ):
        raise MotionDonorOracleError("native donor raw tensor identity differs from receipt")
    provenance = {
        "entry_id": manifest_row["entry_id"],
        "semantic_branch": manifest_row["semantic_branch"],
        "registered_prompt_utf8_sha256": manifest_row["prompt_utf8_sha256"],
        "proposal_cell_id": manifest_row["proposal_cell_id"],
        "execution_group": manifest_row["execution_group"],
        "bank_generation_seed": manifest_row["seed"],
        "output_subdir": manifest_row["output_subdir"],
        "native_receipt_path": str(receipt_path),
        "native_receipt_file_sha256": receipt_file_sha,
        "native_receipt_digest": receipt_digest,
        "clean_latent_path": str(clean_path),
        "clean_latent_file_sha256": clean_file_sha,
        "clean_latent_raw_storage_sha256": identity["raw_storage_sha256"],
        "clean_latent_identity": identity,
        "coordinate": metadata["coordinate"],
        "artifact_role": metadata["artifact_role"],
        "native_sampler_before_vae_decode": True,
        "donor_mp4_opened": False,
        "donor_mp4_decode_reencode_used": False,
        "bank_initial_noise_file_sha256": bank_row["initial_noise_file_sha256"],
        "bank_initial_noise_tensor_value_sha256": bank_row["initial_noise_tensor_value_sha256"],
        "bank_method_source_revision": bank_row["method_source_revision"],
        "bank_method_source_archive_sha256": bank_row["method_source_archive_sha256"],
        "registered_branch_label_is_not_verified_action_success": True,
    }
    return tensor, provenance


@dataclass
class _ActiveAudit:
    completed_steps: int = 0
    patch_source_ids: list[float] = field(default_factory=list)
    shared_lengths: list[int] = field(default_factory=list)
    shared_visual_objects: list[Any] = field(default_factory=list)
    shared_rotary_objects: list[Any] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)


class NativeMultiVideoConditionAudit:
    """Read-only audit of the pinned native source-id/guidance/scheduler path."""

    def __init__(
        self,
        diffusion: Any,
        *,
        condition_list: list[Any],
        condition_roles: Sequence[str],
        guidance_mode: str,
        expected_steps: int,
        expected_seed: int,
        prompt_embeds: Any,
        uncond_prompt_embeds: Any,
    ) -> None:
        self.diffusion = sampler_contract.resolve_diffusion_core(diffusion)
        self.transformer = getattr(self.diffusion, "transformer", None)
        self.scheduler = getattr(self.diffusion, "scheduler", None)
        self.condition_list = condition_list
        self.condition_roles = tuple(condition_roles)
        self.guidance_mode = str(guidance_mode)
        self.expected_steps = int(expected_steps)
        self.expected_seed = int(expected_seed)
        self.prompt_embeds = prompt_embeds
        self.uncond_prompt_embeds = uncond_prompt_embeds
        self._original_sample = getattr(self.diffusion, "sample", None)
        self._original_shared = getattr(self.diffusion, "shared_step", None)
        self._original_patch = getattr(self.transformer, "patch_vae_latent", None)
        self._original_scheduler = getattr(self.scheduler, "step", None)
        self._patches: list[tuple[Any, str, bool, Any]] = []
        self._active: Optional[_ActiveAudit] = None
        self.sample_calls = 0
        self.restored = False
        self.trace: dict[str, Any] = {}
        if (
            not isinstance(condition_list, list)
            or len(condition_list) not in (1, 2)
            or len(condition_list) != len(self.condition_roles)
            or self.guidance_mode not in {"rv2v", "v2v_apg"}
            or self.expected_steps not in ALLOWED_STEPS
            or self.prompt_embeds is None
            or self.uncond_prompt_embeds is None
            or not all(callable(value) for value in (
                self._original_sample, self._original_shared,
                self._original_patch, self._original_scheduler,
            ))
        ):
            raise MotionDonorOracleError("native multi-video audit construction differs")
        if getattr(self.diffusion, "transformer_2", None) is not None:
            raise MotionDonorOracleError("multi-video oracle supports only frozen Bernini 1.3B")
        for owner, name in (
            (self.diffusion, "sample"),
            (self.diffusion, "shared_step"),
            (self.transformer, "patch_vae_latent"),
            (self.scheduler, "step"),
        ):
            if name in vars(owner):
                raise MotionDonorOracleError(f"refusing to stack audit on {name} override")

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        instance = vars(owner)
        had_instance = name in instance
        previous = instance.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._patches:
            raise MotionDonorOracleError("native multi-video audit already installed")

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared(*args, **kwargs)

        def patch_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_patch(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler(*args, **kwargs)

        for wrapper in (sample_wrapper, shared_wrapper, patch_wrapper, scheduler_wrapper):
            setattr(wrapper, "_bernini_native_multivideo_motion_donor_audit", self)
        try:
            self._set_patch(self.transformer, "patch_vae_latent", patch_wrapper)
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self.restored = False

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:  # pragma: no cover - pathological runtime
                errors.append(error)
        self._active = None
        self.restored = not errors
        if errors:
            raise MotionDonorOracleError(f"failed to restore {len(errors)} observer hooks") from errors[0]

    def _expected_patch_ids(self) -> list[float]:
        return [float(index) for index in range(1, len(self.condition_list) + 1)] + [0.0]

    def _expected_shared_lengths(self) -> list[int]:
        count = len(self.condition_list)
        if self.guidance_mode == "rv2v":
            return [PATCH_TOKENS, 2 * PATCH_TOKENS, (count + 1) * PATCH_TOKENS, (count + 1) * PATCH_TOKENS]
        return [(count + 1) * PATCH_TOKENS, (count + 1) * PATCH_TOKENS]

    def _sampling_contract(self) -> dict[str, Any]:
        if self.guidance_mode == "rv2v":
            return native.native_sampling_contract(
                "rv2v", steps=self.expected_steps, seed=self.expected_seed
            )
        return native.legacy.sampler_contract(
            steps=self.expected_steps, seed=self.expected_seed
        )

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None or self.sample_calls:
            raise MotionDonorOracleError("audit permits exactly one native sample call")
        try:
            values = sampler_contract._bind_call(self._original_sample, args, kwargs)
        except Exception as error:
            raise MotionDonorOracleError(str(error)) from error
        expected = self._sampling_contract()
        for name, wanted in expected.items():
            observed = values.get(name)
            if isinstance(wanted, tuple):
                observed = tuple(observed)
            if observed != wanted:
                raise MotionDonorOracleError(f"native sample {name} differs")
        if (
            values.get("multi_video_vae_latents") is not self.condition_list
            or values.get("image_vae_latents") is not None
            or values.get("multi_image_vae_latents") is not None
        ):
            raise MotionDonorOracleError("native video-list identity or no-image closure differs")
        for observed, expected_object in zip(values["multi_video_vae_latents"], self.condition_list):
            if observed is not expected_object:
                raise MotionDonorOracleError("native condition tensor object/order differs")
        if (
            values.get("prompt_embeds") is not self.prompt_embeds
            or values.get("uncond_prompt_embeds") is not self.uncond_prompt_embeds
        ):
            raise MotionDonorOracleError("native action/negative prompt object identity differs")
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
                raise MotionDonorOracleError("native multi-video sample ended with incomplete audit state")
            self.sample_calls = 1
            self.trace = {
                "sample_calls": 1,
                "step_count": self.expected_steps,
                "scheduler_calls": self.expected_steps,
                "guidance_mode": self.guidance_mode,
                "condition_roles_in_list_order": list(self.condition_roles),
                "source_id_order_per_step": self._expected_patch_ids(),
                "shared_visual_token_lengths_per_step": self._expected_shared_lengths(),
                "step_records": list(state.step_records),
                "step_records_digest": object_sha256(state.step_records),
                "original_callables_received_unchanged_arguments": True,
                "original_return_objects_forwarded": True,
                "observer_modified_numerics": False,
            }
            return result
        finally:
            self._active = None

    def _wrapped_patch(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise MotionDonorOracleError("patch_vae_latent ran outside observed sample")
        try:
            values = sampler_contract._bind_call(self._original_patch, args, kwargs)
        except Exception as error:
            raise MotionDonorOracleError(str(error)) from error
        expected_ids = self._expected_patch_ids()
        index = len(state.patch_source_ids)
        if index >= len(expected_ids):
            raise MotionDonorOracleError("too many patch_vae_latent calls before scheduler step")
        source_id = float(values.get("source_id"))
        if not math.isclose(source_id, expected_ids[index], rel_tol=0.0, abs_tol=0.0):
            raise MotionDonorOracleError("native patch_vae_latent source-id order differs")
        result = self._original_patch(*args, **kwargs)
        state.patch_source_ids.append(source_id)
        return result

    def _wrapped_shared(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise MotionDonorOracleError("shared_step ran outside observed sample")
        try:
            values = sampler_contract._bind_call(self._original_shared, args, kwargs)
        except Exception as error:
            raise MotionDonorOracleError(str(error)) from error
        expected = self._expected_shared_lengths()
        index = len(state.shared_lengths)
        if index >= len(expected):
            raise MotionDonorOracleError("too many shared_step calls before scheduler step")
        lengths = values.get("batch_vae_seqlen")
        if not isinstance(lengths, (list, tuple)) or tuple(int(value) for value in lengths) != (expected[index],):
            raise MotionDonorOracleError("native shared visual-token length/order differs")
        visual = values.get("noisy_latents")
        rotary = values.get("rotary_embs")
        if visual is None or rotary is None:
            raise MotionDonorOracleError("native shared visual/rotary objects are absent")
        expected_prompts = (
            [self.uncond_prompt_embeds, self.uncond_prompt_embeds,
             self.uncond_prompt_embeds, self.prompt_embeds]
            if self.guidance_mode == "rv2v"
            else [self.uncond_prompt_embeds, self.prompt_embeds]
        )
        if values.get("cond_embeds") is not expected_prompts[index]:
            raise MotionDonorOracleError("native negative/action forward order differs")
        if values.get("model_id") != "transformer_1":
            raise MotionDonorOracleError("native sampler left the pinned 1.3B expert")
        if self.guidance_mode == "rv2v" and index == 3:
            if visual is not state.shared_visual_objects[2] or rotary is not state.shared_rotary_objects[2]:
                raise MotionDonorOracleError("rv2v VI uncond/action visual or rotary object differs")
        result = self._original_shared(*args, **kwargs)
        state.shared_lengths.append(expected[index])
        state.shared_visual_objects.append(visual)
        state.shared_rotary_objects.append(rotary)
        return result

    def _wrapped_scheduler(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise MotionDonorOracleError("scheduler.step ran outside observed sample")
        expected_ids = self._expected_patch_ids()
        expected_lengths = self._expected_shared_lengths()
        if state.patch_source_ids != expected_ids or state.shared_lengths != expected_lengths:
            raise MotionDonorOracleError("scheduler.step arrived before a complete native guidance step")
        result = self._original_scheduler(*args, **kwargs)
        state.step_records.append({
            "step_index": state.completed_steps,
            "source_ids": list(state.patch_source_ids),
            "condition_roles": list(self.condition_roles),
            "shared_visual_token_lengths": list(state.shared_lengths),
            "target_source_id": 0,
            "image_condition_count": 0,
            "scheduler_original_return_forwarded": True,
        })
        state.completed_steps += 1
        state.patch_source_ids.clear()
        state.shared_lengths.clear()
        state.shared_visual_objects.clear()
        state.shared_rotary_objects.clear()
        return result


def _sampling_values(spec: ArmSpec, *, steps: int) -> dict[str, Any]:
    if spec.guidance_mode == "rv2v":
        return native.native_sampling_contract("rv2v", steps=steps, seed=TARGET_SEED)
    return native.legacy.sampler_contract(steps=steps, seed=TARGET_SEED)


def _condition_list(spec: ArmSpec, *, source: Any, donors: Mapping[str, Any]) -> list[Any]:
    if spec.arm_id in {"O0", "Z0", "A-source-v2v-apg"}:
        return [source]
    if spec.arm_id == "D-duplicate-source":
        conditions = [source, source]
        if conditions[0] is not conditions[1]:
            raise MotionDonorOracleError("duplicate-source arm lost exact tensor-object duplication")
        return conditions
    if spec.donor_branch is None or spec.donor_branch not in donors:
        raise MotionDonorOracleError("arm lacks its registered donor tensor")
    donor = donors[spec.donor_branch]
    if spec.arm_id == "D-action-order-swap":
        return [donor, source]
    return [source, donor]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--factor-manifest", required=True)
    parser.add_argument("--expected-factor-manifest-file-sha256", required=True)
    parser.add_argument("--factor-bank-receipt", required=True)
    parser.add_argument("--expected-factor-bank-receipt-file-sha256", required=True)
    parser.add_argument("--bank-output-root", required=True)
    parser.add_argument("--execution-group", required=True, choices=factor_bank.GROUPS)
    parser.add_argument("--num-inference-steps", type=int, required=True, choices=ALLOWED_STEPS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=native.legacy.trainer.CHECKPOINT_TREE_SHA256)
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    arm_plan()
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        _require_sha(getattr(args, name), length=40, label=name)
    for name in (
        "runtime_source_archive_sha256",
        "launcher_source_sha256",
        "expected_factor_manifest_file_sha256",
        "expected_factor_bank_receipt_file_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        _require_sha(getattr(args, name), length=64, label=name)
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise MotionDonorOracleError("Bernini commit differs from pinned release")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise MotionDonorOracleError("VeOmni commit differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise MotionDonorOracleError("checkpoint tree differs from pinned release")


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MotionDonorOracleError("refusing to overwrite receipt")
    payload = canonical_json_bytes(receipt) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_staging_directory(final: Path) -> Path:
    """Create a hidden sibling used until the complete receipt is durable."""

    transaction = os.environ.get("BERNINI_OUTPUT_TRANSACTION_ID", "")
    if _SAFE_NAME.fullmatch(transaction) is None:
        raise MotionDonorOracleError(
            "BERNINI_OUTPUT_TRANSACTION_ID must be one safe non-empty token"
        )
    staging = Path(
        tempfile.mkdtemp(
            dir=final.parent,
            prefix=f".{final.name}.partial-{transaction}-",
        )
    )
    staging.chmod(0o755)
    if staging.parent != final.parent or staging.is_symlink():
        raise MotionDonorOracleError("output staging directory escaped its parent")
    return staging


def _rebase_artifact_paths(value: Any, *, old_root: Path, new_root: Path) -> Any:
    """Rewrite only exact staging-root path prefixes inside a receipt."""

    old = str(old_root)
    new = str(new_root)
    if isinstance(value, str):
        if value == old:
            return new
        if value.startswith(old + os.sep):
            return new + value[len(old):]
        return value
    if isinstance(value, Mapping):
        return {
            key: _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rebase_artifact_paths(item, old_root=old_root, new_root=new_root)
            for item in value
        )
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_output_transaction(*, staging: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise MotionDonorOracleError("refusing to replace an output directory")
    if staging.parent != final.parent or not staging.is_dir() or staging.is_symlink():
        raise MotionDonorOracleError("output staging directory is invalid")
    _fsync_directory(staging)
    os.replace(staging, final)
    _fsync_directory(final.parent)


def _save_outputs(
    *, output_dir: Path, generated: Mapping[str, Any], vae: Any,
    device: Any, save_output_fn: Any, steps: int,
) -> dict[str, Any]:
    if steps == 40:
        # The matched rollout tensors are deliberately retained as FP32 CPU
        # evidence between arms.  Bernini's decoder helper expects its input
        # on the same device to which ``native._save_outputs`` moves the VAE.
        # Materialize a short-lived device copy here; this is post-sampling and
        # cannot affect any rollout or O0/Z0 parity comparison.
        generated_for_decode = {
            arm: latent.to(device=device).contiguous()
            for arm, latent in generated.items()
        }
        try:
            return native._save_outputs(
                output_dir=output_dir,
                generated=generated_for_decode,
                vae=vae,
                bucket_hw=(HEIGHT, WIDTH),
                device=device,
                save_output_fn=save_output_fn,
            )
        finally:
            generated_for_decode.clear()
    outputs: dict[str, Any] = {}
    for arm, latent in generated.items():
        clean = native._save_normalized_clean_latent_atomically(
            output_dir / f"{arm}.normalized-clean-latent.safetensors", latent
        )
        outputs[arm] = {
            "normalized_clean_latent": clean,
            "mp4": None,
            "decode_performed": False,
            "engineering_canary_only": True,
        }
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_cli(args)
    output_dir = _fresh_output_directory(args.output_dir)

    # Validate the complete sealed donor closure before source decode, model
    # construction, output creation, or any attempt to render.
    manifest, manifest_path, manifest_file_sha = _load_json(args.factor_manifest, label="factor manifest")
    if manifest_file_sha != args.expected_factor_manifest_file_sha256:
        raise MotionDonorOracleError("factor manifest file SHA-256 differs")
    bank_receipt, bank_receipt_path, bank_receipt_file_sha = _load_json(
        args.factor_bank_receipt, label="factor bank receipt"
    )
    if bank_receipt_file_sha != args.expected_factor_bank_receipt_file_sha256:
        raise MotionDonorOracleError("factor bank receipt file SHA-256 differs")
    bound = bind_registered_donors(
        manifest=manifest,
        bank_receipt=bank_receipt,
        execution_group=args.execution_group,
    )
    if bank_receipt.get("manifest_file_sha256") != manifest_file_sha:
        raise MotionDonorOracleError("bank receipt manifest file SHA-256 binding differs")
    bank_root = _canonical_root(args.bank_output_root, label="bank output root")
    loaded_donors_cpu: dict[str, Any] = {}
    donor_provenance: dict[str, Any] = {}
    for branch in DONOR_BRANCHES:
        tensor, provenance = load_registered_clean_donor(
            row=bound["donor_rows"][branch], bank_root=bank_root
        )
        loaded_donors_cpu[branch] = tensor
        donor_provenance[branch] = provenance
    if len({row["clean_latent_raw_storage_sha256"] for row in donor_provenance.values()}) != len(DONOR_BRANCHES):
        raise MotionDonorOracleError("registered donor controls unexpectedly alias one clean tensor")

    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise MotionDonorOracleError("source video must be absolute and non-symlink")
    source_path = source_requested.resolve(strict=True)
    if source_path != source_requested or not source_path.is_file():
        raise MotionDonorOracleError("source video must be a canonical plain file")
    source_contract = bound["manifest"]["source_geometry_video"]
    if source_contract.get("sha256") != CDF_DOG_SOURCE_SHA256:
        raise MotionDonorOracleError("factor manifest is not bound to the CDF dog source")
    if file_sha256(source_path) != source_contract["sha256"]:
        raise MotionDonorOracleError("CDF source video SHA-256 differs from factor manifest")

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
        raise MotionDonorOracleError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise MotionDonorOracleError("Bernini attention heads do not divide Ulysses4")
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
    import bernini.models.transformer_wan as transformer_wan
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS

    if SYSTEM_PROMPTS.get("mv2v") != native.legacy.MV2V_SYSTEM_PROMPT:
        raise MotionDonorOracleError("runtime Bernini mv2v system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise MotionDonorOracleError("runtime Bernini negative prompt differs")

    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != ULYSSES_SIZE or distributed.ulysses_size != ULYSSES_SIZE:
        raise MotionDonorOracleError("runtime requires exact WORLD4/Ulysses4")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise MotionDonorOracleError("runtime requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise MotionDonorOracleError(f"checkpoint content validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = native.source_audit.prepare_hashed_source_snapshot(source_path)
    if source_sha != source_contract["sha256"]:
        raise MotionDonorOracleError("source snapshot SHA-256 differs")
    if tuple(source_metadata["source_derived_bucket_hw"]) != (HEIGHT, WIDTH):
        raise MotionDonorOracleError("source bucket geometry differs from registered exact81 bank")

    target_body = bound["target_prompt"]
    if hashlib.sha256(target_body.encode("utf-8")).hexdigest() != bound["target_prompt_sha256"]:
        raise MotionDonorOracleError("target full-action prompt SHA-256 differs")
    target_prompt = native.legacy.build_training_prompt(target_body, prompt_cleaner=prompt_clean)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    target_ids, target_mask = native.legacy._tokenize_training_prompt(tokenizer, target_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise MotionDonorOracleError(str(error)) from error
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise MotionDonorOracleError("renderer is not pinned UniPC shift5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = native.source_audit.model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    with torch.inference_mode():
        source_latent = _vae_encode(
            vae, source_tensor.to(device=device, dtype=torch.float32)
        ).contiguous()
    if tuple(int(item) for item in source_latent.shape) != LATENT_SHAPE:
        raise MotionDonorOracleError("source VAE latent shape differs")
    source_broadcast = native._broadcast_condition_from_rank_zero(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    source_identity = native._all_rank_tensor_identity(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    if source_identity["identity"]["raw_storage_sha256"] in {
        row["clean_latent_raw_storage_sha256"] for row in donor_provenance.values()
    }:
        raise MotionDonorOracleError("source condition aliases a registered donor clean tensor")

    donors: dict[str, Any] = {}
    donor_broadcasts: dict[str, Any] = {}
    donor_all_rank: dict[str, Any] = {}
    for branch in DONOR_BRANCHES:
        donor = loaded_donors_cpu[branch].to(device=device, dtype=torch.float32).contiguous()
        donor_broadcasts[branch] = native._broadcast_condition_from_rank_zero(
            donor, label=f"registered_{branch}_donor", world_size=ULYSSES_SIZE
        )
        donor_all_rank[branch] = native._all_rank_tensor_identity(
            donor, label=f"registered_{branch}_donor", world_size=ULYSSES_SIZE
        )
        if donor_all_rank[branch]["identity"]["raw_storage_sha256"] != donor_provenance[branch]["clean_latent_raw_storage_sha256"]:
            raise MotionDonorOracleError("GPU donor differs from registered FP32 clean latent")
        donors[branch] = donor

    vae.to("cpu")
    del source_tensor, loaded_donors_cpu
    torch.cuda.empty_cache()
    model.to(device)
    model.t5_text_encoder.to(device)
    with torch.inference_mode():
        prompt_embeds = model.encode_prompt(target_ids.to(device), target_mask.to(device))
        uncond_embeds = model.encode_prompt(negative_ids.to(device), negative_mask.to(device))
    model.t5_text_encoder.to("cpu")
    torch.cuda.empty_cache()

    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    try:
        wan_source_sha = sampler_contract.validate_runtime_source_identity(
            bernini_commit=bernini_revision,
            wan_diffusion_path=Path(wan_diffusion.__file__).resolve(),
        )
        sampler_contract._validate_scheduler_contract(
            diffusion.scheduler, expected_flow_shift=native.FLOW_SHIFT
        )
    except Exception as error:
        raise MotionDonorOracleError(str(error)) from error
    transformer_source_path = Path(transformer_wan.__file__).resolve()
    transformer_source_sha = file_sha256(transformer_source_path)
    if transformer_source_sha != PINNED_TRANSFORMER_WAN_SHA256:
        raise MotionDonorOracleError("bernini/models/transformer_wan.py differs from audited bytes")

    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    noise_captures: dict[str, Any] = {}
    noise_rank_identities: dict[str, Any] = {}
    arm_audits: dict[str, Any] = {}
    condition_orders: dict[str, Any] = {}
    source_condition_raw_sha = source_identity["identity"]["raw_storage_sha256"]
    donor_condition_raw_sha = {
        branch: donor_all_rank[branch]["identity"]["raw_storage_sha256"]
        for branch in DONOR_BRANCHES
    }
    with torch.inference_mode():
        for spec in arm_plan():
            conditions = _condition_list(spec, source=source_latent, donors=donors)
            if spec.arm_id == "D-action-order-swap":
                condition_raw_shas = [
                    donor_condition_raw_sha["full_action"], source_condition_raw_sha
                ]
            elif spec.arm_id == "D-duplicate-source":
                condition_raw_shas = [source_condition_raw_sha, source_condition_raw_sha]
            elif spec.donor_branch is not None:
                condition_raw_shas = [
                    source_condition_raw_sha,
                    donor_condition_raw_sha[spec.donor_branch],
                ]
            else:
                condition_raw_shas = [source_condition_raw_sha]
            condition_orders[spec.arm_id] = {
                "roles": list(spec.condition_roles),
                "source_ids": [index for index in range(1, len(conditions) + 1)],
                "latent_raw_storage_sha256_in_order": condition_raw_shas,
                "source_latent_raw_storage_sha256": source_condition_raw_sha,
                "second_video_latent_raw_storage_sha256": (
                    condition_raw_shas[1] if len(condition_raw_shas) == 2 else None
                ),
                "target_source_id": 0,
                "privileged_v_role": spec.privileged_v_role,
                "first_video_alone_enters_v": True,
                "all_videos_enter_vi": True,
                "image_condition_count": 0,
                "is_image_condition": False,
            }
            sample_kwargs = {
                "prompt_embeds": prompt_embeds,
                "uncond_prompt_embeds": uncond_embeds,
                "image_vae_latents": None,
                "multi_video_vae_latents": conditions,
                "multi_image_vae_latents": None,
                "width": WIDTH,
                "height": HEIGHT,
                "device": device,
                **_sampling_values(spec, steps=args.num_inference_steps),
            }
            observer: Optional[NativeMultiVideoConditionAudit] = None
            if spec.observed:
                observer = NativeMultiVideoConditionAudit(
                    diffusion,
                    condition_list=conditions,
                    condition_roles=spec.condition_roles,
                    guidance_mode=spec.guidance_mode,
                    expected_steps=args.num_inference_steps,
                    expected_seed=TARGET_SEED,
                    prompt_embeds=prompt_embeds,
                    uncond_prompt_embeds=uncond_embeds,
                )
                observer.install()
            try:
                result, noise_capture = native._sample_with_native_initial_noise_observer(
                    sample_fn=lambda kw=sample_kwargs: diffusion.sample(**kw),
                    wan_diffusion_module=wan_diffusion,
                    expected_shape=LATENT_SHAPE,
                    expected_device=device,
                    expected_seed=TARGET_SEED,
                )
            finally:
                if observer is not None:
                    observer.restore()
            if (
                not isinstance(result, torch.Tensor)
                or result.layout != torch.strided
                or result.device != device
                or result.dtype != torch.float32
                or result.requires_grad
                or result.grad_fn is not None
                or not result.is_contiguous()
                or not bool(torch.isfinite(result).all().item())
                or tuple(int(item) for item in result.shape) != LATENT_SHAPE
            ):
                raise MotionDonorOracleError(
                    f"{spec.arm_id} native sampler return must be detached contiguous finite FP32 on-device exact81"
                )
            # The native sampler is required to return contiguous FP32.  The
            # CPU copy changes only device, never dtype/layout/value bytes.
            generated_cpu = result.detach().to(device="cpu").contiguous()
            generated[spec.arm_id] = generated_cpu
            generated_identities[spec.arm_id] = native._all_rank_tensor_identity(
                generated_cpu, label=f"generated_{spec.arm_id}", world_size=ULYSSES_SIZE
            )
            noise_captures[spec.arm_id] = noise_capture
            noise_rank_identities[spec.arm_id] = native._all_rank_tensor_identity(
                noise_capture.tensor,
                label=f"official_initial_gaussian_{spec.arm_id}",
                world_size=ULYSSES_SIZE,
            )
            arm_audits[spec.arm_id] = (
                dict(observer.trace)
                if observer is not None
                else {
                    "observer_installed": False,
                    "multi_video_condition_audit_installed": False,
                    "initial_gaussian_capture_wrapper_installed": True,
                    "guidance_mode": spec.guidance_mode,
                    "step_count_declared_by_native_call": args.num_inference_steps,
                }
            )

    noise_hashes = {capture.raw_value_sha256 for capture in noise_captures.values()}
    if len(noise_hashes) != 1:
        raise MotionDonorOracleError("matched arms did not start from one byte-identical target Gaussian")
    o0_identity = generated_identities["O0"]["identity"]
    z0_identity = generated_identities["Z0"]["identity"]
    if o0_identity.get("raw_storage_sha256") != z0_identity.get("raw_storage_sha256") or not torch.equal(generated["O0"], generated["Z0"]):
        raise MotionDonorOracleError(
            "Z0 condition observer is not byte-exact to the condition-audit-free O0"
        )

    freeze_after = native.source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
        raise MotionDonorOracleError("frozen model certificate changed")
    model.to("cpu")
    torch.cuda.empty_cache()

    checkpoint_after_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_after_rows[0] = {
                "ok": True,
                "identity": native.source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            checkpoint_after_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_after_rows, src=0)
    if (
        not isinstance(checkpoint_after_rows[0], Mapping)
        or checkpoint_after_rows[0].get("ok") is not True
        or checkpoint_after_rows[0].get("identity") != checkpoint_identity
    ):
        raise MotionDonorOracleError("checkpoint content changed during runtime")

    local_evidence = {
        "rank": distributed.rank,
        "arm_audits_digest": object_sha256(arm_audits),
        "generated_identities_digest": object_sha256(generated_identities),
        "noise_raw_sha256": next(iter(noise_hashes)),
        "freeze_digest": object_sha256(freeze_after),
    }
    gathered: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != [0, 1, 2, 3]:
        raise MotionDonorOracleError("WORLD4 rank evidence closure differs")
    for field_name in ("arm_audits_digest", "generated_identities_digest", "noise_raw_sha256", "freeze_digest"):
        if len({row.get(field_name) for row in gathered if isinstance(row, Mapping)}) != 1:
            raise MotionDonorOracleError(f"WORLD4 ranks disagree on {field_name}")

    runtime_versions = {
        "torch": torch.__version__,
        "torch_hip": str(torch.version.hip),
        "transformers": transformers_version,
        "diffusers": diffusers_version,
    }
    if distributed.rank == 0:
        artifact_dir = _output_staging_directory(output_dir)
        noise_artifacts = {
            arm: native._save_initial_noise_atomically(
                artifact_dir / f"{arm}.official-initial-gaussian.safetensors",
                noise_captures[arm],
                all_rank_identity=noise_rank_identities[arm],
            )
            for arm in ARM_ORDER
        }
        source_artifact = native._save_normalized_clean_latent_atomically(
            artifact_dir / "source.normalized-clean-latent.safetensors",
            source_latent,
            artifact_role="source_video_condition",
        )
        outputs = _save_outputs(
            output_dir=artifact_dir,
            generated=generated,
            vae=vae,
            device=device,
            save_output_fn=save_output,
            steps=args.num_inference_steps,
        )
        arm_receipts = {}
        for spec in arm_plan():
            sampling = _sampling_values(spec, steps=args.num_inference_steps)
            condition_file_shas = []
            for role in spec.condition_roles:
                if role in {"source_video", "source_video_duplicate"}:
                    condition_file_shas.append(source_artifact["sha256"])
                else:
                    if spec.donor_branch is None:
                        raise MotionDonorOracleError(
                            "receipt arm role lacks registered donor provenance"
                        )
                    condition_file_shas.append(
                        donor_provenance[spec.donor_branch][
                            "clean_latent_file_sha256"
                        ]
                    )
            arm_receipts[spec.arm_id] = {
                **asdict(spec),
                "condition_order": condition_orders[spec.arm_id],
                "condition_latent_artifact_file_sha256_in_order": condition_file_shas,
                "provenance": {
                    "runtime_source_revision": args.runtime_source_revision,
                    "runtime_source_archive_sha256": args.runtime_source_archive_sha256,
                    "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                    "factor_manifest_digest": bound["manifest"]["manifest_digest"],
                    "factor_bank_receipt_digest": bound["bank_receipt_digest"],
                    "registered_donor_native_receipt_digest": (
                        donor_provenance[spec.donor_branch]["native_receipt_digest"]
                        if spec.donor_branch is not None
                        else None
                    ),
                },
                "sampling": sampling,
                "omega_vid": sampling["omega_vid"],
                "omega_img": sampling["omega_img"],
                "omega_txt": sampling["omega_txt"],
                "target_initialization": native.TARGET_INITIALIZATION,
                "target_mixed_with_source_or_donor": False,
                "donor_is_condition_not_target": spec.donor_branch is not None,
                "native_video_latent_condition": True,
                "image_condition": False,
                "audit": arm_audits[spec.arm_id],
            }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "stage": "engineering_oom_callpath_canary" if args.num_inference_steps == 1 else "matched_exact40_qualitative_causal_pilot",
            "execution_group": args.execution_group,
            "proposal_cell": bound["cell"],
            "runtime_source": {
                "revision": args.runtime_source_revision,
                "archive_sha256": args.runtime_source_archive_sha256,
                "launcher_sha256": args.launcher_source_sha256,
            },
            "pinned_sources": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "wan_diffusion_path": str(Path(wan_diffusion.__file__).resolve()),
                "wan_diffusion_sha256": wan_source_sha,
                "transformer_wan_path": str(transformer_source_path),
                "transformer_wan_sha256": transformer_source_sha,
                "bernini_inference_files": inference_file_hashes,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_before_and_after": checkpoint_identity,
                "unchanged": True,
            },
            "factor_bank": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": manifest_file_sha,
                "manifest_digest": bound["manifest"]["manifest_digest"],
                "bank_receipt_path": str(bank_receipt_path),
                "bank_receipt_file_sha256": bank_receipt_file_sha,
                "bank_receipt_digest": bound["bank_receipt_digest"],
                "bank_root": str(bank_root),
                "bank_renderer_source_revision": bound["manifest"]["renderer_contract"]["method_source_revision"],
                "bank_renderer_source_archive_sha256": bound["manifest"]["renderer_contract"]["method_source_archive_sha256"],
                "donors": donor_provenance,
                "donor_generation_gaussians_are_provenance_only": True,
                "donor_mp4_consumed": False,
            },
            "source": {
                "video_path": str(source_path),
                "video_sha256": source_sha,
                "metadata": source_metadata,
                "vae_condition_identity": source_identity,
                "rank_zero_broadcast": source_broadcast,
                "normalized_clean_latent_artifact": source_artifact,
            },
            "prompt": {
                "registered_full_action_body_sha256": bound["target_prompt_sha256"],
                "mv2v_system_prompt_sha256": hashlib.sha256(native.legacy.MV2V_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "full_prompt_sha256": hashlib.sha256(target_prompt.encode("utf-8")).hexdigest(),
                "same_prompt_all_arms": True,
                "prompt_body_from_registered_full_action_cell": True,
            },
            "matched_target": {
                "seed": TARGET_SEED,
                "frame_count": FRAME_COUNT,
                "latent_shape": list(LATENT_SHAPE),
                "height": HEIGHT,
                "width": WIDTH,
                "fps": FPS,
                "num_inference_steps": args.num_inference_steps,
                "same_target_gaussian_all_arms": True,
                "target_gaussian_raw_storage_sha256": next(iter(noise_hashes)),
                "external_target_or_target_latent": False,
            },
            "arms_in_execution_order": list(ARM_ORDER),
            "arms": arm_receipts,
            "donor_condition_broadcasts": donor_broadcasts,
            "donor_condition_all_rank_identities": donor_all_rank,
            "initial_noise_artifacts": noise_artifacts,
            "generated_identities": generated_identities,
            "outputs": outputs,
            "z0_wrapper_parity": {
                "o0_raw_storage_sha256": o0_identity["raw_storage_sha256"],
                "z0_raw_storage_sha256": z0_identity["raw_storage_sha256"],
                "byte_exact_fp32": True,
                "native_return_dtype_asserted_before_cpu_copy": "torch.float32",
                "o0_without_condition_shared_step_scheduler_hooks": True,
                "o0_initial_gaussian_capture_wrapper": True,
                "z0_observer_read_only": True,
            },
            "frozen_model": freeze_after,
            "world4_evidence": gathered,
            "runtime_versions": runtime_versions,
            "interpretation": {
                "training_performed": False,
                "optimizer": None,
                "backward": False,
                "model_weights_written": False,
                "initial_gaussian_capture_wrapper_all_arms": True,
                "o0_multi_video_shared_step_scheduler_audit_hooks": False,
                "donor_is_supervision_target": False,
                "registered_donor_label_proves_realized_motion": False,
                "action_success_evaluated": False,
                "identity_preservation_evaluated": False,
                "motion_transfer_evaluated": False,
                "quality_claim": False,
                "scientific_claim_authorized": False,
                "order_swap_is_pure_role_swap": False,
                "order_swap_jointly_changes_privileged_v_source_ids_and_order": True,
                "v2v_apg_anchor_is_old_base_sampler_binding_only": True,
                "one_step_stage_is_engineering_only": args.num_inference_steps == 1,
            },
        }
        receipt = _rebase_artifact_paths(
            receipt, old_root=artifact_dir, new_root=output_dir
        )
        receipt["receipt_digest"] = object_sha256(receipt)
        _write_receipt(artifact_dir / "receipt.json", receipt)
        _commit_output_transaction(staging=artifact_dir, final=output_dir)
        print(canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, donors, generated, noise_captures
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STEPS",
    "ARM_ORDER",
    "ARM_SPECS",
    "DONOR_BRANCHES",
    "CDF_DOG_SOURCE_SHA256",
    "LATENT_SHAPE",
    "METHOD",
    "MotionDonorOracleError",
    "NativeMultiVideoConditionAudit",
    "PATCH_TOKENS",
    "SCHEMA_VERSION",
    "TARGET_SEED",
    "arm_plan",
    "bind_registered_donors",
    "load_registered_clean_donor",
    "main",
]
