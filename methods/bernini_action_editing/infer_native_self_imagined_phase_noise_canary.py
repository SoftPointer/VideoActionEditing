#!/usr/bin/env python3
"""Frozen Bernini self-imagined phase-noise exact-81 canary.

This frozen ablation runner compares five *initial-noise* arms under one native Bernini
source-conditioned sampler.  The manifest-labelled references are receipt-bound
FP32 pre-decode T2V latents from one DMIQ factor-bank cell; their event semantics
have not been independently verified.  Proposal MP4 files are
never opened.  Every arm first calls Bernini's original module-global
``wan_diffusion.randn_tensor`` with the untouched arguments and generator.
The returned baseline Gaussian is then either forwarded unchanged or replaced
by a CPU-FP64 Phi-noise realization before native sampling continues.  This is
an explicit, reversible injection -- not a read-only observer.

The only renderer conditions are native R2V with five independently encoded
source frames, or native RV2V with the full source video plus four independently
encoded source frames.  There is no paired edit target, mask, flow, pose,
track, trajectory, donor RGB decode, optimization, or model-weight mutation.

``--num-inference-steps 1`` is only an engineering/OOM canary.  The matched
``40``-step stage is a qualitative mechanism pilot and authorizes no scientific
claim by itself.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
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
import infer_native_multivideo_motion_donor_oracle as donor_runtime  # noqa: E402
import self_imagined_phase_noise as phase_noise  # noqa: E402


METHOD = "frozen-bernini-self-imagined-phase-noise-canary"
SCHEMA_VERSION = "bernini-self-imagined-phase-noise-canary-receipt-v1"
FRAME_COUNT = 81
LATENT_SHAPE = (1, 16, 21, 62, 60)
FPS = 25
HEIGHT = 496
WIDTH = 480
ULYSSES_SIZE = 4
ALLOWED_STEPS = (1, 40)
DEFAULT_SEED = 20_260_810
CONDITION_MODES = ("r2v5", "rv2v4")
PROPOSAL_BRANCHES = ("full_action", "noop", "reverse_action")
SPATIAL_RADIUS = 3
GAMMA = 30.0
SOURCE_DC_RHO = 0.2
CDF_DOG_SOURCE_SHA256 = donor_runtime.CDF_DOG_SOURCE_SHA256

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PhaseNoiseCanaryError(RuntimeError):
    """Raised before ambiguous phase-noise evidence is published."""


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    proposal_branch: Optional[str]
    source_rho: float
    intervention: str


ARM_SPECS = (
    ArmSpec(
        "matched-gaussian", None, 0.0,
        "original_randn_tensor_return_forwarded_without_replacement",
    ),
    ArmSpec(
        "action-phi", "full_action", 0.0,
        "full_action_low_spatial_frequency_phase_injection",
    ),
    ArmSpec(
        "noop-phi", "noop", 0.0,
        "noop_low_spatial_frequency_phase_negative_control",
    ),
    ArmSpec(
        "reverse-phi", "reverse_action", 0.0,
        "reverse_action_low_spatial_frequency_phase_negative_control",
    ),
    ArmSpec(
        "action-phi-source-dc-rho02", "full_action", SOURCE_DC_RHO,
        "full_action_phase_plus_source_temporal_dc_identity_carrier",
    ),
)
ARM_ORDER = tuple(spec.arm_id for spec in ARM_SPECS)


@dataclass(frozen=True)
class PhaseNoiseInjectionCapture:
    baseline_tensor: Any
    injected_tensor: Any
    baseline_raw_storage_sha256: str
    injected_raw_storage_sha256: str
    baseline_content_sha256: str
    injected_content_sha256: str
    requested_shape: tuple[int, ...]
    requested_dtype: str
    requested_device: str
    returned_dtype: str
    returned_device: str
    generator_device: str
    generator_initial_seed: int
    call_count: int
    injection_performed: bool
    original_return_object_forwarded: bool
    operator_receipt: Mapping[str, Any]


def arm_plan() -> tuple[ArmSpec, ...]:
    if tuple(spec.arm_id for spec in ARM_SPECS) != ARM_ORDER:
        raise PhaseNoiseCanaryError("arm registry order changed")
    if len(set(ARM_ORDER)) != len(ARM_ORDER):
        raise PhaseNoiseCanaryError("arm registry contains duplicates")
    for spec in ARM_SPECS:
        if spec.proposal_branch is not None and spec.proposal_branch not in PROPOSAL_BRANCHES:
            raise PhaseNoiseCanaryError("arm requests an unregistered proposal branch")
        if spec.arm_id == "matched-gaussian":
            if spec.proposal_branch is not None or spec.source_rho != 0.0:
                raise PhaseNoiseCanaryError("matched Gaussian arm is not inert")
        elif spec.source_rho not in (0.0, SOURCE_DC_RHO):
            raise PhaseNoiseCanaryError("unregistered source-DC rho")
    return ARM_SPECS


def normalize_arms(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise PhaseNoiseCanaryError("at least one arm is required")
    names = tuple(str(value) for value in values)
    unknown = sorted(set(names) - set(ARM_ORDER))
    if unknown:
        raise PhaseNoiseCanaryError(f"unknown phase-noise arms: {unknown}")
    if len(set(names)) != len(names):
        raise PhaseNoiseCanaryError("phase-noise arm names must be unique")
    return tuple(arm for arm in ARM_ORDER if arm in names)


def spec_for_arm(arm: str) -> ArmSpec:
    for spec in arm_plan():
        if spec.arm_id == arm:
            return spec
    raise PhaseNoiseCanaryError(f"unknown arm: {arm!r}")


def condition_contract(mode: str) -> dict[str, Any]:
    if mode == "r2v5":
        return {
            "native_arm": "r2v",
            "guidance_mode": "r2v_apg",
            "full_source_video_count": 0,
            "source_reference_count": 5,
            "reference_indices": list(native.R2V_REFERENCE_INDICES),
        }
    if mode == "rv2v4":
        return {
            "native_arm": "rv2v",
            "guidance_mode": "rv2v",
            "full_source_video_count": 1,
            "source_reference_count": 4,
            "reference_indices": list(native.RV2V_REFERENCE_INDICES),
        }
    raise PhaseNoiseCanaryError(f"unknown native condition mode: {mode!r}")


def bind_registered_proposals(
    *, manifest: Mapping[str, Any], bank_receipt: Mapping[str, Any], execution_group: str,
) -> dict[str, Any]:
    """Bind one complete cell but expose only the three registered Phi references."""

    try:
        bound = bank_contract.validate_micro_bank_bindings(
            manifest, bank_receipt, execution_group=execution_group
        )
        bank_contract.validate_renderer_contract(bound["manifest"]["renderer_contract"])
    except Exception as error:
        raise PhaseNoiseCanaryError(str(error)) from error
    rows = {row["manifest"]["semantic_branch"]: row for row in bound["prompt_rows"]}
    if any(branch not in rows for branch in PROPOSAL_BRANCHES):
        raise PhaseNoiseCanaryError("registered factor cell lacks a required Phi proposal")
    full_action = rows["full_action"]["manifest"]
    prompt = full_action.get("prompt")
    prompt_sha = full_action.get("prompt_utf8_sha256")
    if not isinstance(prompt, str) or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha:
        raise PhaseNoiseCanaryError("registered full-action prompt binding differs")
    return {
        "manifest": bound["manifest"],
        "bank_receipt_digest": bound["bank_receipt_digest"],
        "cell": bound["cell"],
        "target_prompt": prompt,
        "target_prompt_sha256": prompt_sha,
        "proposal_rows": {branch: rows[branch] for branch in PROPOSAL_BRANCHES},
    }


def _tensor_identity(value: Any, *, label: str) -> dict[str, Any]:
    try:
        return dict(native.value_audit.tensor_identity(value, label=label))
    except Exception as error:
        raise PhaseNoiseCanaryError(str(error)) from error


def _sample_with_phase_noise_injection(
    *,
    sample_fn: Callable[[], Any],
    wan_diffusion_module: Any,
    arm_spec: ArmSpec,
    proposal_reference_cpu: Optional[Any],
    source_normalized_latent_cpu: Any,
    expected_shape: Sequence[int],
    expected_device: Any,
    expected_seed: int,
    canonical_randn_tensor: Optional[Callable[..., Any]] = None,
) -> tuple[Any, PhaseNoiseInjectionCapture]:
    """Replace Bernini's initial Gaussian only after its native RNG call.

    The operator always performs its FFT and source bridge on contiguous CPU
    FP64 tensors.  The tensor returned to Bernini is converted back to the
    exact native FP32 device contract.  The module global is restored on every
    exit path, and a successful call must invoke it exactly once.
    """

    try:
        import torch
        if canonical_randn_tensor is None:
            from diffusers.utils.torch_utils import randn_tensor as canonical
        else:
            canonical = canonical_randn_tensor
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise PhaseNoiseCanaryError("phase-noise injection requires PyTorch and Diffusers") from error
    if not callable(sample_fn):
        raise PhaseNoiseCanaryError("sample_fn must be callable")
    expected = tuple(int(item) for item in expected_shape)
    if expected != LATENT_SHAPE:
        raise PhaseNoiseCanaryError("phase-noise canary requires exact81 latent geometry")
    if type(expected_seed) is not int or not 0 <= expected_seed < 2**63:
        raise PhaseNoiseCanaryError("expected seed is invalid")
    original = getattr(wan_diffusion_module, "randn_tensor", None)
    if original is not canonical:
        raise PhaseNoiseCanaryError("pinned wan_diffusion.randn_tensor is already replaced or differs")

    source_cpu = source_normalized_latent_cpu
    if (
        not isinstance(source_cpu, torch.Tensor)
        or source_cpu.device.type != "cpu"
        or source_cpu.dtype != torch.float32
        or tuple(int(item) for item in source_cpu.shape) != expected
        or not source_cpu.is_contiguous()
        or source_cpu.requires_grad
        or not bool(torch.isfinite(source_cpu).all().item())
    ):
        raise PhaseNoiseCanaryError("source normalized latent must be exact contiguous CPU FP32")
    if arm_spec.proposal_branch is None:
        if proposal_reference_cpu is not None:
            raise PhaseNoiseCanaryError("matched Gaussian arm must not receive a proposal")
    elif (
        not isinstance(proposal_reference_cpu, torch.Tensor)
        or proposal_reference_cpu.device.type != "cpu"
        or proposal_reference_cpu.dtype != torch.float32
        or tuple(int(item) for item in proposal_reference_cpu.shape) != expected
        or not proposal_reference_cpu.is_contiguous()
        or proposal_reference_cpu.requires_grad
        or not bool(torch.isfinite(proposal_reference_cpu).all().item())
    ):
        raise PhaseNoiseCanaryError("proposal reference must be exact contiguous CPU FP32")

    calls: list[dict[str, Any]] = []

    def injected_randn_tensor(*call_args: Any, **call_kwargs: Any) -> Any:
        shape_value = call_args[0] if call_args else call_kwargs.get("shape")
        try:
            requested_shape = tuple(int(item) for item in shape_value)
        except Exception as error:
            raise PhaseNoiseCanaryError("native randn_tensor call has no valid shape") from error
        generator = call_kwargs.get("generator")
        if not isinstance(generator, torch.Generator):
            raise PhaseNoiseCanaryError("native initial noise must use one torch.Generator")
        requested_device = str(call_kwargs.get("device"))
        requested_dtype = str(call_kwargs.get("dtype"))

        # Call first, with the exact original objects.  Only then construct the
        # intervention from the realized baseline Gaussian.
        baseline_native = original(*call_args, **call_kwargs)
        if not isinstance(baseline_native, torch.Tensor):
            raise PhaseNoiseCanaryError("native randn_tensor did not return a tensor")
        if (
            tuple(int(item) for item in baseline_native.shape) != expected
            or baseline_native.dtype != torch.float32
            or str(baseline_native.device) != str(torch.device(expected_device))
            or not baseline_native.is_contiguous()
            or baseline_native.requires_grad
            or not bool(torch.isfinite(baseline_native).all().item())
        ):
            raise PhaseNoiseCanaryError("native baseline Gaussian contract differs")
        baseline_cpu = baseline_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if arm_spec.proposal_branch is None:
            injected_native = baseline_native
            operator_receipt: Mapping[str, Any] = {
                "operator": "identity",
                "original_native_gaussian_object_forwarded": True,
                "injection_performed": False,
                "spatial_radius": None,
                "gamma": None,
                "source_rho": 0.0,
            }
        else:
            try:
                result = phase_noise.build_factorized_phase_noise(
                    baseline_cpu.to(dtype=torch.float64).contiguous(),
                    proposal_reference_cpu.to(dtype=torch.float64).contiguous(),
                    source_cpu.to(dtype=torch.float64).contiguous(),
                    spatial_radius=SPATIAL_RADIUS,
                    gamma=GAMMA,
                    source_rho=arm_spec.source_rho,
                )
            except phase_noise.SelfImaginedPhaseNoiseError as error:
                raise PhaseNoiseCanaryError(str(error)) from error
            injected_native = result.initial_noise.to(
                device=baseline_native.device, dtype=baseline_native.dtype
            ).contiguous()
            if injected_native is baseline_native or torch.equal(injected_native, baseline_native):
                raise PhaseNoiseCanaryError("active Phi arm did not alter native initial noise")
            operator_receipt = dict(result.receipt)
        injected_cpu = injected_native.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        baseline_identity = _tensor_identity(baseline_cpu, label="native_baseline_gaussian")
        injected_identity = _tensor_identity(injected_cpu, label="returned_injected_initial_noise")
        calls.append({
            "baseline_tensor": baseline_cpu,
            "injected_tensor": injected_cpu,
            "baseline_identity": baseline_identity,
            "injected_identity": injected_identity,
            "requested_shape": requested_shape,
            "requested_dtype": requested_dtype,
            "requested_device": requested_device,
            "returned_dtype": str(baseline_native.dtype),
            "returned_device": str(baseline_native.device),
            "generator_device": str(generator.device),
            "generator_initial_seed": int(generator.initial_seed()),
            "operator_receipt": operator_receipt,
            "injection_performed": arm_spec.proposal_branch is not None,
            "original_return_object_forwarded": injected_native is baseline_native,
        })
        return injected_native

    setattr(injected_randn_tensor, "_bernini_phase_noise_injector", True)
    setattr(wan_diffusion_module, "randn_tensor", injected_randn_tensor)
    wrapper_unchanged = True
    try:
        sample_result = sample_fn()
    finally:
        wrapper_unchanged = getattr(wan_diffusion_module, "randn_tensor", None) is injected_randn_tensor
        setattr(wan_diffusion_module, "randn_tensor", original)
    if not wrapper_unchanged:
        raise PhaseNoiseCanaryError("wan_diffusion.randn_tensor changed while injector was active")
    if getattr(wan_diffusion_module, "randn_tensor", None) is not original:
        raise PhaseNoiseCanaryError("wan_diffusion.randn_tensor restoration failed")
    if len(calls) != 1:
        raise PhaseNoiseCanaryError(f"native sampler must call randn_tensor exactly once; observed {len(calls)}")
    call = calls[0]
    expected_device_text = str(torch.device(expected_device))
    if (
        call["requested_shape"] != expected
        or call["requested_dtype"] != str(torch.float32)
        or call["requested_device"] != expected_device_text
        or call["returned_dtype"] != str(torch.float32)
        or call["returned_device"] != expected_device_text
        or call["generator_device"] != "cpu"
        or call["generator_initial_seed"] != expected_seed
    ):
        raise PhaseNoiseCanaryError("native randn_tensor request/seed contract differs")
    baseline_identity = call["baseline_identity"]
    injected_identity = call["injected_identity"]
    return sample_result, PhaseNoiseInjectionCapture(
        baseline_tensor=call["baseline_tensor"],
        injected_tensor=call["injected_tensor"],
        baseline_raw_storage_sha256=str(baseline_identity["raw_storage_sha256"]),
        injected_raw_storage_sha256=str(injected_identity["raw_storage_sha256"]),
        baseline_content_sha256=str(baseline_identity["content_sha256"]),
        injected_content_sha256=str(injected_identity["content_sha256"]),
        requested_shape=call["requested_shape"],
        requested_dtype=call["requested_dtype"],
        requested_device=call["requested_device"],
        returned_dtype=call["returned_dtype"],
        returned_device=call["returned_device"],
        generator_device=call["generator_device"],
        generator_initial_seed=call["generator_initial_seed"],
        call_count=1,
        injection_performed=bool(call["injection_performed"]),
        original_return_object_forwarded=bool(call["original_return_object_forwarded"]),
        operator_receipt=call["operator_receipt"],
    )


def _save_noise_atomically(path: Path, tensor: Any, *, tensor_key: str, role: str) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise PhaseNoiseCanaryError("noise artifact path must be fresh safetensors")
    if tensor_key not in {"baseline_gaussian", "injected_initial_noise"}:
        raise PhaseNoiseCanaryError("noise tensor key is not registered")
    stored = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(int(item) for item in stored.shape) != LATENT_SHAPE or not bool(torch.isfinite(stored).all().item()):
        raise PhaseNoiseCanaryError("noise artifact is not finite exact81 FP32")
    identity = _tensor_identity(stored, label=f"{role}_before_save")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            {tensor_key: stored}, str(temporary),
            metadata={
                "coordinate": "bernini_native_target_latent_before_rearrange",
                "role": role,
                "injection_not_observation": "true" if tensor_key == "injected_initial_noise" else "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [tensor_key]:
                raise PhaseNoiseCanaryError("noise safetensors key differs")
            restored = opened.get_tensor(tensor_key).contiguous()
            metadata = dict(opened.metadata() or {})
        if not torch.equal(restored, stored):
            raise PhaseNoiseCanaryError("noise safetensors round trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": donor_runtime.file_sha256(path),
        "tensor_key": tensor_key,
        "tensor_raw_storage_sha256": identity["raw_storage_sha256"],
        "tensor_content_sha256": identity["content_sha256"],
        "shape": list(LATENT_SHAPE),
        "dtype": str(stored.dtype),
        "role": role,
        "metadata": metadata,
        "roundtrip_byte_exact_fp32": True,
    }


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
    parser.add_argument("--condition-mode", required=True, choices=CONDITION_MODES)
    parser.add_argument("--arms", nargs="+", required=True, choices=ARM_ORDER)
    parser.add_argument("--num-inference-steps", type=int, required=True, choices=ALLOWED_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-source-archive-sha256", required=True)
    parser.add_argument("--launcher-source-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", default=native.legacy.trainer.BERNINI_OFFICIAL_COMMIT)
    parser.add_argument("--expected-veomni-commit", default=native.legacy.trainer.VEOMNI_TESTED_COMMIT)
    parser.add_argument("--expected-checkpoint-tree-sha256", default=native.legacy.trainer.CHECKPOINT_TREE_SHA256)
    return parser


def _validate_cli(args: argparse.Namespace) -> tuple[str, ...]:
    arms = normalize_arms(args.arms)
    condition_contract(args.condition_mode)
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise PhaseNoiseCanaryError("seed must lie in [0,2^63)")
    for name in ("runtime_source_revision", "expected_bernini_commit", "expected_veomni_commit"):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise PhaseNoiseCanaryError(f"{name} must be a full lowercase SHA-1")
    for name in (
        "runtime_source_archive_sha256", "launcher_source_sha256",
        "expected_factor_manifest_file_sha256", "expected_factor_bank_receipt_file_sha256",
        "expected_checkpoint_tree_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise PhaseNoiseCanaryError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT:
        raise PhaseNoiseCanaryError("Bernini commit differs from pinned release")
    if args.expected_veomni_commit != native.legacy.trainer.VEOMNI_TESTED_COMMIT:
        raise PhaseNoiseCanaryError("VeOmni commit differs from pinned release")
    if args.expected_checkpoint_tree_sha256 != native.legacy.trainer.CHECKPOINT_TREE_SHA256:
        raise PhaseNoiseCanaryError("checkpoint tree differs from pinned release")
    return arms


def _sampling_contract(condition_mode: str, *, steps: int, seed: int) -> dict[str, Any]:
    native_arm = condition_contract(condition_mode)["native_arm"]
    return native.native_sampling_contract(native_arm, steps=steps, seed=seed)


def _output_directory(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/") or _SAFE_NAME.fullmatch(requested.name) is None:
        raise PhaseNoiseCanaryError("output directory must be an absolute safe non-root path")
    parent = requested.parent.resolve(strict=True)
    final = parent / requested.name
    if parent.is_symlink() or final.exists() or final.is_symlink():
        raise PhaseNoiseCanaryError("refusing invalid or reused output directory")
    return final


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    arms = _validate_cli(args)
    output_dir = _output_directory(args.output_dir)

    manifest, manifest_path, manifest_file_sha = donor_runtime._load_json(
        args.factor_manifest, label="factor manifest"
    )
    if manifest_file_sha != args.expected_factor_manifest_file_sha256:
        raise PhaseNoiseCanaryError("factor manifest file SHA-256 differs")
    bank_receipt, bank_receipt_path, bank_receipt_file_sha = donor_runtime._load_json(
        args.factor_bank_receipt, label="factor bank receipt"
    )
    if bank_receipt_file_sha != args.expected_factor_bank_receipt_file_sha256:
        raise PhaseNoiseCanaryError("factor bank receipt file SHA-256 differs")
    bound = bind_registered_proposals(
        manifest=manifest, bank_receipt=bank_receipt, execution_group=args.execution_group
    )
    if bank_receipt.get("manifest_file_sha256") != manifest_file_sha:
        raise PhaseNoiseCanaryError("factor-bank manifest file binding differs")
    bank_root = donor_runtime._canonical_root(args.bank_output_root, label="bank output root")
    proposals_cpu: dict[str, Any] = {}
    proposal_provenance: dict[str, Any] = {}
    for branch in PROPOSAL_BRANCHES:
        proposal, provenance = donor_runtime.load_registered_clean_donor(
            row=bound["proposal_rows"][branch], bank_root=bank_root
        )
        proposals_cpu[branch] = proposal
        proposal_provenance[branch] = provenance
    if len({row["clean_latent_raw_storage_sha256"] for row in proposal_provenance.values()}) != 3:
        raise PhaseNoiseCanaryError("registered Phi proposals unexpectedly alias")

    source_requested = Path(args.source_video).expanduser()
    if not source_requested.is_absolute() or source_requested.is_symlink():
        raise PhaseNoiseCanaryError("source video must be absolute and non-symlink")
    source_path = source_requested.resolve(strict=True)
    if source_path != source_requested or not source_path.is_file() or source_path.is_symlink():
        raise PhaseNoiseCanaryError("source video must be a canonical plain file")
    source_contract = bound["manifest"].get("source_geometry_video")
    if not isinstance(source_contract, Mapping) or source_contract.get("sha256") != CDF_DOG_SOURCE_SHA256:
        raise PhaseNoiseCanaryError("factor manifest is not bound to the CDF dog source")
    if donor_runtime.file_sha256(source_path) != CDF_DOG_SOURCE_SHA256:
        raise PhaseNoiseCanaryError("CDF dog source SHA-256 differs")

    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = native.legacy.trainer.validate_source_trees(
            args.bernini_root, args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(args.checkpoint)
    except Exception as error:
        raise PhaseNoiseCanaryError(str(error)) from error
    if int(transformer_config["num_attention_heads"]) % ULYSSES_SIZE:
        raise PhaseNoiseCanaryError("Bernini attention heads do not divide Ulysses4")
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
    from bernini.training.data import SYSTEM_PROMPTS

    contract = condition_contract(args.condition_mode)
    native_arm = contract["native_arm"]
    task_name = native.ARM_TRAINING_TASK_NAMES[native_arm]
    if SYSTEM_PROMPTS.get(task_name) != native.TASK_SYSTEM_PROMPTS[task_name]:
        raise PhaseNoiseCanaryError("runtime native condition system prompt differs")
    if DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT:
        raise PhaseNoiseCanaryError("runtime Bernini negative prompt differs")

    distributed = native.legacy.inference_distributed_contract()
    if distributed.world_size != ULYSSES_SIZE or distributed.ulysses_size != ULYSSES_SIZE:
        raise PhaseNoiseCanaryError("runtime requires exact WORLD4/Ulysses4")
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise PhaseNoiseCanaryError("runtime requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=240),
        rank=distributed.rank, world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=ULYSSES_SIZE)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_manifest = Path(args.checkpoint_content_manifest).expanduser()
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            checkpoint_rows[0] = {"ok": True, "identity": native.source_audit.validate_checkpoint_content(checkpoint, checkpoint_manifest)}
        except Exception as error:
            checkpoint_rows[0] = {"ok": False, "error": str(error)}
    dist.broadcast_object_list(checkpoint_rows, src=0)
    if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
        raise PhaseNoiseCanaryError(f"checkpoint validation failed: {checkpoint_rows[0]}")
    checkpoint_identity = dict(checkpoint_rows[0]["identity"])

    source_tensor, source_metadata, source_sha = native.source_audit.prepare_hashed_source_snapshot(source_path)
    if source_sha != CDF_DOG_SOURCE_SHA256:
        raise PhaseNoiseCanaryError("source snapshot SHA-256 differs")
    if tuple(source_metadata["source_derived_bucket_hw"]) != (HEIGHT, WIDTH):
        raise PhaseNoiseCanaryError("source bucket differs from registered exact81 geometry")
    prompt_body = bound["target_prompt"]
    task_prompt = native.build_task_prompt(native_arm, prompt_body, prompt_cleaner=prompt_clean)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **native.legacy.tokenizer_load_kwargs()
    )
    positive_ids, positive_mask = native.legacy._tokenize_training_prompt(tokenizer, task_prompt)
    negative_ids, negative_mask = native.legacy._tokenize_renderer_negative(
        tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True, **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except Exception as error:
        raise PhaseNoiseCanaryError(str(error)) from error
    if float(config.shift) != native.FLOW_SHIFT or config.use_unipc is not True:
        raise PhaseNoiseCanaryError("renderer is not pinned UniPC shift5")
    model = BerniniRendererModel(config)
    model.requires_grad_(False)
    model.eval()
    freeze_before = native.source_audit.model_freeze_certificate(model)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    vae.eval().requires_grad_(False)
    vae.to(device)
    source_pixels = source_tensor.to(device=device, dtype=torch.float32)
    reference_indices = tuple(int(item) for item in contract["reference_indices"])
    with torch.inference_mode():
        source_latent = _vae_encode(vae, source_pixels).contiguous()
        reference_latents = {
            index: _vae_encode(
                vae, source_pixels[:, :, index:index + 1].contiguous()
            ).contiguous()
            for index in reference_indices
        }
    if tuple(int(item) for item in source_latent.shape) != LATENT_SHAPE:
        raise PhaseNoiseCanaryError("source VAE latent shape differs")
    for index, latent in reference_latents.items():
        if tuple(int(item) for item in latent.shape) != (1, 16, 1, 62, 60):
            raise PhaseNoiseCanaryError(f"source reference {index} VAE shape differs")
    source_broadcast = native._broadcast_condition_from_rank_zero(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    reference_broadcasts = {
        str(index): native._broadcast_condition_from_rank_zero(
            latent, label=f"cdf_source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, latent in reference_latents.items()
    }
    source_identity = native._all_rank_tensor_identity(
        source_latent, label="cdf_source_video", world_size=ULYSSES_SIZE
    )
    reference_identities = {
        str(index): native._all_rank_tensor_identity(
            latent, label=f"cdf_source_reference_{index}", world_size=ULYSSES_SIZE
        )
        for index, latent in reference_latents.items()
    }
    source_latent_cpu = source_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    source_cpu_identity = native._all_rank_tensor_identity(
        source_latent_cpu, label="cdf_source_phase_carrier_cpu", world_size=ULYSSES_SIZE
    )
    proposal_all_rank = {
        branch: native._all_rank_tensor_identity(
            proposal, label=f"registered_{branch}_phase_reference", world_size=ULYSSES_SIZE
        )
        for branch, proposal in proposals_cpu.items()
    }
    for branch in PROPOSAL_BRANCHES:
        if proposal_all_rank[branch]["identity"]["raw_storage_sha256"] != proposal_provenance[branch]["clean_latent_raw_storage_sha256"]:
            raise PhaseNoiseCanaryError("loaded proposal differs from registered FP32 clean latent")

    vae.to("cpu")
    del source_tensor, source_pixels
    torch.cuda.empty_cache()
    model.to(device)

    condition_kwargs = {
        "image_vae_latents": None,
        "multi_video_vae_latents": [source_latent] if args.condition_mode == "rv2v4" else None,
        "multi_image_vae_latents": [reference_latents[index] for index in reference_indices],
    }
    generated: dict[str, Any] = {}
    generated_identities: dict[str, Any] = {}
    captures: dict[str, PhaseNoiseInjectionCapture] = {}
    baseline_rank_identities: dict[str, Any] = {}
    injected_rank_identities: dict[str, Any] = {}
    with torch.inference_mode():
        for arm in arms:
            spec = spec_for_arm(arm)
            proposal = proposals_cpu.get(spec.proposal_branch) if spec.proposal_branch else None
            result, capture = _sample_with_phase_noise_injection(
                sample_fn=lambda: model.sample(
                    input_ids=positive_ids.to(device),
                    attention_mask=positive_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    **condition_kwargs,
                    width=WIDTH, height=HEIGHT, device=device,
                    **_sampling_contract(args.condition_mode, steps=args.num_inference_steps, seed=args.seed),
                ),
                wan_diffusion_module=wan_diffusion,
                arm_spec=spec,
                proposal_reference_cpu=proposal,
                source_normalized_latent_cpu=source_latent_cpu,
                expected_shape=LATENT_SHAPE,
                expected_device=device,
                expected_seed=args.seed,
            )
            if (
                not isinstance(result, torch.Tensor) or result.dtype != torch.float32
                or result.device != device or result.requires_grad or result.grad_fn is not None
                or not result.is_contiguous() or not bool(torch.isfinite(result).all().item())
                or tuple(int(item) for item in result.shape) != LATENT_SHAPE
            ):
                raise PhaseNoiseCanaryError(f"{arm} native sampler return contract differs")
            generated_cpu = result.detach().to(device="cpu").contiguous()
            generated[arm] = generated_cpu
            generated_identities[arm] = native._all_rank_tensor_identity(
                generated_cpu, label=f"generated_{arm}", world_size=ULYSSES_SIZE
            )
            captures[arm] = capture
            baseline_rank_identities[arm] = native._all_rank_tensor_identity(
                capture.baseline_tensor, label=f"baseline_gaussian_{arm}", world_size=ULYSSES_SIZE
            )
            injected_rank_identities[arm] = native._all_rank_tensor_identity(
                capture.injected_tensor, label=f"injected_initial_noise_{arm}", world_size=ULYSSES_SIZE
            )

    baseline_hashes = {capture.baseline_raw_storage_sha256 for capture in captures.values()}
    if len(baseline_hashes) != 1:
        raise PhaseNoiseCanaryError("matched arms did not receive one byte-identical native Gaussian")
    for arm, capture in captures.items():
        if arm == "matched-gaussian":
            if capture.injection_performed or not capture.original_return_object_forwarded:
                raise PhaseNoiseCanaryError("matched Gaussian arm was not an exact native pass-through")
            if capture.baseline_raw_storage_sha256 != capture.injected_raw_storage_sha256:
                raise PhaseNoiseCanaryError("matched Gaussian injected/baseline hashes differ")
        elif not capture.injection_performed or capture.original_return_object_forwarded:
            raise PhaseNoiseCanaryError(f"{arm} was not an explicit noise injection")

    freeze_after = native.source_audit.model_freeze_certificate(model)
    if freeze_after != freeze_before or any(parameter.requires_grad for parameter in model.parameters()):
        raise PhaseNoiseCanaryError("frozen model certificate changed")
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
        raise PhaseNoiseCanaryError("checkpoint content changed during frozen ablation")

    local_evidence = {
        "rank": distributed.rank,
        "arms": list(arms),
        "baseline_raw_storage_sha256": next(iter(baseline_hashes)),
        "injected_hashes": {arm: captures[arm].injected_raw_storage_sha256 for arm in arms},
        "generated_hashes": {
            arm: generated_identities[arm]["identity"]["raw_storage_sha256"] for arm in arms
        },
        "source_hash": source_cpu_identity["identity"]["raw_storage_sha256"],
        "proposal_hashes": {
            branch: proposal_all_rank[branch]["identity"]["raw_storage_sha256"]
            for branch in PROPOSAL_BRANCHES
        },
        "operator_receipt_digests": {
            arm: donor_runtime.object_sha256(captures[arm].operator_receipt)
            for arm in arms
        },
        "freeze_digest": donor_runtime.object_sha256(freeze_after),
    }
    gathered: list[Any] = [None] * ULYSSES_SIZE
    dist.all_gather_object(gathered, local_evidence)
    if sorted(row.get("rank") for row in gathered if isinstance(row, Mapping)) != [0, 1, 2, 3]:
        raise PhaseNoiseCanaryError("WORLD4 rank evidence closure differs")
    comparison = dict(local_evidence)
    comparison.pop("rank")
    if any({key: value for key, value in row.items() if key != "rank"} != comparison for row in gathered):
        raise PhaseNoiseCanaryError("WORLD4 ranks disagree on phase-noise evidence")

    runtime_versions = {
        "torch": torch.__version__, "torch_hip": str(torch.version.hip),
        "transformers": transformers_version, "diffusers": diffusers_version,
    }
    if distributed.rank == 0:
        staging = donor_runtime._output_staging_directory(output_dir)
        baseline_artifacts = {
            arm: _save_noise_atomically(
                staging / f"{arm}.baseline-gaussian.safetensors",
                captures[arm].baseline_tensor,
                tensor_key="baseline_gaussian", role="native_baseline_gaussian_before_injection",
            )
            for arm in arms
        }
        injected_artifacts = {
            arm: _save_noise_atomically(
                staging / f"{arm}.injected-initial-noise.safetensors",
                captures[arm].injected_tensor,
                tensor_key="injected_initial_noise", role=(
                    "matched_native_gaussian_pass_through" if arm == "matched-gaussian"
                    else "explicit_phi_noise_injected_into_native_sampler"
                ),
            )
            for arm in arms
        }
        source_artifact = native._save_normalized_clean_latent_atomically(
            staging / "source.normalized-clean-latent.safetensors",
            source_latent_cpu, artifact_role="source_video_condition",
        )
        proposal_artifacts = {
            branch: native._save_normalized_clean_latent_atomically(
                staging / f"proposal-{branch}.normalized-clean-latent.safetensors",
                proposals_cpu[branch], artifact_role="native_sampler_proposal",
            )
            for branch in PROPOSAL_BRANCHES
        }
        outputs = donor_runtime._save_outputs(
            output_dir=staging, generated=generated, vae=vae,
            device=device, save_output_fn=save_output, steps=args.num_inference_steps,
        )
        arm_receipts = {}
        for arm in arms:
            spec = spec_for_arm(arm)
            capture = captures[arm]
            arm_receipts[arm] = {
                **asdict(spec),
                "native_condition": dict(contract),
                "sampling": _sampling_contract(
                    args.condition_mode, steps=args.num_inference_steps, seed=args.seed
                ),
                "proposal": (
                    {
                        "semantic_branch": spec.proposal_branch,
                        "registered_provenance": proposal_provenance[spec.proposal_branch],
                        "copied_artifact": proposal_artifacts[spec.proposal_branch],
                    }
                    if spec.proposal_branch is not None else None
                ),
                "baseline_noise": baseline_artifacts[arm],
                "injected_noise": injected_artifacts[arm],
                "baseline_raw_storage_sha256": capture.baseline_raw_storage_sha256,
                "injected_raw_storage_sha256": capture.injected_raw_storage_sha256,
                "injection_performed": capture.injection_performed,
                "original_randn_tensor_called_first_with_unchanged_arguments": True,
                "original_return_object_forwarded": capture.original_return_object_forwarded,
                "module_global_replaced_temporarily": True,
                "module_global_restored": True,
                "wrapper_is_injector_not_observer": True,
                "operator_receipt": dict(capture.operator_receipt),
            }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "ablation_only": True,
            "stage": (
                "engineering_oom_callpath_canary" if args.num_inference_steps == 1
                else "matched_exact40_qualitative_phase_noise_pilot"
            ),
            "execution_group": args.execution_group,
            "arms_in_execution_order": list(arms),
            "runtime_source": {
                "revision": args.runtime_source_revision,
                "archive_sha256": args.runtime_source_archive_sha256,
                "launcher_sha256": args.launcher_source_sha256,
            },
            "pinned_sources": {
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "bernini_inference_files": inference_file_hashes,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "tree_sha256": args.expected_checkpoint_tree_sha256,
                "content_before_and_after": checkpoint_identity,
                "unchanged": True,
                "frozen": True,
            },
            "factor_bank": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": manifest_file_sha,
                "manifest_digest": bound["manifest"]["manifest_digest"],
                "bank_receipt_path": str(bank_receipt_path),
                "bank_receipt_file_sha256": bank_receipt_file_sha,
                "bank_receipt_digest": bound["bank_receipt_digest"],
                "bank_root": str(bank_root),
                "proposal_cell": bound["cell"],
                "proposal_branches_loaded": list(PROPOSAL_BRANCHES),
                "proposal_provenance": proposal_provenance,
                "proposal_mp4_consumed": False,
                "only_predecode_fp32_normalized_clean_latents_consumed": True,
                "manifest_branch_labels_are_not_independent_event_verification": True,
                "proposal_semantics_verified": False,
            },
            "source": {
                "video_path": str(source_path), "video_sha256": source_sha,
                "metadata": source_metadata,
                "exact81_25fps": True,
                "native_condition": dict(contract),
                "source_video_identity": source_identity,
                "source_reference_identities": reference_identities,
                "source_rank_zero_broadcast": source_broadcast,
                "reference_rank_zero_broadcasts": reference_broadcasts,
                "phase_carrier_cpu_identity": source_cpu_identity,
                "normalized_clean_latent_artifact": source_artifact,
            },
            "prompt": {
                "registered_full_action_body_sha256": bound["target_prompt_sha256"],
                "native_task": task_name,
                "full_prompt_sha256": hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
                "same_prompt_all_arms": True,
            },
            "matched_design": {
                "seed": args.seed, "frame_count": FRAME_COUNT, "fps": FPS,
                "height": HEIGHT, "width": WIDTH, "latent_shape": list(LATENT_SHAPE),
                "num_inference_steps": args.num_inference_steps,
                "spatial_radius": SPATIAL_RADIUS, "gamma": GAMMA,
                "source_dc_rho": SOURCE_DC_RHO,
                "same_native_baseline_gaussian_all_arms": True,
                "baseline_raw_storage_sha256": next(iter(baseline_hashes)),
                "cpu_fp64_operator": True,
                "returned_sampler_noise_fp32": True,
            },
            "arms": arm_receipts,
            "proposal_all_rank_identities": proposal_all_rank,
            "baseline_all_rank_identities": baseline_rank_identities,
            "injected_all_rank_identities": injected_rank_identities,
            "generated_identities": generated_identities,
            "outputs": outputs,
            "frozen_model": freeze_after,
            "world4_evidence": gathered,
            "runtime_versions": runtime_versions,
            "forbidden_inputs": {
                "paired_edit_target": False,
                "mask": False, "flow": False, "pose": False,
                "track": False, "trajectory": False,
                "proposal_rgb_or_mp4": False,
            },
            "interpretation": {
                "training_performed": False, "optimizer": None, "backward": False,
                "model_weights_written": False,
                "phase_reference_is_not_a_supervision_target": True,
                "randn_wrapper_is_injection_not_observation": True,
                "action_success_evaluated": False,
                "identity_preservation_evaluated": False,
                "quality_claim": False,
                "ablation_only": True,
                "scientific_claim_authorized": False,
                "one_step_stage_is_engineering_only": args.num_inference_steps == 1,
                "current_exact40_scientific_gate": "NO_GO_PENDING_SEMANTIC_SOURCE_ONLY_AND_SIGNAL_AUDITS",
            },
        }
        receipt = donor_runtime._rebase_artifact_paths(
            receipt, old_root=staging, new_root=output_dir
        )
        receipt["receipt_digest"] = donor_runtime.object_sha256(receipt)
        donor_runtime._write_receipt(staging / "receipt.json", receipt)
        donor_runtime._commit_output_transaction(staging=staging, final=output_dir)
        print(donor_runtime.canonical_json_bytes(receipt).decode("ascii"), flush=True)

    dist.barrier()
    del source_latent, reference_latents, generated, captures, proposals_cpu
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_STEPS", "ARM_ORDER", "ARM_SPECS", "CONDITION_MODES",
    "DEFAULT_SEED", "GAMMA", "LATENT_SHAPE", "METHOD", "PROPOSAL_BRANCHES",
    "PhaseNoiseCanaryError", "PhaseNoiseInjectionCapture", "SCHEMA_VERSION",
    "SOURCE_DC_RHO", "SPATIAL_RADIUS", "arm_plan", "bind_registered_proposals",
    "condition_contract", "main", "normalize_arms", "spec_for_arm",
]
