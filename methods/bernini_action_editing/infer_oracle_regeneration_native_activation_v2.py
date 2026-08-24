#!/usr/bin/env python3
"""WORLD4 native-only e02/e03 regeneration diagnostic runner.

The entrypoint is intentionally inert while the activation core/preflight
compiled trust anchors are unset.  A later exact-byte release may run only the
two preregistered arms for e02 (official V2V base and scheduled source-reference
R2V-4 inside the reviewed exact D/C/K union) and the official base for e03.
The self-generated anchor is review context only; no anchor tensor, target,
FlowEdit, connected route, training, optimizer, or automatic selection enters
this runner.  e03 is always recorded as ``ABSTAIN_KEEP_BASE``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
PREFLIGHT_PATH = (
    METHOD_ROOT / "preflight_oracle_regeneration_native_activation_v2.py"
).resolve(strict=True)
while str(METHOD_ROOT) in sys.path:
    sys.path.remove(str(METHOD_ROOT))
sys.path.insert(0, str(METHOD_ROOT))
_preloaded = sys.modules.get("preflight_oracle_regeneration_native_activation_v2")
if _preloaded is not None and Path(
    str(getattr(_preloaded, "__file__", ""))
).resolve(strict=True) != PREFLIGHT_PATH:
    raise RuntimeError("preloaded activation-v2 preflight origin differs")
_preflight_spec = importlib.util.find_spec(
    "preflight_oracle_regeneration_native_activation_v2"
)
if (
    _preflight_spec is None
    or not isinstance(_preflight_spec.origin, str)
    or Path(_preflight_spec.origin).resolve(strict=True) != PREFLIGHT_PATH
):
    raise RuntimeError("activation-v2 preflight import origin differs")
import preflight_oracle_regeneration_native_activation_v2 as release_preflight  # noqa: E402


SCHEMA_VERSION = "bernini-oracle-regeneration-native-activation-v2-run-v1"
METHOD = "round37-native-source-reference-r2v4-local-regeneration-diagnostic"
WORLD_SIZE = 4
NUM_INFERENCE_STEPS = 40
FRAME_COUNT = 81
FPS = 25
ARM_OFFICIAL = "official-v2v-base"
ARM_LOCAL = "local-source-reference-r2v4-in-manual-G"
CASE_ORDER = ("e02", "e03")
ARM_ORDER_BY_CASE = {
    "e02": (ARM_OFFICIAL, ARM_LOCAL),
    "e03": (ARM_OFFICIAL,),
}
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativeActivationV2RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class _CallSurfaceSnapshot:
    tokens: tuple[Any, ...]
    receipt: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _raw_rgb_sha256(frame: Any, *, expected_hw: tuple[int, int]) -> str:
    """Hash one decoded RGB frame with the authoring-receipt byte domain."""

    if (
        str(getattr(frame, "dtype", "")) != "uint8"
        or tuple(getattr(frame, "shape", ())) != (*expected_hw, 3)
        or not bool(getattr(getattr(frame, "flags", None), "c_contiguous", False))
    ):
        raise NativeActivationV2RunnerError("decoded source RGB frame differs")
    header = json.dumps(
        {
            "schema_version": "decoded-uint8-rgb-frame-v1",
            "shape": [*expected_hw, 3],
            "dtype": "uint8",
            "channel_order": "RGB",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + frame.tobytes(order="C")).hexdigest()


def cpu_preflight(*, authority_packet: Path, external_ledger: Path) -> Mapping[str, Any]:
    """Complete the exact CPU trust gate before Torch or model imports."""

    if "torch" in sys.modules:
        raise NativeActivationV2RunnerError("Torch was imported before CPU preflight")
    result = release_preflight.validate_release(
        packet_path=authority_packet,
        ledger_path=external_ledger,
    )
    if (
        result.get("ready") is not True
        or result.get("cpu_only") is not True
        or result.get("torch_imported") is not False
        or result.get("distributed_initialized") is not False
        or result.get("model_loaded") is not False
        or result.get("training") is not False
        or result.get("optimizer") is not False
        or result.get("flowedit") is not False
        or result.get("connected_route") is not False
        or result.get("automatic_replacement") is not False
        or result.get("selection_authority") is not None
    ):
        raise NativeActivationV2RunnerError("CPU activation preflight differs")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-packet", required=True)
    parser.add_argument("--external-ledger", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    return parser


def _fresh_output_dir(value: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise NativeActivationV2RunnerError("output directory must be absolute/non-root")
    parent = requested.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise NativeActivationV2RunnerError("output parent differs")
    output = parent / requested.name
    if (
        output != requested
        or output.exists()
        or output.is_symlink()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", output.name) is None
    ):
        raise NativeActivationV2RunnerError("refusing ambiguous/overwriting output")
    return output


def _callable_token(value: Any) -> Any:
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", None)
    return (owner, function) if function is not None else value


def _callable_row(owner: Any, name: str) -> tuple[Any, Mapping[str, Any]]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise NativeActivationV2RunnerError(f"runtime callable {name} is absent")
    try:
        instance_override = name in vars(owner)
    except TypeError as error:
        raise NativeActivationV2RunnerError(f"cannot inspect {name} owner") from error
    function = getattr(value, "__func__", value)
    return _callable_token(value), {
        "name": name,
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "instance_override": instance_override,
    }


def _capture_call_surface(diffusion: Any) -> _CallSurfaceSnapshot:
    transformer = getattr(diffusion, "transformer", None)
    scheduler = getattr(diffusion, "scheduler", None)
    owners = (
        (diffusion, "sample"),
        (diffusion, "shared_step"),
        (transformer, "patch_vae_latent"),
        (scheduler, "step"),
    )
    rows = tuple(_callable_row(owner, name) for owner, name in owners)
    return _CallSurfaceSnapshot(
        tokens=tuple(row[0] for row in rows),
        receipt={"all_instance_overrides_absent": not any(row[1]["instance_override"] for row in rows),
                 "callables": [dict(row[1]) for row in rows]},
    )


def _certify_call_surface(
    diffusion: Any, before: _CallSurfaceSnapshot, *, label: str
) -> Mapping[str, Any]:
    after = _capture_call_surface(diffusion)
    if before.tokens != after.tokens or before.receipt != after.receipt:
        raise NativeActivationV2RunnerError(f"{label} callable surface changed")
    if before.receipt.get("all_instance_overrides_absent") is not True:
        raise NativeActivationV2RunnerError(f"{label} entered with instance override")
    return dict(after.receipt)


def _all_rank_object(value: Any, *, dist: Any, label: str) -> list[Any]:
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, value)
    if any(row != rows[0] for row in rows[1:]):
        raise NativeActivationV2RunnerError(f"{label} differs across WORLD4")
    return rows


def _rank_zero_only_load_rows(
    *, dist: Any, rank: int, role: str, label: str
) -> list[Mapping[str, Any]]:
    local = {"rank": rank, role: rank == 0}
    rows: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(rows, local)
    expected = [
        {"rank": expected_rank, role: expected_rank == 0}
        for expected_rank in range(WORLD_SIZE)
    ]
    if rows != expected:
        raise NativeActivationV2RunnerError(f"{label} load roles differ")
    return [dict(row) for row in rows]


@contextmanager
def _all_rank_t5_constructor_bypass(
    *, t5_encoder_class: Any, checkpoint: Path, dtype: Any, placeholder_factory: Callable[[], Any]
) -> Any:
    """Prevent arm renderers from deserializing an unused text encoder."""

    own = vars(t5_encoder_class).get("from_pretrained")
    had_own = "from_pretrained" in vars(t5_encoder_class)
    audit = {"call_count": 0, "placeholder": None}

    def bypassed(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            cls is not t5_encoder_class
            or len(args) != 1
            or str(args[0]) != str(checkpoint)
            or kwargs != {"subfolder": "text_encoder", "torch_dtype": dtype}
        ):
            raise NativeActivationV2RunnerError("arm T5 constructor ABI differs")
        audit["call_count"] += 1
        if audit["call_count"] != 1:
            raise NativeActivationV2RunnerError("arm renderer requested T5 repeatedly")
        audit["placeholder"] = placeholder_factory()
        return audit["placeholder"]

    setattr(t5_encoder_class, "from_pretrained", classmethod(bypassed))
    try:
        yield audit
    finally:
        if had_own:
            setattr(t5_encoder_class, "from_pretrained", own)
        else:
            delattr(t5_encoder_class, "from_pretrained")
    if audit["call_count"] != 1 or audit["placeholder"] is None:
        raise NativeActivationV2RunnerError("arm T5 bypass was not exercised")


def _sampling_contract(native: Any, *, seed: int) -> Mapping[str, Any]:
    value = native.native_sampling_contract("rv2v", steps=40, seed=seed)
    value.update(
        {
            "guidance_mode": "v2v_apg",
            "omega_img": 4.5,
            "omega_txt": 4.0,
            "flow_shift": 5.0,
            "eta": 0.5,
            "norm_threshold": (50.0, 50.0),
            "momentum": 0.0,
        }
    )
    expected = {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 4.5,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": seed,
        "eta": 0.5,
        "norm_threshold": (50.0, 50.0),
        "momentum": 0.0,
    }
    if value != expected:
        raise NativeActivationV2RunnerError("native exact40 sampling contract differs")
    return value


def _materialize_source_references(
    case: Any,
    *,
    authority: Any,
    activation: Any,
    native: Any,
    source_audit: Any,
    materialize_vae: Any,
    autoencoder_class: Any,
    vae_encode: Callable[..., Any],
    checkpoint: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any]]:
    """Encode the full source and four RGB frames on rank zero only."""

    distributed = native.legacy.inference_distributed_contract()
    source_seal, source_bytes = activation._seal_plain_file_v2(
        case.source_video_path,
        label=f"{case.case_id} live source before VAE",
        retain_bytes=True,
    )
    if source_seal.sha256 != case.source_sha256 or source_bytes is None:
        raise NativeActivationV2RunnerError(f"{case.case_id} source bytes differ")
    latent_shape = tuple(case.full_source_latent_geometry)
    reference_shape = tuple(case.reference_latent_geometry)
    bucket_hw = (latent_shape[-2] * 8, latent_shape[-1] * 8)
    status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"activation-v2-{case.case_id}-source-"
            ) as temporary:
                snapshot = Path(temporary) / "source.mp4"
                snapshot.write_bytes(source_bytes)
                if _sha256_file(snapshot) != case.source_sha256:
                    raise NativeActivationV2RunnerError(
                        f"{case.case_id} private source snapshot differs"
                    )
                raw_frames, reported_fps, input_hw = (
                    materialize_vae._decode_exact_video(snapshot)
                )
                source_pixels, metadata, source_sha = (
                    source_audit.prepare_hashed_source_snapshot(snapshot)
                )
            if (
                source_sha != case.source_sha256
                or tuple(source_pixels.shape) != (1, 3, 81, *bucket_hw)
                or metadata.get("frame_count") != 81
                or float(metadata.get("fps", -1.0)) != 25.0
                or tuple(metadata.get("source_derived_bucket_hw", ())) != bucket_hw
                or len(raw_frames) != 81
                or float(reported_fps) != 25.0
                or tuple(input_hw) != activation.EXPECTED_SOURCE_INPUT_HW[case.case_id]
            ):
                raise NativeActivationV2RunnerError(
                    f"{case.case_id} live source geometry differs"
                )
            raw_rgb_sha = tuple(
                _raw_rgb_sha256(raw_frames[index], expected_hw=tuple(input_hw))
                for index in activation.REFERENCE_RGB_INDICES
            )
            full_preprocessed_sha = activation.safe_core.tensor_content_sha256_v1(
                source_pixels
            )
            preprocessed_sha = tuple(
                activation.safe_core.tensor_content_sha256_v1(
                    source_pixels[:, :, index : index + 1].contiguous()
                )
                for index in activation.REFERENCE_RGB_INDICES
            )
            vae = autoencoder_class.from_pretrained(
                str(checkpoint),
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            vae.eval().requires_grad_(False).to(device)
            pixels_device = source_pixels.to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                source_latent = vae_encode(vae, pixels_device).float().contiguous()
                references = tuple(
                    vae_encode(
                        vae,
                        pixels_device[:, :, index : index + 1].contiguous(),
                    )
                    .float()
                    .contiguous()
                    for index in activation.REFERENCE_RGB_INDICES
                )
            del pixels_device, source_pixels, vae
            torch.cuda.empty_cache()
            status[0] = {
                "ok": True,
                "raw_rgb_sha256": list(raw_rgb_sha),
                "preprocessed_sha256": list(preprocessed_sha),
                "full_preprocessed_sha256": full_preprocessed_sha,
                "input_hw": list(input_hw),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        source_latent = torch.zeros(latent_shape, device=device, dtype=torch.float32)
        references = tuple(
            torch.zeros(reference_shape, device=device, dtype=torch.float32)
            for _ in range(4)
        )
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"{case.case_id} rank-zero source/VAE failed: {status[0]}"
        )
    broadcasts = {
        "source": native._broadcast_condition_from_rank_zero(
            source_latent,
            label=f"{case.case_id}_source",
            world_size=WORLD_SIZE,
        ),
        "references": [
            native._broadcast_condition_from_rank_zero(
                value,
                label=f"{case.case_id}_reference_{index}",
                world_size=WORLD_SIZE,
            )
            for index, value in zip(activation.REFERENCE_RGB_INDICES, references)
        ],
    }
    source_latent = source_latent.detach().contiguous()
    references = tuple(value.detach().contiguous() for value in references)
    reference_receipt = activation.validate_reference_receipt_v2(
        authority,
        case_id=case.case_id,
        source_video_latent=source_latent,
        source_reference_latents=references,
    )
    if (
        tuple(status[0]["raw_rgb_sha256"])
        != tuple(reference_receipt.reference_raw_rgb_sha256)
        or tuple(status[0]["preprocessed_sha256"])
        != tuple(reference_receipt.reference_preprocessed_sha256)
        or status[0]["full_preprocessed_sha256"]
        != reference_receipt.source_preprocessed_sha256
    ):
        raise NativeActivationV2RunnerError(
            f"{case.case_id} live RGB/reference receipt differs"
        )
    source_after, _ = activation._seal_plain_file_v2(
        case.source_video_path,
        label=f"{case.case_id} live source after VAE",
        retain_bytes=False,
    )
    if source_after != source_seal:
        raise NativeActivationV2RunnerError(f"{case.case_id} source changed")
    identities = {
        "source": native._all_rank_tensor_identity(
            source_latent,
            label=f"{case.case_id}_source_latent",
            world_size=WORLD_SIZE,
        ),
        "references": [
            native._all_rank_tensor_identity(
                value,
                label=f"{case.case_id}_reference_{index}",
                world_size=WORLD_SIZE,
            )
            for index, value in zip(activation.REFERENCE_RGB_INDICES, references)
        ],
        "rank_zero_broadcasts": broadcasts,
        "four_independent_source_rgb_frame_vae_calls": True,
        "full_source_latent_slicing_used_for_references": False,
        "all_rank_vae_load_roles": _rank_zero_only_load_rows(
            dist=dist,
            rank=distributed.rank,
            role="vae_loaded",
            label=f"{case.case_id} VAE load role",
        ),
    }
    return source_latent, references, identities


def _materialize_prompts(
    case: Any,
    *,
    authority: Any,
    activation: Any,
    native: Any,
    prompt_builder: Any,
    prompt_clean: Callable[[str], str],
    tokenizer_class: Any,
    renderer_config_class: Any,
    renderer_model_class: Any,
    checkpoint: Path,
    bernini_root: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Encode low/high/negative prompts on rank zero, then broadcast."""

    distributed = native.legacy.inference_distributed_contract()
    low_text = prompt_builder.build_mode_native_prompt(
        "low-vr2v", case.action_caption, prompt_cleaner=prompt_clean
    )
    high_text = prompt_builder.build_mode_native_prompt(
        "high-r2v4", case.action_caption, prompt_cleaner=prompt_clean
    )
    rendered = {
        "low_action": low_text,
        "high_action": high_text,
        "negative": native.legacy.DEFAULT_NEGATIVE_PROMPT,
    }
    status: list[Any] = [None]
    if distributed.rank == 0:
        try:
            tokenizer = tokenizer_class.from_pretrained(
                str(checkpoint),
                subfolder="tokenizer",
                **native.legacy.tokenizer_load_kwargs(),
            )
            tokenized = {
                "low_action": native.legacy._tokenize_training_prompt(
                    tokenizer, low_text
                ),
                "high_action": native.legacy._tokenize_training_prompt(
                    tokenizer, high_text
                ),
                "negative": native.legacy._tokenize_renderer_negative(
                    tokenizer, native.legacy.DEFAULT_NEGATIVE_PROMPT
                ),
            }
            token_rows = {
                name: {
                    "token_ids_sha256": activation.safe_core.tensor_content_sha256_v1(
                        pair[0]
                    ),
                    "attention_mask_sha256": activation.safe_core.tensor_content_sha256_v1(
                        pair[1]
                    ),
                }
                for name, pair in tokenized.items()
            }
            config = renderer_config_class.from_pretrained(
                str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
                local_files_only=True,
                **native.legacy.inference_renderer_config_overrides(checkpoint),
            )
            config.dtype = torch.bfloat16
            native.legacy.trainer.validate_renderer_config_mapping(
                config.to_dict(), checkpoint
            )
            model = renderer_model_class(config)
            model.eval().requires_grad_(False)
            model.t5_text_encoder.to(device)
            prompt_bank: dict[str, Any] = {}
            with torch.inference_mode():
                for name, (ids, mask) in tokenized.items():
                    prompt_bank[name] = (
                        model.encode_prompt(ids.to(device), mask.to(device))
                        .detach()
                        .contiguous()
                    )
            model.t5_text_encoder.to("cpu")
            del tokenizer, tokenized, model
            gc.collect()
            torch.cuda.empty_cache()
            status[0] = {"ok": True, "token_rows": token_rows}
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        prompt_bank = {
            name: torch.zeros(
                (1, 512, 4096), device=device, dtype=torch.bfloat16
            )
            for name in ("low_action", "high_action", "negative")
        }
    dist.broadcast_object_list(status, src=0)
    if not isinstance(status[0], Mapping) or status[0].get("ok") is not True:
        raise NativeActivationV2RunnerError(
            f"{case.case_id} rank-zero prompt encoding failed: {status[0]}"
        )
    for name in ("low_action", "high_action", "negative"):
        dist.broadcast(prompt_bank[name], src=0)
        prompt_bank[name] = prompt_bank[name].detach().contiguous()
    receipt = activation.validate_prompt_receipt_v2(
        authority,
        case_id=case.case_id,
        low_action_prompt_embeds=prompt_bank["low_action"],
        high_action_prompt_embeds=prompt_bank["high_action"],
        negative_prompt_embeds=prompt_bank["negative"],
    )
    expected_rendered = tuple(
        hashlib.sha256(rendered[name].encode("utf-8")).hexdigest()
        for name in ("low_action", "high_action", "negative")
    )
    token_rows = status[0]["token_rows"]
    expected_tokens = tuple(
        token_rows[name]["token_ids_sha256"]
        for name in ("low_action", "high_action", "negative")
    )
    expected_masks = tuple(
        token_rows[name]["attention_mask_sha256"]
        for name in ("low_action", "high_action", "negative")
    )
    if (
        tuple(receipt.rendered_text_sha256) != expected_rendered
        or tuple(receipt.token_ids_sha256) != expected_tokens
        or tuple(receipt.attention_mask_sha256) != expected_masks
    ):
        raise NativeActivationV2RunnerError(
            f"{case.case_id} live prompt/token receipt differs"
        )
    identities = {
        name: native._all_rank_tensor_identity(
            prompt_bank[name],
            label=f"{case.case_id}_{name}_prompt",
            world_size=WORLD_SIZE,
        )
        for name in ("low_action", "high_action", "negative")
    }
    identities["rendered_text_sha256"] = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in rendered.items()
    }
    identities["rank0_only_text_encoder_load_and_encode"] = True
    identities["all_rank_prompt_byte_identity"] = True
    identities["all_rank_text_encoder_load_roles"] = _rank_zero_only_load_rows(
        dist=dist,
        rank=distributed.rank,
        role="real_text_encoder_loaded",
        label=f"{case.case_id} text encoder",
    )
    return prompt_bank, identities


def _load_fresh_arm_renderer(
    *,
    native: Any,
    renderer_config_class: Any,
    renderer_model_class: Any,
    t5_encoder_class: Any,
    bernini_root: Path,
    checkpoint: Path,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    config = renderer_config_class.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **native.legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    native.legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    if float(config.shift) != 5.0 or config.use_unipc is not True:
        raise NativeActivationV2RunnerError("fresh arm renderer config differs")
    bypass_audit: dict[str, Any] = {}

    def factory(value: Any) -> Any:
        nonlocal bypass_audit
        with _all_rank_t5_constructor_bypass(
            t5_encoder_class=t5_encoder_class,
            checkpoint=checkpoint,
            dtype=torch.bfloat16,
            placeholder_factory=torch.nn.Identity,
        ) as audit:
            model = renderer_model_class(value)
        bypass_audit = dict(audit)
        if model.t5_text_encoder is not audit["placeholder"]:
            raise NativeActivationV2RunnerError("arm renderer retained unexpected T5")
        model.t5_text_encoder = None
        return model

    model = native._load_frozen_renderer_gpu_resident_serialized(
        factory, config, device
    )
    dist.barrier()
    if (
        bypass_audit.get("call_count") != 1
        or bypass_audit.get("placeholder") is None
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise NativeActivationV2RunnerError("fresh arm model freeze/T5 closure differs")
    return model, config, {
        "fresh_model_and_scheduler_instance": True,
        "unused_t5_deserialization_bypassed_all_ranks": True,
        "bypass_call_count": 1,
    }


def _rng_receipt(torch: Any, *, device: Any) -> Mapping[str, Any]:
    cpu = torch.get_rng_state().detach().cpu().contiguous()
    cuda = torch.cuda.get_rng_state(device).detach().cpu().contiguous()
    return {
        "cpu_rng_sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(cuda.numpy().tobytes(order="C")).hexdigest(),
    }


def _run_one_arm(
    *,
    case: Any,
    arm: str,
    authority: Any,
    source_latent: Any,
    references: Sequence[Any],
    prompts: Mapping[str, Any],
    activation: Any,
    native: Any,
    sampler_contract: Any,
    strong_freeze: Callable[[Any], Mapping[str, Any]],
    renderer_config_class: Any,
    renderer_model_class: Any,
    t5_encoder_class: Any,
    wan_diffusion: Any,
    bernini_root: Path,
    checkpoint: Path,
    bernini_revision: str,
    torch: Any,
    dist: Any,
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    if arm not in ARM_ORDER_BY_CASE[case.case_id]:
        raise NativeActivationV2RunnerError("arm is outside case preregistration")
    model, config, load_receipt = _load_fresh_arm_renderer(
        native=native,
        renderer_config_class=renderer_config_class,
        renderer_model_class=renderer_model_class,
        t5_encoder_class=t5_encoder_class,
        bernini_root=bernini_root,
        checkpoint=checkpoint,
        torch=torch,
        dist=dist,
        device=device,
    )
    diffusion = sampler_contract.resolve_diffusion_core(model.diff_dec)
    wan_sha = sampler_contract.validate_runtime_source_identity(
        bernini_commit=bernini_revision,
        wan_diffusion_path=Path(wan_diffusion.__file__).resolve(strict=True),
    )
    sampler_contract._validate_scheduler_contract(
        diffusion.scheduler, expected_flow_shift=5.0
    )
    if diffusion.transformer_2 is not None:
        raise NativeActivationV2RunnerError("runner requires single DiT transformer_1")
    surface_before = _capture_call_surface(diffusion)
    if surface_before.receipt.get("all_instance_overrides_absent") is not True:
        raise NativeActivationV2RunnerError("arm entered with stacked override")
    freeze_before = strong_freeze(model)
    torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)
    rng_before = _rng_receipt(torch, device=device)
    sample_kwargs = {
        "prompt_embeds": prompts["low_action"],
        "uncond_prompt_embeds": prompts["negative"],
        "multi_video_vae_latents": [source_latent],
        "multi_image_vae_latents": list(references),
        "image_vae_latents": None,
        "width": int(case.full_source_latent_geometry[-1] * 8),
        "height": int(case.full_source_latent_geometry[-2] * 8),
        "device": device,
        **_sampling_contract(native, seed=case.seed),
    }
    patch = None
    capability = None
    if arm == ARM_LOCAL:
        capability = activation.mint_native_local_execution_capability_v2(
            authority,
            case_id=case.case_id,
            source_video_latent=source_latent,
            source_reference_latents=references,
            low_action_prompt_embeds=prompts["low_action"],
            high_action_prompt_embeds=prompts["high_action"],
            negative_prompt_embeds=prompts["negative"],
        )
        patch = activation.LocalOracleNativeBranchRuntimePatchV2(
            diffusion,
            config=activation.native_runtime.NativeBranchHomotopyRuntimeConfig(
                target_latent_shape=tuple(case.full_source_latent_geometry),
                expected_steps=40,
                expected_flow_shift=5.0,
                omega_image=4.5,
                omega_text=4.0,
                eta=0.5,
                image_norm_threshold=50.0,
                text_norm_threshold=50.0,
                momentum=0.0,
            ),
            capability=capability,
            expected_bernini_commit=bernini_revision,
            observed_wan_diffusion_sha256=wan_sha,
        )
        patch.install()
    try:
        with torch.inference_mode():
            result, noise_capture = native._sample_with_native_initial_noise_observer(
                sample_fn=lambda: diffusion.sample(**sample_kwargs),
                wan_diffusion_module=wan_diffusion,
                expected_shape=tuple(case.full_source_latent_geometry),
                expected_device=device,
                expected_seed=case.seed,
            )
    finally:
        if patch is not None:
            patch.restore()
    surface_after = _certify_call_surface(
        diffusion, surface_before, label=f"{case.case_id}/{arm}"
    )
    runtime_trace = (
        dict(patch.finalize())
        if patch is not None
        else {
            "arm": ARM_OFFICIAL,
            "runtime_local_patch_installed": False,
            "model_shared_step_scheduler_patch_vae_latent_override": False,
            "official_initial_gaussian_observer_only": True,
            "vendor_wan_diffusion_sha256": wan_sha,
        }
    )
    if (
        not isinstance(result, torch.Tensor)
        or tuple(result.shape) != tuple(case.full_source_latent_geometry)
        or result.dtype != torch.float32
        or result.requires_grad
        or result.grad_fn is not None
        or not bool(torch.isfinite(result).all().item())
    ):
        raise NativeActivationV2RunnerError(f"{case.case_id}/{arm} result differs")
    freeze_after = strong_freeze(model)
    if freeze_after != freeze_before or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise NativeActivationV2RunnerError(f"{case.case_id}/{arm} model changed")
    rng_after = _rng_receipt(torch, device=device)
    stored = result.detach().to(device="cpu", dtype=torch.float32).contiguous()
    result_identity = native._all_rank_tensor_identity(
        stored,
        label=f"{case.case_id}_{arm}_clean_latent",
        world_size=WORLD_SIZE,
    )
    noise_identity = native._all_rank_tensor_identity(
        noise_capture.tensor,
        label=f"{case.case_id}_{arm}_official_gaussian",
        world_size=WORLD_SIZE,
    )
    arm_receipt = {
        "case_id": case.case_id,
        "arm": arm,
        "sample_kwargs_without_tensors": {
            key: value
            for key, value in sample_kwargs.items()
            if key
            not in {
                "prompt_embeds",
                "uncond_prompt_embeds",
                "multi_video_vae_latents",
                "multi_image_vae_latents",
                "device",
            }
        },
        "sample_kwargs_digest": _canonical_sha256(
            {
                **{
                    key: value
                    for key, value in sample_kwargs.items()
                    if key
                    not in {
                        "prompt_embeds",
                        "uncond_prompt_embeds",
                        "multi_video_vae_latents",
                        "multi_image_vae_latents",
                        "device",
                    }
                },
                "source_sha256": activation.safe_core.tensor_content_sha256_v1(
                    source_latent
                ),
                "reference_sha256": [
                    activation.safe_core.tensor_content_sha256_v1(value)
                    for value in references
                ],
                "low_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["low_action"]
                ),
                "high_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["high_action"]
                ),
                "negative_prompt_sha256": activation.safe_core.tensor_content_sha256_v1(
                    prompts["negative"]
                ),
            }
        ),
        "official_gaussian_raw_sha256": noise_capture.raw_value_sha256,
        "official_gaussian_content_sha256": noise_capture.content_sha256,
        "official_gaussian_all_rank_identity": noise_identity,
        "clean_latent_all_rank_identity": result_identity,
        "call_surface_before_after_exact": surface_after,
        "fresh_model_load": load_receipt,
        "freeze_certificate": freeze_after,
        "rng_before": rng_before,
        "rng_after": rng_after,
        "runtime_trace": runtime_trace,
        "training": False,
        "optimizer": False,
        "backward": False,
        "self_generated_anchor_tensor_used": False,
        "target_video_or_latent_used": False,
    }
    model.to("cpu")
    del model, diffusion, result, patch, capability
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return stored, noise_capture, arm_receipt, runtime_trace


def _revalidate_live_case(
    *,
    activation: Any,
    authority: Any,
    case: Any,
    source_latent: Any,
    references: Sequence[Any],
    prompts: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebind every live condition used by either arm to frozen receipts."""

    activation.revalidate_compiled_activation_authority_v2(authority)
    reference = activation.validate_reference_receipt_v2(
        authority,
        case_id=case.case_id,
        source_video_latent=source_latent,
        source_reference_latents=tuple(references),
    )
    prompt = activation.validate_prompt_receipt_v2(
        authority,
        case_id=case.case_id,
        low_action_prompt_embeds=prompts["low_action"],
        high_action_prompt_embeds=prompts["high_action"],
        negative_prompt_embeds=prompts["negative"],
    )
    return {
        "case_id": case.case_id,
        "source_latent_sha256": reference.source_latent_sha256,
        "reference_latent_sha256": list(reference.reference_latent_sha256),
        "low_action_prompt_sha256": prompt.low_action_sha256,
        "high_action_prompt_sha256": prompt.high_action_sha256,
        "negative_prompt_sha256": prompt.negative_sha256,
        "authority_graph_revalidated": True,
    }


def _checkpoint_identity_rank_zero(
    *,
    source_audit: Any,
    checkpoint: Path,
    checkpoint_manifest: Path,
    dist: Any,
    rank: int,
    label: str,
) -> Mapping[str, Any]:
    status: list[Any] = [None]
    if rank == 0:
        try:
            status[0] = {
                "ok": True,
                "identity": source_audit.validate_checkpoint_content(
                    checkpoint, checkpoint_manifest
                ),
            }
        except Exception as error:
            status[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(status, src=0)
    row = status[0]
    if not isinstance(row, Mapping) or row.get("ok") is not True or not isinstance(
        row.get("identity"), Mapping
    ):
        raise NativeActivationV2RunnerError(
            f"{label} checkpoint validation failed: {row}"
        )
    identity = dict(row["identity"])
    _all_rank_object(
        _canonical_sha256(identity), dist=dist, label=f"{label} checkpoint identity"
    )
    return identity


def _owned_file_identity(path: Path, *, label: str) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise NativeActivationV2RunnerError(f"{label} open failed") from error
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
    except OSError as error:
        raise NativeActivationV2RunnerError(f"{label} named file disappeared") from error
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mode,
        row.st_nlink,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if (
        identity(before) != identity(after)
        or identity(after) != identity(named)
        or not stat.S_ISREG(after.st_mode)
        or size != after.st_size
    ):
        raise NativeActivationV2RunnerError(f"{label} changed during owned read")
    return {
        "sha256": digest.hexdigest(),
        "size": int(after.st_size),
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": int(after.st_nlink),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def _publish_staged_file_no_replace(
    *, staging_path: Path, destination: Path, label: str
) -> Mapping[str, Any]:
    """Publish one staged regular file with a hard-link O_EXCL boundary."""

    if (
        destination.parent.is_symlink()
        or destination.exists()
        or destination.is_symlink()
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", destination.name)
        is None
    ):
        raise NativeActivationV2RunnerError(f"{label} destination differs")
    before = _owned_file_identity(staging_path, label=f"{label} staged file")
    if before["nlink"] != 1:
        raise NativeActivationV2RunnerError(f"{label} staged link count differs")
    descriptor = os.open(
        str(staging_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    before = _owned_file_identity(staging_path, label=f"{label} frozen staged file")
    try:
        os.link(staging_path, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            f"{label} destination appeared; refusing overwrite"
        ) from error
    linked = _owned_file_identity(destination, label=f"{label} linked destination")
    if (
        linked["device"] != before["device"]
        or linked["inode"] != before["inode"]
        or linked["sha256"] != before["sha256"]
        or linked["mode"] != 0o444
        or linked["nlink"] != 2
    ):
        raise NativeActivationV2RunnerError(f"{label} published inode differs")
    staging_path.unlink()
    final = _owned_file_identity(destination, label=f"{label} final destination")
    if (
        final["sha256"] != before["sha256"]
        or final["size"] != before["size"]
        or final["mode"] != 0o444
        or final["nlink"] != 1
    ):
        raise NativeActivationV2RunnerError(f"{label} final identity differs")
    return {
        key: final[key] for key in ("sha256", "size", "mode", "nlink")
    }


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Write canonical JSON with no destination replacement window."""

    if path.exists() or path.is_symlink():
        raise NativeActivationV2RunnerError("receipt destination already exists")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-pid-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise NativeActivationV2RunnerError("stale receipt temporary exists")
    descriptor: Optional[int] = None
    published = False
    try:
        descriptor = os.open(
            str(temporary),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        final = _owned_file_identity(path, label="run receipt")
        if (
            final["sha256"] != hashlib.sha256(payload).hexdigest()
            or final["mode"] != 0o444
            or final["nlink"] != 1
        ):
            raise NativeActivationV2RunnerError("published receipt bytes differ")
        return {
            key: final[key] for key in ("sha256", "size", "mode", "nlink")
        }
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            "receipt destination appeared; refusing overwrite"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published and (temporary.exists() or temporary.is_symlink()):
            temporary.unlink()


def _stage_rank_zero_outputs(
    *,
    staging_dir: Path,
    output_dir: Path,
    generated: Mapping[str, Mapping[str, Any]],
    noises: Mapping[str, Mapping[str, Any]],
    arm_receipts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    authority: Any,
    activation: Any,
    native: Any,
    autoencoder_class: Any,
    vae_decode: Callable[..., Any],
    save_output_fn: Callable[..., Any],
    materialize_vae: Any,
    checkpoint: Path,
    torch: Any,
    device: Any,
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """Stage every latent/noise/video before the release directory exists."""

    if staging_dir.is_symlink() or stat.S_IMODE(staging_dir.stat().st_mode) != 0o700:
        raise NativeActivationV2RunnerError("private output staging directory differs")
    vae = autoencoder_class.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False).to(device)
    staged: dict[str, dict[str, Mapping[str, Any]]] = {}
    try:
        for case_id in CASE_ORDER:
            case = authority.cases[case_id]
            expected_hw = (
                int(case.full_source_latent_geometry[-2] * 8),
                int(case.full_source_latent_geometry[-1] * 8),
            )
            staged[case_id] = {}
            for arm in ARM_ORDER_BY_CASE[case_id]:
                prefix = f"{case_id}.{arm}"
                noise_path = staging_dir / f"{prefix}.official-gaussian.safetensors"
                noise_row = native._save_initial_noise_atomically(
                    noise_path,
                    noises[case_id][arm],
                    all_rank_identity=arm_receipts[case_id][arm][
                        "official_gaussian_all_rank_identity"
                    ],
                )
                latent_path = staging_dir / f"{prefix}.clean-latent.safetensors"
                latent_row = native._save_normalized_clean_latent_atomically(
                    latent_path,
                    generated[case_id][arm],
                )
                latent_device = generated[case_id][arm].to(
                    device=device, dtype=torch.float32
                ).contiguous()
                with torch.inference_mode():
                    decoded = vae_decode(vae, latent_device)
                del latent_device
                if tuple(int(item) for item in decoded.shape) != (
                    FRAME_COUNT,
                    expected_hw[0],
                    expected_hw[1],
                    3,
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} decoded video geometry differs"
                    )
                video_path = staging_dir / f"{prefix}.mp4"
                save_output_fn(decoded, str(video_path), fps=FPS)
                del decoded
                if video_path.is_symlink() or not video_path.is_file():
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} encoder output differs"
                    )
                raw_frames, encoded_fps, encoded_hw = (
                    materialize_vae._decode_exact_video(video_path)
                )
                if (
                    len(raw_frames) != FRAME_COUNT
                    or float(encoded_fps) != float(FPS)
                    or tuple(int(item) for item in encoded_hw) != expected_hw
                ):
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} encoded video metadata differs"
                    )
                video_identity = _owned_file_identity(
                    video_path, label=f"{case_id}/{arm} staged video"
                )
                staged[case_id][arm] = {
                    "official_gaussian": {
                        "staging_path": str(noise_path),
                        "destination_path": str(output_dir / noise_path.name),
                        "metadata": {
                            **dict(noise_row),
                            "path": str(output_dir / noise_path.name),
                        },
                    },
                    "clean_latent": {
                        "staging_path": str(latent_path),
                        "destination_path": str(output_dir / latent_path.name),
                        "metadata": {
                            **dict(latent_row),
                            "path": str(output_dir / latent_path.name),
                        },
                    },
                    "video": {
                        "staging_path": str(video_path),
                        "destination_path": str(output_dir / video_path.name),
                        "metadata": {
                            "path": str(output_dir / video_path.name),
                            "sha256": video_identity["sha256"],
                            "frame_count": FRAME_COUNT,
                            "fps": FPS,
                            "height": expected_hw[0],
                            "width": expected_hw[1],
                        },
                    },
                }
    finally:
        vae.to("cpu")
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return staged


def _publish_staged_outputs(
    *, output_dir: Path, staged: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    """Create the final directory once, then hard-link every artifact once."""

    try:
        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as error:
        raise NativeActivationV2RunnerError(
            "output directory appeared; refusing overwrite"
        ) from error
    published: dict[str, dict[str, Mapping[str, Any]]] = {}
    for case_id in CASE_ORDER:
        published[case_id] = {}
        for arm in ARM_ORDER_BY_CASE[case_id]:
            role_rows: dict[str, Any] = {}
            for role in ("official_gaussian", "clean_latent", "video"):
                row = staged[case_id][arm][role]
                staging_path = Path(str(row["staging_path"]))
                destination = Path(str(row["destination_path"]))
                if destination.parent != output_dir:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm}/{role} output parent differs"
                    )
                identity = _publish_staged_file_no_replace(
                    staging_path=staging_path,
                    destination=destination,
                    label=f"{case_id}/{arm}/{role}",
                )
                metadata = dict(row["metadata"])
                if metadata.get("sha256") != identity["sha256"]:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm}/{role} staged digest differs"
                    )
                metadata["file_identity"] = dict(identity)
                role_rows[role] = metadata
            published[case_id][arm] = role_rows
    return published


def _freeze_output_release(
    *, output_dir: Path, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    receipt_identity = _write_json_no_replace(output_dir / "receipt.json", receipt)
    expected_names = {"receipt.json"}
    for case_id in CASE_ORDER:
        for arm in ARM_ORDER_BY_CASE[case_id]:
            prefix = f"{case_id}.{arm}"
            expected_names.update(
                {
                    f"{prefix}.official-gaussian.safetensors",
                    f"{prefix}.clean-latent.safetensors",
                    f"{prefix}.mp4",
                }
            )
    observed_names = {path.name for path in output_dir.iterdir()}
    if observed_names != expected_names:
        raise NativeActivationV2RunnerError("output release file set differs")
    for name in expected_names:
        identity = _owned_file_identity(output_dir / name, label=f"output {name}")
        if identity["mode"] != 0o444 or identity["nlink"] != 1:
            raise NativeActivationV2RunnerError(f"output {name} is not frozen")
    output_dir.chmod(0o555)
    if output_dir.is_symlink() or stat.S_IMODE(output_dir.stat().st_mode) != 0o555:
        raise NativeActivationV2RunnerError("output release directory is not frozen")
    return {
        "receipt": dict(receipt_identity),
        "directory_mode": "0555",
        "all_artifacts_mode": "0444",
        "all_artifacts_nlink": 1,
        "no_overwrite_publish": True,
    }


def _certify_runtime_import_closure(
    *,
    activation: Any,
    authority: Any,
    native: Any,
    prompt_builder: Any,
    autoencoder_class: Any,
    auto_tokenizer_class: Any,
    text_encoder_class: Any,
    renderer_model_class: Any,
    vae_encode: Callable[..., Any],
    prompt_clean: Callable[[str], str],
) -> Mapping[str, str]:
    """Bind runtime objects to the exact implementation files in receipts."""

    observed_objects = {
        "vae_code": Path(vae_encode.__code__.co_filename).resolve(strict=True),
        "autoencoder_class_module": Path(
            inspect.getfile(autoencoder_class)
        ).resolve(strict=True),
        "tokenizer_code": Path(native.legacy.__file__).resolve(strict=True),
        "renderer_code": Path(inspect.getfile(renderer_model_class)).resolve(
            strict=True
        ),
        "prompt_builder_code": Path(prompt_builder.__file__).resolve(strict=True),
        "native_prompt_code": Path(native.__file__).resolve(strict=True),
        "prompt_cleaner_code": Path(prompt_clean.__code__.co_filename).resolve(
            strict=True
        ),
        "auto_tokenizer_module": Path(
            inspect.getfile(auto_tokenizer_class)
        ).resolve(strict=True),
        "text_encoder_class_module": Path(
            inspect.getfile(text_encoder_class)
        ).resolve(strict=True),
        "python_executable": Path(sys.executable).resolve(strict=True),
    }
    digests: dict[str, str] = {}
    for case_id in CASE_ORDER:
        case = authority.cases[case_id]
        reference = activation._bound_authority_json_v2(
            case,
            artifact_key="vae_reference_receipt",
            path=case.reference_receipt_path,
            expected_sha256=case.reference_receipt_sha256,
            label=f"{case_id} runtime VAE receipt",
        )
        prompt = activation._bound_authority_json_v2(
            case,
            artifact_key="prompt_receipt",
            path=case.prompt_receipt_path,
            expected_sha256=case.prompt_receipt_sha256,
            label=f"{case_id} runtime prompt receipt",
        )
        vae_contract = reference.get("vae_contract")
        prompt_contract = prompt.get("prompt_contract")
        if not isinstance(vae_contract, Mapping) or not isinstance(
            prompt_contract, Mapping
        ):
            raise NativeActivationV2RunnerError(
                f"{case_id} runtime implementation receipt differs"
            )
        expected_rows = {
            "vae_code": (
                vae_contract.get("vae_code_path"),
                vae_contract.get("vae_code_sha256"),
            ),
            "autoencoder_class_module": (
                vae_contract.get("autoencoder_class_module_path"),
                vae_contract.get("autoencoder_class_module_sha256"),
            ),
            **{
                key: (
                    prompt_contract.get(f"{key}_path"),
                    prompt_contract.get(f"{key}_sha256"),
                )
                for key in (
                    "tokenizer_code",
                    "renderer_code",
                    "prompt_builder_code",
                    "native_prompt_code",
                    "prompt_cleaner_code",
                    "auto_tokenizer_module",
                    "text_encoder_class_module",
                    "python_executable",
                )
            },
        }
        for key, path in observed_objects.items():
            expected_path, expected_sha = expected_rows[key]
            if (
                not isinstance(expected_path, str)
                or not isinstance(expected_sha, str)
                or path != Path(expected_path)
                or _sha256_file(path) != expected_sha
            ):
                raise NativeActivationV2RunnerError(
                    f"{case_id} runtime import {key} differs"
                )
            previous = digests.setdefault(key, expected_sha)
            if previous != expected_sha:
                raise NativeActivationV2RunnerError(
                    f"{case_id} runtime import receipt disagrees"
                )
    return digests


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    packet_requested = Path(args.authority_packet).expanduser()
    ledger_requested = Path(args.external_ledger).expanduser()
    if (
        not packet_requested.is_absolute()
        or packet_requested.is_symlink()
        or not ledger_requested.is_absolute()
        or ledger_requested.is_symlink()
    ):
        raise NativeActivationV2RunnerError("authority paths must be plain absolute files")
    packet_path = packet_requested.resolve(strict=True)
    ledger_path = ledger_requested.resolve(strict=True)
    if packet_path != packet_requested or ledger_path != ledger_requested:
        raise NativeActivationV2RunnerError("authority path identity differs")

    # This is the only route to the code below.  In the moving candidate the
    # compiled roots are None, so execution stops here before Torch/model/dist.
    preflight_receipt = cpu_preflight(
        authority_packet=packet_path,
        external_ledger=ledger_path,
    )
    output_dir = _fresh_output_dir(args.output_dir)

    import oracle_regeneration_activation_v2 as activation
    import infer_native_branch_homotopy_canary as prompt_builder
    import infer_native_identity_generation_canary as native
    import infer_source_kv_carrier_oracle as source_audit
    from infer_native_self_guided_action_field_canary import (
        _strong_model_freeze_certificate,
    )
    import tri_branch_unipc as sampler_contract

    authority = activation.load_compiled_activation_authority_v2(
        packet_path, ledger_path
    )
    if tuple(authority.cases) != CASE_ORDER or tuple(
        preflight_receipt.get("cases", ())
    ) != CASE_ORDER:
        raise NativeActivationV2RunnerError("activation cases/order differ")
    material_preflight = {
        case_id: activation.preflight_case_material_receipts_v2(
            authority, case_id=case_id
        )
        for case_id in CASE_ORDER
    }
    checkpoint_manifest = activation._plain_absolute_file(
        args.checkpoint_content_manifest,
        label="runtime checkpoint content manifest",
    )
    if any(
        Path(str(row["checkpoint_content_manifest_path"])) != checkpoint_manifest
        for row in material_preflight.values()
    ):
        raise NativeActivationV2RunnerError(
            "runtime checkpoint manifest differs from material receipts"
        )
    if (
        _SHA1.fullmatch(args.expected_bernini_commit) is None
        or _SHA1.fullmatch(args.expected_veomni_commit) is None
        or _SHA256.fullmatch(args.expected_checkpoint_tree_sha256) is None
        or args.expected_bernini_commit
        != native.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit
        != native.legacy.trainer.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != native.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise NativeActivationV2RunnerError("source/checkpoint CLI identity differs")
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            native.legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = native.legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
        inference_source_hashes = native.legacy.validate_inference_source_files(
            bernini_root
        )
        checkpoint_identity_initial = source_audit.validate_checkpoint_content(
            checkpoint, checkpoint_manifest
        )
    except Exception as error:
        raise NativeActivationV2RunnerError(str(error)) from error
    checkpoint_identity_sha256 = _canonical_sha256(checkpoint_identity_initial)
    if (
        int(transformer_config["num_attention_heads"]) % WORLD_SIZE
        or any(
            row["checkpoint_content_identity_sha256"]
            != checkpoint_identity_sha256
            for row in material_preflight.values()
        )
    ):
        raise NativeActivationV2RunnerError("checkpoint/material receipt identity differs")
    activation.verify_frozen_dependency_pins_v2()
    native.legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.io_utils import save_output
    import bernini.models.wan_diffusion as wan_diffusion
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from bernini.pipeline import _vae_decode, _vae_encode
    from bernini.training.data import SYSTEM_PROMPTS
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from tools import materialize_vae
    from transformers import AutoTokenizer, UMT5EncoderModel
    from transformers import __version__ as transformers_version

    if (
        SYSTEM_PROMPTS.get("r2v") != native.TASK_SYSTEM_PROMPTS["r2v"]
        or SYSTEM_PROMPTS.get("vr2v") != native.TASK_SYSTEM_PROMPTS["vr2v"]
        or DEFAULT_NEG_PROMPT != native.legacy.DEFAULT_NEGATIVE_PROMPT
    ):
        raise NativeActivationV2RunnerError("native prompt constants differ")
    import_closure = _certify_runtime_import_closure(
        activation=activation,
        authority=authority,
        native=native,
        prompt_builder=prompt_builder,
        autoencoder_class=AutoencoderKLWan,
        auto_tokenizer_class=AutoTokenizer,
        text_encoder_class=UMT5EncoderModel,
        renderer_model_class=BerniniRendererModel,
        vae_encode=_vae_encode,
        prompt_clean=prompt_clean,
    )
    distributed = native.legacy.inference_distributed_contract()
    if (
        distributed.world_size != WORLD_SIZE
        or distributed.ulysses_size != WORLD_SIZE
        or distributed.rank not in range(WORLD_SIZE)
        or distributed.local_rank not in range(WORLD_SIZE)
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
        or dist.is_initialized()
    ):
        raise NativeActivationV2RunnerError("runner requires fresh AUH WORLD4/SP4 ROCm")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    initialized = True
    try:
        init_parallel_state(ulysses_size=WORLD_SIZE)
        device = torch.device("cuda", distributed.local_rank)
        host_rows: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(
            host_rows,
            {"rank": distributed.rank, "hostname": socket.gethostname()},
        )
        if host_rows != [
            {"rank": rank, "hostname": host_rows[0]["hostname"]}
            for rank in range(WORLD_SIZE)
        ]:
            raise NativeActivationV2RunnerError("WORLD4 is not one-node rank ordered")
        checkpoint_identity = _checkpoint_identity_rank_zero(
            source_audit=source_audit,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            dist=dist,
            rank=distributed.rank,
            label="pre-condition",
        )
        if _canonical_sha256(checkpoint_identity) != checkpoint_identity_sha256:
            raise NativeActivationV2RunnerError("checkpoint changed before conditions")

        condition_tensors: dict[str, Mapping[str, Any]] = {}
        condition_receipts: dict[str, Mapping[str, Any]] = {}
        for case_id in CASE_ORDER:
            case = authority.cases[case_id]
            source_latent, references, reference_identity = (
                _materialize_source_references(
                    case,
                    authority=authority,
                    activation=activation,
                    native=native,
                    source_audit=source_audit,
                    materialize_vae=materialize_vae,
                    autoencoder_class=AutoencoderKLWan,
                    vae_encode=_vae_encode,
                    checkpoint=checkpoint,
                    torch=torch,
                    dist=dist,
                    device=device,
                )
            )
            prompts, prompt_identity = _materialize_prompts(
                case,
                authority=authority,
                activation=activation,
                native=native,
                prompt_builder=prompt_builder,
                prompt_clean=prompt_clean,
                tokenizer_class=AutoTokenizer,
                renderer_config_class=BerniniRendererConfig,
                renderer_model_class=BerniniRendererModel,
                checkpoint=checkpoint,
                bernini_root=bernini_root,
                torch=torch,
                dist=dist,
                device=device,
            )
            condition_tensors[case_id] = {
                "source": source_latent,
                "references": references,
                "prompts": prompts,
            }
            condition_receipts[case_id] = {
                "source_references": reference_identity,
                "prompts": prompt_identity,
                "live_binding": _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=source_latent,
                    references=references,
                    prompts=prompts,
                ),
            }

        generated: dict[str, dict[str, Any]] = {case_id: {} for case_id in CASE_ORDER}
        noises: dict[str, dict[str, Any]] = {case_id: {} for case_id in CASE_ORDER}
        arm_receipts: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in CASE_ORDER
        }
        runtime_traces: dict[str, dict[str, Mapping[str, Any]]] = {
            case_id: {} for case_id in CASE_ORDER
        }
        model_state_sha256: Optional[str] = None
        for case_id in CASE_ORDER:
            case = authority.cases[case_id]
            tensors = condition_tensors[case_id]
            for arm in ARM_ORDER_BY_CASE[case_id]:
                binding_before = _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                )
                checkpoint_before_arm = _checkpoint_identity_rank_zero(
                    source_audit=source_audit,
                    checkpoint=checkpoint,
                    checkpoint_manifest=checkpoint_manifest,
                    dist=dist,
                    rank=distributed.rank,
                    label=f"{case_id}/{arm} pre-arm",
                )
                if _canonical_sha256(checkpoint_before_arm) != checkpoint_identity_sha256:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} checkpoint changed before arm"
                    )
                stored, noise, arm_receipt, trace = _run_one_arm(
                    case=case,
                    arm=arm,
                    authority=authority,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                    activation=activation,
                    native=native,
                    sampler_contract=sampler_contract,
                    strong_freeze=_strong_model_freeze_certificate,
                    renderer_config_class=BerniniRendererConfig,
                    renderer_model_class=BerniniRendererModel,
                    t5_encoder_class=UMT5EncoderModel,
                    wan_diffusion=wan_diffusion,
                    bernini_root=bernini_root,
                    checkpoint=checkpoint,
                    bernini_revision=bernini_revision,
                    torch=torch,
                    dist=dist,
                    device=device,
                )
                binding_after = _revalidate_live_case(
                    activation=activation,
                    authority=authority,
                    case=case,
                    source_latent=tensors["source"],
                    references=tensors["references"],
                    prompts=tensors["prompts"],
                )
                if binding_after != binding_before:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} live binding changed"
                    )
                checkpoint_after_arm = _checkpoint_identity_rank_zero(
                    source_audit=source_audit,
                    checkpoint=checkpoint,
                    checkpoint_manifest=checkpoint_manifest,
                    dist=dist,
                    rank=distributed.rank,
                    label=f"{case_id}/{arm} post-arm",
                )
                if _canonical_sha256(checkpoint_after_arm) != checkpoint_identity_sha256:
                    raise NativeActivationV2RunnerError(
                        f"{case_id}/{arm} checkpoint changed during arm"
                    )
                state_sha = str(arm_receipt["freeze_certificate"]["state_content_sha256"])
                if model_state_sha256 is None:
                    model_state_sha256 = state_sha
                elif state_sha != model_state_sha256:
                    raise NativeActivationV2RunnerError(
                        "fresh arm model state differs across matched arms"
                    )
                trace_digest = _canonical_sha256(trace)
                _all_rank_object(
                    trace_digest,
                    dist=dist,
                    label=f"{case_id}/{arm} runtime trace",
                )
                full_arm_receipt = {
                    **dict(arm_receipt),
                    "live_binding_before": binding_before,
                    "live_binding_after": binding_after,
                    "checkpoint_identity_sha256_before_after": checkpoint_identity_sha256,
                    "runtime_trace_sha256": trace_digest,
                }
                _all_rank_object(
                    _canonical_sha256(full_arm_receipt),
                    dist=dist,
                    label=f"{case_id}/{arm} arm receipt",
                )
                generated[case_id][arm] = stored
                noises[case_id][arm] = noise
                arm_receipts[case_id][arm] = full_arm_receipt
                runtime_traces[case_id][arm] = trace

        e02_base = arm_receipts["e02"][ARM_OFFICIAL]
        e02_local = arm_receipts["e02"][ARM_LOCAL]
        if (
            e02_base["sample_kwargs_digest"] != e02_local["sample_kwargs_digest"]
            or e02_base["official_gaussian_raw_sha256"]
            != e02_local["official_gaussian_raw_sha256"]
            or e02_base["official_gaussian_content_sha256"]
            != e02_local["official_gaussian_content_sha256"]
            or noises["e02"][ARM_OFFICIAL].generator_initial_seed
            != noises["e02"][ARM_LOCAL].generator_initial_seed
            or runtime_traces["e02"][ARM_LOCAL].get(
                "outside_G_official_bytes_exact_all_steps"
            )
            is not True
        ):
            raise NativeActivationV2RunnerError(
                "e02 matched-arm Gaussian/input/local trace differs"
            )

        staging_context: Optional[tempfile.TemporaryDirectory[str]] = None
        staged_rows: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None
        stage_status: list[Any] = [None]
        if distributed.rank == 0:
            try:
                staging_context = tempfile.TemporaryDirectory(
                    prefix=f".{output_dir.name}.staging-",
                    dir=str(output_dir.parent),
                )
                staging_dir = Path(staging_context.name)
                staging_dir.chmod(0o700)
                staged_rows = _stage_rank_zero_outputs(
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                    generated=generated,
                    noises=noises,
                    arm_receipts=arm_receipts,
                    authority=authority,
                    activation=activation,
                    native=native,
                    autoencoder_class=AutoencoderKLWan,
                    vae_decode=_vae_decode,
                    save_output_fn=save_output,
                    materialize_vae=materialize_vae,
                    checkpoint=checkpoint,
                    torch=torch,
                    device=device,
                )
                stage_status[0] = {"ok": True}
            except Exception as error:
                stage_status[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(stage_status, src=0)
        if not isinstance(stage_status[0], Mapping) or stage_status[0].get("ok") is not True:
            if staging_context is not None:
                staging_context.cleanup()
            raise NativeActivationV2RunnerError(
                f"rank-zero output staging failed: {stage_status[0]}"
            )

        checkpoint_identity_final = _checkpoint_identity_rank_zero(
            source_audit=source_audit,
            checkpoint=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            dist=dist,
            rank=distributed.rank,
            label="post-decode",
        )
        if _canonical_sha256(checkpoint_identity_final) != checkpoint_identity_sha256:
            raise NativeActivationV2RunnerError("checkpoint changed during run/decode")
        activation.revalidate_compiled_activation_authority_v2(authority)
        for case_id in CASE_ORDER:
            tensors = condition_tensors[case_id]
            final_binding = _revalidate_live_case(
                activation=activation,
                authority=authority,
                case=authority.cases[case_id],
                source_latent=tensors["source"],
                references=tensors["references"],
                prompts=tensors["prompts"],
            )
            if final_binding != condition_receipts[case_id]["live_binding"]:
                raise NativeActivationV2RunnerError(
                    f"{case_id} final live binding changed"
                )
        activation.verify_frozen_dependency_pins_v2()
        final_import_closure = _certify_runtime_import_closure(
            activation=activation,
            authority=authority,
            native=native,
            prompt_builder=prompt_builder,
            autoencoder_class=AutoencoderKLWan,
            auto_tokenizer_class=AutoTokenizer,
            text_encoder_class=UMT5EncoderModel,
            renderer_model_class=BerniniRendererModel,
            vae_encode=_vae_encode,
            prompt_clean=prompt_clean,
        )
        if final_import_closure != import_closure:
            raise NativeActivationV2RunnerError("runtime import closure changed")

        publish_status: list[Any] = [None]
        if distributed.rank == 0:
            try:
                if staged_rows is None or staging_context is None:
                    raise NativeActivationV2RunnerError(
                        "rank-zero staged artifact state is absent"
                    )
                outputs = _publish_staged_outputs(
                    output_dir=output_dir, staged=staged_rows
                )
                receipt: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "method": METHOD,
                    "scope": "experimental diagnostic canary only",
                    "authority": {
                        "kind": "diagnostic_exact_packet_and_code_review_trust_root",
                        "packet_id": authority.packet_id,
                        "packet_sha256": authority.packet_sha256,
                        "external_ledger_sha256": authority.ledger_sha256,
                        "formal_authority": False,
                        "training_authority": False,
                    },
                    "cpu_preflight": dict(preflight_receipt),
                    "material_preflight": material_preflight,
                    "cases": {
                        case_id: {
                            "decision": authority.cases[case_id].decision,
                            "source_iid": authority.cases[case_id].source_iid,
                            "source_video_path": str(
                                authority.cases[case_id].source_video_path
                            ),
                            "source_video_sha256": authority.cases[
                                case_id
                            ].source_sha256,
                            "action_caption": authority.cases[case_id].action_caption,
                            "action_caption_sha256": authority.cases[
                                case_id
                            ].action_caption_sha256,
                            "structured_action_program_sha256": authority.cases[
                                case_id
                            ].structured_action_program_sha256,
                            "seed": authority.cases[case_id].seed,
                            "arms": list(ARM_ORDER_BY_CASE[case_id]),
                            "condition_receipts": condition_receipts[case_id],
                            "arm_receipts": arm_receipts[case_id],
                            "runtime_traces": runtime_traces[case_id],
                            "outputs": outputs[case_id],
                            "selection": (
                                "ABSTAIN_KEEP_BASE"
                                if case_id == "e03"
                                else "SIDE_BY_SIDE_DIAGNOSTIC_NO_SELECTION"
                            ),
                        }
                        for case_id in CASE_ORDER
                    },
                    "matched_e02_contract": {
                        "same_seed": True,
                        "same_official_gaussian_raw_bytes": True,
                        "same_sample_kwargs_and_live_conditions": True,
                        "fresh_model_and_scheduler_per_arm": True,
                        "only_experimental_difference": "scheduled source-reference R2V4 velocity inside exact G",
                        "outside_G_claim": "within local arm each scheduler model_output is byte-exact to that same-step official V2V output outside exact G",
                        "cross_arm_final_outside_pixel_identity_claimed": False,
                    },
                    "source_revisions": {
                        "bernini_root": str(bernini_root),
                        "veomni_root": str(veomni_root),
                        "bernini_revision": bernini_revision,
                        "veomni_revision": veomni_revision,
                        "inference_source_hashes": inference_source_hashes,
                        "runtime_import_closure": import_closure,
                    },
                    "checkpoint": {
                        "content_identity": checkpoint_identity_final,
                        "content_identity_sha256": checkpoint_identity_sha256,
                        "validated_before_conditions_before_after_each_arm_and_after_decode": True,
                        "fresh_model_state_sha256": model_state_sha256,
                    },
                    "world": {
                        "world_size": WORLD_SIZE,
                        "sequence_parallel_size": WORLD_SIZE,
                        "one_node": True,
                        "rank_rows": host_rows,
                    },
                    "runtime_versions": {
                        "python": sys.version,
                        "torch": str(torch.__version__),
                        "torch_hip": str(torch.version.hip),
                        "diffusers": str(diffusers_version),
                        "transformers": str(transformers_version),
                    },
                    "scientific_boundary": {
                        "source_reference_r2v4_regeneration_expert": True,
                        "self_generated_anchor_tensor_used": False,
                        "anchor_used_only_as_review_context": True,
                        "anchor_reference_or_quotient_arm_deferred": True,
                        "global_source_reference_r2v4_upper_bound_arm_deferred": True,
                        "local_G_step0_domain_separated_gaussian_arm_deferred": True,
                        "this_run_tests_self_generated_anchor_action_representation": False,
                    },
                    "output_contract": {
                        "side_by_side_only": True,
                        "automatic_selection": False,
                        "background_cosine_selection": False,
                        "e03_keep_base": True,
                        "post_run_output_bound_independent_review_required_for_any_selection": True,
                        "no_overwrite_publish": True,
                    },
                    "training": False,
                    "optimizer": False,
                    "backward": False,
                    "parameter_update": False,
                    "flowedit": False,
                    "connected_route": False,
                    "learned_gate": False,
                    "automatic_replacement": False,
                    "selection_authority": None,
                }
                receipt["receipt_digest"] = _canonical_sha256(receipt)
                freeze_receipt = _freeze_output_release(
                    output_dir=output_dir, receipt=receipt
                )
                publish_status[0] = {
                    "ok": True,
                    "receipt": receipt,
                    "release": freeze_receipt,
                }
                print(
                    json.dumps(
                        publish_status[0],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    flush=True,
                )
            except Exception as error:
                publish_status[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            finally:
                if staging_context is not None:
                    staging_context.cleanup()
        dist.broadcast_object_list(publish_status, src=0)
        if (
            not isinstance(publish_status[0], Mapping)
            or publish_status[0].get("ok") is not True
        ):
            raise NativeActivationV2RunnerError(
                f"rank-zero output publish failed: {publish_status[0]}"
            )
        dist.barrier()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_LOCAL",
    "ARM_OFFICIAL",
    "ARM_ORDER_BY_CASE",
    "CASE_ORDER",
    "METHOD",
    "NativeActivationV2RunnerError",
    "SCHEMA_VERSION",
    "build_parser",
    "cpu_preflight",
    "main",
]
