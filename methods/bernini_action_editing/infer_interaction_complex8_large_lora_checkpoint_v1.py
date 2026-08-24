#!/usr/bin/env python3
"""Decode one large-action adapter through the pinned native RV2V runner.

The wrapper changes only renderer construction: after the frozen base is
loaded, it installs the exact all-30-block rank-256 PEFT surface, strictly
loads one checkpoint, freezes/casts the adapter for inference, and then hands
the base renderer back to the existing exact40 native sampler.  Source-video
conditioning, four RGB-derived references, Gaussian draw, scheduler, guidance
and VAE decode remain owned by the pinned native implementation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native
import interaction_large_action_adapter_v1 as large_adapter


SCHEMA_VERSION = "bernini-interaction-complex8-large-lora-inference-v1"
_HOLDERS: list[Any] = []


class LargeActionInferenceError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def serialized_adapter_load() -> Any:
    """Serialize the large host checkpoint read across one SP4 group."""

    import fcntl

    lock_value = os.environ.get("NATIVE_V_AXIS_LOAD_LOCK")
    if not lock_value:
        raise LargeActionInferenceError("adapter load lock is absent")
    lock = Path(lock_value)
    if not lock.is_absolute() or not lock.is_file() or lock.is_symlink():
        raise LargeActionInferenceError("adapter load lock is not a plain absolute file")
    with lock.open("r+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def trim_host_allocator() -> None:
    import ctypes
    import gc

    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    malloc_trim = libc.malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--adapter", required=True)
    result.add_argument("--expected-adapter-sha256", required=True)
    result.add_argument("--adapter-label", required=True)
    result.add_argument("--bernini-root", required=True)
    result.add_argument("--veomni-root", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--checkpoint-content-manifest", required=True)
    result.add_argument("--source-video", required=True)
    result.add_argument("--expected-source-sha256", required=True)
    result.add_argument("--action-prompt", required=True)
    result.add_argument("--expected-action-prompt-sha256", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--num-inference-steps", type=int, choices=(40,), default=40)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--method-source-revision", required=True)
    result.add_argument("--method-source-archive-sha256", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    adapter = Path(args.adapter)
    if (
        not adapter.is_absolute()
        or not adapter.is_file()
        or adapter.is_symlink()
        or file_sha256(adapter) != args.expected_adapter_sha256
    ):
        raise LargeActionInferenceError("adapter path/hash differs")
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise LargeActionInferenceError("output-dir must be a fresh absolute directory")

    from safetensors import safe_open
    with safe_open(str(adapter), framework="pt", device="cpu") as opened:
        metadata = dict(opened.metadata() or {})
        keys = tuple(opened.keys())
    if (
        metadata.get("schema_version")
        != "bernini-interaction-complex8-large-lora-dpo-run-v1"
        or metadata.get("pure_t2v_anchor_is_absent") != "true"
        or not keys
    ):
        raise LargeActionInferenceError("adapter checkpoint metadata/key closure differs")

    load_receipt: dict[str, Any] = {}
    legacy = native.legacy
    try:
        bernini_root, veomni_root, _, _ = legacy.trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.trainer.VEOMNI_TESTED_COMMIT,
        )
        legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise LargeActionInferenceError(str(error)) from error
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)
    import bernini.models.renderer as renderer_module
    import source_self_native_ref_contrastive_v3 as native_schedule

    original_factory = renderer_module.BerniniRendererModel
    original_freeze_certificate = native.source_audit.model_freeze_certificate
    routed_forward_counts = [0] * 40
    freeze_state: dict[str, Any] = {}

    def hooked_factory(config: Any) -> Any:
        import torch
        from safetensors import safe_open

        base = original_factory(config)
        handle = large_adapter.install_pair_v5_action_adapter(base)
        named = dict(handle.trainable_named_parameters())
        adapter_contract = dict(handle.receipt())
        if set(named) != set(keys):
            raise LargeActionInferenceError("installed adapter parameter names differ")
        # The pinned be31323 native runner constructs the renderer on CPU and
        # moves it to the local GPU only after source VAE encoding.  Load the
        # adapter at that exact constructor seam without changing native main.
        with serialized_adapter_load():
            with safe_open(str(adapter), framework="pt", device="cpu") as opened:
                for name in sorted(named):
                    tensor = opened.get_tensor(name).contiguous()
                    parameter = named[name]
                    if (
                        tensor.dtype != torch.float32
                        or tuple(tensor.shape) != tuple(parameter.shape)
                        or not bool(torch.isfinite(tensor).all().item())
                    ):
                        raise LargeActionInferenceError(f"adapter tensor differs: {name}")
                    with torch.no_grad():
                        parameter.copy_(tensor.to(dtype=parameter.dtype))
                    del tensor
            trim_host_allocator()
        with torch.no_grad():
            for parameter in named.values():
                parameter.requires_grad_(False)
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if any(
            parameter.requires_grad or parameter.dtype != torch.bfloat16
            for parameter in named.values()
        ):
            raise LargeActionInferenceError("inference adapter freeze/cast differs")
        handle.model.eval()
        adapted_base = handle.model.get_base_model()
        inference_adapter_rows = [
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in adapted_base.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]
        if (
            not inference_adapter_rows
            or sum(row["numel"] for row in inference_adapter_rows)
            != large_adapter.EXPECTED_TRAINABLE_PARAMETERS
            or any(row["dtype"] != "torch.bfloat16" for row in inference_adapter_rows)
        ):
            raise LargeActionInferenceError(
                "inference adapter inventory/count/dtype closure differs"
            )
        inference_lora_modules = sorted(
            name
            for name, _ in adapted_base.named_modules()
            if "lora_" in name.lower() or ".lora" in name.lower()
        )
        if not inference_lora_modules:
            raise LargeActionInferenceError("inference LoRA module inventory is absent")
        freeze_state.update(
            {
                "model_identity": id(adapted_base),
                "adapter_rows": inference_adapter_rows,
                "adapter_rows_sha256": hashlib.sha256(
                    canonical_bytes(inference_adapter_rows)
                ).hexdigest(),
                "lora_modules": inference_lora_modules,
                "lora_modules_sha256": hashlib.sha256(
                    canonical_bytes(inference_lora_modules)
                ).hexdigest(),
                "adapter_contract_digest": adapter_contract["digest"],
            }
        )
        diffusion = adapted_base.diff_dec
        original_shared_step = diffusion.shared_step

        def routed_shared_step(*shared_args: Any, **shared_kwargs: Any) -> Any:
            import inspect

            try:
                bound = inspect.signature(original_shared_step).bind(
                    *shared_args, **shared_kwargs
                )
                timestep = bound.arguments["timesteps"]
                values = timestep.detach().reshape(-1).float()
                if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
                    raise ValueError("non-finite/empty timestep")
                observed = float(values[0].item())
                if not bool((values == observed).all().item()):
                    raise ValueError("non-uniform timestep")
                matches = [
                    index
                    for index, registered in enumerate(
                        native_schedule.NATIVE_UNIPC40_TIMESTEPS
                    )
                    if float(registered) == observed
                ]
                if len(matches) != 1:
                    raise ValueError("timestep is outside exact40")
                schedule_index = matches[0]
            except Exception as error:
                raise LargeActionInferenceError(
                    f"cannot authenticate native exact40 shared_step: {error}"
                ) from error
            route = large_adapter.PairV5ActionRoute(
                total_tokens=2,
                condition_tokens=1,
                sequence_parallel_rank=int(os.environ.get("LOCAL_RANK", "0")) % 4,
                sequence_parallel_size=4,
                branch_name="V",
                sigma_schedule_index=schedule_index,
                enabled=True,
            )
            try:
                with handle.route(route):
                    result = original_shared_step(*shared_args, **shared_kwargs)
            finally:
                # PEFT's disable_adapter() context re-enables the selected
                # adapter on exit and, in the pinned version, also restores
                # its default trainable flags.  Exact40 indices 38/39 use
                # that context for base-only inference, so restore the
                # inference-only freeze after every routed call.
                for parameter in named.values():
                    parameter.requires_grad_(False)
            routed_forward_counts[schedule_index] += 1
            return result

        diffusion.shared_step = routed_shared_step
        _HOLDERS.extend((handle, handle.model, adapted_base, original_shared_step))
        load_receipt.update(
            {
                "adapter_contract": adapter_contract,
                "checkpoint_tensor_count": len(named),
                "checkpoint_parameters_loaded_before_bfloat16_cast": True,
                "inference_dtype": "torch.bfloat16",
                "inference_requires_grad": False,
                "native_be31323_renderer_constructor_hook": True,
                "exact40_high_mid_adapter_indices": list(range(38)),
                "exact40_low_base_only_indices": [38, 39],
            }
        )
        return adapted_base

    def adapter_aware_freeze_certificate(model: Any) -> dict[str, Any]:
        """Prove a loaded inference adapter is frozen without calling it base-only."""

        if not freeze_state or id(model) != freeze_state["model_identity"]:
            raise LargeActionInferenceError(
                "adapter-aware freeze certificate saw an unauthenticated model"
            )
        trainable = [
            (name, int(parameter.numel()))
            for name, parameter in model.named_parameters()
            if bool(parameter.requires_grad)
        ]
        if trainable:
            raise LargeActionInferenceError(
                f"inference model has {len(trainable)} trainable parameter tensors"
            )
        observed_rows = [
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        ]
        observed_modules = sorted(
            name
            for name, _ in model.named_modules()
            if "lora_" in name.lower() or ".lora" in name.lower()
        )
        if (
            observed_rows != freeze_state["adapter_rows"]
            or observed_modules != freeze_state["lora_modules"]
        ):
            raise LargeActionInferenceError(
                "inference adapter inventory changed during native decode"
            )
        adapter_elements = sum(row["numel"] for row in observed_rows)
        if adapter_elements != large_adapter.EXPECTED_TRAINABLE_PARAMETERS:
            raise LargeActionInferenceError("inference adapter parameter count differs")
        return {
            "base_and_adapter_frozen": True,
            "trainable_parameter_tensors": 0,
            "trainable_parameter_elements": 0,
            "lora_module_count": len(observed_modules),
            "adapter_parameter_tensors": len(observed_rows),
            "adapter_parameter_elements": adapter_elements,
            "adapter_parameter_inventory_sha256": freeze_state[
                "adapter_rows_sha256"
            ],
            "lora_module_inventory_sha256": freeze_state[
                "lora_modules_sha256"
            ],
            "adapter_checkpoint_sha256": args.expected_adapter_sha256,
            "adapter_contract_digest": freeze_state["adapter_contract_digest"],
        }

    renderer_module.BerniniRendererModel = hooked_factory
    native.source_audit.model_freeze_certificate = adapter_aware_freeze_certificate
    native_args = [
        "--bernini-root", args.bernini_root,
        "--veomni-root", args.veomni_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-content-manifest", args.checkpoint_content_manifest,
        "--source-video", args.source_video,
        "--expected-source-sha256", args.expected_source_sha256,
        "--action-prompt", args.action_prompt,
        "--expected-action-prompt-sha256", args.expected_action_prompt_sha256,
        "--output-dir", args.output_dir,
        "--arms", "rv2v",
        "--num-inference-steps", str(args.num_inference_steps),
        "--seed", str(args.seed),
        "--method-source-revision", args.method_source_revision,
        "--method-source-archive-sha256", args.method_source_archive_sha256,
    ]
    try:
        status = native.main(native_args)
    finally:
        renderer_module.BerniniRendererModel = original_factory
        native.source_audit.model_freeze_certificate = original_freeze_certificate
    if status != 0:
        return status
    if any(count <= 0 for count in routed_forward_counts):
        raise LargeActionInferenceError(
            f"native exact40 adapter route coverage is incomplete: {routed_forward_counts}"
        )
    load_receipt["routed_shared_step_calls_by_schedule_index"] = routed_forward_counts
    rank = int(__import__("os").environ.get("RANK", "0"))
    if rank == 0:
        native_receipt = output / "receipt.json"
        rv2v = output / "rv2v.mp4"
        if not native_receipt.is_file() or not rv2v.is_file() or not load_receipt:
            raise LargeActionInferenceError("native adapted inference artifacts are incomplete")
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "adapter_label": args.adapter_label,
            "adapter": {
                "path": str(adapter),
                "sha256": args.expected_adapter_sha256,
                "metadata": metadata,
                **load_receipt,
            },
            "native_receipt": {
                "path": str(native_receipt),
                "sha256": file_sha256(native_receipt),
            },
            "output_video": {"path": str(rv2v), "sha256": file_sha256(rv2v)},
            "input_closure": {
                "pure_t2v_anchor_loaded": False,
                "pure_t2v_anchor_appearance_loaded": False,
                "source_video_and_source_rgb_refs_only_visual_conditions": True,
                "native_exact40_sampler_unmodified": True,
            },
        }
        receipt = {**unsigned, "receipt_digest": hashlib.sha256(canonical_bytes(unsigned)).hexdigest()}
        (output / "adapter-inference-receipt.json").write_bytes(canonical_bytes(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
