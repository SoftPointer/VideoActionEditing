#!/usr/bin/env python3
"""Inject a RAFT-transported matched Gaussian into native Bernini inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import flow_noise_action_canary_v1 as operator
import infer_lora as legacy


class FlowNoiseInferenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    return hashlib.sha256(
        value.detach().to(device="cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _option(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError) as error:
        raise FlowNoiseInferenceError(f"legacy arguments lack {name}") from error


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--flow-bundle", required=True)
    parser.add_argument(
        "--flow-field", choices=("raw", "camera_residual"), required=True
    )
    parser.add_argument("--flow-degradation", type=float, required=True)
    parser.add_argument("--flow-ignore-validity", action="store_true")
    args, legacy_argv = parser.parse_known_args()
    bundle_path = Path(args.flow_bundle).expanduser().resolve(strict=True)
    metadata_path = bundle_path.with_suffix(".json")
    if bundle_path.suffix != ".safetensors" or not metadata_path.is_file():
        raise FlowNoiseInferenceError("flow bundle or metadata sidecar is missing")
    output_path = Path(_option(legacy_argv, "--output"))
    if not output_path.is_absolute():
        raise FlowNoiseInferenceError("legacy output must be absolute")
    sidecar_path = output_path.with_name(output_path.name + ".flow-noise.json")
    if sidecar_path.exists():
        raise FlowNoiseInferenceError("refusing to overwrite flow-noise sidecar")

    from safetensors.torch import load_file

    tensors = load_file(str(bundle_path), device="cpu")
    flow_key = (
        "backward_raw"
        if args.flow_field == "raw"
        else "backward_camera_residual"
    )
    if set(tensors) != {"backward_raw", "backward_camera_residual", "validity"}:
        raise FlowNoiseInferenceError("flow bundle tensor registry differs")
    capture: dict[str, Any] = {"calls": 0}
    original_activate = legacy.trainer.activate_source_trees

    def patched_activate(*activate_args: Any, **activate_kwargs: Any) -> Any:
        result = original_activate(*activate_args, **activate_kwargs)
        import bernini.models.wan_diffusion as wan_diffusion

        original_randn = wan_diffusion.randn_tensor

        def injected_randn(*randn_args: Any, **randn_kwargs: Any) -> Any:
            baseline = original_randn(*randn_args, **randn_kwargs)
            capture["calls"] += 1
            if capture["calls"] != 1:
                raise FlowNoiseInferenceError(
                    "native sampler called randn_tensor more than once"
                )
            try:
                transformed = operator.build_flow_transported_noise(
                    baseline,
                    tensors[flow_key],
                    tensors["validity"],
                    degradation=args.flow_degradation,
                    use_validity=not args.flow_ignore_validity,
                )
            except operator.FlowNoiseActionError as error:
                raise FlowNoiseInferenceError(str(error)) from error
            capture.update(
                {
                    "baseline_sha256": _tensor_sha256(baseline),
                    "injected_sha256": _tensor_sha256(transformed.initial_noise),
                    "operator": transformed.receipt,
                }
            )
            return transformed.initial_noise

        wan_diffusion.randn_tensor = injected_randn
        capture["wan_module"] = wan_diffusion
        capture["original_randn"] = original_randn
        return result

    legacy.trainer.activate_source_trees = patched_activate
    status = legacy.main(legacy_argv)
    if capture.get("calls") != 1:
        raise FlowNoiseInferenceError(
            f"expected one native noise draw, observed {capture.get('calls')}"
        )
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        inference_receipt = output_path.with_name(output_path.name + ".receipt.json")
        if not inference_receipt.is_file():
            raise FlowNoiseInferenceError("native inference receipt is missing")
        metadata = json.loads(metadata_path.read_text())
        sidecar = {
            "schema_version": "bernini-flow-noise-inference-sidecar-v1",
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "native_inference_receipt": str(inference_receipt),
            "native_inference_receipt_sha256": _sha256(inference_receipt),
            "flow_bundle": str(bundle_path),
            "flow_bundle_sha256": _sha256(bundle_path),
            "flow_metadata_sha256": _sha256(metadata_path),
            "anchor_sha256": metadata["anchor_sha256"],
            "source_sha256": metadata["source_sha256"],
            "flow_field": args.flow_field,
            "flow_degradation": float(args.flow_degradation),
            "flow_ignore_validity": bool(args.flow_ignore_validity),
            "native_randn_call_count": capture["calls"],
            "native_baseline_noise_sha256": capture["baseline_sha256"],
            "injected_initial_noise_sha256": capture["injected_sha256"],
            "operator": capture["operator"],
            "base_model_weights_modified": False,
            "anchor_rgb_or_latent_used_by_model": False,
        }
        temporary = sidecar_path.with_name(sidecar_path.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, sidecar_path)
        print(json.dumps(sidecar, sort_keys=True), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
