#!/usr/bin/env python3
"""Create fail-closed temporal controls for an extracted RAFT flow bundle.

The input contract is the exact three-tensor bundle emitted by
``extract_anchor_raft_flow_v1.py``.  The generated bundle is create-only and
uses one common temporal permutation for raw flow, camera-residual flow, and
validity.  No model or video is needed for this operation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-anchor-raft-flow-counterfactual-control-v1"
PERMUTATION_ALGORITHM = "sha256-signed-decimal-seed-index-sort-v1"
REQUIRED_TENSORS = (
    "backward_raw",
    "backward_camera_residual",
    "validity",
)


class FlowCounterfactualControlError(RuntimeError):
    """Raised when an input or publication violates the control contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


def _temporal_permutation(mode: str, length: int, seed: int | None) -> list[int]:
    if length <= 0:
        raise FlowCounterfactualControlError("temporal length must be positive")
    if mode not in {"zero", "shuffle", "reverse"}:
        raise FlowCounterfactualControlError(f"unsupported control mode: {mode!r}")
    if mode == "shuffle" and seed is None:
        raise FlowCounterfactualControlError("shuffle mode requires --seed")
    if mode != "shuffle" and seed is not None:
        raise FlowCounterfactualControlError(
            "--seed is only valid for shuffle mode"
        )
    if mode == "reverse":
        return list(range(length - 1, -1, -1))
    if mode == "zero":
        return list(range(length))

    assert seed is not None
    seed_text = str(seed).encode("ascii")
    order = sorted(
        range(length),
        key=lambda index: hashlib.sha256(
            seed_text + b":" + str(index).encode("ascii")
        ).digest(),
    )
    # An identity shuffle would be an invalid negative control whenever a
    # non-trivial temporal axis exists.  A one-step rotation is deterministic
    # and keeps the result a permutation without introducing RNG dependence.
    if length > 1 and order == list(range(length)):
        order = order[1:] + order[:1]
    return order


def _load_bundle(path: Path) -> tuple[dict[str, Any], dict[str, str], str]:
    try:
        import torch
        from safetensors import safe_open
    except ModuleNotFoundError as error:
        raise FlowCounterfactualControlError(
            "PyTorch and safetensors are required to materialize flow controls"
        ) from error

    hash_before = _sha256(path)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            metadata = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key).contiguous() for key in keys}
    except Exception as error:
        raise FlowCounterfactualControlError(
            f"cannot load input safetensors: {path}"
        ) from error
    if _sha256(path) != hash_before:
        raise FlowCounterfactualControlError("input changed while it was being read")

    if set(tensors) != set(REQUIRED_TENSORS) or len(tensors) != len(REQUIRED_TENSORS):
        raise FlowCounterfactualControlError(
            "input tensor keys must be exactly " + repr(REQUIRED_TENSORS)
        )
    raw = tensors["backward_raw"]
    camera = tensors["backward_camera_residual"]
    validity = tensors["validity"]
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise FlowCounterfactualControlError(
                f"{name} must be 4D [time, channels, height, width]"
            )
        if any(int(size) <= 0 for size in tensor.shape):
            raise FlowCounterfactualControlError(f"{name} has an empty dimension")
        if not tensor.dtype.is_floating_point:
            raise FlowCounterfactualControlError(f"{name} must be floating point")
        if not bool(torch.isfinite(tensor).all().item()):
            raise FlowCounterfactualControlError(f"{name} contains non-finite values")
    if int(raw.shape[1]) != 2 or int(camera.shape[1]) != 2:
        raise FlowCounterfactualControlError(
            "raw and camera-residual flow must each have two channels"
        )
    if int(validity.shape[1]) != 1:
        raise FlowCounterfactualControlError("validity must have one channel")
    if tuple(raw.shape) != tuple(camera.shape):
        raise FlowCounterfactualControlError(
            "raw and camera-residual flow shapes must match"
        )
    expected_validity_shape = (int(raw.shape[0]), 1, *map(int, raw.shape[2:]))
    if tuple(map(int, validity.shape)) != expected_validity_shape:
        raise FlowCounterfactualControlError(
            "validity time/spatial dimensions must match both flow tensors"
        )
    if raw.dtype != camera.dtype:
        raise FlowCounterfactualControlError(
            "raw and camera-residual flow dtypes must match"
        )
    return tensors, metadata, hash_before


def _transform_bundle(
    tensors: Mapping[str, Any], *, mode: str, permutation: Sequence[int]
) -> dict[str, Any]:
    import torch

    if mode == "zero":
        return {
            name: torch.zeros_like(tensors[name]).contiguous()
            for name in REQUIRED_TENSORS
        }
    index = torch.tensor(tuple(permutation), dtype=torch.long, device="cpu")
    return {
        name: tensors[name].index_select(0, index).contiguous()
        for name in REQUIRED_TENSORS
    }


def _output_metadata(
    input_metadata: Mapping[str, str],
    *,
    input_sha256: str,
    mode: str,
    seed: int | None,
) -> dict[str, str]:
    reserved = {
        "bernini_counterfactual_schema_version": SCHEMA_VERSION,
        "bernini_counterfactual_mode": mode,
        "bernini_counterfactual_input_sha256": input_sha256,
        "bernini_counterfactual_seed": "none" if seed is None else str(seed),
    }
    collisions = sorted(set(input_metadata).intersection(reserved))
    if collisions:
        raise FlowCounterfactualControlError(
            "input metadata collides with reserved control keys: " + repr(collisions)
        )
    output = dict(input_metadata)
    output.update(reserved)
    return output


def _load_optional_input_sidecar(input_path: Path) -> dict[str, Any] | None:
    sidecar = input_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    if sidecar.is_symlink() or not sidecar.is_file():
        raise FlowCounterfactualControlError(
            f"input sidecar must be a regular non-symlink file: {sidecar}"
        )
    try:
        payload = sidecar.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FlowCounterfactualControlError(
            f"input sidecar is not valid UTF-8 JSON: {sidecar}"
        ) from error
    if not isinstance(document, dict):
        raise FlowCounterfactualControlError("input sidecar JSON must be an object")
    return {
        "path": str(sidecar),
        "sha256": _sha256_bytes(payload),
        "document": document,
    }


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FlowCounterfactualControlError(
            f"refusing to overwrite existing output: {path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def materialize(
    input_path: Path | str,
    output_path: Path | str,
    *,
    mode: str,
    seed: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve(strict=True)
    output_path = Path(output_path).expanduser()
    if output_path.suffix != ".safetensors":
        raise FlowCounterfactualControlError("output must end in .safetensors")
    output_path = output_path.absolute()
    sidecar_path = output_path.with_suffix(".json")
    if output_path.exists() or output_path.is_symlink():
        raise FlowCounterfactualControlError(
            f"refusing to overwrite existing output: {output_path}"
        )
    if sidecar_path.exists() or sidecar_path.is_symlink():
        raise FlowCounterfactualControlError(
            f"refusing to overwrite existing sidecar: {sidecar_path}"
        )
    if input_path.is_symlink() or not input_path.is_file():
        raise FlowCounterfactualControlError(
            "input must resolve to a regular safetensors file"
        )
    if input_path.suffix != ".safetensors":
        raise FlowCounterfactualControlError("input must end in .safetensors")

    tensors, input_metadata, input_sha256 = _load_bundle(input_path)
    length = int(tensors["backward_raw"].shape[0])
    permutation = _temporal_permutation(mode, length, seed)
    transformed = _transform_bundle(
        tensors, mode=mode, permutation=permutation
    )
    metadata = _output_metadata(
        input_metadata,
        input_sha256=input_sha256,
        mode=mode,
        seed=seed,
    )

    try:
        import torch
        from safetensors.torch import load as load_safetensors
        from safetensors.torch import save as save_safetensors
    except ModuleNotFoundError as error:
        raise FlowCounterfactualControlError(
            "PyTorch and safetensors are required to materialize flow controls"
        ) from error
    payload = save_safetensors(transformed, metadata=metadata)
    try:
        roundtrip = load_safetensors(payload)
    except Exception as error:
        raise FlowCounterfactualControlError(
            "generated safetensors bytes failed an in-memory round trip"
        ) from error
    if set(roundtrip) != set(REQUIRED_TENSORS):
        raise FlowCounterfactualControlError("round-trip tensor key closure differs")
    for name in REQUIRED_TENSORS:
        if not torch.equal(roundtrip[name], transformed[name]):
            raise FlowCounterfactualControlError(
                f"round-trip tensor differs: {name}"
            )
        if mode == "zero" and int(torch.count_nonzero(roundtrip[name]).item()) != 0:
            raise FlowCounterfactualControlError(f"zero control is nonzero: {name}")

    tensor_shapes = {
        name: list(map(int, transformed[name].shape)) for name in REQUIRED_TENSORS
    }
    tensor_dtypes = {
        name: _dtype_name(transformed[name].dtype) for name in REQUIRED_TENSORS
    }
    output_sha256 = _sha256_bytes(payload)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "mode": mode,
        "seed": seed,
        "permutation": permutation,
        "permutation_semantics": (
            "output[time] = input[permutation[time]]; zero mode records identity "
            "correspondence but replaces all three tensors with zeros"
        ),
        "permutation_algorithm": (
            PERMUTATION_ALGORITHM if mode == "shuffle" else mode
        ),
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "required_tensor_key_closure": list(REQUIRED_TENSORS),
        "input_safetensors_metadata": input_metadata,
        "output_safetensors_metadata": metadata,
        "input_sidecar": _load_optional_input_sidecar(input_path),
        "validation": {
            "all_input_tensors_finite": True,
            "time_and_spatial_geometry_consistent": True,
            "roundtrip_exact": True,
            "zero_all_three_tensors_exact": mode == "zero",
            "create_only_publication": True,
        },
    }
    sidecar_payload = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if _sha256(input_path) != input_sha256:
        raise FlowCounterfactualControlError(
            "input changed after validation and before publication"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_create_only(output_path, payload)
    try:
        _write_create_only(sidecar_path, sidecar_payload)
    except Exception:
        # The bundle was created by this invocation and is not a valid
        # publication without its receipt.  Remove only if its bytes are still
        # ours; never touch a pre-existing or concurrently replaced file.
        if output_path.is_file() and not output_path.is_symlink():
            if _sha256(output_path) == output_sha256:
                output_path.unlink()
        raise
    if _sha256(output_path) != output_sha256:
        raise FlowCounterfactualControlError(
            "published output SHA differs from the pre-publication receipt"
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize zero or temporal controls for a RAFT flow bundle."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("zero", "shuffle", "reverse"))
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = materialize(
        args.input,
        args.output,
        mode=args.mode,
        seed=args.seed,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
