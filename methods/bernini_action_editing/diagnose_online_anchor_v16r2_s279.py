#!/usr/bin/env python3
"""Reproduce the v16r2 S279 action backward without an optimizer update.

This is a diagnostic-only wrapper.  It loads the frozen S256 adapter, moves
the original S279 IID to diagnostic step one, restores the original two
transform seeds, reports local LoRA-gradient finiteness after each action
micro, and deliberately stops at the first action-component reduction.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r2 as v16r2


v16 = v16r2.v16
base = v16r2.base
legacy = base.legacy
TARGET_INDEX = 278
TARGET_IID = "4aeb0557a94b4db3"
ORIGINAL_RECORD_OFFSET = 2 * TARGET_INDEX
EXPECTED_TENSOR_COUNT = 480
STOP_MESSAGE = "v16r2 S279 diagnostic completed before all-reduce/optimizer step"


_NAMED: tuple[tuple[str, Any], ...] = ()
_BACKWARD_COUNT = 0


def fail(message: str) -> None:
    raise RuntimeError(message)


def _finite_summary(named: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    import torch

    rows = []
    total_nan = 0
    total_posinf = 0
    total_neginf = 0
    nonfinite_names = []
    for name, parameter in named:
        gradient = parameter.grad
        if gradient is None:
            rows.append({"name": name, "missing": True})
            nonfinite_names.append(name)
            continue
        detached = gradient.detach()
        nan_count = int(torch.isnan(detached).count_nonzero().item())
        posinf_count = int(torch.isposinf(detached).count_nonzero().item())
        neginf_count = int(torch.isneginf(detached).count_nonzero().item())
        finite = detached[torch.isfinite(detached)]
        max_abs_finite = (
            float(finite.abs().max().item()) if int(finite.numel()) else None
        )
        total_nan += nan_count
        total_posinf += posinf_count
        total_neginf += neginf_count
        if nan_count or posinf_count or neginf_count:
            nonfinite_names.append(name)
            rows.append(
                {
                    "name": name,
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                    "nan": nan_count,
                    "posinf": posinf_count,
                    "neginf": neginf_count,
                    "max_abs_finite": max_abs_finite,
                }
            )
    return {
        "tensor_count": len(named),
        "nonfinite_tensor_count": len(nonfinite_names),
        "nonfinite_element_count": total_nan + total_posinf + total_neginf,
        "nan_count": total_nan,
        "posinf_count": total_posinf,
        "neginf_count": total_neginf,
        "nonfinite_names": nonfinite_names,
        "nonfinite_details": rows,
    }


def _emit(kind: str, payload: Mapping[str, Any]) -> None:
    import torch.distributed as dist

    rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
    document = {"kind": kind, "rank": rank, **dict(payload)}
    print(
        "V16R2_S279_DIAGNOSTIC "
        + json.dumps(document, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _rotated_manifest_loader(path: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    manifest, rows = _ORIGINAL_MANIFEST_LOADER(path)
    if len(rows) != 644 or str(rows[TARGET_INDEX].get("iid")) != TARGET_IID:
        fail("diagnostic full644 target closure differs")
    rotated = list(rows[TARGET_INDEX:]) + list(rows[:TARGET_INDEX])
    if len(rotated) != 644 or str(rotated[0].get("iid")) != TARGET_IID:
        fail("diagnostic manifest rotation differs")
    return manifest, rotated


def _mapped_step_seed(base_seed: int, record_index: int, row_index: int) -> int:
    if record_index not in (0, 1) or row_index != 0:
        fail("diagnostic executed beyond the isolated original S279 update")
    return _ORIGINAL_STEP_SEED(
        base_seed,
        record_index + ORIGINAL_RECORD_OFFSET,
        row_index + TARGET_INDEX,
    )


def _load_s256_then_synchronize(
    named: Sequence[tuple[str, Any]], *, source_rank: int
) -> str:
    from safetensors.torch import load_file
    import torch.distributed as dist

    global _NAMED
    if len(named) != EXPECTED_TENSOR_COUNT:
        fail("diagnostic trainable tensor closure is not 480")
    adapter_path = Path(os.environ["V16R2_S256_ADAPTER"]).resolve(strict=True)
    state = load_file(str(adapter_path), device="cpu")
    if len(state) != EXPECTED_TENSOR_COUNT:
        fail("diagnostic S256 adapter tensor closure is not 480")
    expected_keys = []
    for name, parameter in named:
        key = name.replace(".default.", ".")
        expected_keys.append(key)
        value = state.get(key)
        if value is None:
            fail(f"diagnostic S256 adapter lacks {key}")
        if tuple(value.shape) != tuple(parameter.shape):
            fail(f"diagnostic S256 adapter geometry differs: {key}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if set(expected_keys) != set(state):
        fail("diagnostic S256 adapter key closure differs")
    del state
    _NAMED = tuple(named)
    digest = _ORIGINAL_SYNCHRONIZE(named, source_rank=source_rank)
    _emit(
        "s256_adapter_loaded",
        {
            "adapter_path": str(adapter_path),
            "adapter_tensor_count": len(expected_keys),
            "synchronized_parameter_digest": digest,
        },
    )
    return digest


def _scaled_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
    global _BACKWARD_COUNT

    if not _NAMED:
        fail("diagnostic action backward preceded parameter capture")
    _BACKWARD_COUNT += 1
    if _BACKWARD_COUNT not in (1, 2):
        fail("diagnostic observed an unexpected explicit backward")
    scale = float(os.environ.get("V16R2_DIAGNOSTIC_BACKWARD_SCALE", "1"))
    if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
        fail("diagnostic backward scale must be finite in (0, 1]")
    scalar = float(self.detach().float().item())
    result = _ORIGINAL_TENSOR_BACKWARD(self * scale, *args, **kwargs)
    _emit(
        "action_micro_backward",
        {
            "micro": _BACKWARD_COUNT - 1,
            "scaled_action_scalar_before_backward": scalar,
            "backward_scale": scale,
            "gradient_summary_before_inverse_scale": _finite_summary(_NAMED),
        },
    )
    return result


def _stop_before_action_all_reduce(
    named: Sequence[tuple[str, Any]], *, bucket_bytes: int = 64 * 1024 * 1024
) -> float:
    import torch
    import torch.distributed as dist

    del bucket_bytes
    if (
        len(named) != len(_NAMED)
        or any(
            actual_name != expected_name or actual_parameter is not expected_parameter
            for (actual_name, actual_parameter), (
                expected_name,
                expected_parameter,
            ) in zip(named, _NAMED)
        )
        or _BACKWARD_COUNT != 2
    ):
        fail("diagnostic action-gradient boundary closure differs")
    scale = float(os.environ.get("V16R2_DIAGNOSTIC_BACKWARD_SCALE", "1"))
    before = _finite_summary(named)
    if before["nonfinite_tensor_count"] == 0 and scale != 1.0:
        inverse = 1.0 / scale
        with torch.no_grad():
            for _, parameter in named:
                if parameter.grad is None:
                    fail("diagnostic inverse scale found a missing gradient")
                parameter.grad.mul_(inverse)
    after = _finite_summary(named)
    _emit(
        "action_boundary_before_collective",
        {
            "backward_scale": scale,
            "gradient_summary_before_inverse_scale": before,
            "gradient_summary_after_inverse_scale": after,
            "all_reduce_performed": False,
            "optimizer_step_performed": False,
            "target_iid": TARGET_IID,
            "original_step": 279,
            "original_record_indices": [556, 557],
        },
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    raise RuntimeError(STOP_MESSAGE)


_ORIGINAL_MANIFEST_LOADER = v16.load_manifest_full644_v16
_ORIGINAL_STEP_SEED = legacy.step_seed
_ORIGINAL_SYNCHRONIZE = legacy.synchronize_trainable_parameters


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch

    global _BACKWARD_COUNT, _NAMED, _ORIGINAL_TENSOR_BACKWARD
    _BACKWARD_COUNT = 0
    _NAMED = ()
    _ORIGINAL_TENSOR_BACKWARD = torch.Tensor.backward
    original_manifest_loader = v16.load_manifest_full644_v16
    original_seed = legacy.step_seed
    original_sync = legacy.synchronize_trainable_parameters
    original_reduce = legacy.all_reduce_lora_gradients
    v16.load_manifest_full644_v16 = _rotated_manifest_loader
    legacy.step_seed = _mapped_step_seed
    legacy.synchronize_trainable_parameters = _load_s256_then_synchronize
    legacy.all_reduce_lora_gradients = _stop_before_action_all_reduce
    torch.Tensor.backward = _scaled_backward
    try:
        return v16r2.main(argv)
    finally:
        torch.Tensor.backward = _ORIGINAL_TENSOR_BACKWARD
        legacy.all_reduce_lora_gradients = original_reduce
        legacy.synchronize_trainable_parameters = original_sync
        legacy.step_seed = original_seed
        v16.load_manifest_full644_v16 = original_manifest_loader
        _NAMED = ()


if __name__ == "__main__":
    raise SystemExit(main())
