#!/usr/bin/env python3
"""Evaluate one PAIR-v5 current-family SP4 group for source preservation.

Each of four ranks owns exactly one sealed action-only native RV2V candidate.
The rank verifies the rollout receipt and candidate/source media, verifies and
loads one frozen local plain DINOv2 checkpoint through Transformers 4.53.2's
official slow BitImageProcessor, deterministically decodes exact
81-frame RGB, and emits raw source-bound visual evidence.  Rank zero validates
the four candidate receipts and writes a group receipt.  Two independent
instances cover the full eight-GPU/two-group population.

This program consumes no target, proposal, donor, caption condition, mask,
flow, pose, track, or trajectory.  It performs no training and makes no
absolute source-preservation or action-editing-success claim.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_source_bound_preservation_evaluator_v1 as contract  # noqa: E402


_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
_PAIR_FIELDS = frozenset(
    {
        "schema_version", "root_spec_raw_sha256", "candidate_envelope_sha256",
        "group_id", "visible_gpus", "runtime_topology", "ordinal", "candidate",
        "sampling_contract", "semantic_input_closure", "native_receipt_path",
        "native_receipt_sha256", "native_receipt_digest", "artifacts", "receipt_digest",
    }
)
_NATIVE_FIELDS = frozenset(
    {
        "schema_version", "method", "method_source_revision",
        "method_source_archive_sha256", "bernini_commit", "veomni_commit",
        "bernini_inference_files", "checkpoint", "arms", "input", "preprocessing",
        "prompt_contract", "conditioning", "sampling", "latent_geometry",
        "condition_identities", "source_condition_artifact", "initial_noise_artifacts",
        "generated_identities", "outputs", "freeze_certificate", "runtime_versions",
        "interpretation", "experimental_canary", "production_claim_forbidden",
        "scientific_claim_authorized", "receipt_digest",
    }
)
_CLEAN_LATENT_FIELDS = frozenset(
    {
        "path", "sha256", "tensor_key", "shape", "stored_dtype",
        "sampler_return_dtype", "coordinate", "artifact_role", "origin",
        "native_sampler_before_vae_decode", "source_video_vae_encode_before_any_decode",
        "mp4_decode_reencode_used", "roundtrip_byte_exact_fp32",
    }
)
_GAUSSIAN_FIELDS = frozenset(
    {
        "path", "sha256", "tensor_key", "tensor_value_sha256", "raw_value_sha256",
        "content_sha256", "shape", "dtype", "stored_dtype", "original_device",
        "stored_device", "numel", "byte_count", "randn_tensor_call_count",
        "official_randn_tensor_call_count", "requested_device", "requested_dtype",
        "generator_device", "generator_initial_seed", "all_rank_identity", "coordinate",
        "origin", "observer_only", "captured_from_native_sampler",
        "observer_changed_return_value", "source_or_target_derived",
        "observer_added_device_to_cpu_readback", "official_module_global_symbol",
        "original_callable_invoked_once_with_unchanged_arguments",
        "original_return_tensor_forwarded_by_identity", "external_initial_noise_injection",
        "sampler_noise_replacement", "roundtrip_raw_value_exact",
    }
)
_PAIR_SAMPLING = {
    "condition_mode": "rv2v4", "num_frames": 81, "latent_frames": 21, "fps": 25,
    "num_inference_steps": 40, "source_reference_indices": [0, 27, 53, 80],
    "target_initialization": "official_gen_wanx22_fresh_gaussian",
}
_PAIR_SEMANTIC_CLOSURE = {
    "accepted": ["source_video", "complete_caption"], "target_video": False,
    "t2v_proposal_media": False, "donor_video": False, "external_reference": False,
    "mask": False, "flow": False, "pose": False, "track": False, "trajectory": False,
}


class PairV5SourceBoundScoringError(contract.PairV5SourceBoundEvaluationError):
    """The runtime cannot produce valid source-bound evidence."""


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise PairV5SourceBoundScoringError(f"{label} field closure differs")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return contract.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return contract.object_sha256(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PairV5SourceBoundScoringError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5SourceBoundScoringError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5SourceBoundScoringError(
            f"{label} must be an absolute non-symlink directory"
        )
    return path.resolve(strict=True)


def _strict_json_file(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    source = _plain_file(path, label=label)
    raw = source.read_bytes()

    def reject_constant(token: str) -> None:
        raise PairV5SourceBoundScoringError(f"{label} contains {token}")

    def reject_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairV5SourceBoundScoringError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5SourceBoundScoringError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise PairV5SourceBoundScoringError(f"{label} root differs")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_embedded_digest(
    value: Mapping[str, Any], *, field: str, label: str, ensure_ascii: bool
) -> str:
    unsigned = dict(value)
    declared = _sha256(unsigned.pop(field, None), label=f"{label} {field}")
    raw = json.dumps(
        unsigned,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii" if ensure_ascii else "utf-8")
    if hashlib.sha256(raw).hexdigest() != declared:
        raise PairV5SourceBoundScoringError(f"{label} embedded digest differs")
    return declared


def _hash_stable_file(
    path: str | Path, *, expected_sha256: str, label: str
) -> tuple[Path, str]:
    source = _plain_file(path, label=label)
    before = source.stat()
    actual = contract.file_sha256(source)
    after = source.stat()
    if actual != _sha256(expected_sha256, label=f"{label} expected SHA-256"):
        raise PairV5SourceBoundScoringError(f"{label} hash differs")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PairV5SourceBoundScoringError(f"{label} changed while hashing")
    return source, actual


def runtime_versions() -> dict[str, str]:
    import av
    import numpy
    import PIL
    import torch
    import transformers

    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_hip_version": str(torch.version.hip),
        "transformers_version": str(transformers.__version__),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "av_version": str(av.__version__),
        "numpy_version": str(numpy.__version__),
        "pillow_version": str(PIL.__version__),
    }


def _golden_rgb() -> Any:
    """One asymmetric deterministic RGB image; no filesystem decoder involved."""

    import numpy as np

    y, x, c = np.indices((19, 31, 3), dtype=np.int64)
    return ((17 * x + 29 * y + 71 * c + 11 * x * y) % 256).astype(np.uint8)[None]


def inspect_official_processor(checkpoint_root: str | Path) -> dict[str, Any]:
    """Load and execute Transformers 4.53.2's official slow Bit processor."""

    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

    root = _plain_directory(checkpoint_root, label="visual checkpoint root")
    processor = AutoImageProcessor.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=False
    )
    if type(processor).__name__ != "BitImageProcessor":
        raise PairV5SourceBoundScoringError("official processor is not BitImageProcessor")
    expected = {
        "do_resize": True,
        "size": {"shortest_edge": 256},
        "resample": 3,
        "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224},
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
    }
    observed = processor.to_dict()
    for key, value in expected.items():
        if observed.get(key) != value:
            raise PairV5SourceBoundScoringError(f"BitImageProcessor field differs: {key}")
    golden = _golden_rgb()
    images = [Image.fromarray(np.ascontiguousarray(golden[0]), mode="RGB")]
    pixels = processor(images=images, return_tensors="pt").pixel_values
    pixels = pixels.to(dtype=__import__("torch").float32).contiguous()
    if list(pixels.shape) != [1, 3, 224, 224]:
        raise PairV5SourceBoundScoringError("official processor golden geometry differs")
    return {
        "processor": processor,
        "preprocessor_golden_input_sha256": _rgb_array_sha256(golden),
        "preprocessor_golden_output_sha256": tensor_sha256(pixels),
        "preprocessor_golden_output_shape": list(pixels.shape),
    }


def verify_checkpoint_content(
    checkpoint_root: str | Path,
    manifest_path: str | Path,
    *,
    evaluator_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the complete non-cache checkpoint tree against canonical SHA256SUMS."""

    root = _plain_directory(checkpoint_root, label="visual checkpoint root")
    manifest = _plain_file(manifest_path, label="visual checkpoint manifest")
    model_spec = evaluator_spec["model"]
    manifest_sha = contract.file_sha256(manifest)
    if manifest_sha != model_spec["checkpoint_manifest_sha256"]:
        raise PairV5SourceBoundScoringError("visual checkpoint manifest hash differs")
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PairV5SourceBoundScoringError("cannot read visual checkpoint manifest") from error
    if len(lines) != model_spec["checkpoint_file_count"]:
        raise PairV5SourceBoundScoringError("visual checkpoint manifest count differs")
    expected: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise PairV5SourceBoundScoringError(
                "visual checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PairV5SourceBoundScoringError("checkpoint manifest path escapes root")
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise PairV5SourceBoundScoringError("checkpoint manifest path is empty/duplicate")
        expected[normalized] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise PairV5SourceBoundScoringError("checkpoint contains a symlink")
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise PairV5SourceBoundScoringError(
                "checkpoint contains a non-regular filesystem entry"
            )
    if actual != set(expected):
        raise PairV5SourceBoundScoringError("checkpoint file closure differs")
    verified: list[dict[str, str]] = []
    for relative in sorted(expected):
        path = _plain_file(root / relative, label=f"checkpoint file {relative}")
        digest = contract.file_sha256(path)
        if digest != expected[relative]:
            raise PairV5SourceBoundScoringError(
                f"checkpoint content hash differs: {relative}"
            )
        verified.append({"path": relative, "sha256": digest})
    config_path = _plain_file(root / "config.json", label="checkpoint config")
    config_sha = contract.file_sha256(config_path)
    if config_sha != model_spec["checkpoint_config_sha256"]:
        raise PairV5SourceBoundScoringError("checkpoint config hash differs")
    config, _ = _strict_json_file(config_path, label="checkpoint config")
    if config.get("model_type") != model_spec["architecture_id"]:
        raise PairV5SourceBoundScoringError("checkpoint architecture differs")
    if config.get("num_register_tokens", 0) != model_spec["num_register_tokens"]:
        raise PairV5SourceBoundScoringError(
            "checkpoint register-token geometry differs"
        )
    if (
        config.get("image_size") != model_spec["image_size"]
        or config.get("patch_size") != model_spec["patch_size"]
    ):
        raise PairV5SourceBoundScoringError("checkpoint image/patch geometry differs")
    preprocessor_path = _plain_file(
        root / "preprocessor_config.json", label="checkpoint preprocessor config"
    )
    preprocessor_sha = contract.file_sha256(preprocessor_path)
    if preprocessor_sha != model_spec["preprocessor_config_sha256"]:
        raise PairV5SourceBoundScoringError("checkpoint preprocessor config hash differs")
    processor_evidence = inspect_official_processor(root)
    if (
        processor_evidence["preprocessor_golden_input_sha256"]
        != model_spec["preprocessor_golden_input_sha256"]
        or processor_evidence["preprocessor_golden_output_sha256"]
        != model_spec["preprocessor_golden_output_sha256"]
        or processor_evidence["preprocessor_golden_output_shape"]
        != model_spec["preprocessor_golden_output_shape"]
    ):
        raise PairV5SourceBoundScoringError("official processor golden binding differs")
    return {
        "root": root,
        "adapter_id": model_spec["adapter_id"],
        "architecture_id": model_spec["architecture_id"],
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_config_sha256": config_sha,
        "preprocessor_config_sha256": preprocessor_sha,
        "checkpoint_file_count": len(verified),
        "verified_entries_digest": object_sha256(verified),
        "every_checkpoint_file_verified": True,
        **processor_evidence,
    }


def tensor_sha256(value: Any) -> str:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5SourceBoundScoringError("tensor hash requires one finite tensor")
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": int(cpu.numel() * cpu.element_size()),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    if len(raw) != metadata["byte_count"]:
        raise PairV5SourceBoundScoringError("tensor byte count differs")
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def _rgb_array_sha256(frames: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(frames)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise PairV5SourceBoundScoringError("RGB hash requires uint8 [T,H,W,3]")
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return hashlib.sha256(
        canonical_json_bytes(metadata) + b"\x00" + value.tobytes(order="C")
    ).hexdigest()


def decode_exact81_rgb(
    path: str | Path, *, expected_sha256: str
) -> tuple[Any, dict[str, Any]]:
    """Decode stream zero in presentation order and bind exact RGB bytes."""

    import av
    import numpy as np

    source, artifact_sha = _hash_stable_file(
        path, expected_sha256=expected_sha256, label="video artifact"
    )
    before = source.stat()
    try:
        with av.open(str(source), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise PairV5SourceBoundScoringError(
                    "video must expose exactly one video stream"
                )
            stream = streams[0]
            stream.thread_count = contract.PREPROCESS_CONTRACT["decode_thread_count"]
            if stream.average_rate is None:
                raise PairV5SourceBoundScoringError("video average rate is absent")
            fps = Fraction(stream.average_rate)
            if stream.time_base is None:
                raise PairV5SourceBoundScoringError("video time base is absent")
            time_base = Fraction(stream.time_base)
            arrays: list[Any] = []
            pts: list[int] = []
            width = height = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise PairV5SourceBoundScoringError("decoded frame PTS is absent")
                array = frame.to_ndarray(format="rgb24")
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise PairV5SourceBoundScoringError("decoded RGB frame differs")
                if width is None:
                    height, width = int(array.shape[0]), int(array.shape[1])
                if tuple(array.shape[:2]) != (height, width):
                    raise PairV5SourceBoundScoringError("decoded frame geometry changes")
                arrays.append(np.ascontiguousarray(array))
                pts.append(int(frame.pts))
    except PairV5SourceBoundScoringError:
        raise
    except Exception as error:
        raise PairV5SourceBoundScoringError("PyAV decode failed") from error
    if len(arrays) != contract.FRAME_COUNT:
        raise PairV5SourceBoundScoringError("video does not decode to exact81")
    if fps != Fraction(25, 1):
        raise PairV5SourceBoundScoringError("video FPS is not exact25")
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise PairV5SourceBoundScoringError("decoded PTS order is not strictly increasing")
    if len({right - left for left, right in zip(pts, pts[1:])}) != 1:
        raise PairV5SourceBoundScoringError("decoded PTS cadence is not uniform")
    pts_step = pts[1] - pts[0]
    if time_base * pts_step != Fraction(1, 25):
        raise PairV5SourceBoundScoringError("decoded PTS cadence is not exact25")
    frames = np.stack(arrays, axis=0)
    selected = np.ascontiguousarray(frames[list(contract.EVAL_FRAME_INDICES)])
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or contract.file_sha256(source) != artifact_sha:
        raise PairV5SourceBoundScoringError("video changed while decoding")
    evidence = {
        "artifact_sha256": artifact_sha,
        "decoded_rgb_sha256": _rgb_array_sha256(frames),
        "frame_count": int(frames.shape[0]),
        "fps_numerator": fps.numerator,
        "fps_denominator": fps.denominator,
        "time_base_numerator": time_base.numerator,
        "time_base_denominator": time_base.denominator,
        "pts_step": pts_step,
        "pts_sha256": object_sha256(pts),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "selected_frame_indices": list(contract.EVAL_FRAME_INDICES),
        "selected_rgb_sha256": _rgb_array_sha256(selected),
        "preprocessed_tensor_sha256": "0" * 64,
    }
    return frames, evidence


def preprocess_selected_rgb(frames: Any, processor: Any) -> tuple[Any, Any]:
    """Run the checkpoint's sealed official slow BitImageProcessor."""

    import numpy as np
    from PIL import Image
    import torch

    selected = np.ascontiguousarray(frames[list(contract.EVAL_FRAME_INDICES)])
    images = [Image.fromarray(frame, mode="RGB") for frame in selected]
    normalized = processor(images=images, return_tensors="pt").pixel_values
    normalized = normalized.to(dtype=torch.float32).contiguous()
    if list(normalized.shape) != [len(contract.EVAL_FRAME_INDICES), 3, 224, 224]:
        raise PairV5SourceBoundScoringError("official processor output geometry differs")
    mean = torch.tensor(
        contract.PREPROCESS_CONTRACT["normalization_mean"], dtype=torch.float32
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        contract.PREPROCESS_CONTRACT["normalization_std"], dtype=torch.float32
    ).reshape(1, 3, 1, 1)
    raw = (normalized * std + mean).clamp(0.0, 1.0).contiguous()
    if not bool(torch.isfinite(normalized).all().item()):
        raise PairV5SourceBoundScoringError("preprocessed tensor is non-finite")
    return raw, normalized


def load_frozen_model(
    checkpoint_evidence: Mapping[str, Any], *, device: Any
) -> tuple[Any, dict[str, int]]:
    import torch
    from transformers import AutoConfig, AutoModel

    root = checkpoint_evidence["root"]
    config = AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    if getattr(config, "model_type", None) != checkpoint_evidence["architecture_id"]:
        raise PairV5SourceBoundScoringError("loaded model architecture differs")
    loaded = AutoModel.from_pretrained(
        str(root),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
        attn_implementation="eager",
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], Mapping):
        raise PairV5SourceBoundScoringError("model loading-info contract differs")
    model, loading_info = loaded
    loading_counts = {
        "missing_key_count": len(loading_info.get("missing_keys", [])),
        "unexpected_key_count": len(loading_info.get("unexpected_keys", [])),
        "mismatched_key_count": len(loading_info.get("mismatched_keys", [])),
        "loading_error_count": len(loading_info.get("error_msgs", [])),
    }
    if any(loading_counts.values()):
        raise PairV5SourceBoundScoringError("visual checkpoint did not load exactly")
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    trainable = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0 or model.training:
        raise PairV5SourceBoundScoringError("visual evaluator is not frozen/eval")
    return model, loading_counts


def extract_features(
    model: Any,
    normalized_pixels: Any,
    *,
    device: Any,
    num_register_tokens: int,
    evaluation_image_size: int = contract.EVALUATION_IMAGE_SIZE,
    patch_size: int = contract.MODEL_PATCH_SIZE,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    pixels = normalized_pixels.to(device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        output = model(pixel_values=pixels)
    hidden = getattr(output, "last_hidden_state", None)
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.dtype != torch.float32
        or hidden.ndim != 3
        or hidden.shape[0] != len(contract.EVAL_FRAME_INDICES)
        or not bool(torch.isfinite(hidden).all().item())
    ):
        raise PairV5SourceBoundScoringError("model last_hidden_state differs")
    global_feature = functional.normalize(hidden[:, 0], p=2, dim=-1, eps=1.0e-12)
    dense = hidden[:, 1 + num_register_tokens :]
    patch_count = int(dense.shape[1])
    side = evaluation_image_size // patch_size
    if side <= 0 or patch_count != side * side:
        raise PairV5SourceBoundScoringError("dense token count differs from sealed patch grid")
    dense_feature = functional.normalize(dense, p=2, dim=-1, eps=1.0e-12)
    if not bool(torch.isfinite(global_feature).all().item()) or not bool(
        torch.isfinite(dense_feature).all().item()
    ):
        raise PairV5SourceBoundScoringError("normalized feature is non-finite")
    global_cpu = global_feature.detach().to(device="cpu", dtype=torch.float32).contiguous()
    dense_cpu = dense_feature.detach().to(device="cpu", dtype=torch.float32).contiguous()
    evidence = {
        "global_feature_sha256": tensor_sha256(global_cpu),
        "dense_feature_sha256": tensor_sha256(dense_cpu),
        "selected_frame_count": int(global_cpu.shape[0]),
        "dense_grid_height": side,
        "dense_grid_width": side,
        "feature_dimension": int(global_cpu.shape[-1]),
    }
    return global_cpu, dense_cpu, evidence


def _mapped_cosine(left: Any, right: Any) -> Any:
    import torch

    value = ((left * right).sum(dim=-1) + 1.0) * 0.5
    return torch.clamp(value, min=0.0, max=1.0)


def _strict_finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise PairV5SourceBoundScoringError(f"{label} is non-finite")
    return result


def global_similarity(left: Any, right: Any) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise PairV5SourceBoundScoringError("global feature shape differs")
    return _strict_finite_float(_mapped_cosine(left, right).mean().item(), label="global similarity")


def dense_similarity(left: Any, right: Any) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise PairV5SourceBoundScoringError("dense feature shape differs")
    return _strict_finite_float(
        _mapped_cosine(left, right).reshape(-1).median().item(),
        label="dense similarity",
    )


def spatial_layout_viewpoint_similarity(left: Any, right: Any) -> float:
    """Absolute source-bound same-location patch similarity, with no camera decomposition."""

    return dense_similarity(left, right)


def temporal_consistency(left: Any, right: Any) -> float:
    import torch

    if tuple(left.shape) != tuple(right.shape) or left.shape[0] < 2:
        raise PairV5SourceBoundScoringError("temporal feature geometry differs")
    left_change = 1.0 - (left[1:] * left[:-1]).sum(dim=-1)
    right_change = 1.0 - (right[1:] * right[:-1]).sum(dim=-1)
    # Median pooling over the complete fixed grid is robust to a minority of
    # legitimately edited actor tokens without constructing a target mask.
    discrepancy = (left_change - right_change).abs().reshape(-1).median()
    score = torch.exp(-discrepancy).clamp(0.0, 1.0)
    return _strict_finite_float(score.item(), label="temporal consistency")


def _phase_correlation_step(previous: Any, current: Any) -> tuple[float, float]:
    import torch
    import torch.nn.functional as functional

    if tuple(previous.shape) != tuple(current.shape) or previous.ndim != 3:
        raise PairV5SourceBoundScoringError("phase-correlation frame geometry differs")
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).reshape(3, 1, 1)
    left = (previous * weights).sum(dim=0, keepdim=True)[None]
    right = (current * weights).sum(dim=0, keepdim=True)[None]
    left = functional.interpolate(left, size=(96, 96), mode="bilinear", align_corners=False)[0, 0]
    right = functional.interpolate(right, size=(96, 96), mode="bilinear", align_corners=False)[0, 0]
    window = torch.outer(torch.hann_window(96, periodic=False), torch.hann_window(96, periodic=False))
    left = (left - left.mean()) * window
    right = (right - right.mean()) * window
    spectrum = torch.fft.fft2(left) * torch.fft.fft2(right).conj()
    magnitude = spectrum.abs()
    valid = magnitude > 1.0e-12
    if int(valid.count_nonzero().item()) < 96:
        if bool(torch.equal(previous, current)):
            return 0.0, 0.0
        raise PairV5SourceBoundScoringError("phase-correlation spectrum is degenerate")
    normalized = torch.where(valid, spectrum / magnitude.clamp_min(1.0e-12), torch.zeros_like(spectrum))
    correlation = torch.fft.ifft2(normalized).real
    if not bool(torch.isfinite(correlation).all().item()):
        raise PairV5SourceBoundScoringError("phase correlation is non-finite")
    index = int(correlation.reshape(-1).argmax().item())
    y, x = divmod(index, 96)
    if y > 48:
        y -= 96
    if x > 48:
        x -= 96
    return float(y / 96.0), float(x / 96.0)


def temporal_translation_agreement(candidate_raw: Any, source_raw: Any) -> float:
    import torch

    if tuple(candidate_raw.shape) != tuple(source_raw.shape):
        raise PairV5SourceBoundScoringError("temporal translation diagnostic geometry differs")
    candidate_steps = torch.tensor(
        [
            _phase_correlation_step(candidate_raw[index - 1], candidate_raw[index])
            for index in range(1, int(candidate_raw.shape[0]))
        ],
        dtype=torch.float32,
    )
    source_steps = torch.tensor(
        [
            _phase_correlation_step(source_raw[index - 1], source_raw[index])
            for index in range(1, int(source_raw.shape[0]))
        ],
        dtype=torch.float32,
    )
    score = torch.exp(-(candidate_steps - source_steps).abs().mean()).clamp(0.0, 1.0)
    return _strict_finite_float(score.item(), label="temporal translation agreement diagnostic")


def quality_diagnostics(candidate_raw: Any, source_raw: Any) -> dict[str, float]:
    import torch

    if tuple(candidate_raw.shape) != tuple(source_raw.shape) or candidate_raw.shape[0] < 3:
        raise PairV5SourceBoundScoringError("quality diagnostic geometry differs")

    def sharpness(frames: Any) -> Any:
        dx = frames[:, :, :, 1:] - frames[:, :, :, :-1]
        dy = frames[:, :, 1:, :] - frames[:, :, :-1, :]
        return 0.5 * (dx.square().mean() + dy.square().mean())

    candidate_sharp = sharpness(candidate_raw)
    source_sharp = sharpness(source_raw)
    if float(source_sharp.item()) <= 1.0e-12:
        sharpness_retention = torch.ones_like(source_sharp) if float(candidate_sharp.item()) <= 1.0e-12 else torch.zeros_like(source_sharp)
    else:
        sharpness_retention = torch.clamp(candidate_sharp / source_sharp, 0.0, 1.0)
    clipped = ((candidate_raw <= (2.0 / 255.0)) | (candidate_raw >= (253.0 / 255.0))).to(torch.float32)
    exposure = torch.clamp(1.0 - clipped.mean(), 0.0, 1.0)
    candidate_step = (candidate_raw[1:] - candidate_raw[:-1]).abs().mean()
    source_step = (source_raw[1:] - source_raw[:-1]).abs().mean()
    if float(source_step.item()) <= 1.0e-12:
        nonfreeze = torch.ones_like(source_step)
    else:
        nonfreeze = torch.clamp(candidate_step / source_step, 0.0, 1.0)
    candidate_mean = candidate_raw.mean(dim=(1, 2, 3))
    source_mean = source_raw.mean(dim=(1, 2, 3))
    candidate_second = candidate_mean[2:] - 2.0 * candidate_mean[1:-1] + candidate_mean[:-2]
    source_second = source_mean[2:] - 2.0 * source_mean[1:-1] + source_mean[:-2]
    flicker = torch.exp(-10.0 * (candidate_second - source_second).abs().mean()).clamp(0.0, 1.0)
    terms = torch.stack((sharpness_retention, exposure, nonfreeze, flicker))
    score = torch.exp(torch.log(terms.clamp_min(1.0e-12)).mean()).clamp(0.0, 1.0)
    result = {
        "decode_video_quality_diagnostic": score.item(),
        "quality_sharpness_retention": sharpness_retention.item(),
        "quality_exposure_score": exposure.item(),
        "quality_nonfreeze_score": nonfreeze.item(),
        "quality_flicker_score": flicker.item(),
    }
    return {key: _strict_finite_float(value, label=key) for key, value in result.items()}


def compute_metrics(
    *,
    candidate_global: Any,
    candidate_dense: Any,
    correct_global: Any,
    correct_dense: Any,
    wrong_global: Any,
    wrong_dense: Any,
    candidate_raw: Any,
    correct_raw: Any,
) -> dict[str, float]:
    correct_identity = global_similarity(candidate_global, correct_global)
    wrong_identity = global_similarity(candidate_global, wrong_global)
    source_self_upper = global_similarity(correct_global, correct_global)
    identity_denominator = source_self_upper - wrong_identity
    identity_contrast = (
        (correct_identity - wrong_identity) / identity_denominator
        if identity_denominator > 0.0
        else 0.0
    )
    correct_background = dense_similarity(candidate_dense, correct_dense)
    wrong_background = dense_similarity(candidate_dense, wrong_dense)
    layout_correct = spatial_layout_viewpoint_similarity(candidate_dense, correct_dense)
    layout_wrong = spatial_layout_viewpoint_similarity(candidate_dense, wrong_dense)
    layout_denominator = 1.0 - layout_wrong
    layout_contrast = (
        (layout_correct - layout_wrong) / layout_denominator
        if layout_denominator > 0.0
        else 0.0
    )
    metrics = {
        "source_identity_appearance_proxy": correct_identity,
        "source_identity_appearance_wrong_source_proxy": wrong_identity,
        "source_identity_appearance_correct_minus_wrong_margin": correct_identity - wrong_identity,
        "source_identity_appearance_source_self_upper_bound": source_self_upper,
        "source_identity_appearance_upper_bound_minus_correct_headroom": source_self_upper - correct_identity,
        "source_identity_appearance_wrong_normalized_contrast": identity_contrast,
        "background_appearance_fixed_grid_proxy": correct_background,
        "background_appearance_wrong_source_fixed_grid_proxy": wrong_background,
        "background_appearance_correct_minus_wrong_margin": correct_background - wrong_background,
        "non_target_temporal_consistency_proxy": temporal_consistency(candidate_dense, correct_dense),
        "non_target_temporal_consistency_wrong_source_proxy": temporal_consistency(candidate_dense, wrong_dense),
        "source_bound_spatial_layout_viewpoint_proxy": layout_correct,
        "source_bound_spatial_layout_wrong_source_proxy": layout_wrong,
        "source_bound_spatial_layout_correct_minus_wrong_margin": layout_correct - layout_wrong,
        "source_bound_spatial_layout_wrong_normalized_contrast_proxy": layout_contrast,
        "temporal_global_translation_agreement_diagnostic": temporal_translation_agreement(candidate_raw, correct_raw),
        **quality_diagnostics(candidate_raw, correct_raw),
    }
    return {key: _strict_finite_float(value, label=key) for key, value in metrics.items()}


def _artifact_in_candidate_dir(value: Any, *, candidate_dir: Path, label: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise PairV5SourceBoundScoringError(f"{label} artifact differs")
    digest = _sha256(value.get("sha256"), label=f"{label} SHA-256")
    path, _ = _hash_stable_file(value.get("path"), expected_sha256=digest, label=label)
    if path.parent != candidate_dir:
        raise PairV5SourceBoundScoringError(f"{label} escaped candidate directory")
    return dict(value), path


def _verify_native_receipt(
    native: Mapping[str, Any], *, candidate: Mapping[str, Any], candidate_dir: Path,
    evaluator_spec: Mapping[str, Any], native_file_sha256: str,
) -> dict[str, Any]:
    _closed(native, _NATIVE_FIELDS, label="native receipt")
    native_digest = _verify_embedded_digest(
        native, field="receipt_digest", label="native receipt", ensure_ascii=False
    )
    if (
        native["schema_version"] != contract.EXPECTED_NATIVE_SCHEMA
        or native["method"] != contract.EXPECTED_NATIVE_METHOD
        or native["arms"] != ["rv2v"]
    ):
        raise PairV5SourceBoundScoringError("native schema/method/arm closure differs")

    sealed_generation = contract.generation_provenance_from_native_receipt(
        native,
        reference_file_sha256=evaluator_spec["generation_provenance"][
            "reference_native_receipt_file_sha256"
        ],
    )
    if sealed_generation != evaluator_spec["generation_provenance"]:
        raise PairV5SourceBoundScoringError("native method/checkpoint/runtime provenance differs")

    native_input = _closed(
        native["input"],
        {
            "source_video_path", "source_video_sha256", "action_prompt_utf8_sha256",
            "action_prompt_utf8_bytes", "accepted_external_conditions", "target_video",
            "external_reference_image_or_video", "external_mask_flow_pose_track_trajectory",
            "external_first_frame_anchor",
        },
        label="native input",
    )
    caption_bytes = candidate["complete_caption"].encode("utf-8")
    if native_input != {
        "source_video_path": candidate["source_video"],
        "source_video_sha256": candidate["source_video_sha256"],
        "action_prompt_utf8_sha256": candidate["complete_caption_sha256"],
        "action_prompt_utf8_bytes": len(caption_bytes),
        "accepted_external_conditions": ["source_video", "action_prompt"],
        "target_video": False,
        "external_reference_image_or_video": False,
        "external_mask_flow_pose_track_trajectory": False,
        "external_first_frame_anchor": False,
    }:
        raise PairV5SourceBoundScoringError("native input/privileged-condition closure differs")

    checkpoint = _closed(native["checkpoint"], {"path", "tree_sha256", "content"}, label="native checkpoint")
    content = _closed(
        checkpoint["content"],
        {"manifest_path", "manifest_sha256_computed", "manifest_sha256_expected",
         "verified_file_count", "every_file_sha256_verified", "verified_entries_digest"},
        label="native checkpoint content",
    )
    checkpoint_path = Path(str(checkpoint["path"]))
    manifest_path = Path(str(content["manifest_path"]))
    if (
        not checkpoint_path.is_absolute() or checkpoint_path == Path("/")
        or not manifest_path.is_absolute() or manifest_path == Path("/")
        or "\x00" in str(checkpoint_path) or "\x00" in str(manifest_path)
    ):
        raise PairV5SourceBoundScoringError("native checkpoint/manifest path form differs")
    generation = evaluator_spec["generation_provenance"]
    if (
        checkpoint["tree_sha256"] != generation["checkpoint_tree_sha256"]
        or content["manifest_sha256_computed"] != generation["checkpoint_manifest_sha256"]
        or content["manifest_sha256_expected"] != generation["checkpoint_manifest_sha256"]
        or content["verified_file_count"] != generation["checkpoint_file_count"]
        or content["every_file_sha256_verified"] is not True
        or content["verified_entries_digest"] != generation["checkpoint_entries_digest"]
    ):
        raise PairV5SourceBoundScoringError("native checkpoint content seal differs")

    preprocessing = _closed(
        native["preprocessing"],
        {"frame_count", "fps", "reported_fps", "source_input_hw", "source_derived_bucket_hw",
         "max_pixels", "stride", "temporal_policy", "spatial_policy", "resize",
         "external_shared_i0", "decoded_from_private_byte_snapshot", "snapshot_sha256",
         "original_pre_snapshot_sha256", "original_post_snapshot_sha256",
         "original_stat_identity_stable"},
        label="native preprocessing",
    )
    bucket = preprocessing["source_derived_bucket_hw"]
    source_hw = preprocessing["source_input_hw"]
    if (
        preprocessing["frame_count"] != 81 or preprocessing["fps"] != 25
        or float(preprocessing["reported_fps"]) != 25.0
        or not isinstance(source_hw, list) or len(source_hw) != 2
        or not isinstance(bucket, list) or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
        or preprocessing["max_pixels"] != 245760 or preprocessing["stride"] != 16
        or preprocessing["temporal_policy"] != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing["spatial_policy"] != "sqrt_max_pixels_then_floor_each_dimension_to_stride"
        or preprocessing["resize"] != "torchvision_bicubic_antialias_true"
        or preprocessing["external_shared_i0"] is not False
        or preprocessing["decoded_from_private_byte_snapshot"] is not True
        or preprocessing["snapshot_sha256"] != candidate["source_video_sha256"]
        or preprocessing["original_pre_snapshot_sha256"] != candidate["source_video_sha256"]
        or preprocessing["original_post_snapshot_sha256"] != candidate["source_video_sha256"]
        or preprocessing["original_stat_identity_stable"] is not True
    ):
        raise PairV5SourceBoundScoringError("native exact81 source preprocessing differs")

    prompt_root = _closed(native["prompt_contract"], {"rv2v"}, label="native prompt root")
    prompt = _closed(
        prompt_root["rv2v"],
        {"training_task_name", "inference_arm", "guidance_mode", "system_prompt_sha256",
         "binding_clause_sha256", "full_prompt_sha256", "cleaner", "tokenizer_fix_mistral_regex"},
        label="native prompt contract",
    )
    if (
        prompt["training_task_name"] != "vr2v" or prompt["inference_arm"] != "rv2v"
        or prompt["guidance_mode"] != "rv2v"
        or prompt["cleaner"] != "diffusers.pipelines.wan.pipeline_wan.prompt_clean"
        or prompt["tokenizer_fix_mistral_regex"] is not True
    ):
        raise PairV5SourceBoundScoringError("native prompt contract differs")
    for key in ("system_prompt_sha256", "binding_clause_sha256", "full_prompt_sha256"):
        _sha256(prompt[key], label=f"native prompt {key}")

    source_ids = {
        "target_source_id": 0, "video_source_ids": [1], "reference_source_ids": [2, 3, 4, 5],
        "conditioning_source_count": 5, "max_conditioning_source_id": 5,
        "within_pretrained_source_ids_1_through_5": True,
        "source_id_interpolation_required": False,
    }
    conditioning_root = _closed(native["conditioning"], {"rv2v"}, label="native conditioning root")
    conditioning = _closed(
        conditioning_root["rv2v"],
        {"full_source_video_count", "source_derived_reference_count", "source_frame_indices",
         "reference_encoding", "reference_from_temporal_video_latent_slice", "source_ids"},
        label="native conditioning",
    )
    if conditioning != {
        "full_source_video_count": 1, "source_derived_reference_count": 4,
        "source_frame_indices": [0, 27, 53, 80],
        "reference_encoding": "independent_rgb_frame_to_wan_vae_[1,C,1,H,W]",
        "reference_from_temporal_video_latent_slice": False, "source_ids": source_ids,
    }:
        raise PairV5SourceBoundScoringError("native conditioning/source-id closure differs")

    sampling_root = _closed(native["sampling"], {"rv2v"}, label="native sampling root")
    expected_sampling = {
        "num_frames": 81, "num_inference_steps": 40, "guidance_mode": "rv2v",
        "omega_vid": 1.25, "omega_img": 4.5, "omega_txt": 4.0, "omega_scale": 0.8,
        "flow_shift": 5.0, "seed": candidate["seed"], "eta": 0.5,
        "norm_threshold": [50.0, 50.0], "momentum": 0.0,
        "target_initialization": contract.EXPECTED_TARGET_INITIALIZATION,
        "target_mixed_with_source_latent": False, "custom_sampler_or_scheduler": False,
        "same_seed_and_target_shape_across_arms": True, "single_expert": "transformer_1",
        "ulysses_size": 4, **candidate["guidance"],
    }
    if sampling_root["rv2v"] != expected_sampling:
        raise PairV5SourceBoundScoringError("native sampling/guidance closure differs")

    geometry = _closed(
        native["latent_geometry"],
        {"video_latent_shape", "reference_latent_shape", "target_patch_tokens",
         "one_reference_patch_tokens", "per_arm_total_visual_tokens"},
        label="native latent geometry",
    )
    video_shape = [1, 16, 21, bucket[0] // 8, bucket[1] // 8]
    reference_shape = [1, 16, 1, bucket[0] // 8, bucket[1] // 8]
    spatial_tokens = (bucket[0] // 16) * (bucket[1] // 16)
    if (
        geometry["video_latent_shape"] != video_shape
        or geometry["reference_latent_shape"] != reference_shape
        or geometry["target_patch_tokens"] != 21 * spatial_tokens
        or geometry["one_reference_patch_tokens"] != spatial_tokens
        or geometry["per_arm_total_visual_tokens"] != {
            "t2v": 21 * spatial_tokens,
            "r2v": 26 * spatial_tokens,
            "rv2v": 46 * spatial_tokens,
        }
    ):
        raise PairV5SourceBoundScoringError("native latent/token geometry differs")

    identities = _closed(
        native["condition_identities"],
        {"rank_zero_broadcasts", "references", "full_source_video"},
        label="native condition identities",
    )
    if not isinstance(identities["rank_zero_broadcasts"], Mapping) or not isinstance(identities["references"], Mapping) or list(identities["references"]) != ["0", "27", "53", "80"] or not isinstance(identities["full_source_video"], Mapping):
        raise PairV5SourceBoundScoringError("native condition identity closure differs")

    source_artifact, _ = _artifact_in_candidate_dir(
        native["source_condition_artifact"], candidate_dir=candidate_dir,
        label="native source condition",
    )
    _closed(source_artifact, _CLEAN_LATENT_FIELDS, label="native source condition artifact")
    if (
        source_artifact.get("shape") != video_shape
        or source_artifact.get("tensor_key") != "normalized_clean_latent"
        or source_artifact.get("stored_dtype") != "torch.float32"
        or source_artifact.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or source_artifact.get("origin") != "source_video_vae_encode_before_any_decode"
        or source_artifact.get("artifact_role") != "source_video_condition"
        or source_artifact.get("source_video_vae_encode_before_any_decode") is not True
        or source_artifact.get("native_sampler_before_vae_decode") is not False
        or source_artifact.get("mp4_decode_reencode_used") is not False
        or source_artifact.get("roundtrip_byte_exact_fp32") is not True
    ):
        raise PairV5SourceBoundScoringError("native source-condition artifact differs")

    outputs = _closed(native["outputs"], {"rv2v"}, label="native outputs")
    output = _closed(
        outputs["rv2v"],
        {"path", "sha256", "frame_count", "fps", "height", "width", "normalized_clean_latent"},
        label="native RV2V output",
    )
    output_copy, output_path = _artifact_in_candidate_dir(output, candidate_dir=candidate_dir, label="native RV2V MP4")
    clean, _ = _artifact_in_candidate_dir(
        output["normalized_clean_latent"], candidate_dir=candidate_dir,
        label="native RV2V clean latent",
    )
    _closed(clean, _CLEAN_LATENT_FIELDS, label="native clean latent artifact")
    if (
        output["frame_count"] != 81 or output["fps"] != 25
        or output["height"] != bucket[0] or output["width"] != bucket[1]
        or clean.get("shape") != video_shape or clean.get("tensor_key") != "normalized_clean_latent"
        or clean.get("stored_dtype") != "torch.float32"
        or clean.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or clean.get("artifact_role") != "native_sampler_proposal"
        or clean.get("origin") != "native_sampler_before_vae_decode"
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("source_video_vae_encode_before_any_decode") is not False
        or clean.get("mp4_decode_reencode_used") is not False
        or clean.get("roundtrip_byte_exact_fp32") is not True
    ):
        raise PairV5SourceBoundScoringError("native exact81 output/clean latent differs")

    noise_root = _closed(native["initial_noise_artifacts"], {"rv2v"}, label="native noise root")
    gaussian, _ = _artifact_in_candidate_dir(
        noise_root["rv2v"], candidate_dir=candidate_dir,
        label="native official initial Gaussian",
    )
    _closed(gaussian, _GAUSSIAN_FIELDS, label="native Gaussian artifact")
    if (
        gaussian.get("shape") != video_shape or gaussian.get("tensor_key") != "official_initial_gaussian"
        or gaussian.get("generator_initial_seed") != candidate["seed"]
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("observer_only") is not True
        or gaussian.get("external_initial_noise_injection") is not False
        or gaussian.get("source_or_target_derived") is not False
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("official_randn_tensor_call_count") != 1
        or gaussian.get("original_return_tensor_forwarded_by_identity") is not True
        or gaussian.get("roundtrip_raw_value_exact") is not True
        or gaussian.get("stored_dtype") != "torch.float32"
        or gaussian.get("stored_device") != "cpu"
        or gaussian.get("tensor_value_sha256") != gaussian.get("raw_value_sha256")
        or gaussian.get("randn_tensor_call_count") != 1
        or gaussian.get("numel") != math.prod(video_shape)
        or gaussian.get("byte_count") != 4 * math.prod(video_shape)
        or gaussian.get("official_module_global_symbol") != "bernini.models.wan_diffusion.randn_tensor"
        or gaussian.get("original_callable_invoked_once_with_unchanged_arguments") is not True
        or gaussian.get("sampler_noise_replacement") is not False
    ):
        raise PairV5SourceBoundScoringError("native official Gaussian provenance differs")
    for key in ("tensor_value_sha256", "raw_value_sha256", "content_sha256"):
        _sha256(gaussian[key], label=f"native Gaussian {key}")

    generated = _closed(native["generated_identities"], {"rv2v"}, label="native generated identities")
    if not isinstance(generated["rv2v"], Mapping):
        raise PairV5SourceBoundScoringError("native generated identity differs")
    freeze = _closed(
        native["freeze_certificate"],
        {"base_frozen", "trainable_parameter_tensors", "trainable_parameter_elements", "lora_module_count"},
        label="native freeze certificate",
    )
    if freeze != {"base_frozen": True, "trainable_parameter_tensors": 0, "trainable_parameter_elements": 0, "lora_module_count": 0}:
        raise PairV5SourceBoundScoringError("native model was not frozen adapter-free")
    interpretation = _closed(
        native["interpretation"], {"purpose", "quality_claim", "training_performed", "best_arm_selected"},
        label="native interpretation",
    )
    if interpretation != {
        "purpose": "test_native_identity_conditioned_generation_before_training",
        "quality_claim": False, "training_performed": False, "best_arm_selected": False,
    } or native["experimental_canary"] is not True or native["production_claim_forbidden"] is not True or native["scientific_claim_authorized"] is not False:
        raise PairV5SourceBoundScoringError("native interpretation/training flags differ")
    return {
        "native_receipt_digest": native_digest,
        "native_generation_provenance_digest": sealed_generation["provenance_digest"],
        "mp4": output_copy, "mp4_path": output_path,
        "predecode_clean_latent": clean, "official_initial_gaussian": gaussian,
    }


def audit_rollout_candidate(
    *, candidate: Mapping[str, Any], normalized_rollout: Mapping[str, Any],
    rollout_root: Path, rollout_spec_raw_sha256: str,
    evaluator_spec: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    candidate_requested = rollout_root / candidate_id
    if not candidate_requested.is_dir() or candidate_requested.is_symlink():
        raise PairV5SourceBoundScoringError("candidate directory escaped rollout root")
    candidate_dir = candidate_requested.resolve(strict=True)
    if candidate_dir.parent != rollout_root:
        raise PairV5SourceBoundScoringError("candidate directory escaped rollout root")
    receipt_path = _plain_file(candidate_dir / "pair-v5-rollout-receipt.json", label="candidate rollout receipt")
    if receipt_path.parent != candidate_dir:
        raise PairV5SourceBoundScoringError("PAIR receipt escaped candidate directory")
    receipt, receipt_file_sha = _strict_json_file(receipt_path, label="candidate rollout receipt")
    _closed(receipt, _PAIR_FIELDS, label="candidate rollout receipt")
    receipt_digest = _verify_embedded_digest(receipt, field="receipt_digest", label="candidate rollout receipt", ensure_ascii=False)
    raw_candidate = {key: value for key, value in candidate.items() if key != "group_id"}
    group_rows = [row for row in normalized_rollout["candidates"] if row["group_id"] == candidate["group_id"]]
    ordinal = [row["candidate_id"] for row in group_rows].index(candidate_id)
    visible_gpus = contract.EXPECTED_GROUP_GPUS[candidate["group_id"]]
    envelope = {
        "schema_version": "pair-v5-native-rv2v4-candidate-v1",
        "root_spec_raw_sha256": rollout_spec_raw_sha256,
        "group_id": candidate["group_id"], "visible_gpus": visible_gpus,
        "ordinal": ordinal, "sampling_contract": _PAIR_SAMPLING,
        "semantic_input_closure": _PAIR_SEMANTIC_CLOSURE, "candidate": raw_candidate,
    }
    expected_envelope_sha = hashlib.sha256(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    ).hexdigest()
    expected_topology = {"world_size": 4, "ulysses_size": 4, "rocr_visible_devices": ",".join(str(item) for item in visible_gpus)}
    if (
        receipt["schema_version"] != contract.EXPECTED_ROLLOUT_RECEIPT_SCHEMA
        or receipt["root_spec_raw_sha256"] != rollout_spec_raw_sha256
        or receipt["candidate_envelope_sha256"] != expected_envelope_sha
        or receipt["group_id"] != candidate["group_id"]
        or receipt["visible_gpus"] != visible_gpus
        or receipt["runtime_topology"] != expected_topology
        or receipt["ordinal"] != ordinal or receipt["candidate"] != raw_candidate
        or receipt["sampling_contract"] != _PAIR_SAMPLING
        or receipt["semantic_input_closure"] != _PAIR_SEMANTIC_CLOSURE
    ):
        raise PairV5SourceBoundScoringError("PAIR envelope/topology/input closure differs")
    native_path = _plain_file(receipt["native_receipt_path"], label="native receipt")
    if native_path != candidate_dir / "receipt.json":
        raise PairV5SourceBoundScoringError("native receipt escaped candidate directory")
    native_file_sha = contract.file_sha256(native_path)
    if native_file_sha != _sha256(receipt["native_receipt_sha256"], label="native receipt file SHA-256"):
        raise PairV5SourceBoundScoringError("native receipt file hash differs")
    native, observed_native_sha = _strict_json_file(native_path, label="native receipt")
    if observed_native_sha != native_file_sha:
        raise PairV5SourceBoundScoringError("native receipt changed while reading")
    verified = _verify_native_receipt(
        native, candidate=candidate, candidate_dir=candidate_dir,
        evaluator_spec=evaluator_spec, native_file_sha256=native_file_sha,
    )
    if receipt["native_receipt_digest"] != verified["native_receipt_digest"]:
        raise PairV5SourceBoundScoringError("PAIR/native receipt digest differs")
    expected_artifacts = {
        "mp4": verified["mp4"],
        "predecode_clean_latent": verified["predecode_clean_latent"],
        "official_initial_gaussian": verified["official_initial_gaussian"],
    }
    if receipt["artifacts"] != expected_artifacts:
        raise PairV5SourceBoundScoringError("PAIR/native artifact binding differs")
    source_path, source_sha = _hash_stable_file(candidate["source_video"], expected_sha256=candidate["source_video_sha256"], label="candidate correct source")
    return {
        "receipt_digest": receipt_digest, "receipt_file_sha256": receipt_file_sha,
        "candidate_envelope_sha256": expected_envelope_sha,
        "native_rollout_receipt_digest": verified["native_receipt_digest"],
        "native_rollout_receipt_file_sha256": native_file_sha,
        "native_generation_provenance_digest": verified["native_generation_provenance_digest"],
        "mp4_path": verified["mp4_path"], "mp4_sha256": verified["mp4"]["sha256"],
        "predecode_clean_latent_sha256": verified["predecode_clean_latent"]["sha256"],
        "official_initial_gaussian_sha256": verified["official_initial_gaussian"]["sha256"],
        "source_path": source_path, "source_sha256": source_sha,
    }


def model_evidence(
    checkpoint: Mapping[str, Any],
    *,
    evaluator_spec: Mapping[str, Any],
    model: Any,
    loading_counts: Mapping[str, int],
) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    trainable = sum(1 for _, parameter in named_parameters if parameter.requires_grad)
    parameter_metadata = [
        {
            "name": name,
            "shape": [int(item) for item in parameter.shape],
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named_parameters
    ]
    return {
        "adapter_id": checkpoint["adapter_id"],
        "architecture_id": checkpoint["architecture_id"],
        "checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
        "checkpoint_config_sha256": checkpoint["checkpoint_config_sha256"],
        "preprocessor_config_sha256": checkpoint["preprocessor_config_sha256"],
        "checkpoint_file_count": checkpoint["checkpoint_file_count"],
        "verified_entries_digest": checkpoint["verified_entries_digest"],
        "preprocessor_golden_input_sha256": checkpoint[
            "preprocessor_golden_input_sha256"
        ],
        "preprocessor_golden_output_sha256": checkpoint[
            "preprocessor_golden_output_sha256"
        ],
        "preprocessor_golden_output_shape": checkpoint[
            "preprocessor_golden_output_shape"
        ],
        "every_checkpoint_file_verified": checkpoint["every_checkpoint_file_verified"],
        "all_parameters_frozen": not model.training and trainable == 0,
        "trainable_parameter_tensors": trainable,
        "parameter_tensor_count": len(named_parameters),
        "parameter_element_count": sum(
            int(parameter.numel()) for _, parameter in named_parameters
        ),
        "parameter_metadata_digest": object_sha256(parameter_metadata),
        **dict(loading_counts),
        "runtime_versions": runtime_versions(),
    }


def evaluate_one_candidate(
    *,
    candidate: Mapping[str, Any],
    candidate_ordinal: int,
    normalized_rollout: Mapping[str, Any],
    rollout_root: Path,
    rollout_spec_raw_sha256: str,
    evaluator_spec: Mapping[str, Any],
    evaluator_spec_raw_sha256: str,
    checkpoint_evidence: Mapping[str, Any],
    model: Any,
    loading_counts: Mapping[str, int],
    device: Any,
) -> dict[str, Any]:
    audit = audit_rollout_candidate(
        candidate=candidate,
        normalized_rollout=normalized_rollout,
        rollout_root=rollout_root,
        rollout_spec_raw_sha256=rollout_spec_raw_sha256,
        evaluator_spec=evaluator_spec,
    )
    wrong_sha = evaluator_spec["wrong_source_by_source_sha256"][audit["source_sha256"]]
    wrong_rows = [
        row for row in normalized_rollout["candidates"] if row["source_video_sha256"] == wrong_sha
    ]
    if len(wrong_rows) != 2 or wrong_rows[0]["source_video"] != wrong_rows[1]["source_video"]:
        raise PairV5SourceBoundScoringError("wrong source path closure differs")
    wrong_path, _ = _hash_stable_file(
        wrong_rows[0]["source_video"], expected_sha256=wrong_sha, label="wrong source"
    )
    media = {
        "candidate": (audit["mp4_path"], audit["mp4_sha256"]),
        "correct_source": (audit["source_path"], audit["source_sha256"]),
        "wrong_source": (wrong_path, wrong_sha),
    }
    raw_by_role: dict[str, Any] = {}
    normalized_by_role: dict[str, Any] = {}
    decode_evidence: dict[str, dict[str, Any]] = {}
    for role in ("candidate", "correct_source", "wrong_source"):
        frames, evidence = decode_exact81_rgb(media[role][0], expected_sha256=media[role][1])
        raw, normalized = preprocess_selected_rgb(frames, checkpoint_evidence["processor"])
        evidence["preprocessed_tensor_sha256"] = tensor_sha256(normalized)
        raw_by_role[role] = raw
        normalized_by_role[role] = normalized
        decode_evidence[role] = evidence
    feature_tensors: dict[str, tuple[Any, Any]] = {}
    feature_evidence: dict[str, dict[str, Any]] = {}
    for role in ("candidate", "correct_source", "wrong_source"):
        global_feature, dense_feature, evidence = extract_features(
            model,
            normalized_by_role[role],
            device=device,
            num_register_tokens=evaluator_spec["model"]["num_register_tokens"],
            evaluation_image_size=evaluator_spec["model"][
                "preprocessor_golden_output_shape"
            ][-1],
            patch_size=evaluator_spec["model"]["patch_size"],
        )
        feature_tensors[role] = (global_feature, dense_feature)
        feature_evidence[role] = evidence
    metrics = compute_metrics(
        candidate_global=feature_tensors["candidate"][0],
        candidate_dense=feature_tensors["candidate"][1],
        correct_global=feature_tensors["correct_source"][0],
        correct_dense=feature_tensors["correct_source"][1],
        wrong_global=feature_tensors["wrong_source"][0],
        wrong_dense=feature_tensors["wrong_source"][1],
        candidate_raw=raw_by_role["candidate"],
        correct_raw=raw_by_role["correct_source"],
    )
    return contract.make_candidate_receipt(
        evaluator_spec=evaluator_spec,
        evaluator_spec_raw_sha256=evaluator_spec_raw_sha256,
        candidate_id=candidate["candidate_id"],
        candidate_ordinal=candidate_ordinal,
        group_id=candidate["group_id"],
        candidate_envelope_sha256=audit["candidate_envelope_sha256"],
        rollout_receipt_digest=audit["receipt_digest"],
        rollout_receipt_file_sha256=audit["receipt_file_sha256"],
        native_rollout_receipt_digest=audit["native_rollout_receipt_digest"],
        native_rollout_receipt_file_sha256=audit[
            "native_rollout_receipt_file_sha256"
        ],
        native_generation_provenance_digest=audit[
            "native_generation_provenance_digest"
        ],
        candidate_mp4_sha256=audit["mp4_sha256"],
        predecode_clean_latent_sha256=audit["predecode_clean_latent_sha256"],
        official_initial_gaussian_sha256=audit[
            "official_initial_gaussian_sha256"
        ],
        correct_source_video_sha256=audit["source_sha256"],
        wrong_source_video_sha256=wrong_sha,
        decode_evidence_by_role=decode_evidence,
        feature_evidence_by_role=feature_evidence,
        model_evidence=model_evidence(
            checkpoint_evidence,
            evaluator_spec=evaluator_spec,
            model=model,
            loading_counts=loading_counts,
        ),
        metrics=metrics,
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise PairV5SourceBoundScoringError("refusing to overwrite evaluator receipt")
    raw = canonical_json_bytes(value) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise PairV5SourceBoundScoringError("receipt appeared concurrently") from error
    os.chmod(path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--evaluator-spec", required=True)
    parser.add_argument("--expected-evaluator-spec-sha256", required=True)
    parser.add_argument("--rollout-output-dir", required=True)
    parser.add_argument("--group-id", required=True, choices=contract.EXPECTED_GROUPS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _SHA1.fullmatch(args.method_source_revision) is None:
        raise PairV5SourceBoundScoringError("method source revision differs")
    _sha256(args.method_source_archive_sha256, label="method source archive SHA-256")
    normalized_rollout, rollout_sha = contract.load_current_family_rollout_spec(
        Path(args.root_spec), args.expected_root_spec_sha256
    )
    evaluator_spec, evaluator_spec_raw_sha = contract.load_evaluator_spec(
        Path(args.evaluator_spec),
        args.expected_evaluator_spec_sha256,
        normalized_rollout=normalized_rollout,
        rollout_spec_raw_sha256=rollout_sha,
        implementation_path=Path(__file__),
        contract_path=Path(contract.__file__),
    )
    if runtime_versions() != evaluator_spec["runtime_versions"]:
        raise PairV5SourceBoundScoringError("runtime version binding differs")
    if (
        args.method_source_revision != evaluator_spec["method_source_revision"]
        or args.method_source_archive_sha256
        != evaluator_spec["method_source_archive_sha256"]
    ):
        raise PairV5SourceBoundScoringError("method source archive/spec binding differs")
    rollout_root = _plain_directory(args.rollout_output_dir, label="rollout output root")
    checkpoint_evidence = verify_checkpoint_content(
        args.checkpoint,
        args.checkpoint_content_manifest,
        evaluator_spec=evaluator_spec,
    )
    group_id = args.group_id
    group_candidates = [
        row for row in normalized_rollout["candidates"] if row["group_id"] == group_id
    ]
    if len(group_candidates) != 4:
        raise PairV5SourceBoundScoringError("group candidate count differs")

    import torch
    import torch.distributed as distributed

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise PairV5SourceBoundScoringError("evaluator requires one four-rank group")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise PairV5SourceBoundScoringError("rank4 evaluator requires exactly four visible GPUs")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group(backend="nccl")
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/"):
        raise PairV5SourceBoundScoringError("output must be absolute non-root")
    if rank == 0:
        if output.exists() or output.is_symlink():
            raise PairV5SourceBoundScoringError("refusing evaluator output reuse")
        output.mkdir(parents=False, exist_ok=False)
    distributed.barrier()
    if not output.is_dir() or output.is_symlink():
        raise PairV5SourceBoundScoringError("evaluator output differs")

    candidate = group_candidates[rank]
    candidate_id = candidate["candidate_id"]
    candidate_ordinal = evaluator_spec["candidate_order"].index(candidate_id)
    stage = "model_load"
    failed = False
    try:
        model, loading_counts = load_frozen_model(checkpoint_evidence, device=device)
        stage = "candidate_evaluation"
        receipt = evaluate_one_candidate(
            candidate=candidate,
            candidate_ordinal=candidate_ordinal,
            normalized_rollout=normalized_rollout,
            rollout_root=rollout_root,
            rollout_spec_raw_sha256=rollout_sha,
            evaluator_spec=evaluator_spec,
            evaluator_spec_raw_sha256=evaluator_spec_raw_sha,
            checkpoint_evidence=checkpoint_evidence,
            model=model,
            loading_counts=loading_counts,
            device=device,
        )
        _write_new_json(output / f"{candidate_id}.json", receipt)
    except Exception as error:
        failed = True
        failure = contract.make_failure_receipt(
            evaluator_spec=evaluator_spec,
            evaluator_spec_raw_sha256=evaluator_spec_raw_sha,
            candidate_id=candidate_id,
            candidate_ordinal=candidate_ordinal,
            group_id=group_id,
            failure_stage=stage,
            error=error,
        )
        _write_new_json(output / f"{candidate_id}.failure.json", failure)
    failure_tensor = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    distributed.all_reduce(failure_tensor, op=distributed.ReduceOp.SUM)
    failure_count = int(failure_tensor.item())
    distributed.barrier()
    if failure_count:
        distributed.destroy_process_group()
        return 2
    aggregate_status = torch.zeros(1, dtype=torch.int32, device=device)
    if rank == 0:
        try:
            receipts: list[dict[str, Any]] = []
            file_hashes: dict[str, str] = {}
            for row in group_candidates:
                path = _plain_file(
                    output / f"{row['candidate_id']}.json",
                    label="candidate evidence receipt",
                )
                value, digest = _strict_json_file(
                    path, label="candidate evidence receipt"
                )
                checked = contract.validate_candidate_receipt(
                    value,
                    evaluator_spec=evaluator_spec,
                    evaluator_spec_raw_sha256=evaluator_spec_raw_sha,
                )
                receipts.append(checked)
                file_hashes[row["candidate_id"]] = digest
            group_receipt = contract.make_group_receipt(
                evaluator_spec=evaluator_spec,
                evaluator_spec_raw_sha256=evaluator_spec_raw_sha,
                group_id=group_id,
                candidate_receipts=receipts,
                candidate_receipt_file_sha256_by_id=file_hashes,
            )
            _write_new_json(
                output / f"pair-v5-source-bound-preservation-{group_id}-v1.json",
                group_receipt,
            )
        except Exception as error:
            aggregate_status.fill_(1)
            unsigned_failure = {
                "schema_version": "bernini-pair-v5-source-bound-group-failure-v1",
                "evaluator_spec_digest": evaluator_spec["spec_digest"],
                "group_id": group_id,
                "failure_stage": "group_aggregation",
                "error_class": type(error).__name__,
                "error_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "evidence_valid": False,
                "eligible_for_downstream_calibration": False,
            }
            _write_new_json(
                output / f"pair-v5-source-bound-preservation-{group_id}-failure-v1.json",
                {
                    **unsigned_failure,
                    "receipt_digest": object_sha256(unsigned_failure),
                },
            )
    distributed.broadcast(aggregate_status, src=0)
    if int(aggregate_status.item()) != 0:
        distributed.destroy_process_group()
        return 2
    distributed.barrier()
    distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
