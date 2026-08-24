#!/usr/bin/env python3
"""Encode the exact ELAL-3 simulator C1 packet with the frozen Bernini VAE.

This is a data-preparation diagnostic, not a trainer.  It accepts exactly the
``c1-two-entity-push-to-goal`` row from the pinned simulator packet, verifies
all eight media byte identities, applies Bernini's native inference resize,
and performs eight independent full-video VAE encodes.  The output is one
safe-tensors bundle plus a canonical receipt.  It grants no formal C1,
exact160, real-video, or scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_lora as legacy
from tools import materialize_vae


SCHEMA_VERSION = "bernini-elal3-simulator-c1-latent-bundle-v1"
RECEIPT_SCHEMA = "bernini-elal3-simulator-c1-latent-bundle-receipt-v1"
ROW_ID = "c1-two-entity-push-to-goal"
PACKET_MANIFEST_SHA256 = (
    "2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"
)
DERIVATIVE_AUTHORITY_SCHEMA = (
    "bernini-elal3-simulator-optimizer-derivative-authority-v1"
)
DERIVATIVE_AUTHORITY_SHA256 = (
    "298e0f31027e1c085196fd23401268d4113da9201dd95e57fa8c6b6f13ee0a5b"
)
MODEL_AUTHORITY_SCHEMA = "bernini-elal3-c1-real-model-authority-v1"
MODEL_AUTHORITY_SHA256 = (
    "4c2f4d28af646ab39bdeb775e1b651d523d83b3fc0b8e5c1dd4bc78fbd4f25ed"
)
MODEL_AUTHORITY_DIGEST = (
    "25255902f4c5ce6de94ce6c3666bcf85eae4bf8e360a217f327c6febd049d21b"
)
MEDIA_ORDER = (
    "source",
    "target",
    "anchor",
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
FRAME_COUNT = 81
LATENT_PHASES = 21
LATENT_CHANNELS = 16
MAX_PIXELS = 245_760
SPATIAL_STRIDE = 16
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUNTIME_DEPENDENCY_SHA256 = {
    "train_lora.py": "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5",
    "tools/materialize_vae.py": "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0",
    "tools/build_renderer_dataset.py": "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5",
}


class ELAL3SimulatorVAEError(RuntimeError):
    """Raised before an ambiguous simulator latent can be published."""


def fail(message: str) -> None:
    raise ELAL3SimulatorVAEError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ELAL3SimulatorVAEError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_file(path: Path, *, expected_sha256: Optional[str] = None) -> tuple[bytes, dict[str, Any]]:
    requested = path.expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"file must be one absolute non-symlink path: {requested}")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorVAEError(f"file is unavailable: {requested}: {error}") from error
    if resolved != requested or not stat.S_ISREG(resolved.lstat().st_mode):
        fail(f"file type/canonical path differs: {requested}")
    with resolved.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    named = resolved.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named):
        fail(f"file changed while reading: {resolved}")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        fail(f"file SHA-256 differs: {resolved}")
    return payload, {
        "path": str(resolved),
        "sha256": digest,
        "size": len(payload),
        "mode": stat.S_IMODE(named.st_mode),
        "device": named.st_dev,
        "inode": named.st_ino,
        "nlink": named.st_nlink,
    }


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def reject(value: str) -> None:
        raise ELAL3SimulatorVAEError(f"{label} contains non-finite JSON: {value}")

    def pairs(rows: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ELAL3SimulatorVAEError(f"{label} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            parse_constant=reject,
            object_pairs_hook=pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ELAL3SimulatorVAEError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    return value


def validate_packet(packet_root: Path, manifest_sha256: str) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, dict[str, Any]]]:
    if manifest_sha256 != PACKET_MANIFEST_SHA256:
        fail("simulator packet manifest literal differs")
    root = packet_root.expanduser()
    if not root.is_absolute() or root.is_symlink():
        fail("packet root must be one absolute non-symlink directory")
    root = root.resolve(strict=True)
    if not stat.S_ISDIR(root.lstat().st_mode):
        fail("packet root is not a directory")
    raw, manifest_binding = stable_file(
        root / "manifest.json", expected_sha256=manifest_sha256
    )
    manifest = _strict_json(raw, label="simulator manifest")
    authority = manifest.get("authority")
    rows = manifest.get("rows")
    if (
        manifest.get("schema_version") != "elal3-simulator-gt-canary-v1"
        or manifest.get("status") != "ELAL3_SIM_DIAGNOSTIC"
        or manifest.get("row_count") != 3
        or manifest.get("media_count") != 24
        or not isinstance(authority, Mapping)
        or authority.get("simulator_only") is not True
        or authority.get("training_authorized") is not False
        or authority.get("training_use_forbidden") is not True
        or authority.get("formal_c0_c1_c2_go_authorized") is not False
        or authority.get("exact160_claim_authorized") is not False
        or authority.get("scientific_claim_authorized") is not False
        or not isinstance(rows, list)
    ):
        fail("simulator manifest authority/count closure differs")
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("row_id") == ROW_ID]
    if len(selected) != 1:
        fail("exact C1 simulator row is absent or duplicated")
    row = selected[0]
    media = row.get("media")
    if (
        row.get("gate") != "C1_TWO_ENTITY_ONE_ROW_OVERFIT"
        or row.get("entity_count") != 2
        or not isinstance(media, Mapping)
        or set(media) != set(MEDIA_ORDER)
    ):
        fail("C1 row/media closure differs")
    bindings: dict[str, dict[str, Any]] = {}
    for role in MEDIA_ORDER:
        entry = media[role]
        if not isinstance(entry, Mapping) or entry.get("variant") != role:
            fail(f"C1 media role differs: {role}")
        expected_sha = entry.get("sha256")
        relative = entry.get("path")
        if (
            not isinstance(expected_sha, str)
            or _SHA256.fullmatch(expected_sha) is None
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            fail(f"C1 media binding differs: {role}")
        _, binding = stable_file(root / relative, expected_sha256=expected_sha)
        bindings[role] = binding
    return manifest_binding, row, bindings


def validate_derivative_authority(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if expected_sha256 != DERIVATIVE_AUTHORITY_SHA256:
        fail("derivative authority literal SHA-256 differs")
    payload, binding = stable_file(path, expected_sha256=expected_sha256)
    value = _strict_json(payload, label="derivative optimizer authority")
    unsigned = dict(value)
    stored = unsigned.pop("authority_digest", None)
    restrictions = value.get("training_objective_restrictions")
    disallowed = value.get("disallowed_claims")
    if (
        value.get("schema_version") != DERIVATIVE_AUTHORITY_SCHEMA
        or value.get("status")
        != "AUTHORIZED_SIMULATOR_ORACLE_Q_DIAGNOSTIC_ONLY"
        or value.get("authorized_row_id") != ROW_ID
        or value.get("packet_manifest_sha256") != PACKET_MANIFEST_SHA256
        or value.get("supersedes_packet_training_use_forbidden_for_exact_scope_only")
        is not True
        or value.get("fresh_optimizer_run_required") is not True
        or value.get("max_optimizer_updates_per_arm") != 20
        or value.get("oracle_q_teacher_forced_required") is not True
        or not isinstance(disallowed, Mapping)
        or set(disallowed.values()) != {True}
        or not isinstance(restrictions, Mapping)
        or restrictions.get("frozen_base_velocity_reference_forbidden") is not True
        or restrictions.get("frozen_teacher_self_distillation_forbidden") is not True
        or restrictions.get("hand_tuned_reward_scalar_forbidden") is not True
        or restrictions.get("target_grounded_event_and_context_flow_only") is not True
        or not isinstance(stored, str)
        or object_sha256(unsigned) != stored
    ):
        fail("derivative optimizer authority closure/digest differs")
    return value, binding


def validate_model_authority(
    path: Path,
    expected_sha256: str,
    *,
    bernini_root: Path,
    checkpoint_root: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Replay the exact model/runtime bytes before and after VAE encoding."""

    if expected_sha256 != MODEL_AUTHORITY_SHA256:
        fail("real-model authority literal SHA-256 differs")
    payload, authority_binding = stable_file(path, expected_sha256=expected_sha256)
    value = _strict_json(payload, label="real-model authority")
    if set(value) != {
        "authority_digest",
        "bernini_root",
        "checkpoint_root",
        "constraints",
        "file_count",
        "files",
        "model_family",
        "python_env_root",
        "row_id",
        "runtime_versions",
        "schema_version",
    }:
        fail("real-model authority top-level fields differ")
    unsigned = dict(value)
    stored_digest = unsigned.pop("authority_digest", None)
    constraints = value.get("constraints")
    rows = value.get("files")
    if (
        value.get("schema_version") != MODEL_AUTHORITY_SCHEMA
        or value.get("row_id") != ROW_ID
        or stored_digest != MODEL_AUTHORITY_DIGEST
        or object_sha256(unsigned) != stored_digest
        or str(bernini_root.resolve(strict=True)) != value.get("bernini_root")
        or str(checkpoint_root.resolve(strict=True)) != value.get("checkpoint_root")
        or value.get("file_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
        or not isinstance(constraints, Mapping)
        or constraints.get("allowed_operation")
        != "elal3_c1_simulator_oracle_q_optimizer_diagnostic"
        or constraints.get("max_optimizer_updates_per_arm") != 20
        or constraints.get("formal_c1_authorized") is not False
        or constraints.get("exact160_authorized") is not False
        or constraints.get("scientific_claim_authorized") is not False
        or constraints.get("real_video_claim_authorized") is not False
        or constraints.get("source_instruction_inference_claim_authorized") is not False
    ):
        fail("real-model authority closure/digest differs")
    root_values = {
        "bernini": value.get("bernini_root"),
        "checkpoint": value.get("checkpoint_root"),
        "python_env": value.get("python_env_root"),
    }
    if any(
        not isinstance(item, str)
        or not Path(item).is_absolute()
        or Path(item).is_symlink()
        for item in root_values.values()
    ):
        fail("real-model authority root closure differs")
    bindings: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "mode",
            "relative_path",
            "root",
            "sha256",
            "size",
        }:
            fail("real-model authority file row fields differ")
        root_name = row.get("root")
        relative = row.get("relative_path")
        expected_file_sha = row.get("sha256")
        if (
            root_name not in root_values
            or not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_file_sha, str)
            or _SHA256.fullmatch(expected_file_sha) is None
            or not isinstance(row.get("size"), int)
            or not isinstance(row.get("mode"), int)
        ):
            fail("real-model authority file row value differs")
        identity = (str(root_name), relative)
        if identity in identities:
            fail("real-model authority file row is duplicated")
        identities.add(identity)
        _, binding = stable_file(
            Path(str(root_values[str(root_name)])) / relative,
            expected_sha256=expected_file_sha,
        )
        if (
            binding["size"] != row["size"]
            or binding["mode"] != row["mode"]
            or binding["nlink"] != 1
        ):
            fail("real-model authority physical file binding differs")
        bindings.append(binding)
    return value, authority_binding, bindings


def tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        fail("tensor hash input differs")
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(map(int, tensor.shape))}
    )
    raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--packet-root", required=True)
    value.add_argument("--packet-manifest-sha256", required=True)
    value.add_argument("--derivative-authority", required=True)
    value.add_argument("--derivative-authority-sha256", required=True)
    value.add_argument("--model-authority", required=True)
    value.add_argument("--model-authority-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--ack-simulator-oracle-diagnostic", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if not args.ack_simulator_oracle_diagnostic:
        fail("explicit simulator oracle diagnostic acknowledgement is required")
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be one fresh absolute path")
    packet_root = Path(args.packet_root).expanduser().resolve(strict=True)
    derivative_authority, derivative_authority_binding = validate_derivative_authority(
        Path(args.derivative_authority).expanduser(),
        args.derivative_authority_sha256,
    )
    model_authority, model_authority_binding, model_files_pre = validate_model_authority(
        Path(args.model_authority).expanduser(),
        args.model_authority_sha256,
        bernini_root=Path(args.bernini_root).expanduser(),
        checkpoint_root=Path(args.checkpoint).expanduser(),
    )
    runtime_dependencies = {
        "train_lora.py": Path(legacy.__file__).resolve(strict=True),
        "tools/materialize_vae.py": Path(materialize_vae.__file__).resolve(strict=True),
        "tools/build_renderer_dataset.py": Path(
            materialize_vae.raw_builder.__file__
        ).resolve(strict=True),
    }
    runtime_dependency_bindings = {
        name: stable_file(path, expected_sha256=RUNTIME_DEPENDENCY_SHA256[name])[1]
        for name, path in runtime_dependencies.items()
    }
    manifest_binding, row, media_bindings = validate_packet(
        packet_root, args.packet_manifest_sha256
    )
    bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    from diffusers import AutoencoderKLWan
    from safetensors.torch import save_file
    from bernini.pipeline import _vae_encode

    if not torch.cuda.is_available():
        fail("Bernini simulator VAE materialization requires one GPU")
    device = torch.device("cuda", 0)
    bundle_path = output / "c1-latents.safetensors"
    receipt_path = output / "latent-bundle-receipt.json"

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    tensors: dict[str, Any] = {}
    tensor_rows: list[dict[str, Any]] = []
    bucket_hw: Optional[tuple[int, int]] = None
    for role in MEDIA_ORDER:
        media_path = Path(media_bindings[role]["path"])
        frames, reported_fps, source_hw = materialize_vae._decode_exact_video(media_path)
        if int(frames.shape[0]) != FRAME_COUNT or abs(float(reported_fps) - 25.0) > 1e-3:
            fail(f"C1 decoded video metadata differs: {role}")
        current_bucket = materialize_vae.source_aspect_bucket(
            *source_hw, max_pixels=MAX_PIXELS, stride=SPATIAL_STRIDE
        )
        pixels = materialize_vae._resize_video(frames, current_bucket, None).unsqueeze(0)
        if bucket_hw is None:
            bucket_hw = current_bucket
        if current_bucket != bucket_hw:
            fail("all C1 simulator media must share one Bernini bucket")
        with torch.no_grad():
            latent = _vae_encode(
                vae, pixels.to(device=device, dtype=torch.float32)
            ).float().cpu().contiguous()
        expected_shape = (
            1,
            LATENT_CHANNELS,
            LATENT_PHASES,
            bucket_hw[0] // 8,
            bucket_hw[1] // 8,
        )
        if (
            tuple(map(int, latent.shape)) != expected_shape
            or latent.dtype != torch.float32
            or not bool(torch.isfinite(latent).all().item())
        ):
            fail(f"C1 VAE latent geometry/value differs: {role}")
        tensors[role] = latent
        tensor_rows.append(
            {
                "role": role,
                "shape": list(expected_shape),
                "dtype": str(latent.dtype),
                "sha256": tensor_sha256(latent),
                "source_media_sha256": media_bindings[role]["sha256"],
            }
        )
        del pixels, latent
        torch.cuda.empty_cache()
    if bucket_hw is None or set(tensors) != set(MEDIA_ORDER):
        fail("C1 latent tensor closure differs")
    model_authority_post, model_authority_binding_post, model_files_post = (
        validate_model_authority(
            Path(args.model_authority).expanduser(),
            args.model_authority_sha256,
            bernini_root=Path(args.bernini_root).expanduser(),
            checkpoint_root=Path(args.checkpoint).expanduser(),
        )
    )
    if (
        model_authority_post != model_authority
        or model_authority_binding_post != model_authority_binding
        or model_files_post != model_files_pre
    ):
        fail("real-model authority changed during VAE materialization")
    output.mkdir(mode=0o700)
    save_file(
        tensors,
        str(bundle_path),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "row_id": ROW_ID,
            "packet_manifest_sha256": PACKET_MANIFEST_SHA256,
            "bucket_hw": f"{bucket_hw[0]},{bucket_hw[1]}",
        },
    )
    os.chmod(bundle_path, 0o444)
    bundle_payload, bundle_binding = stable_file(bundle_path)
    if len(bundle_payload) <= 0:
        fail("C1 latent bundle is empty")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "row_id": ROW_ID,
        "bundle": bundle_binding,
        "bundle_format": "safetensors-exact8-fp32-v1",
        "tensor_order": list(MEDIA_ORDER),
        "tensor_rows": tensor_rows,
        "bucket_hw": list(bucket_hw),
        "latent_shape": [1, 16, 21, bucket_hw[0] // 8, bucket_hw[1] // 8],
        "packet_manifest": manifest_binding,
        "derivative_optimizer_authority": {
            "file": derivative_authority_binding,
            "authority_digest": derivative_authority["authority_digest"],
            "schema_version": derivative_authority["schema_version"],
        },
        "real_model_authority": {
            "file": model_authority_binding,
            "authority_digest": model_authority["authority_digest"],
            "schema_version": model_authority["schema_version"],
            "verified_file_bindings": model_files_pre,
            "verified_before_and_after_encoding": True,
        },
        "media_bindings": {key: media_bindings[key] for key in MEDIA_ORDER},
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
            "transformer_layers": transformer_config.get("num_layers"),
            "transformer_hidden_width": (
                int(transformer_config.get("num_attention_heads", 0))
                * int(transformer_config.get("attention_head_dim", 0))
            ),
        },
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "runtime_dependencies": runtime_dependency_bindings,
        "vae_encode_count": len(MEDIA_ORDER),
        "each_media_independently_full_video_vae_encoded": True,
        "simulator_optimizer_diagnostic_authorized": True,
        "oracle_q_required_for_training": True,
        "source_instruction_inference": False,
        "formal_c1_authorized": False,
        "exact160_authorized": False,
        "scientific_claim_authorized": False,
        "real_video_data": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail("receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    stable_file(receipt_path, expected_sha256=hashlib.sha256(payload).hexdigest())
    os.chmod(receipt_path, 0o444)
    os.chmod(output, 0o555)
    print(
        json.dumps(
            {
                "status": "ELAL3_SIMULATOR_C1_VAE_GO",
                "bundle": str(bundle_path),
                "bundle_sha256": bundle_binding["sha256"],
                "receipt": str(receipt_path),
                "receipt_sha256": hashlib.sha256(payload).hexdigest(),
                "bucket_hw": list(bucket_hw),
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
