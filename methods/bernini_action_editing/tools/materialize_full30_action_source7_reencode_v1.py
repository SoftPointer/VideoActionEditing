#!/usr/bin/env python3
"""Encode the BOX-EXP-014 exact7 source MP4s once with the pinned Wan VAE.

Only the seven source-video paths in the frozen plan are opened.  Each video
is decoded as exact81 RGB, resized with the official source-aspect bucket, and
encoded in exactly one VAE call.  Every output is a bare, CPU, contiguous,
finite FP32 ``posterior_parameters`` tensor serialized by
``materialize_vae.tensor_to_bytes``.  No posterior sample is materialized.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full30_action_source7_reencode_plan_v1 as plan_contract  # noqa: E402
from tools import materialize_vae as base  # noqa: E402
from tools import materialize_ramp_motion_analogy_vae as pinned  # noqa: E402
SCHEMA_VERSION = "bernini-full30-action-source7-reencode-receipt-v1"
ROW_SCHEMA = "bernini-full30-action-source7-reencode-row-receipt-v1"
METHOD_NAME = "bernini-full30-action-source7-reencode-v1"
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
NEGATIVE_ACCESS_FIELDS = frozenset(
    {
        "source_only_reencode_from_source_video",
        "vae_encode_calls_per_source",
        "paired_dataset_accessed",
        "legacy_source_target_container_opened",
        "synthetic_target_index1_path_read",
        "synthetic_target_index1_bytes_read",
        "synthetic_target_index1_decoded",
        "synthetic_target_index1_filtered_on",
        "synthetic_target_index1_hashed",
        "target_video_path_present",
        "target_video_accessed",
    }
)
ROW_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "analysis_split",
        "event_id",
        "actor_kind",
        "q0_id",
        "group_id",
        "actor_id",
        "scene_id",
        "source_video_path",
        "source_video_sha256",
        "source_video_stat_identity",
        "source_video_sha256_before_decode",
        "source_video_sha256_after_decode",
        "source_video_pre_post_stat_and_hash_stable",
        "frame_count",
        "expected_fps",
        "reported_fps",
        "input_hw",
        "source_aspect_bucket_hw",
        "posterior_parameters_path",
        "posterior_parameters_file_sha256",
        "posterior_parameters_tensor_sha256",
        "posterior_parameters_tensor_raw_sha256",
        "posterior_parameters_shape",
        "posterior_parameters_dtype",
        "posterior_parameters_device",
        "posterior_parameters_layout",
        "posterior_parameters_contiguous",
        "posterior_parameters_finite",
        "posterior_parameters_bare_tensor",
        "posterior_sample_materialized",
        "physical_file_reopened_after_write",
        "physical_tensor_reopened_after_write",
        "physical_tensor_equal_to_encoded_tensor",
        "peak_allocated_bytes",
    }
) | NEGATIVE_ACCESS_FIELDS
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "experiment_id",
        "complete",
        "plan",
        "output_root",
        "row_count",
        "rows",
        "external_existing_index0",
        "vae_identity",
        "output_filenames",
        "output_exact_member_closure",
        "distinct_source_mp4_count",
        "total_vae_encode_calls",
        "posterior_sample_materialized",
        "external_existing_index0_opened",
        "external_existing_index0_reencoded",
        "inventory_snapshot_only",
        "exact8_authority_go_claimed",
        "teacher_cross_disjointness_pending",
        "optimizer_created",
        "optimizer_updates",
        "training_authorized",
        "receipt_digest",
    }
) | NEGATIVE_ACCESS_FIELDS


class Source7ReencodeMaterializationError(RuntimeError):
    """Raised before incomplete or widened source-only output can pass."""


def _bind_official_source_self_primitives() -> Any:
    """Bind the exact primitives used by the committed source-self materializer.

    The import is deliberately delayed so static release audits do not require
    torch.  The live AUH materialization must import the reference module and
    prove that its decoder, bucket rule, and pinned encoder are these exact
    function/class objects before any source MP4 is opened.
    """

    from tools import materialize_source_self_role_repaint as source_self

    require(
        source_self.base._decode_exact_video is base._decode_exact_video
        and source_self.base.source_aspect_bucket is base.source_aspect_bucket
        and source_self.pinned.PinnedBerniniWanPosteriorEncoder
        is pinned.PinnedBerniniWanPosteriorEncoder,
        "official source-self decoder/bucket/pinned encoder binding differs",
    )
    return source_self


def fail(message: str) -> NoReturn:
    raise Source7ReencodeMaterializationError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return plan_contract.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return plan_contract.object_sha256(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    require(type(value) is str and bool(value), f"{label} path differs")
    requested = Path(value)
    require(requested.is_absolute() and not requested.is_symlink(), f"{label} must be absolute and non-symlink")
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.lstat()
    except OSError as error:
        raise Source7ReencodeMaterializationError(f"{label} is unavailable") from error
    require(
        resolved == requested
        and stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be one canonical plain file",
    )
    return resolved


def _load_plan(path_value: str | Path, expected_sha256: str) -> tuple[Mapping[str, Any], Path, str]:
    path = _plain_absolute_file(str(path_value), label="source7 plan")
    before = _identity(path)
    raw = path.read_bytes()
    after = _identity(path)
    require(before == after and bool(raw), "source7 plan changed while reading")
    observed = hashlib.sha256(raw).hexdigest()
    require(observed == expected_sha256, "source7 plan file SHA-256 differs")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=lambda pairs: _closed_pairs(pairs),
            parse_constant=lambda token: (_ for _ in ()).throw(
                Source7ReencodeMaterializationError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Source7ReencodeMaterializationError("source7 plan is not valid JSON") from error
    require(type(value) is dict, "source7 plan must be one object")
    require(raw == canonical_json_bytes(value) + b"\n", "source7 plan is not canonical JSON")
    try:
        plan_contract.validate_plan(value)
    except plan_contract.Source7ReencodePlanError as error:
        raise Source7ReencodeMaterializationError(str(error)) from error
    return value, path, observed


def _closed_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _tensor_raw_sha256(value: Any) -> str:
    import torch

    require(isinstance(value, torch.Tensor), "tensor hash input differs")
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(
        tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    ).hexdigest()


def _validate_tensor(value: Any, expected_shape: Sequence[int], *, label: str) -> Any:
    import torch

    require(isinstance(value, torch.Tensor), f"{label} must be a tensor")
    require(value.layout == torch.strided, f"{label} layout differs")
    require(value.device.type == "cpu", f"{label} must be on CPU")
    require(value.dtype == torch.float32, f"{label} must be FP32")
    require(value.is_contiguous(), f"{label} must be contiguous")
    require(tuple(int(item) for item in value.shape) == tuple(expected_shape), f"{label} shape differs")
    require(bool(torch.isfinite(value).all().item()), f"{label} contains non-finite values")
    return value


def _decode_bare_tensor(raw: bytes, expected_shape: Sequence[int], *, label: str) -> Any:
    import torch

    buffer = io.BytesIO(raw)
    try:
        value = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - pinned AUH torch accepts weights_only
        buffer.seek(0)
        value = torch.load(buffer, map_location="cpu")
    except Exception as error:
        raise Source7ReencodeMaterializationError(f"cannot reopen {label}") from error
    return _validate_tensor(value, expected_shape, label=label)


def _write_create_only(path: Path, raw: bytes) -> None:
    require(path.is_absolute() and path.parent.is_dir(), "posterior output parent differs")
    require(not path.exists() and not path.is_symlink() and bool(raw), "posterior output must be fresh")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_json_bytes(value) + b"\n"
    _write_create_only(path, raw)
    require(path.read_bytes() == raw, "receipt physical reopen differs")
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one same-parent directory without replacement.

    A preflight ``exists`` check cannot close a publication race.  Linux uses
    ``renameat2(RENAME_NOREPLACE)`` and Darwin uses
    ``renamex_np(RENAME_EXCL)``.  There is deliberately no check-then-rename
    fallback: a platform without one of these primitives fails closed.
    """

    require(
        source.is_absolute()
        and destination.is_absolute()
        and source.parent == destination.parent
        and source != destination,
        "atomic publication requires one same-parent source/destination pair",
    )
    try:
        source_before = os.lstat(source)
        parent_before = os.lstat(source.parent)
    except OSError as error:
        raise Source7ReencodeMaterializationError(
            "atomic publication source/parent is unavailable"
        ) from error
    require(
        stat.S_ISDIR(source_before.st_mode)
        and not stat.S_ISLNK(source_before.st_mode)
        and stat.S_ISDIR(parent_before.st_mode)
        and not stat.S_ISLNK(parent_before.st_mode),
        "atomic publication source/parent must be plain directories",
    )
    parent_descriptor = os.open(
        source.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        parent_opened = os.fstat(parent_descriptor)
        parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
        require(
            parent_identity == (parent_before.st_dev, parent_before.st_ino),
            "atomic publication parent changed while opening",
        )
        library = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            function = getattr(library, "renameat2", None)
            require(function is not None, "renameat2(RENAME_NOREPLACE) is unavailable")
            function.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            function.restype = ctypes.c_int
            result = function(
                parent_descriptor,
                os.fsencode(source.name),
                parent_descriptor,
                os.fsencode(destination.name),
                1,
            )
        elif sys.platform == "darwin":
            function = getattr(library, "renamex_np", None)
            require(function is not None, "renamex_np(RENAME_EXCL) is unavailable")
            function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
            function.restype = ctypes.c_int
            result = function(
                os.fsencode(str(source)),
                os.fsencode(str(destination)),
                0x00000004,
            )
        else:
            fail("atomic no-replace directory publication is unsupported")
        if result != 0:
            observed_errno = ctypes.get_errno()
            if observed_errno in (errno.EEXIST, errno.ENOTEMPTY):
                fail("atomic publication target already exists")
            raise Source7ReencodeMaterializationError(
                f"atomic no-replace publication failed: errno={observed_errno}"
            )
        os.fsync(parent_descriptor)
        try:
            source_after = os.stat(
                source.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            source_after = None
        destination_after = os.stat(
            destination.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        parent_after = os.fstat(parent_descriptor)
        require(
            source_after is None
            and stat.S_ISDIR(destination_after.st_mode)
            and not stat.S_ISLNK(destination_after.st_mode)
            and (destination_after.st_dev, destination_after.st_ino)
            == (source_before.st_dev, source_before.st_ino)
            and (parent_after.st_dev, parent_after.st_ino) == parent_identity,
            "atomic no-replace publication postcondition differs",
        )
    finally:
        os.close(parent_descriptor)


def _negative_access_closure() -> dict[str, Any]:
    value = {
        "source_only_reencode_from_source_video": True,
        "vae_encode_calls_per_source": 1,
        "paired_dataset_accessed": False,
        "legacy_source_target_container_opened": False,
        "synthetic_target_index1_path_read": False,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_index1_decoded": False,
        "synthetic_target_index1_filtered_on": False,
        "synthetic_target_index1_hashed": False,
        "target_video_path_present": False,
        "target_video_accessed": False,
    }
    require(set(value) == NEGATIVE_ACCESS_FIELDS, "negative-access field closure differs")
    return value


def _encode_one(row: Mapping[str, Any], *, encoder: Any, stage: Path, final_root: Path) -> Mapping[str, Any]:
    import torch

    iid = row["iid"]
    source_path = _plain_absolute_file(row["source_video_path"], label=f"{iid} source MP4")
    before_identity = _identity(source_path)
    require(0 < before_identity[2] <= MAX_SOURCE_BYTES, f"{iid} source MP4 size differs")
    before_sha256 = file_sha256(source_path)
    require(before_sha256 == row["source_video_sha256"], f"{iid} source MP4 SHA-256 differs before decode")
    frames, reported_fps, input_hw = base._decode_exact_video(source_path)
    after_decode_identity = _identity(source_path)
    after_sha256 = file_sha256(source_path)
    final_identity = _identity(source_path)
    require(
        before_identity == after_decode_identity == final_identity
        and after_sha256 == before_sha256,
        f"{iid} source MP4 changed during decode",
    )
    bucket_hw = base.source_aspect_bucket(*input_hw)
    expected_shape = tuple(row["expected_posterior_shape"])
    require(
        bucket_hw == (expected_shape[3] * 8, expected_shape[4] * 8),
        f"{iid} official source aspect bucket differs from expected posterior shape",
    )
    rgb = base._resize_video(frames, bucket_hw, None).clamp_(-1.0, 1.0).contiguous()
    require(
        rgb.dtype == torch.float32
        and rgb.is_contiguous()
        and tuple(int(item) for item in rgb.shape) == (3, 81, *bucket_hw)
        and bool(torch.isfinite(rgb).all().item()),
        f"{iid} source RGB tensor differs",
    )
    value = rgb.unsqueeze(0).to(encoder.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(encoder.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        posterior_parameters = encoder.model.encode(value).latent_dist.parameters
    posterior_parameters = posterior_parameters.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    _validate_tensor(posterior_parameters, expected_shape, label=f"{iid} posterior_parameters")
    blob = base.tensor_to_bytes(posterior_parameters)
    output_path = stage / row["output_filename"]
    _write_create_only(output_path, blob)
    reopened_raw = output_path.read_bytes()
    reopened = _decode_bare_tensor(
        reopened_raw, expected_shape, label=f"{iid} physically reopened posterior_parameters"
    )
    require(torch.equal(reopened, posterior_parameters), f"{iid} physical tensor replay differs")
    tensor_sha256 = base._tensor_sha256(reopened)
    tensor_raw_sha256 = _tensor_raw_sha256(reopened)
    file_digest = hashlib.sha256(reopened_raw).hexdigest()
    require(file_digest == file_sha256(output_path), f"{iid} physical file hash replay differs")
    receipt = {
        "schema_version": ROW_SCHEMA,
        "iid": iid,
        "analysis_split": row["analysis_split"],
        "event_id": row["event_id"],
        "actor_kind": row["actor_kind"],
        "q0_id": row["q0_id"],
        "group_id": row["group_id"],
        "actor_id": row["actor_id"],
        "scene_id": row["scene_id"],
        "source_video_path": str(source_path),
        "source_video_sha256": before_sha256,
        "source_video_stat_identity": list(before_identity),
        "source_video_sha256_before_decode": before_sha256,
        "source_video_sha256_after_decode": after_sha256,
        "source_video_pre_post_stat_and_hash_stable": True,
        "frame_count": 81,
        "expected_fps": 25.0,
        "reported_fps": reported_fps,
        "input_hw": list(input_hw),
        "source_aspect_bucket_hw": list(bucket_hw),
        "posterior_parameters_path": str(final_root / row["output_filename"]),
        "posterior_parameters_file_sha256": file_digest,
        "posterior_parameters_tensor_sha256": tensor_sha256,
        "posterior_parameters_tensor_raw_sha256": tensor_raw_sha256,
        "posterior_parameters_shape": list(expected_shape),
        "posterior_parameters_dtype": "torch.float32",
        "posterior_parameters_device": "cpu",
        "posterior_parameters_layout": "torch.strided",
        "posterior_parameters_contiguous": True,
        "posterior_parameters_finite": True,
        "posterior_parameters_bare_tensor": True,
        "posterior_sample_materialized": False,
        "physical_file_reopened_after_write": True,
        "physical_tensor_reopened_after_write": True,
        "physical_tensor_equal_to_encoded_tensor": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(encoder.device)),
        **_negative_access_closure(),
    }
    require(set(receipt) == ROW_RECEIPT_FIELDS, f"{iid} row receipt field closure differs")
    return receipt


def materialize(
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    checkpoint: str | Path,
    checkpoint_content_manifest: str | Path,
    expected_checkpoint_content_manifest_sha256: str,
    output_root: str | Path,
    device: str = "cuda:0",
) -> Mapping[str, Any]:
    require(device == "cuda:0", "only one explicit cuda:0 encoder is authorized")
    plan, resolved_plan, plan_sha256 = _load_plan(plan_path, expected_plan_sha256)
    checkpoint_path = Path(checkpoint)
    require(checkpoint_path.is_absolute() and checkpoint_path.is_dir() and not checkpoint_path.is_symlink(), "checkpoint root differs")
    checkpoint_path = checkpoint_path.resolve(strict=True)
    manifest_path = _plain_absolute_file(
        str(checkpoint_content_manifest), label="checkpoint content manifest"
    )
    output = Path(output_root)
    require(output.is_absolute(), "output root must be absolute")
    output = output.resolve(strict=False)
    require(not output.exists() and not output.is_symlink(), "output root must be create-only")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "output parent differs")
    stage = output.parent / f".{output.name}.source7-staging"
    require(not stage.exists() and not stage.is_symlink(), "source7 staging root must be fresh")
    stage.mkdir(mode=0o700)
    try:
        _bind_official_source_self_primitives()
        encoder = pinned.PinnedBerniniWanPosteriorEncoder(
            checkpoint_path,
            content_manifest=manifest_path,
            device=device,
            expected_manifest_sha256=expected_checkpoint_content_manifest_sha256,
        )
        rows = [
            _encode_one(row, encoder=encoder, stage=stage, final_root=output)
            for row in plan["rows"]
        ]
        require(len(rows) == 7 and len({row["iid"] for row in rows}) == 7, "materialized exact7 closure differs")
        external = dict(plan["external_existing_index0"])
        external.update(
            {
                "opened_by_materializer": False,
                "included_in_exact7_output_files": False,
                "reencoded": False,
            }
        )
        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD_NAME,
            "experiment_id": plan_contract.EXPERIMENT_ID,
            "complete": True,
            "plan": {
                "path": str(resolved_plan),
                "file_sha256": plan_sha256,
                "plan_digest": plan["plan_digest"],
            },
            "output_root": str(output),
            "row_count": 7,
            "rows": rows,
            "external_existing_index0": external,
            "vae_identity": dict(encoder.identity),
            "output_filenames": [row["output_filename"] for row in plan["rows"]],
            "output_exact_member_closure": True,
            "distinct_source_mp4_count": 7,
            "total_vae_encode_calls": 7,
            "posterior_sample_materialized": False,
            "external_existing_index0_opened": False,
            "external_existing_index0_reencoded": False,
            "inventory_snapshot_only": True,
            "exact8_authority_go_claimed": False,
            "teacher_cross_disjointness_pending": True,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "training_authorized": False,
            **_negative_access_closure(),
        }
        receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        require(set(receipt) == RECEIPT_FIELDS, "materialization receipt field closure differs")
        require(
            all(type(row) is dict and set(row) == ROW_RECEIPT_FIELDS for row in rows),
            "materialization row receipt field closure differs",
        )
        _write_json_create_only(stage / "materialization_receipt.json", receipt)
        require(
            {item.name for item in stage.iterdir()}
            == set(receipt["output_filenames"]) | {"materialization_receipt.json"},
            "staged output member closure differs",
        )
        _fsync_directory(stage)
        _rename_noreplace(stage, output)
        _fsync_directory(output.parent)
        return receipt
    finally:
        if stage.exists():
            try:
                stage.rmdir()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        checkpoint=args.checkpoint,
        checkpoint_content_manifest=args.checkpoint_content_manifest,
        expected_checkpoint_content_manifest_sha256=(
            args.expected_checkpoint_content_manifest_sha256
        ),
        output_root=args.output_root,
        device=args.device,
    )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
