#!/usr/bin/env python3
"""Materialize a source-self role/repaint pretext cohort.

Each input row binds only one raw exact-81 source MP4.  The pinned Wan VAE is
called independently for the clean source target, two temporally shared RGB
style donors, and RGB frames 0/40/80: exactly six calls per row.  No paired
dataset, prior posterior, or edited target is opened.

The output is a create-only parquet plus a sealed receipt.  It is pretext data,
contains no edited/action target, and makes no semantic-motion claim.
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
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_self_role_repaint as core  # noqa: E402
from tools import materialize_ramp_motion_analogy_vae as pinned  # noqa: E402
from tools import materialize_vae as base  # noqa: E402


METHOD_NAME = "bernini-source-self-role-repaint-materializer-v2"
SPEC_SCHEMA = "bernini-source-self-role-repaint-materialization-spec-v2"
ROW_SCHEMA = "bernini-source-self-role-repaint-row-v2"
RECEIPT_SCHEMA = "bernini-source-self-role-repaint-dataset-receipt-v2"
FRAME_COUNT = 81
FPS = 25.0
LATENT_PHASES = 21
REFERENCE_PHASES = 1
MIN_ROWS = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

POSTERIOR_FIELDS = (
    "clean_target_posterior_blob",
    "style1_donor_posterior_blob",
    "style2_donor_posterior_blob",
    "ref0_posterior_blob",
    "ref40_posterior_blob",
    "ref80_posterior_blob",
)
SPEC_KEYS = frozenset({"schema_version", "rows", "spec_digest"})
SPEC_ROW_KEYS = frozenset(
    {
        "iid",
        "source_video_path",
        "source_video_sha256",
    }
)


class SourceSelfMaterializationError(RuntimeError):
    """Raised before a weakly bound dataset artifact can be published."""


def canonical_json_bytes(value: Any) -> bytes:
    return core.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return core.object_sha256(value)


def file_sha256(path: Path) -> str:
    return pinned.file_sha256(path)


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SourceSelfMaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise SourceSelfMaterializationError(f"{label} path must be non-empty text")
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise SourceSelfMaterializationError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise SourceSelfMaterializationError(f"{label} is unavailable: {error}") from error
    if resolved != requested or not stat.S_ISREG(mode) or resolved.is_symlink():
        raise SourceSelfMaterializationError(f"{label} must be a canonical plain file")
    return resolved


def _reject_constant(value: str) -> None:
    raise SourceSelfMaterializationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceSelfMaterializationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_spec(path_value: str | Path, expected_sha256: str) -> tuple[Path, dict[str, Any], str]:
    path = _plain_absolute_file(str(path_value), label="materialization spec")
    actual = file_sha256(path)
    if actual != _sha(expected_sha256, label="materialization spec SHA-256"):
        raise SourceSelfMaterializationError("materialization spec SHA-256 differs")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != actual:
        raise SourceSelfMaterializationError("materialization spec changed while reading")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SourceSelfMaterializationError(f"cannot parse materialization spec: {error}") from error
    if (
        not isinstance(value, dict)
        or set(value) != SPEC_KEYS
        or value.get("schema_version") != SPEC_SCHEMA
    ):
        raise SourceSelfMaterializationError("materialization spec schema differs")
    candidate = dict(value)
    declared = candidate.pop("spec_digest", None)
    if object_sha256(candidate) != declared:
        raise SourceSelfMaterializationError("materialization spec digest differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) < MIN_ROWS:
        raise SourceSelfMaterializationError("spec requires at least two source rows")
    if any(not isinstance(row, dict) or set(row) != SPEC_ROW_KEYS for row in rows):
        raise SourceSelfMaterializationError("materialization spec row field closure differs")
    return path, value, actual


def build_materialization_spec(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the canonical in-memory spec that an external writer may seal.

    Paths and SHA values are still revalidated against live bytes by the
    materializer.  This helper only removes ambiguity about field closure and
    digest construction; it does not write or authorize an artifact.
    """

    copied = [dict(row) for row in rows]
    if len(copied) < MIN_ROWS or any(set(row) != SPEC_ROW_KEYS for row in copied):
        raise SourceSelfMaterializationError("spec builder requires two or more closed rows")
    iids = [row.get("iid") for row in copied]
    if any(type(iid) is not str or _IID.fullmatch(iid) is None for iid in iids):
        raise SourceSelfMaterializationError("spec builder row IID is invalid")
    if len(set(iids)) != len(iids):
        raise SourceSelfMaterializationError("spec builder IIDs must be unique")
    value: dict[str, Any] = {"schema_version": SPEC_SCHEMA, "rows": copied}
    value["spec_digest"] = object_sha256(value)
    return value


def _blob_sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _style_transform(video: Any, *, style: int) -> tuple[Any, Mapping[str, Any]]:
    import torch

    if (
        not isinstance(video, torch.Tensor)
        or video.dtype != torch.float32
        or tuple(int(item) for item in video.shape[:2]) != (3, FRAME_COUNT)
        or not video.is_contiguous()
    ):
        raise SourceSelfMaterializationError("style input must be FP32 [3,81,H,W]")
    if style == 1:
        matrix_values = ((0.15, 0.75, 0.10), (0.70, 0.10, 0.20), (0.10, 0.20, 0.70))
        bias_values = (0.16, -0.10, 0.08)
    elif style == 2:
        matrix_values = ((0.65, 0.25, 0.10), (0.10, 0.20, 0.70), (0.30, 0.60, 0.10))
        bias_values = (-0.12, 0.14, -0.04)
    else:
        raise SourceSelfMaterializationError("only registered styles 1 and 2 are supported")
    matrix = torch.tensor(matrix_values, dtype=torch.float32, device=video.device)
    bias = torch.tensor(bias_values, dtype=torch.float32, device=video.device).view(3, 1, 1, 1)
    transformed = torch.einsum("oc,cthw->othw", matrix, video).add_(bias).clamp_(-1.0, 1.0).contiguous()
    if torch.equal(transformed, video) or not bool(torch.isfinite(transformed).all().item()):
        raise SourceSelfMaterializationError("registered style transform is ineffective")
    receipt = {
        "style_id": f"registered_rgb_channel_affine_v1_style{style}",
        "matrix": [list(item) for item in matrix_values],
        "bias": list(bias_values),
        "applied_identically_to_all_81_frames": True,
        "temporal_order_changed": False,
        "spatial_coordinates_changed": False,
        "post_transform_clamp": [-1.0, 1.0],
        "semantic_motion_preservation_claimed": False,
    }
    return transformed, {**receipt, "digest": object_sha256(receipt)}


def _encode_posterior(encoder: Any, video: Any, *, phases: int, role: str, call_index: int) -> tuple[bytes, Mapping[str, Any]]:
    import torch

    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise SourceSelfMaterializationError(f"{role} RGB tensor must be [3,T,H,W]")
    value = video.unsqueeze(0).to(encoder.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(encoder.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        parameters = encoder.model.encode(value).latent_dist.parameters
    parameters = parameters.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(int(item) for item in parameters.shape[:3]) != (1, 32, phases):
        raise SourceSelfMaterializationError(
            f"{role} posterior shape differs: {list(parameters.shape)}"
        )
    blob = base.tensor_to_bytes(parameters)
    metadata = {
        "artifact_role": role,
        "encode_call_index": call_index,
        "encoded_independently": True,
        "posterior_from_video_latent_slice": False,
        "posterior_shape": list(parameters.shape),
        "posterior_dtype": str(parameters.dtype),
        "posterior_tensor_sha256": base._tensor_sha256(parameters),
        "posterior_blob_sha256": _blob_sha(blob),
        "posterior_sample_materialized": False,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(encoder.device)),
    }
    return blob, metadata


def _reference_order(iid: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(("source-self-ref-order\0" + iid).encode("utf-8")).digest()
    values = list(core.REFERENCE_RGB_INDICES)
    # Deterministic Fisher-Yates with digest bytes; independent of Python hash seed.
    for index in range(len(values) - 1, 0, -1):
        other = digest[index] % (index + 1)
        values[index], values[other] = values[other], values[index]
    return tuple(values)


def _row_digest(row: Mapping[str, Any]) -> str:
    candidate = {key: value for key, value in row.items() if key not in POSTERIOR_FIELDS and key != "row_digest"}
    candidate["posterior_blob_sha256"] = {
        field: _blob_sha(bytes(row[field])) for field in POSTERIOR_FIELDS
    }
    return object_sha256(candidate)


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _materialize_row(row: Mapping[str, Any], *, encoder: Any) -> dict[str, Any]:
    iid = row.get("iid")
    if type(iid) is not str or _IID.fullmatch(iid) is None:
        raise SourceSelfMaterializationError("row IID is invalid")
    source_path = _plain_absolute_file(row.get("source_video_path"), label=f"{iid} source video")
    source_sha = _sha(row.get("source_video_sha256"), label=f"{iid} source video SHA")
    source_identity_before = _stat_identity(source_path)
    source_sha_before = file_sha256(source_path)
    source_identity_after_hash = _stat_identity(source_path)
    if source_identity_before != source_identity_after_hash:
        raise SourceSelfMaterializationError(f"{iid} source changed while hashing")
    if source_sha_before != source_sha:
        raise SourceSelfMaterializationError(f"{iid} source video SHA differs")
    frames, reported_fps, input_hw = base._decode_exact_video(source_path)
    source_identity_after_decode = _stat_identity(source_path)
    source_sha_after = file_sha256(source_path)
    source_identity_final = _stat_identity(source_path)
    if (
        source_identity_before != source_identity_after_decode
        or source_identity_before != source_identity_final
        or source_sha_after != source_sha
    ):
        raise SourceSelfMaterializationError(
            f"{iid} source changed while decoding"
        )
    bucket_hw = base.source_aspect_bucket(*input_hw)
    rgb = base._resize_video(frames, bucket_hw, None).clamp_(-1.0, 1.0).contiguous()
    style1, style1_receipt = _style_transform(rgb, style=1)
    style2, style2_receipt = _style_transform(rgb, style=2)
    encoded: dict[str, bytes] = {}
    metadata: dict[str, Any] = {}
    call_index = 0
    for field, tensor, phases, role in (
        ("clean_target_posterior_blob", rgb, LATENT_PHASES, "clean_raw_source_reconstruction_target"),
        ("style1_donor_posterior_blob", style1, LATENT_PHASES, "appearance_corrupted_ordered_donor_style1"),
        ("style2_donor_posterior_blob", style2, LATENT_PHASES, "appearance_corrupted_ordered_donor_style2"),
        ("ref0_posterior_blob", rgb[:, 0:1], REFERENCE_PHASES, "independent_clean_source_rgb_frame_0"),
        ("ref40_posterior_blob", rgb[:, 40:41], REFERENCE_PHASES, "independent_clean_source_rgb_frame_40"),
        ("ref80_posterior_blob", rgb[:, 80:81], REFERENCE_PHASES, "independent_clean_source_rgb_frame_80"),
    ):
        blob, details = _encode_posterior(
            encoder, tensor.contiguous(), phases=phases, role=role, call_index=call_index
        )
        encoded[field] = blob
        metadata[field] = details
        call_index += 1
    if call_index != 6 or tuple(sorted(item["encode_call_index"] for item in metadata.values())) != tuple(range(6)):
        raise SourceSelfMaterializationError(f"{iid} must use exactly six independent VAE calls")
    clean_shape = tuple(metadata["clean_target_posterior_blob"]["posterior_shape"])
    if any(
        tuple(metadata[field]["posterior_shape"]) != clean_shape
        for field in ("style1_donor_posterior_blob", "style2_donor_posterior_blob")
    ):
        raise SourceSelfMaterializationError(f"{iid} clean/style posterior geometry differs")
    if any(
        tuple(metadata[f"ref{index}_posterior_blob"]["posterior_shape"][:2])
        != tuple(clean_shape[:2])
        or tuple(metadata[f"ref{index}_posterior_blob"]["posterior_shape"][3:])
        != tuple(clean_shape[3:])
        for index in core.REFERENCE_RGB_INDICES
    ):
        raise SourceSelfMaterializationError(f"{iid} clean/reference posterior geometry differs")
    if len({_blob_sha(value) for key, value in encoded.items() if key.startswith("ref")}) != 3:
        raise SourceSelfMaterializationError(f"{iid} independently encoded refs collide")
    output: dict[str, Any] = {
        "schema_version": ROW_SCHEMA,
        "iid": iid,
        "source_video_path": str(source_path),
        "source_video_sha256": source_sha,
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "reported_fps": reported_fps,
        "input_hw_json": canonical_json_bytes(list(input_hw)).decode("ascii"),
        "bucket_hw_json": canonical_json_bytes(list(bucket_hw)).decode("ascii"),
        "clean_posterior_shape_json": canonical_json_bytes(list(clean_shape)).decode("ascii"),
        "reference_indices_json": canonical_json_bytes(list(core.REFERENCE_RGB_INDICES)).decode("ascii"),
        "reference_order_json": canonical_json_bytes(list(_reference_order(iid))).decode("ascii"),
        "raw_source_stability_json": canonical_json_bytes(
            {
                "stat_identity": list(source_identity_before),
                "sha256_before_decode": source_sha_before,
                "sha256_after_decode": source_sha_after,
                "pre_post_stat_and_hash_stable": True,
            }
        ).decode("ascii"),
        "style_receipts_json": canonical_json_bytes([style1_receipt, style2_receipt]).decode("ascii"),
        "independent_encode_metadata_json": canonical_json_bytes(metadata).decode("ascii"),
        "independent_vae_encode_calls": 6,
        "clean_target_encoded_from_same_raw_source_rgb": True,
        "clean_style_refs_share_one_pinned_vae_identity": True,
        "paired_dataset_accessed": False,
        "prior_posterior_accessed": False,
        "edited_target_accessed": False,
        "reference_from_video_posterior_slice": False,
        "action_label_present": False,
        "semantic_motion_preservation_claimed": False,
        **encoded,
    }
    output["row_digest"] = _row_digest(output)
    return output


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path, spec, spec_sha = _load_spec(args.spec, args.expected_spec_sha256)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    content_manifest = _plain_absolute_file(
        args.checkpoint_content_manifest, label="checkpoint content manifest"
    )
    encoder = pinned.PinnedBerniniWanPosteriorEncoder(
        checkpoint,
        content_manifest=content_manifest,
        device=args.device,
        expected_manifest_sha256=_sha(
            args.expected_checkpoint_content_manifest_sha256,
            label="checkpoint content manifest SHA-256",
        ),
    )
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise SourceSelfMaterializationError("output must be absolute")
    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SourceSelfMaterializationError("output is create-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.staging"
    if stage.exists() or stage.is_symlink():
        raise SourceSelfMaterializationError("hidden staging path already exists")
    stage.mkdir(mode=0o750)
    try:
        rows = [_materialize_row(row, encoder=encoder) for row in spec["rows"]]
        if len({row["iid"] for row in rows}) != len(rows):
            raise SourceSelfMaterializationError("materialized IIDs are not unique")
        shapes = {row["clean_posterior_shape_json"] for row in rows}
        if len(shapes) != 1:
            raise SourceSelfMaterializationError(
                "one materialized cohort must use one bucket for clean wrong-ref controls"
            )
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            dataset_path = stage / "dataset.parquet"
            pq.write_table(table, dataset_path, compression="zstd")
        except Exception as error:
            raise SourceSelfMaterializationError(f"cannot write dataset parquet: {error}") from error
        _fsync(dataset_path)
        dataset_sha = file_sha256(dataset_path)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "spec": {"path": str(spec_path), "sha256": spec_sha, "digest": spec["spec_digest"]},
            "dataset": {
                "path": str(output / "dataset.parquet"),
                "sha256": dataset_sha,
                "rows": len(rows),
                "iids": [row["iid"] for row in rows],
                "row_digests": [row["row_digest"] for row in rows],
                "single_bucket": json.loads(rows[0]["bucket_hw_json"]),
            },
            "vae_identity": dict(encoder.identity),
            "roles": {
                "clean_target": "independent_pinned_vae_encode_of_raw_clean_source_rgb",
                "ordered_donors": ["registered_rgb_style1", "registered_rgb_style2"],
                "source_refs": list(core.REFERENCE_RGB_INDICES),
                "reference_order": "iid_hash_preregistered_permutation",
            },
            "independent_vae_encode_calls_per_row": 6,
            "all_six_calls_share_one_pinned_vae_identity": True,
            "paired_dataset_accessed": False,
            "prior_posterior_accessed": False,
            "target_video_path_present": False,
            "target_video_accessed": False,
            "references_independently_encoded_from_rgb": True,
            "references_from_video_posterior_slice": False,
            "edited_target_accessed": False,
            "synthetic_edited_target_present": False,
            "action_supervision_present": False,
            "mask_flow_pose_track_present": False,
            "style_applied_temporally_shared_before_donor_vae_encode": True,
            "temporal_order_changed_by_style": False,
            "semantic_motion_preservation_claimed": False,
            "preview_pretext_only": True,
            "scientific_claim_authorized": False,
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        _atomic_json(stage / "receipt.json", receipt)
        _fsync(stage / "receipt.json")
        if set(path.name for path in stage.iterdir()) != {"dataset.parquet", "receipt.json"}:
            raise SourceSelfMaterializationError("staged dataset closure differs")
        _fsync_directory(stage)
        os.replace(stage, output)
        _fsync_directory(output.parent)
    finally:
        if stage.exists():
            try:
                stage.rmdir()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
