#!/usr/bin/env python3
"""Materialize sealed v3 RV2V-4 source/variant-A/variant-B identity orbits.

For every IID this program decodes three independently hash-bound exact81
MP4s: one raw source and two frozen native variants.  Each variant explicitly
binds its actual R2V or RV2V arm, so RV2V+RV2V is a first-class orbit.  The
program then performs exactly fifteen pinned Wan-VAE calls: three full-video
posterior encodes and independent RGB-frame encodes at 0/27/53/80 per member.

No posterior reference is sliced from a full-video posterior.  No target,
mask, flow, pose, track, box, trajectory, training program, or locality route
is accepted.  The create-only parquet remains scientifically unauthorized
unless every IID carries a valid external full-video qualification seal.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import appearance_counterfactual_identity_orbit as contract  # noqa: E402
import source_self_runtime as runtime  # noqa: E402
from tools import materialize_ramp_motion_analogy_vae as pinned  # noqa: E402
from tools import materialize_vae as base  # noqa: E402


METHOD_NAME = "bernini-appearance-counterfactual-identity-orbit-materializer-v3"
ROW_SCHEMA = "bernini-appearance-counterfactual-identity-orbit-vae-row-v3"
RECEIPT_SCHEMA = (
    "bernini-appearance-counterfactual-identity-orbit-dataset-receipt-v3"
)
LATENT_PHASES = 21
POSTERIOR_CHANNELS = 32
INDEPENDENT_ENCODE_CALLS_PER_ROW = contract.INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")

POSTERIOR_FIELDS = tuple(
    [f"{member}_full_posterior_blob" for member in contract.MEMBER_NAMES]
    + [
        f"{member}_ref{index}_posterior_blob"
        for member in contract.MEMBER_NAMES
        for index in contract.REFERENCE_INDICES
    ]
)
POSTERIOR_ARTIFACT_ROLES = {
    **{
        f"{member}_full_posterior_blob": f"{member}_full_exact81_rgb"
        for member in contract.MEMBER_NAMES
    },
    **{
        f"{member}_ref{index}_posterior_blob": (
            f"{member}_independent_rgb_frame_{index}"
        )
        for member in contract.MEMBER_NAMES
        for index in contract.REFERENCE_INDICES
    },
}
if tuple(POSTERIOR_ARTIFACT_ROLES) != POSTERIOR_FIELDS:
    raise RuntimeError("posterior artifact-role registry differs from field order")


class OrbitMaterializationError(RuntimeError):
    """Raised before an incomplete or mutable dataset is published."""


def _blob_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode_posterior(
    encoder: Any,
    rgb: Any,
    *,
    phases: int,
    role: str,
    call_index: int,
) -> tuple[bytes, Mapping[str, Any]]:
    """Run one independent pinned VAE posterior-parameter encode."""

    import torch

    if (
        not isinstance(rgb, torch.Tensor)
        or rgb.dtype != torch.float32
        or rgb.ndim != 4
        or int(rgb.shape[0]) != 3
        or int(rgb.shape[1]) != (contract.FRAME_COUNT if phases == LATENT_PHASES else 1)
        or not rgb.is_contiguous()
        or not bool(torch.isfinite(rgb).all().item())
    ):
        raise OrbitMaterializationError(f"{role} RGB tensor geometry/dtype differs")
    value = rgb.unsqueeze(0).to(encoder.device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(encoder.device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        parameters = encoder.model.encode(value).latent_dist.parameters
    parameters = parameters.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if (
        parameters.ndim != 5
        or tuple(int(item) for item in parameters.shape[:3])
        != (1, POSTERIOR_CHANNELS, phases)
        or not bool(torch.isfinite(parameters).all().item())
    ):
        raise OrbitMaterializationError(
            f"{role} posterior parameters differ: {list(parameters.shape)}"
        )
    blob = base.tensor_to_bytes(parameters)
    metadata = {
        "artifact_role": role,
        "encode_call_index": call_index,
        "encoded_independently": True,
        "encoded_directly_from_rgb": True,
        "reference_from_full_video_posterior_slice": False,
        "posterior_parameters_shape": list(parameters.shape),
        "posterior_parameters_dtype": str(parameters.dtype),
        "posterior_parameters_tensor_sha256": base._tensor_sha256(parameters),
        "posterior_parameters_blob_sha256": _blob_sha256(blob),
        "posterior_sample_materialized": False,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(encoder.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(encoder.device)),
    }
    return blob, metadata


def _encoding_jobs(videos: Mapping[str, Any]) -> tuple[tuple[str, Any, int, str], ...]:
    """Return the registered three-full then twelve-independent-ref call order."""

    if tuple(videos) != contract.MEMBER_NAMES:
        raise OrbitMaterializationError(
            "RGB member order must be source/variant_a/variant_b"
        )
    jobs: list[tuple[str, Any, int, str]] = []
    for member in contract.MEMBER_NAMES:
        jobs.append(
            (
                f"{member}_full_posterior_blob",
                videos[member],
                LATENT_PHASES,
                f"{member}_full_exact81_rgb",
            )
        )
    for member in contract.MEMBER_NAMES:
        for index in contract.REFERENCE_INDICES:
            # This is deliberately RGB -> VAE.  Never replace this expression
            # with a slice of the already encoded full-video posterior.
            reference = videos[member][:, index : index + 1].contiguous()
            jobs.append(
                (
                    f"{member}_ref{index}_posterior_blob",
                    reference,
                    1,
                    f"{member}_independent_rgb_frame_{index}",
                )
            )
    if (
        tuple(job[0] for job in jobs) != POSTERIOR_FIELDS
        or tuple(job[3] for job in jobs)
        != tuple(POSTERIOR_ARTIFACT_ROLES[field] for field in POSTERIOR_FIELDS)
        or len(jobs) != INDEPENDENT_ENCODE_CALLS_PER_ROW
    ):
        raise OrbitMaterializationError("registered VAE encode job closure differs")
    return tuple(jobs)


def _encode_all_posteriors(
    videos: Mapping[str, Any],
    *,
    encoder: Any,
    encode_one: Callable[..., tuple[bytes, Mapping[str, Any]]] = _encode_posterior,
) -> tuple[dict[str, bytes], dict[str, Mapping[str, Any]]]:
    blobs: dict[str, bytes] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for call_index, (field, rgb, phases, role) in enumerate(_encoding_jobs(videos)):
        blob, details = encode_one(
            encoder,
            rgb,
            phases=phases,
            role=role,
            call_index=call_index,
        )
        if not isinstance(blob, bytes) or not blob or not isinstance(details, Mapping):
            raise OrbitMaterializationError(f"{role} VAE result is absent")
        details = dict(details)
        if (
            details.get("encode_call_index") != call_index
            or details.get("encoded_independently") is not True
            or details.get("reference_from_full_video_posterior_slice") is not False
        ):
            raise OrbitMaterializationError(f"{role} independent-encode receipt differs")
        blobs[field] = blob
        metadata[field] = details
    if (
        tuple(blobs) != POSTERIOR_FIELDS
        or tuple(metadata) != POSTERIOR_FIELDS
        or tuple(item["encode_call_index"] for item in metadata.values())
        != tuple(range(INDEPENDENT_ENCODE_CALLS_PER_ROW))
    ):
        raise OrbitMaterializationError("one orbit row must use exactly fifteen VAE calls")
    full_shapes = {
        tuple(metadata[f"{member}_full_posterior_blob"]["posterior_parameters_shape"])
        for member in contract.MEMBER_NAMES
    }
    reference_shapes = {
        tuple(
            metadata[f"{member}_ref{index}_posterior_blob"][
                "posterior_parameters_shape"
            ]
        )
        for member in contract.MEMBER_NAMES
        for index in contract.REFERENCE_INDICES
    }
    if len(full_shapes) != 1 or len(reference_shapes) != 1:
        raise OrbitMaterializationError("orbit member posterior geometries differ")
    full_shape = next(iter(full_shapes))
    reference_shape = next(iter(reference_shapes))
    if (
        full_shape[:2] != reference_shape[:2]
        or full_shape[3:] != reference_shape[3:]
        or full_shape[2] != LATENT_PHASES
        or reference_shape[2] != 1
    ):
        raise OrbitMaterializationError("full/reference posterior spatial geometry differs")
    return blobs, metadata


def _row_digest(row: Mapping[str, Any]) -> str:
    candidate = {
        key: value
        for key, value in row.items()
        if key not in set(POSTERIOR_FIELDS) | {"row_digest"}
    }
    candidate["posterior_blob_sha256"] = {
        field: _blob_sha256(bytes(row[field])) for field in POSTERIOR_FIELDS
    }
    return contract.object_sha256(candidate)


def _member_bindings(row: contract.OrbitSpecRow) -> dict[str, Mapping[str, Any]]:
    return {
        "source": {
            "video_path": str(row.source.path),
            "video_sha256": row.source.sha256,
            "native_receipt": None,
        },
        "variant_a": {
            "video_path": str(row.variant_a.video.path),
            "video_sha256": row.variant_a.video.sha256,
            "native_arm": row.variant_a.native_arm,
            "native_receipt": {
                "path": str(row.variant_a.receipt_path),
                "file_sha256": row.variant_a.receipt_file_sha256,
                "digest": row.variant_a.receipt_digest,
                "arm": row.variant_a.native_arm,
            },
        },
        "variant_b": {
            "video_path": str(row.variant_b.video.path),
            "video_sha256": row.variant_b.video.sha256,
            "native_arm": row.variant_b.native_arm,
            "native_receipt": {
                "path": str(row.variant_b.receipt_path),
                "file_sha256": row.variant_b.receipt_file_sha256,
                "digest": row.variant_b.receipt_digest,
                "arm": row.variant_b.native_arm,
            },
        },
    }


def _materialize_row(
    row: contract.OrbitSpecRow,
    *,
    encoder: Any,
    audit: contract.FileMutationAudit,
) -> dict[str, Any]:
    import torch

    members = {
        "source": row.source,
        "variant_a": row.variant_a.video,
        "variant_b": row.variant_b.video,
    }
    decoded: dict[str, Any] = {}
    reported_fps: dict[str, float] = {}
    input_hw: dict[str, list[int]] = {}
    for name in contract.MEMBER_NAMES:
        audit.assert_current(members[name].path)
        frames, fps, hw = base._decode_exact_video(members[name].path)
        audit.assert_current(members[name].path)
        decoded[name] = frames
        reported_fps[name] = float(fps)
        input_hw[name] = [int(hw[0]), int(hw[1])]

    bucket_hw = base.source_aspect_bucket(
        *input_hw["source"],
        max_pixels=base.DEFAULT_MAX_PIXELS,
        stride=base.DEFAULT_STRIDE,
    )
    for member_name, native_member in (
        ("variant_a", row.variant_a),
        ("variant_b", row.variant_b),
    ):
        native_arm = native_member.native_arm
        native_output = native_member.receipt["outputs"][native_arm]
        native_bucket = native_member.receipt["preprocessing"].get(
            "source_derived_bucket_hw"
        )
        if (
            native_bucket != list(bucket_hw)
            or input_hw[member_name] != list(bucket_hw)
            or native_output.get("height") != input_hw[member_name][0]
            or native_output.get("width") != input_hw[member_name][1]
        ):
            raise OrbitMaterializationError(
                f"{row.iid} {member_name}/{native_arm} decoded/native bucket differs"
            )

    videos = {
        name: base._resize_video(decoded[name], bucket_hw, None)
        .clamp_(-1.0, 1.0)
        .contiguous()
        for name in contract.MEMBER_NAMES
    }
    expected_shape = (3, contract.FRAME_COUNT, bucket_hw[0], bucket_hw[1])
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(int(item) for item in value.shape) != expected_shape
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
        for value in videos.values()
    ):
        raise OrbitMaterializationError(f"{row.iid} aligned RGB geometry differs")
    rgb_sha256 = {name: base._tensor_sha256(value) for name, value in videos.items()}
    if rgb_sha256["variant_a"] == rgb_sha256["variant_b"]:
        raise OrbitMaterializationError(
            f"{row.iid} variant_a and variant_b RGB content must be distinct"
        )
    rgb_members_distinct = len(set(rgb_sha256.values())) == 3
    if row.scientific_use_authorized and not rgb_members_distinct:
        raise OrbitMaterializationError(
            f"{row.iid} qualified orbit members collide after RGB preprocessing"
        )

    blobs, vae_metadata = _encode_all_posteriors(videos, encoder=encoder)
    full_shape = vae_metadata["source_full_posterior_blob"][
        "posterior_parameters_shape"
    ]
    qualification_binding: Mapping[str, Any]
    if row.qualification is None:
        qualification_binding = {
            "present": False,
            "path": None,
            "file_sha256": None,
            "digest": None,
            "scientific_use_authorized": False,
        }
    else:
        qualification_binding = {
            "present": True,
            "path": str(row.qualification.path),
            "file_sha256": row.qualification.file_sha256,
            "digest": row.qualification.digest,
            "scientific_use_authorized": True,
        }
    output: dict[str, Any] = {
        "schema_version": ROW_SCHEMA,
        "iid": row.iid,
        "source_video_path": str(row.source.path),
        "source_video_sha256": row.source.sha256,
        "variant_a_native_arm": row.variant_a.native_arm,
        "variant_a_video_path": str(row.variant_a.video.path),
        "variant_a_video_sha256": row.variant_a.video.sha256,
        "variant_a_native_receipt_path": str(row.variant_a.receipt_path),
        "variant_a_native_receipt_file_sha256": (
            row.variant_a.receipt_file_sha256
        ),
        "variant_a_native_receipt_digest": row.variant_a.receipt_digest,
        "variant_b_native_arm": row.variant_b.native_arm,
        "variant_b_video_path": str(row.variant_b.video.path),
        "variant_b_video_sha256": row.variant_b.video.sha256,
        "variant_b_native_receipt_path": str(row.variant_b.receipt_path),
        "variant_b_native_receipt_file_sha256": (
            row.variant_b.receipt_file_sha256
        ),
        "variant_b_native_receipt_digest": row.variant_b.receipt_digest,
        "qualification_seal_path": (
            str(row.qualification.path) if row.qualification is not None else None
        ),
        "qualification_seal_file_sha256": (
            row.qualification.file_sha256 if row.qualification is not None else None
        ),
        "qualification_seal_digest": (
            row.qualification.digest if row.qualification is not None else None
        ),
        "member_order_json": contract.canonical_json_bytes(
            list(contract.MEMBER_NAMES)
        ).decode("ascii"),
        "v4_member_aliases_json": contract.canonical_json_bytes(
            contract.V4_MEMBER_ALIASES
        ).decode("ascii"),
        "member_bindings_json": contract.canonical_json_bytes(
            _member_bindings(row)
        ).decode("ascii"),
        "appearance_intervention_generation_factors_json": contract.canonical_json_bytes(
            {
                "variant_a": {
                    "native_arm": row.variant_a.native_arm,
                    "action_prompt_utf8_sha256": row.variant_a.receipt["input"][
                        "action_prompt_utf8_sha256"
                    ],
                    "seed": row.variant_a.receipt["sampling"][
                        row.variant_a.native_arm
                    ]["seed"],
                },
                "variant_b": {
                    "native_arm": row.variant_b.native_arm,
                    "action_prompt_utf8_sha256": row.variant_b.receipt["input"][
                        "action_prompt_utf8_sha256"
                    ],
                    "seed": row.variant_b.receipt["sampling"][
                        row.variant_b.native_arm
                    ]["seed"],
                },
                "distinct_action_prompt_hashes_required": True,
                "distinct_variant_mp4_and_rgb_content_required": True,
                "same_seed_required": False,
                "semantic_identity_distinction_requires_external_qualification": True,
            }
        ).decode("ascii"),
        "qualification_binding_json": contract.canonical_json_bytes(
            qualification_binding
        ).decode("ascii"),
        "frame_count": contract.FRAME_COUNT,
        "fps": contract.FPS,
        "reported_fps_json": contract.canonical_json_bytes(reported_fps).decode(
            "ascii"
        ),
        "input_hw_json": contract.canonical_json_bytes(input_hw).decode("ascii"),
        "source_derived_bucket_hw_json": contract.canonical_json_bytes(
            list(bucket_hw)
        ).decode("ascii"),
        "rgb_tensor_sha256_json": contract.canonical_json_bytes(rgb_sha256).decode(
            "ascii"
        ),
        "full_posterior_parameters_shape_json": contract.canonical_json_bytes(
            full_shape
        ).decode("ascii"),
        "reference_indices_json": contract.canonical_json_bytes(
            list(contract.REFERENCE_INDICES)
        ).decode("ascii"),
        "reference_count": contract.REFERENCE_COUNT,
        "reference_encoding_contract_digest": contract.reference_encoding_contract()[
            "digest"
        ],
        "independent_vae_encode_metadata_json": contract.canonical_json_bytes(
            vae_metadata
        ).decode("ascii"),
        "independent_vae_encode_calls": INDEPENDENT_ENCODE_CALLS_PER_ROW,
        "three_full_videos_independently_encoded": True,
        "twelve_rgb_references_independently_encoded": True,
        "references_from_full_video_posterior_slice": False,
        "native_deployment_visual_conditioning": "one_video_plus_four_rgb_refs",
        "native_receipt_source_output_exactly_bound": True,
        "native_exact81_25fps_40steps_frozen": True,
        "rgb_member_tensors_content_distinct": rgb_members_distinct,
        "external_target_accepted": False,
        "mask_flow_pose_track_box_trajectory_used": False,
        "paired_action_target_present": False,
        "synthetic_action_target_present": False,
        "member_role": "appearance_identity_orbit_same_motion_candidate",
        "scientific_use_authorized": row.scientific_use_authorized,
        "direct_action_edit_claim_authorized": False,
        **blobs,
    }
    # Filled only after all rows, receipts, media and VAE files pass a final
    # post-encode mutation audit.
    output["input_file_mutation_audit_json"] = ""
    output["row_digest"] = ""
    return output


def _row_audit_records(
    iid: str, records: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    prefix = f"{iid}:"
    return [
        dict(record)
        for record in records
        if any(str(role).startswith(prefix) for role in record.get("roles", []))
    ]


def _plain_stage_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise OrbitMaterializationError(f"cannot inspect staged {label}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise OrbitMaterializationError(f"staged {label} is not a plain file")


def _verify_staged_bundle(
    stage: Path,
    *,
    receipt: Mapping[str, Any],
    dataset_sha256: str,
    expected_rows: int,
    expected_columns: Sequence[str],
) -> None:
    entries = list(stage.iterdir())
    if {path.name for path in entries} != {"dataset.parquet", "receipt.json"}:
        raise OrbitMaterializationError("staged dataset file closure differs")
    for path in entries:
        _plain_stage_file(path, label=path.name)
    if runtime.file_sha256(stage / "dataset.parquet") != dataset_sha256:
        raise OrbitMaterializationError("staged dataset hash differs")
    expected_receipt = contract.canonical_json_bytes(receipt) + b"\n"
    actual_receipt = (stage / "receipt.json").read_bytes()
    if actual_receipt != expected_receipt:
        raise OrbitMaterializationError("staged receipt bytes differ")
    unsigned = dict(receipt)
    declared = unsigned.pop("receipt_digest", None)
    if contract.object_sha256(unsigned) != declared:
        raise OrbitMaterializationError("staged receipt embedded digest differs")
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(stage / "dataset.parquet")
        if (
            parquet.metadata.num_rows != expected_rows
            or tuple(parquet.schema_arrow.names) != tuple(expected_columns)
        ):
            raise OrbitMaterializationError("staged parquet row/column closure differs")
    except OrbitMaterializationError:
        raise
    except Exception as error:
        raise OrbitMaterializationError(f"cannot verify staged parquet: {error}") from error


def _publish_create_only_bundle(stage: Path, output: Path) -> None:
    """Publish with an exclusive final-directory create and receipt-last commit.

    Renaming a directory is atomic but can replace a concurrently created empty
    directory on POSIX.  Here ``mkdir(exist_ok=False)`` is the no-replace
    primitive.  The parquet is hard-linked first and the receipt last; readers
    must treat the receipt as the bundle commit marker.  A crash can therefore
    leave an obviously incomplete, never-overwritten directory, but cannot
    turn a pre-existing path into this dataset.
    """

    expected_stage = output.parent / f".{output.name}.staging"
    if stage != expected_stage or stage.parent != output.parent:
        raise OrbitMaterializationError("stage/output transaction paths differ")
    if output.exists() or output.is_symlink():
        raise OrbitMaterializationError("create-only output appeared before publication")
    entries = {path.name: path for path in stage.iterdir()}
    if set(entries) != {"dataset.parquet", "receipt.json"}:
        raise OrbitMaterializationError("publish stage file closure differs")
    for name, path in entries.items():
        _plain_stage_file(path, label=name)

    try:
        output.mkdir(mode=0o750, exist_ok=False)
    except FileExistsError as error:
        raise OrbitMaterializationError("create-only output already exists") from error
    runtime.fsync_directory(output.parent)
    try:
        os.link(stage / "dataset.parquet", output / "dataset.parquet")
        # Receipt is deliberately the last link and therefore the commit marker.
        os.link(stage / "receipt.json", output / "receipt.json")
        runtime.fsync_directory(output)
        runtime.fsync_directory(output.parent)
        published = list(output.iterdir())
        if {path.name for path in published} != {"dataset.parquet", "receipt.json"}:
            raise OrbitMaterializationError("published dataset file closure differs")
        for path in published:
            _plain_stage_file(path, label=f"published {path.name}")
    except Exception:
        # Never remove a final path after its exclusive creation: another
        # process could have observed or augmented the incomplete directory.
        # Absence of receipt.json remains the fail-closed recovery signal.
        raise

    # These are the two exact staging hard links created by this transaction;
    # unlinking them leaves the sealed final inodes intact.
    (stage / "dataset.parquet").unlink()
    (stage / "receipt.json").unlink()
    stage.rmdir()
    runtime.fsync_directory(output.parent)


def _resolve_output(value: str | Path) -> tuple[Path, Path]:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or requested.suffix
        or _SAFE_OUTPUT_NAME.fullmatch(requested.name) is None
    ):
        raise OrbitMaterializationError(
            "output must be an absolute safe suffix-free directory"
        )
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise OrbitMaterializationError(f"output parent is unavailable: {error}") from error
    if parent.is_symlink() or not parent.is_dir() or requested != parent / requested.name:
        raise OrbitMaterializationError("output path/parent is not canonical")
    stage = parent / f".{requested.name}.staging"
    if (
        requested.exists()
        or requested.is_symlink()
        or stage.exists()
        or stage.is_symlink()
    ):
        raise OrbitMaterializationError("output and hidden stage are create-only")
    return requested, stage


def _register_pinned_vae_inputs(
    *,
    checkpoint: Path,
    content_manifest: Path,
    expected_manifest_sha256: str,
    expected_vae_config_sha256: str,
    audit: contract.FileMutationAudit,
) -> Mapping[str, Any]:
    manifest_path = audit.register(
        str(content_manifest),
        expected_sha256=expected_manifest_sha256,
        role="pinned_vae:checkpoint_content_manifest",
    )
    try:
        identity = pinned.validate_pinned_vae_checkpoint(
            checkpoint,
            manifest_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_vae_config_sha256=expected_vae_config_sha256,
        )
    except pinned.RampVaeMaterializationError as error:
        raise OrbitMaterializationError(str(error)) from error
    for relative, digest in identity["vae_files"].items():
        audit.register(
            str(Path(identity["checkpoint_root"]) / relative),
            expected_sha256=digest,
            role=f"pinned_vae:{relative}",
        )
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-vae-config-sha256", default=pinned.EXPECTED_VAE_CONFIG_SHA256
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output, stage = _resolve_output(args.output)
    if re.fullmatch(r"cuda:[0-9]+", args.device) is None:
        raise OrbitMaterializationError("device must be one explicit cuda:N device")
    if (
        args.expected_checkpoint_content_manifest_sha256
        != pinned.EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or args.expected_vae_config_sha256 != pinned.EXPECTED_VAE_CONFIG_SHA256
    ):
        raise OrbitMaterializationError("only the pinned Bernini/Wan VAE is supported")

    audit = contract.FileMutationAudit()
    try:
        spec = contract.load_materialization_spec(
            args.spec,
            expected_sha256=args.expected_spec_sha256,
            audit=audit,
        )
        checkpoint = args.checkpoint.expanduser()
        pinned_identity = _register_pinned_vae_inputs(
            checkpoint=checkpoint,
            content_manifest=args.checkpoint_content_manifest,
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
            expected_vae_config_sha256=args.expected_vae_config_sha256,
            audit=audit,
        )
        encoder = pinned.PinnedBerniniWanPosteriorEncoder(
            checkpoint,
            content_manifest=args.checkpoint_content_manifest,
            device=args.device,
            expected_manifest_sha256=args.expected_checkpoint_content_manifest_sha256,
            expected_vae_config_sha256=args.expected_vae_config_sha256,
        )
        if dict(encoder.identity) != dict(pinned_identity):
            raise OrbitMaterializationError("loaded pinned VAE identity differs")
        if any(parameter.requires_grad for parameter in encoder.model.parameters()):
            raise OrbitMaterializationError("pinned VAE unexpectedly has trainable parameters")

        rows = [
            _materialize_row(row, encoder=encoder, audit=audit) for row in spec.rows
        ]
        mutation_records = audit.finalize()
        mutation_receipt = {
            "files": list(mutation_records),
            "file_count": len(mutation_records),
            "all_files_pre_post_stat_and_hash_stable": all(
                record["pre_post_stat_and_hash_stable"] for record in mutation_records
            ),
        }
        mutation_receipt["digest"] = contract.object_sha256(mutation_receipt)
        for row in rows:
            row_records = _row_audit_records(row["iid"], mutation_records)
            if not row_records or not all(
                record["pre_post_stat_and_hash_stable"] for record in row_records
            ):
                raise OrbitMaterializationError(
                    f"{row['iid']} per-file mutation audit is incomplete"
                )
            row["input_file_mutation_audit_json"] = contract.canonical_json_bytes(
                {
                    "files": row_records,
                    "all_files_pre_post_stat_and_hash_stable": True,
                    "digest": contract.object_sha256(row_records),
                }
            ).decode("ascii")
            row["row_digest"] = _row_digest(row)

        if output.exists() or output.is_symlink() or stage.exists() or stage.is_symlink():
            raise OrbitMaterializationError("output/stage appeared during materialization")
        stage.mkdir(mode=0o750)
        runtime.fsync_directory(stage.parent)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            dataset_path = stage / "dataset.parquet"
            pq.write_table(table, dataset_path, compression="zstd")
        except Exception as error:
            raise OrbitMaterializationError(f"cannot write dataset parquet: {error}") from error
        with dataset_path.open("rb") as handle:
            os.fsync(handle.fileno())
        dataset_sha = runtime.file_sha256(dataset_path)
        qualified_iids = [
            row.iid for row in spec.rows if row.scientific_use_authorized
        ]
        missing_qualification_iids = [
            row.iid for row in spec.rows if not row.scientific_use_authorized
        ]
        scientific_use_authorized = not missing_qualification_iids
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "complete": True,
            "create_only": True,
            "spec": {
                "path": str(spec.path),
                "file_sha256": spec.file_sha256,
                "digest": spec.digest,
                "reference_encoding_contract_digest": (
                    spec.reference_encoding_contract["digest"]
                ),
            },
            "dataset": {
                "path": str(output / "dataset.parquet"),
                "sha256": dataset_sha,
                "rows": len(rows),
                "columns": list(table.column_names),
                "iids": [row["iid"] for row in rows],
                "row_digests": [row["row_digest"] for row in rows],
            },
            "pinned_vae_identity": dict(encoder.identity),
            "encoding_contract": {
                "member_order": list(contract.MEMBER_NAMES),
                "v4_member_aliases": dict(contract.V4_MEMBER_ALIASES),
                "reference_count": contract.REFERENCE_COUNT,
                "reference_rgb_indices": list(contract.REFERENCE_INDICES),
                "reference_encoding_contract_digest": (
                    contract.reference_encoding_contract()["digest"]
                ),
                "full_video_encode_calls_per_row": (
                    contract.FULL_VIDEO_ENCODE_CALLS_PER_ROW
                ),
                "independent_rgb_reference_encode_calls_per_row": (
                    contract.INDEPENDENT_RGB_REFERENCE_ENCODE_CALLS_PER_ROW
                ),
                "independent_vae_encode_calls_per_row": (
                    contract.INDEPENDENT_VAE_ENCODE_CALLS_PER_ROW
                ),
                "all_calls_share_one_pinned_vae_identity": True,
                "posterior_representation": "latent_dist.parameters_fp32",
                "posterior_sample_materialized": False,
                "references_from_full_video_posterior_slice": False,
                "native_deployment_visual_conditioning": (
                    "one_video_plus_four_rgb_refs"
                ),
            },
            "native_input_contract": {
                "receipt_schema": contract.NATIVE_RECEIPT_SCHEMA,
                "allowed_native_arms": list(contract.ALLOWED_NATIVE_ARMS),
                "member_native_arms_by_iid": {
                    row.iid: {
                        "variant_a": row.variant_a.native_arm,
                        "variant_b": row.variant_b.native_arm,
                    }
                    for row in spec.rows
                },
                "frame_count": contract.FRAME_COUNT,
                "fps": contract.FPS,
                "num_inference_steps": contract.NUM_INFERENCE_STEPS,
                "base_frozen": True,
                "training_performed": False,
                "external_target_accepted": False,
                "mask_flow_pose_track_box_trajectory_used": False,
                "variant_prompt_hashes_distinct_required": True,
                "variant_mp4_and_rgb_content_distinct_required": True,
                "variant_same_seed_required": False,
            },
            "input_file_mutation_audit": mutation_receipt,
            "qualification": {
                "external_schema": contract.QUALIFICATION_SCHEMA,
                "qualified_iids": qualified_iids,
                "missing_qualification_iids": missing_qualification_iids,
                "all_rows_externally_qualified": scientific_use_authorized,
                "absence_defaults_to_unauthorized": True,
                "semantic_identity_distinction_attested": scientific_use_authorized,
                "same_motion_camera_scene_attested": scientific_use_authorized,
            },
            "appearance_orbit_pretext_only": True,
            "paired_action_target_present": False,
            "synthetic_action_target_present": False,
            "direct_action_edit_supervision_present": False,
            "mask_flow_pose_track_box_trajectory_used": False,
            "scientific_use_authorized": scientific_use_authorized,
            "direct_action_edit_claim_authorized": False,
        }
        receipt["receipt_digest"] = contract.object_sha256(receipt)
        runtime.atomic_json(stage / "receipt.json", receipt)
        with (stage / "receipt.json").open("rb") as handle:
            os.fsync(handle.fileno())
        _verify_staged_bundle(
            stage,
            receipt=receipt,
            dataset_sha256=dataset_sha,
            expected_rows=len(rows),
            expected_columns=table.column_names,
        )
        runtime.fsync_directory(stage)
        _publish_create_only_bundle(stage, output)
        return 0
    except contract.AppearanceOrbitError as error:
        raise OrbitMaterializationError(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INDEPENDENT_ENCODE_CALLS_PER_ROW",
    "LATENT_PHASES",
    "METHOD_NAME",
    "OrbitMaterializationError",
    "POSTERIOR_ARTIFACT_ROLES",
    "POSTERIOR_FIELDS",
    "RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "_encode_all_posteriors",
    "_encoding_jobs",
    "_materialize_row",
    "_publish_create_only_bundle",
    "_resolve_output",
    "_row_digest",
    "_verify_staged_bundle",
    "build_parser",
    "main",
]
