#!/usr/bin/env python3
"""Pure fail-closed contract for the first decoded GRAFT Phase-A diagnostic.

The live runner that consumes this module must replay every parameter-changing
Phase-A operation in one process.  A Slurm ``afterok`` edge is a queue gate,
not a weight transport.  The upstream short, field14 and active14 jobs do not
publish a checkpoint and this contract consequently rejects any checkpoint
lineage or load/save claim.

The only positive claim emitted here is operational: two preregistered WORLD4
arms completed an exact40 native source-conditioned rollout, produced an
exact81/25fps decoded artifact, agreed byte-for-byte within each SP4 group,
and left frozen/base and in-memory trainable bytes unchanged during decode.
No visual evaluator is present, so identity, action and quality authority stay
false even when every operational gate passes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-decoded-core-v1"
TRACE_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-trace-v1"
LOCAL_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-local-result-v1"
WORLD8_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-world8-result-v1"
MEDIA_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-media-v1"
ARTIFACT_SCHEMA_VERSION = "bernini-graft-phase-a-full-exact81-artifact-v1"

FRAME_COUNT = 81
LATENT_PHASES = 21
FPS_NUMERATOR = 25
FPS_DENOMINATOR = 1
NUM_INFERENCE_STEPS = 40
WORLD_SIZE = 8
SP_SIZE = 4
DP_SIZE = 2
FAMILY_ORDER = ("dog", "human")

AUTHORITY_FIELDS = (
    "action_authority",
    "identity_authority",
    "cross_clip_identity_authority",
    "quality_authority",
    "training_authority",
    "checkpoint_authority",
    "publication_authority",
    "production_authority",
    "data_governance_authority",
    "data_license_authority",
    "scientific_success_claimed",
    "semantic_action_editing_success_claimed",
)


class FullExact81ContractError(RuntimeError):
    """Reject an ambiguous decoded result before it becomes authoritative."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise FullExact81ContractError("canonical mapping key is not a string")
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise FullExact81ContractError("value contains a non-JSON runtime object")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise FullExact81ContractError(
            "value is not finite canonical ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seal_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    plain = dict(value)
    if "digest" in plain:
        raise FullExact81ContractError("mapping is already sealed")
    plain["digest"] = object_sha256(plain)
    return MappingProxyType(plain)


def validate_sealed_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullExact81ContractError(f"{label} must be a mapping")
    plain = dict(value)
    digest = plain.pop("digest", None)
    if not _is_sha256(digest) or object_sha256(plain) != digest:
        raise FullExact81ContractError(f"{label} digest differs")
    plain["digest"] = digest
    return plain


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise FullExact81ContractError(f"{label} must be lowercase SHA256")
    return value


def false_authority() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_FIELDS}


def assert_no_elevated_authority(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "authority":
                if (
                    not isinstance(item, Mapping)
                    or set(item) != set(AUTHORITY_FIELDS)
                    or any(item.get(name) is not False for name in AUTHORITY_FIELDS)
                ):
                    raise FullExact81ContractError(
                        "decoded result elevated authority"
                    )
            elif (
                key in AUTHORITY_FIELDS
                or key.endswith("_authorized")
                or "authority" in key
            ) and item is not False:
                raise FullExact81ContractError("decoded result elevated authority")
            assert_no_elevated_authority(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_elevated_authority(item)


_TENSOR_IDENTITY_KEYS = {
    "shape",
    "dtype",
    "device_type_at_observation",
    "finite",
    "byte_count",
    "raw_sha256",
    "content_sha256",
}


def validate_tensor_identity(
    value: Any,
    *,
    label: str,
    expected_shape: Sequence[int] | None = None,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TENSOR_IDENTITY_KEYS:
        raise FullExact81ContractError(f"{label} identity schema differs")
    shape = value.get("shape")
    if (
        type(shape) is not list
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
        or value.get("finite") is not True
        or type(value.get("dtype")) is not str
        or type(value.get("device_type_at_observation")) is not str
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
    ):
        raise FullExact81ContractError(f"{label} identity values differ")
    if expected_shape is not None and shape != list(expected_shape):
        raise FullExact81ContractError(f"{label} shape differs")
    if expected_dtype is not None and value.get("dtype") != expected_dtype:
        raise FullExact81ContractError(f"{label} dtype differs")
    require_sha256(value.get("raw_sha256"), label=f"{label} raw SHA")
    require_sha256(value.get("content_sha256"), label=f"{label} content SHA")
    return dict(value)


_TRACE_ROW_KEYS = {
    "schedule_index",
    "timestep",
    "sigma_float32_be_hex",
    "state_before",
    "native_visual_pack",
    "native_rotary_pack",
    "negative_raw",
    "action_raw",
    "guided_velocity",
    "state_after",
    "route_receipts",
    "scheduler_step_call_count",
    "source_conditioned",
    "action_positive_condition",
    "target_video_used",
    "clean_source_initial_latent_used",
}

_ROUTE_PROJECTION_KEYS = {
    "schema_version",
    "schedule_index",
    "branch_name",
    "total_tokens",
    "condition_tokens",
    "target_tokens",
    "sequence_parallel_size",
    "sigma_hex",
    "gate_hex",
    "atlas_receipt_digest",
    "source_memory_owned_by_V_VI_only",
    "enabled",
    "all_sp4_ranks_apply_same_global_route",
    "local_rank_validated_before_projection",
    "rank_specific_receipt_digest_not_cross_rank_comparable",
    "digest",
}


def _expected_route_gate(sigma: float) -> float:
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise FullExact81ContractError("route sigma lies outside [0,1]")
    if sigma <= 0.25:
        return 1.0
    if sigma >= 0.75:
        return 0.0
    u = (0.75 - sigma) / (0.75 - 0.25)
    return u * u * (3.0 - 2.0 * u)


def _validate_route_projection(
    value: Any, *, schedule_index: int, sigma_float32_be_hex: str
) -> dict[str, Any]:
    route = validate_sealed_mapping(value, label="exact40 route projection")
    if set(route) != _ROUTE_PROJECTION_KEYS:
        raise FullExact81ContractError("exact40 route projection schema differs")
    total_tokens = route.get("total_tokens")
    condition_tokens = route.get("condition_tokens")
    sigma_hex = route.get("sigma_hex")
    gate_hex = route.get("gate_hex")
    try:
        sigma = float.fromhex(sigma_hex)
        gate = float.fromhex(gate_hex)
    except (TypeError, ValueError, OverflowError) as error:
        raise FullExact81ContractError(
            "exact40 route projection float encoding differs"
        ) from error
    expected_gate = _expected_route_gate(sigma)
    if (
        route.get("schema_version")
        != "bernini-graft-phase-a-full-exact81-route-projection-v1"
        or route.get("schedule_index") != schedule_index
        or route.get("branch_name") != "V"
        or type(total_tokens) is not int
        or total_tokens <= 0
        or type(condition_tokens) is not int
        or not 0 <= condition_tokens < total_tokens
        or route.get("target_tokens") != total_tokens - condition_tokens
        or route.get("sequence_parallel_size") != SP_SIZE
        or type(sigma_hex) is not str
        or sigma.hex() != sigma_hex
        or struct.pack(">f", sigma).hex() != sigma_float32_be_hex
        or type(gate_hex) is not str
        or not math.isfinite(gate)
        or gate.hex() != gate_hex
        or gate_hex != expected_gate.hex()
        or not _is_sha256(route.get("atlas_receipt_digest"))
        or route.get("source_memory_owned_by_V_VI_only") is not True
        or route.get("enabled") is not True
        or route.get("all_sp4_ranks_apply_same_global_route") is not True
        or route.get("local_rank_validated_before_projection") is not True
        or route.get("rank_specific_receipt_digest_not_cross_rank_comparable")
        is not True
        or (schedule_index < 26 and gate != 0.0)
        or (schedule_index >= 26 and gate <= 0.0)
    ):
        raise FullExact81ContractError("exact40 route projection differs")
    return route


def validate_exact40_trace(value: Any) -> dict[str, Any]:
    trace = validate_sealed_mapping(value, label="exact40 trace")
    if set(trace) != {
        "schema_version",
        "rows",
        "official_unipc_step_count",
        "initial_state_role",
        "source_condition_role",
        "positive_condition_role",
        "same_gaussian_seed_across_sp4",
        "cross_index_selection_used",
        "checkpoint_loaded_from_dependency",
        "checkpoint_written",
        "digest",
    }:
        raise FullExact81ContractError("exact40 trace root schema differs")
    rows = trace.get("rows")
    if (
        trace.get("schema_version") != TRACE_SCHEMA_VERSION
        or not isinstance(rows, list)
        or len(rows) != NUM_INFERENCE_STEPS
        or trace.get("official_unipc_step_count") != NUM_INFERENCE_STEPS
        or trace.get("initial_state_role") != "fresh-source-keyed-standard-gaussian"
        or trace.get("source_condition_role") != "full-confirmation-source-v-pack"
        or trace.get("positive_condition_role") != "preregistered-action-text"
        or trace.get("same_gaussian_seed_across_sp4") is not True
        or trace.get("cross_index_selection_used") is not False
        or trace.get("checkpoint_loaded_from_dependency") is not False
        or trace.get("checkpoint_written") is not False
    ):
        raise FullExact81ContractError("exact40 trace contract differs")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _TRACE_ROW_KEYS:
            raise FullExact81ContractError(f"exact40 trace row {index} schema differs")
        sigma_hex = row.get("sigma_float32_be_hex")
        if (
            row.get("schedule_index") != index
            or type(row.get("timestep")) is not int
            or type(sigma_hex) is not str
            or len(sigma_hex) != 8
            or any(character not in "0123456789abcdef" for character in sigma_hex)
            or row.get("scheduler_step_call_count") != 1
            or row.get("source_conditioned") is not True
            or row.get("action_positive_condition") is not True
            or row.get("target_video_used") is not False
            or row.get("clean_source_initial_latent_used") is not False
        ):
            raise FullExact81ContractError(f"exact40 trace row {index} differs")
        for name in (
            "state_before",
            "native_visual_pack",
            "native_rotary_pack",
            "negative_raw",
            "action_raw",
            "guided_velocity",
            "state_after",
        ):
            validate_tensor_identity(row.get(name), label=f"trace {index} {name}")
        routes = row.get("route_receipts")
        if (
            not isinstance(routes, Mapping)
            or set(routes) != {"negative", "action"}
            or routes["negative"] != routes["action"]
        ):
            raise FullExact81ContractError(f"exact40 trace row {index} route differs")
        try:
            _validate_route_projection(
                routes["negative"],
                schedule_index=index,
                sigma_float32_be_hex=sigma_hex,
            )
        except FullExact81ContractError as error:
            raise FullExact81ContractError(
                f"exact40 trace row {index} route differs"
            ) from error
    for left, right in zip(rows, rows[1:]):
        if left["state_after"] != right["state_before"]:
            raise FullExact81ContractError("exact40 state chain is discontinuous")
    return trace


def validate_media_record(
    value: Any, *, expected_height: int, expected_width: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "frame_count",
        "fps_numerator",
        "fps_denominator",
        "reported_fps_numerator",
        "reported_fps_denominator",
        "height",
        "width",
        "decoded_tensor_shape",
        "codec_content_interpreted_for_semantics",
    }:
        raise FullExact81ContractError("decoded media record schema differs")
    if (
        type(expected_height) is not int
        or type(expected_width) is not int
        or expected_height <= 0
        or expected_width <= 0
        or value.get("schema_version") != MEDIA_SCHEMA_VERSION
        or value.get("frame_count") != FRAME_COUNT
        or value.get("fps_numerator") != FPS_NUMERATOR
        or value.get("fps_denominator") != FPS_DENOMINATOR
        or value.get("reported_fps_numerator") != FPS_NUMERATOR
        or value.get("reported_fps_denominator") != FPS_DENOMINATOR
        or value.get("height") != expected_height
        or value.get("width") != expected_width
        or value.get("decoded_tensor_shape")
        != [FRAME_COUNT, expected_height, expected_width, 3]
        or value.get("codec_content_interpreted_for_semantics") is not False
    ):
        raise FullExact81ContractError("decoded media record differs")
    return dict(value)


def validate_artifact_record(
    value: Any,
    *,
    role: str,
    output_root: str,
    suffix: str,
    endpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "role",
        "path",
        "relative_path",
        "size_bytes",
        "mode",
        "regular_file",
        "link_count_one",
        "sha256",
        "opened_nofollow_and_revalidated",
        "content_binding",
    }:
        raise FullExact81ContractError(f"{role} artifact schema differs")
    relative = value.get("relative_path")
    path = value.get("path")
    pure_relative = PurePosixPath(relative) if type(relative) is str else None
    pure_root = PurePosixPath(output_root) if type(output_root) is str else None
    if (
        value.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or value.get("role") != role
        or pure_relative is None
        or pure_relative.is_absolute()
        or ".." in pure_relative.parts
        or not str(pure_relative).endswith(suffix)
        or pure_root is None
        or not pure_root.is_absolute()
        or type(path) is not str
        or PurePosixPath(path) != pure_root / pure_relative
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
        or value.get("mode") != "0444"
        or value.get("regular_file") is not True
        or value.get("link_count_one") is not True
        or value.get("opened_nofollow_and_revalidated") is not True
    ):
        raise FullExact81ContractError(f"{role} artifact differs")
    require_sha256(value.get("sha256"), label=f"{role} artifact SHA")
    endpoint = validate_tensor_identity(
        endpoint_identity, label=f"{role} endpoint binding"
    )
    binding = value.get("content_binding")
    if role == "normalized-clean-latent":
        if not isinstance(binding, Mapping) or set(binding) != {
            "kind",
            "tensor_key",
            "tensor_shape",
            "tensor_dtype",
            "tensor_raw_sha256",
            "tensor_content_sha256",
            "endpoint_raw_sha256",
            "endpoint_content_sha256",
            "safetensors_roundtrip_verified",
        }:
            raise FullExact81ContractError("latent content binding schema differs")
        if (
            binding.get("kind") != "safetensors-exact-endpoint-tensor"
            or binding.get("tensor_key") != "normalized_clean_latent"
            or binding.get("tensor_shape") != endpoint["shape"]
            or binding.get("tensor_dtype") != endpoint["dtype"]
            or binding.get("tensor_raw_sha256") != endpoint["raw_sha256"]
            or binding.get("tensor_content_sha256")
            != endpoint["content_sha256"]
            or binding.get("endpoint_raw_sha256") != endpoint["raw_sha256"]
            or binding.get("endpoint_content_sha256")
            != endpoint["content_sha256"]
            or binding.get("safetensors_roundtrip_verified") is not True
        ):
            raise FullExact81ContractError("latent content binding differs")
    elif role == "decoded-exact81-video":
        if not isinstance(binding, Mapping) or binding != {
            "kind": "same-call-vae-decode-from-sealed-endpoint",
            "decoded_from_endpoint_raw_sha256": endpoint["raw_sha256"],
            "decoded_from_endpoint_content_sha256": endpoint["content_sha256"],
            "endpoint_unchanged_after_decode": True,
            "semantic_content_interpreted": False,
        }:
            raise FullExact81ContractError("video content binding differs")
    else:
        raise FullExact81ContractError("artifact role binding differs")
    return dict(value)


def build_local_result(
    *,
    global_rank: int,
    dp_arm: int,
    family: str,
    confirmation_iid: str,
    source_sha256: str,
    action_prompt_sha256: str,
    seed: int,
    short_receipt_digest: str,
    field14_receipt_digest: str,
    active14_precommit_digest: str,
    initial_gaussian_identity: Mapping[str, Any],
    endpoint_identity: Mapping[str, Any],
    exact40_trace: Mapping[str, Any],
    media: Mapping[str, Any],
    latent_artifact: Mapping[str, Any],
    video_artifact: Mapping[str, Any],
    output_root: str,
    expected_height: int,
    expected_width: int,
    trainable_sha256_before_decode: str,
    trainable_sha256_after_decode: str,
    base_sha256_before_decode: str,
    base_sha256_after_decode: str,
) -> Mapping[str, Any]:
    if (
        type(global_rank) is not int
        or type(dp_arm) is not int
        or not 0 <= global_rank < WORLD_SIZE
        or dp_arm not in range(DP_SIZE)
        or global_rank // SP_SIZE != dp_arm
        or family != FAMILY_ORDER[dp_arm]
        or type(confirmation_iid) is not str
        or not confirmation_iid
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise FullExact81ContractError("local rank/family registration differs")
    digests = {
        "source_sha256": source_sha256,
        "action_prompt_sha256": action_prompt_sha256,
        "short_receipt_digest": short_receipt_digest,
        "field14_receipt_digest": field14_receipt_digest,
        "active14_precommit_digest": active14_precommit_digest,
        "trainable_sha256_before_decode": trainable_sha256_before_decode,
        "trainable_sha256_after_decode": trainable_sha256_after_decode,
        "base_sha256_before_decode": base_sha256_before_decode,
        "base_sha256_after_decode": base_sha256_after_decode,
    }
    for label, digest in digests.items():
        require_sha256(digest, label=label)
    if (
        trainable_sha256_before_decode != trainable_sha256_after_decode
        or base_sha256_before_decode != base_sha256_after_decode
    ):
        raise FullExact81ContractError("decode changed parameter bytes")
    latent_shape = [1, 16, LATENT_PHASES, expected_height // 8, expected_width // 8]
    gaussian = validate_tensor_identity(
        initial_gaussian_identity,
        label="initial Gaussian",
        expected_shape=latent_shape,
        expected_dtype="torch.float32",
    )
    endpoint = validate_tensor_identity(
        endpoint_identity,
        label="exact81 endpoint",
        expected_shape=latent_shape,
        expected_dtype="torch.float32",
    )
    trace = validate_exact40_trace(exact40_trace)
    if (
        trace["rows"][0]["state_before"] != gaussian
        or trace["rows"][-1]["state_after"] != endpoint
    ):
        raise FullExact81ContractError("Gaussian/endpoint does not bind trace")
    media_row = validate_media_record(
        media, expected_height=expected_height, expected_width=expected_width
    )
    latent_row = validate_artifact_record(
        latent_artifact,
        role="normalized-clean-latent",
        output_root=output_root,
        suffix=".safetensors",
        endpoint_identity=endpoint,
    )
    video_row = validate_artifact_record(
        video_artifact,
        role="decoded-exact81-video",
        output_root=output_root,
        suffix=".mp4",
        endpoint_identity=endpoint,
    )
    value = {
        "schema_version": LOCAL_SCHEMA_VERSION,
        "global_rank": global_rank,
        "dp_arm": dp_arm,
        "sp_rank": global_rank % SP_SIZE,
        "family": family,
        "confirmation_iid": confirmation_iid,
        "source_sha256": source_sha256,
        "action_prompt_sha256": action_prompt_sha256,
        "seed": seed,
        "state_continuity": {
            "same_process_from_base": True,
            "short_replayed": True,
            "field14_replayed": True,
            "active14_replayed": True,
            "dependency_afterok_used_only_as_queue_gate": True,
            "weights_inherited_from_dependency": False,
            "checkpoint_loaded_from_dependency": False,
        },
        "upstream_in_memory_receipts": {
            "short": short_receipt_digest,
            "field14": field14_receipt_digest,
            "active14_precommit": active14_precommit_digest,
        },
        "initial_gaussian": gaussian,
        "endpoint": endpoint,
        "exact40_trace": trace,
        "media": media_row,
        "artifacts": {"latent": latent_row, "video": video_row},
        "parameter_integrity": {
            "trainable_sha256_before_decode": trainable_sha256_before_decode,
            "trainable_sha256_after_decode": trainable_sha256_after_decode,
            "base_sha256_before_decode": base_sha256_before_decode,
            "base_sha256_after_decode": base_sha256_after_decode,
            "trainable_bytes_unchanged_during_decode": True,
            "base_bytes_unchanged_during_decode": True,
        },
        "checkpoint_policy": {
            "checkpoint_written": False,
            "checkpoint_publishable": False,
            "optimizer_state_exposed_to_continuation": False,
        },
        "rollback_policy": {
            "active14_owner_restores_adapter_and_process_group": True,
            "failed_staging_never_renamed_to_final_output": True,
            "no_partial_success_receipt": True,
            "no_checkpoint_to_roll_back": True,
        },
        "decoded_media_created": True,
        "decoded_media_semantically_evaluated": False,
        **false_authority(),
    }
    assert_no_elevated_authority(value)
    return seal_mapping(value)


def assemble_world8_result(packets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(packets) != WORLD_SIZE:
        raise FullExact81ContractError("WORLD8 packet count differs")
    admitted = [
        validate_sealed_mapping(packet, label=f"WORLD8 packet {index}")
        for index, packet in enumerate(packets)
    ]
    if [row.get("global_rank") for row in admitted] != list(range(WORLD_SIZE)):
        raise FullExact81ContractError("WORLD8 rank order differs")
    upstream_receipt_rows = []
    for rank, row in enumerate(admitted):
        upstream = row.get("upstream_in_memory_receipts")
        if not isinstance(upstream, Mapping) or set(upstream) != {
            "short",
            "field14",
            "active14_precommit",
        }:
            raise FullExact81ContractError(
                f"WORLD8 packet {rank} upstream receipt schema differs"
            )
        for name, digest in upstream.items():
            require_sha256(digest, label=f"WORLD8 rank {rank} {name} receipt")
        upstream_receipt_rows.append(
            {"global_rank": rank, **dict(upstream)}
        )
    upstream_receipt_matrix_digest = object_sha256(upstream_receipt_rows)
    representatives = []
    for dp_arm, family in enumerate(FAMILY_ORDER):
        group = admitted[dp_arm * SP_SIZE : (dp_arm + 1) * SP_SIZE]
        if any(
            row.get("dp_arm") != dp_arm
            or row.get("sp_rank") != offset
            or row.get("family") != family
            for offset, row in enumerate(group)
        ):
            raise FullExact81ContractError(f"{family} SP4 topology differs")
        projection_keys = (
            "family",
            "confirmation_iid",
            "source_sha256",
            "action_prompt_sha256",
            "seed",
            "state_continuity",
            "initial_gaussian",
            "endpoint",
            "exact40_trace",
            "media",
            "artifacts",
            "parameter_integrity",
            "checkpoint_policy",
            "rollback_policy",
            "decoded_media_created",
            "decoded_media_semantically_evaluated",
        ) + AUTHORITY_FIELDS
        projections = [
            {key: row[key] for key in projection_keys}
            for row in group
        ]
        if any(row != projections[0] for row in projections[1:]):
            raise FullExact81ContractError(f"{family} SP4 result differs")
        representatives.append(
            {
                "dp_arm": dp_arm,
                "family": family,
                "representative_global_rank": dp_arm * SP_SIZE,
                "local_result_digest": group[0]["digest"],
                "sp4_projection_digest": object_sha256(projections[0]),
                "sp4_exact": True,
                "video_sha256": group[0]["artifacts"]["video"]["sha256"],
                "latent_sha256": group[0]["artifacts"]["latent"]["sha256"],
            }
        )
    value = {
        "schema_version": WORLD8_SCHEMA_VERSION,
        "rank_order": list(range(WORLD_SIZE)),
        "family_order": list(FAMILY_ORDER),
        "rows": admitted,
        "arm_representatives": representatives,
        "rank_local_upstream_receipt_bindings": upstream_receipt_rows,
        "rank_local_upstream_receipt_bindings_digest": (
            upstream_receipt_matrix_digest
        ),
        "both_sp4_arms_exact": True,
        "both_exact40_completed": True,
        "both_exact81_decoded": True,
        "all_parameter_bytes_unchanged_during_decode": True,
        "checkpoint_written": False,
        "publication_performed": False,
        "visual_semantics_evaluated": False,
        **false_authority(),
    }
    assert_no_elevated_authority(value)
    return seal_mapping(value)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "AUTHORITY_FIELDS",
    "DP_SIZE",
    "FAMILY_ORDER",
    "FPS_DENOMINATOR",
    "FPS_NUMERATOR",
    "FRAME_COUNT",
    "FullExact81ContractError",
    "LATENT_PHASES",
    "LOCAL_SCHEMA_VERSION",
    "MEDIA_SCHEMA_VERSION",
    "NUM_INFERENCE_STEPS",
    "SCHEMA_VERSION",
    "SP_SIZE",
    "TRACE_SCHEMA_VERSION",
    "WORLD8_SCHEMA_VERSION",
    "WORLD_SIZE",
    "assemble_world8_result",
    "assert_no_elevated_authority",
    "build_local_result",
    "canonical_json_bytes",
    "false_authority",
    "object_sha256",
    "require_sha256",
    "seal_mapping",
    "validate_artifact_record",
    "validate_exact40_trace",
    "validate_media_record",
    "validate_sealed_mapping",
    "validate_tensor_identity",
]
