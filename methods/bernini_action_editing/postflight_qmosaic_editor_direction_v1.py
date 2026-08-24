#!/usr/bin/env python3
"""Postflight the fixed Q-MOSAIC base/plus/minus direction artifacts.

The postflight owns only deterministic integrity/numerical/media checks.  It
does not accept action, identity, camera, or quality booleans from a caller.
Until a method-owned decoded evaluator exists, a successful postflight remains
``SEMANTICS_UNASSESSED_NO_LORA`` and cannot authorize a LoRA VJP or update.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import run_qmosaic_editor_direction_sp4_v1 as runtime  # noqa: E402


POSTFLIGHT_SCHEMA = "bernini-qmosaic-editor-direction-postflight-v3"
POSTFLIGHT_FILENAME = "postflight.receipt.json"


class QMosaicDirectionPostflightError(RuntimeError):
    """The run receipt, artifact closure, or live decode proof differed."""


def _read_run_receipt(
    path: str | Path, *, expected_file_sha256: str, _p_qmosaic: bool = False
) -> tuple[Mapping[str, Any], Path]:
    p_profile: Any = None
    if _p_qmosaic:
        import p_qmosaic_direction_envelope_v1 as p_profile
    source = Path(path)
    expected = runtime._sha256(  # noqa: SLF001 - same method receipt primitive
        expected_file_sha256, label="run receipt file SHA-256"
    )
    try:
        observed = runtime._file_sha256(source)  # noqa: SLF001
    except runtime.QMosaicEditorDirectionError as error:
        raise QMosaicDirectionPostflightError(str(error)) from error
    if observed != expected:
        raise QMosaicDirectionPostflightError("run receipt bytes changed")
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QMosaicDirectionPostflightError("run receipt JSON differs") from error
    if (
        not isinstance(value, Mapping)
        or raw != runtime._canonical_json_bytes(value) + b"\n"  # noqa: SLF001
        or value.get("schema_version")
        != (p_profile.RUN_RECEIPT_SCHEMA if _p_qmosaic else runtime.RUN_RECEIPT_SCHEMA)
    ):
        raise QMosaicDirectionPostflightError("run receipt schema/canonical bytes differ")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if digest != runtime.object_sha256(unsigned):
        raise QMosaicDirectionPostflightError("run receipt seal differs")
    return dict(value), source.resolve(strict=True)


def validate_run_artifacts(
    *,
    run_receipt_path: str | Path,
    expected_run_receipt_file_sha256: str,
    artifact_root: str | Path,
    probe_fn: Callable[[str | Path], Mapping[str, Any]] = runtime._probe_exact81_25fps,  # noqa: SLF001
    _p_qmosaic: bool = False,
) -> Mapping[str, Any]:
    """Reopen every artifact and enforce the non-semantic fail-closed gate."""

    receipt, receipt_path = _read_run_receipt(
        run_receipt_path,
        expected_file_sha256=expected_run_receipt_file_sha256,
        _p_qmosaic=_p_qmosaic,
    )
    p_profile: Any = None
    if _p_qmosaic:
        import p_qmosaic_direction_envelope_v1 as p_profile
    root = Path(artifact_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise QMosaicDirectionPostflightError("artifact root must be an absolute plain directory")
    resolved_root = root.resolve(strict=True)
    try:
        receipt_path.relative_to(resolved_root)
    except ValueError as error:
        raise QMosaicDirectionPostflightError("run receipt escaped artifact root") from error
    expected_top_keys = {
        "schema_version",
        "method_name",
        "experiment_scope",
        "registry",
        "native_coordinate",
        "predecode_parity",
        "symmetric_direction",
        "terminal_full_seal",
        "published_arms",
        "all_fixed_arms_published",
        "parameter_invariance",
        "semantic_assessment",
        "authorization",
        "output_contract",
        "method_source",
        "receipt_digest",
    }
    if _p_qmosaic:
        expected_top_keys.add("direction_variant")
    if (
        set(receipt) != expected_top_keys
        or receipt.get("method_name")
        != (p_profile.METHOD_NAME if _p_qmosaic else runtime.METHOD_NAME)
        or (
            _p_qmosaic
            and receipt.get("direction_variant") != p_profile.variant_lock()
        )
    ):
        raise QMosaicDirectionPostflightError("run receipt field closure differs")
    registry = receipt.get("registry")
    scope = receipt.get("experiment_scope")
    native = receipt.get("native_coordinate")
    parity = receipt.get("predecode_parity")
    direction = receipt.get("symmetric_direction")
    terminal = receipt.get("terminal_full_seal")
    invariance = receipt.get("parameter_invariance")
    semantic = receipt.get("semantic_assessment")
    authorization = receipt.get("authorization")
    output_contract = receipt.get("output_contract")
    method_source = receipt.get("method_source")
    if not all(
        isinstance(value, Mapping)
        for value in (
            registry,
            scope,
            native,
            parity,
            direction,
            terminal,
            invariance,
            semantic,
            authorization,
            output_contract,
            method_source,
        )
    ):
        raise QMosaicDirectionPostflightError("run receipt nested closure differs")
    cell_id = registry.get("cell_id")
    query_seed = registry.get("query_seed")
    if dict(scope) != {
        "classification": "ENGINEERING_ONLY" if _p_qmosaic else "ENGINEERING_SMOKE_ONLY",
        "scientific_evidence_authority": False,
        "semantic_authority": False,
        "lora_or_parameter_update_authority": False,
    }:
        raise QMosaicDirectionPostflightError(
            "run receipt must remain engineering-smoke-only"
        )
    if (
        registry.get("file_sha256") != runtime.FIXED_REGISTRY_SHA256
        or cell_id not in runtime.FIXED_QUERY_SEEDS
        or query_seed not in runtime.FIXED_QUERY_SEEDS[cell_id]
        or registry.get("fixed_query_seeds_for_cell")
        != list(runtime.FIXED_QUERY_SEEDS[cell_id])
        or native.get("world_size") != runtime.WORLD_SIZE
        or native.get("ulysses_size") != runtime.SP_SIZE
        or native.get("schedule_index") != runtime.NATIVE_SCHEDULE_INDEX
        or native.get("timestep") != runtime.NATIVE_TIMESTEP
        or native.get("frame_count") != runtime.EXPECTED_FRAMES
        or native.get("fps") != runtime.EXPECTED_FPS
        or native.get("relative_l2_dose") != runtime.RELATIVE_L2_DOSE
        or native.get("editor_method_source_revision")
        != method_source.get("revision")
        or native.get("editor_method_source_archive_sha256")
        != method_source.get("archive_sha256")
    ):
        raise QMosaicDirectionPostflightError("fixed registry/native coordinate differs")
    materialization_path = Path(
        str(native.get("editor_materialization_receipt_path"))
    )
    try:
        materialization_sha = runtime._file_sha256(  # noqa: SLF001
            materialization_path
        )
        materialization_raw = materialization_path.read_bytes()
        materialization = json.loads(materialization_raw.decode("ascii"))
    except (
        runtime.QMosaicEditorDirectionError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise QMosaicDirectionPostflightError(
            "editor materialization receipt cannot be reopened"
        ) from error
    materialization_unsigned = dict(materialization)
    materialization_digest = materialization_unsigned.pop("receipt_digest", None)
    materialization_source = materialization.get("method_source")
    if (
        materialization_sha
        != native.get("editor_materialization_receipt_file_sha256")
        or materialization_digest
        != native.get("editor_materialization_receipt_digest")
        or materialization_digest
        != runtime.object_sha256(materialization_unsigned)
        or materialization_raw
        != runtime._canonical_json_bytes(materialization) + b"\n"  # noqa: SLF001
        or not isinstance(materialization_source, Mapping)
        or materialization_source.get("revision")
        != method_source.get("revision")
        or materialization_source.get("archive_file_sha256")
        != method_source.get("archive_sha256")
    ):
        raise QMosaicDirectionPostflightError(
            "editor materialization/source triple differs"
        )
    if (
        parity.get("b0_z0_predecode_exact_parity") is not True
        or parity.get("b0_tensor_sha256") != parity.get("z0_tensor_sha256")
        or parity.get(
            "native_zero_lora_structural_forward_identity_proven"
        )
        is not True
        or parity.get(
            "separate_off_enabled_sketch_comparison_used_for_authority"
        )
        is not False
        or parity.get(
            "b0_z0_and_all_direction_arms_share_source_noise_prompt_scheduler"
        )
        is not True
        or parity.get("native_schedule_index") != runtime.NATIVE_SCHEDULE_INDEX
        or parity.get("native_timestep") != runtime.NATIVE_TIMESTEP
        or any(
            not isinstance(parity.get(name), str)
            or runtime._SHA256_RE.fullmatch(parity[name]) is None  # noqa: SLF001
            for name in (
                "source_latent_tensor_sha256",
                "official_initial_noise_tensor_sha256",
                "action_prompt_sha256",
                "noop_prompt_sha256",
                "prompt_condition_binding_digest",
                "checkpoint_content_receipt_digest",
            )
        )
        or (
            not _p_qmosaic
            and (
                direction.get("relative_l2_dose") != runtime.RELATIVE_L2_DOSE
                or direction.get("formula_recomputed_exact_fp32") is not True
                or direction.get("latent_symmetry_passed") is not True
                or direction.get("formula")
                != "q=g/l2(g);scale=0.01*l2(base);plus=base+scale*q;minus=base-scale*q"
                or not isinstance(direction.get("symmetry_tolerance"), (int, float))
                or float(direction["midpoint_max_abs_error"])
                > float(direction["symmetry_tolerance"])
                or float(direction["delta_antisymmetry_max_abs_error"])
                > 2.0 * float(direction["symmetry_tolerance"])
            )
        )
    ):
        raise QMosaicDirectionPostflightError("predecode parity/symmetry evidence differs")
    try:
        runtime.validate_world4_zero_route_proof(
            parity.get("world4_zero_lora_structural_proof"), parity=parity
        )
    except runtime.QMosaicEditorDirectionError as error:
        raise QMosaicDirectionPostflightError(
            "WORLD4 structural zero-LoRA proof differs"
        ) from error

    route_rows = parity.get("separate_off_enabled_sketch_diagnostic_by_sp_rank")
    diagnostic_fields = {
        "sp_rank",
        "role",
        "shape",
        "dtype",
        "adapter_off_tensor_sha256",
        "enabled_zero_b_tensor_sha256",
        "numeric_exact_equal",
        "raw_byte_exact_equal",
        "numeric_mismatch_element_count",
        "max_absolute_difference",
        "confounded_by_separate_forward_rocm_reduction_and_route_mode",
        "authoritative_for_zero_route_identity",
        "allclose_or_tolerance_used",
    }
    if (
        not isinstance(route_rows, list)
        or [row.get("sp_rank") if isinstance(row, Mapping) else None for row in route_rows]
        != list(range(runtime.SP_SIZE))
        or any(
            set(row) != {"sp_rank", "roles"}
            or not isinstance(row.get("roles"), list)
            or [
                item.get("role") if isinstance(item, Mapping) else None
                for item in row.get("roles", [])
            ]
            != ["action", "noop"]
            or any(
                not isinstance(item, Mapping)
                or set(item) != diagnostic_fields
                or item.get("sp_rank") != row.get("sp_rank")
                or item.get("dtype") != "torch.float32"
                or not isinstance(item.get("shape"), list)
                or not item.get("shape")
                or any(
                    runtime._SHA256_RE.fullmatch(str(item.get(name))) is None  # noqa: SLF001
                    for name in (
                        "adapter_off_tensor_sha256",
                        "enabled_zero_b_tensor_sha256",
                    )
                )
                or type(item.get("numeric_exact_equal")) is not bool
                or type(item.get("raw_byte_exact_equal")) is not bool
                or type(item.get("numeric_mismatch_element_count")) is not int
                or item.get("numeric_mismatch_element_count") < 0
                or not isinstance(item.get("max_absolute_difference"), (int, float))
                or not math.isfinite(float(item["max_absolute_difference"]))
                or float(item["max_absolute_difference"]) < 0.0
                or item.get(
                    "confounded_by_separate_forward_rocm_reduction_and_route_mode"
                )
                is not True
                or item.get("authoritative_for_zero_route_identity") is not False
                or item.get("allclose_or_tolerance_used") is not False
                or (
                    item.get("numeric_exact_equal") is True
                    and (
                        item.get("numeric_mismatch_element_count") != 0
                        or float(item["max_absolute_difference"]) != 0.0
                    )
                )
                or (
                    item.get("raw_byte_exact_equal")
                    != (
                        item.get("adapter_off_tensor_sha256")
                        == item.get("enabled_zero_b_tensor_sha256")
                    )
                )
                for item in row.get("roles", [])
            )
            for row in route_rows
        )
    ):
        raise QMosaicDirectionPostflightError(
            "WORLD4 non-authoritative separate-forward diagnostic differs"
        )
    if (
        terminal.get("called_before_any_mp4_or_receipt_publication") is not True
        or terminal.get("deep_full_byte_revalidated") is not True
        or set(terminal)
        != {
            "called_before_any_mp4_or_receipt_publication",
            "deep_full_byte_revalidated",
            "rank_receipts",
        }
        or not isinstance(terminal.get("rank_receipts"), list)
        or [
            row.get("sp_rank") if isinstance(row, Mapping) else None
            for row in terminal["rank_receipts"]
        ]
        != list(range(runtime.SP_SIZE))
        or any(
            set(row)
            != {
                "sp_rank",
                "terminal_full_seal_receipt_digest",
                "deep_full_byte_revalidated",
            }
            or row.get("deep_full_byte_revalidated") is not True
            or runtime._SHA256_RE.fullmatch(  # noqa: SLF001
                str(row.get("terminal_full_seal_receipt_digest"))
            )
            is None
            for row in terminal["rank_receipts"]
        )
    ):
        raise QMosaicDirectionPostflightError(
            "terminal full-byte pre-publication seal differs"
        )
    if (
        invariance.get("parameter_bytes_unchanged") is not True
        or invariance.get("lora_b_exact_zero_before") is not True
        or invariance.get("lora_b_exact_zero_after") is not True
        or invariance.get("action_lora_state_sha256_before")
        != invariance.get("action_lora_state_sha256_after")
        or invariance.get("lora_b_state_sha256_before")
        != invariance.get("lora_b_state_sha256_after")
        or invariance.get("optimizer_created") is not False
        or invariance.get("parameter_update_performed") is not False
    ):
        raise QMosaicDirectionPostflightError("parameter byte-invariance proof differs")
    expected_semantic = {
        "action": runtime.SEMANTIC_UNASSESSED,
        "identity": runtime.SEMANTIC_UNASSESSED,
        "camera": runtime.SEMANTIC_UNASSESSED,
        "background": runtime.SEMANTIC_UNASSESSED,
        "quality": runtime.SEMANTIC_UNASSESSED,
        "method_owned_decoded_evaluator_available": False,
        "caller_boolean_or_callback_consumed": False,
        "self_reported_semantic_score_consumed": False,
        "decoded_semantic_gate_passed": False,
    }
    if dict(semantic) != expected_semantic:
        raise QMosaicDirectionPostflightError("semantic status must remain method-owned UNASSESSED")
    false_authorities = (
        "lora_vjp_requested",
        "lora_vjp_executed",
        "lora_vjp_authorized",
        "optimizer_created",
        "parameter_update_authorized",
        "parameter_update_performed",
        "adapter_checkpoint_written",
        "scientific_action_editing_success_claim",
    )
    if (
        authorization.get("cli_no_lora_vjp_required") is not True
        or authorization.get("clean_latent_vjp_executed") is not True
        or any(authorization.get(name) is not False for name in false_authorities)
        or output_contract.get("receipt_and_video_only") is not True
        or output_contract.get("latent_or_gradient_tensor_artifact_written") is not False
        or receipt.get("all_fixed_arms_published") is not True
    ):
        raise QMosaicDirectionPostflightError("no-LoRA/output authority boundary differs")

    raw_arms = receipt.get("published_arms")
    required_arm_fields = {
        "role",
        "mp4_path",
        "mp4_file_sha256",
        "latent_tensor_sha256",
        "decode_seed",
        "frame_count",
        "fps",
        "decode_probe",
    }
    if (
        not isinstance(raw_arms, list)
        or [row.get("role") if isinstance(row, Mapping) else None for row in raw_arms]
        != list(runtime.ARM_ORDER)
    ):
        raise QMosaicDirectionPostflightError("published arm order/closure differs")
    live_rows = []
    paths = []
    for role, row in zip(runtime.ARM_ORDER, raw_arms):
        if not isinstance(row, Mapping) or set(row) != required_arm_fields:
            raise QMosaicDirectionPostflightError("published arm field closure differs")
        path = Path(str(row.get("mp4_path")))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise QMosaicDirectionPostflightError("published arm must be an absolute plain MP4")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise QMosaicDirectionPostflightError("published arm escaped artifact root") from error
        observed_sha = runtime._file_sha256(resolved)  # noqa: SLF001
        live_probe = dict(probe_fn(resolved))
        if (
            row.get("role") != role
            or row.get("mp4_file_sha256") != observed_sha
            or row.get("decode_seed") != query_seed
            or row.get("frame_count") != runtime.EXPECTED_FRAMES
            or row.get("fps") != runtime.EXPECTED_FPS
            or row.get("decode_probe") != live_probe
            or set(live_probe) != runtime.EXACT81_25FPS_PROBE_FIELDS
            or live_probe.get("schema_version")
            != runtime.qmosaic.EXACT81_MEDIA_PROBE_SCHEMA_VERSION
            or live_probe.get("pyav_decoded_frame_count")
            != runtime.EXPECTED_FRAMES
            or live_probe.get("bundled_ffmpeg_framemd5_frame_count")
            != runtime.EXPECTED_FRAMES
            or live_probe.get("pyav_exact_25fps_pts_cadence") is not True
            or live_probe.get("fps_exact_integer") != runtime.EXPECTED_FPS
        ):
            raise QMosaicDirectionPostflightError(f"published {role} live proof differs")
        runtime._sha256(row.get("latent_tensor_sha256"), label=f"{role} latent SHA")  # noqa: SLF001
        paths.append(resolved)
        live_rows.append(
            {
                "role": role,
                "mp4_path": str(resolved),
                "mp4_file_sha256": observed_sha,
                "decoded_frame_transcript_sha256": live_probe[
                    "decoded_frame_transcript_sha256"
                ],
                "frame_count": runtime.EXPECTED_FRAMES,
                "fps": runtime.EXPECTED_FPS,
            }
        )
    if _p_qmosaic:
        by_role = {row["role"]: row for row in raw_arms}
        hashes = direction.get("tensor_sha256")
        try:
            p_profile.validate_envelope(
                direction,
                cell_id=str(cell_id),
                query_seed=int(query_seed),
                clean_vjp_receipt_digest=str(
                    native.get("sp4_clean_vjp_receipt_digest")
                ),
                clean_vjp_value_sha256=str(
                    hashes.get("upstream_raw_clean_latent_vjp")
                    if isinstance(hashes, Mapping)
                    else None
                ),
                base_tensor_sha256=str(parity.get("b0_tensor_sha256")),
                plus_tensor_sha256=str(by_role["plus"]["latent_tensor_sha256"]),
                minus_tensor_sha256=str(by_role["minus"]["latent_tensor_sha256"]),
            )
        except (p_profile.PQMosaicDirectionEnvelopeError, TypeError, ValueError) as error:
            raise QMosaicDirectionPostflightError(
                "P-Q direction envelope differs"
            ) from error
        if by_role["base"]["latent_tensor_sha256"] != parity.get(
            "b0_tensor_sha256"
        ):
            raise QMosaicDirectionPostflightError("P-Q base tensor hash differs")
    if (
        len(set(paths)) != len(runtime.ARM_ORDER)
        or len({row["mp4_file_sha256"] for row in live_rows}) != len(runtime.ARM_ORDER)
        or len({row["decoded_frame_transcript_sha256"] for row in live_rows})
        != len(runtime.ARM_ORDER)
        or len({row["latent_tensor_sha256"] for row in raw_arms})
        != len(runtime.ARM_ORDER)
    ):
        raise QMosaicDirectionPostflightError("published base/plus/minus arms alias")
    expected_names = {runtime.RUN_RECEIPT_FILENAME, *(f"{role}.mp4" for role in runtime.ARM_ORDER)}
    children = tuple(resolved_root.iterdir())
    observed_names = {child.name for child in children}
    if (
        observed_names != expected_names
        or any(child.is_symlink() or not child.is_file() for child in children)
    ):
        raise QMosaicDirectionPostflightError("artifact root file closure differs before postflight")
    return {
        "run_receipt": receipt,
        "run_receipt_path": str(receipt_path),
        "run_receipt_file_sha256": expected_run_receipt_file_sha256,
        "artifact_root": str(resolved_root),
        "editor_materialization_receipt_path": str(
            materialization_path.resolve(strict=True)
        ),
        "editor_materialization_receipt_file_sha256": materialization_sha,
        "editor_materialization_receipt_digest": materialization_digest,
        "editor_method_source_revision": method_source["revision"],
        "editor_method_source_archive_sha256": method_source["archive_sha256"],
        "live_arms": live_rows,
        "_p_qmosaic": _p_qmosaic,
    }


def build_postflight_receipt(
    validated: Mapping[str, Any], *, _p_qmosaic: bool = False
) -> Mapping[str, Any]:
    run = validated["run_receipt"]
    p_profile: Any = None
    if _p_qmosaic:
        import p_qmosaic_direction_envelope_v1 as p_profile
    if validated.get("_p_qmosaic", False) is not _p_qmosaic:
        raise QMosaicDirectionPostflightError("postflight profile binding differs")
    unsigned = {
        "schema_version": p_profile.POSTFLIGHT_SCHEMA if _p_qmosaic else POSTFLIGHT_SCHEMA,
        "method_name": p_profile.METHOD_NAME if _p_qmosaic else runtime.METHOD_NAME,
        "run_receipt_digest": run["receipt_digest"],
        "run_receipt_file_sha256": validated["run_receipt_file_sha256"],
        "artifact_root": validated["artifact_root"],
        "editor_materialization_receipt_path": validated[
            "editor_materialization_receipt_path"
        ],
        "editor_materialization_receipt_file_sha256": validated[
            "editor_materialization_receipt_file_sha256"
        ],
        "editor_materialization_receipt_digest": validated[
            "editor_materialization_receipt_digest"
        ],
        "editor_method_source_revision": validated[
            "editor_method_source_revision"
        ],
        "editor_method_source_archive_sha256": validated[
            "editor_method_source_archive_sha256"
        ],
        "cell_id": run["registry"]["cell_id"],
        "query_seed": run["registry"]["query_seed"],
        "fixed_registry_sha256": runtime.FIXED_REGISTRY_SHA256,
        "experiment_scope": "ENGINEERING_ONLY" if _p_qmosaic else "ENGINEERING_SMOKE_ONLY",
        "b0_z0_predecode_exact_parity": True,
        "native_zero_lora_structural_forward_identity_proven": True,
        "separate_forward_sketch_diagnostic_used_for_authority": False,
        "plus_minus_latent_symmetry_passed": True,
        "terminal_deep_full_byte_revalidated_before_publication": True,
        "terminal_full_seal_receipt_digests_by_sp_rank": [
            row["terminal_full_seal_receipt_digest"]
            for row in run["terminal_full_seal"]["rank_receipts"]
        ],
        "all_base_plus_minus_mp4_live_decoded_exact81_25fps": True,
        "live_arms": list(validated["live_arms"]),
        "artifact_integrity_postflight_passed": True,
        "semantic_assessment": {
            "status": "SEMANTICS_UNASSESSED_NO_LORA",
            "action": runtime.SEMANTIC_UNASSESSED,
            "identity": runtime.SEMANTIC_UNASSESSED,
            "caller_boolean_or_callback_consumed": False,
            "decoded_semantic_gate_passed": False,
        },
        "lora_vjp_authorized": False,
        "parameter_update_authorized": False,
        "scientific_action_editing_success_claim": False,
        "disposition": "NUMERIC_AND_MEDIA_PASS_SEMANTICS_UNASSESSED_NO_LORA",
    }
    if _p_qmosaic:
        direction = run["symmetric_direction"]
        adapter_receipt = direction["runtime_adapter_receipt"]
        unsigned["direction_variant"] = dict(p_profile.variant_lock())
        unsigned["p_qmosaic_direction_evidence"] = {
            "schema_version": p_profile.EVIDENCE_SCHEMA,
            "receipt_digest": direction["receipt_digest"],
            "runtime_adapter_receipt_digest": adapter_receipt["receipt_digest"],
            "live_tensor_hashes_recomputed_before_run_receipt": True,
            "postflight_seal_and_cross_hashes_recomputed": True,
            "fixed_nuisance_nulls": list(direction["fixed_nuisance_nulls"]),
            "semantic_authority": False,
        }
    return runtime._seal(unsigned)  # noqa: SLF001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-receipt", required=True)
    parser.add_argument("--expected-run-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validated = validate_run_artifacts(
        run_receipt_path=args.run_receipt,
        expected_run_receipt_file_sha256=args.expected_run_receipt_sha256,
        artifact_root=args.artifact_root,
    )
    receipt = build_postflight_receipt(validated)
    output = Path(args.output)
    if output.name != POSTFLIGHT_FILENAME or output.parent.resolve(strict=True) != Path(
        validated["artifact_root"]
    ):
        raise QMosaicDirectionPostflightError("postflight output path differs")
    try:
        runtime._write_create_only_json(output, receipt)  # noqa: SLF001
    except runtime.QMosaicEditorDirectionError as error:
        raise QMosaicDirectionPostflightError(str(error)) from error
    print(runtime._canonical_json_bytes(receipt).decode("ascii"), flush=True)  # noqa: SLF001
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "POSTFLIGHT_FILENAME",
    "POSTFLIGHT_SCHEMA",
    "QMosaicDirectionPostflightError",
    "build_parser",
    "build_postflight_receipt",
    "main",
    "validate_run_artifacts",
]
