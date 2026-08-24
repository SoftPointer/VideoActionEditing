#!/usr/bin/env python3
"""Render or audit PAIR-v5's frozen, source-free native-T2V bank.

The pinned Bernini entry point calls its geometry probe a ``source-video``.
For the T2V-only arm that video is decoded only to derive an exact-81 spatial
bucket: no source VAE latent or reference is created or passed to the
transformer.  This wrapper verifies that fact from the native receipt.

Each action-case cell contains ``action`` plus the nine MACE hard negatives.
The final bank receipt is emitted only when all ten branches have byte-equal
official native Gaussian *tensor values*.  Every safetensors container is
individually hash-verified, but container hashes need not match: safetensors
metadata/header serialization is not the random variable consumed by the
sampler.  Generated media is calibration evidence only; generation alone
never event-qualifies an action or authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Any, Iterable, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
_PREPROCESSING_TOOL_SHA256 = {
    "tools.build_renderer_dataset": (
        "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
    ),
    "tools.materialize_vae": (
        "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
    ),
}


def _file_sha256_early(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_release_preprocessing_tools(method_root: Path) -> Mapping[str, str]:
    """Authenticate lazy VAE helpers before importing legacy model code."""

    try:
        root = method_root.resolve(strict=True)
        tools_root = (root / "tools").resolve(strict=True)
    except OSError as error:
        raise RuntimeError("release preprocessing-tools root is unavailable") from error
    if (
        not method_root.is_absolute()
        or root != method_root
        or method_root.is_symlink()
        or not method_root.is_dir()
        or tools_root != root / "tools"
        or (root / "tools").is_symlink()
        or not tools_root.is_dir()
    ):
        raise RuntimeError("release preprocessing-tools root is not canonical")
    expected = {
        "tools.build_renderer_dataset": tools_root / "build_renderer_dataset.py",
        "tools.materialize_vae": tools_root / "materialize_vae.py",
    }
    for label, path in expected.items():
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
            raise RuntimeError(f"release preprocessing member differs: {label}")

    package = sys.modules.get("tools")
    if package is None:
        package = types.ModuleType("tools")
        package.__package__ = "tools"
        package.__path__ = [str(tools_root)]
        specification = importlib.machinery.ModuleSpec(
            "tools", loader=None, is_package=True
        )
        specification.submodule_search_locations = [str(tools_root)]
        package.__spec__ = specification
        sys.modules["tools"] = package
    else:
        try:
            locations = tuple(
                Path(value).resolve(strict=True)
                for value in getattr(package, "__path__", ())
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError("preloaded tools package is ambiguous") from error
        if locations != (tools_root,):
            raise RuntimeError("preloaded tools package is outside this release")

    identities: dict[str, str] = {}
    for label, expected_path in expected.items():
        try:
            module = importlib.import_module(label)
            origin = Path(module.__file__).resolve(strict=True)
        except (AttributeError, ImportError, OSError, TypeError) as error:
            raise RuntimeError(
                f"cannot import exact release preprocessing member: {label}"
            ) from error
        identity = _file_sha256_early(origin)
        if origin != expected_path or identity != _PREPROCESSING_TOOL_SHA256[label]:
            raise RuntimeError(f"release preprocessing identity differs: {label}")
        identities[label] = identity
    return identities


RELEASE_PREPROCESSING_TOOL_IDENTITIES = bind_release_preprocessing_tools(METHOD_ROOT)

import infer_native_identity_generation_canary as native  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as contract  # noqa: E402


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PAIR_RECEIPT_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "candidate_envelope_sha256",
    "group_id",
    "visible_gpus",
    "runtime_topology",
    "ordinal",
    "candidate",
    "sampling_contract",
    "semantic_input_closure",
    "artifact_use_contract",
    "split_contract",
    "geometry_use_certificate",
    "native_receipt_path",
    "native_receipt_sha256",
    "native_receipt_digest",
    "artifacts",
    "interpretation",
    "receipt_digest",
}


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise contract.PairT2VCalibrationSpecError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise contract.PairT2VCalibrationSpecError(f"{label} is absent or not plain")
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                contract.PairT2VCalibrationSpecError(
                    f"non-finite JSON constant is forbidden: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise contract.PairT2VCalibrationSpecError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise contract.PairT2VCalibrationSpecError(f"{label} must be an object")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise contract.PairT2VCalibrationSpecError(f"{label} must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = _require_mapping(value, label)
    path_text = artifact.get("path")
    expected = artifact.get("sha256")
    if (
        not isinstance(path_text, str)
        or not isinstance(expected, str)
        or _SHA256_RE.fullmatch(expected) is None
    ):
        raise contract.PairT2VCalibrationSpecError(f"{label} identity differs")
    path = Path(path_text)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise contract.PairT2VCalibrationSpecError(f"{label} is absent or not plain")
    if _file_sha256(path) != expected:
        raise contract.PairT2VCalibrationSpecError(f"{label} SHA-256 differs")
    return dict(artifact)


def _verify_embedded_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or contract.sha256_bytes(contract.canonical_json_bytes(unsigned)) != declared
    ):
        raise contract.PairT2VCalibrationSpecError(f"{label} digest differs")
    return declared


def _verify_native_receipt(
    native_receipt: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    declared_digest = _verify_embedded_digest(native_receipt, label="native receipt")
    if (
        native_receipt.get("schema_version") != native.SCHEMA_VERSION
        or native_receipt.get("method") != native.METHOD
        or native_receipt.get("arms") != ["t2v"]
    ):
        raise contract.PairT2VCalibrationSpecError(
            "native runner did not execute the pinned frozen T2V-only arm"
        )
    native_input = _require_mapping(native_receipt.get("input"), "native input")
    if (
        native_input.get("source_video_sha256")
        != candidate["geometry_source_video_sha256"]
        or native_input.get("action_prompt_utf8_sha256")
        != candidate["full_t2v_caption_utf8_sha256"]
        or native_input.get("target_video") is not False
        or native_input.get("external_reference_image_or_video") is not False
        or native_input.get("external_mask_flow_pose_track_trajectory") is not False
        or native_input.get("external_first_frame_anchor") is not False
    ):
        raise contract.PairT2VCalibrationSpecError("native geometry/prompt binding differs")
    preprocessing = _require_mapping(
        native_receipt.get("preprocessing"), "native preprocessing"
    )
    bucket = preprocessing.get("source_derived_bucket_hw")
    if (
        preprocessing.get("frame_count") != 81
        or preprocessing.get("fps") != 25
        or not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
    ):
        raise contract.PairT2VCalibrationSpecError(
            "geometry probe did not produce one exact81 stride-16 bucket"
        )
    conditioning = _require_mapping(
        _require_mapping(native_receipt.get("conditioning"), "conditioning").get("t2v"),
        "T2V conditioning",
    )
    expected_source_ids = {
        "target_source_id": 0,
        "video_source_ids": [],
        "reference_source_ids": [],
        "conditioning_source_count": 0,
        "max_conditioning_source_id": 0,
        "within_pretrained_source_ids_1_through_5": True,
        "source_id_interpolation_required": False,
    }
    if (
        conditioning.get("full_source_video_count") != 0
        or conditioning.get("source_derived_reference_count") != 0
        or conditioning.get("source_frame_indices") != []
        or conditioning.get("reference_encoding") != "none"
        or conditioning.get("source_ids") != expected_source_ids
    ):
        raise contract.PairT2VCalibrationSpecError("source content entered native T2V")
    identities = _require_mapping(
        native_receipt.get("condition_identities"), "condition identities"
    )
    if (
        identities.get("references") != {}
        or identities.get("full_source_video") is not None
        or identities.get("rank_zero_broadcasts")
        != {"references": {}, "full_source_video": None}
        or native_receipt.get("source_condition_artifact") is not None
    ):
        raise contract.PairT2VCalibrationSpecError(
            "native T2V created or consumed a source latent/reference"
        )
    sampling = _require_mapping(
        _require_mapping(native_receipt.get("sampling"), "sampling").get("t2v"),
        "T2V sampling",
    )
    guidance = contract.SAMPLING_CONTRACT["guidance"]
    if (
        sampling.get("num_frames") != 81
        or sampling.get("num_inference_steps") != 40
        or sampling.get("guidance_mode") != "t2v_apg"
        or sampling.get("seed") != candidate["seed"]
        or sampling.get("omega_txt") != guidance["omega_txt"]
        or sampling.get("omega_vid") != guidance["omega_vid"]
        or sampling.get("omega_img") != guidance["omega_img"]
        or sampling.get("target_initialization") != contract.TARGET_INITIALIZATION
        or sampling.get("target_mixed_with_source_latent") is not False
        or sampling.get("custom_sampler_or_scheduler") is not False
        or sampling.get("ulysses_size") != 4
    ):
        raise contract.PairT2VCalibrationSpecError("native T2V sampling differs")
    lifecycle = _require_mapping(
        native_receipt.get("resource_lifecycle"), "native resource lifecycle"
    )
    try:
        lifecycle = native.validate_t2v_resource_lifecycle(
            lifecycle, require_serialized_load=True
        )
    except native.NativeIdentityCanaryError as error:
        raise contract.PairT2VCalibrationSpecError(
            "native T2V host/model/VAE resource lifecycle differs"
        ) from error
    geometry = _require_mapping(native_receipt.get("latent_geometry"), "latent geometry")
    expected_shape = [1, 16, 21, int(bucket[0]) // 8, int(bucket[1]) // 8]
    if geometry.get("video_latent_shape") != expected_shape:
        raise contract.PairT2VCalibrationSpecError("native latent/bucket geometry differs")
    output = _require_mapping(
        _require_mapping(native_receipt.get("outputs"), "outputs").get("t2v"),
        "T2V output",
    )
    clean = _require_mapping(output.get("normalized_clean_latent"), "clean latent")
    gaussian = _require_mapping(
        _require_mapping(
            native_receipt.get("initial_noise_artifacts"), "initial noise artifacts"
        ).get("t2v"),
        "official Gaussian",
    )
    if (
        output.get("frame_count") != 81
        or output.get("fps") != 25
        or output.get("height") != bucket[0]
        or output.get("width") != bucket[1]
        or clean.get("shape") != expected_shape
        or clean.get("native_sampler_before_vae_decode") is not True
        or clean.get("mp4_decode_reencode_used") is not False
    ):
        raise contract.PairT2VCalibrationSpecError(
            "clean latent is not exact81 native predecode state"
        )
    if (
        gaussian.get("shape") != expected_shape
        or gaussian.get("generator_initial_seed") != candidate["seed"]
        or gaussian.get("captured_from_native_sampler") is not True
        or gaussian.get("external_initial_noise_injection") is not False
        or gaussian.get("source_or_target_derived") is not False
        or gaussian.get("observer_changed_return_value") is not False
        or gaussian.get("official_randn_tensor_call_count") != 1
        or _SHA256_RE.fullmatch(str(gaussian.get("raw_value_sha256"))) is None
        or _SHA256_RE.fullmatch(str(gaussian.get("content_sha256"))) is None
    ):
        raise contract.PairT2VCalibrationSpecError("official Gaussian provenance differs")
    interpretation = _require_mapping(
        native_receipt.get("interpretation"), "native interpretation"
    )
    if interpretation.get("training_performed") is not False:
        raise contract.PairT2VCalibrationSpecError("native generation performed training")
    return {
        "native_receipt_digest": declared_digest,
        "bucket_hw": list(bucket),
        "latent_shape": expected_shape,
        "mp4": dict(output),
        "predecode_clean_latent": dict(clean),
        "official_initial_gaussian": dict(gaussian),
        "resource_lifecycle": dict(lifecycle),
    }


def bind_receipt(args: argparse.Namespace, envelope: Mapping[str, Any]) -> Path:
    output = Path(args.output_dir)
    native_path = output / "receipt.json"
    native_receipt = _load_json(native_path, "native receipt")
    candidate = envelope["candidate"]
    artifacts = _verify_native_receipt(native_receipt, candidate)
    mp4 = _verify_file_artifact(artifacts["mp4"], "T2V MP4")
    clean = _verify_file_artifact(
        artifacts["predecode_clean_latent"], "T2V predecode clean latent"
    )
    gaussian = _verify_file_artifact(
        artifacts["official_initial_gaussian"], "official initial Gaussian"
    )
    native_digest = artifacts["native_receipt_digest"]
    bucket = artifacts["bucket_hw"]
    latent_shape = artifacts["latent_shape"]
    receipt = {
        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
        "root_spec_raw_sha256": envelope["root_spec_raw_sha256"],
        "candidate_envelope_sha256": envelope["candidate_envelope_sha256"],
        "group_id": envelope["group_id"],
        "visible_gpus": envelope["visible_gpus"],
        "runtime_topology": {
            "world_size": 4,
            "ulysses_size": 4,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "ordinal": envelope["ordinal"],
        "candidate": candidate,
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
        "split_contract": contract.SPLIT_CONTRACT,
        "geometry_use_certificate": {
            "video_sha256": candidate["geometry_source_video_sha256"],
            "bucket_hw": bucket,
            "latent_shape": latent_shape,
            "used_to_derive_bucket_shape": True,
            "vae_latent_created": False,
            "pixels_entered_transformer": False,
            "content_conditioning_count": 0,
        },
        "native_receipt_path": str(native_path),
        "native_receipt_sha256": _file_sha256(native_path),
        "native_receipt_digest": native_digest,
        "artifacts": {
            "mp4": mp4,
            "predecode_clean_latent": clean,
            "official_initial_gaussian": gaussian,
        },
        "interpretation": {
            "calibration_evidence_only": True,
            "event_qualified_from_generation_receipt": False,
            "action_success_not_implied": True,
            "training_performed": False,
            "parameter_update_performed": False,
            "optimizer_authorized": False,
            "t2v_media_as_rv2v_policy_candidate_forbidden": True,
            "donor_or_pseudo_target_use_forbidden": True,
        },
    }
    receipt["receipt_digest"] = contract.sha256_bytes(
        contract.canonical_json_bytes(receipt)
    )
    receipt_path = output / "pair-v5-t2v-calibration-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise contract.PairT2VCalibrationSpecError("refusing to overwrite PAIR receipt")
    receipt_path.write_bytes(contract.canonical_json_bytes(receipt) + b"\n")
    os.chmod(receipt_path, 0o400)
    return receipt_path


def _load_pair_receipt(path: Path) -> dict[str, Any]:
    receipt = dict(_load_json(path, "PAIR T2V calibration receipt"))
    if set(receipt) != _PAIR_RECEIPT_FIELDS:
        raise contract.PairT2VCalibrationSpecError("PAIR receipt field closure differs")
    if receipt.get("schema_version") != contract.RECEIPT_SCHEMA_VERSION:
        raise contract.PairT2VCalibrationSpecError("PAIR receipt schema differs")
    _verify_embedded_digest(receipt, label="PAIR receipt")
    return receipt


def audit_rendered_bank(
    *, root_spec: str | Path, expected_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    spec, root_digest = contract.load_sealed_spec(root_spec, expected_sha256)
    root = Path(output_dir)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise contract.PairT2VCalibrationSpecError(
            "bank output must be an absolute plain directory"
        )
    candidate_rows: list[dict[str, Any]] = []
    cells: dict[tuple[str, str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    split_groups = {
        split: {axis: set() for axis in contract.SPLIT_GROUP_AXES}
        for split in contract.ANALYSIS_SPLITS
    }
    for group in spec["groups"]:
        expected_visible = ",".join(str(item) for item in group["visible_gpus"])
        for ordinal, candidate in enumerate(group["candidates"]):
            receipt_path = (
                root / candidate["candidate_id"] / "pair-v5-t2v-calibration-receipt.json"
            )
            receipt = _load_pair_receipt(receipt_path)
            if (
                receipt["root_spec_raw_sha256"] != root_digest
                or receipt["candidate"] != candidate
                or receipt["group_id"] != group["group_id"]
                or receipt["visible_gpus"] != group["visible_gpus"]
                or receipt["ordinal"] != ordinal
                or receipt["runtime_topology"]
                != {
                    "world_size": 4,
                    "ulysses_size": 4,
                    "rocr_visible_devices": expected_visible,
                }
                or receipt["sampling_contract"] != contract.SAMPLING_CONTRACT
                or receipt["semantic_input_closure"] != contract.SEMANTIC_INPUT_CLOSURE
                or receipt["artifact_use_contract"] != contract.ARTIFACT_USE_CONTRACT
                or receipt["split_contract"] != contract.SPLIT_CONTRACT
            ):
                raise contract.PairT2VCalibrationSpecError(
                    "PAIR candidate receipt/spec binding differs"
                )
            expected_interpretation = {
                "calibration_evidence_only": True,
                "event_qualified_from_generation_receipt": False,
                "action_success_not_implied": True,
                "training_performed": False,
                "parameter_update_performed": False,
                "optimizer_authorized": False,
                "t2v_media_as_rv2v_policy_candidate_forbidden": True,
                "donor_or_pseudo_target_use_forbidden": True,
            }
            if receipt["interpretation"] != expected_interpretation:
                raise contract.PairT2VCalibrationSpecError(
                    "generation receipt exceeds calibration-only authority"
                )
            native_path = Path(receipt["native_receipt_path"])
            native_receipt = _load_json(native_path, "receipt-bound native receipt")
            if _file_sha256(native_path) != receipt["native_receipt_sha256"]:
                raise contract.PairT2VCalibrationSpecError("native receipt file hash differs")
            native_artifacts = _verify_native_receipt(native_receipt, candidate)
            if native_artifacts["native_receipt_digest"] != receipt["native_receipt_digest"]:
                raise contract.PairT2VCalibrationSpecError("native receipt digest binding differs")
            expected_artifacts = {
                "mp4": native_artifacts["mp4"],
                "predecode_clean_latent": native_artifacts["predecode_clean_latent"],
                "official_initial_gaussian": native_artifacts[
                    "official_initial_gaussian"
                ],
            }
            if receipt["artifacts"] != expected_artifacts:
                raise contract.PairT2VCalibrationSpecError(
                    "PAIR artifact declarations differ from the bound native receipt"
                )
            geometry_certificate = receipt["geometry_use_certificate"]
            if (
                not isinstance(geometry_certificate, Mapping)
                or geometry_certificate.get("bucket_hw") != native_artifacts["bucket_hw"]
                or geometry_certificate.get("latent_shape")
                != native_artifacts["latent_shape"]
                or geometry_certificate.get("video_sha256")
                != candidate["geometry_source_video_sha256"]
                or geometry_certificate.get("used_to_derive_bucket_shape") is not True
                or geometry_certificate.get("vae_latent_created") is not False
                or geometry_certificate.get("pixels_entered_transformer") is not False
                or geometry_certificate.get("content_conditioning_count") != 0
            ):
                raise contract.PairT2VCalibrationSpecError(
                    "PAIR geometry-use certificate differs from native evidence"
                )
            for label, artifact in receipt["artifacts"].items():
                _verify_file_artifact(
                    artifact, f"{candidate['candidate_id']} {label}"
                )
            cell_key = (
                candidate["analysis_split"],
                candidate["action_family_id"],
                candidate["calibration_group_id"],
            )
            cells.setdefault(cell_key, []).append((candidate, receipt))
            for axis in contract.SPLIT_GROUP_AXES:
                split_groups[candidate["analysis_split"]][axis].add(candidate[axis])
            candidate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "analysis_split": candidate["analysis_split"],
                    "action_family_id": candidate["action_family_id"],
                    "calibration_group_id": candidate["calibration_group_id"],
                    "semantic_branch": candidate["semantic_branch"],
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": _file_sha256(receipt_path),
                    "receipt_digest": receipt["receipt_digest"],
                    "mp4_sha256": receipt["artifacts"]["mp4"]["sha256"],
                    "predecode_clean_latent_sha256": receipt["artifacts"][
                        "predecode_clean_latent"
                    ]["sha256"],
                    "official_initial_gaussian_sha256": receipt["artifacts"][
                        "official_initial_gaussian"
                    ]["sha256"],
                }
            )
    cell_proofs = []
    for cell_key, rows in cells.items():
        branches = [candidate["semantic_branch"] for candidate, _ in rows]
        if branches != list(contract.MACE_BRANCH_ORDER):
            raise contract.PairT2VCalibrationSpecError(
                f"rendered cell {cell_key!r} lost exact MACE branch order"
            )
        gaussians = [
            receipt["artifacts"]["official_initial_gaussian"]
            for _, receipt in rows
        ]
        # The native observer independently serializes one provenance
        # container per branch.  Container bytes are verified above, but are
        # not the sampled Gaussian: identical tensor bytes can legally have
        # different safetensors headers.  Cross-branch equality is therefore
        # proved from the tensor-value/content identities plus geometry,
        # dtype, and seed -- never from the container-file SHA.
        proof_fields = (
            "raw_value_sha256",
            "content_sha256",
            "shape",
            "dtype",
            "stored_dtype",
            "generator_initial_seed",
        )
        identities = {
            contract.sha256_bytes(
                contract.canonical_json_bytes(
                    {field: gaussian.get(field) for field in proof_fields}
                )
            )
            for gaussian in gaussians
        }
        if len(identities) != 1:
            raise contract.PairT2VCalibrationSpecError(
                f"rendered cell {cell_key!r} did not reuse one exact official Gaussian"
            )
        first = gaussians[0]
        cell_proofs.append(
            {
                "analysis_split": cell_key[0],
                "action_family_id": cell_key[1],
                "calibration_group_id": cell_key[2],
                "semantic_branch_count": len(rows),
                "semantic_branch_order": branches,
                "all_ten_official_gaussian_tensor_values_byte_equal": True,
                "all_container_files_individually_sha256_verified": True,
                "official_gaussian_file_sha256_by_branch": {
                    candidate["semantic_branch"]: receipt["artifacts"][
                        "official_initial_gaussian"
                    ]["sha256"]
                    for candidate, receipt in rows
                },
                "official_gaussian_raw_value_sha256": first["raw_value_sha256"],
                "official_gaussian_content_sha256": first["content_sha256"],
                "seed": first["generator_initial_seed"],
            }
        )
    split_membership = {
        split: {axis: sorted(values) for axis, values in axes.items()}
        for split, axes in split_groups.items()
    }
    for axis in contract.SPLIT_GROUP_AXES:
        if set(split_membership["fit"][axis]) & set(
            split_membership["confirmation"][axis]
        ):
            raise contract.PairT2VCalibrationSpecError(
                f"rendered bank lost fit/confirmation isolation on {axis}"
            )
    bank = {
        "schema_version": contract.BANK_RECEIPT_SCHEMA_VERSION,
        "root_spec_raw_sha256": root_digest,
        "candidate_count": len(candidate_rows),
        "cell_count": len(cell_proofs),
        "mace_branch_order": list(contract.MACE_BRANCH_ORDER),
        "sampling_contract": contract.SAMPLING_CONTRACT,
        "semantic_input_closure": contract.SEMANTIC_INPUT_CLOSURE,
        "artifact_use_contract": contract.ARTIFACT_USE_CONTRACT,
        "split_contract": contract.SPLIT_CONTRACT,
        "split_group_membership": split_membership,
        "fit_confirmation_all_registered_axes_disjoint": True,
        "same_cell_gaussian_proofs": cell_proofs,
        "candidate_receipts": candidate_rows,
        "interpretation": {
            "calibration_evidence_only": True,
            "event_qualification_performed": False,
            "action_success_not_implied": True,
            "training_performed": False,
            "parameter_update_performed": False,
            "optimizer_authorized": False,
            "t2v_negative_media_are_rv2v_policy_candidates": False,
            "t2v_media_as_condition_target_donor_or_noise_forbidden": True,
        },
    }
    bank["receipt_digest"] = contract.sha256_bytes(contract.canonical_json_bytes(bank))
    bank_path = root / "pair-v5-t2v-calibration-bank-receipt.json"
    if bank_path.exists() or bank_path.is_symlink():
        raise contract.PairT2VCalibrationSpecError("refusing to overwrite bank receipt")
    bank_path.write_bytes(contract.canonical_json_bytes(bank) + b"\n")
    os.chmod(bank_path, 0o400)
    return bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-bank", action="store_true")
    parser.add_argument("--candidate-spec")
    parser.add_argument("--root-spec")
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bernini-root")
    parser.add_argument("--veomni-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-content-manifest")
    parser.add_argument("--method-source-revision")
    parser.add_argument("--method-source-archive-sha256")
    return parser


def _render(args: argparse.Namespace) -> int:
    required = (
        "candidate_spec",
        "bernini_root",
        "veomni_root",
        "checkpoint",
        "checkpoint_content_manifest",
        "method_source_revision",
        "method_source_archive_sha256",
    )
    if any(not getattr(args, name) for name in required) or args.root_spec is not None:
        raise contract.PairT2VCalibrationSpecError(
            "render mode requires only the candidate and frozen-runtime arguments"
        )
    envelope = contract.load_candidate_envelope(
        args.candidate_spec, args.expected_root_spec_sha256
    )
    expected_visible = ",".join(str(value) for value in envelope["visible_gpus"])
    if os.environ.get("ROCR_VISIBLE_DEVICES") != expected_visible:
        raise contract.PairT2VCalibrationSpecError(
            "runtime ROCR_VISIBLE_DEVICES differs from sealed SP4 assignment"
        )
    if os.environ.get("WORLD_SIZE") not in (None, "4"):
        raise contract.PairT2VCalibrationSpecError("runtime world size is not SP4")
    candidate = envelope["candidate"]
    guidance = contract.SAMPLING_CONTRACT["guidance"]
    native.OMEGA_TEXT = guidance["omega_txt"]
    native.OMEGA_VIDEO = guidance["omega_vid"]
    native.OMEGA_IMAGE = guidance["omega_img"]
    native_argv = [
        "--bernini-root", args.bernini_root,
        "--veomni-root", args.veomni_root,
        "--checkpoint", args.checkpoint,
        "--checkpoint-content-manifest", args.checkpoint_content_manifest,
        # Geometry only: the receipt must prove no VAE source/reference state.
        "--source-video", candidate["geometry_source_video"],
        "--expected-source-sha256", candidate["geometry_source_video_sha256"],
        "--action-prompt", candidate["full_t2v_caption"],
        "--expected-action-prompt-sha256", candidate["full_t2v_caption_utf8_sha256"],
        "--output-dir", args.output_dir,
        "--arms", "t2v",
        "--num-inference-steps", "40",
        "--seed", str(candidate["seed"]),
        "--method-source-revision", args.method_source_revision,
        "--method-source-archive-sha256", args.method_source_archive_sha256,
    ]
    status = native.main(native_argv)
    if status == 0 and int(os.environ.get("RANK", "0")) == 0:
        bind_receipt(args, envelope)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.audit_bank:
        if args.root_spec is None or args.candidate_spec is not None or any(
            getattr(args, name) is not None
            for name in (
                "bernini_root",
                "veomni_root",
                "checkpoint",
                "checkpoint_content_manifest",
                "method_source_revision",
                "method_source_archive_sha256",
            )
        ):
            raise contract.PairT2VCalibrationSpecError(
                "audit mode accepts only root spec, root digest, and output directory"
            )
        receipt = audit_rendered_bank(
            root_spec=args.root_spec,
            expected_sha256=args.expected_root_spec_sha256,
            output_dir=args.output_dir,
        )
        print(contract.canonical_json_bytes(receipt).decode("utf-8"), flush=True)
        return 0
    return _render(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_rendered_bank",
    "bind_receipt",
    "main",
]
