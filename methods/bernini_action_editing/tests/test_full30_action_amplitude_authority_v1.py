#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for value in (METHOD_ROOT, TEST_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import full30_action_amplitude_authority_v1 as amplitude
import full30_action_data_teacher_authority_v1 as parent_authority
from test_full30_action_data_teacher_authority_v1 import (
    _build_manifest,
    _materialization_condition_fixture,
    _materialization_output_policy,
    _materialization_state_and_forwards,
    _resign_nested,
    _write_fp32_artifact,
    _write_json,
)
import full30_action_psiout_materializer_v1 as materializer


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path.resolve()), hashlib.sha256(payload).hexdigest()


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    return amplitude.seal_record(value, field)


def _resign(value: dict[str, object], field: str) -> None:
    value[field] = amplitude.object_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def _basis(index: int, scale: float) -> tuple[float, ...]:
    values = [0.0] * amplitude.TENSOR_ELEMENTS
    values[index] = scale
    return tuple(values)


def _write_container(
    root: Path,
    relative: str,
    *,
    evidence_id: str,
    pair_id: str,
    source_iid: str,
    teacher_cell_id: str,
    branch: str,
    scale_offset: float,
) -> tuple[str, str, dict[int, tuple[bytes, float]]]:
    payload_parts: list[bytes] = []
    entries: list[dict[str, object]] = []
    slices: dict[int, tuple[bytes, float]] = {}
    for ordinal, sigma_index in enumerate(amplitude.SIGMA_INDICES):
        norm = scale_offset + 0.01 * (ordinal + 1)
        tensor_bytes = struct.pack(
            f"<{amplitude.TENSOR_ELEMENTS}f", *_basis(ordinal, norm)
        )
        actual_norm = math.sqrt(
            math.fsum(
                float(value) * float(value)
                for value in struct.unpack(
                    f"<{amplitude.TENSOR_ELEMENTS}f", tensor_bytes
                )
            )
        )
        slices[sigma_index] = (tensor_bytes, actual_norm)
        payload_parts.append(tensor_bytes)
        entries.append(
            {
                "name": amplitude._tensor_name(sigma_index),
                "sigma_index": sigma_index,
                "dtype": amplitude.TENSOR_DTYPE,
                "shape": list(amplitude.TENSOR_SHAPE),
                "offset": ordinal * amplitude.TENSOR_SLICE_BYTES,
                "length": amplitude.TENSOR_SLICE_BYTES,
                "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
            }
        )
    payload = b"".join(payload_parts)
    header = {
        "schema_version": amplitude.CONTAINER_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "pair_id": pair_id,
        "source_iid": source_iid,
        "teacher_cell_id": teacher_cell_id,
        "branch": branch,
        "dtype": amplitude.TENSOR_DTYPE,
        "shape": list(amplitude.TENSOR_SHAPE),
        "sigma_indices": list(amplitude.SIGMA_INDICES),
        "layout": amplitude.CONTAINER_LAYOUT,
        "tensor_count": len(amplitude.SIGMA_INDICES),
        "payload_bytes": len(payload),
        "entries": entries,
    }
    header_bytes = amplitude.canonical_json_bytes(header)
    raw = (
        amplitude.CONTAINER_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + payload
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(amplitude.CONTAINER_MODE)
    return str(path.resolve()), hashlib.sha256(raw).hexdigest(), slices


def _rewrite_container_tensor(
    path_value: object, *, sigma_index: int, values: tuple[float, ...]
) -> str:
    path = Path(path_value)  # type: ignore[arg-type]
    raw = path.read_bytes()
    prefix = len(amplitude.CONTAINER_MAGIC) + 4
    header_length = struct.unpack(">I", raw[len(amplitude.CONTAINER_MAGIC):prefix])[0]
    header = json.loads(raw[prefix:prefix + header_length])
    payload = bytearray(raw[prefix + header_length:])
    entry = next(row for row in header["entries"] if row["sigma_index"] == sigma_index)
    tensor_bytes = struct.pack(f"<{amplitude.TENSOR_ELEMENTS}f", *values)
    start = entry["offset"]
    payload[start:start + entry["length"]] = tensor_bytes
    entry["sha256"] = hashlib.sha256(tensor_bytes).hexdigest()
    header_bytes = amplitude.canonical_json_bytes(header)
    rewritten = (
        amplitude.CONTAINER_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + bytes(payload)
    )
    path.write_bytes(rewritten)
    path.chmod(amplitude.CONTAINER_MODE)
    return hashlib.sha256(rewritten).hexdigest()


def _review(
    *,
    evidence_id: str,
    pair: dict[str, object],
    output_sha256: str,
    action_result: str,
) -> dict[str, object]:
    return _seal(
        {
            "schema_version": amplitude.REVIEW_SCHEMA_VERSION,
            "review_id": f"amplitude-review:{evidence_id}",
            "evidence_id": evidence_id,
            "pair_id": pair["pair_id"],
            "source_iid": pair["source_iid"],
            "branch": pair["branch"],
            "baseline_output_sha256": output_sha256,
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
            "entire_full81_video_viewed": True,
            "independent_reviewer": True,
            "reviewer_blinded_to_amplitude_metrics": True,
            "sealed_before_sidecar_extraction": True,
            "sealed_before_optimizer_authority": True,
            "action_result": action_result,
        },
        "review_digest",
    )


def _evidence(
    root: Path,
    *,
    role: str,
    pair: dict[str, object],
    source: dict[str, object],
    ordinal: int,
) -> tuple[dict[str, object], dict[int, tuple[bytes, float]] | None]:
    evidence_id = f"amplitude:{role}:{pair['pair_id']}"
    output_path, output_sha = _write(
        root,
        f"baseline/{evidence_id.replace(':', '_')}.mp4",
        f"frozen-output-{evidence_id}".encode("utf-8"),
    )
    action_result = "partial" if role == "calibrator" else "fail"
    unsigned: dict[str, object] = {
        "schema_version": amplitude.EVIDENCE_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "evidence_role": role,
        "teacher_cell_id": pair["teacher_cell_id"],
        "branch": pair["branch"],
        "pair_id": pair["pair_id"],
        "source_iid": pair["source_iid"],
        "source_posterior_index0_sha256": source[
            "source_posterior_index0_sha256"
        ],
        "instruction_utf8_sha256": pair["instruction_utf8_sha256"],
        "baseline_output_path": output_path,
        "baseline_output_sha256": output_sha,
        "initial_gaussian_sha256": hashlib.sha256(
            f"initial-gaussian-{pair['pair_id']}".encode("utf-8")
        ).hexdigest(),
        "same_source_noise_sigma_state": True,
        "official_frozen_native_only": True,
        "pre_admission_review": _review(
            evidence_id=evidence_id,
            pair=pair,
            output_sha256=output_sha,
            action_result=action_result,
        ),
    }
    slices = None
    if role == "calibrator":
        container_path, container_sha, slices = _write_container(
            root,
            f"amplitude/{evidence_id.replace(':', '_')}.f30ac",
            evidence_id=evidence_id,
            pair_id=str(pair["pair_id"]),
            source_iid=str(pair["source_iid"]),
            teacher_cell_id=str(pair["teacher_cell_id"]),
            branch=str(pair["branch"]),
            scale_offset=0.04 + ordinal * 0.01,
        )
        unsigned.update(
            {
                "calibrator_noise_seed": amplitude.calibrator_noise_seed_v1(
                    str(pair["pair_id"])
                ),
                "calibrator_noise_sha256": hashlib.sha256(
                    f"calibrator-noise-{pair['pair_id']}".encode("utf-8")
                ).hexdigest(),
                "amplitude_container_path": container_path,
                "amplitude_container_sha256": container_sha,
            }
        )
    return _seal(unsigned, "evidence_digest"), slices


def _attach_amplitude_materialization_provenance(
    root: Path,
    *,
    parent: dict[str, object],
    bundles: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    parent_binding = parent["materialization_run_receipt"]
    parent_run = json.loads(Path(parent_binding["path"]).read_bytes())
    runtime_identity = copy.deepcopy(parent_run["runtime_identity"])
    runtime_plan = copy.deepcopy(parent_run["plan_authority"]["runtime"])
    sigma_authority = copy.deepcopy(parent_run["sigma_authority"])
    fit_teacher_records = [
        copy.deepcopy(row)
        for row in parent_run["plan_authority"]["records"]
        if row["record_kind"] == "teacher_anchor"
        and str(row["teacher_cell_id"]).startswith("fit-")
    ]
    parent_receipts = {
        reference["record_id"]: json.loads(Path(reference["path"]).read_bytes())
        for reference in parent_run["record_receipts"]
    }
    parent_representation_fragments = [
        copy.deepcopy(fragment)
        for fragment in parent_run["representation_sigma_evidence_candidates"]
        if str(fragment["teacher_cell_id"]).startswith("fit-")
    ]
    sources = {str(row["source_iid"]): row for row in parent["sources"]}  # type: ignore[index]
    pairs = {str(row["pair_id"]): row for row in parent["pairs"]}  # type: ignore[index]
    representations = {
        (str(row["teacher_cell_id"]), str(row["branch"])): row
        for row in parent["representation_admissions"]  # type: ignore[index]
        if row["analysis_split"] == "fit"
    }

    amplitude_records: list[dict[str, object]] = []
    amplitude_candidates: dict[str, dict[str, object]] = {}
    for bundle in bundles:
        for evidence in bundle["calibrator_evidence"]:  # type: ignore[index]
            pair = pairs[str(evidence["pair_id"])]
            source = sources[str(evidence["source_iid"])]
            record_id = f"materialize:{evidence['evidence_id']}"
            slug = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
            clean = _write_fp32_artifact(
                root,
                f"materialization-amplitude-latent/{slug}-clean.safetensors",
                tensor_key="latent",
                seed_value=2.0 + len(amplitude_records) * 0.01,
            )
            noise = _write_fp32_artifact(
                root,
                f"materialization-amplitude-latent/{slug}-noise.safetensors",
                tensor_key="noise",
                seed_value=5.0 + len(amplitude_records) * 0.01,
            )
            evidence["initial_gaussian_sha256"] = noise["tensor_raw_sha256"]
            evidence["calibrator_noise_sha256"] = noise["tensor_raw_sha256"]
            _resign(evidence, "evidence_digest")
            candidate = copy.deepcopy(evidence)
            amplitude_candidates[str(evidence["evidence_id"])] = candidate
            review_path, review_sha = _write_json(
                root,
                f"materialization-amplitude-review/{slug}.json",
                evidence["pre_admission_review"],
            )
            conditions = [
                _materialization_condition_fixture(
                    root,
                    record_id=record_id,
                    role="branch",
                    control_anchor_id=None,
                    instruction_override=str(pair["instruction"]),
                ),
                _materialization_condition_fixture(
                    root,
                    record_id=record_id,
                    role="noop",
                    control_anchor_id=None,
                    instruction_override=materializer.EXACT_NOOP_INSTRUCTION,
                ),
            ]
            record = _seal(
                {
                    "schema_version": parent_authority._MATERIALIZATION_PLAN_RECORD_SCHEMA,
                    "record_id": record_id,
                    "record_kind": "amplitude_calibrator",
                    "evidence_id": evidence["evidence_id"],
                    "evidence_role": "calibrator",
                    "teacher_cell_id": evidence["teacher_cell_id"],
                    "analysis_split": "fit",
                    "branch": evidence["branch"],
                    "event_id": pair["event_id"],
                    "actor_kind": pair["actor_kind"],
                    "q0_id": pair["q0_id"],
                    "actor_id": source["actor_id"],
                    "scene_id": source["scene_id"],
                    "anchor_id": None,
                    "anchor_iid": None,
                    "pair_id": evidence["pair_id"],
                    "source_iid": evidence["source_iid"],
                    "review": {
                        "schema_version": parent_authority._MATERIALIZATION_REVIEW_BINDING_SCHEMA,
                        "path": review_path,
                        "file_sha256": review_sha,
                        "review_digest": evidence["pre_admission_review"][
                            "review_digest"
                        ],
                    },
                    "reviewed_media": {
                        "path": evidence["baseline_output_path"],
                        "file_sha256": evidence["baseline_output_sha256"],
                    },
                    "target_clean_latent": clean,
                    "target_clean_latent_authority": None,
                    "source_clean_latent": copy.deepcopy(clean),
                    "source_posterior_index0_path": source[
                        "source_posterior_index0_path"
                    ],
                    "source_posterior_index0_sha256": source[
                        "source_posterior_index0_sha256"
                    ],
                    "source_posterior_tensor_key": source[
                        "source_posterior_tensor_key"
                    ],
                    "noise": {
                        "artifact": noise,
                        "seed": evidence["calibrator_noise_seed"],
                        "generator": "torch-cpu-generator-manual-seed-randn-fp32-v1",
                    },
                    "conditions": conditions,
                },
                "record_digest",
            )
            amplitude_records.append(record)

    plan_records = fit_teacher_records + amplitude_records
    population = _seal(
        {
            "schema_version": parent_authority._MATERIALIZATION_POPULATION_SCHEMA,
            "population_id": "amplitude-authority-fixture",
            "record_count": len(plan_records),
            "teacher_record_count": len(fit_teacher_records),
            "amplitude_record_count": len(amplitude_records),
            "teacher_cell_ids": sorted(
                {str(row["teacher_cell_id"]) for row in plan_records},
                key=lambda item: item.encode("utf-8"),
            ),
            "record_order_sha256": amplitude.object_sha256(
                [str(row["record_id"]) for row in plan_records]
            ),
            "finite_closed_population": True,
            "block_probe": False,
        },
        "population_digest",
    )
    plan = _seal(
        {
            "schema_version": parent_authority._MATERIALIZATION_PLAN_SCHEMA,
            "plan_id": "amplitude-authority-fixture-plan",
            "status": "SEALED_REVIEWED_PRE_OPTIMIZER",
            "runtime": runtime_plan,
            "population": population,
            "records": plan_records,
            "output_policy": _materialization_output_policy(),
        },
        "plan_digest",
    )

    references: list[dict[str, object]] = []
    receipt_by_evidence: dict[str, tuple[str, str, str]] = {}
    amplitude_receipts: dict[str, dict[str, object]] = {}
    for ordinal, record in enumerate(fit_teacher_records):
        receipt = copy.deepcopy(parent_receipts[str(record["record_id"])])
        receipt["plan_id"] = plan["plan_id"]
        receipt["plan_digest"] = plan["plan_digest"]
        receipt["record_ordinal"] = ordinal
        _resign(receipt, "record_receipt_digest")
        receipt_path, receipt_sha = _write_json(
            root,
            f"materialization-amplitude-receipts/teacher-{ordinal:04d}.json",
            receipt,
            mode=parent_authority.MATERIALIZATION_RECEIPT_MODE,
        )
        references.append(
            {
                "record_id": record["record_id"],
                "record_kind": record["record_kind"],
                "path": receipt_path,
                "file_sha256": receipt_sha,
                "record_receipt_digest": receipt["record_receipt_digest"],
                "candidate_evidence_digest": receipt[
                    "candidate_authority_evidence"
                ]["evidence_digest"],
            }
        )

    for local_ordinal, record in enumerate(amplitude_records):
        ordinal = len(fit_teacher_records) + local_ordinal
        evidence_id = str(record["evidence_id"])
        candidate = amplitude_candidates[evidence_id]
        tensors = amplitude._validate_container(
            candidate["amplitude_container_path"],
            candidate["amplitude_container_sha256"],
            evidence_id=evidence_id,
            pair_id=str(record["pair_id"]),
            source_iid=str(record["source_iid"]),
            teacher_cell_id=str(record["teacher_cell_id"]),
            branch=str(record["branch"]),
            label="fixture amplitude container",
        )
        representation = representations[
            (str(record["teacher_cell_id"]), str(record["branch"]))
        ]
        origin = representation["origin_evidence"]
        nuisance = parent_authority._validate_tensor_container(
            origin["nuisance_packet_path"],
            origin["nuisance_packet_sha256"],
            container_kind="nuisance",
            evidence_id=str(origin["evidence_id"]),
            evidence_role="teacher_origin",
            teacher_cell_id=str(record["teacher_cell_id"]),
            branch=str(record["branch"]),
            label="fixture parent nuisance",
        )
        states, forwards = _materialization_state_and_forwards(
            record=record,
            runtime_digest=str(runtime_identity["runtime_digest"]),
        )
        noise_receipt = _seal(
            {
                "schema_version": parent_authority.MATERIALIZATION_NOISE_RECEIPT_SCHEMA,
                "provider_abi": parent_authority.MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "record_id": record["record_id"],
                "seed": record["noise"]["seed"],
                "generator": record["noise"]["generator"],
                "shape": record["noise"]["artifact"]["shape"],
                "artifact_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "replayed_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "byte_exact_replay": True,
            },
            "noise_digest",
        )
        sigma_metrics = []
        for sigma_ordinal, sigma_index in enumerate(amplitude.SIGMA_INDICES):
            tensor = tensors[sigma_index]
            sigma_metrics.append(
                {
                    "sigma_index": sigma_index,
                    "state_digest": states[sigma_ordinal]["state_digest"],
                    "projected_slice_sha256": tensor.sha256,
                    "amplitude_norm": amplitude._norm(tensor.values),
                    "teacher_nuisance_camera_sha256": nuisance[
                        parent_authority._tensor_name(sigma_index, "camera_unit")
                    ][2],
                    "teacher_nuisance_appearance_sha256": nuisance[
                        parent_authority._tensor_name(
                            sigma_index, "appearance_unit"
                        )
                    ][2],
                }
            )
        receipt = _seal(
            {
                "schema_version": parent_authority.MATERIALIZATION_RECORD_RECEIPT_SCHEMA,
                "plan_id": plan["plan_id"],
                "plan_digest": plan["plan_digest"],
                "runtime_digest": runtime_identity["runtime_digest"],
                "provider_abi": parent_authority.MATERIALIZATION_PROVIDER_ABI,
                "official_provider": True,
                "test_only": False,
                "record_ordinal": ordinal,
                "record_id": record["record_id"],
                "record_digest": record["record_digest"],
                "record_kind": record["record_kind"],
                "evidence_id": record["evidence_id"],
                "evidence_role": record["evidence_role"],
                "teacher_cell_id": record["teacher_cell_id"],
                "branch": record["branch"],
                "record_authority": record,
                "record_conditions": record["conditions"],
                "review_digest": record["review"]["review_digest"],
                "reviewed_media_sha256": record["reviewed_media"]["file_sha256"],
                "target_clean_latent_raw_sha256": record["target_clean_latent"][
                    "tensor_raw_sha256"
                ],
                "target_clean_latent_authority_digest": None,
                "source_clean_latent_raw_sha256": record["source_clean_latent"][
                    "tensor_raw_sha256"
                ],
                "source_posterior_index0_sha256": record[
                    "source_posterior_index0_sha256"
                ],
                "noise_seed": record["noise"]["seed"],
                "noise_raw_sha256": record["noise"]["artifact"][
                    "tensor_raw_sha256"
                ],
                "noise_replay_receipt": noise_receipt,
                "sigma_authority_digest": sigma_authority[
                    "sigma_authority_digest"
                ],
                "state_receipts": states,
                "forward_receipts": forwards,
                "container_bindings": [
                    {
                        "container_kind": "amplitude",
                        "path": candidate["amplitude_container_path"],
                        "file_sha256": candidate["amplitude_container_sha256"],
                        "slice_sha256": {
                            amplitude._tensor_name(sigma_index): tensor.sha256
                            for sigma_index, tensor in tensors.items()
                        },
                    }
                ],
                "sigma_metrics": sigma_metrics,
                "candidate_authority_evidence": candidate,
                "generated_rgb_decoded": False,
                "generated_rgb_used_as_model_input": False,
                "generated_rgb_used_as_regression_target": False,
                "generated_latent_used_as_absolute_regression_target": False,
                "model_parameters_updated": False,
                "optimizer_created": False,
            },
            "record_receipt_digest",
        )
        receipt_path, receipt_sha = _write_json(
            root,
            f"materialization-amplitude-receipts/calibrator-{local_ordinal:04d}.json",
            receipt,
            mode=parent_authority.MATERIALIZATION_RECEIPT_MODE,
        )
        references.append(
            {
                "record_id": record["record_id"],
                "record_kind": record["record_kind"],
                "path": receipt_path,
                "file_sha256": receipt_sha,
                "record_receipt_digest": receipt["record_receipt_digest"],
                "candidate_evidence_digest": candidate["evidence_digest"],
            }
        )
        receipt_by_evidence[evidence_id] = (
            receipt_path,
            receipt_sha,
            str(receipt["record_receipt_digest"]),
        )
        amplitude_receipts[evidence_id] = receipt

    amplitude_fragments = []
    for bundle in bundles:
        ordered = sorted(
            bundle["calibrator_evidence"],  # type: ignore[index]
            key=lambda row: str(row["evidence_id"]),
        )
        amplitude_fragments.append(
            {
                "teacher_cell_id": bundle["teacher_cell_id"],
                "branch": bundle["branch"],
                "calibrator_record_ids": [
                    amplitude_receipts[str(row["evidence_id"])]["record_id"]
                    for row in ordered
                ],
                "calibrator_evidence_candidates": [
                    copy.deepcopy(
                        amplitude_receipts[str(row["evidence_id"])][
                            "candidate_authority_evidence"
                        ]
                    )
                    for row in ordered
                ],
                "sigma_calibrations": copy.deepcopy(bundle["sigma_calibrations"]),
            }
        )

    run = _seal(
        {
            "schema_version": parent_authority.MATERIALIZATION_RUN_RECEIPT_SCHEMA,
            "plan_id": plan["plan_id"],
            "plan_digest": plan["plan_digest"],
            "plan_authority": plan,
            "population_digest": population["population_digest"],
            "record_order_sha256": population["record_order_sha256"],
            "runtime_identity": runtime_identity,
            "runtime_plan_digest": runtime_plan["runtime_plan_digest"],
            "official_helper_sources": runtime_plan["official_helper_sources"],
            "provider_abi": parent_authority.MATERIALIZATION_PROVIDER_ABI,
            "official_provider": True,
            "test_only": False,
            "world_size": 4,
            "dp_size": 1,
            "sp_size": 4,
            "sigma_indices": list(amplitude.SIGMA_INDICES),
            "sigma_authority": sigma_authority,
            "record_count": len(plan_records),
            "computation_digest": hashlib.sha256(
                b"amplitude-materialization-computation"
            ).hexdigest(),
            "record_receipts": references,
            "representation_sigma_evidence_candidates": parent_representation_fragments,
            "amplitude_sigma_calibration_candidates": amplitude_fragments,
            "output_policy": _materialization_output_policy(),
            "generated_rgb_decoded": False,
            "generated_rgb_used_as_model_input": False,
            "generated_rgb_used_as_regression_target": False,
            "generated_latent_used_as_absolute_regression_target": False,
            "model_parameters_updated": False,
            "optimizer_created": False,
        },
        "run_digest",
    )
    run_path, run_sha = _write_json(
        root,
        "materialization-amplitude-receipts/materialization-run.json",
        run,
        mode=parent_authority.MATERIALIZATION_RECEIPT_MODE,
    )
    for bundle in bundles:
        for evidence in bundle["calibrator_evidence"]:  # type: ignore[index]
            path, file_sha, receipt_digest = receipt_by_evidence[
                str(evidence["evidence_id"])
            ]
            evidence.update(
                {
                    "materialization_record_receipt_path": path,
                    "materialization_record_receipt_sha256": file_sha,
                    "materialization_record_receipt_digest": receipt_digest,
                    "materialization_run_digest": run["run_digest"],
                }
            )
            _resign(evidence, "evidence_digest")
        _resign(bundle, "bundle_digest")
    binding = _seal(
        {
            "schema_version": parent_authority.MATERIALIZATION_RUN_BINDING_SCHEMA,
            "path": run_path,
            "file_sha256": run_sha,
            "run_digest": run["run_digest"],
        },
        "binding_digest",
    )
    return binding, runtime_identity


def _build_amplitude_manifest(
    root: Path,
) -> tuple[dict[str, object], Path, str, dict[str, object]]:
    parent = _build_manifest(root / "parent-assets")
    parent_path = root / "parent-authority.json"
    parent_path.write_bytes(parent_authority.canonical_json_bytes(parent) + b"\n")
    parent_sha = parent_authority.file_sha256(parent_path)
    parent_receipt = parent_authority.validate_manifest_file(parent_path, parent_sha)
    sources = {str(row["source_iid"]): row for row in parent["sources"]}  # type: ignore[index]
    pairs = [
        row
        for row in parent["pairs"]  # type: ignore[index]
        if row["analysis_split"] == "fit" and row["optimizer_admitted"] is True
    ]
    representations = [
        row
        for row in parent["representation_admissions"]  # type: ignore[index]
        if row["analysis_split"] == "fit" and row["optimizer_admitted"] is True
    ]
    bundles: list[dict[str, object]] = []
    for bundle_ordinal, representation in enumerate(representations):
        cell = str(representation["teacher_cell_id"])
        branch = str(representation["branch"])
        eligible = sorted(
            (
                row
                for row in pairs
                if row["teacher_cell_id"] == cell and row["branch"] == branch
            ),
            key=lambda row: str(row["pair_id"]),
        )
        if len(eligible) != 8:
            raise AssertionError("fixture fit assignment capacity differs")
        fail_rows: list[dict[str, object]] = []
        calibrator_rows: list[dict[str, object]] = []
        calibrator_slices: dict[str, dict[int, tuple[bytes, float]]] = {}
        for ordinal, pair in enumerate(eligible[:4]):
            role = "frozen_fail" if ordinal < 2 else "calibrator"
            evidence, slices = _evidence(
                root,
                role=role,
                pair=pair,
                source=sources[str(pair["source_iid"])],
                ordinal=ordinal - 2,
            )
            if role == "frozen_fail":
                fail_rows.append(evidence)
            else:
                calibrator_rows.append(evidence)
                assert slices is not None
                calibrator_slices[str(evidence["evidence_id"])] = slices
        ordered = sorted(calibrator_rows, key=lambda row: str(row["evidence_id"]))
        sigma_rows: list[dict[str, object]] = []
        for sigma_index in amplitude.SIGMA_INDICES:
            metrics: list[dict[str, object]] = []
            norms: list[float] = []
            for evidence in ordered:
                payload, norm = calibrator_slices[str(evidence["evidence_id"])][
                    sigma_index
                ]
                norms.append(norm)
                metrics.append(
                    {
                        "evidence_id": evidence["evidence_id"],
                        "pair_id": evidence["pair_id"],
                        "projected_slice_sha256": hashlib.sha256(payload).hexdigest(),
                        "amplitude_norm": norm,
                    }
                )
            median = math.fsum(sorted(norms)) / 2.0
            _value, float_hex, scalar_sha = amplitude._float32(
                amplitude.AMPLITUDE_SCALE * median
            )
            sigma_rows.append(
                {
                    "sigma_index": sigma_index,
                    "calibrator_metrics": metrics,
                    "median_amplitude": median,
                    "a_min_scale": amplitude.AMPLITUDE_SCALE,
                    "a_min_float32_be_hex": float_hex,
                    "a_min_float32_le_sha256": scalar_sha,
                }
            )
        bundles.append(
            _seal(
                {
                    "schema_version": amplitude.BUNDLE_SCHEMA_VERSION,
                    "calibration_id": f"calibration:{cell}:{branch}",
                    "teacher_cell_id": cell,
                    "branch": branch,
                    "parent_representation_admission_digest": representation[
                        "admission_digest"
                    ],
                    "frozen_fail_evidence": fail_rows,
                    "calibrator_evidence": calibrator_rows,
                    "sigma_calibrations": sigma_rows,
                    "optimizer_admitted": True,
                },
                "bundle_digest",
            )
        )
    compute_contract = {
        "schema_version": amplitude.COMPUTE_CONTRACT_SCHEMA_VERSION,
        "model_eval": True,
        "torch_inference_mode": True,
        "official_frozen_native_only": True,
        "calibrator_peft_adapter_present": False,
        "frozen_effective_adapter_enabled": False,
        "frozen_effective_typed_patch_role_enabled": False,
        "base_compute_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "observer_output_dtype": "torch.float32",
        "observer_output_stage": "post-final-norm-proj-out-target-velocity",
        "observer_output_detached": True,
        "observer_output_contiguous": True,
        "same_state_counterfactual": True,
        "branch_and_noop_share_input_state": True,
        "world_size": 4,
        "dp_size": 1,
        "sp_size": 4,
        "sp_order_contract": "official-world4-sp4-rank-order-v1",
        "all_rank_consensus": True,
    }
    runtime = _seal(
        {
            "schema_version": amplitude.RUNTIME_IDENTITY_SCHEMA_VERSION,
            "bernini_revision": "1" * 40,
            "veomni_revision": "6" * 40,
            "official_checkpoint_tree_sha256": "2" * 64,
            "transformer_config_sha256": "3" * 64,
            "sigma_table_sha256": "4" * 64,
            "psiout_protocol_sha256": "5" * 64,
            "official_provider_source_sha256": "6" * 64,
            "official_provider_abi": "full30-psiout-official-provider-v1",
            "compute_contract": compute_contract,
            "compute_contract_digest": amplitude.object_sha256(compute_contract),
            "frame_count": 81,
            "fps": 25.0,
            "sampler_steps": 40,
        },
        "runtime_digest",
    )
    materialization_run_receipt, runtime = (
        _attach_amplitude_materialization_provenance(
            root,
            parent=parent,
            bundles=bundles,
        )
    )
    parent_binding = _seal(
        {
            "schema_version": amplitude.PARENT_BINDING_SCHEMA_VERSION,
            "manifest_file_sha256": parent_sha,
            "manifest_digest": parent["manifest_digest"],
            "validation_digest": parent_receipt["validation_digest"],
        },
        "binding_digest",
    )
    counts = {
        "optimizer_bundles": 16,
        "calibrator_evidence": 32,
        "frozen_fail_evidence": 32,
        "sigma_floor_rows": 96,
    }
    manifest = _seal(
        {
            "schema_version": amplitude.SCHEMA_VERSION,
            "parent_authority": parent_binding,
            "materialization_run_receipt": materialization_run_receipt,
            "frozen_runtime_identity": runtime,
            "calibration_bundles": bundles,
            "authority_counts": counts,
            "authority": {
                "status": "optimizer_admitted",
                "calibration_complete": True,
                "current_optimizer_bundles": 16,
                "current_calibrator_evidence": 32,
                "current_frozen_fail_evidence": 32,
                "optimizer_authorized": True,
            },
        },
        "manifest_digest",
    )
    return manifest, parent_path, parent_sha, parent


def _write_manifest(root: Path, manifest: dict[str, object]) -> tuple[Path, str]:
    path = root / "amplitude-authority.json"
    path.write_bytes(amplitude.canonical_json_bytes(manifest) + b"\n")
    return path, amplitude.file_sha256(path)


class Full30ActionAmplitudeAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (
            self.manifest,
            self.parent_path,
            self.parent_sha,
            self.parent,
        ) = _build_amplitude_manifest(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(self, manifest: dict[str, object] | None = None):
        path, digest = _write_manifest(self.root, manifest or self.manifest)
        return amplitude.load_amplitude_authority_v1(
            manifest_path=path,
            expected_manifest_sha256=digest,
            parent_manifest_path=self.parent_path,
            expected_parent_manifest_sha256=self.parent_sha,
        )

    def _reject(self, manifest: dict[str, object], pattern: str) -> None:
        with self.assertRaisesRegex(amplitude.Full30AmplitudeAuthorityError, pattern):
            self._load(manifest)

    def test_complete_authority_loads_and_resolves_exact_float32_floor(self) -> None:
        validated = self._load()
        self.assertTrue(validated.validation_receipt["optimizer_authorized"])
        self.assertEqual(len(validated.floors), 96)
        bundle = self.manifest["calibration_bundles"][0]  # type: ignore[index]
        sigma = bundle["sigma_calibrations"][0]
        floor = validated.resolve(
            bundle["teacher_cell_id"], bundle["branch"], sigma["sigma_index"]
        )
        self.assertEqual(floor.float32_be_hex, sigma["a_min_float32_be_hex"])
        self.assertEqual(floor.float32_le_sha256, sigma["a_min_float32_le_sha256"])
        self.assertGreater(floor.value_float32, 1e-6)
        with self.assertRaisesRegex(
            amplitude.Full30AmplitudeAuthorityError, "not optimizer-admitted"
        ):
            validated.resolve("unknown-cell", "action", 4)

    def test_floor_is_recomputed_not_trusted(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        row = manifest["calibration_bundles"][0]["sigma_calibrations"][0]  # type: ignore[index]
        row["median_amplitude"] *= 2.0
        _resign(manifest["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(manifest, "manifest_digest")
        self._reject(manifest, "median differs")

        manifest = copy.deepcopy(self.manifest)
        row = manifest["calibration_bundles"][0]["sigma_calibrations"][0]  # type: ignore[index]
        row["a_min_float32_be_hex"] = "00000000"
        _resign(manifest["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(manifest, "manifest_digest")
        self._reject(manifest, "a_min float32 bytes differ")

    def test_requires_two_distinct_fail_and_two_distinct_calibrator_sources(self) -> None:
        missing = copy.deepcopy(self.manifest)
        missing["calibration_bundles"][0]["frozen_fail_evidence"].pop()  # type: ignore[index]
        _resign(missing["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(missing, "manifest_digest")
        self._reject(missing, "exactly two")

        reused = copy.deepcopy(self.manifest)
        bundle = reused["calibration_bundles"][0]  # type: ignore[index]
        bundle["frozen_fail_evidence"][1] = copy.deepcopy(
            bundle["frozen_fail_evidence"][0]
        )
        _resign(bundle, "bundle_digest")
        _resign(reused, "manifest_digest")
        self._reject(reused, "evidence IDs are not distinct|pairs are not distinct")

    def test_fail_cannot_masquerade_as_partial_calibrator_or_reverse(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        evidence = manifest["calibration_bundles"][0]["frozen_fail_evidence"][0]  # type: ignore[index]
        review = evidence["pre_admission_review"]
        review["action_result"] = "partial"
        _resign(review, "review_digest")
        _resign(evidence, "evidence_digest")
        _resign(manifest["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(manifest, "manifest_digest")
        self._reject(manifest, "action_result differs")

        manifest = copy.deepcopy(self.manifest)
        evidence = manifest["calibration_bundles"][0]["calibrator_evidence"][0]  # type: ignore[index]
        review = evidence["pre_admission_review"]
        review["action_result"] = "fail"
        _resign(review, "review_digest")
        _resign(evidence, "evidence_digest")
        _resign(manifest["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(manifest, "manifest_digest")
        self._reject(manifest, "action_result differs")

    def test_evidence_review_and_artifact_identity_cannot_be_reused(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        first = manifest["calibration_bundles"][0]["frozen_fail_evidence"][0]  # type: ignore[index]
        second = manifest["calibration_bundles"][1]["frozen_fail_evidence"][0]  # type: ignore[index]
        second["baseline_output_path"] = first["baseline_output_path"]
        second["baseline_output_sha256"] = first["baseline_output_sha256"]
        review = second["pre_admission_review"]
        review["baseline_output_sha256"] = first["baseline_output_sha256"]
        _resign(review, "review_digest")
        _resign(second, "evidence_digest")
        _resign(manifest["calibration_bundles"][1], "bundle_digest")  # type: ignore[index]
        _resign(manifest, "manifest_digest")
        self._reject(manifest, "output_shas are reused across bundles")

    def test_sidecar_nonfinite_extra_bytes_symlink_and_mode_are_rejected(self) -> None:
        def evidence(manifest: dict[str, object]) -> dict[str, object]:
            return manifest["calibration_bundles"][0]["calibrator_evidence"][0]  # type: ignore[index]

        cases: list[tuple[str, str]] = []
        for kind in ("nonfinite", "extra", "mode", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                manifest, parent_path, parent_sha, _parent = _build_amplitude_manifest(root)
                item = evidence(manifest)
                path = Path(item["amplitude_container_path"])
                if kind == "nonfinite":
                    values = list(_basis(0, 0.1))
                    values[0] = float("nan")
                    item["amplitude_container_sha256"] = _rewrite_container_tensor(
                        path,
                        sigma_index=amplitude.SIGMA_INDICES[0],
                        values=tuple(values),
                    )
                    pattern = "non-finite"
                elif kind == "extra":
                    path.write_bytes(path.read_bytes() + b"x")
                    path.chmod(amplitude.CONTAINER_MODE)
                    item["amplitude_container_sha256"] = amplitude.file_sha256(path)
                    pattern = "payload length/extra bytes differ"
                elif kind == "mode":
                    path.chmod(0o640)
                    pattern = "mode must be exactly"
                else:
                    link = root / "amplitude-symlink.f30ac"
                    link.symlink_to(path)
                    item["amplitude_container_path"] = str(link.absolute())
                    pattern = "plain non-symlink"
                _resign(item, "evidence_digest")
                _resign(manifest["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
                _resign(manifest, "manifest_digest")
                manifest_path, manifest_sha = _write_manifest(root, manifest)
                with self.assertRaisesRegex(
                    amplitude.Full30AmplitudeAuthorityError, pattern
                ):
                    amplitude.load_amplitude_authority_v1(
                        manifest_path=manifest_path,
                        expected_manifest_sha256=manifest_sha,
                        parent_manifest_path=parent_path,
                        expected_parent_manifest_sha256=parent_sha,
                    )

    def test_parent_authority_and_runtime_identity_are_hard_bound(self) -> None:
        parent_tamper = copy.deepcopy(self.parent)
        parent_tamper["authority"]["optimizer_authorized"] = False  # type: ignore[index]
        parent_tamper["manifest_digest"] = parent_authority.object_sha256(
            {
                key: value
                for key, value in parent_tamper.items()
                if key != "manifest_digest"
            }
        )
        self.parent_path.write_bytes(
            parent_authority.canonical_json_bytes(parent_tamper) + b"\n"
        )
        self.parent_sha = parent_authority.file_sha256(self.parent_path)
        self._reject(self.manifest, "parent action authority is not admitted|manifest_file_sha256 differs")

        self.parent_path.write_bytes(
            parent_authority.canonical_json_bytes(self.parent) + b"\n"
        )
        self.parent_sha = parent_authority.file_sha256(self.parent_path)
        runtime_tamper = copy.deepcopy(self.manifest)
        runtime_tamper["frozen_runtime_identity"]["sampler_steps"] = 41  # type: ignore[index]
        _resign(runtime_tamper["frozen_runtime_identity"], "runtime_digest")  # type: ignore[index]
        _resign(runtime_tamper, "manifest_digest")
        self._reject(runtime_tamper, "sampler steps differ")

    def test_runtime_compute_contract_and_provider_identity_are_closed(self) -> None:
        for field, hostile_value in (
            ("veomni_revision", "not-a-revision"),
            ("official_provider_source_sha256", "not-a-sha"),
            ("official_provider_abi", "forged/provider"),
        ):
            hostile = copy.deepcopy(self.manifest)
            hostile["frozen_runtime_identity"][field] = hostile_value  # type: ignore[index]
            _resign(hostile["frozen_runtime_identity"], "runtime_digest")  # type: ignore[index]
            _resign(hostile, "manifest_digest")
            self._reject(hostile, "revision differs|lowercase SHA-256|safe identifier")

        hostile = copy.deepcopy(self.manifest)
        hostile["frozen_runtime_identity"]["compute_contract"][  # type: ignore[index]
            "model_eval"
        ] = False
        hostile["frozen_runtime_identity"]["compute_contract_digest"] = (  # type: ignore[index]
            amplitude.object_sha256(
                hostile["frozen_runtime_identity"]["compute_contract"]  # type: ignore[index]
            )
        )
        _resign(hostile["frozen_runtime_identity"], "runtime_digest")  # type: ignore[index]
        _resign(hostile, "manifest_digest")
        self._reject(hostile, "compute contract model_eval differs")

    def test_calibrator_materialization_provenance_and_physical_source_are_hard(self) -> None:
        missing = copy.deepcopy(self.manifest)
        evidence = missing["calibration_bundles"][0]["calibrator_evidence"][0]  # type: ignore[index]
        del evidence["materialization_record_receipt_digest"]
        _resign(evidence, "evidence_digest")
        _resign(missing["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(missing, "manifest_digest")
        self._reject(missing, "field closure differs")

        wrong_record = copy.deepcopy(self.manifest)
        first = wrong_record["calibration_bundles"][0]["calibrator_evidence"][0]  # type: ignore[index]
        second = wrong_record["calibration_bundles"][0]["calibrator_evidence"][1]  # type: ignore[index]
        for field in (
            "materialization_record_receipt_path",
            "materialization_record_receipt_sha256",
            "materialization_record_receipt_digest",
        ):
            first[field] = second[field]
        _resign(first, "evidence_digest")
        _resign(wrong_record["calibration_bundles"][0], "bundle_digest")  # type: ignore[index]
        _resign(wrong_record, "manifest_digest")
        self._reject(wrong_record, "base candidate|record authority")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, parent_path, parent_sha, _parent = _build_amplitude_manifest(root)
            evidence = manifest["calibration_bundles"][0]["calibrator_evidence"][0]  # type: ignore[index]
            receipt = json.loads(
                Path(evidence["materialization_record_receipt_path"]).read_bytes()
            )
            posterior = Path(
                receipt["record_authority"]["source_posterior_index0_path"]
            )
            posterior.write_bytes(posterior.read_bytes() + b"tamper")
            manifest_path, manifest_sha = _write_manifest(root, manifest)
            with self.assertRaisesRegex(
                amplitude.Full30AmplitudeAuthorityError,
                "materialization run is not admitted|source_posterior_index0.*SHA-256 differs|parent action authority is not admitted",
            ):
                amplitude.load_amplitude_authority_v1(
                    manifest_path=manifest_path,
                    expected_manifest_sha256=manifest_sha,
                    parent_manifest_path=parent_path,
                    expected_parent_manifest_sha256=parent_sha,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest, parent_path, parent_sha, _parent = _build_amplitude_manifest(root)
            binding = manifest["materialization_run_receipt"]
            run_path = Path(binding["path"])
            run = json.loads(run_path.read_bytes())
            run["official_provider"] = False
            run["test_only"] = True
            _resign(run, "run_digest")
            raw = amplitude.canonical_json_bytes(run) + b"\n"
            run_path.write_bytes(raw)
            run_path.chmod(parent_authority.MATERIALIZATION_RECEIPT_MODE)
            binding["file_sha256"] = hashlib.sha256(raw).hexdigest()
            binding["run_digest"] = run["run_digest"]
            _resign(binding, "binding_digest")
            _resign(manifest, "manifest_digest")
            manifest_path, manifest_sha = _write_manifest(root, manifest)
            with self.assertRaisesRegex(
                amplitude.Full30AmplitudeAuthorityError,
                "materialization run is not admitted",
            ):
                amplitude.load_amplitude_authority_v1(
                    manifest_path=manifest_path,
                    expected_manifest_sha256=manifest_sha,
                    parent_manifest_path=parent_path,
                    expected_parent_manifest_sha256=parent_sha,
                )

    def test_cli_requires_both_manifest_file_hashes(self) -> None:
        path, digest = _write_manifest(self.root, self.manifest)
        script = METHOD_ROOT / "full30_action_amplitude_authority_v1.py"
        success = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manifest",
                str(path),
                "--expected-sha256",
                digest,
                "--parent-manifest",
                str(self.parent_path),
                "--expected-parent-sha256",
                self.parent_sha,
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(success.stdout)["optimizer_authorized"])
        rejected = subprocess.run(
            [
                sys.executable,
                str(script),
                "--manifest",
                str(path),
                "--expected-sha256",
                "0" * 64,
                "--parent-manifest",
                str(self.parent_path),
                "--expected-parent-sha256",
                self.parent_sha,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("file SHA-256 differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
