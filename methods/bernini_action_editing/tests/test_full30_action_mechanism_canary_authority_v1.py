#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for value in (METHOD_ROOT, TEST_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import full30_action_amplitude_authority_v1 as amplitude
import full30_action_checkpoint_v1 as checkpoint
import full30_action_data_teacher_authority_v1 as formal_data
import full30_action_learning_v1 as learning
import full30_action_mechanism_canary_authority_v1 as canary
import full30_action_psiout_materializer_v1 as materializer
import train_full30_action_lora_v1 as trainer
from test_full30_action_amplitude_authority_v1 import (
    _attach_amplitude_materialization_provenance,
    _evidence,
)
from test_full30_action_data_teacher_authority_v1 import (
    _attach_teacher_materialization_provenance,
    _build_manifest,
    _pair_v5_seed_truth_fixture,
    _resign_nested,
)
from test_full30_action_psiout_materializer_v1 import (
    FakeFrozenProvider,
    FixtureBuilder as MaterializerFixtureBuilder,
)


def _write_manifest(path: Path, value: object) -> tuple[Path, str]:
    path.write_bytes(formal_data.canonical_json_bytes(value) + b"\n")
    return path.resolve(), formal_data.file_sha256(path)


def _teacher_seed_bindings(
    materialization: dict[str, object],
    teachers: list[dict[str, object]],
) -> list[dict[str, object]]:
    run = json.loads(Path(materialization["path"]).read_bytes())
    references = {
        str(row["record_id"]): row for row in run["record_receipts"]
    }
    records = [
        row
        for row in run["plan_authority"]["records"]
        if row["record_kind"] == "teacher_anchor"
        and row["evidence_role"] == "teacher_origin"
    ]
    result: list[dict[str, object]] = []
    for teacher in teachers:
        cell = str(teacher["teacher_cell_id"])
        candidates: list[dict[str, object]] = []
        generation_seed: int | None = None
        for branch in canary.BRANCHES:
            record = next(
                row
                for row in records
                if row["teacher_cell_id"] == cell and row["branch"] == branch
            )
            authority_binding = record["target_clean_latent_authority"]
            authority_receipt = json.loads(
                Path(authority_binding["path"]).read_bytes()
            )
            pair_v5_candidate = authority_receipt["pair_v5_candidate"]
            native_binding = authority_receipt["native_receipt"]
            candidate_envelope = json.loads(
                Path(pair_v5_candidate["path"]).read_bytes()
            )
            native_receipt = json.loads(
                Path(native_binding["path"]).read_bytes()
            )
            candidate_seed = int(candidate_envelope["candidate"]["seed"])
            native_seed = int(
                native_receipt["initial_noise_artifacts"]["t2v"][
                    "generator_initial_seed"
                ]
            )
            if candidate_seed != native_seed:
                raise AssertionError("fixture candidate/native seed differs")
            if generation_seed is None:
                generation_seed = candidate_seed
            elif generation_seed != candidate_seed:
                raise AssertionError("fixture teacher seed differs across branches")
            reference = references[str(record["record_id"])]
            candidates.append(
                formal_data.seal_record(
                    {
                        "schema_version": canary.TEACHER_SEED_CANDIDATE_BINDING_SCHEMA_VERSION,
                        "authority_kind": "pair-v5-candidate-plus-native-receipt",
                        "branch": branch,
                        "latent_authority_receipt_path": authority_binding["path"],
                        "latent_authority_receipt_file_sha256": authority_binding[
                            "file_sha256"
                        ],
                        "latent_authority_receipt_digest_field": authority_binding[
                            "digest_field"
                        ],
                        "latent_authority_receipt_digest": authority_binding["digest"],
                        "candidate_envelope_path": pair_v5_candidate["path"],
                        "candidate_envelope_file_sha256": pair_v5_candidate[
                            "sha256"
                        ],
                        "candidate_seed_json_pointer": "/candidate/seed",
                        "candidate_branch_json_pointer": "/candidate/semantic_branch",
                        "candidate_analysis_split_json_pointer": "/candidate/analysis_split",
                        "candidate_id_json_pointer": "/candidate/candidate_id",
                        "native_receipt_path": native_binding["path"],
                        "native_receipt_file_sha256": native_binding["sha256"],
                        "native_receipt_digest": native_binding[
                            "receipt_digest"
                        ],
                        "native_sampling_seed_json_pointer": "/sampling/t2v/seed",
                        "native_gaussian_seed_json_pointer": "/initial_noise_artifacts/t2v/generator_initial_seed",
                        "native_gaussian_raw_sha256_json_pointer": "/initial_noise_artifacts/t2v/raw_value_sha256",
                        "native_media_json_pointer": "/outputs/t2v",
                        "native_predecode_latent_json_pointer": "/outputs/t2v/normalized_clean_latent",
                        "materialization_record_id": record["record_id"],
                        "materialization_record_receipt_path": reference["path"],
                        "materialization_record_receipt_file_sha256": reference[
                            "file_sha256"
                        ],
                        "materialization_record_receipt_digest": reference[
                            "record_receipt_digest"
                        ],
                    },
                    "candidate_binding_digest",
                )
            )
        if generation_seed is None:
            raise AssertionError("fixture teacher seed is absent")
        result.append(
            formal_data.seal_record(
                {
                    "schema_version": canary.TEACHER_SEED_BINDING_SCHEMA_VERSION,
                    "population_profile": canary.POPULATION_PROFILE,
                    "teacher_cell_id": cell,
                    "origin_iid": teacher["origin_iid"],
                    "generation_seed": generation_seed,
                    "candidate_bindings": candidates,
                },
                "binding_digest",
            )
        )
    return result


def _reduced_data_manifest(root: Path) -> dict[str, object]:
    full = _build_manifest(root / "full-fixture")
    fit_teachers = [
        copy.deepcopy(row)
        for row in full["teacher_origins"]  # type: ignore[index]
        if row["analysis_split"] == "fit"
    ]
    first = fit_teachers[0]
    second = next(
        row
        for row in fit_teachers
        if row["event_id"] == first["event_id"]
        and row["teacher_cell_id"] != first["teacher_cell_id"]
    )
    for field in (
        "origin_iid",
        "origin_source_path",
        "origin_source_sha256",
        "origin_group_id",
        "event_id",
        "actor_kind",
        "q0_id",
        "actor_id",
        "scene_id",
    ):
        second[field] = first[field]
    _resign_nested(second, "origin_digest")
    teachers = [first, second]
    cell_order = [str(row["teacher_cell_id"]) for row in teachers]
    full_sources = {
        str(row["source_iid"]): copy.deepcopy(row)
        for row in full["sources"]  # type: ignore[index]
        if row["analysis_split"] == "fit"
    }
    full_pairs = {
        (str(row["source_iid"]), str(row["branch"])): copy.deepcopy(row)
        for row in full["pairs"]  # type: ignore[index]
        if row["analysis_split"] == "fit"
    }
    sources: list[dict[str, object]] = []
    for cell in cell_order:
        eligible = sorted(
            {
                source_iid
                for (source_iid, branch), row in full_pairs.items()
                if branch == "action" and row["teacher_cell_id"] == cell
            }
        )
        if len(eligible) != 8:
            raise AssertionError("formal fixture assignment capacity differs")
        sources.extend(full_sources[source_iid] for source_iid in eligible[:4])
    pairs = [
        full_pairs[(str(source["source_iid"]), branch)]
        for source in sources
        for branch in canary.BRANCHES
    ]
    representation_index = {
        (str(row["teacher_cell_id"]), str(row["branch"])): copy.deepcopy(row)
        for row in full["representation_admissions"]  # type: ignore[index]
        if row["analysis_split"] == "fit"
    }
    representations = [
        representation_index[(cell, branch)]
        for cell in cell_order
        for branch in canary.BRANCHES
    ]
    teacher_by_cell = {
        str(row["teacher_cell_id"]): row for row in teachers
    }
    for representation in representations:
        teacher = teacher_by_cell[str(representation["teacher_cell_id"])]
        origin = representation["origin_evidence"]
        origin["anchor_iid"] = teacher["origin_iid"]
        for field in (
            "event_id",
            "actor_kind",
            "q0_id",
            "actor_id",
            "scene_id",
        ):
            origin[field] = teacher[field]
            origin["pre_admission_blind_review"][field] = teacher[field]
        _resign_nested(
            origin["pre_admission_blind_review"], "review_digest"
        )
        _resign_nested(origin, "evidence_digest")
        representation["event_id"] = teacher["event_id"]
        cross = representation["cross_anchor_evidence"]
        cross["anchor_split"] = "fit"
        cross["pre_admission_blind_review"]["anchor_split"] = "fit"
        _resign_nested(
            cross["pre_admission_blind_review"], "review_digest"
        )
        for evidence_field in ("origin_evidence", "cross_anchor_evidence"):
            evidence = representation[evidence_field]
            for field in formal_data._MATERIALIZATION_PROVENANCE_FIELDS:
                evidence.pop(field, None)
            _resign_nested(evidence, "evidence_digest")
        _resign_nested(representation, "admission_digest")
    materialization = _attach_teacher_materialization_provenance(
        root / "reduced-teacher-materialization", representations
    )
    teacher_seed_bindings = _teacher_seed_bindings(materialization, teachers)
    unsigned = {
        "schema_version": canary.SCHEMA_VERSION,
        "population_profile": canary.POPULATION_PROFILE,
        "materialization_run_receipt": materialization,
        "authority": copy.deepcopy(canary._EXPECTED_AUTHORITY),
        "source_io_policy": copy.deepcopy(full["source_io_policy"]),
        "teacher_origins": teachers,
        "teacher_seed_bindings": teacher_seed_bindings,
        "sources": sources,
        "pairs": pairs,
        "representation_admissions": representations,
        "authority_counts": copy.deepcopy(canary._EXPECTED_COUNTS),
    }
    return formal_data.seal_record(unsigned, "manifest_digest")


def _amplitude_bundles(
    root: Path, parent: dict[str, object]
) -> list[dict[str, object]]:
    sources = {
        str(row["source_iid"]): row
        for row in parent["sources"]  # type: ignore[index]
    }
    pairs = [
        row for row in parent["pairs"]  # type: ignore[index]
    ]
    representations = [
        row for row in parent["representation_admissions"]  # type: ignore[index]
    ]
    bundles: list[dict[str, object]] = []
    for representation in representations:
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
        if len(eligible) != 4:
            raise AssertionError("canary bundle requires four source pairs")
        fails: list[dict[str, object]] = []
        calibrators: list[dict[str, object]] = []
        slices_by_evidence: dict[str, dict[int, tuple[bytes, float]]] = {}
        calibrator_ordinals = (
            {2, 3} if branch == "action" else {0, 1}
        )
        calibrator_rank = 0
        for ordinal, pair in enumerate(eligible):
            role = (
                "calibrator"
                if ordinal in calibrator_ordinals
                else "frozen_fail"
            )
            evidence, slices = _evidence(
                root,
                role=role,
                pair=pair,
                source=sources[str(pair["source_iid"])],
                ordinal=calibrator_rank,
            )
            if role == "frozen_fail":
                fails.append(evidence)
            else:
                calibrator_rank += 1
                if slices is None:
                    raise AssertionError("calibrator fixture lacks tensor slices")
                calibrators.append(evidence)
                slices_by_evidence[str(evidence["evidence_id"])] = slices
        ordered = sorted(calibrators, key=lambda row: str(row["evidence_id"]))
        sigma_rows: list[dict[str, object]] = []
        for sigma_index in amplitude.SIGMA_INDICES:
            metrics: list[dict[str, object]] = []
            norms: list[float] = []
            for evidence in ordered:
                payload, norm = slices_by_evidence[str(evidence["evidence_id"])][
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
            _value, value_hex, value_sha = amplitude._float32(
                amplitude.AMPLITUDE_SCALE * median
            )
            sigma_rows.append(
                {
                    "sigma_index": sigma_index,
                    "calibrator_metrics": metrics,
                    "median_amplitude": median,
                    "a_min_scale": amplitude.AMPLITUDE_SCALE,
                    "a_min_float32_be_hex": value_hex,
                    "a_min_float32_le_sha256": value_sha,
                }
            )
        bundles.append(
            amplitude.seal_record(
                {
                    "schema_version": amplitude.BUNDLE_SCHEMA_VERSION,
                    "calibration_id": f"canary-calibration:{cell}:{branch}",
                    "teacher_cell_id": cell,
                    "branch": branch,
                    "parent_representation_admission_digest": representation[
                        "admission_digest"
                    ],
                    "frozen_fail_evidence": fails,
                    "calibrator_evidence": calibrators,
                    "sigma_calibrations": sigma_rows,
                    "optimizer_admitted": True,
                },
                "bundle_digest",
            )
        )
    return bundles


def _reduced_amplitude_manifest(
    root: Path,
    *,
    parent: dict[str, object],
    parent_path: Path,
    parent_sha: str,
) -> dict[str, object]:
    parent_receipt = canary.load_data_authority_v1(
        manifest_path=parent_path, expected_manifest_sha256=parent_sha
    ).validation_receipt
    bundles = _amplitude_bundles(root / "amplitude-assets", parent)
    materialization, runtime = _attach_amplitude_materialization_provenance(
        root / "combined-materialization", parent=parent, bundles=bundles
    )
    parent_binding = amplitude.seal_record(
        {
            "schema_version": canary.PARENT_BINDING_SCHEMA_VERSION,
            "manifest_file_sha256": parent_sha,
            "manifest_digest": parent["manifest_digest"],
            "validation_digest": parent_receipt["validation_digest"],
        },
        "binding_digest",
    )
    unsigned: dict[str, object] = {
        "schema_version": canary.AMPLITUDE_SCHEMA_VERSION,
        "population_profile": canary.POPULATION_PROFILE,
        "parent_authority": parent_binding,
        "materialization_run_receipt": materialization,
        "frozen_runtime_identity": runtime,
        "calibration_bundles": bundles,
        "authority_counts": copy.deepcopy(canary._EXPECTED_AMPLITUDE_COUNTS),
        "authority": copy.deepcopy(canary._EXPECTED_AMPLITUDE_AUTHORITY),
    }
    projection_digest = canary._amplitude_authority_projection_digest(
        unsigned
    )
    run = json.loads(Path(materialization["path"]).read_bytes())
    run_plan = run["plan_authority"]
    plan = dict(
        canary._official_materializer_plan_projection_v1(
            run_plan, materializer_module=materializer
        )
    )
    plan_path, plan_sha = _write_manifest(
        root / "exact-official-materializer-plan.json", plan
    )
    rows = tuple(
        learning.ActionPairRow(
            row_id=row["pair_id"],
            source_id=row["source_iid"],
            branch=row["branch"],
            teacher_cell_id=row["teacher_cell_id"],
        )
        for row in parent["pairs"]
    )
    schedule_run_seed = 20260815
    schedule = canary.build_checkpoint_scaffold_schedule_v1(
        rows,
        run_seed=schedule_run_seed,
        learning_module=learning,
        checkpoint_module=checkpoint,
    )
    admission = canary.admit_reduced_materialization_plan_v1(
        plan,
        schedule=schedule,
        materializer_module=materializer,
        checkpoint_module=checkpoint,
        authority_projection_digest=projection_digest,
        parent_manifest_file_sha256=parent_sha,
        parent_manifest_digest=parent["manifest_digest"],
    )
    admission_path, admission_sha = _write_manifest(
        root / "exact-official-materializer-admission.json",
        dict(admission.validation_receipt),
    )
    unsigned["materializer_plan_admission"] = amplitude.seal_record(
        {
            "schema_version": canary.MATERIALIZER_PLAN_BINDING_SCHEMA_VERSION,
            "population_profile": canary.POPULATION_PROFILE,
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_sha,
            "plan_id": plan["plan_id"],
            "plan_digest": plan["plan_digest"],
            "run_plan_id": run_plan["plan_id"],
            "run_plan_digest": run_plan["plan_digest"],
            "run_record_bridge_digest": canary._plan_run_bridge_digest_v1(
                plan, run_plan
            ),
            "run_fragment_binding_digest": canary._run_fragment_binding_digest_v1(
                run
            ),
            "schedule_run_seed": schedule_run_seed,
            "admission_receipt_path": str(admission_path),
            "admission_receipt_file_sha256": admission_sha,
            "admission_validation_digest": admission.validation_receipt[
                "validation_digest"
            ],
            "authority_projection_digest": projection_digest,
            "parent_manifest_file_sha256": parent_sha,
            "parent_manifest_digest": parent["manifest_digest"],
            "materialization_run_receipt_file_sha256": materialization[
                "file_sha256"
            ],
            "materialization_run_digest": materialization["run_digest"],
        },
        "binding_digest",
    )
    return amplitude.seal_record(unsigned, "manifest_digest")


def _schedule_rows() -> tuple[learning.ActionPairRow, ...]:
    return tuple(
        learning.ActionPairRow(
            row_id=f"row:{source}:{branch}",
            source_id=f"real-source-{source:02d}",
            branch=branch,
            teacher_cell_id=f"real-cell-{source // 4}",
        )
        for source in range(8)
        for branch in canary.BRANCHES
    )


def _rewrite_materializer_review(
    row: dict[str, object], *, teacher: bool
) -> None:
    binding = row["review"]
    path = Path(binding["path"])
    review = json.loads(path.read_bytes())
    review["branch"] = row["branch"]
    if teacher:
        for field in (
            "event_id",
            "actor_kind",
            "q0_id",
            "actor_id",
            "scene_id",
        ):
            review[field] = row[field]
    review = materializer.seal_record(review, "review_digest")
    raw = materializer.canonical_json_bytes(review) + b"\n"
    path.write_bytes(raw)
    binding["file_sha256"] = hashlib.sha256(raw).hexdigest()
    binding["review_digest"] = review["review_digest"]


def _rewrite_materializer_generation_authority(
    record: dict[str, object],
    *,
    generation_seed: int,
    gaussian_raw_sha256_override: str | None = None,
    gaussian_content_sha256_override: str | None = None,
) -> None:
    binding = record["target_clean_latent_authority"]
    path = Path(binding["path"])
    authority_receipt = json.loads(path.read_bytes())
    candidate = {
        "branch": record["branch"],
        "anchor_split": record["analysis_split"],
        "event_id": record["event_id"],
        "teacher_cell_id": record["teacher_cell_id"],
        "actor_id": record["actor_id"],
        "scene_id": record["scene_id"],
        "q0_id": record["q0_id"],
        "anchor_video_path": record["reviewed_media"]["path"],
        "anchor_video_sha256": record["reviewed_media"]["file_sha256"],
    }
    pair_v5_candidate, native_receipt = _pair_v5_seed_truth_fixture(
        path.parent.parent,
        candidate=candidate,
        target=record["target_clean_latent"],
        seed=generation_seed,
        record_slug=hashlib.sha256(
            str(record["record_id"]).encode("utf-8")
        ).hexdigest(),
    )
    if (
        gaussian_raw_sha256_override is not None
        or gaussian_content_sha256_override is not None
    ):
        native_path = Path(native_receipt["path"])
        native = json.loads(native_path.read_bytes())
        gaussian = native["initial_noise_artifacts"]["t2v"]
        if gaussian_raw_sha256_override is not None:
            gaussian["raw_value_sha256"] = gaussian_raw_sha256_override
        if gaussian_content_sha256_override is not None:
            gaussian["content_sha256"] = gaussian_content_sha256_override
        _resign_nested(native, "receipt_digest")
        _, native_sha = _write_manifest(native_path, native)
        native_receipt["sha256"] = native_sha
        native_receipt["receipt_digest"] = native["receipt_digest"]
    authority_receipt.update(
        {
            "pair_v5_candidate": pair_v5_candidate,
            "native_receipt": native_receipt,
        }
    )
    digest_field = str(binding["digest_field"])
    authority_receipt = materializer.seal_record(
        authority_receipt, digest_field
    )
    raw = materializer.canonical_json_bytes(authority_receipt) + b"\n"
    path.write_bytes(raw)
    binding["file_sha256"] = hashlib.sha256(raw).hexdigest()
    binding["digest"] = authority_receipt[digest_field]


def _official_reduced_materializer_plan(
    root: Path,
    *,
    same_origin: bool = True,
    same_generation_seed: bool = False,
    reuse_gaussian_fields_across_cells: frozenset[str] = frozenset(),
) -> tuple[dict[str, object], tuple[learning.ActionPairRow, ...]]:
    if not reuse_gaussian_fields_across_cells <= {
        "raw_value_sha256",
        "content_sha256",
    }:
        raise AssertionError("unknown Gaussian reuse fixture field")
    builder = MaterializerFixtureBuilder(root)
    teacher_records: list[dict[str, object]] = []
    amplitude_records: list[dict[str, object]] = []
    schedule_pair_ids: dict[tuple[str, str], str] = {}
    source_ids = [f"{1000000000000001 + index:016d}" for index in range(8)]
    for cell_index in range(2):
        cell = f"canary-cell-{cell_index}"
        event = "canary-shared-event"
        q0_id = "canary-shared-q0"
        generation_seed = 7000 + (0 if same_generation_seed else cell_index)
        for branch_index, branch in enumerate(canary.BRANCHES):
            noise = builder.artifact(
                f"{cell}-{branch}-teacher-noise",
                3.0 + cell_index + branch_index * 0.1,
            )
            for role in ("teacher_origin", "same_event_cross_anchor"):
                stem = f"{cell}-{branch}-{role}"
                if role == "teacher_origin":
                    anchor_iid = (
                        "0000000000000001"
                        if same_origin
                        else f"{1 + cell_index:016d}"
                    )
                    actor_id = (
                        "canary-shared-origin-actor"
                        if same_origin
                        else f"canary-distinct-origin-actor-{cell_index}"
                    )
                    scene_id = (
                        "canary-shared-origin-scene"
                        if same_origin
                        else f"canary-distinct-origin-scene-{cell_index}"
                    )
                else:
                    anchor_iid = f"{100 + cell_index:016d}"
                    actor_id = f"canary-cross-actor-{cell_index}"
                    scene_id = f"canary-cross-scene-{cell_index}"
                record = dict(
                    builder.teacher_record(
                        stem,
                        role,
                        anchor_iid,
                        actor_id,
                        scene_id,
                        0.1 + len(teacher_records) * 0.01,
                        noise,
                    )
                )
                record.update(
                    {
                        "teacher_cell_id": cell,
                        "branch": branch,
                        "event_id": event,
                        "q0_id": q0_id,
                    }
                )
                record["noise"]["seed"] = materializer.teacher_noise_seed_v1(
                    cell, branch
                )
                _rewrite_materializer_generation_authority(
                    record,
                    generation_seed=generation_seed,
                    gaussian_raw_sha256_override=(
                        hashlib.sha256(
                            b"official-initial-gaussian:7000"
                        ).hexdigest()
                        if cell_index == 1
                        and "raw_value_sha256"
                        in reuse_gaussian_fields_across_cells
                        else None
                    ),
                    gaussian_content_sha256_override=(
                        hashlib.sha256(
                            b"official-initial-gaussian-content:7000"
                        ).hexdigest()
                        if cell_index == 1
                        and "content_sha256"
                        in reuse_gaussian_fields_across_cells
                        else None
                    ),
                )
                _rewrite_materializer_review(record, teacher=True)
                teacher_records.append(
                    dict(materializer.seal_record(record, "record_digest"))
                )

            local_sources = (
                (0, 1) if branch == "action" else (2, 3)
            )
            for local_source in local_sources:
                source_index = cell_index * 4 + local_source
                source_iid = source_ids[source_index]
                stem = f"{cell}-{branch}-calibrator-{local_source}"
                record = dict(
                    builder.amplitude_record(
                        stem,
                        source_iid,
                        0.4 + len(amplitude_records) * 0.01,
                    )
                )
                record.update(
                    {
                        "teacher_cell_id": cell,
                        "branch": branch,
                        "event_id": event,
                        "q0_id": q0_id,
                    }
                )
                _rewrite_materializer_review(record, teacher=False)
                record = dict(
                    materializer.seal_record(record, "record_digest")
                )
                amplitude_records.append(record)
                schedule_pair_ids[(source_iid, branch)] = str(
                    record["pair_id"]
                )

    records = [*teacher_records, *amplitude_records]
    population = materializer.seal_record(
        {
            "schema_version": materializer.PLAN_POPULATION_SCHEMA_VERSION,
            "population_id": "canary-reduced-materializer-fixture",
            "record_count": len(records),
            "teacher_record_count": len(teacher_records),
            "amplitude_record_count": len(amplitude_records),
            "teacher_cell_ids": ["canary-cell-0", "canary-cell-1"],
            "record_order_sha256": materializer.object_sha256(
                [row["record_id"] for row in records]
            ),
            "finite_closed_population": True,
            "block_probe": False,
        },
        "population_digest",
    )
    policy = {
        "schema_version": materializer.PLAN_OUTPUT_POLICY_SCHEMA_VERSION,
        "create_only": True,
        "container_mode_octal": "0600",
        "generated_rgb_decoded": False,
        "generated_rgb_used_as_model_input": False,
        "generated_rgb_used_as_regression_target": False,
        "generated_latent_used_as_absolute_regression_target": False,
        "model_parameters_updated": False,
        "optimizer_created": False,
        "persisted_tensor_role": "detached-post-head-psiout-or-same-mode-amplitude-evidence-only",
    }
    plan = dict(
        materializer.seal_record(
            {
                "schema_version": materializer.PLAN_SCHEMA_VERSION,
                "plan_id": "canary-reduced-materializer-fixture-plan",
                "status": "SEALED_REVIEWED_PRE_OPTIMIZER",
                "runtime": builder.runtime,
                "population": population,
                "records": records,
                "output_policy": policy,
            },
            "plan_digest",
        )
    )
    rows = tuple(
        learning.ActionPairRow(
            row_id=schedule_pair_ids.get(
                (source_iid, branch), f"runtime-pair:{source_iid}:{branch}"
            ),
            source_id=source_iid,
            branch=branch,
            teacher_cell_id=f"canary-cell-{source_index // 4}",
        )
        for source_index, source_iid in enumerate(source_ids)
        for branch in canary.BRANCHES
    )
    return plan, rows


class Full30MechanismCanaryAuthorityTests(unittest.TestCase):
    def test_population_profile_accepts_same_origin_and_rejects_distinct_origin_or_confirmation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _reduced_data_manifest(Path(temporary).resolve())
            teachers = manifest["teacher_origins"]  # type: ignore[index]
            for field in (
                "origin_iid",
                "origin_source_path",
                "origin_source_sha256",
                "origin_group_id",
                "event_id",
                "actor_kind",
                "q0_id",
                "actor_id",
                "scene_id",
            ):
                self.assertEqual(teachers[0][field], teachers[1][field])
            self.assertEqual(len(canary._validate_teacher_origins(teachers)), 2)
            representations = manifest["representation_admissions"]
            origin_media = {
                row["origin_evidence"]["anchor_video_sha256"]
                for row in representations
            }
            cross_media = {
                row["cross_anchor_evidence"]["anchor_video_sha256"]
                for row in representations
            }
            self.assertEqual(len(origin_media), 4)
            self.assertEqual(len(cross_media), 2)
            self.assertFalse(origin_media & cross_media)

            for field, replacement in (
                ("origin_iid", "ffffffffffffffff"),
                ("origin_group_id", "forbidden-distinct-origin-group"),
                ("event_id", "forbidden-distinct-origin-event"),
                ("actor_id", "forbidden-distinct-origin-actor"),
            ):
                with self.subTest(field=field):
                    distinct_origin = copy.deepcopy(teachers)
                    distinct_origin[1][field] = replacement
                    _resign_nested(distinct_origin[1], "origin_digest")
                    with self.assertRaisesRegex(
                        canary.Full30MechanismCanaryAuthorityError,
                        "requires exact shared origin",
                    ):
                        canary._validate_teacher_origins(distinct_origin)

            confirmation = copy.deepcopy(teachers)
            confirmation[1]["analysis_split"] = "confirmation"
            _resign_nested(confirmation[1], "origin_digest")
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "is not fit",
            ):
                canary._validate_teacher_origins(confirmation)

    def test_teacher_seed_bindings_fail_closed_on_missing_same_or_resigned_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = _reduced_data_manifest(root)

            missing = copy.deepcopy(manifest)
            missing["teacher_seed_bindings"][0].pop("generation_seed")
            _resign_nested(
                missing["teacher_seed_bindings"][0], "binding_digest"
            )
            _resign_nested(missing, "manifest_digest")
            missing_path, missing_sha = _write_manifest(
                root / "missing-seed-binding.json", missing
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "field closure",
            ):
                canary.load_data_authority_v1(
                    manifest_path=missing_path,
                    expected_manifest_sha256=missing_sha,
                )

            same_seed = copy.deepcopy(manifest)
            seed_rows = same_seed["teacher_seed_bindings"]
            seed_rows[1]["generation_seed"] = seed_rows[0]["generation_seed"]
            _resign_nested(seed_rows[1], "binding_digest")
            _resign_nested(same_seed, "manifest_digest")
            same_path, same_sha = _write_manifest(
                root / "same-seed-binding.json", same_seed
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "declared generation seed differs|generation seed is reused",
            ):
                canary.load_data_authority_v1(
                    manifest_path=same_path,
                    expected_manifest_sha256=same_sha,
                )

            binding = manifest["teacher_seed_bindings"][0]
            original_candidate = binding["candidate_bindings"][0]
            candidate = copy.deepcopy(original_candidate)
            native_receipt = json.loads(
                Path(candidate["native_receipt_path"]).read_bytes()
            )
            native_receipt["sampling"]["t2v"]["seed"] = (
                int(binding["generation_seed"]) + 1
            )
            _resign_nested(native_receipt, "receipt_digest")
            replacement_path, replacement_sha = _write_manifest(
                root / "resigned-native-receipt.json", native_receipt
            )
            candidate["native_receipt_path"] = str(replacement_path)
            candidate["native_receipt_file_sha256"] = replacement_sha
            candidate["native_receipt_digest"] = native_receipt[
                "receipt_digest"
            ]
            _resign_nested(candidate, "candidate_binding_digest")
            materialization = json.loads(
                Path(manifest["materialization_run_receipt"]["path"]).read_bytes()
            )
            record = next(
                row
                for row in materialization["plan_authority"]["records"]
                if row["record_id"] == candidate["materialization_record_id"]
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "candidate/native/official-Gaussian seeds differ",
            ):
                canary._validate_pair_v5_seed_truth(
                    candidate,
                    branch=candidate["branch"],
                    record=record,
                    label="hostile-resigned-native-receipt",
                )

            valid_runtime_digest = canary._validate_pair_v5_seed_truth(
                original_candidate,
                branch=original_candidate["branch"],
                record=record,
                label="valid-native-runtime",
            )[4]
            runtime_candidate = copy.deepcopy(original_candidate)
            runtime_receipt = json.loads(
                Path(runtime_candidate["native_receipt_path"]).read_bytes()
            )
            runtime_receipt["runtime_versions"] = {
                "torch": "hostile-resigned-runtime"
            }
            _resign_nested(runtime_receipt, "receipt_digest")
            runtime_path, runtime_sha = _write_manifest(
                root / "resigned-runtime-native-receipt.json",
                runtime_receipt,
            )
            runtime_candidate["native_receipt_path"] = str(runtime_path)
            runtime_candidate["native_receipt_file_sha256"] = runtime_sha
            runtime_candidate["native_receipt_digest"] = runtime_receipt[
                "receipt_digest"
            ]
            hostile_runtime_digest = canary._validate_pair_v5_seed_truth(
                runtime_candidate,
                branch=runtime_candidate["branch"],
                record=record,
                label="hostile-resigned-native-runtime",
            )[4]
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "fixed runtime identities differ",
            ):
                canary._validate_pair_v5_runtime_identity_closure_v1(
                    {valid_runtime_digest, hostile_runtime_digest},
                    label="hostile",
                )

    def test_action_and_incomplete_seed_truth_may_reopen_from_distinct_exp_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = _reduced_data_manifest(root)
            binding = manifest["teacher_seed_bindings"][0]
            materialization = json.loads(
                Path(manifest["materialization_run_receipt"]["path"]).read_bytes()
            )
            records = {
                row["record_id"]: row
                for row in materialization["plan_authority"]["records"]
            }
            results = []
            roots = []
            for exp_name, source_candidate in zip(
                ("EXP011-action", "EXP013-incomplete"),
                binding["candidate_bindings"],
            ):
                exp_root = root / exp_name
                exp_root.mkdir()
                roots.append(exp_root)
                candidate = copy.deepcopy(source_candidate)
                envelope = json.loads(
                    Path(candidate["candidate_envelope_path"]).read_bytes()
                )
                native = json.loads(
                    Path(candidate["native_receipt_path"]).read_bytes()
                )
                envelope_path, envelope_sha = _write_manifest(
                    exp_root / "candidate.json", envelope
                )
                native_path, native_sha = _write_manifest(
                    exp_root / "native-receipt.json", native
                )
                candidate["candidate_envelope_path"] = str(envelope_path)
                candidate["candidate_envelope_file_sha256"] = envelope_sha
                candidate["native_receipt_path"] = str(native_path)
                candidate["native_receipt_file_sha256"] = native_sha
                candidate["native_receipt_digest"] = native["receipt_digest"]
                results.append(
                    canary._validate_pair_v5_seed_truth(
                        candidate,
                        branch=candidate["branch"],
                        record=records[candidate["materialization_record_id"]],
                        label=f"cross-exp-{exp_name}",
                    )
                )
            self.assertNotEqual(roots[0], roots[1])
            self.assertEqual(len({row[0] for row in results}), 1)
            self.assertEqual(len({row[2] for row in results}), 1)
            self.assertEqual(len({row[3] for row in results}), 1)
            self.assertEqual(len({row[4] for row in results}), 1)

    def test_checkpoint_scaffold_is_canonical_and_only_first_sixteen_rows_are_live(self) -> None:
        schedule = canary.build_checkpoint_scaffold_schedule_v1(
            _schedule_rows(),
            run_seed=20260815,
            learning_module=learning,
            checkpoint_module=checkpoint,
        )
        canonical = checkpoint.canonical_schedule_v2(schedule)
        self.assertEqual(len(canonical), 1280)
        first = canonical[:16]
        self.assertEqual(
            [row["global_index"] for row in first], list(range(16))
        )
        self.assertEqual(
            len({row["row"]["source_id"] for row in first}), 8
        )
        for update in range(2):
            group = first[update * 8 : (update + 1) * 8]
            self.assertEqual(
                Counter(row["row"]["branch"] for row in group),
                Counter({"action": 4, "incomplete": 4}),
            )
            self.assertEqual(
                len({row["row"]["source_id"] for row in group}), 4
            )
        self.assertTrue(
            all(
                canary.is_serialization_tail_source_v1(
                    row["row"]["source_id"]
                )
                for row in canonical[16:24]
            )
        )
        self.assertEqual(
            len(
                {
                    row["row"]["source_id"]
                    for row in canonical
                    if canary.is_serialization_tail_source_v1(
                        row["row"]["source_id"]
                    )
                }
            ),
            56,
        )
        receipt = canary.schedule_authority_receipt_v1(
            schedule, checkpoint_module=checkpoint
        )
        self.assertFalse(receipt["formal_authority"])
        self.assertEqual(receipt["population_profile"], canary.POPULATION_PROFILE)
        self.assertTrue(receipt["tail_serialization_only"])
        self.assertFalse(receipt["u3_authorized"])
        self.assertFalse(receipt["event_family_generalization"])
        self.assertFalse(receipt["synthetic_target_bytes_read"])
        self.assertEqual(receipt["executable_global_indices"], list(range(16)))

    def test_u3_is_rejected_before_any_tail_authority_can_resolve(self) -> None:
        schedule = canary.build_checkpoint_scaffold_schedule_v1(
            _schedule_rows(),
            run_seed=7,
            learning_module=learning,
            checkpoint_module=checkpoint,
        )
        admitted = {
            row.row_id: {
                "source_iid": row.source_id,
                "branch": row.branch,
                "teacher_cell_id": row.teacher_cell_id,
            }
            for row in _schedule_rows()
        }
        canary.authorize_scheduled_row_v1(schedule[0], admitted_pairs=admitted)
        with self.assertRaisesRegex(
            canary.Full30MechanismCanaryAuthorityError,
            "serialization-only",
        ):
            canary.authorize_scheduled_row_v1(
                schedule[16], admitted_pairs=admitted
            )

    def test_reduced_plan_reuses_official_materializer_with_nonexecuted_sigma_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan, rows = _official_reduced_materializer_plan(root)
            schedule = canary.build_checkpoint_scaffold_schedule_v1(
                rows,
                run_seed=20260815,
                learning_module=learning,
                checkpoint_module=checkpoint,
            )
            admission = canary.admit_reduced_materialization_plan_v1(
                plan,
                schedule=schedule,
                materializer_module=materializer,
                checkpoint_module=checkpoint,
                authority_projection_digest="a" * 64,
                parent_manifest_file_sha256="b" * 64,
                parent_manifest_digest="c" * 64,
            )
            receipt = admission.validation_receipt
            self.assertEqual(receipt["record_count"], 16)
            self.assertEqual(receipt["shared_origin_identities"], 1)
            self.assertEqual(receipt["teacher_generation_media_outputs"], 4)
            self.assertEqual(receipt["teacher_generation_reviews"], 4)
            self.assertEqual(
                receipt["distinct_teacher_gaussian_raw_sha256"], 2
            )
            self.assertEqual(
                receipt["distinct_teacher_gaussian_content_sha256"], 2
            )
            self.assertEqual(
                receipt["physically_reopened_real_source_index0_records"], 8
            )
            self.assertTrue(
                receipt["official_materializer_plan_validator_reused"]
            )
            self.assertTrue(
                receipt["official_materializer_six_sigma_abi_retained"]
            )
            self.assertGreater(
                len(receipt["non_executable_evidence_coordinates"]), 0
            )
            self.assertFalse(
                receipt["non_executable_evidence_trainer_read_authorized"]
            )
            self.assertFalse(
                receipt["training_noise_materialized_by_official_materializer"]
            )
            self.assertFalse(receipt["confirmation_population_admitted"])
            self.assertFalse(receipt["synthetic_target_index1_bytes_read"])
            self.assertFalse(receipt["synthetic_target_bytes_read"])
            self.assertFalse(receipt["event_family_generalization"])
            self.assertFalse(receipt["identity_generalization"])
            self.assertFalse(receipt["generalization"])
            self.assertEqual(receipt["shared_origin_identities"], 1)
            self.assertEqual(
                receipt["population_profile"], canary.POPULATION_PROFILE
            )

            provider = FakeFrozenProvider(admission.plan)
            result = materializer.materialize_with_test_provider_v1(
                admission.plan,
                output_directory=root / "materialized-reduced-plan",
                provider=provider,
            )
            self.assertEqual(result.record_count, 16)
            self.assertTrue(result.test_only)
            run = json.loads(result.run_receipt_path.read_bytes())
            self.assertEqual(run["sigma_indices"], list(canary.SIGMA_INDICES))
            self.assertEqual(
                len(run["representation_sigma_evidence_candidates"]), 4
            )
            self.assertEqual(
                len(run["amplitude_sigma_calibration_candidates"]), 4
            )

            different_root = root / "distinct-origin-plan"
            different_root.mkdir()
            different_plan, different_rows = (
                _official_reduced_materializer_plan(
                    different_root, same_origin=False
                )
            )
            different_schedule = canary.build_checkpoint_scaffold_schedule_v1(
                different_rows,
                run_seed=20260815,
                learning_module=learning,
                checkpoint_module=checkpoint,
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "must share one exact origin identity",
            ):
                canary.admit_reduced_materialization_plan_v1(
                    different_plan,
                    schedule=different_schedule,
                    materializer_module=materializer,
                    checkpoint_module=checkpoint,
                    authority_projection_digest="a" * 64,
                    parent_manifest_file_sha256="b" * 64,
                    parent_manifest_digest="c" * 64,
                )

            same_seed_root = root / "same-generation-seed-plan"
            same_seed_root.mkdir()
            same_seed_plan, same_seed_rows = (
                _official_reduced_materializer_plan(
                    same_seed_root, same_generation_seed=True
                )
            )
            same_seed_schedule = canary.build_checkpoint_scaffold_schedule_v1(
                same_seed_rows,
                run_seed=20260815,
                learning_module=learning,
                checkpoint_module=checkpoint,
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "generation seed is reused|Gaussian .*reused",
            ):
                canary.admit_reduced_materialization_plan_v1(
                    same_seed_plan,
                    schedule=same_seed_schedule,
                    materializer_module=materializer,
                    checkpoint_module=checkpoint,
                    authority_projection_digest="a" * 64,
                    parent_manifest_file_sha256="b" * 64,
                    parent_manifest_digest="c" * 64,
                )

    def test_reduced_plan_rejects_resigned_cross_cell_gaussian_reuse(
        self,
    ) -> None:
        cases = (
            (
                "raw-only",
                frozenset({"raw_value_sha256"}),
                "Gaussian raw SHA-256 is reused",
            ),
            (
                "content-only",
                frozenset({"content_sha256"}),
                "Gaussian content SHA-256 is reused",
            ),
            (
                "raw-and-content",
                frozenset({"raw_value_sha256", "content_sha256"}),
                "Gaussian raw SHA-256 is reused",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for case_name, reused_fields, expected_error in cases:
                with self.subTest(case=case_name):
                    case_root = root / case_name
                    case_root.mkdir()
                    plan, rows = _official_reduced_materializer_plan(
                        case_root,
                        reuse_gaussian_fields_across_cells=reused_fields,
                    )
                    seed_values_by_cell: dict[str, set[int]] = {}
                    for record in plan["records"]:
                        if (
                            record["record_kind"] != "teacher_anchor"
                            or record["evidence_role"] != "teacher_origin"
                        ):
                            continue
                        authority = json.loads(
                            Path(
                                record["target_clean_latent_authority"]["path"]
                            ).read_bytes()
                        )
                        candidate = json.loads(
                            Path(authority["pair_v5_candidate"]["path"]).read_bytes()
                        )
                        seed_values_by_cell.setdefault(
                            str(record["teacher_cell_id"]), set()
                        ).add(int(candidate["candidate"]["seed"]))
                    self.assertEqual(
                        set(seed_values_by_cell),
                        {"canary-cell-0", "canary-cell-1"},
                    )
                    self.assertEqual(
                        {next(iter(values)) for values in seed_values_by_cell.values()},
                        {7000, 7001},
                    )
                    self.assertTrue(
                        all(len(values) == 1 for values in seed_values_by_cell.values())
                    )
                    schedule = canary.build_checkpoint_scaffold_schedule_v1(
                        rows,
                        run_seed=20260815,
                        learning_module=learning,
                        checkpoint_module=checkpoint,
                    )
                    with self.assertRaisesRegex(
                        canary.Full30MechanismCanaryAuthorityError,
                        expected_error,
                    ):
                        canary.admit_reduced_materialization_plan_v1(
                            plan,
                            schedule=schedule,
                            materializer_module=materializer,
                            checkpoint_module=checkpoint,
                            authority_projection_digest="a" * 64,
                            parent_manifest_file_sha256="b" * 64,
                            parent_manifest_digest="c" * 64,
                        )

    def test_reduced_data_and_amplitude_reopen_formal_physical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data_manifest = _reduced_data_manifest(root)
            data_path, data_sha = _write_manifest(
                root / "canary-data.json", data_manifest
            )
            amplitude_manifest = _reduced_amplitude_manifest(
                root,
                parent=data_manifest,
                parent_path=data_path,
                parent_sha=data_sha,
            )
            amplitude_path, amplitude_sha = _write_manifest(
                root / "canary-amplitude.json", amplitude_manifest
            )
            admitted = canary.load_mechanism_canary_authority_v1(
                manifest_path=data_path,
                expected_manifest_sha256=data_sha,
                amplitude_manifest_path=amplitude_path,
                expected_amplitude_manifest_sha256=amplitude_sha,
                materializer_module=materializer,
                checkpoint_module=checkpoint,
                learning_module=learning,
            )
            self.assertEqual(
                admitted.data.validation_receipt["source_units"], 8
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "distinct_teacher_generation_seeds"
                ],
                2,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "distinct_teacher_gaussian_raw_sha256"
                ],
                2,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "distinct_teacher_gaussian_content_sha256"
                ],
                2,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "teacher_pair_v5_candidate_files_reopened"
                ],
                4,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "teacher_native_receipts_reopened"
                ],
                4,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "teacher_fixed_runtime_identity_digests"
                ],
                1,
            )
            self.assertFalse(
                admitted.data.validation_receipt[
                    "action_incomplete_candidate_roots_required_equal"
                ]
            )
            self.assertFalse(
                admitted.data.validation_receipt[
                    "materializer_wrapper_is_seed_truth"
                ]
            )
            self.assertEqual(
                admitted.data.validation_receipt["population_profile"],
                canary.POPULATION_PROFILE,
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "shared_origin_identities"
                ],
                1,
            )
            self.assertTrue(
                admitted.data.validation_receipt[
                    "same_origin_profile_verified"
                ]
            )
            self.assertEqual(
                admitted.data.validation_receipt[
                    "teacher_generation_media_outputs"
                ],
                4,
            )
            self.assertTrue(
                admitted.data.validation_receipt[
                    "teacher_generation_media_outputs_unique"
                ]
            )
            self.assertTrue(
                admitted.data.validation_receipt[
                    "physical_source_index0_reopened"
                ]
            )
            self.assertTrue(
                admitted.amplitude.validation_receipt[
                    "formal_private_amplitude_validators_reused"
                ]
            )
            self.assertEqual(len(admitted.amplitude.floors), 24)
            self.assertFalse(
                admitted.data.validation_receipt[
                    "synthetic_target_index1_bytes_read"
                ]
            )
            self.assertFalse(
                admitted.data.validation_receipt[
                    "synthetic_target_bytes_read"
                ]
            )
            self.assertFalse(
                admitted.data.validation_receipt[
                    "event_family_generalization"
                ]
            )
            self.assertEqual(
                admitted.amplitude.validation_receipt[
                    "population_profile"
                ],
                canary.POPULATION_PROFILE,
            )

            projections = trainer._authority_projection_digests(
                data_manifest,
                amplitude_manifest_sha256=amplitude_sha,
                amplitude_validation_digest=admitted.amplitude.validation_receipt[
                    "validation_digest"
                ],
            )
            runtime_index = trainer.Full30AuthorityRuntimeIndexV1(
                path=data_path,
                expected_sha256=data_sha,
                amplitude_path=amplitude_path,
                expected_amplitude_sha256=amplitude_sha,
                modules=SimpleNamespace(
                    authority=formal_data,
                    amplitude_authority=amplitude,
                    canary_authority=canary,
                    learning=learning,
                    checkpoint=checkpoint,
                    psiout_materializer=materializer,
                    torch=object(),
                ),
                expected_data_sha256=projections["data_sha256"],
                expected_teacher_sha256=projections["teacher_sha256"],
                expected_nuisance_sha256=projections["nuisance_sha256"],
                profile="disposable-canary-2",
            )
            runtime_schedule = runtime_index.build_schedule(run_seed=20260815)
            self.assertTrue(runtime_index.is_disposable_canary)
            self.assertEqual(len(runtime_schedule), 1280)
            self.assertEqual(
                runtime_index.schedule_authority_receipt[
                    "executable_global_indices"
                ],
                list(range(16)),
            )
            executable_coordinates = {
                (
                    str(row.row.teacher_cell_id),
                    str(row.row.branch),
                    int(row.sigma_index),
                )
                for row in runtime_schedule[:16]
            }
            all_evidence_coordinates = {
                (cell, branch, sigma_index)
                for cell, branch in runtime_index.representations
                for sigma_index in canary.SIGMA_INDICES
            }
            non_executable_coordinate = next(
                iter(all_evidence_coordinates - executable_coordinates)
            )
            with self.assertRaisesRegex(
                trainer.Full30ActionTrainingError,
                "non-executable evidence",
            ):
                runtime_index.teacher_packet(
                    teacher_cell_id=non_executable_coordinate[0],
                    branch=non_executable_coordinate[1],
                    sigma_index=non_executable_coordinate[2],
                    device=None,
                )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "serialization-only",
            ):
                trainer.prepare_runtime_record_v1(
                    scheduled=runtime_schedule[16],
                    authority_index=runtime_index,
                    conditions={},
                    noop_condition=None,
                    vae_mean=None,
                    vae_std=None,
                    rope=None,
                    device=None,
                )

            first_source = data_manifest["sources"][0]  # type: ignore[index]
            Path(first_source["source_posterior_index0_path"]).write_bytes(
                b"tampered-after-validation"
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "physical reopen failed",
            ):
                canary.load_data_authority_v1(
                    manifest_path=data_path,
                    expected_manifest_sha256=data_sha,
                )

    def test_final_fit_only_and_global_intrinsic_video_reuse_are_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = _reduced_data_manifest(root)
            representations = manifest["representation_admissions"]
            canary._validate_global_anchor_video_reuse_v1(representations)

            reused = copy.deepcopy(representations)
            reused[1]["cross_anchor_evidence"]["anchor_video_sha256"] = (
                reused[0]["cross_anchor_evidence"]["anchor_video_sha256"]
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "intrinsic identity differs",
            ):
                canary._validate_global_anchor_video_reuse_v1(reused)

            confirmation = copy.deepcopy(manifest)
            cross = confirmation["representation_admissions"][0][
                "cross_anchor_evidence"
            ]
            cross["anchor_split"] = "confirmation"
            cross["pre_admission_blind_review"]["anchor_split"] = (
                "confirmation"
            )
            _resign_nested(
                cross["pre_admission_blind_review"], "review_digest"
            )
            _resign_nested(cross, "evidence_digest")
            _resign_nested(
                confirmation["representation_admissions"][0],
                "admission_digest",
            )
            _resign_nested(confirmation, "manifest_digest")
            confirmation_path, confirmation_sha = _write_manifest(
                root / "confirmation-cross-final.json", confirmation
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "fit-only|differs|validation failed",
            ):
                canary.load_data_authority_v1(
                    manifest_path=confirmation_path,
                    expected_manifest_sha256=confirmation_sha,
                )

            reused_evidence = copy.deepcopy(manifest)
            first_origin = reused_evidence["representation_admissions"][0][
                "origin_evidence"
            ]
            second_origin = reused_evidence["representation_admissions"][2][
                "origin_evidence"
            ]
            second_origin["evidence_id"] = first_origin["evidence_id"]
            second_origin["pre_admission_blind_review"]["evidence_id"] = (
                first_origin["evidence_id"]
            )
            _resign_nested(
                second_origin["pre_admission_blind_review"], "review_digest"
            )
            _resign_nested(second_origin, "evidence_digest")
            _resign_nested(
                reused_evidence["representation_admissions"][2],
                "admission_digest",
            )
            _resign_nested(reused_evidence, "manifest_digest")
            reused_path, reused_sha = _write_manifest(
                root / "reused-origin-evidence.json", reused_evidence
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "evidence|reused|differs",
            ):
                canary.load_data_authority_v1(
                    manifest_path=reused_path,
                    expected_manifest_sha256=reused_sha,
                )

            plan_root = root / "confirmation-record-plan"
            plan_root.mkdir()
            plan, rows = _official_reduced_materializer_plan(plan_root)
            hostile_plan = copy.deepcopy(plan)
            record = next(
                row
                for row in hostile_plan["records"]
                if row["record_kind"] == "teacher_anchor"
                and row["evidence_role"] == "same_event_cross_anchor"
            )
            record["analysis_split"] = "confirmation"
            _resign_nested(record, "record_digest")
            _resign_nested(hostile_plan, "plan_digest")
            schedule = canary.build_checkpoint_scaffold_schedule_v1(
                rows,
                run_seed=20260815,
                learning_module=learning,
                checkpoint_module=checkpoint,
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "confirmation|split|review",
            ):
                canary.admit_reduced_materialization_plan_v1(
                    hostile_plan,
                    schedule=schedule,
                    materializer_module=materializer,
                    checkpoint_module=checkpoint,
                    authority_projection_digest="a" * 64,
                    parent_manifest_file_sha256="b" * 64,
                    parent_manifest_digest="c" * 64,
                )

    def test_final_loader_replays_official_plan_and_rejects_resigned_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data_manifest = _reduced_data_manifest(root)
            data_path, data_sha = _write_manifest(
                root / "bound-data.json", data_manifest
            )
            amplitude_manifest = _reduced_amplitude_manifest(
                root,
                parent=data_manifest,
                parent_path=data_path,
                parent_sha=data_sha,
            )

            fragment_hostile = copy.deepcopy(amplitude_manifest)
            fragment_binding = fragment_hostile["materializer_plan_admission"]
            fragment_binding["run_fragment_binding_digest"] = "f" * 64
            _resign_nested(fragment_binding, "binding_digest")
            _resign_nested(fragment_hostile, "manifest_digest")
            fragment_path, fragment_sha = _write_manifest(
                root / "resigned-fragment-hostile.json", fragment_hostile
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "record-fragment bridge differs",
            ):
                canary.load_mechanism_canary_authority_v1(
                    manifest_path=data_path,
                    expected_manifest_sha256=data_sha,
                    amplitude_manifest_path=fragment_path,
                    expected_amplitude_manifest_sha256=fragment_sha,
                    materializer_module=materializer,
                    checkpoint_module=checkpoint,
                    learning_module=learning,
                )

            admission_hostile = copy.deepcopy(amplitude_manifest)
            admission_binding = admission_hostile[
                "materializer_plan_admission"
            ]
            admission_receipt = json.loads(
                Path(admission_binding["admission_receipt_path"]).read_bytes()
            )
            admission_receipt["record_count"] -= 1
            _resign_nested(admission_receipt, "validation_digest")
            admission_path, admission_sha = _write_manifest(
                root / "resigned-admission-hostile.json", admission_receipt
            )
            admission_binding["admission_receipt_path"] = str(admission_path)
            admission_binding["admission_receipt_file_sha256"] = admission_sha
            admission_binding["admission_validation_digest"] = (
                admission_receipt["validation_digest"]
            )
            _resign_nested(admission_binding, "binding_digest")
            _resign_nested(admission_hostile, "manifest_digest")
            hostile_path, hostile_sha = _write_manifest(
                root / "resigned-admission-manifest.json", admission_hostile
            )
            with self.assertRaisesRegex(
                canary.Full30MechanismCanaryAuthorityError,
                "does not replay exactly",
            ):
                canary.load_mechanism_canary_authority_v1(
                    manifest_path=data_path,
                    expected_manifest_sha256=data_sha,
                    amplitude_manifest_path=hostile_path,
                    expected_amplitude_manifest_sha256=hostile_sha,
                    materializer_module=materializer,
                    checkpoint_module=checkpoint,
                    learning_module=learning,
                )


if __name__ == "__main__":
    unittest.main()
