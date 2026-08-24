from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from motive.r7_p1_diagnostic import (
    DiagnosticGateConfig,
    DownstreamAuditConfig,
    P1DiagnosticConfig,
    _audit_defaults,
    _commit,
    _evaluate_side,
    _selection_score,
    compute_p1_gate,
    deterministic_downstream_perturbation,
    evaluate_cache,
    load_final_cache,
    run_diagnostic,
    validate_diagnostic_rows,
)
from motive.r7_coherent_actor import (
    R7_COHERENT_ACTOR_SCHEMA,
    CoherentActorConfig,
)
from motive.r7_preflight_extract import (
    R7_VIDEO_SAMPLING,
    _file_digest,
    _object_digest,
)
from motive.r7_temporal_teacher import TemporalTeacherConfig
from motive.r7_track_cache import (
    FINAL_WORLD_SIZE,
    R7_TRACK_CACHE_ROW_SCHEMA,
    _commit as _cache_commit,
    _empty_arrays as _empty_cache_arrays,
    _rank_directory,
    build_cache_contract,
    cotracker_source_provenance,
    finalize_shards,
)


def _tracks(*, coherent: bool, track_count: int = 16) -> np.ndarray:
    times = np.linspace(0.0, 1.24, 32)
    tracks = np.zeros((32, track_count, 2), dtype=np.float32)
    columns = int(np.ceil(np.sqrt(track_count)))
    for index in range(track_count):
        tracks[:, index, 0] = 0.12 + 0.18 * (index % columns)
        tracks[:, index, 1] = 0.12 + 0.18 * (index // columns)
    if coherent:
        # Three spatially local tracks move together after compensation.
        members = (0, 1, 4)
        displacement = np.linspace(0.0, 0.12, len(times))
        tracks[:, members, 0] += displacement[:, None]
    return tracks


def _cache(
    *,
    positives: list[bool],
    coherent: bool = False,
    camera_valid: list[bool] | None = None,
) -> dict[str, object]:
    row_count = len(positives)
    track_count = 16
    valid = (
        np.ones(row_count, dtype=bool)
        if camera_valid is None
        else np.asarray(camera_valid, dtype=bool)
    )
    arrays: dict[str, np.ndarray] = {
        "positive": np.asarray(positives, dtype=bool),
    }
    times = np.linspace(0.0, 1.24, 32, dtype=np.float64)
    for side in ("source", "target"):
        arrays[f"{side}_camera_valid"] = valid.copy()
        arrays[f"{side}_stabilized_tracks"] = np.stack(
            [
                _tracks(coherent=coherent, track_count=track_count)
                if value
                else np.zeros((32, track_count, 2), dtype=np.float32)
                for value in valid
            ]
        )
        arrays[f"{side}_visibility"] = np.stack(
            [
                np.ones((32, track_count), dtype=np.float32)
                if value
                else np.zeros((32, track_count), dtype=np.float32)
                for value in valid
            ]
        )
        arrays[f"{side}_frame_times"] = np.stack(
            [times if value else np.zeros(32) for value in valid]
        )
    rows = []
    for index, positive in enumerate(positives):
        negative_type = None if positive else "static"
        rows.append(
            {
                "input_index": index,
                "iid": f"iid-{index:03d}",
                "label_type": "positive" if positive else "negative",
                "negative_type": negative_type,
                "action_signature": "walk" if positive else "negative:static",
            }
        )
    return {
        "rows": rows,
        "arrays": arrays,
        "contract": {"tracker": {"track_count": track_count}},
    }


def _input_row(index: int) -> dict[str, object]:
    positive = index < 100
    return {
        "iid": f"real-cache-{index:03d}",
        "src_video": f"videos/{index:03d}/source.mp4",
        "tgt_video": f"videos/{index:03d}/target.mp4",
        "input_digest": f"digest-{index:03d}",
        "prompt": f"action {index}",
        "r5_pilot_label": {
            "class": "positive" if positive else "negative",
            "negative_type": None if positive else "static",
            "action_signature": "walk" if positive else "negative:static",
        },
    }


def _runtime(seed: int) -> dict[str, object]:
    return {
        "schema_version": "motive-r7-runtime-v1",
        "python_version": "3.12.0",
        "python_implementation": "CPython",
        "python_executable": "/fake/python",
        "platform": "test-linux",
        "numpy_version": np.__version__,
        "opencv_version": "test",
        "torch_version": "test",
        "device_type": "cuda-hip",
        "device_name": "unit-test-device",
        "torch_hip_version": "unit-test-hip",
        "visible_device_count": FINAL_WORLD_SIZE,
        "device_total_memory": 1,
        "device_capability": [9, 0],
        "torch_build_config_sha256": "c" * 64,
        "determinism": {
            "schema_version": "motive-r7-determinism-v1",
            "seed": seed,
            "rank_seed_policy": (
                "identical-base-seed-on-all-eight-ranks-v1"
            ),
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_cpu_seeded": True,
            "torch_all_visible_devices_seeded": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "python_hash_seed": str(seed),
        },
    }


def _cache_arrays(
    input_rows: list[dict[str, object]],
    indices: list[int],
) -> dict[str, np.ndarray]:
    track_count = 4
    arrays = _empty_cache_arrays(len(indices), track_count=track_count)
    arrays["input_indices"][:] = indices
    arrays["positive"][:] = [
        input_rows[index]["r5_pilot_label"]["class"] == "positive"
        for index in indices
    ]
    times = np.arange(32, dtype=np.float64) / 25.0
    frame_indices = np.arange(32, dtype=np.int64)
    tracks = np.zeros((32, track_count, 2), dtype=np.float32)
    tracks[..., 0] = np.linspace(0.2, 0.8, track_count)[None]
    tracks[..., 1] = np.linspace(0.3, 0.7, track_count)[None]
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    for side in ("source", "target"):
        arrays[f"{side}_track_valid"][:] = True
        arrays[f"{side}_camera_valid"][:] = True
        arrays[f"{side}_normalized_tracks"][:] = tracks
        arrays[f"{side}_stabilized_tracks"][:] = tracks
        arrays[f"{side}_visibility"][:] = 1.0
        arrays[f"{side}_frame_times"][:] = times
        arrays[f"{side}_source_frame_indices"][:] = frame_indices
        arrays[f"{side}_resized_size"][:] = [100, 200]
        arrays[f"{side}_source_fps"][:] = 25.0
        arrays[f"{side}_transition_affines"][:] = identity
        arrays[f"{side}_cumulative_affines"][:] = identity
    return arrays


def _cache_row(
    *,
    input_row: dict[str, object],
    input_index: int,
    local_index: int,
    rank: int,
    data_root: Path,
    checkpoint: Path,
) -> dict[str, object]:
    decode = {
        "sampling_version": R7_VIDEO_SAMPLING,
        "decoded_frames": 32,
        "source_frame_indices": list(range(32)),
        "source_fps": 25.0,
        "source_frame_count": 32,
        "source_size": [100, 200],
        "resized_size": [100, 200],
    }

    def side(field: str) -> dict[str, object]:
        resolved_path = (data_root / str(input_row[field])).resolve()
        return {
            "status": "camera_ready",
            "track_valid": True,
            "camera_valid": True,
            "failure_stage": None,
            "failure_reason": None,
            "failure_message": None,
            "resolved_path": str(resolved_path),
            "video_sha256": _file_digest(resolved_path),
            "decode": dict(decode),
            "tracker": {
                "backend": "cotracker",
                "provenance": {
                    "checkpoint": str(checkpoint.resolve()),
                    "grid_size": 2,
                    "query_frame": 0,
                    "backward_tracking": False,
                    "device": f"cuda:{rank}",
                },
                "tracks": 4,
            },
            "camera_crossfit": {
                "valid": False,
                "raw_median": 0.0,
                "residual_median": 0.0,
                "residual_reduction": 0.0,
            },
        }

    pilot = input_row["r5_pilot_label"]
    positive = pilot["class"] == "positive"
    return {
        "schema_version": R7_TRACK_CACHE_ROW_SCHEMA,
        "input_index": input_index,
        "shard_array_index": local_index,
        "shard_rank": rank,
        "world_size": FINAL_WORLD_SIZE,
        "iid": input_row["iid"],
        "input_row": dict(input_row),
        "input_row_sha256": _object_digest(input_row),
        "input_digest": input_row.get("input_digest"),
        "prompt": input_row.get("prompt"),
        "label_type": pilot["class"],
        "negative_type": pilot.get("negative_type"),
        "positive": positive,
        "action_signature": pilot.get("action_signature"),
        "source": side("src_video"),
        "target": side("tgt_video"),
        "paired_track_valid": True,
        "paired_camera_valid": True,
    }


def _build_semantic_eight_shard_cache(
    root: Path,
) -> tuple[Path, Path]:
    manifest = root / "input.jsonl"
    data_root = root / "data"
    cache_root = root / "cache"
    checkpoint = root / "tracker.pth"
    data_root.mkdir()
    checkpoint.write_bytes(b"unit-test-cotracker")
    input_rows = [_input_row(index) for index in range(181)]
    for row in input_rows:
        for field in ("src_video", "tgt_video"):
            video = data_root / str(row[field])
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(
                f"{row['iid']}:{field}".encode("utf-8")
            )
    with manifest.open("w", encoding="utf-8") as handle:
        for row in input_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    source_root = root / "cotracker-source"
    predictor = source_root / "cotracker" / "predictor.py"
    predictor.parent.mkdir(parents=True)
    predictor.write_text("class CoTrackerPredictor: pass\n", encoding="utf-8")
    subprocess.run(
        ("git", "init", str(source_root)),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ("git", "-C", str(source_root), "add", "cotracker/predictor.py"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(source_root),
            "-c",
            "user.name=R7 Unit Test",
            "-c",
            "user.email=r7@example.invalid",
            "commit",
            "-m",
            "freeze test source",
        ),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    source = cotracker_source_provenance(source_root)
    seed = 23
    for rank in range(FINAL_WORLD_SIZE):
        indices = list(range(rank, len(input_rows), FINAL_WORLD_SIZE))
        arrays = _cache_arrays(input_rows, indices)
        rows = [
            _cache_row(
                input_row=input_rows[input_index],
                input_index=input_index,
                local_index=local_index,
                rank=rank,
                data_root=data_root,
                checkpoint=checkpoint,
            )
            for local_index, input_index in enumerate(indices)
        ]
        contract = build_cache_contract(
            input_manifest=manifest,
            data_root=data_root,
            tracker_checkpoint=checkpoint,
            cotracker_provenance=source,
            runtime=_runtime(seed),
            tracker_grid_size=2,
            rank=rank,
            world_size=FINAL_WORLD_SIZE,
            device=f"cuda:{rank}",
            seed=seed,
        )
        _cache_commit(
            directory=_rank_directory(
                cache_root, rank, FINAL_WORLD_SIZE
            ),
            rows=rows,
            arrays=arrays,
            contract=contract,
            input_rows=len(input_rows),
            final=False,
        )
    finalize_shards(
        input_manifest=manifest,
        output_root=cache_root,
    )
    return manifest, cache_root


def _gate_row(
    *,
    positive: bool,
    target_ready: bool,
    source_ready: bool,
    audit_pass: bool,
    score: float,
    negative_type: str | None = None,
    camera_valid: bool = True,
) -> dict[str, object]:
    return {
        "positive": positive,
        "negative_type": negative_type,
        "target_camera_valid": camera_valid,
        "source": {"diagnostic_ready": source_ready, "score": score},
        "target": {"diagnostic_ready": target_ready, "score": score},
        "target_audit": {"joint_pass": audit_pass},
    }


def _passing_gate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(100):
        rows.append(
            _gate_row(
                positive=True,
                target_ready=index < 65,
                source_ready=index < 50,
                audit_pass=index < 70,
                score=2.0,
            )
        )
    for index in range(40):
        rows.append(
            _gate_row(
                positive=False,
                target_ready=index < 4,
                source_ready=False,
                audit_pass=False,
                score=0.0,
                negative_type=(
                    "static" if index % 2 == 0 else "endpoint_only"
                ),
            )
        )
    return rows


class P1PerturbationAndAuditTests(unittest.TestCase):
    def test_rejected_component_score_never_enters_gate(self) -> None:
        rejected = SimpleNamespace(
            diagnostic_ready=False,
            selected_component=None,
            components=[
                SimpleNamespace(accepted=False, selection_score=999.0)
            ],
        )
        self.assertEqual(_selection_score(rejected), 0.0)
        accepted = SimpleNamespace(
            diagnostic_ready=True,
            selected_component=0,
            components=[
                SimpleNamespace(accepted=True, selection_score=0.125)
            ],
        )
        self.assertEqual(_selection_score(accepted), 0.125)

    def test_component_energy_localizes_zero_centroid_shape_motion(self) -> None:
        phase_steps = 32
        component = SimpleNamespace(accepted=True, selection_score=0.2)
        selection = SimpleNamespace(
            diagnostic_ready=True,
            failure_reason=None,
            failure_detail=None,
            components=(component,),
            selected_component=0,
            actor_track_mask=np.asarray(
                [True, True, True, False, False, False, False, False]
            ),
            actor_track_indices=np.asarray([0, 1, 2], dtype=np.int64),
            # Symmetric shape change: the aggregate centroid is stationary.
            actor_trajectory=np.zeros((phase_steps, 2), dtype=np.float32),
            actor_track_trajectories=np.zeros(
                (3, phase_steps, 2), dtype=np.float32
            ),
            actor_track_phase_mask=np.ones(
                (3, phase_steps), dtype=bool
            ),
            phase_times=np.linspace(0.0, 1.24, phase_steps),
            phase_energy=np.concatenate(
                (
                    np.zeros(8),
                    np.ones(16) * 0.2,
                    np.zeros(8),
                )
            ).astype(np.float32),
            phase_visibility=np.ones(phase_steps, dtype=np.float32),
            to_summary=lambda: {
                "schema_version": (
                    R7_COHERENT_ACTOR_SCHEMA
                ),
                "actor_track_indices": [0, 1, 2],
                "components": [],
            },
        )
        with patch(
            "motive.r7_p1_diagnostic.select_coherent_actor",
            return_value=selection,
        ):
            record = _evaluate_side(
                np.zeros((32, 8, 2), dtype=np.float32),
                np.ones((32, 8), dtype=np.float32),
                np.linspace(0.0, 1.24, 32),
                selector_config=CoherentActorConfig(),
                event_config=TemporalTeacherConfig(),
            )
        self.assertTrue(record["diagnostic_ready"])
        self.assertEqual(max(record["actor_phase_speed"]), 0.0)
        self.assertGreater(max(record["event_transition_energy"]), 0.0)
        self.assertIsNotNone(record["event_window"])

    def test_perturbation_is_deterministic_and_applies_all_four(self) -> None:
        tracks = _tracks(coherent=True)
        visibility = np.ones((32, 16), dtype=np.float32)
        times = np.linspace(0.0, 1.24, 32)
        config = DownstreamAuditConfig()
        first = deterministic_downstream_perturbation(
            tracks, visibility, times, seed=17, config=config
        )
        second = deterministic_downstream_perturbation(
            tracks, visibility, times, seed=17, config=config
        )
        for left, right in zip(first[:3], second[:3]):
            self.assertTrue(np.array_equal(left, right))
        provenance = first[3]
        self.assertGreater(provenance["track_drop_count"], 0)
        self.assertGreater(provenance["visibility_drop_count"], 0)
        self.assertFalse(np.array_equal(first[0], tracks))
        self.assertFalse(np.array_equal(first[2], times))

    def test_base_rejection_does_not_mask_positive_target_audit(self) -> None:
        # Static tracks are rejected by the base selector, but the audit still
        # runs and receives zero joint credit.
        cache = _cache(positives=[True], coherent=False)
        rows = evaluate_cache(cache=cache, config=P1DiagnosticConfig())
        self.assertFalse(rows[0]["target"]["diagnostic_ready"])
        audit = rows[0]["target_audit"]
        self.assertTrue(audit["eligible"])
        self.assertTrue(audit["performed"])
        self.assertIsNotNone(audit["perturbation"])
        self.assertFalse(audit["comparison_available"])
        self.assertEqual(audit["metrics"], _audit_defaults())
        self.assertFalse(audit["joint_pass"])

    def test_missing_audit_cannot_gain_credit(self) -> None:
        cache = _cache(positives=[True], coherent=False)
        rows = evaluate_cache(cache=cache, config=P1DiagnosticConfig())
        rows[0]["target_audit"]["metrics"]["actor_mask_iou"] = 1.0
        with self.assertRaisesRegex(ValueError, "gained credit"):
            validate_diagnostic_rows(
                rows,
                cache=cache,
                config=P1DiagnosticConfig(),
            )


class P1GateTests(unittest.TestCase):
    def test_exact_frozen_development_boundaries_pass(self) -> None:
        gate = compute_p1_gate(_passing_gate_rows())
        self.assertTrue(gate["diagnostic_gate_passed"])
        self.assertEqual(
            gate["criteria"]["positive_target_ready_fraction"]["value"],
            0.65,
        )
        self.assertEqual(
            gate["criteria"][
                "positive_target_audit_joint_pass_fraction"
            ]["value"],
            0.70,
        )
        self.assertEqual(
            gate["criteria"]["no_action_false_event_fraction"]["value"],
            0.10,
        )

    def test_every_gate_axis_fails_closed(self) -> None:
        mutations = {}

        def low_coverage(rows):
            for row in rows[:15]:
                row["target_camera_valid"] = False

        mutations["target_camera_cache_coverage"] = low_coverage

        def low_ready(rows):
            rows[64]["target"]["diagnostic_ready"] = False

        mutations["positive_target_ready_fraction"] = low_ready

        def low_paired(rows):
            rows[49]["source"]["diagnostic_ready"] = False

        mutations["paired_positive_ready_count"] = low_paired

        def too_few_negatives(rows):
            for row in rows[129:]:
                row["negative_type"] = "instruction_mismatch"

        mutations["no_action_negative_samples"] = too_few_negatives

        def high_false_event(rows):
            rows[104]["target"]["diagnostic_ready"] = True

        mutations["no_action_false_event_fraction"] = high_false_event

        def low_audit(rows):
            rows[69]["target_audit"]["joint_pass"] = False

        mutations[
            "positive_target_audit_joint_pass_fraction"
        ] = low_audit

        def low_auroc(rows):
            for row in rows[:100]:
                row["target"]["score"] = 0.0
            for row in rows[100:]:
                row["target"]["score"] = 1.0

        mutations["positive_vs_no_action_score_auroc"] = low_auroc

        for criterion, mutate in mutations.items():
            with self.subTest(criterion=criterion):
                rows = copy.deepcopy(_passing_gate_rows())
                mutate(rows)
                gate = compute_p1_gate(rows)
                self.assertFalse(gate["criteria"][criterion]["passed"])
                self.assertFalse(gate["diagnostic_gate_passed"])

    def test_missing_audits_use_all_camera_valid_denominator(self) -> None:
        rows = _passing_gate_rows()
        for row in rows[:50]:
            row["target_audit"]["joint_pass"] = False
        gate = compute_p1_gate(rows)
        criterion = gate["criteria"][
            "positive_target_audit_joint_pass_fraction"
        ]
        self.assertEqual(criterion["denominator"], 100)
        self.assertEqual(criterion["numerator"], 20)
        self.assertEqual(criterion["value"], 0.20)

    def test_instruction_mismatch_is_excluded(self) -> None:
        rows = _passing_gate_rows()
        baseline = compute_p1_gate(rows)
        for _ in range(20):
            rows.append(
                _gate_row(
                    positive=False,
                    target_ready=True,
                    source_ready=True,
                    audit_pass=True,
                    score=1000.0,
                    negative_type="instruction_mismatch",
                )
            )
        gate = compute_p1_gate(rows)
        self.assertEqual(
            gate["counts"]["negative_no_action_rows"],
            baseline["counts"]["negative_no_action_rows"],
        )
        self.assertEqual(
            gate["criteria"]["positive_vs_no_action_score_auroc"]["value"],
            baseline["criteria"][
                "positive_vs_no_action_score_auroc"
            ]["value"],
        )
        self.assertEqual(
            gate["counts"]["negative_instruction_mismatch_excluded"], 20
        )


class P1CommitTests(unittest.TestCase):
    def test_resume_revalidates_bytes_without_re_evaluation_and_tamper_fails(
        self,
    ) -> None:
        cache = _cache(positives=[True], coherent=False)
        contract = {
            "schema_version": "unit-test-contract",
            "formal_status": "INSUFFICIENT",
            "generation_authorized": False,
        }
        config = P1DiagnosticConfig()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "input.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            cache_root = root / "cache"
            cache_root.mkdir()
            output = root / "output"
            with patch(
                "motive.r7_p1_diagnostic.load_final_cache",
                return_value=cache,
            ), patch(
                "motive.r7_p1_diagnostic.build_diagnostic_contract",
                return_value=contract,
            ):
                first = run_diagnostic(
                    input_manifest=manifest,
                    cache_root=cache_root,
                    output_directory=output,
                    config=config,
                )
                self.assertTrue(first["committed"])
                with patch(
                    "motive.r7_p1_diagnostic.evaluate_cache",
                    side_effect=AssertionError("resume reran selector"),
                ):
                    resumed = run_diagnostic(
                        input_manifest=manifest,
                        cache_root=cache_root,
                        output_directory=output,
                        config=config,
                        resume=True,
                    )
                self.assertEqual(resumed, first)
                rows_path = output / "rows.jsonl"
                rows_path.write_bytes(rows_path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "byte digest"):
                    run_diagnostic(
                        input_manifest=manifest,
                        cache_root=cache_root,
                        output_directory=output,
                        config=config,
                        resume=True,
                    )

    def test_commit_refuses_overwrite_and_partial_resume(self) -> None:
        cache = _cache(positives=[False], coherent=False)
        rows = evaluate_cache(cache=cache, config=P1DiagnosticConfig())
        contract = {"schema_version": "unit"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            (output / "rows.jsonl").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                _commit(
                    output_directory=output,
                    rows=rows,
                    contract=contract,
                    cache=cache,
                    config=P1DiagnosticConfig(),
                )


class P1FinalCacheTests(unittest.TestCase):
    def test_semantic_eight_shard_final_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cache_root = _build_semantic_eight_shard_cache(root)
            with patch(
                "motive.r7_p1_diagnostic.validate_commit",
                wraps=__import__(
                    "motive.r7_track_cache",
                    fromlist=["validate_commit"],
                ).validate_commit,
            ) as mocked:
                result = load_final_cache(
                    input_manifest=manifest,
                    cache_root=cache_root,
                )
            self.assertEqual(len(result["rows"]), 181)
            self.assertTrue(
                any(call.kwargs.get("final") for call in mocked.call_args_list)
            )
            self.assertEqual(
                result["contract"]["merge_world_size"], FINAL_WORLD_SIZE
            )

    def test_non_eight_way_final_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cache_root = _build_semantic_eight_shard_cache(root)
            final = cache_root / "final"
            summary_path = final / "summary.json"
            done_path = final / "done.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["contract"]["merge_world_size"] = 7
            summary["contract_sha256"] = _object_digest(summary["contract"])
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["summary_sha256"] = _file_digest(summary_path)
            done["contract_sha256"] = summary["contract_sha256"]
            done_path.write_text(
                json.dumps(done, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "merge contract"):
                load_final_cache(
                    input_manifest=manifest,
                    cache_root=cache_root,
                )

    def test_missing_source_shard_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cache_root = _build_semantic_eight_shard_cache(root)
            shard = _rank_directory(cache_root, 7, FINAL_WORLD_SIZE)
            moved = root / "missing-rank-7"
            shutil.move(str(shard), str(moved))
            with self.assertRaises(FileNotFoundError):
                load_final_cache(
                    input_manifest=manifest,
                    cache_root=cache_root,
                )


if __name__ == "__main__":
    unittest.main()
