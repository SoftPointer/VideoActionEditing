from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive.r10b_bernini_pilot_manifest import (
    AUDIT_ROW_SCHEMA,
    BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256,
    BOUNDED_ACTION_NEAR_MISS_TIER,
    QUEUE_ROW_SCHEMA,
    QUEUE_SUMMARY_SCHEMA,
    R10BBerniniPilotError,
    _BOUNDED_ACTION_NEAR_MISS_POLICY,
    _candidate_expansion_evidence,
    _load_queue_commit,
    _screen_gates,
    write_qwen_audit_queue,
)
from motive.r10b_family_qwen_audit import (
    BLIND_SCHEMA,
    DONE_NAME,
    PROMPT_CONTRACT_SHA256,
    RECORDS_NAME,
    SUMMARY_NAME,
    run_audit,
    validate_published_audit,
)
from motive.r10b_qwen_audit_shards import (
    R10BQwenAuditShardsError,
    merge_audits,
    split_queue,
    validate_split,
)
from motive.r10b_tangent_core import canonical_json, file_digest


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct@test-revision"
_AUTHORIZATION = {
    "human_label": False,
    "formal_evidence": False,
    "representation_promoted": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}


def _queue_bytes(rows: list[dict]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _write_full_queue(
    root: Path,
    *,
    rows: int = 6,
    expansion_rows: int = 0,
) -> tuple[Path, Path, list[dict]]:
    if expansion_rows < 0 or expansion_rows > rows:
        raise ValueError("expansion_rows must be within the queue")
    data_root = root / "data"
    data_root.mkdir()
    queue_rows = []
    for index in range(rows):
        iid = f"case-wave-{index:02d}"
        sample = data_root / iid
        sample.mkdir()
        source = sample / "source.mp4"
        target = sample / "target.mp4"
        source.write_bytes(f"source-{index}".encode())
        target.write_bytes(f"target-{index}".encode())
        queue_rows.append(
            {
                "schema_version": QUEUE_ROW_SCHEMA,
                "iid": iid,
                "component_id": f"component-{index:02d}",
                "screen_cell": "positive:wave",
                "screen_role_hint": "positive",
                "intended_family": "wave",
                "canonical_prompt": (
                    "Make the subject wave one forelimb toward the viewer."
                ),
                "prompt_variants": {
                    "canonical": (
                        "Make the subject wave one forelimb toward the viewer."
                    ),
                    "noop": "Keep the video unchanged.",
                    "cross_family_shuffle": "Make the quadruped lie down.",
                    "cross_family_shuffle_family": "quadruped_lie_down",
                },
                "media_binding": {
                    "data_root": str(data_root.resolve()),
                    "src_video": {
                        "relative_path": f"{iid}/source.mp4",
                        "sha256": file_digest(source),
                    },
                    "tgt_video": {
                        "relative_path": f"{iid}/target.mp4",
                        "sha256": file_digest(target),
                    },
                },
                "authorization": dict(_AUTHORIZATION),
            }
        )
    expanded = queue_rows[-expansion_rows:] if expansion_rows else []
    for row in expanded:
        metrics = {
            "paired_track_camera_crossfit_valid": True,
            "source_stabilized_motion_p90": 0.001,
            "target_stabilized_motion_p90": 0.001,
            "target_raw_motion_p90": 0.001,
            "edit_delta_p90": 0.0022,
            "source_visibility_mean": 0.6,
            "target_visibility_mean": 0.6,
            "paired_visibility_mean": 0.6,
            "target_visibility_drop": 0.0,
            "camera_crossfit_residual_median_max": 0.0015,
            "target_camera_residual_reduction": 0.0,
            "target_acceleration_p90": 0.0001,
            "target_acceleration_to_speed_p90": 0.1,
        }
        gates = _screen_gates(metrics)
        assert gates["action"]["pass"] is False
        row.update(
            {
                "upstream_label": {"class": "positive"},
                "feature_metrics": metrics,
                "feature_gates": gates,
                "motion_gate_applicable": True,
                "motion_gate_pass": False,
                "candidate_expansion_tier": BOUNDED_ACTION_NEAR_MISS_TIER,
                "candidate_expansion_policy_sha256": (
                    BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256
                ),
                "candidate_expansion_check_evidence": (
                    _candidate_expansion_evidence(
                        metrics,
                        strict_action_pass=False,
                    )
                ),
                "audit_only": True,
                "final_pilot_eligible": False,
            }
        )
    raw = _queue_bytes(queue_rows)
    summary = {
        "schema_version": QUEUE_SUMMARY_SCHEMA,
        "experiment_role": "qwen_audit_queue_only",
        "selection_seed": 17,
        "audit_oversample": 1,
        "rows": len(queue_rows),
        "unique_components": len(queue_rows),
        "component_disjoint": True,
        "screen_cell_counts": {"positive:wave": len(queue_rows)},
        "screen_shortfalls": {},
        "thresholds": {"action": {"motion": 0.25}},
        "qwen_audit": {
            "schema_version": AUDIT_ROW_SCHEMA,
            "qwen_model_id": MODEL_ID,
            "qwen_prompt_sha256": PROMPT_CONTRACT_SHA256,
            "semantic_precedence": ["effect", "camera", "static", "positive"],
        },
        "inputs": {"fixture": "immutable"},
        "queue_sha256": hashlib.sha256(raw).hexdigest(),
        "exclusion_counts": {},
        "video_bytes_copied": False,
        "authorization": dict(_AUTHORIZATION),
    }
    if expansion_rows:
        summary["candidate_expansion"] = {
            "tier": BOUNDED_ACTION_NEAR_MISS_TIER,
            "policy": dict(_BOUNDED_ACTION_NEAR_MISS_POLICY),
            "policy_sha256": BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256,
            "eligible_before_component_dedup": expansion_rows,
            "selected_rows": expansion_rows,
            "selected_cell_counts": {"positive:wave": expansion_rows},
            "audit_only": True,
            "final_pilot_eligible": False,
        }
    queue_dir = root / "full_queue"
    write_qwen_audit_queue(
        {"rows": queue_rows, "summary": summary},
        queue_dir,
    )
    return queue_dir, data_root, queue_rows


def _blind() -> dict:
    return {
        "schema_version": BLIND_SCHEMA,
        "subject_morphology": "adult_human",
        "source_wave": {
            "limb_part": "none",
            "event_frames": [],
            "direction_sequence": [],
            "directed_toward_viewer": "unclear",
        },
        "target_wave": {
            "limb_part": "arm",
            "event_frames": [2, 5, 8],
            "direction_sequence": ["left", "right", "left"],
            "directed_toward_viewer": "yes",
        },
        "source_lie_down": {
            "start_posture": "unclear",
            "start_frame": -1,
            "lowering_frame": -1,
            "final_frame": -1,
            "final_posture": "unclear",
        },
        "target_lie_down": {
            "start_posture": "unclear",
            "start_frame": -1,
            "lowering_frame": -1,
            "final_frame": -1,
            "final_posture": "unclear",
        },
        "source_actor_motion": "none",
        "target_actor_motion": "clear",
        "camera_motion": "none",
        "background_motion": "none",
        "artifact_level": "none",
        "preservation_quality": "acceptable",
        "identity_appearance_change": "none",
        "nonphysical_effect": "none",
        "deformation": "none",
        "flicker": "none",
        "reflection_or_sunglasses_artifact": "none",
        "secondary_action": "none",
        "uncertainty_codes": [],
    }


class _FakeBackend:
    model_revision = "fake-model-revision"
    transformers_version = "fake-transformers"

    def __init__(self, **kwargs) -> None:
        self.text_calls = 0
        self.visual_calls = 0

    def generate_visual_observation(self, **kwargs) -> tuple[str, str]:
        self.visual_calls += 1
        return canonical_json(_blind()), "a" * 64

    def generate_text(self, **kwargs) -> str:
        self.text_calls += 1
        raise AssertionError("deterministic stage 2 must not call Qwen text")


class _ScriptedBackend(_FakeBackend):
    def __init__(self, *, script: list[object], **kwargs) -> None:
        super().__init__(**kwargs)
        self.script = list(script)

    def generate_visual_observation(self, **kwargs) -> tuple[str, str]:
        self.visual_calls += 1
        outcome = self.script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if not isinstance(outcome, str):  # pragma: no cover - fixture guard
            raise AssertionError("script outcome must be text or exception")
        return outcome, "a" * 64


class R10BQwenAuditShardsTests(unittest.TestCase):
    def test_round_robin_split_is_exact_and_preserves_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _data_root, full_rows = _write_full_queue(root)
            shards = root / "shard_queues"
            result = split_queue(
                full_queue_dir=queue_dir,
                output_root=shards,
                shard_count=3,
            )
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["strategy"], "round_robin")
            expected = ([0, 3], [1, 4], [2, 5])
            _rows, full_summary, _files = _load_queue_commit(queue_dir)
            for shard_index, indices in enumerate(expected):
                rows, summary, _files = _load_queue_commit(
                    shards / f"shard_{shard_index:03d}"
                )
                self.assertEqual(rows, [full_rows[index] for index in indices])
                for field in ("qwen_audit", "inputs", "thresholds"):
                    self.assertEqual(summary[field], full_summary[field])
                self.assertEqual(
                    summary["shard"]["full_queue_indices"],
                    list(indices),
                )
            validated = validate_split(
                shards,
                full_queue_dir=queue_dir,
            )
            self.assertEqual(validated["rows"], 6)

    def test_split_recomputes_candidate_expansion_summary_per_shard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _data_root, full_rows = _write_full_queue(
                root,
                rows=8,
                expansion_rows=2,
            )
            shards = root / "expanded_shards"
            split_queue(
                full_queue_dir=queue_dir,
                output_root=shards,
                shard_count=3,
            )
            expected = ([0, 3, 6], [1, 4, 7], [2, 5])
            total_expansion_rows = 0
            for shard_index, indices in enumerate(expected):
                rows, summary, _files = _load_queue_commit(
                    shards / f"shard_{shard_index:03d}"
                )
                self.assertEqual(rows, [full_rows[index] for index in indices])
                expected_expansion = sum(
                    row.get("candidate_expansion_tier")
                    == BOUNDED_ACTION_NEAR_MISS_TIER
                    for row in rows
                )
                expansion = summary["candidate_expansion"]
                self.assertEqual(
                    expansion["selected_rows"],
                    expected_expansion,
                )
                self.assertEqual(
                    expansion["eligible_before_component_dedup"],
                    expected_expansion,
                )
                self.assertEqual(
                    expansion["selected_cell_counts"],
                    (
                        {"positive:wave": expected_expansion}
                        if expected_expansion
                        else {}
                    ),
                )
                self.assertEqual(
                    expansion["scope"],
                    "shard_selected_rows",
                )
                total_expansion_rows += expected_expansion
            self.assertEqual(total_expansion_rows, 2)
            self.assertEqual(
                validate_split(
                    shards,
                    full_queue_dir=queue_dir,
                )["rows"],
                8,
            )

    def test_balanced_contiguous_split_and_invalid_shard_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _data_root, full_rows = _write_full_queue(root, rows=7)
            shards = root / "balanced"
            split_queue(
                full_queue_dir=queue_dir,
                output_root=shards,
                shard_count=3,
                strategy="balanced_contiguous",
            )
            expected = ([0, 1, 2], [3, 4], [5, 6])
            for shard_index, indices in enumerate(expected):
                rows, _summary, _files = _load_queue_commit(
                    shards / f"shard_{shard_index:03d}"
                )
                self.assertEqual(rows, [full_rows[index] for index in indices])
            with self.assertRaisesRegex(
                R10BQwenAuditShardsError,
                "between one",
            ):
                split_queue(
                    full_queue_dir=queue_dir,
                    output_root=root / "too_many",
                    shard_count=8,
                )

    def test_queue_and_split_reject_nonclosed_artifact_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, _data_root, _full_rows = _write_full_queue(root)

            # Reproduce the dynamic contamination that previously passed both
            # the queue loader and split entry point.
            extra = queue_dir / "EXTRA.txt"
            extra.write_text("not part of the queue commit", encoding="utf-8")
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "queue directory closure differs",
            ):
                _load_queue_commit(queue_dir)
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "queue directory closure differs",
            ):
                split_queue(
                    full_queue_dir=queue_dir,
                    output_root=root / "must_not_publish",
                    shard_count=3,
                )
            self.assertFalse((root / "must_not_publish").exists())
            extra.unlink()

            extra_dir = queue_dir / "extra_dir"
            extra_dir.mkdir()
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "queue directory closure differs",
            ):
                _load_queue_commit(queue_dir)
            extra_dir.rmdir()

            done_path = queue_dir / "done.json"
            done_target = root / "done-target.json"
            done_path.replace(done_target)
            done_path.symlink_to(done_target)
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "regular non-symlink files",
            ):
                _load_queue_commit(queue_dir)
            done_path.unlink()
            done_target.replace(done_path)

            shards = root / "shard_queues"
            split_queue(
                full_queue_dir=queue_dir,
                output_root=shards,
                shard_count=3,
            )
            shard_extra = shards / "shard_000" / "EXTRA.txt"
            shard_extra.write_text(
                "not part of the shard queue commit",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "queue directory closure differs",
            ):
                validate_split(shards, full_queue_dir=queue_dir)
            shard_extra.unlink()

            root_extra = shards / "EXTRA.txt"
            root_extra.write_text(
                "not part of the split-tree commit",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                R10BQwenAuditShardsError,
                "shard split root closure differs",
            ):
                validate_split(shards, full_queue_dir=queue_dir)

    def test_three_shard_audits_merge_to_original_validator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, full_rows = _write_full_queue(root)
            shard_queues = root / "shard_queues"
            split_queue(
                full_queue_dir=queue_dir,
                output_root=shard_queues,
                shard_count=3,
            )
            shard_audits = root / "shard_audits"
            shard_audits.mkdir()
            model = root / "model"
            model.mkdir()
            instances: list[_FakeBackend] = []

            def factory(**kwargs) -> _FakeBackend:
                instance = _FakeBackend(**kwargs)
                instances.append(instance)
                return instance

            for shard_index in range(3):
                name = f"shard_{shard_index:03d}"
                result = run_audit(
                    queue_dir=shard_queues / name,
                    data_root=data_root,
                    model_path=model,
                    output_dir=shard_audits / name,
                    nframes=12,
                    backend_factory=factory,
                )
                self.assertEqual(result["status"], "VALID")
            self.assertEqual(
                sum(instance.visual_calls for instance in instances),
                len(full_rows),
            )
            self.assertEqual(
                sum(instance.text_calls for instance in instances),
                0,
            )

            merged = root / "merged"
            result = merge_audits(
                full_queue_dir=queue_dir,
                shard_queues_root=shard_queues,
                shard_audits_root=shard_audits,
                output_dir=merged,
            )
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["rows"], len(full_rows))
            self.assertEqual(
                result["hard_role_counts"],
                {"positive": len(full_rows)},
            )
            validated = validate_published_audit(merged)
            self.assertEqual(validated["status"], "VALID")
            records = [
                json.loads(line)
                for line in (merged / RECORDS_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [record["iid"] for record in records],
                [row["iid"] for row in full_rows],
            )
            self.assertEqual(
                Counter(record["screen_cell"] for record in records),
                Counter(row["screen_cell"] for row in full_rows),
            )

    def test_merge_preserves_partial_generation_failure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_dir, data_root, _full_rows = _write_full_queue(
                root, rows=4
            )
            shard_queues = root / "shard_queues"
            split_queue(
                full_queue_dir=queue_dir,
                output_root=shard_queues,
                shard_count=2,
            )
            shard_audits = root / "shard_audits"
            shard_audits.mkdir()
            model = root / "model"
            model.mkdir()

            partial = run_audit(
                queue_dir=shard_queues / "shard_000",
                data_root=data_root,
                model_path=model,
                output_dir=shard_audits / "shard_000",
                backend_factory=lambda **kwargs: _ScriptedBackend(
                    script=[
                        canonical_json(_blind()),
                        RuntimeError("CUDA out of memory"),
                    ],
                    **kwargs,
                ),
            )
            self.assertEqual(
                partial["status"], "PARTIAL_GENERATION_FAILURE"
            )
            complete = run_audit(
                queue_dir=shard_queues / "shard_001",
                data_root=data_root,
                model_path=model,
                output_dir=shard_audits / "shard_001",
                backend_factory=_FakeBackend,
            )
            self.assertEqual(complete["status"], "VALID")

            merged = root / "merged"
            result = merge_audits(
                full_queue_dir=queue_dir,
                shard_queues_root=shard_queues,
                shard_audits_root=shard_audits,
                output_dir=merged,
            )
            self.assertEqual(
                result["status"], "PARTIAL_GENERATION_FAILURE"
            )
            self.assertEqual(result["successful_rows"], 3)
            self.assertEqual(result["schema_error_rows"], 0)
            self.assertEqual(result["generation_error_rows"], 1)
            for name in (SUMMARY_NAME, DONE_NAME):
                metadata = json.loads(
                    (merged / name).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    metadata["status"], "partial_generation_failure"
                )
                self.assertEqual(metadata["successful_rows"], 3)
                self.assertEqual(metadata["schema_error_rows"], 0)
                self.assertEqual(metadata["generation_error_rows"], 1)


if __name__ == "__main__":
    unittest.main()
