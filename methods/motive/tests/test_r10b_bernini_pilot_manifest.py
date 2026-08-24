from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from motive.r10b_bernini_pilot_manifest import (
    AUDIT_ROW_SCHEMA,
    BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256,
    BOUNDED_ACTION_NEAR_MISS_TIER,
    FINAL_QUOTAS,
    MAX_FINAL_ROWS,
    R10BBerniniPilotError,
    derive_qwen_audit_queue,
    finalize_controlled_pilot,
    write_qwen_audit_queue,
)
from motive.r10b_tangent_core import canonical_json, object_digest


MODEL_ID = "Qwen/Qwen2.5-VL-test"
PROMPT_SHA = "a" * 64
DATA_ROOT = (
    "/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.candidate_path = root / "candidates.jsonl"
        self.track_manifest_path = root / "track_manifest.jsonl"
        self.track_cache_path = root / "tracks.npz"
        specifications = [
            *(("positive", "wave") for _ in range(4)),
            *(("positive", "lie_down") for _ in range(4)),
            *(("instruction_mismatch", "wave") for _ in range(2)),
            *(("instruction_mismatch", "lie_down") for _ in range(2)),
            *(("static", "wave") for _ in range(2)),
            *(("static", "lie_down") for _ in range(2)),
            *(("camera_motion", "wave") for _ in range(2)),
            *(("camera_motion", "lie_down") for _ in range(2)),
            *(("artifact", "wave") for _ in range(2)),
            *(("artifact", "lie_down") for _ in range(2)),
        ]
        rows: list[dict] = []
        track_rows: list[dict] = []
        count, frames, tracks = len(specifications), 6, 4
        base = np.zeros((count, frames, tracks, 2), dtype=np.float32)
        base[..., 0] = np.linspace(0.2, 0.8, tracks)
        base[..., 1] = 0.5
        source_stabilized = base.copy()
        target_stabilized = base.copy()
        source_normalized = base.copy()
        target_normalized = base.copy()
        source_visibility = np.ones((count, frames, tracks), dtype=np.float32)
        target_visibility = np.ones((count, frames, tracks), dtype=np.float32)
        for index, (kind, family) in enumerate(specifications):
            iid = f"case-{index:03d}"
            prompt = (
                "Make the subject wave."
                if family == "wave"
                else "Make the dog lie down."
            )
            positive = kind == "positive"
            label = {
                "class": "positive" if positive else "negative",
                "negative_type": None if positive else kind,
                "primary_family": family,
                "provenance_kind": "synthetic-test",
                "human_label": False,
            }
            row = {
                "iid": iid,
                "input_digest": f"{index + 1:064x}",
                "prompt": prompt,
                "label": label,
                "assignment": {
                    "fresh": True,
                    "split": "train",
                    "component_id": f"component-{index:03d}",
                },
                "source_bindings": {
                    "schema_version": "fixture",
                    "media": {
                        "data_root": DATA_ROOT,
                        "src_video": {
                            "relative_path": f"{iid}/source.mp4",
                            "sha256": "b" * 64,
                        },
                        "tgt_video": {
                            "relative_path": f"{iid}/target.mp4",
                            "sha256": "c" * 64,
                        },
                    },
                },
            }
            rows.append(row)
            track_rows.append(
                {
                    "iid": iid,
                    "input_index": index,
                    "final_array_index": index,
                    "input_digest": row["input_digest"],
                    "input_row_sha256": object_digest(row),
                    "paired_track_valid": True,
                    "paired_camera_valid": True,
                    "source": {"video_sha256": "b" * 64},
                    "target": {"video_sha256": "c" * 64},
                }
            )
            if kind in {"positive", "instruction_mismatch"}:
                target_stabilized[index, :, 0, 1] += (
                    np.arange(frames) * 0.006
                )
                target_normalized[index] = target_stabilized[index]
            elif kind == "camera_motion":
                target_normalized[index, :, :, 0] += (
                    np.arange(frames)[:, None] * 0.004
                )
            elif kind == "artifact":
                alternating = np.asarray(
                    [0.0, 0.006, 0.0, 0.006, 0.0, 0.006],
                    dtype=np.float32,
                )
                target_stabilized[index, :, 0, 1] += alternating
                target_normalized[index] = target_stabilized[index]
                target_visibility[index] = 0.72
        _write_jsonl(self.candidate_path, rows)
        _write_jsonl(self.track_manifest_path, track_rows)
        np.savez_compressed(
            self.track_cache_path,
            input_indices=np.arange(count, dtype=np.int64),
            source_normalized_tracks=source_normalized,
            target_normalized_tracks=target_normalized,
            source_stabilized_tracks=source_stabilized,
            target_stabilized_tracks=target_stabilized,
            source_visibility=source_visibility,
            target_visibility=target_visibility,
            source_track_valid=np.ones(count, dtype=np.bool_),
            target_track_valid=np.ones(count, dtype=np.bool_),
            source_camera_valid=np.ones(count, dtype=np.bool_),
            target_camera_valid=np.ones(count, dtype=np.bool_),
            source_camera_crossfit_valid=np.ones(count, dtype=np.bool_),
            target_camera_crossfit_valid=np.ones(count, dtype=np.bool_),
            source_camera_crossfit_residual_median=np.full(
                count, 0.0005, dtype=np.float32
            ),
            target_camera_crossfit_residual_median=np.full(
                count, 0.0005, dtype=np.float32
            ),
            target_camera_crossfit_residual_reduction=np.full(
                count, 0.8, dtype=np.float32
            ),
        )

    def queue(self, root: Path) -> tuple[Path, list[dict]]:
        payload = derive_qwen_audit_queue(
            candidate_manifest=self.candidate_path,
            track_manifest=self.track_manifest_path,
            track_cache=self.track_cache_path,
            qwen_model_id=MODEL_ID,
            qwen_prompt_sha256=PROMPT_SHA,
            audit_oversample=1,
        )
        queue_dir = root / "queue"
        write_qwen_audit_queue(payload, queue_dir)
        return queue_dir, list(payload["rows"])

    def set_target_camera_residuals(
        self,
        residuals: dict[int, float],
    ) -> None:
        with np.load(self.track_cache_path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
        target = arrays["target_camera_crossfit_residual_median"]
        for index, value in residuals.items():
            target[index] = value
        np.savez_compressed(self.track_cache_path, **arrays)

    def expanded_queue_payload(self) -> dict:
        return derive_qwen_audit_queue(
            candidate_manifest=self.candidate_path,
            track_manifest=self.track_manifest_path,
            track_cache=self.track_cache_path,
            qwen_model_id=MODEL_ID,
            qwen_prompt_sha256=PROMPT_SHA,
            audit_oversample=1,
            candidate_expansion_tier=BOUNDED_ACTION_NEAR_MISS_TIER,
        )


def _audit(queue_row: dict, morphology: str = "adult_human") -> dict:
    role = queue_row["screen_role_hint"]
    family = queue_row["intended_family"]
    values = {
        "schema_version": AUDIT_ROW_SCHEMA,
        "iid": queue_row["iid"],
        "queue_row_sha256": object_digest(queue_row),
        "qwen_model_id": MODEL_ID,
        "qwen_prompt_sha256": PROMPT_SHA,
        "intended_atomic": family,
        "observed_atomic_or_none": family,
        "source_state": "source state is visible",
        "target_state": "target state is visible",
        "subject_morphology": morphology,
        "onset": "clear",
        "periodicity": "repeated" if family == "wave" else "single",
        "direction": "toward_viewer" if family == "wave" else "other",
        "success": "yes",
        "actor_motion": "clear",
        "camera_motion": "none",
        "identity_appearance_change": "none",
        "nonphysical_effect": "none",
        "deformation": "none",
        "flicker": "none",
        "confidence": "high",
        "reflection_or_sunglasses_artifact": "none",
        "secondary_action": "none",
    }
    if role == "wrong":
        values.update(
            observed_atomic_or_none="other",
            success="no",
        )
    elif role == "static":
        values.update(
            observed_atomic_or_none="none",
            onset="none",
            periodicity="none",
            direction="none",
            success="no",
            actor_motion="none",
        )
    elif role == "camera":
        values.update(camera_motion="high", success="no")
    elif role == "effect":
        values.update(nonphysical_effect="high", success="no")
    return values


class R10BBerniniPilotManifestTests(unittest.TestCase):
    def _complete_audits(self, rows: list[dict]) -> list[dict]:
        wave_morphology = iter(
            [
                "adult_human",
                "child_human",
                "character_or_nonhuman",
                "adult_human",
            ]
        )
        lie_morphology = iter(
            ["dog", "bulldog", "cat", "other_quadruped"]
        )
        audits = []
        for row in rows:
            if row["screen_role_hint"] == "positive":
                morphology = (
                    next(wave_morphology)
                    if row["intended_family"] == "wave"
                    else next(lie_morphology)
                )
            else:
                morphology = (
                    "adult_human"
                    if row["intended_family"] == "wave"
                    else "dog"
                )
            audits.append(_audit(row, morphology))
        return audits

    def test_queue_and_complete_finalize_are_component_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            queue_dir, queue_rows = fixture.queue(root)
            self.assertEqual(len(queue_rows), MAX_FINAL_ROWS)
            self.assertEqual(
                len({row["component_id"] for row in queue_rows}),
                MAX_FINAL_ROWS,
            )
            static_rows = [
                row
                for row in queue_rows
                if row["screen_role_hint"] == "static"
            ]
            self.assertTrue(static_rows)
            self.assertTrue(
                all(not row["motion_gate_applicable"] for row in static_rows)
            )
            self.assertTrue(
                all(row["feature_gates"]["static"]["pass"] for row in static_rows)
            )
            audits = self._complete_audits(queue_rows)
            audit_path = root / "audits.jsonl"
            _write_jsonl(audit_path, audits)
            output = root / "final"
            summary = finalize_controlled_pilot(
                queue_dir=queue_dir,
                audit_records=audit_path,
                output_dir=output,
            )
            self.assertTrue(summary["balanced_pilot_ready"])
            self.assertEqual(summary["rows"], MAX_FINAL_ROWS)
            self.assertEqual(summary["quota_selected"], FINAL_QUOTAS)
            manifest = [
                json.loads(line)
                for line in (output / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                len({row["component_id"] for row in manifest}),
                MAX_FINAL_ROWS,
            )
            self.assertTrue(
                all(row["authorization"]["training_authorized"] is False for row in manifest)
            )
            self.assertTrue(
                all(row["data_root"] == DATA_ROOT for row in manifest)
            )
            self.assertTrue(
                all("original_prompt" in row and "canonical_prompt" in row for row in manifest)
            )

    def test_semantic_shortfall_is_recorded_without_fabrication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            queue_dir, queue_rows = fixture.queue(root)
            audits = self._complete_audits(queue_rows)
            static_control = next(
                record
                for record, row in zip(audits, queue_rows)
                if row["screen_cell"] == "static:global"
            )
            static_control["observed_atomic_or_none"] = "ambiguous"
            audit_path = root / "audits.jsonl"
            _write_jsonl(audit_path, audits)
            output = root / "final"
            summary = finalize_controlled_pilot(
                queue_dir=queue_dir,
                audit_records=audit_path,
                output_dir=output,
            )
            self.assertFalse(summary["balanced_pilot_ready"])
            self.assertEqual(
                summary["experiment_role"],
                "engineering_unbalanced_evidence_only",
            )
            self.assertEqual(summary["rows"], MAX_FINAL_ROWS - 1)
            self.assertEqual(
                summary["shortfalls"]["control:static:global"]["selected"],
                FINAL_QUOTAS["control:static:global"] - 1,
            )
            self.assertFalse(summary["controls_fabricated"])

    def test_qwen_binding_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            queue_dir, queue_rows = fixture.queue(root)
            audits = self._complete_audits(queue_rows)
            audits[0]["queue_row_sha256"] = "f" * 64
            audit_path = root / "audits.jsonl"
            _write_jsonl(audit_path, audits)
            with self.assertRaisesRegex(
                R10BBerniniPilotError, "audit binding differs"
            ):
                finalize_controlled_pilot(
                    queue_dir=queue_dir,
                    audit_records=audit_path,
                    output_dir=root / "final",
                )

    def test_bounded_near_miss_is_audit_only_and_r2_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            # case-003 fails only the strict camera-residual threshold and
            # passes bounded_action_near_miss_v1.  case-007 also exceeds the
            # frozen near-miss threshold and must remain excluded.
            fixture.set_target_camera_residuals({3: 0.0015, 7: 0.0035})
            payload = fixture.expanded_queue_payload()
            rows = list(payload["rows"])
            expansion_rows = [
                row
                for row in rows
                if row.get("candidate_expansion_tier")
                == BOUNDED_ACTION_NEAR_MISS_TIER
            ]
            self.assertEqual([row["iid"] for row in expansion_rows], ["case-003"])
            expansion = expansion_rows[0]
            self.assertFalse(expansion["feature_gates"]["action"]["pass"])
            self.assertFalse(expansion["motion_gate_pass"])
            self.assertTrue(
                expansion["candidate_expansion_check_evidence"]["pass"]
            )
            self.assertEqual(
                expansion["candidate_expansion_policy_sha256"],
                BOUNDED_ACTION_NEAR_MISS_POLICY_SHA256,
            )
            self.assertTrue(expansion["audit_only"])
            self.assertFalse(expansion["final_pilot_eligible"])
            self.assertEqual(rows[-1]["iid"], "case-003")
            self.assertNotIn("case-007", {row["iid"] for row in rows})
            self.assertTrue(
                all(
                    row["upstream_label"]["class"] == "positive"
                    for row in expansion_rows
                )
            )
            self.assertEqual(
                len({row["component_id"] for row in rows}),
                len(rows),
            )
            expansion_summary = payload["summary"]["candidate_expansion"]
            self.assertEqual(expansion_summary["selected_rows"], 1)
            self.assertEqual(
                expansion_summary["selected_cell_counts"],
                {"positive:wave": 1},
            )
            self.assertEqual(
                payload["summary"]["screen_cell_counts"]["positive:wave"],
                4,
            )
            self.assertEqual(
                payload["summary"]["screen_cell_counts"][
                    "positive:quadruped_lie_down"
                ],
                3,
            )

    def test_qwen_positive_near_miss_cannot_balance_or_open_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            fixture.set_target_camera_residuals({3: 0.0015, 7: 0.0035})
            payload = fixture.expanded_queue_payload()
            queue_dir = root / "expanded_queue"
            write_qwen_audit_queue(payload, queue_dir)
            rows = list(payload["rows"])
            audits = self._complete_audits(rows)
            audit_path = root / "expanded_audits.jsonl"
            _write_jsonl(audit_path, audits)
            summary = finalize_controlled_pilot(
                queue_dir=queue_dir,
                audit_records=audit_path,
                output_dir=root / "expanded_final",
            )
            self.assertFalse(summary["balanced_pilot_ready"])
            self.assertEqual(
                summary["candidate_expansion_audit"],
                {
                    "tier": BOUNDED_ACTION_NEAR_MISS_TIER,
                    "rows": 1,
                    "classification_counts": {"positive": 1},
                    "rejection_count": 1,
                    "admitted_to_final_quota": 0,
                    "audit_only": True,
                    "final_pilot_eligible": False,
                },
            )
            self.assertEqual(
                summary["rejection_counts"][
                    "candidate_expansion_audit_only"
                ],
                1,
            )
            for field in (
                "formal_evidence",
                "representation_promoted",
                "renderer_probe_authorized",
                "generation_authorized",
                "training_authorized",
            ):
                self.assertFalse(summary["authorization"][field])

    def test_near_miss_evidence_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _Fixture(root)
            fixture.set_target_camera_residuals({3: 0.0015})
            payload = fixture.expanded_queue_payload()
            queue_dir = root / "expanded_queue"
            write_qwen_audit_queue(payload, queue_dir)

            queue_path = queue_dir / "qwen_audit_queue.jsonl"
            rows = [
                json.loads(line)
                for line in queue_path.read_text(encoding="utf-8").splitlines()
            ]
            expansion = next(
                row
                for row in rows
                if row.get("candidate_expansion_tier")
                == BOUNDED_ACTION_NEAR_MISS_TIER
            )
            expansion["candidate_expansion_check_evidence"]["checks"][
                "camera_crossfit_residual_median"
            ] = False
            queue_bytes = "".join(
                canonical_json(row) + "\n" for row in rows
            ).encode("utf-8")
            queue_path.write_bytes(queue_bytes)

            summary_path = queue_dir / "summary.json"
            queue_summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            queue_summary["queue_sha256"] = hashlib.sha256(
                queue_bytes
            ).hexdigest()
            summary_bytes = (
                json.dumps(
                    queue_summary,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            summary_path.write_bytes(summary_bytes)

            done_path = queue_dir / "done.json"
            done = json.loads(done_path.read_text(encoding="utf-8"))
            done["files"]["qwen_audit_queue.jsonl"] = hashlib.sha256(
                queue_bytes
            ).hexdigest()
            done["files"]["summary.json"] = hashlib.sha256(
                summary_bytes
            ).hexdigest()
            done_path.write_text(
                json.dumps(
                    done,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                R10BBerniniPilotError,
                "candidate expansion evidence differs",
            ):
                finalize_controlled_pilot(
                    queue_dir=queue_dir,
                    audit_records=root / "unused.jsonl",
                    output_dir=root / "unused_final",
                )


if __name__ == "__main__":
    unittest.main()
