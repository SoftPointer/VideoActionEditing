from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import goku_paired_stage0_admission_0817_v1 as admission  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class Stage0Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.media = root / "media"
        self.stage = root / "stage0"
        self.media.mkdir()
        self.stage.mkdir()
        self.original = root / "original_selected.jsonl"
        self.rows: list[dict[str, object]] = []
        self.audits: list[dict[str, object]] = []
        self.output_index = 0
        self.add_dynamic(
            iid="dynamic-a",
            media_key="a",
            instruction="Make the person walk across the room.",
            content_group="actor-scene-a",
            seed=7,
        )

    def _media_paths(self, key: str) -> tuple[Path, Path]:
        source = self.media / f"{key}-source.mp4"
        target = self.media / f"{key}-target.mp4"
        if not source.exists():
            source.write_bytes((f"physical-source-{key}-" * 4).encode("utf-8"))
        if not target.exists():
            target.write_bytes((f"physical-target-{key}-" * 4).encode("utf-8"))
        return source, target

    def add_dynamic(
        self,
        *,
        iid: str,
        media_key: str,
        instruction: str,
        content_group: str,
        seed: int,
        action_family: str = "walk",
        selected: bool = True,
    ) -> None:
        source, target = self._media_paths(media_key)
        row: dict[str, object] = {
            "iid": iid,
            "src_video": str(source.relative_to(self.root)),
            "tgt_video": str(target.relative_to(self.root)),
            "prompt": instruction,
            "source_video_sha256": _sha(source),
            "content_group_id": content_group,
            "action_family": action_family,
            "seed": seed,
        }
        delta = 0.50 if selected else 0.10
        motive: dict[str, object] = {
            "status": "ok",
            "id": iid,
            "target_path": str(target),
            "target_label": "dynamic_object",
            "target_metrics": {"residual_speed_p90": 0.02},
            "paired": True,
            "selected": selected,
            "instruction": instruction,
            "instruction_semantics": {
                "label": "continuous_action",
                "matched_motion_terms": ["walk"],
                "matched_static_cues": [],
                "matched_endpoint_terms": [],
                "matched_appearance_terms": [],
            },
            "source_path": str(source),
            "source_label": "static",
            "source_metrics": {"residual_speed_p90": 0.001},
            "descriptor_delta_norm": delta,
        }
        self.rows.append(row)
        self.audits.append(motive)

    def add_suppression(
        self,
        *,
        iid: str = "suppression-b",
        media_key: str = "b",
        content_group: str = "actor-scene-b",
    ) -> None:
        source, target = self._media_paths(media_key)
        instruction = "Make the dog stop and remain completely still."
        row: dict[str, object] = {
            "iid": iid,
            "src_video": str(source.relative_to(self.root)),
            "tgt_video": str(target.relative_to(self.root)),
            "prompt": instruction,
            "source_video_sha256": _sha(source),
            "content_group_id": content_group,
            "action_family": "stop",
        }
        motive: dict[str, object] = {
            "status": "ok",
            "id": iid,
            "target_path": str(target),
            "target_label": "static",
            "target_metrics": {"residual_speed_p90": 0.004},
            "paired": True,
            "selected": True,
            "instruction": instruction,
            "instruction_semantics": {
                "label": "motion_suppression",
                "matched_motion_terms": ["stop"],
                "matched_static_cues": [],
                "matched_endpoint_terms": [],
                "matched_appearance_terms": [],
            },
            "source_path": str(source),
            "source_label": "dynamic_object",
            "source_metrics": {"residual_speed_p90": 0.02},
            "descriptor_delta_norm": 0.50,
        }
        self.rows.append(row)
        self.audits.append(motive)

    def materialize(self) -> None:
        _write_jsonl(self.original, self.rows)
        audited_rows: list[dict[str, object]] = []
        selected_rows: list[dict[str, object]] = []
        labels: dict[str, int] = {}
        semantics: dict[str, int] = {}
        ids: list[str] = []
        for index, (row, motive_template) in enumerate(zip(self.rows, self.audits)):
            motive = deepcopy(motive_template)
            motive["feature_index"] = index
            audited = deepcopy(row)
            audited["motive_audit"] = motive
            audited_rows.append(audited)
            if motive["selected"]:
                selected_rows.append(deepcopy(audited))
            label = f"{motive['source_label']}->{motive['target_label']}"
            labels[label] = labels.get(label, 0) + 1
            semantic = str(motive["instruction_semantics"]["label"])
            semantics[semantic] = semantics.get(semantic, 0) + 1
            ids.append(str(row["iid"]))
        _write_jsonl(self.stage / "audit.jsonl", audited_rows)
        _write_jsonl(self.stage / "selected.jsonl", selected_rows)
        metadata = {
            "schema_version": admission.FEATURE_SCHEMA_VERSION,
            "feature_kind": "geometry_action_delta",
            "dimension": 4,
            "provenance": {
                "descriptor_version": "camera_compensated_hoof_v2",
                "descriptor_config": admission.EXPECTED_DESCRIPTOR_CONFIG,
                "motion_backend": "opencv_farneback_partial_affine_v1",
                "motion_config": admission.EXPECTED_MOTION_CONFIG,
                "speed_units": "frame_width_per_second",
            },
        }
        metadata["compatibility_digest"] = _object_sha(metadata)
        with (self.stage / "descriptors.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.arange(len(ids) * 4, dtype=np.float32).reshape(len(ids), 4),
                ids=np.asarray(ids),
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
        summary = {
            "input": str(self.original),
            "root": str(self.root),
            "total": len(self.rows),
            "successful": len(self.rows),
            "selected": len(selected_rows),
            "errors": 0,
            "labels": dict(sorted(labels.items())),
            "instruction_semantics": dict(sorted(semantics.items())),
            "selection_semantic_classes": None,
            "config": admission.EXPECTED_MOTION_CONFIG,
            "descriptor_semantics": (
                "paired target-minus-source geometry action delta"
            ),
            "archive_compatibility_digest": metadata["compatibility_digest"],
        }
        (self.stage / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def expected(self) -> dict[str, str]:
        return {
            "expected_original_sha256": _sha(self.original),
            "expected_summary_sha256": _sha(self.stage / "summary.json"),
            "expected_audit_sha256": _sha(self.stage / "audit.jsonl"),
            "expected_selected_sha256": _sha(self.stage / "selected.jsonl"),
            "expected_descriptors_sha256": _sha(self.stage / "descriptors.npz"),
        }

    def run(
        self,
        *,
        equivalence_authority: Path | None = None,
    ) -> tuple[dict[str, object], Path]:
        output = self.root / f"admitted-{self.output_index}"
        self.output_index += 1
        kwargs: dict[str, object] = {}
        if equivalence_authority is not None:
            kwargs = {
                "equivalence_authority": equivalence_authority,
                "expected_equivalence_authority_sha256": _sha(
                    equivalence_authority
                ),
            }
        receipt = admission.admit_stage0(
            stage0_dir=self.stage,
            original_selected=self.original,
            output_dir=output,
            **self.expected(),
            **kwargs,
        )
        return receipt, output

    def authority(self, overrides: dict[str, dict[str, str]] | None = None) -> Path:
        rows: list[dict[str, object]] = []
        overrides = overrides or {}
        for row in self.rows:
            iid = str(row["iid"])
            source = self.root / str(row["src_video"])
            target = self.root / str(row["tgt_video"])
            values: dict[str, object] = {
                "schema_version": admission.EQUIVALENCE_SCHEMA_VERSION,
                "iid": iid,
                "source_sha256": _sha(source),
                "target_sha256": _sha(target),
                "canonical_source_id": f"canonical-source-{iid}",
                "canonical_target_id": f"canonical-target-{iid}",
                "instruction_equivalence_id": f"instruction-{iid}",
                "upstream_group_id": f"upstream-{iid}",
                "actor_scene_group_id": f"actor-scene-{iid}",
            }
            values.update(overrides.get(iid, {}))
            rows.append(values)
        path = self.root / "equivalence_authority.jsonl"
        _write_jsonl(path, rows)
        return path


class GokuPairedStage0AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = Stage0Fixture(self.root)
        self.fixture.materialize()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_rejected(self, pattern: str) -> None:
        with self.assertRaisesRegex(admission.GokuStage0AdmissionError, pattern):
            self.fixture.run()

    def test_success_is_diagnostic_only_and_media_are_rehashed(self) -> None:
        receipt, output = self.fixture.run()
        self.assertEqual(receipt["closure_status"], "PASS")
        self.assertEqual(receipt["role"], admission.ROLE)
        self.assertFalse(receipt["training_authorized"])
        self.assertEqual(receipt["qualification_status"], "unqualified")
        self.assertEqual(receipt["formal_d0_count_contribution"], 0)
        closure = receipt["physical_media_closure"]
        self.assertTrue(closure["source_and_target_bytes_rehashed"])
        self.assertFalse(closure["trusted_declared_hashes"])
        candidate = json.loads(
            (output / "candidate_manifest.jsonl").read_text(encoding="utf-8")
        )
        self.assertEqual(candidate["source"]["sha256"], _sha(self.root / "media/a-source.mp4"))
        self.assertEqual(candidate["target"]["sha256"], _sha(self.root / "media/a-target.mp4"))
        self.assertFalse(candidate["training_authorized"])
        self.assertEqual(candidate["human_review"], "pending")
        review = json.loads((output / "review_queue.jsonl").read_text())
        self.assertEqual(review["review_status"], "pending")
        self.assertEqual(review["reviewer_receipts"], [])
        self.assertFalse(review["training_authorized"])

    def test_all_output_flags_remain_false_even_if_input_claims_human_acceptance(self) -> None:
        self.fixture.rows[0]["human_review"] = "accepted"
        self.fixture.rows[0]["training_authorized"] = True
        self.fixture.materialize()
        receipt, output = self.fixture.run()
        candidate = json.loads((output / "candidate_manifest.jsonl").read_text())
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(candidate["training_authorized"])
        self.assertEqual(candidate["qualification_status"], "unqualified")

    def test_declared_source_sha_is_not_trusted(self) -> None:
        self.fixture.rows[0]["source_video_sha256"] = "0" * 64
        self.fixture.materialize()
        self.assert_rejected("declared source SHA disagrees")

    def test_declared_target_sha_is_checked_when_present(self) -> None:
        self.fixture.rows[0]["target_video_sha256"] = "1" * 64
        self.fixture.materialize()
        self.assert_rejected("declared target SHA disagrees")

    def test_symlink_media_is_rejected(self) -> None:
        real = self.root / "media/real-source.mp4"
        real.write_bytes(b"real-source-physical-bytes")
        source = self.root / "media/a-source.mp4"
        source.unlink()
        source.symlink_to(real)
        self.fixture.rows[0]["source_video_sha256"] = _sha(real)
        self.fixture.materialize()
        self.assert_rejected("may not be a symlink")

    def test_pinned_artifact_sha_rejects_mutation(self) -> None:
        expected = self.fixture.expected()
        with (self.fixture.stage / "audit.jsonl").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(
            admission.GokuStage0AdmissionError, "audit SHA-256 mismatch"
        ):
            admission.admit_stage0(
                stage0_dir=self.fixture.stage,
                original_selected=self.fixture.original,
                output_dir=self.root / "never-created",
                **expected,
            )
        self.assertFalse((self.root / "never-created").exists())

    def test_selected_flag_is_recomputed_not_trusted(self) -> None:
        self.fixture.audits[0]["descriptor_delta_norm"] = 0.10
        self.fixture.audits[0]["selected"] = True
        self.fixture.materialize()
        self.assert_rejected("selected flag disagrees")

    def test_selected_jsonl_must_be_exact_ordered_subset(self) -> None:
        self.fixture.add_suppression()
        self.fixture.materialize()
        rows = [json.loads(line) for line in (self.fixture.stage / "selected.jsonl").read_text().splitlines()]
        _write_jsonl(self.fixture.stage / "selected.jsonl", list(reversed(rows)))
        self.assert_rejected("exact ordered selected=True subset")

    def test_stage0_semantic_allowlist_drift_is_rejected(self) -> None:
        summary_path = self.fixture.stage / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["selection_semantic_classes"] = ["continuous_action"]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self.assert_rejected("summary semantic selection classes")

    def test_null_allowlist_replays_noncontinuous_semantic_selection(self) -> None:
        self.fixture.audits[0]["instruction_semantics"]["label"] = "endpoint_pose"
        self.fixture.materialize()
        receipt, _ = self.fixture.run()
        self.assertEqual(
            receipt["candidate_semantic_counts"], {"endpoint_pose": 1}
        )

    def test_upstream_family_field_is_the_action_family_authority(self) -> None:
        self.fixture.rows[0].pop("action_family")
        self.fixture.rows[0]["family"] = "sit_down"
        self.fixture.materialize()
        receipt, _ = self.fixture.run()
        self.assertEqual(
            receipt["candidate_action_family_counts"], {"sit_down": 1}
        )

    def test_original_to_audit_row_copy_closure(self) -> None:
        audit_rows = [json.loads(line) for line in (self.fixture.stage / "audit.jsonl").read_text().splitlines()]
        audit_rows[0]["seed"] = 999
        _write_jsonl(self.fixture.stage / "audit.jsonl", audit_rows)
        self.assert_rejected("not an exact copy")

    def test_feature_id_and_index_closure(self) -> None:
        audit_rows = [json.loads(line) for line in (self.fixture.stage / "audit.jsonl").read_text().splitlines()]
        audit_rows[0]["motive_audit"]["feature_index"] = 4
        _write_jsonl(self.fixture.stage / "audit.jsonl", audit_rows)
        selected = deepcopy(audit_rows)
        _write_jsonl(self.fixture.stage / "selected.jsonl", selected)
        self.assert_rejected("feature_index")

    def test_nan_descriptor_is_rejected(self) -> None:
        with np.load(self.fixture.stage / "descriptors.npz", allow_pickle=False) as archive:
            ids = archive["ids"]
            metadata = archive["metadata_json"]
        with (self.fixture.stage / "descriptors.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.asarray([[np.nan, 0, 0, 0]], dtype=np.float32),
                ids=ids,
                metadata_json=metadata,
            )
        self.assert_rejected("NaN/Inf")

    def test_extra_npz_member_is_rejected(self) -> None:
        with np.load(self.fixture.stage / "descriptors.npz", allow_pickle=False) as archive:
            features = archive["features"]
            ids = archive["ids"]
            metadata = archive["metadata_json"]
        with (self.fixture.stage / "descriptors.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                features=features,
                ids=ids,
                metadata_json=metadata,
                surprise=np.asarray([1]),
            )
        self.assert_rejected("keys must be exactly")

    def test_partial_stage0_errors_are_rejected(self) -> None:
        summary = json.loads((self.fixture.stage / "summary.json").read_text())
        summary["errors"] = 1
        summary["successful"] = 0
        (self.fixture.stage / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n"
        )
        self.assert_rejected("partial stage0 output")

    def test_motion_config_is_frozen(self) -> None:
        summary = json.loads((self.fixture.stage / "summary.json").read_text())
        summary["config"]["analysis_frames"] = 31
        (self.fixture.stage / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n"
        )
        self.assert_rejected("motion config does not match")

    def test_ambiguous_path_alias_declarations_are_rejected(self) -> None:
        self.fixture.rows[0]["source_video"] = "media/different.mp4"
        self.fixture.materialize()
        self.assert_rejected("ambiguous source video path")

    def test_seed_and_path_variants_collapse_to_one_effective_n(self) -> None:
        self.fixture.add_dynamic(
            iid="dynamic-a-seed-99",
            media_key="a",
            instruction="Make the person walk across the room.",
            content_group="actor-scene-a",
            seed=99,
        )
        self.fixture.materialize()
        receipt, output = self.fixture.run()
        self.assertEqual(receipt["counts"]["selected_physical_rows"], 2)
        self.assertEqual(receipt["counts"]["effective_diagnostic_candidates"], 1)
        self.assertEqual(receipt["counts"]["seed_path_endpoint_aliases_collapsed"], 1)
        candidate = json.loads((output / "candidate_manifest.jsonl").read_text())
        self.assertEqual(candidate["physical_variant_count"], 2)
        self.assertEqual(
            candidate["alias_iids"], ["dynamic-a", "dynamic-a-seed-99"]
        )

    def test_one_physical_source_cannot_claim_two_canonical_groups(self) -> None:
        self.fixture.add_dynamic(
            iid="same-bytes-conflicting-group",
            media_key="a",
            instruction="Make the person run across the room.",
            content_group="conflicting-canonical-group",
            seed=10,
            action_family="run",
        )
        self.fixture.materialize()
        self.assert_rejected("physical source SHA maps to multiple canonical source IDs")

    def test_external_authority_collapses_transcoded_endpoint_variants(self) -> None:
        self.fixture.add_dynamic(
            iid="dynamic-transcode",
            media_key="transcoded-a",
            instruction="Make the person walk across the room.",
            content_group="different-embedded-group",
            seed=88,
        )
        self.fixture.materialize()
        common = {
            "canonical_source_id": "canonical-source-same",
            "instruction_equivalence_id": "instruction-same",
            "upstream_group_id": "upstream-same",
            "actor_scene_group_id": "actor-scene-same",
        }
        authority = self.fixture.authority(
            {
                "dynamic-a": {**common, "canonical_target_id": "target-a"},
                "dynamic-transcode": {
                    **common,
                    "canonical_target_id": "target-transcode",
                },
            }
        )
        receipt, output = self.fixture.run(equivalence_authority=authority)
        self.assertEqual(receipt["counts"]["effective_diagnostic_candidates"], 1)
        candidate = json.loads((output / "candidate_manifest.jsonl").read_text())
        self.assertTrue(candidate["group_closure"]["external_equivalence_authority"])

    def test_authority_content_hash_is_bound_to_physical_bytes(self) -> None:
        authority = self.fixture.authority(
            {"dynamic-a": {"target_sha256": "f" * 64}}
        )
        with self.assertRaisesRegex(
            admission.GokuStage0AdmissionError,
            "authority target SHA disagrees",
        ):
            self.fixture.run(equivalence_authority=authority)

    def test_group_connected_component_is_partition_disjoint(self) -> None:
        self.fixture.add_dynamic(
            iid="dynamic-a-run",
            media_key="a-run",
            instruction="Make the person run across the room.",
            content_group="actor-scene-a",
            seed=8,
            action_family="run",
        )
        self.fixture.materialize()
        _, output = self.fixture.run()
        candidates = [
            json.loads(line)
            for line in (output / "candidate_manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {row["candidate_partition"] for row in candidates}
            .__len__(),
            1,
        )
        self.assertEqual(
            {row["group_closure"]["connected_component_id"] for row in candidates}
            .__len__(),
            1,
        )

    def test_actor_scene_partition_is_stable_when_new_instruction_is_added(self) -> None:
        _, output_before = self.fixture.run()
        before = json.loads(
            (output_before / "candidate_manifest.jsonl").read_text()
        )
        self.fixture.add_dynamic(
            iid="dynamic-a-run-later",
            media_key="a-run-later",
            instruction="Make the person run across the room.",
            content_group="actor-scene-a",
            seed=11,
            action_family="run",
        )
        self.fixture.materialize()
        _, output_after = self.fixture.run()
        after_rows = [
            json.loads(line)
            for line in (output_after / "candidate_manifest.jsonl").read_text().splitlines()
        ]
        after = next(
            row
            for row in after_rows
            if row["effective_row_id"] == before["effective_row_id"]
        )
        self.assertEqual(after["candidate_partition"], before["candidate_partition"])
        self.assertEqual(
            after["group_closure"]["connected_component_id"],
            before["group_closure"]["connected_component_id"],
        )

    def test_semantic_action_family_and_motion_strata_are_reported(self) -> None:
        self.fixture.add_suppression()
        self.fixture.materialize()
        receipt, _ = self.fixture.run()
        self.assertEqual(
            receipt["candidate_semantic_counts"],
            {"continuous_action": 1, "motion_suppression": 1},
        )
        self.assertEqual(
            receipt["candidate_motion_stratum_counts"],
            {"dynamic": 1, "suppression": 1},
        )
        self.assertEqual(receipt["candidate_action_family_counts"], {"stop": 1, "walk": 1})

    def test_future_licensed500_quota_is_never_claimed_satisfied(self) -> None:
        receipt, _ = self.fixture.run()
        plan = receipt["future_d0_licensed500_review_plan"]
        self.assertEqual(plan["desired_quotas"], {"general": 300, "strict_action": 100, "noop": 100})
        self.assertEqual(plan["human_qualified"], {"general": 0, "strict_action": 0, "noop": 0})
        self.assertEqual(plan["qualification_shortfall"], plan["desired_quotas"])
        self.assertFalse(plan["quota_gate_passed"])
        self.assertEqual(receipt["counts"]["formal_d0_qualified_rows"], 0)

    def test_without_external_equivalence_authority_receipt_blocks_transcodes(self) -> None:
        receipt, _ = self.fixture.run()
        self.assertIn(
            "external_transcode_actor_scene_equivalence_authority_absent",
            receipt["blocking_reasons"],
        )

    def test_outputs_are_deterministic_across_output_directories(self) -> None:
        receipt_a, output_a = self.fixture.run()
        receipt_b, output_b = self.fixture.run()
        self.assertEqual(receipt_a, receipt_b)
        for name in (
            "candidate_manifest.jsonl",
            "review_queue.jsonl",
            "admission_receipt.json",
            "DONE.json",
        ):
            self.assertEqual((output_a / name).read_bytes(), (output_b / name).read_bytes())

    def test_existing_output_directory_is_never_overwritten(self) -> None:
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "user-data.txt"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(
            admission.GokuStage0AdmissionError, "already exists"
        ):
            admission.admit_stage0(
                stage0_dir=self.fixture.stage,
                original_selected=self.fixture.original,
                output_dir=existing,
                **self.fixture.expected(),
            )
        self.assertEqual(marker.read_text(), "preserve me")

    def test_duplicate_json_keys_are_rejected(self) -> None:
        self.fixture.original.write_text(
            '{"iid":"dynamic-a","iid":"shadow"}\n', encoding="utf-8"
        )
        self.assert_rejected("duplicate JSON key")


if __name__ == "__main__":
    unittest.main()
