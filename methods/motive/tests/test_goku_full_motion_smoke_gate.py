from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from motive.goku_full_motion_qwen import (
    _canonical_jsonl_bytes,
    _receipt_digest,
    qwen_provenance_digest,
    run_audit,
)
from motive.goku_full_motion_smoke_gate import (
    CANARY_ORACLE_SCHEMA,
    FAILURE_SCHEMA_VERSION,
    FullMotionSmokeGateError,
    _match_canary_actors,
    _validate_runtime_closure,
    gate_smoke,
    main,
)
from motive.goku_full_motion_contract import object_sha256
from motive.qwen_filter import _file_digest
from methods.motive.tests.test_goku_full_motion_qwen import (
    _A0aSelfNegatedHeadLabelBackend,
    _FakeBackend,
    _args,
    _input_row,
    _source_census,
    _write_jsonl,
)


def _oracle(row: dict) -> dict:
    return {
        "schema_version": CANARY_ORACLE_SCHEMA,
        "iid": row["iid"],
        "source_video_sha256": row["source_video_sha256"],
        "anchor_sha256": row["anchor_sha256"],
        "expected_dynamic_entities": [
            {
                "oracle_id": "left_person",
                "entity_type": "person",
                "viewer_region": "center_left",
                "i0_bbox_xyxy_1000": [50, 350, 300, 900],
                "required_motion_component_types": ["gesture"],
            },
            {
                "oracle_id": "right_person",
                "entity_type": "person",
                "viewer_region": "center_right",
                "i0_bbox_xyxy_1000": [700, 350, 950, 900],
                "required_motion_component_types": ["gesture"],
            },
        ],
        "expected_camera": {"dynamic": False, "motion_class": "locked_off"},
    }


class FullMotionSmokeGateTests(unittest.TestCase):
    def _finished_smoke(
        self, root: Path, *, backend_factory=_FakeBackend
    ) -> tuple[Path, Path, str]:
        canary = "two-people-wave-001"
        rows = [_input_row(root, canary)]
        rows.extend(_input_row(root, f"candidate-{index:02d}") for index in range(1, 8))
        selected = root / "selected.jsonl"
        _write_jsonl(selected, rows)
        qwen_root = root / "qwen"
        qwen_root.mkdir()
        for shard_index in range(8):
            output = qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
            args = _args(selected, output, rows[0])
            args.shard_index = shard_index
            self.assertEqual(
                run_audit(args, backend_factory=backend_factory), 0
            )
        return selected, qwen_root, canary

    @staticmethod
    def _rewrite_first_record(qwen_root: Path, mutate) -> None:
        shard = next(
            path
            for path in sorted(qwen_root.glob("qwen_shard_*.jsonl"))
            if path.stat().st_size
        )
        rows = [json.loads(line) for line in shard.read_text().splitlines()]
        mutate(rows[0])
        shard.write_bytes(_canonical_jsonl_bytes(rows))
        receipt_path = shard.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text())
        receipt["output"]["sha256"] = _file_digest(shard)
        receipt["output"]["bytes"] = shard.stat().st_size
        receipt["receipt_digest"] = _receipt_digest(receipt)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )

    def test_accepts_closed_two_person_canary(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            output = root / "gate.json"
            result = gate_smoke(
                input_path=selected,
                qwen_root=qwen_root,
                output=output,
                canary_iid=canary,
                canary_oracle=_oracle(json.loads(selected.read_text().splitlines()[0])),
                minimum_hard_passes=8,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["schema_version"],
                "motive-goku-full-motion-qwen-smoke-gate-v6",
            )
            self.assertEqual(result["hard_passes"], 8)
            runtime = result["qwen_runtime"]["run_config"]
            self.assertRegex(
                runtime["prompt_template_digests"]["pass_a0a"],
                r"^[0-9a-f]{64}$",
            )
            self.assertRegex(
                runtime["prompt_schema_digests"][
                    "coverage_authority_inventory"
                ],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                runtime["generation"]["visual_input"],
                "blind_two_stage_coverage_authority_plus_i0_only_grounding_plus_"
                "dense_source_mosaic_plus_temporal_triptych_lr_zoom_motion_"
                "attention_authority_grid_and_grounded_actor_zoom",
            )
            self.assertEqual(
                runtime["generation"]["high_resolution_checkpoints"],
                [
                    "exact_i0_only_grounding",
                    "full_frame_temporal_triptych",
                    "overlapping_left_right_temporal_zoom",
                    "fixed_full_frame_4x4_f0_f20_f40_f60_f80_grid",
                    "fixed_bbox_subject_f0_f20_f40_f60_f80_zoom",
                ],
            )
            self.assertEqual(
                result["qwen_lineage"],
                {
                    "record": "goku-full-motion-qwen-record-v6",
                    "hard_gate": "goku-full-motion-hard-gate-v6",
                    "provenance": "goku-full-motion-qwen-provenance-v6",
                    "source_inventory_alignment": (
                        "motive-goku-full-motion-source-inventory-alignment-v4"
                    ),
                    "change_region_proposals": (
                        "motive-goku-full-motion-change-region-proposals-v1"
                    ),
                    "coverage_authority": (
                        "motive-goku-full-motion-coverage-authority-v2"
                    ),
                    "coverage_authority_inventory": (
                        "motive-goku-full-motion-coverage-authority-inventory-v1"
                    ),
                    "coverage_authority_assignments": (
                        "motive-goku-full-motion-coverage-authority-assignments-v1"
                    ),
                    "coverage_authority_allowed_owner_map": (
                        "motive-goku-full-motion-coverage-authority-allowed-owner-map-v1"
                    ),
                    "coverage_authority_alignment": (
                        "motive-goku-full-motion-coverage-authority-alignment-v2"
                    ),
                },
            )
            self.assertEqual(len(result["hard_pass_bindings"]), 8)
            self.assertEqual(
                [binding["iid"] for binding in result["hard_pass_bindings"]],
                result["hard_pass_iids"],
            )
            for binding in result["hard_pass_bindings"]:
                self.assertEqual(
                    binding["source_inventory_alignment_schema_version"],
                    "motive-goku-full-motion-source-inventory-alignment-v4",
                )
                for field, value in binding.items():
                    if field.endswith("_sha256"):
                        self.assertRegex(value, r"^[0-9a-f]{64}$")
            self.assertEqual(
                result["canary"]["dynamic_unit_ids"], ["unit_01", "unit_02"]
            )
            self.assertTrue(
                all(
                    "gesture" in match["matched_motion_component_types"]
                    for label in (
                        "primary_actor_matches",
                        "secondary_actor_matches",
                    )
                    for match in result["canary"][label]
                )
            )
            self.assertIn("tattooed man", result["canary"]["edit_instruction"])
            self.assertIn("camera locked off", result["canary"]["camera_clause"])
            self.assertEqual(
                result["canary"]["qwen_record_schema_version"],
                "goku-full-motion-qwen-record-v6",
            )
            self.assertEqual(
                result["canary"]["qwen_hard_gate_schema_version"],
                "goku-full-motion-hard-gate-v6",
            )
            self.assertEqual(
                result["canary"]["qwen_i0_grounding_schema_version"],
                "motive-goku-full-motion-i0-grounding-v1",
            )
            for field in (
                "qwen_i0_grounding_sha256",
                "qwen_i0_grounding_prompt_sha256",
                "qwen_i0_grounding_visual_input_sha256",
                "source_census_canonicalization_sha256",
                "secondary_source_census_canonicalization_sha256",
                "target_plan_canonicalization_sha256",
                "change_region_proposals_sha256",
                "coverage_authority_inventory_prompt_sha256",
                "coverage_authority_inventory_visual_input_sha256",
                "coverage_authority_inventory_sha256",
                "coverage_authority_assignments_prompt_sha256",
                "coverage_authority_assignments_visual_input_sha256",
                "coverage_authority_assignments_sha256",
                "coverage_authority_sha256",
                "source_inventory_alignment_sha256",
                "coverage_authority_alignment_sha256",
                "qwen_hard_gate_sha256",
            ):
                self.assertRegex(result["canary"][field], r"^[0-9a-f]{64}$")
            self.assertEqual(json.loads(output.read_text())["gate_digest"], result["gate_digest"])

    def test_accepts_and_replays_canonicalized_original_a0a(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(
                root, backend_factory=_A0aSelfNegatedHeadLabelBackend
            )
            oracle = _oracle(json.loads(selected.read_text().splitlines()[0]))
            result = gate_smoke(
                input_path=selected,
                qwen_root=qwen_root,
                output=root / "gate.json",
                canary_iid=canary,
                canary_oracle=oracle,
                minimum_hard_passes=8,
            )
            self.assertEqual(result["status"], "pass")

            def forge_validated_from(record: dict) -> None:
                self.assertEqual(
                    record["coverage_authority_inventory_validated_from"],
                    "canonicalized_original",
                )
                record["coverage_authority_inventory_validated_from"] = (
                    "original"
                )
                record["provenance_digest"] = qwen_provenance_digest(record)

            self._rewrite_first_record(qwen_root, forge_validated_from)
            with self.assertRaisesRegex(
                FullMotionSmokeGateError, "canonicalization path differs"
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "forged_gate.json",
                    canary_iid=canary,
                    canary_oracle=oracle,
                    minimum_hard_passes=8,
                )

    def test_rejects_self_consistent_v6_runtime_contract_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, _ = self._finished_smoke(root)
            receipt_path = sorted(qwen_root.glob("*.receipt.json"))[0]
            receipt = json.loads(receipt_path.read_text())
            shard_index = receipt["shard_index"]
            input_sha256 = _file_digest(selected)
            self.assertEqual(
                _validate_runtime_closure(
                    receipt=receipt,
                    selected_path=selected.resolve(),
                    input_sha256=input_sha256,
                    shard_index=shard_index,
                ),
                receipt["run_config"],
            )
            for case in (
                "pass_a0a_digest",
                "pass_a0b_digest",
                "authority_schema_digest",
                "visual_input",
                "checkpoint_list",
            ):
                with self.subTest(case=case):
                    altered = copy.deepcopy(receipt)
                    config = altered["run_config"]
                    if case == "pass_a0a_digest":
                        config["prompt_template_digests"]["pass_a0a"] = "0" * 64
                    elif case == "pass_a0b_digest":
                        config["prompt_template_digests"]["pass_a0b"] = "0" * 64
                    elif case == "authority_schema_digest":
                        config["prompt_schema_digests"][
                            "coverage_authority_assignments"
                        ] = "0" * 64
                    elif case == "visual_input":
                        config["generation"]["visual_input"] = "legacy_visuals"
                    else:
                        config["generation"]["high_resolution_checkpoints"] = config[
                            "generation"
                        ]["high_resolution_checkpoints"][:-1]
                    altered["run_config_digest"] = object_sha256(config)
                    altered["config_digest"] = object_sha256(
                        {
                            "run_config_digest": altered["run_config_digest"],
                            "execution_manifest": str(selected.resolve()),
                            "execution_manifest_sha256": input_sha256,
                            "root": str(selected.resolve().parent),
                            "shard_index": shard_index,
                            "num_shards": 8,
                        }
                    )
                    with self.assertRaisesRegex(
                        FullMotionSmokeGateError,
                        "fixed v6 runtime differs",
                    ):
                        _validate_runtime_closure(
                            receipt=altered,
                            selected_path=selected.resolve(),
                            input_sha256=input_sha256,
                            shard_index=shard_index,
                        )

    def test_rejects_self_consistent_false_i0_visual_digest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            shard = next(
                path
                for path in sorted(qwen_root.glob("qwen_shard_*.jsonl"))
                if path.stat().st_size
            )
            rows = [json.loads(line) for line in shard.read_text().splitlines()]
            rows[0]["i0_grounding_visual_input_digest"] = "0" * 64
            rows[0]["provenance_digest"] = qwen_provenance_digest(rows[0])
            shard.write_bytes(_canonical_jsonl_bytes(rows))
            receipt_path = shard.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            receipt["output"]["sha256"] = _file_digest(shard)
            receipt["output"]["bytes"] = shard.stat().st_size
            receipt["receipt_digest"] = _receipt_digest(receipt)
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                FullMotionSmokeGateError,
                "I0 grounding visual media replay differs",
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                    minimum_hard_passes=8,
                )

    def test_independently_replays_exact_media_if_record_validator_is_bypassed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            self._rewrite_first_record(
                qwen_root,
                lambda row: row["media_verification"].__setitem__(
                    "width", row["media_verification"]["width"] + 1
                ),
            )
            with patch(
                "motive.goku_full_motion_smoke_gate.validate_output_record",
                return_value={},
            ), self.assertRaisesRegex(
                FullMotionSmokeGateError,
                "exact media verification binding differs",
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                    minimum_hard_passes=8,
                )

    def test_independently_replays_authority_grid_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)

            def mutate(row: dict) -> None:
                proposals = row["change_region_proposals"]
                proposals["global_changed_fraction_ppm"] = (
                    proposals["global_changed_fraction_ppm"] + 1
                ) % 1_000_001
                row["change_region_proposals_digest"] = object_sha256(
                    proposals
                )

            self._rewrite_first_record(qwen_root, mutate)
            with patch(
                "motive.goku_full_motion_smoke_gate.validate_output_record",
                return_value={},
            ), self.assertRaisesRegex(
                FullMotionSmokeGateError,
                "change-region proposals differ from media replay",
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                    minimum_hard_passes=8,
                )

    def test_independently_rejects_broken_a0_g_a1_a2_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)

            def mutate(row: dict) -> None:
                alignment = row["coverage_authority_alignment"]
                alignment["camera_aligned"] = False
                row["coverage_authority_alignment_digest"] = object_sha256(
                    alignment
                )

            self._rewrite_first_record(qwen_root, mutate)
            with patch(
                "motive.goku_full_motion_smoke_gate.validate_output_record",
                return_value={},
            ), self.assertRaisesRegex(
                FullMotionSmokeGateError,
                "independent A0/G/A1/A2 replay",
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                    minimum_hard_passes=8,
                )

    def test_rejects_modified_shard_after_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            shard = next(path for path in sorted(qwen_root.glob("qwen_shard_*.jsonl")) if path.stat().st_size)
            shard.write_bytes(shard.read_bytes() + b"\n")
            with self.assertRaisesRegex(FullMotionSmokeGateError, "output bytes differ"):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                )

    def test_rejects_uniformly_tampered_visual_runtime_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            for receipt_path in sorted(qwen_root.glob("*.receipt.json")):
                receipt = json.loads(receipt_path.read_text())
                for field in (
                    "nframes",
                    "tile_width",
                    "mosaic_columns",
                    "max_pixels",
                    "max_new_tokens",
                ):
                    receipt["run_config"][field] = 1
                receipt["receipt_digest"] = _receipt_digest(receipt)
                receipt_path.write_text(
                    json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(
                FullMotionSmokeGateError, "run config digest differs"
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=_oracle(
                        json.loads(selected.read_text().splitlines()[0])
                    ),
                    minimum_hard_passes=8,
                )

    def test_cli_publishes_closed_non_authorizing_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            shard = next(
                path
                for path in sorted(qwen_root.glob("qwen_shard_*.jsonl"))
                if path.stat().st_size
            )
            shard.write_bytes(shard.read_bytes() + b"\n")
            output = root / "gate.json"
            oracle_path = root / "oracle.json"
            oracle_path.write_text(
                json.dumps(_oracle(json.loads(selected.read_text().splitlines()[0])))
            )
            self.assertEqual(
                main(
                    [
                        "--input",
                        str(selected),
                        "--qwen-root",
                        str(qwen_root),
                        "--output",
                        str(output),
                        "--canary-iid",
                        canary,
                        "--canary-oracle-json",
                        str(oracle_path),
                    ]
                ),
                2,
            )
            receipt = json.loads(output.read_text())
            self.assertEqual(
                set(receipt),
                {
                    "schema_version",
                    "status",
                    "authorizes_full_run",
                    "input_path",
                    "qwen_root",
                    "canary_iid",
                    "error_type",
                    "error",
                    "failure_digest",
                },
            )
            self.assertEqual(receipt["schema_version"], FAILURE_SCHEMA_VERSION)
            self.assertEqual(receipt["status"], "fail")
            self.assertFalse(receipt["authorizes_full_run"])
            digest_payload = dict(receipt)
            digest = digest_payload.pop("failure_digest")
            self.assertEqual(digest, object_sha256(digest_payload))

    def test_rejects_canary_oracle_media_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            selected, qwen_root, canary = self._finished_smoke(root)
            row = json.loads(selected.read_text().splitlines()[0])
            oracle = _oracle(row)
            oracle["source_video_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                FullMotionSmokeGateError, "oracle media binding differs"
            ):
                gate_smoke(
                    input_path=selected,
                    qwen_root=qwen_root,
                    output=root / "gate.json",
                    canary_iid=canary,
                    canary_oracle=oracle,
                )

    def test_rejects_left_person_plus_nonperson_decoy(self) -> None:
        census = _source_census("two-people-wave-001")
        census["i0_entity_registry"][1]["entity_type"] = "articulated_object"
        census["dynamic_units"][1]["entity_type"] = "articulated_object"
        row = {
            "iid": census["iid"],
            "source_video_sha256": "1" * 64,
            "anchor_sha256": "2" * 64,
        }
        with self.assertRaisesRegex(
            FullMotionSmokeGateError, "right_person"
        ):
            _match_canary_actors(
                census=census,
                oracle=_oracle(row),
                label="primary",
            )

    def test_rejects_right_person_without_gesture_component(self) -> None:
        census = _source_census("two-people-wave-001")
        for component in census["dynamic_units"][1][
            "source_motion_components"
        ]:
            component["component_type"] = "body_pose"
        row = {
            "iid": census["iid"],
            "source_video_sha256": "1" * 64,
            "anchor_sha256": "2" * 64,
        }
        with self.assertRaisesRegex(FullMotionSmokeGateError, "right_person"):
            _match_canary_actors(
                census=census,
                oracle=_oracle(row),
                label="secondary",
            )

    def test_rejects_right_person_with_semantically_static_gesture(self) -> None:
        census = _source_census("two-people-wave-001")
        unit = census["dynamic_units"][1]
        unit["source_action_signature"] = "remains_completely_still"
        unit["source_motion"] = "the right person has no visible motion"
        unit["motion_evidence"] = [
            {
                **unit["motion_evidence"][0],
                "description": "the right person remains completely still",
            }
        ]
        component = unit["source_motion_components"][0]
        component["motion_signature"] = "no_gesture_change"
        component["motion_description"] = "there is no visible gesture change"
        component["motion_evidence"] = [
            {
                **component["motion_evidence"][0],
                "description": "the hand shows no visible gesture change",
            }
        ]
        row = {
            "iid": census["iid"],
            "source_video_sha256": "1" * 64,
            "anchor_sha256": "2" * 64,
        }
        with self.assertRaisesRegex(
            FullMotionSmokeGateError, "source census is invalid"
        ):
            _match_canary_actors(
                census=census,
                oracle=_oracle(row),
                label="secondary",
            )


if __name__ == "__main__":
    unittest.main()
