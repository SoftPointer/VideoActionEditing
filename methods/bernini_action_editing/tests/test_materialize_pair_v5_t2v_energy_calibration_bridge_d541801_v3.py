#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
CPU_LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3_cpu.sbatch"
)
for search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import author_pair_v5_core4_event_labels_d541801_v3 as author  # noqa: E402
import materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3 as bridge  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def fixture_rows(root: Path) -> tuple[list[dict], dict]:
    cells = (
        ("sp4-a", "fit", "dog-sit", "dog-fit"),
        ("sp4-a", "confirmation", "human-stand", "human-confirm"),
        ("sp4-b", "fit", "human-stand", "human-fit"),
        ("sp4-b", "confirmation", "dog-sit", "dog-confirm"),
    )
    joined: list[dict] = []
    labels: list[dict] = []
    for cell_index, (group_id, split, family, stem) in enumerate(cells):
        for branch in calibration.BRANCH_ORDER:
            candidate_id = f"pair5-t2v-core4-v2-{stem}-{branch}"
            generation_digest = sha(f"generation-{candidate_id}")
            candidate = {
                "candidate_id": candidate_id,
                "analysis_split": split,
                "action_family_id": family,
                "calibration_group_id": f"cell-{stem}",
                "actor_group_id": f"actor-{stem}",
                "scene_group_id": f"scene-{stem}",
                "action_group_id": f"action-{stem}",
                "semantic_branch": branch,
                "geometry_source_video_sha256": sha(f"geometry-{stem}"),
                "full_t2v_caption": f"Complete caption {cell_index} {branch}.",
                "full_t2v_caption_utf8_sha256": sha(
                    f"caption-hash-fixture-{cell_index}-{branch}"
                ),
            }
            is_action = branch == calibration.ACTION_BRANCH
            score = {
                "schema_version": bridge.REQUIRED_SCORE_RECEIPT_SCHEMA,
                "receipt_digest": sha(f"score-receipt-{candidate_id}"),
                "frozen_scorer_receipt_digest": sha("one-frozen-scorer"),
                "raw_global_action_energy_score": 2.0 if is_action else 0.0,
            }
            score_path = root / f"{candidate_id}-score.json"
            joined.append(
                {
                    "group_id": group_id,
                    "candidate": candidate,
                    "candidate_envelope_sha256": sha(f"envelope-{candidate_id}"),
                    "generation_receipt_digest": generation_digest,
                    "generation_receipt_file_sha256": sha(
                        f"generation-file-{candidate_id}"
                    ),
                    "native_rollout_receipt_digest": sha(f"native-{candidate_id}"),
                    "native_rollout_receipt_file_sha256": sha(
                        f"native-file-{candidate_id}"
                    ),
                    "score": score,
                    "score_path": str(score_path),
                    "score_file_sha256": sha(f"score-file-{candidate_id}"),
                }
            )
            labels.append(
                {
                    "ordinal": len(labels),
                    "group_id": group_id,
                    **{
                        name: candidate[name]
                        for name in (
                            "candidate_id",
                            "analysis_split",
                            "action_family_id",
                            "calibration_group_id",
                            "actor_group_id",
                            "scene_group_id",
                            "action_group_id",
                            "semantic_branch",
                        )
                    },
                    "generation_receipt_digest": generation_digest,
                    "audit_source_kind": "manual_detached",
                    "external_audit_artifact_path": str(root / "audit.json"),
                    "external_audit_artifact_sha256": sha("audit"),
                    "complete_target_transition_observed": is_action,
                    "terminal_hold_observed": is_action,
                    "full_target_action_observed": is_action,
                    "full_target_action_false_confirmed": not is_action,
                }
            )
    manifest_unsigned = {
        "schema_version": author.LABEL_MANIFEST_SCHEMA,
        "root_spec_raw_sha256": sha("spec"),
        "bank_receipt_digest": sha("bank"),
        "candidate_count": 40,
        "candidate_order": [row["candidate"]["candidate_id"] for row in joined],
        "rows": labels,
        "author_acknowledgements": {
            name: True for name in author.ACKNOWLEDGEMENT_FIELDS
        },
        "labels_are_external_and_detached": True,
        "labels_may_enter_model_condition": False,
        "ambiguity_fails_calibration_closed": True,
    }
    manifest = {
        **manifest_unsigned,
        "manifest_digest": author.object_sha256(manifest_unsigned),
    }
    return joined, manifest


def group_bindings(joined: list[dict]) -> list[dict]:
    result = []
    for group_id in bridge.GROUP_IDS:
        rows = [row for row in joined if row["group_id"] == group_id]
        result.append(
            {
                "group_id": group_id,
                "path": f"/sealed/scores/{group_id}/group.json",
                "file_sha256": sha(f"{group_id}-group-file"),
                "receipt_digest": sha(f"{group_id}-group-receipt"),
                "candidate_count": 20,
                "candidate_order": [row["candidate"]["candidate_id"] for row in rows],
                "candidate_receipt_digests": [
                    row["score"]["receipt_digest"] for row in rows
                ],
                "score_receipt_files": [],
                "frozen_scorer_authority": {
                    "authority": "fixture",
                    "method_source_revision": bridge.REQUIRED_SCORER_SOURCE_REVISION,
                },
            }
        )
    return result


class CalibrationBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.joined, self.labels = fixture_rows(self.root)
        self.spec_sha = sha("spec")
        self.bank_digest = sha("bank")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def payloads(self, *, labels: dict | None = None):
        return bridge.build_calibration_payloads(
            joined_rows=self.joined,
            label_manifest=labels or self.labels,
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
        )

    def test_exact_40_scalar_bridge_authorizes_only_existing_gates(self) -> None:
        audits, rows, prereg, receipt = self.payloads()
        self.assertEqual(len(audits), 40)
        self.assertEqual(len(rows), 40)
        self.assertEqual(prereg["calibrator_id"], bridge.CALIBRATOR_ID)
        self.assertTrue(receipt["optimizer_authorized"])
        self.assertEqual(receipt["failure_reasons"], [])
        self.assertFalse(receipt["confirmation_rows_consumed_by_optimizer"])
        self.assertFalse(receipt["t2v_media_consumed_by_calibrator"])
        self.assertTrue(all(row["media_fields_present"] is False for row in rows))

    def test_calibrator_receives_only_scalar_boolean_receipts(self) -> None:
        original = calibration.calibrate_global_action_energy
        observed: dict[str, object] = {}

        def spy(score_rows, event_audits, preregistration, **kwargs):
            observed["score_rows"] = score_rows
            observed["event_audits"] = event_audits
            observed["preregistration"] = preregistration
            return original(score_rows, event_audits, preregistration, **kwargs)

        with mock.patch.object(
            calibration, "calibrate_global_action_energy", side_effect=spy
        ):
            self.payloads()
        for collection in (observed["score_rows"], observed["event_audits"]):
            for row in collection:  # type: ignore[union-attr]
                self.assertNotIn("path", row)
                self.assertFalse(any("tensor" in key for key in row))
                self.assertFalse(any("latent" in key for key in row))
                self.assertFalse(any("prompt" in key for key in row))
                for key in row:
                    if "model" in key or "media" in key:
                        self.assertIs(
                            row[key],
                            False,
                            msg=f"information-flow certificate {key} must fail closed",
                        )

    def test_ambiguity_is_preserved_and_fails_closed(self) -> None:
        labels = copy.deepcopy(self.labels)
        first = labels["rows"][0]
        first["complete_target_transition_observed"] = False
        first["terminal_hold_observed"] = False
        first["full_target_action_observed"] = False
        first["full_target_action_false_confirmed"] = False
        _audits, _rows, _prereg, receipt = self.payloads(labels=labels)
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertIn(
            f"event_audit:{first['candidate_id']}:action",
            receipt["failure_reasons"],
        )

    def test_score_order_mutation_rejected(self) -> None:
        self.joined[0], self.joined[1] = self.joined[1], self.joined[0]
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "label/score order"
        ):
            self.payloads()

    def test_label_generation_digest_mutation_rejected(self) -> None:
        labels = copy.deepcopy(self.labels)
        labels["rows"][0]["generation_receipt_digest"] = sha("wrong-generation")
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "label identity differs"
        ):
            self.payloads(labels=labels)

    def make_bridge(self, *, labels: dict | None = None) -> dict:
        label_value = labels or self.labels
        audits, rows, prereg, calibration_receipt = self.payloads(labels=label_value)
        return bridge.make_bridge_receipt(
            output_dir=self.root / "output",
            root_spec_path=self.root / "spec.json",
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_path=self.root / "bank.json",
            bank_receipt_file_sha256=sha("bank-file"),
            bank_receipt_digest=self.bank_digest,
            label_manifest_path=self.root / "labels.json",
            label_manifest_file_sha256=sha("label-file"),
            label_manifest=label_value,
            score_group_bindings=group_bindings(self.joined),
            joined_rows=self.joined,
            audits=audits,
            score_rows=rows,
            preregistration=prereg,
            calibration_receipt=calibration_receipt,
        )

    @staticmethod
    def reseal(receipt: dict) -> None:
        unsigned = dict(receipt)
        unsigned.pop("receipt_digest")
        receipt["receipt_digest"] = bridge.object_sha256(unsigned)

    def test_bridge_binds_every_generation_score_audit_and_row_in_order(self) -> None:
        receipt = bridge.validate_bridge_receipt(self.make_bridge())
        self.assertEqual(receipt["candidate_count"], 40)
        self.assertEqual(
            [row["candidate_id"] for row in receipt["score_receipt_bindings"]],
            receipt["candidate_order"],
        )
        self.assertTrue(receipt["optimizer_authorized"])
        self.assertEqual(
            receipt["optimizer_authorization_source"],
            "pair_v5_t2v_energy_calibration_v3_existing_fit_and_confirmation_gates_only",
        )
        contract = receipt["calibrator_input_contract"]
        self.assertFalse(contract["media_or_path_fields_present"])
        self.assertFalse(contract["tensor_or_latent_fields_present"])
        self.assertFalse(contract["model_or_prompt_fields_present"])

    def test_score_receipt_digest_mutation_rejected_after_reseal(self) -> None:
        receipt = self.make_bridge()
        receipt["score_receipt_bindings"][0]["receipt_digest"] = sha(
            "mutated-score"
        )
        self.reseal(receipt)
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "digest/order closure"
        ):
            bridge.validate_bridge_receipt(receipt)

    def test_score_row_digest_mutation_rejected_after_reseal(self) -> None:
        receipt = self.make_bridge()
        receipt["score_row_bindings"][0]["row_digest"] = sha("mutated-row")
        self.reseal(receipt)
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "digest/order closure"
        ):
            bridge.validate_bridge_receipt(receipt)

    def test_candidate_binding_order_mutation_rejected_after_reseal(self) -> None:
        receipt = self.make_bridge()
        receipt["generation_bindings"][0], receipt["generation_bindings"][1] = (
            receipt["generation_bindings"][1],
            receipt["generation_bindings"][0],
        )
        self.reseal(receipt)
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "identity/order"
        ):
            bridge.validate_bridge_receipt(receipt)

    def test_bridge_acknowledgement_is_mandatory(self) -> None:
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError,
            "acknowledge-reviewed-label-manifest",
        ):
            bridge.materialize(
                root_spec="/not/read",
                expected_root_spec_sha256=self.spec_sha,
                bank_output_dir="/not/read",
                bank_receipt="/not/read",
                expected_bank_receipt_sha256=sha("bank-file"),
                score_root="/not/read",
                expected_sp4_a_score_group_sha256=sha("a"),
                expected_sp4_b_score_group_sha256=sha("b"),
                detached_label_manifest="/not/read",
                expected_detached_label_manifest_sha256=sha("labels"),
                output_dir=self.root / "new-output",
                acknowledge_reviewed_label_manifest=False,
            )

    def test_score_generation_join_rejects_generation_mutation(self) -> None:
        bound = copy.deepcopy(self.joined[0])
        candidate = bound["candidate"]
        gaussian = {
            "sha256": sha("gaussian-artifact"),
            "raw_value_sha256": sha("gaussian-raw"),
            "content_sha256": sha("gaussian-content"),
        }
        bound["artifacts"] = {
            "mp4": {"sha256": sha("mp4")},
            "predecode_clean_latent": {"sha256": sha("latent")},
            "official_initial_gaussian": gaussian,
        }
        runtime_registry = {
            branch: {"binding": branch} for branch in calibration.BRANCH_ORDER
        }
        caption_registry = {
            branch: f"caption {branch}" for branch in calibration.BRANCH_ORDER
        }
        caption_registry[candidate["semantic_branch"]] = candidate[
            "full_t2v_caption"
        ]
        group = {
            "frozen_checkpoint_receipt_digest": sha("checkpoint"),
            "checkpoint_content_binding": {"binding": "checkpoint"},
            "schedule_coordinate": {"coordinate": "native-516"},
        }
        score = {
            "schema_version": bridge.REQUIRED_SCORE_RECEIPT_SCHEMA,
            **{
                name: candidate[name]
                for name in (
                    "candidate_id",
                    "analysis_split",
                    "action_family_id",
                    "calibration_group_id",
                    "actor_group_id",
                    "scene_group_id",
                    "action_group_id",
                    "semantic_branch",
                )
            },
            "candidate_envelope_sha256": bound["candidate_envelope_sha256"],
            "root_spec_raw_sha256": self.spec_sha,
            "bank_receipt_digest": self.bank_digest,
            "generation_receipt_digest": sha("wrong"),
            "generation_receipt_file_sha256": bound[
                "generation_receipt_file_sha256"
            ],
            "native_rollout_receipt_digest": bound[
                "native_rollout_receipt_digest"
            ],
            "native_rollout_receipt_file_sha256": bound[
                "native_rollout_receipt_file_sha256"
            ],
            "generated_mp4_sha256": sha("mp4"),
            "clean_latent_artifact_sha256": sha("latent"),
            "geometry_source_video_sha256": candidate[
                "geometry_source_video_sha256"
            ],
            "full_t2v_caption_utf8_sha256": candidate[
                "full_t2v_caption_utf8_sha256"
            ],
            "official_gaussian_artifact_sha256": gaussian["sha256"],
            "official_gaussian_raw_value_sha256": gaussian["raw_value_sha256"],
            "official_gaussian_content_sha256": gaussian["content_sha256"],
            "frozen_checkpoint_receipt_digest": sha("checkpoint"),
            "checkpoint_content_binding": {"binding": "checkpoint"},
            "schedule_coordinate": {"coordinate": "native-516"},
            "generation_runtime_binding_by_branch": runtime_registry,
            "full_t2v_caption_by_branch": caption_registry,
        }
        with self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError, "score/generation join"
        ):
            bridge.validate_score_generation_join(
                score,
                bound,
                group_receipt=group,
                root_spec_raw_sha256=self.spec_sha,
                bank_receipt_digest=self.bank_digest,
                expected_generation_registry=runtime_registry,
                expected_caption_registry=caption_registry,
            )

    def test_optional_cpu_launcher_has_no_gpu_and_binds_every_authority(self) -> None:
        text = CPU_LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", text)
        self.assertIn('export ROCR_VISIBLE_DEVICES=""', text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_SP4_A_GROUP_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_SP4_B_GROUP_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_LABEL_MANIFEST_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_SOURCE_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_AUTHOR_SOURCE_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_CALIBRATOR_SOURCE_SHA256", text)
        self.assertIn("PAIR_V5_CAL_BRIDGE_SCORER_SOURCE_SHA256", text)
        self.assertIn(
            "I_REVIEWED_THE_SEALED_40_ROW_DETACHED_LABEL_MANIFEST", text
        )
        self.assertIn("--acknowledge-reviewed-label-manifest", text)
        self.assertIn('!= 40', text)
        self.assertIn('confirmation_samples_consumed_by_optimizer', text)
        self.assertIn(
            "materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3.py",
            text,
        )
        self.assertIn(
            "author_pair_v5_core4_event_labels_d541801_v3.py",
            text,
        )
        self.assertNotIn(
            'author_source="${method_root}/tools/author_pair_v5_core4_event_labels_v3.py"',
            text,
        )
        self.assertIn("pair-v5-t2v-global-energy-sp4-a-v3.json", text)
        self.assertIn("pair-v5-t2v-global-energy-sp4-b-v3.json", text)
        self.assertNotIn("global-energy-sp4-a-v4", text)
        self.assertIn(bridge.REQUIRED_SCORER_SOURCE_REVISION, text)

    def test_unique_bridge_hard_binds_completed_d541801_v3_evidence(self) -> None:
        self.assertEqual(
            bridge.REQUIRED_SCORER_SOURCE_REVISION,
            "d541801a162796aacde34c2bfc2b1f0472d954d2",
        )
        self.assertEqual(
            bridge.REQUIRED_SCORE_RECEIPT_SCHEMA,
            "bernini-pair-v5-frozen-t2v-global-energy-score-v3",
        )
        self.assertEqual(
            bridge.REQUIRED_GROUP_RECEIPT_SCHEMA,
            "bernini-pair-v5-frozen-t2v-global-energy-group-v3",
        )
        self.assertEqual(
            bridge.SCORE_FILENAME, "pair-v5-t2v-global-energy-score-v3.json"
        )
        self.assertEqual(
            bridge.GROUP_FILENAME,
            "pair-v5-t2v-global-energy-{group_id}-v3.json",
        )

    def test_unique_bridge_rejects_a_v4_scorer_runtime(self) -> None:
        fake_v4 = SimpleNamespace(
            SCORE_RECEIPT_SCHEMA="bernini-pair-v5-frozen-t2v-global-energy-score-v4",
            GROUP_RECEIPT_SCHEMA="bernini-pair-v5-frozen-t2v-global-energy-group-v4",
        )
        with mock.patch.object(
            bridge.importlib, "import_module", return_value=fake_v4
        ), self.assertRaisesRegex(
            bridge.PairV5Core4CalibrationBridgeError,
            "requires the sealed d541801 v3 scorer runtime",
        ):
            bridge._scorer_runtime()


if __name__ == "__main__":
    unittest.main()
