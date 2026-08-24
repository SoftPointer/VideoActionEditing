#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = METHOD_ROOT / "tools"
for search_root in (METHOD_ROOT, TOOLS_ROOT, TEST_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import author_pair_v5_core4_event_labels_d541801_v3 as label_author  # noqa: E402
import materialize_pair_v5_t2v_energy_calibration_bridge_d541801_v3 as audit_bridge  # noqa: E402
import materialize_temporal_counterfactual_calibration_v1 as materializer  # noqa: E402
import temporal_counterfactual_calibration_v1 as calibration  # noqa: E402
import temporal_counterfactual_contract_v1 as contract  # noqa: E402
import test_temporal_counterfactual_contract_v1 as fixtures  # noqa: E402


LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_materialize_temporal_counterfactual_calibration_v1_cpu.sbatch"
)


def write_json(path: Path, value: dict) -> str:
    raw = contract.canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class TemporalCounterfactualMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="temporal-cf-materializer-test-", dir="/private/tmp"
        )
        cls.root = Path(cls.temporary.name)
        cls.root_spec = (
            METHOD_ROOT
            / "assets"
            / "pair_v5_t2v_calibration_core4_bank_v2.json"
        ).resolve()
        cls.bank_receipt = (
            METHOD_ROOT.parents[1]
            / "tmp"
            / "pair_v5_t2v_core4_v2_event_review_17cc2c7"
            / "pair-v5-t2v-calibration-bank-receipt.json"
        ).resolve()
        cls.preregistration = (
            METHOD_ROOT
            / "assets"
            / "temporal_counterfactual_preregistration_v1.json"
        ).resolve()
        cls.calibrator_archive = Path(
            "/tmp/temporal_cf_full_f7ffa59.tar"
        ).resolve()
        if not cls.bank_receipt.is_file():
            raise unittest.SkipTest("sealed local core4-v2 bank receipt is absent")
        if not cls.calibrator_archive.is_file():
            raise unittest.SkipTest("sealed f7 temporal source archive is absent")

        cls.scores, cls.audits = fixtures.make_population()
        cls.groups = fixtures.make_group_receipts(cls.scores)
        by_candidate = {
            row["candidate_identity"]["candidate_id"]: row for row in cls.scores
        }

        cls.score_root = cls.root / "scores"
        cls.score_root.mkdir()
        (cls.score_root / "sp4-a.log").write_text("sp4-a complete\n", encoding="ascii")
        (cls.score_root / "sp4-b.log").write_text("sp4-b complete\n", encoding="ascii")
        cls.group_file_shas: dict[str, str] = {}
        for group in cls.groups:
            group_id = group["group_id"]
            group_root = cls.score_root / group_id
            group_root.mkdir()
            for candidate_id in group["candidate_order"]:
                candidate_root = group_root / candidate_id
                candidate_root.mkdir()
                write_json(
                    candidate_root / materializer.SCORE_FILENAME,
                    by_candidate[candidate_id],
                )
            group_path = group_root / materializer.GROUP_FILENAME.format(
                group_id=group_id
            )
            cls.group_file_shas[group_id] = write_json(group_path, group)

        cls.audit_root = cls.root / "event_audits"
        cls.audit_root.mkdir()
        audit_bindings = []
        label_rows = []
        generation_bindings = []
        score_receipt_bindings = []
        score_row_bindings = []
        candidate_order = []
        for ordinal, (score, audit) in enumerate(zip(cls.scores, cls.audits)):
            identity = score["candidate_identity"]
            candidate_id = identity["candidate_id"]
            candidate_order.append(candidate_id)
            audit_path = cls.audit_root / f"{ordinal:02d}-{candidate_id}.json"
            audit_sha = write_json(audit_path, audit)
            audit_bindings.append(
                {
                    "ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "path": str(audit_path),
                    "file_sha256": audit_sha,
                    "receipt_digest": audit["receipt_digest"],
                }
            )
            label_rows.append(
                {
                    "ordinal": ordinal,
                    "group_id": score["group_id"],
                    **identity,
                    "generation_receipt_digest": score["generation_binding"][
                        "generation_receipt_digest"
                    ],
                    "audit_source_kind": audit["audit_source_kind"],
                    "external_audit_artifact_path": str(
                        cls.root / "detached-review" / f"{candidate_id}.json"
                    ),
                    "external_audit_artifact_sha256": audit[
                        "external_audit_artifact_sha256"
                    ],
                    **{
                        name: audit[name]
                        for name in label_author.LABEL_BOOLEAN_FIELDS
                    },
                }
            )
            generation_bindings.append(
                {
                    "ordinal": ordinal,
                    "group_id": score["group_id"],
                    "candidate_id": candidate_id,
                    "generation_receipt_digest": score["generation_binding"][
                        "generation_receipt_digest"
                    ],
                }
            )
            score_row_digest = hashlib.sha256(
                f"score-row-{ordinal}".encode("ascii")
            ).hexdigest()
            score_receipt_bindings.append(
                {
                    "ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "receipt_digest": score["receipt_digest"],
                    "score_row_digest": score_row_digest,
                }
            )
            score_row_bindings.append(
                {
                    "ordinal": ordinal,
                    "candidate_id": candidate_id,
                    "row_digest": score_row_digest,
                }
            )

        acknowledgements = {
            name: True for name in label_author.ACKNOWLEDGEMENT_FIELDS
        }
        label_unsigned = {
            "schema_version": label_author.LABEL_MANIFEST_SCHEMA,
            "root_spec_raw_sha256": contract.REQUIRED_CORE4_V2_SPEC_SHA256,
            "bank_receipt_digest": contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
            "candidate_count": 40,
            "candidate_order": candidate_order,
            "rows": label_rows,
            "author_acknowledgements": acknowledgements,
            "labels_are_external_and_detached": True,
            "labels_may_enter_model_condition": False,
            "ambiguity_fails_calibration_closed": True,
        }
        cls.labels = {
            **label_unsigned,
            "manifest_digest": label_author.object_sha256(label_unsigned),
        }
        cls.label_path = cls.root / "detached-labels.json"
        cls.label_file_sha = write_json(cls.label_path, cls.labels)

        bridge_score_groups = []
        for group in cls.groups:
            bridge_score_groups.append(
                {
                    "group_id": group["group_id"],
                    "candidate_count": 20,
                    "candidate_order": list(group["candidate_order"]),
                    "candidate_receipt_digests": list(
                        group["candidate_receipt_digests"]
                    ),
                    "frozen_scorer_authority": {
                        "method_source_revision": (
                            audit_bridge.REQUIRED_SCORER_SOURCE_REVISION
                        )
                    },
                }
            )
        bridge_unsigned = {
            "schema_version": audit_bridge.BRIDGE_SCHEMA,
            "required_frozen_scorer_source_revision": (
                audit_bridge.REQUIRED_SCORER_SOURCE_REVISION
            ),
            "source_root_spec": {
                "path": str(cls.root_spec),
                "file_sha256": contract.REQUIRED_CORE4_V2_SPEC_SHA256,
                "schema_version": (
                    "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2"
                ),
            },
            "source_bank_receipt": {
                "path": str(cls.bank_receipt),
                "file_sha256": contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256,
                "receipt_digest": contract.REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST,
            },
            "detached_event_label_manifest": {
                "path": str(cls.label_path),
                "file_sha256": cls.label_file_sha,
                "manifest_digest": cls.labels["manifest_digest"],
                "candidate_count": 40,
                "author_acknowledgements": acknowledgements,
            },
            "score_group_receipts": bridge_score_groups,
            "candidate_count": 40,
            "candidate_order": candidate_order,
            "generation_bindings": generation_bindings,
            "score_receipt_bindings": score_receipt_bindings,
            "event_audit_receipt_bindings": audit_bindings,
            "score_row_bindings": score_row_bindings,
            "preregistration_binding": {},
            "calibration_binding": {"optimizer_authorized": False},
            "calibrator_input_contract": {
                "input_object_kinds": [
                    "scalar_score_rows",
                    "detached_boolean_event_audit_receipts",
                    "preregistration",
                ],
                "score_scalar_field": "raw_global_action_energy_score",
                "media_or_path_fields_present": False,
                "tensor_or_latent_fields_present": False,
                "model_or_prompt_fields_present": False,
                "external_audit_artifact_contents_enter_calibrator": False,
                "generation_artifact_contents_enter_calibrator": False,
                "provenance_files_verified_before_calibrator": True,
            },
            "ambiguous_candidate_ids": [],
            "optimizer_authorized": False,
            "optimizer_authorization_source": (
                "pair_v5_t2v_energy_calibration_v3_existing_fit_and_confirmation_gates_only"
            ),
            "confirmation_samples_consumed_by_optimizer": False,
            "training_performed": False,
            "scientific_action_editing_claim": False,
        }
        cls.bridge = {
            **bridge_unsigned,
            "receipt_digest": audit_bridge.object_sha256(bridge_unsigned),
        }
        audit_bridge.validate_bridge_receipt(cls.bridge)
        cls.bridge_path = cls.root / "audit-bridge.json"
        cls.bridge_file_sha = write_json(cls.bridge_path, cls.bridge)

    @classmethod
    def tearDownClass(cls) -> None:
        for directory, directories, filenames in os.walk(cls.root, topdown=False):
            for filename in filenames:
                os.chmod(Path(directory) / filename, 0o600)
            for name in directories:
                os.chmod(Path(directory) / name, 0o700)
        os.chmod(cls.root, 0o700)
        cls.temporary.cleanup()

    def authority_patches(self):
        return mock.patch.multiple(
            materializer,
            REQUIRED_AUDIT_BRIDGE_FILE_SHA256=self.bridge_file_sha,
            REQUIRED_AUDIT_BRIDGE_RECEIPT_DIGEST=self.bridge["receipt_digest"],
            REQUIRED_LABEL_MANIFEST_FILE_SHA256=self.label_file_sha,
        )

    def materialize_arguments(self, output: Path) -> dict:
        return {
            "root_spec": self.root_spec,
            "expected_root_spec_sha256": contract.REQUIRED_CORE4_V2_SPEC_SHA256,
            "bank_receipt": self.bank_receipt,
            "expected_bank_receipt_sha256": (
                contract.REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
            ),
            "score_root": self.score_root,
            "expected_sp4_a_score_group_sha256": self.group_file_shas["sp4-a"],
            "expected_sp4_b_score_group_sha256": self.group_file_shas["sp4-b"],
            "score_group_file_sha256_authority": "presealed_external",
            "audit_bridge_receipt": self.bridge_path,
            "expected_audit_bridge_sha256": self.bridge_file_sha,
            "event_audit_root": self.audit_root,
            "detached_label_manifest": self.label_path,
            "expected_detached_label_manifest_sha256": self.label_file_sha,
            "preregistration": self.preregistration,
            "expected_preregistration_sha256": (
                materializer.REQUIRED_PREREGISTRATION_FILE_SHA256
            ),
            "calibrator_source_archive": self.calibrator_archive,
            "calibrator_source_revision": (
                materializer.REQUIRED_CALIBRATOR_SOURCE_REVISION
            ),
            "calibrator_source_archive_sha256": (
                materializer.REQUIRED_CALIBRATOR_SOURCE_ARCHIVE_SHA256
            ),
            "expected_calibrator_source_sha256": (
                materializer.REQUIRED_CALIBRATOR_SOURCE_SHA256
            ),
            "expected_materializer_source_sha256": materializer.file_sha256(
                materializer.__file__
            ),
            "output_dir": output,
            "acknowledge_reviewed_label_manifest": True,
        }

    def test_exact40_materialization_replays_before_and_after_write(self) -> None:
        output = self.root / "output-pass"
        with self.authority_patches():
            receipt = materializer.materialize(**self.materialize_arguments(output))
            self.assertTrue(receipt["optimizer_authorized"])
            self.assertEqual(receipt["failure_reasons"], [])
            self.assertEqual(receipt["candidate_count"], 40)
            self.assertEqual(
                {entry.name for entry in output.iterdir()},
                {
                    materializer.PREREGISTRATION_FILENAME,
                    materializer.CALIBRATION_FILENAME,
                    materializer.MATERIALIZATION_FILENAME,
                },
            )
            self.assertTrue(all(not entry.is_symlink() for entry in output.iterdir()))
            replayed = materializer.replay_materialized_output(
                **self.materialize_arguments(output)
            )
            self.assertEqual(replayed, receipt)
            with self.assertRaisesRegex(
                materializer.TemporalCounterfactualMaterializationError,
                "GO materialization requires exact external replay",
            ):
                materializer.validate_materialization_receipt(receipt)

    def test_score_root_exactly_allows_two_groups_and_two_nonsemantic_logs(self) -> None:
        groups, scores, bindings, logs, root = (
            materializer.load_ordered_score_population(
                score_root=self.score_root,
                expected_group_file_sha256_by_id=self.group_file_shas,
            )
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(scores), 40)
        self.assertEqual(len(bindings), 2)
        self.assertEqual([row["name"] for row in logs], ["sp4-a.log", "sp4-b.log"])
        self.assertEqual(root, self.score_root)

        extra = self.score_root / "unsealed-extra.log"
        extra.write_text("unexpected\n", encoding="ascii")
        try:
            with self.assertRaisesRegex(
                materializer.TemporalCounterfactualMaterializationError,
                "score root closure differs",
            ):
                materializer.load_ordered_score_population(
                    score_root=self.score_root,
                    expected_group_file_sha256_by_id=self.group_file_shas,
                )
        finally:
            extra.unlink()

    def test_expected_log_name_cannot_be_a_symlink(self) -> None:
        root = self.root / "symlink-root"
        root.mkdir()
        (root / "sp4-a").mkdir()
        (root / "sp4-b").mkdir()
        target = root / "outside.log"
        target.write_text("outside\n", encoding="ascii")
        (root / "sp4-a.log").symlink_to(target)
        (root / "sp4-b.log").write_text("plain\n", encoding="ascii")
        # Remove the target from root closure without deleting the symlink target.
        moved_target = self.root / "outside.log"
        target.rename(moved_target)
        (root / "sp4-a.log").unlink()
        (root / "sp4-a.log").symlink_to(moved_target)
        with self.assertRaisesRegex(
            materializer.TemporalCounterfactualMaterializationError,
            "symlink component",
        ):
            materializer._validate_score_root_closure(root)

    def test_truthy_forged_optimizer_authority_is_rejected(self) -> None:
        output = self.root / "output-forged-base"
        with self.authority_patches():
            receipt = materializer.materialize(**self.materialize_arguments(output))
            forged = copy.deepcopy(receipt)
            forged["optimizer_authorized"] = "true"
            unsigned = dict(forged)
            unsigned.pop("receipt_digest")
            forged["receipt_digest"] = materializer.object_sha256(unsigned)
            with self.assertRaisesRegex(
                materializer.TemporalCounterfactualMaterializationError,
                "optimizer/failure fields differ",
            ):
                materializer.validate_materialization_receipt(forged)

    def test_create_only_writer_uses_kernel_exclusive_creation(self) -> None:
        path = self.root / "exclusive-create.json"
        first = {"value": 1}
        materializer._write_create_only(path, first)
        original = path.read_bytes()
        with self.assertRaisesRegex(
            materializer.TemporalCounterfactualMaterializationError,
            "refusing to overwrite",
        ):
            materializer._write_create_only(path, {"value": 2})
        self.assertEqual(path.read_bytes(), original)
        source = Path(materializer.__file__).read_text(encoding="utf-8")
        self.assertIn("os.O_EXCL", source)

    def test_cli_requires_review_ack_and_all_formal_hash_inputs(self) -> None:
        parser = materializer.build_parser()
        options = [
            "--root-spec", "/x/spec.json",
            "--expected-root-spec-sha256", "1" * 64,
            "--bank-receipt", "/x/bank.json",
            "--expected-bank-receipt-sha256", "2" * 64,
            "--score-root", "/x/scores",
            "--expected-sp4-a-score-group-sha256", "3" * 64,
            "--expected-sp4-b-score-group-sha256", "4" * 64,
            "--score-group-file-sha256-authority", "presealed_external",
            "--audit-bridge-receipt", "/x/bridge.json",
            "--expected-audit-bridge-sha256", "5" * 64,
            "--event-audit-root", "/x/audits",
            "--detached-label-manifest", "/x/labels.json",
            "--expected-detached-label-manifest-sha256", "6" * 64,
            "--preregistration", "/x/prereg.json",
            "--expected-preregistration-sha256", "7" * 64,
            "--calibrator-source-archive", "/x/source.tar",
            "--calibrator-source-revision", "8" * 40,
            "--calibrator-source-archive-sha256", "9" * 64,
            "--expected-calibrator-source-sha256", "a" * 64,
            "--expected-materializer-source-sha256", "b" * 64,
            "--output-dir", "/x/output",
        ]
        args = parser.parse_args(options)
        self.assertFalse(args.acknowledge_reviewed_label_manifest)

    def test_cpu_launcher_is_hash_bound_afterok_ready_and_not_self_launching(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("dependency=afterok:131237", text)
        self.assertIn("#SBATCH --gres=gpu:mi210:1", text)
        self.assertIn("scheduler-required GPU is allocated but hidden", text)
        self.assertNotIn("sbatch ", text)
        self.assertIn("sp4-a.log", Path(materializer.__file__).read_text("utf-8"))
        self.assertIn("sp4-b.log", Path(materializer.__file__).read_text("utf-8"))
        self.assertIn(materializer.REQUIRED_AUDIT_BRIDGE_FILE_SHA256, text)
        self.assertIn(materializer.REQUIRED_LABEL_MANIFEST_FILE_SHA256, text)
        self.assertIn(materializer.REQUIRED_PREREGISTRATION_FILE_SHA256, text)
        self.assertIn(materializer.REQUIRED_CALIBRATOR_SOURCE_SHA256, text)
        self.assertIn("TEMPORAL_CF_CAL_SP4_A_GROUP_SHA256", text)
        self.assertIn("TEMPORAL_CF_CAL_SP4_B_GROUP_SHA256", text)
        self.assertIn("TEMPORAL_CF_CAL_CALIBRATOR_ARCHIVE", text)
        self.assertIn("git get-tar-commit-id", text)
        self.assertIn("observed_afterok_runtime", text)
        self.assertIn("--verify-existing-output", text)
        self.assertIn("ROCR_VISIBLE_DEVICES=\"\"", text)
        self.assertIn("--acknowledge-reviewed-label-manifest", text)
        self.assertIn("exact40_replay=true", text)

    def test_materializer_source_has_no_torch_subprocess_or_training_call(self) -> None:
        source = Path(materializer.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("backward(", source)
        self.assertNotIn("VideoReader", source)


if __name__ == "__main__":
    unittest.main()
