#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for search_root in (METHOD_ROOT, TOOLS_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import author_pair_v5_core4_event_labels_v3 as author  # noqa: E402
import pair_v5_t2v_energy_calibration_v3 as calibration  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def make_bound_rows(media_path: Path) -> list[dict]:
    rows = []
    cells = (
        ("sp4-a", "fit", "dog-sit", "dog-fit"),
        ("sp4-a", "confirmation", "human-stand", "human-confirm"),
        ("sp4-b", "fit", "human-stand", "human-fit"),
        ("sp4-b", "confirmation", "dog-sit", "dog-confirm"),
    )
    for cell_index, (group_id, split, family, stem) in enumerate(cells):
        for branch in calibration.BRANCH_ORDER:
            candidate_id = f"pair5-t2v-core4-v2-{stem}-{branch}"
            rows.append(
                {
                    "group_id": group_id,
                    "candidate": {
                        "candidate_id": candidate_id,
                        "analysis_split": split,
                        "action_family_id": family,
                        "calibration_group_id": f"cell-{stem}",
                        "actor_group_id": f"actor-{stem}",
                        "scene_group_id": f"scene-{stem}",
                        "action_group_id": f"action-{stem}",
                        "semantic_branch": branch,
                        "full_t2v_caption": f"Complete caption {cell_index} {branch}.",
                    },
                    "generation_receipt_digest": sha(f"generation-{candidate_id}"),
                    "artifacts": {
                        "mp4": {
                            "path": str(media_path),
                            "sha256": author.file_sha256(media_path),
                        }
                    },
                }
            )
    return rows


class Core4LabelAuthoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media = self.root / "review.mp4"
        self.media.write_bytes(b"review media provenance")
        self.audit = self.root / "detached-audit.json"
        self.audit.write_bytes(b'{"reviewed":true}\n')
        self.rows = make_bound_rows(self.media)
        self.spec_sha = sha("spec")
        self.bank_digest = sha("bank")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def template(self) -> dict:
        return author.make_authoring_template(
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
        )

    def completed_template(self) -> dict:
        value = self.template()
        artifact_sha = author.file_sha256(self.audit)
        for row in value["rows"]:
            action = row["semantic_branch"] == calibration.ACTION_BRANCH
            row.update(
                {
                    "audit_source_kind": "manual_detached",
                    "external_audit_artifact_path": str(self.audit),
                    "external_audit_artifact_sha256": artifact_sha,
                    "complete_target_transition_observed": action,
                    "terminal_hold_observed": action,
                    "full_target_action_observed": action,
                    "full_target_action_false_confirmed": not action,
                    "annotation_complete": True,
                }
            )
        return value

    @staticmethod
    def acknowledgements(**overrides: bool) -> dict[str, bool]:
        result = {name: True for name in author.ACKNOWLEDGEMENT_FIELDS}
        result.update(overrides)
        return result

    def test_template_contains_no_invented_event_label(self) -> None:
        template = self.template()
        checked = author.validate_authoring_template(
            template,
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
            require_complete=False,
        )
        self.assertEqual(checked["candidate_count"], 40)
        for row in checked["rows"]:
            self.assertFalse(row["annotation_complete"])
            for name in (
                "audit_source_kind",
                "external_audit_artifact_path",
                "external_audit_artifact_sha256",
                *author.LABEL_BOOLEAN_FIELDS,
            ):
                self.assertIsNone(row[name])

    def test_completed_template_cannot_seal_without_every_acknowledgement(self) -> None:
        acknowledgements = self.acknowledgements(
            labels_not_inferred_from_semantic_branch=False
        )
        with self.assertRaisesRegex(
            author.PairV5Core4LabelAuthoringError, "acknowledgements"
        ):
            author.seal_label_manifest(
                completed_template=self.completed_template(),
                root_spec_raw_sha256=self.spec_sha,
                bank_receipt_digest=self.bank_digest,
                bound_rows=self.rows,
                acknowledgements=acknowledgements,
            )

    def test_sealed_manifest_is_exact_order_and_hash_bound(self) -> None:
        manifest = author.seal_label_manifest(
            completed_template=self.completed_template(),
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
            acknowledgements=self.acknowledgements(),
        )
        checked = author.validate_label_manifest(
            manifest,
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
        )
        self.assertEqual(
            checked["candidate_order"],
            [row["candidate"]["candidate_id"] for row in self.rows],
        )
        self.assertFalse(checked["labels_may_enter_model_condition"])
        self.assertTrue(checked["ambiguity_fails_calibration_closed"])

    def test_ambiguous_row_is_preserved_not_guessed(self) -> None:
        template = self.completed_template()
        row = template["rows"][0]
        row["complete_target_transition_observed"] = False
        row["terminal_hold_observed"] = False
        row["full_target_action_observed"] = False
        row["full_target_action_false_confirmed"] = False
        manifest = author.seal_label_manifest(
            completed_template=template,
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
            acknowledgements=self.acknowledgements(),
        )
        self.assertFalse(manifest["rows"][0]["full_target_action_observed"])
        self.assertFalse(
            manifest["rows"][0]["full_target_action_false_confirmed"]
        )

    def test_contradictory_observed_and_false_label_rejected(self) -> None:
        template = self.completed_template()
        template["rows"][0]["full_target_action_false_confirmed"] = True
        with self.assertRaisesRegex(
            author.PairV5Core4LabelAuthoringError, "both observed and false"
        ):
            author.seal_label_manifest(
                completed_template=template,
                root_spec_raw_sha256=self.spec_sha,
                bank_receipt_digest=self.bank_digest,
                bound_rows=self.rows,
                acknowledgements=self.acknowledgements(),
            )

    def test_candidate_order_mutation_rejected_even_with_resealed_manifest(self) -> None:
        manifest = author.seal_label_manifest(
            completed_template=self.completed_template(),
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
            acknowledgements=self.acknowledgements(),
        )
        mutated = copy.deepcopy(manifest)
        mutated["rows"][0], mutated["rows"][1] = (
            mutated["rows"][1],
            mutated["rows"][0],
        )
        unsigned = dict(mutated)
        unsigned.pop("manifest_digest")
        mutated["manifest_digest"] = author.object_sha256(unsigned)
        with self.assertRaisesRegex(
            author.PairV5Core4LabelAuthoringError, "identity/order"
        ):
            author.validate_label_manifest(
                mutated,
                root_spec_raw_sha256=self.spec_sha,
                bank_receipt_digest=self.bank_digest,
                bound_rows=self.rows,
            )

    def test_artifact_byte_mutation_rejected(self) -> None:
        manifest = author.seal_label_manifest(
            completed_template=self.completed_template(),
            root_spec_raw_sha256=self.spec_sha,
            bank_receipt_digest=self.bank_digest,
            bound_rows=self.rows,
            acknowledgements=self.acknowledgements(),
        )
        self.audit.write_bytes(b"changed")
        with self.assertRaisesRegex(
            author.PairV5Core4LabelAuthoringError, "artifact changed"
        ):
            author.validate_label_manifest(
                manifest,
                root_spec_raw_sha256=self.spec_sha,
                bank_receipt_digest=self.bank_digest,
                bound_rows=self.rows,
            )

    def test_seal_cli_requires_all_four_explicit_ack_flags(self) -> None:
        parser = author.build_parser()
        args = parser.parse_args(
            [
                "seal",
                "--root-spec",
                "/x/spec.json",
                "--expected-root-spec-sha256",
                self.spec_sha,
                "--bank-output-dir",
                "/x/bank",
                "--bank-receipt",
                "/x/bank/receipt.json",
                "--expected-bank-receipt-sha256",
                sha("bank-file"),
                "--completed-template",
                "/x/completed.json",
                "--expected-completed-template-sha256",
                sha("completed"),
                "--output",
                "/x/labels.json",
            ]
        )
        self.assertFalse(args.ack_all_40_rows_individually_reviewed)
        self.assertFalse(args.ack_no_semantic_branch_defaults)
        self.assertFalse(args.ack_ambiguity_left_unresolved)
        self.assertFalse(args.ack_detached_artifacts_never_model_conditioning)


if __name__ == "__main__":
    unittest.main()
