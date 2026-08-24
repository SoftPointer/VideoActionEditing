from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for entry in (str(METHOD_ROOT), str(TOOLS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import stage_a_source_edge_confirmation_contract_v1 as contract  # noqa: E402
import materialize_stage_a_source_edge_confirmation_manifest_v1 as tool  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class StageASourceEdgeConfirmationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.review_path = self.root / "review.json"
        _write_json(self.review_path, {"manifest_digest": "a" * 64})
        self.rows = []
        shapes = ([1, 16, 21, 60, 62], [1, 16, 21, 70, 52])
        for index, sentinel_id in enumerate(contract.SENTINEL_ORDER):
            source = self.root / f"source-{index}.mp4"
            source.write_bytes(f"source-{index}".encode("ascii"))
            instructions = {
                "forward": f"sentinel {index} forward",
                "noop": f"sentinel {index} noop",
                "reverse": f"sentinel {index} reverse",
                "incomplete": f"sentinel {index} incomplete",
                "camera-only": f"sentinel {index} camera",
                "appearance-only": f"sentinel {index} appearance",
            }
            self.rows.append(
                {
                    "sentinel_id": sentinel_id,
                    "diversity_role": f"role-{index}",
                    "source_entity_type": f"entity-{index}",
                    "iid": f"iid-{index}",
                    "action_family": f"action-{index}",
                    "source_caption": f"source caption {index}",
                    "source_video": str(source),
                    "source_video_sha256": contract.file_sha256(source),
                    "latent_shape": list(shapes[0 if index in (0, 2) else 1]),
                    "seed": 9000 + index,
                    "instructions": instructions,
                    "instruction_sha256": {
                        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
                        for key, value in instructions.items()
                    },
                    "wrong_owner_iid": f"iid-{2 if index == 0 else 3 if index == 1 else 0 if index == 2 else 1}",
                }
            )
        by_iid = {row["iid"]: row for row in self.rows}
        for row in self.rows:
            row["wrong_owner_source_video_sha256"] = by_iid[
                row["wrong_owner_iid"]
            ]["source_video_sha256"]
        self.review_value = {
            "manifest_digest": "a" * 64,
            "sentinels": self.rows,
        }
        self.formal_cells = []
        for family in ("dog", "human"):
            receipt = self.root / f"{family}-receipt.json"
            correct_iid = f"a1-{family}-correct"
            wrong_iid = f"a1-{family}-wrong"
            correct_sha = ("d" if family == "dog" else "e") * 64
            wrong_sha = ("f" if family == "dog" else "0") * 64
            receipt_unsigned = {
                "complete": True,
                "shard": {"family": family},
                "authority": {
                    "correct_row": {"iid": correct_iid},
                    "wrong_owner_row": {"iid": wrong_iid},
                },
                "source": {
                    "correct_sha256": correct_sha,
                    "wrong_owner_sha256": wrong_sha,
                },
            }
            receipt_digest = contract.object_sha256(receipt_unsigned)
            _write_json(
                receipt,
                {**receipt_unsigned, "receipt_digest": receipt_digest},
            )
            self.formal_cells.append(
                {
                    "family": family,
                    "receipt_path": receipt,
                    "receipt_file_sha256": contract.file_sha256(receipt),
                    "receipt_digest": receipt_digest,
                    "correct_iid": correct_iid,
                    "wrong_iid": wrong_iid,
                    "correct_source_sha256": correct_sha,
                    "wrong_source_sha256": wrong_sha,
                }
            )
        self.authorization_path = self.root / "authorization.json"
        self._write_authorization()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_authorization(self, *, cell: object = None) -> dict:
        admitted = {"schedule_index": 29, "block_band": "early_middle"} if cell is None else cell
        unsigned = {
            "schema_version": contract.AUTHORIZATION_SCHEMA,
            "authorization_id": "manual-a1-winner-20260814",
            "evidence_role": contract.EVIDENCE_ROLE,
            "a1_formal_receipts": [
                contract._formal_receipt_record(item) for item in self.formal_cells
            ],
            "admitted_cell": admitted,
            "manual_review": {
                "reviewer": "human-reviewer",
                "reviewed_at_utc": "2026-08-14T12:00:00Z",
                "rationale": "Externally reviewed single-cell confirmation request.",
                "exactly_one_cell_authorized": True,
                "automatic_choice_used": False,
            },
            "scope": {
                "winner_robustness_evidence_only": True,
                "stage_b_admission": False,
                "stage_b_two_band_rule_unchanged": True,
                "quality_claim_deferred_to_human_review": True,
            },
        }
        value = {**unsigned, "authorization_digest": contract.object_sha256(unsigned)}
        _write_json(self.authorization_path, value)
        return value

    def _materialize(self) -> dict:
        with mock.patch.object(
            contract.review, "load_manifest", return_value=self.review_value
        ), mock.patch.object(
            contract.formal,
            "_validate_cell",
            side_effect=self.formal_cells,
        ):
            return dict(
                contract.materialize_manifest_value(
                    review_manifest_path=self.review_path,
                    expected_review_manifest_sha256=contract.file_sha256(self.review_path),
                    dog_formal_output=self.root,
                    human_formal_output=self.root,
                    authorization_path=self.authorization_path,
                    expected_authorization_sha256=contract.file_sha256(self.authorization_path),
                )
            )

    def test_exact14_manifest_and_roundtrip(self) -> None:
        value = self._materialize()
        self.assertEqual(len(value["plan"]), 14)
        self.assertEqual(
            [row["key"] for row in value["plan"][:6]],
            [f"native-correct-{branch}" for branch in contract.BRANCHES],
        )
        self.assertEqual(value["admitted_cell"]["schedule_index"], 29)
        self.assertEqual(value["admitted_cell"]["block_indices"], list(range(8, 16)))
        self.assertFalse(value["scope"]["stage_b_admission"])
        output = self.root / "execution.json"
        contract.write_create_only_json(output, value)
        with mock.patch.object(
            contract.review, "load_manifest", return_value=self.review_value
        ):
            loaded = contract.load_manifest(
                output,
                expected_file_sha256=contract.file_sha256(output),
                verify_files=True,
            )
        self.assertEqual(loaded["manifest_digest"], value["manifest_digest"])

    def test_no_schedule_or_band_cli_escape_hatch(self) -> None:
        options = {action.dest for action in tool.build_parser()._actions}
        self.assertNotIn("schedule_index", options)
        self.assertNotIn("block_band", options)

    def test_zero_or_multiple_cells_fail(self) -> None:
        for invalid in ([], [{"schedule_index": 29, "block_band": "early_middle"}] * 2):
            with self.subTest(invalid=invalid):
                self._write_authorization(cell=invalid)
                with self.assertRaisesRegex(
                    contract.SourceEdgeConfirmationError, "exactly one"
                ):
                    self._materialize()

    def test_unregistered_cell_fails(self) -> None:
        self._write_authorization(
            cell={"schedule_index": 17, "block_band": "early_middle"}
        )
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "outside the A1 formal registry"
        ):
            self._materialize()

    def test_a1_receipt_byte_drift_fails_after_manifest_seal(self) -> None:
        value = self._materialize()
        output = self.root / "execution.json"
        contract.write_create_only_json(output, value)
        Path(self.formal_cells[0]["receipt_path"]).write_text("drift\n", encoding="utf-8")
        with mock.patch.object(
            contract.review, "load_manifest", return_value=self.review_value
        ), self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "A1 formal receipt bytes differ"
        ):
            contract.load_manifest(
                output,
                expected_file_sha256=contract.file_sha256(output),
                verify_files=True,
            )

    def test_forbidden_evaluator_key_fails_even_when_resigned(self) -> None:
        value = self._materialize()
        value["score"] = None
        value.pop("manifest_digest")
        value["manifest_digest"] = contract.object_sha256(value)
        output = self.root / "bad.json"
        contract.write_create_only_json(output, value)
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "schema/role|forbidden"
        ):
            contract.load_manifest(
                output,
                expected_file_sha256=contract.file_sha256(output),
                verify_files=False,
            )

    def test_confirmation_source_overlap_with_a1_fails(self) -> None:
        self.formal_cells[0]["correct_source_sha256"] = self.rows[0][
            "source_video_sha256"
        ]
        self._write_authorization()
        with self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "source-disjoint"
        ):
            self._materialize()

    def test_resigned_self_owner_manifest_is_rejected(self) -> None:
        value = self._materialize()
        row = value["sentinels"][0]
        row["wrong_owner_sentinel_id"] = row["sentinel_id"]
        row["wrong_owner_iid"] = row["iid"]
        row["wrong_owner_source_video"] = row["source_video"]
        row["wrong_owner_source_video_sha256"] = row["source_video_sha256"]
        row["wrong_owner_latent_shape"] = row["latent_shape"]
        value.pop("manifest_digest")
        value["manifest_digest"] = contract.object_sha256(value)
        output = self.root / "self-owner.json"
        contract.write_create_only_json(output, value)
        with mock.patch.object(
            contract.review, "load_manifest", return_value=self.review_value
        ), self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError, "sentinel/wrong-owner"
        ):
            contract.load_manifest(
                output,
                expected_file_sha256=contract.file_sha256(output),
                verify_files=True,
            )

    def test_resigned_source_label_drift_is_rejected(self) -> None:
        value = self._materialize()
        value["sentinels"][1]["source_caption"] = "fabricated caption"
        value.pop("manifest_digest")
        value["manifest_digest"] = contract.object_sha256(value)
        output = self.root / "label-drift.json"
        contract.write_create_only_json(output, value)
        with mock.patch.object(
            contract.review, "load_manifest", return_value=self.review_value
        ), self.assertRaisesRegex(
            contract.SourceEdgeConfirmationError,
            "drifted from persistent review authority",
        ):
            contract.load_manifest(
                output,
                expected_file_sha256=contract.file_sha256(output),
                verify_files=True,
            )


if __name__ == "__main__":
    unittest.main()
