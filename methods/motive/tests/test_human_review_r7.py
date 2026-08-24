from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.human_review import (
    OPTIONAL_REVIEW_TEXT_FIELDS,
    REVIEW_ITEM_DIGEST_FIELDS,
    R7_ASSIGNMENT_FIELD,
    R7_ASSIGNMENT_SCHEMA,
    R7_MEDIA_FIELD,
    R7_RATE_AUDIT_REVIEW_SCHEMA,
    R7_REVIEW_ITEM_DIGEST_FIELDS,
    _atomic_jsonl,
    _review_item_digest,
    merge,
    prepare,
)
from motive.r7_human_audit_policy import policy_sha256
from motive.r7_human_audit_sample import build_media_binding
from motive.train_action_repr import (
    HUMAN_REVIEW_SCHEMA,
    _human_review_verdict,
)


PRIMARY_REVIEWER_ID = "reviewer-primary"
SECONDARY_REVIEWER_ID = "reviewer-secondary"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _prepare(manifest: Path, output: Path, *, overwrite: bool = False) -> None:
    result = prepare(
        argparse.Namespace(
            input=manifest,
            output=output,
            include_automation_hints=False,
            overwrite=overwrite,
        )
    )
    if result != 0:
        raise AssertionError(f"human_review.prepare returned {result}")


def _merge(
    manifest: Path,
    labels: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    result = merge(
        argparse.Namespace(
            manifest=manifest,
            labels=labels,
            output=output,
            overwrite=overwrite,
        )
    )
    if result != 0:
        raise AssertionError(f"human_review.merge returned {result}")


def _legacy_row() -> dict[str, object]:
    return {
        "iid": "legacy-001",
        "input_digest": _sha("legacy input"),
        "prompt": "Make the subject wave.",
        "src_video": "legacy/source.mp4",
        "tgt_video": "legacy/edited.mp4",
        "final_triage": {"decision": "review"},
        "qwen_evidence": {"visual": {"status": "not_run"}},
    }


def _r7_row(
    root: Path,
    *,
    slot: str,
    reviewer_id: str,
) -> dict[str, object]:
    iid = "r7-001"
    data_root = root / "media"
    src_path = data_root / "clips" / iid / "source.mp4"
    tgt_path = data_root / "clips" / iid / "edited.mp4"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_bytes(b"immutable source bytes")
    tgt_path.write_bytes(b"immutable target bytes")
    row: dict[str, object] = {
        "iid": iid,
        "input_digest": _sha("r7 input"),
        "prompt": "Make the subject jump.",
        "src_video": f"clips/{iid}/source.mp4",
        "tgt_video": f"clips/{iid}/edited.mp4",
    }
    row[R7_MEDIA_FIELD] = build_media_binding(
        row,
        data_root=data_root,
        diagnostic_unbound_media=False,
    )
    row[R7_ASSIGNMENT_FIELD] = {
        "schema_version": R7_ASSIGNMENT_SCHEMA,
        "review_instance_id": _sha(f"{iid}:{slot}:{reviewer_id}"),
        "iid": iid,
        "annotator_slot": slot,
        "assigned_reviewer_id": reviewer_id,
        "independent_review_required": slot == "secondary",
        "assignment_set_digest": _sha("shared assignment set"),
        "policy_sha256": policy_sha256(),
    }
    return row


def _complete_label(
    label: dict[str, object],
    *,
    reviewer: str,
) -> dict[str, object]:
    completed = copy.deepcopy(label)
    completed["verdict"] = "valid_action"
    completed["reviewer"] = reviewer
    completed["action_signature"] = "jump"
    completed["event_start_frame"] = 2
    completed["event_end_frame"] = 11
    return completed


class HumanReviewLegacyAndR7Tests(unittest.TestCase):
    def test_legacy_prepare_and_merge_keep_the_v1_digest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "legacy.jsonl"
            labels = root / "legacy-labels.jsonl"
            merged_path = root / "legacy-merged.jsonl"
            row = _legacy_row()
            _write_jsonl(manifest, [row])

            _prepare(manifest, labels)
            prepared = _jsonl_rows(labels)
            self.assertEqual(len(prepared), 1)
            template = prepared[0]
            self.assertNotIn(R7_ASSIGNMENT_FIELD, template)
            self.assertNotIn(R7_MEDIA_FIELD, template)
            expected_payload = {
                "schema_version": HUMAN_REVIEW_SCHEMA,
                "iid": row["iid"],
                "input_digest": row["input_digest"],
                "prompt": row["prompt"],
                "src_video": row["src_video"],
                "tgt_video": row["tgt_video"],
            }
            expected_digest = hashlib.sha256(
                json.dumps(
                    expected_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(template["review_item_digest"], expected_digest)

            prepare_summary = json.loads(
                labels.with_suffix(".jsonl.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(prepare_summary["r7_contract_bound"])
            self.assertFalse(prepare_summary["media_bytes_bound"])
            self.assertEqual(
                prepare_summary["review_item_digest_fields"],
                list(REVIEW_ITEM_DIGEST_FIELDS),
            )

            completed = _complete_label(
                template,
                reviewer="legacy-reviewer",
            )
            _write_jsonl(labels, [completed])
            _merge(manifest, labels, merged_path)
            merged = _jsonl_rows(merged_path)
            self.assertEqual(len(merged), 1)
            review = merged[0]["human_review"]
            self.assertEqual(review["review_item_digest"], expected_digest)
            self.assertEqual(review["reviewer"], "legacy-reviewer")
            self.assertEqual(review["event_start_frame"], 2)
            self.assertEqual(review["event_end_frame"], 11)
            for r7_only_field in (
                "review_instance_id",
                "annotator_slot",
                "assigned_reviewer_id",
                "assignment_set_digest",
                "policy_sha256",
                "media_binding_sha256",
            ):
                self.assertNotIn(r7_only_field, review)
            self.assertTrue(
                set(OPTIONAL_REVIEW_TEXT_FIELDS).issubset(review)
            )

            merge_summary = json.loads(
                merged_path.with_suffix(".jsonl.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(merge_summary["r7_contract_bound"])
            self.assertEqual(
                merge_summary["review_item_digest_fields"],
                list(REVIEW_ITEM_DIGEST_FIELDS),
            )

    def test_r7_happy_path_freezes_slot_reviewer_and_media_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "primary.jsonl"
            labels = root / "primary-labels.jsonl"
            merged_path = root / "primary-merged.jsonl"
            row = _r7_row(
                root,
                slot="primary",
                reviewer_id=PRIMARY_REVIEWER_ID,
            )
            _write_jsonl(manifest, [row])
            _prepare(manifest, labels)
            template = _jsonl_rows(labels)[0]
            self.assertEqual(
                template[R7_ASSIGNMENT_FIELD],
                row[R7_ASSIGNMENT_FIELD],
            )
            self.assertEqual(template[R7_MEDIA_FIELD], row[R7_MEDIA_FIELD])

            summary = json.loads(
                labels.with_suffix(".jsonl.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["r7_contract_bound"])
            self.assertTrue(summary["media_bytes_bound"])
            self.assertEqual(
                summary["review_item_digest_fields"],
                list(R7_REVIEW_ITEM_DIGEST_FIELDS),
            )
            self.assertEqual(summary["annotator_slots"], ["primary"])

            completed_rate_label = _complete_label(
                template,
                reviewer=PRIMARY_REVIEWER_ID,
            )
            completed_rate_label["action_signature"] = ""
            completed_rate_label["actor"] = ""
            _write_jsonl(labels, [completed_rate_label])
            _merge(manifest, labels, merged_path)
            merged_row = _jsonl_rows(merged_path)[0]
            review = merged_row["human_review"]
            assignment = row[R7_ASSIGNMENT_FIELD]
            self.assertEqual(
                review["schema_version"],
                R7_RATE_AUDIT_REVIEW_SCHEMA,
            )
            self.assertEqual(review["action_signature"], "")
            self.assertEqual(review["actor"], "")
            self.assertEqual(review["annotator_slot"], "primary")
            self.assertEqual(
                review["assigned_reviewer_id"],
                PRIMARY_REVIEWER_ID,
            )
            self.assertEqual(
                review["review_instance_id"],
                assignment["review_instance_id"],
            )
            self.assertEqual(
                review["assignment_set_digest"],
                assignment["assignment_set_digest"],
            )
            self.assertEqual(
                review["policy_sha256"],
                assignment["policy_sha256"],
            )
            self.assertRegex(
                review["media_binding_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(review["label_scope"], "rate_audit_only")
            self.assertFalse(
                review["direct_training_supervision_allowed"]
            )
            self.assertFalse(review["training_authorized"])
            with self.assertRaisesRegex(
                ValueError,
                "unsupported human_review schema",
            ):
                _human_review_verdict(
                    merged_row,
                    context="R7 rate-audit row",
                )

    def test_primary_label_copy_and_reviewer_alias_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_manifest = root / "primary.jsonl"
            secondary_manifest = root / "secondary.jsonl"
            primary_labels = root / "primary-labels.jsonl"
            secondary_labels = root / "secondary-labels.jsonl"
            primary_row = _r7_row(
                root,
                slot="primary",
                reviewer_id=PRIMARY_REVIEWER_ID,
            )
            secondary_row = _r7_row(
                root,
                slot="secondary",
                reviewer_id=SECONDARY_REVIEWER_ID,
            )
            _write_jsonl(primary_manifest, [primary_row])
            _write_jsonl(secondary_manifest, [secondary_row])
            _prepare(primary_manifest, primary_labels)
            _prepare(secondary_manifest, secondary_labels)

            copied_primary = _complete_label(
                _jsonl_rows(primary_labels)[0],
                reviewer=PRIMARY_REVIEWER_ID,
            )
            copied_path = root / "copied-primary-as-secondary.jsonl"
            _write_jsonl(copied_path, [copied_primary])
            with self.assertRaisesRegex(
                ValueError,
                "review_item_digest mismatch",
            ):
                _merge(
                    secondary_manifest,
                    copied_path,
                    root / "copied-merge.jsonl",
                )

            aliased = _complete_label(
                _jsonl_rows(primary_labels)[0],
                reviewer="Reviewer-Primary",
            )
            alias_path = root / "reviewer-alias.jsonl"
            _write_jsonl(alias_path, [aliased])
            with self.assertRaisesRegex(
                ValueError,
                "reviewer must exactly equal assigned R7 reviewer ID",
            ):
                _merge(
                    primary_manifest,
                    alias_path,
                    root / "alias-merge.jsonl",
                )

    def test_assignment_mutation_fails_even_if_label_digest_is_rebound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "primary.jsonl"
            labels = root / "primary-labels.jsonl"
            row = _r7_row(
                root,
                slot="primary",
                reviewer_id=PRIMARY_REVIEWER_ID,
            )
            _write_jsonl(manifest, [row])
            _prepare(manifest, labels)
            mutated = _complete_label(
                _jsonl_rows(labels)[0],
                reviewer=PRIMARY_REVIEWER_ID,
            )
            mutated[R7_ASSIGNMENT_FIELD]["assignment_set_digest"] = "f" * 64
            mutated["review_item_digest"] = _review_item_digest(
                mutated,
                context="adversarial rebound label",
            )
            mutated_path = root / "assignment-mutated.jsonl"
            _write_jsonl(mutated_path, [mutated])

            with self.assertRaisesRegex(
                ValueError,
                "review_item_digest mismatch",
            ):
                _merge(
                    manifest,
                    mutated_path,
                    root / "mutated-merge.jsonl",
                )

    def test_atomic_writer_preserves_existing_commit_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "committed.jsonl"
            committed = b'{"committed":true}\n'
            output.write_bytes(committed)

            with self.assertRaises(FileExistsError):
                _atomic_jsonl(
                    output,
                    [{"replacement": True}],
                    overwrite=False,
                )
            self.assertEqual(output.read_bytes(), committed)

            with self.assertRaises(TypeError):
                _atomic_jsonl(
                    output,
                    [
                        {"serializable": True},
                        {"not_serializable": {1, 2, 3}},
                    ],
                    overwrite=True,
                )
            self.assertEqual(output.read_bytes(), committed)
            self.assertEqual(
                list(root.glob(f".{output.name}.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
