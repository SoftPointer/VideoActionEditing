#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_saic_t2v_event_bank_detached_review_v1 as review  # noqa: E402


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): review.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fake_diagnostic(
    *,
    source_video: str,
    expected_source_sha256: str,
    candidate_video: str,
    expected_candidate_sha256: str,
) -> dict[str, object]:
    decode = {
        "decoded_rgb24_sha256": "1" * 64,
        "frame_count": 81,
        "fps_numerator": 25,
        "fps_denominator": 1,
        "width": 64,
        "height": 64,
        "decoder_contract": "fixture-exact81",
    }
    body = {
        "schema_version": "bernini-saic-exact81-media-diagnostics-v1",
        "media": {
            "source": {
                "path": str(Path(source_video).resolve()),
                "sha256": expected_source_sha256,
                "bytes": Path(source_video).stat().st_size,
                "decode": decode,
            },
            "candidate": {
                "path": str(Path(candidate_video).resolve()),
                "sha256": expected_candidate_sha256,
                "bytes": Path(candidate_video).stat().st_size,
                "decode": decode,
            },
        },
        "runtime": {},
        "input_closure": {
            "source_video_read": True,
            "candidate_video_read": True,
            "decoded_exact81_whole_frames_read": True,
            "external_mask_read": False,
            "external_track_read": False,
            "external_pose_read": False,
            "external_flow_read": False,
            "external_trajectory_read": False,
            "internally_computed_optical_flow_diagnostic_only": True,
        },
        "availability": {
            "identity": "unavailable",
            "appearance": "unavailable",
            "background": "unavailable",
            "non_target": "unavailable",
            "event": "unavailable",
            "source_bind": "unavailable",
            "inverse": "unavailable",
            "camera": "diagnostic_only",
            "technical": "diagnostic_only",
            "temporal_consistency": "diagnostic_only",
        },
        "source": {"motion_summary": {"transition_count": 80}},
        "candidate": {"motion_summary": {"transition_count": 80}},
        "comparisons": {
            "camera_trajectory": {
                "cumulative_global_endpoint_l2_difference": 0.25,
                "interpretation": "diagnostic_only_no_absolute_camera_pass_threshold",
            },
            "scene_cut_ratio_absolute_difference": 0.0,
            "temporal_energy_cv_absolute_difference": 0.1,
            "technical": {
                "geometric_mean_technical_diagnostic": 0.75,
                "interpretation": "diagnostic_only_no_absolute_technical_pass_threshold",
            },
        },
        "authority": {
            "measurement_runtime_qualified": False,
            "candidate_selection_allowed": False,
            "training_allowed": False,
            "optimizer_step_allowed": False,
            "absolute_action_editing_success_claimed": False,
        },
        "remaining_gaps": ["fixture-no-semantic-observer"],
    }
    return {**body, "diagnostic_digest": review.object_sha256(body)}


class Fixture:
    def __init__(self, parent: Path) -> None:
        self.input_root = parent / "input"
        self.output_root = parent / "audit" / review.PACKET_ID
        self.output_root.parent.mkdir(parents=True)
        self.input_root.mkdir()
        self.master_path = self.input_root / review.MASTER_BASENAME
        self.source_manifest_path = self.input_root / review.SOURCE_MANIFEST_BASENAME
        self.event_spec_path = self.input_root / review.EVENT_SPEC_BASENAME
        _write(self.master_path, b"master-receipt-fixture\n")
        _write(self.source_manifest_path, b"source-manifest-fixture\n")
        _write(self.event_spec_path, b"event-spec-fixture\n")
        self.candidate_rows = self._rows()

    def _rows(self) -> list[dict[str, object]]:
        rows = []
        index = 0
        for source_index in range(8):
            iid = f"{source_index + 1:016x}"
            row_id = f"{'fit' if source_index < 4 else 'confirmation'}-fixture-{source_index:02d}-{iid}"
            source = self.input_root / "sources" / f"{iid}.mp4"
            _write(source, f"source-{iid}".encode("ascii"))
            source_sha = review.file_sha256(source)
            seed_count = 2 if source_index in (0, 1, 4, 5) else 3
            for seed_offset in range(seed_count):
                seed = 2026082101 + source_index * 10 + seed_offset
                for branch in review.BRANCH_ORDER:
                    index += 1
                    candidate_id = f"saic-{iid}-{branch}-s{seed}"
                    candidate = self.input_root / "attempts" / candidate_id / "t2v.mp4"
                    attempt = (
                        self.input_root
                        / "attempts"
                        / candidate_id
                        / review.ATTEMPT_RECEIPT_BASENAME
                    )
                    _write(candidate, f"candidate-{candidate_id}".encode("ascii"))
                    _write(attempt, f"attempt-{candidate_id}".encode("ascii"))
                    rows.append(
                        {
                            "registered_candidate_index": index,
                            "candidate_id": candidate_id,
                            "row_id": row_id,
                            "iid": iid,
                            "analysis_split": "fit" if source_index < 4 else "confirmation",
                            "actor_family": "dog" if source_index % 2 == 0 else "human",
                            "action_family_id": "fixture-reversible-v1",
                            "branch": branch,
                            "seed": seed,
                            "initial_state_type": "fixture_i0",
                            "terminal_state_type": "fixture_i1",
                            "branch_start_state_caption": "Fixture start state.",
                            "branch_instruction": f"Fixture {branch} instruction.",
                            "full_t2v_caption": f"Fixture full {branch} prompt.",
                            "source_input_path": str(source),
                            "source_sha256": source_sha,
                            "candidate_input_path": str(candidate),
                            "candidate_sha256": review.file_sha256(candidate),
                            "attempt_receipt_input_path": str(attempt),
                            "attempt_receipt_sha256": review.file_sha256(attempt),
                            "attempt_receipt_digest": review.object_sha256(
                                {"candidate_id": candidate_id}
                            ),
                            "semantic_status": "UNASSESSED",
                            "event_verified": False,
                            "identity_preservation_verified": False,
                        }
                    )
        if len(rows) != 60:
            raise AssertionError(len(rows))
        return rows

    def validated(self) -> dict[str, object]:
        return {
            "input_root": self.input_root.resolve(),
            "master": {},
            "master_digest": "a" * 64,
            "master_path": self.master_path.resolve(),
            "source_manifest": {},
            "source_manifest_path": self.source_manifest_path.resolve(),
            "source_manifest_summary": {},
            "event_spec": {},
            "event_spec_path": self.event_spec_path.resolve(),
            "event_spec_raw_sha256": "b" * 64,
            "candidate_rows": self.candidate_rows,
        }


class DetachedReviewBuilderTests(unittest.TestCase):
    def test_builds_fresh_complete_zero_authority_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            before = _tree_hashes(fixture.input_root)
            with mock.patch.object(
                review, "_load_and_validate_inputs", return_value=fixture.validated()
            ), mock.patch.object(
                review.diagnostics, "build_diagnostic", side_effect=_fake_diagnostic
            ):
                receipt = review.build_review(
                    input_root=fixture.input_root,
                    output_root=fixture.output_root,
                    job_id="999001",
                    workers=1,
                )
            self.assertEqual(_tree_hashes(fixture.input_root), before)
            self.assertEqual(receipt["candidate_count"], 60)
            self.assertEqual(receipt["machine_diagnostic_count"], 60)
            self.assertTrue(receipt["exact81_machine_diagnostics_complete"])
            self.assertTrue(receipt["machine_diagnostics_zero_authority"])
            self.assertFalse(receipt["detached_full81_event_review_complete"])
            self.assertFalse(receipt["event_verified"])
            self.assertFalse(receipt["identity_preservation_verified"])
            self.assertFalse(receipt["seed_selection_authorized"])
            self.assertFalse(receipt["training_target_authorized"])
            self.assertFalse(receipt["optimizer_step_authorized"])
            self.assertEqual(receipt["semantic_status"], "UNASSESSED")

            manifest = json.loads(
                (fixture.output_root / "review-manifest.json").read_text()
            )
            self.assertEqual(manifest["candidate_count"], 60)
            self.assertEqual(manifest["source_count"], 8)
            self.assertEqual(len(manifest["items"]), 60)
            self.assertEqual(
                {item["semantic_status"] for item in manifest["items"]},
                {"UNASSESSED"},
            )
            self.assertEqual(
                {item["diagnostic_summary"]["authority"] for item in manifest["items"]},
                {"diagnostic_only"},
            )
            self.assertEqual(
                len(list((fixture.output_root / "diagnostics").glob("*.json"))), 60
            )

            page = (fixture.output_root / "index.html").read_text()
            self.assertIn("Diagnostics are not semantic evidence", page)
            self.assertIn("no seed is ranked or selected", page)
            self.assertEqual(page.count("<video "), 68)
            self.assertEqual(page.count("semantic: <strong>UNASSESSED</strong>"), 60)
            self.assertNotIn(str(fixture.input_root), page)
            self.assertIn("Do not open this page before both independent", page)

            blind_page = (fixture.output_root / "blind-review.html").read_text()
            self.assertIn("Do not open index.html or diagnostics/*.json yet", blind_page)
            self.assertIn("machine measurements are absent", blind_page)
            self.assertNotIn("Technical geometric mean", blind_page)
            self.assertNotIn("SEED ", blind_page)

            protocol = json.loads(
                (fixture.output_root / "observer-protocol.json").read_text()
            )
            self.assertEqual(
                protocol["aggregation_rule"]["observer_disagreement_result"],
                "UNASSESSED",
            )
            self.assertEqual(
                protocol["observer_contract"]["minimum_independent_observers"], 2
            )
            self.assertTrue(
                protocol["machine_diagnostic_contract"][
                    "human_labels_must_precede_machine_diagnostic_access"
                ]
            )
            self.assertFalse(
                protocol["aggregation_rule"][
                    "event_verified_may_be_set_by_this_packet"
                ]
            )

            for slot in (1, 2):
                template = json.loads(
                    (
                        fixture.output_root
                        / "observer-templates"
                        / f"observer-{slot}-blank.json"
                    ).read_text()
                )
                self.assertTrue(template["template_only"])
                self.assertIsNone(template["observer_id"])
                self.assertEqual(len(template["responses"]), 60)
                self.assertTrue(
                    all(
                        response["event_reaches_required_terminal_state"] is None
                        for response in template["responses"]
                    )
                )
                self.assertFalse(template["authority"]["event_verified"])
                self.assertEqual(
                    template["observer_protocol_artifact"]["protocol_digest"],
                    protocol["protocol_digest"],
                )

            validated = review.validate_packet(fixture.output_root)
            self.assertEqual(validated["receipt_digest"], receipt["receipt_digest"])

    def test_refuses_existing_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.output_root.mkdir()
            with self.assertRaisesRegex(review.DetachedReviewError, "fresh"):
                review.build_review(
                    input_root=fixture.input_root,
                    output_root=fixture.output_root,
                    job_id="999002",
                    workers=1,
                )

    def test_observer_slots_share_items_but_cannot_claim_independence(self) -> None:
        items = [
            {
                "review_item_id": "review-0001",
                "candidate_media_sha256": "1" * 64,
                "source_media_sha256": "2" * 64,
                "branch": "forward",
                "seed": 7,
            }
        ]
        protocol = review._observer_protocol(review_items=items)
        protocol_binding = {
            "portable_path": "observer-protocol.json",
            "file_sha256": "3" * 64,
            "protocol_digest": protocol["protocol_digest"],
            "review_item_set_digest": protocol["review_item_set_digest"],
        }
        left = review._observer_template(
            slot=1, review_items=items, protocol_binding=protocol_binding
        )
        right = review._observer_template(
            slot=2, review_items=items, protocol_binding=protocol_binding
        )
        self.assertEqual(left["review_item_set_digest"], right["review_item_set_digest"])
        self.assertNotEqual(left["template_digest"], right["template_digest"])
        self.assertFalse(left["blindness_or_independence_established_by_template"])
        self.assertTrue(left["same_person_must_not_fill_both_slots"])
        self.assertEqual(left["observer_protocol_artifact"], protocol_binding)


if __name__ == "__main__":
    unittest.main()
