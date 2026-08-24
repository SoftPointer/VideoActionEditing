from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import postvideo_quality_router_v4_1 as router  # noqa: E402


def quality(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "action_implemented": "yes",
        "identity_preserved": "yes",
        "species_preserved": "yes",
        "clothing_preserved": "yes",
        "non_edited_content_preserved": "yes",
        "camera_preserved": "yes",
        "blur_level": "low",
        "flicker_level": "low",
        "artifact_level": "low",
        "confidence": "high",
        "uncertainty_codes": [],
        "evidence": {
            "action": [{"frames": ["T0", "T3"], "observation": "ordered motion"}],
            "identity": [{"frames": ["S0", "T0"], "observation": "same subject"}],
            "preservation": [
                {"frames": ["S0", "T0"], "observation": "same scene"}
            ],
            "technical": [
                {"frames": ["T0", "T3"], "observation": "stable target"}
            ],
        },
    }
    value.update(changes)
    return value


def qwen_record(iid: str, value: dict[str, object] | None = None) -> dict[str, object]:
    value = value or quality()
    row: dict[str, object] = {
        "iid": iid,
        "audit_outcome": "success",
        "quality": value,
        "quality_sha256": router.object_sha(value),
        "record_digest": "a" * 64,
        "input": {
            "source_video": {"sha256": "b" * 64},
            "target_video": {"sha256": "c" * 64},
        },
    }
    return row


def gate_row(iid: str, status: str) -> dict[str, object]:
    return {
        "iid": iid,
        "report_path": "/evidence/gate.json",
        "report_sha256": "d" * 64,
        "qwen_record_digest": "a" * 64,
        "gate": {
            "status": status,
            "hard": status in {"fail", "error"},
            "unresolved": status == "unresolved",
            "failure_codes": ["quality_noise"] if status == "fail" else [],
            "unresolved_codes": (
                ["quality_routeoff_structure_requires_external_verifier"]
                if status == "unresolved"
                else []
            ),
            "families": {},
            "media": {
                "source": {"sha256": "b" * 64},
                "candidate": {"sha256": "c" * 64},
                "frozen_base": {"sha256": "e" * 64},
            },
        },
    }


class RouterV41PolicyTest(unittest.TestCase):
    iid = "s00000359-case04"

    def test_all_four_axes_pass_is_only_training_eligible_route(self) -> None:
        row = router.combine_one(gate_row(self.iid, "pass"), qwen_record(self.iid))
        self.assertEqual(row["decision"]["route"], "PROMOTE")
        self.assertTrue(row["decision"]["training_eligible"])
        self.assertEqual(
            row["decision"]["execution_action"], "ACCEPT_FOR_TRAINING"
        )
        self.assertFalse(row["decision"]["manual_review_required"])
        self.assertFalse(row["decision"]["human_review_dependency"])
        self.assertEqual({axis["status"] for axis in row["axes"].values()}, {"PASS"})

    def test_hard_artifact_is_non_compensating_against_positive_qwen(self) -> None:
        row = router.combine_one(gate_row(self.iid, "fail"), qwen_record(self.iid))
        self.assertEqual(row["decision"]["route"], "REJECT")
        self.assertEqual(row["decision"]["execution_action"], "AUTO_RETRY")
        self.assertEqual(row["axes"]["artifact_quality"]["status"], "FAIL")
        self.assertFalse(
            row["axes"]["artifact_quality"]["qwen_can_override_hard_artifact"]
        )
        self.assertFalse(row["decision"]["training_eligible"])

    def test_artifact_unresolved_never_auto_promotes(self) -> None:
        row = router.combine_one(
            gate_row(self.iid, "unresolved"), qwen_record(self.iid)
        )
        self.assertEqual(row["decision"]["route"], "REVIEW")
        self.assertEqual(row["decision"]["overall_status"], "UNRESOLVED")
        self.assertEqual(row["decision"]["execution_action"], "AUTO_ADJUDICATE")
        self.assertFalse(row["decision"]["manual_review_required"])
        self.assertFalse(row["decision"]["human_review_dependency"])
        self.assertFalse(row["decision"]["training_eligible"])

    def test_semantic_or_action_failure_rejects_gate_pass(self) -> None:
        row = router.combine_one(
            gate_row(self.iid, "pass"),
            qwen_record(
                self.iid,
                quality(
                    action_implemented="no",
                    identity_preserved="no",
                ),
            ),
        )
        self.assertEqual(row["decision"]["route"], "REJECT")
        self.assertEqual(row["axes"]["action_alignment"]["status"], "FAIL")
        self.assertEqual(
            row["axes"]["identity_content_preservation"]["status"], "FAIL"
        )

    def test_qwen_technical_label_does_not_replace_v31_artifact_authority(self) -> None:
        row = router.combine_one(
            gate_row(self.iid, "pass"),
            qwen_record(
                self.iid,
                quality(blur_level="high", artifact_level="high"),
            ),
        )
        self.assertEqual(row["axes"]["artifact_quality"]["status"], "PASS")
        self.assertEqual(
            row["axes"]["artifact_quality"]["qwen_artifact_level_non_authoritative"],
            "high",
        )

    def test_v31_replay_family_payload_recomputes(self) -> None:
        replay = json.loads(
            (
                REPOSITORY_ROOT
                / "artifacts/checkpoint_visual_quality_gate_v3_1_20260824/evidence/replay-suite.json"
            ).read_text(encoding="utf-8")
        )
        families = replay["counterexample_replay"]["rows"][0]["gate"][
            "evidence_families"
        ]
        validated = router._validate_families(families, iid="replay-case00")
        self.assertEqual(set(validated), set(router._FAMILIES))
        self.assertFalse(any(row["triggered"] for row in validated.values()))
        self.assertTrue(any(row["unresolved"] for row in validated.values()))

    def test_formal_schema_and_fixed_policy_are_v41(self) -> None:
        schema = json.loads(router.JSON_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$defs"]["record"]["properties"]["schema_version"]["const"],
            router.RECORD_SCHEMA,
        )
        self.assertFalse(router.FIXED_POLICY["qwen_can_override_hard_artifact"])
        self.assertTrue(router.FIXED_POLICY["all_four_axes_must_pass_for_training"])
        self.assertFalse(router.FIXED_POLICY["human_review_dependency"])


if __name__ == "__main__":
    unittest.main()
