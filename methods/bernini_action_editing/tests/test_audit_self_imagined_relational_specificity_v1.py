from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import audit_self_imagined_relational_specificity_v1 as audit


class SIRMSpecificityGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(audit.__file__).read_text(encoding="utf-8")

    def test_roles_and_threshold_are_fixed_and_not_cli_exposed(self) -> None:
        self.assertEqual(
            audit.CORE_CONTROL_ROLES,
            (
                "same_video_reverse",
                "same_video_phase_shuffle",
                "semantic_generic_wrong_motion",
            ),
        )
        self.assertTrue(
            math.isclose(
                audit.MINIMUM_CORE_CONTROL_MISMATCH,
                0.05,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        destinations = {action.dest for action in audit.build_parser()._actions}
        self.assertNotIn("threshold", destinations)
        self.assertNotIn("minimum_mismatch", destinations)
        self.assertNotIn("seed", destinations)

    def test_all_core_controls_must_clear_the_fixed_floor(self) -> None:
        passing = {role: 0.05 for role in audit.CORE_CONTROL_ROLES}
        self.assertTrue(audit.core_specificity_passes(passing))
        for role in audit.CORE_CONTROL_ROLES:
            row = dict(passing)
            row[role] = 0.049999
            self.assertFalse(audit.core_specificity_passes(row))
        self.assertFalse(audit.core_specificity_passes({}))
        nonfinite = dict(passing)
        nonfinite[audit.CORE_CONTROL_ROLES[0]] = float("nan")
        self.assertFalse(audit.core_specificity_passes(nonfinite))

    def test_controls_are_cross_bound_to_one_frozen_query_cell(self) -> None:
        for token in (
            '"group_id"',
            '"episode_id"',
            '"action_family_id"',
            '"same_state_query_binding"',
            '"spatial_sketch_binding"',
            '"hidden_binding"',
            "materializer.validate_arm_receipt(checked, verify_artifact=True)",
            'control.get("prompt_binding") != positive.get("prompt_binding")',
            'control.get("model_binding") != model_binding',
            '"all_controls_same_episode_prompt_model_sketch_geometry_and_query_contract": True',
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
