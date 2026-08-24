from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import mdr_exact_motion_analogy as mdr
else:  # pragma: no cover - dependency-light local environment
    mdr = None


def _video(offset: float = 0.0):
    timeline = torch.linspace(-0.8, 0.8, mdr.FRAME_COUNT, dtype=torch.float32)
    value = timeline.view(1, 1, mdr.FRAME_COUNT, 1, 1).expand(1, 3, -1, 2, 2)
    return (value + offset).clamp(-1.0, 1.0).contiguous()


SOURCE_ID = "1" * 64
DONOR_ID = "2" * 64


@unittest.skipIf(torch is None, "torch is unavailable")
class TemporalProgramTests(unittest.TestCase):
    def test_registered_grid_is_exact_81_frame_and_row_stochastic(self) -> None:
        programs = mdr.registered_program_grid()
        self.assertEqual(tuple(item.kind for item in programs), mdr.PROGRAM_KINDS)
        for program in programs:
            coordinates = program.output_to_input
            matrix = program.interpolation_matrix
            self.assertEqual(tuple(coordinates.shape), (81,))
            self.assertEqual(tuple(matrix.shape), (81, 81))
            self.assertEqual(coordinates.dtype, torch.float64)
            self.assertEqual(matrix.dtype, torch.float64)
            self.assertTrue(torch.equal(matrix.sum(1), torch.ones(81, dtype=torch.float64)))
            self.assertTrue(bool((matrix >= 0).all()))
            self.assertTrue(bool((matrix <= 1).all()))

    def test_identity_and_reverse_are_exact(self) -> None:
        source = _video()
        identity = mdr.apply_temporal_program(source, mdr.TemporalProgram("identity"))
        reverse = mdr.apply_temporal_program(source, mdr.TemporalProgram("reverse"))
        self.assertTrue(torch.equal(identity, source))
        self.assertTrue(torch.equal(reverse, source.flip(2)))

    def test_pause_has_exact_hold_and_endpoint(self) -> None:
        source = _video()
        program = mdr.TemporalProgram("pause_then_catch_up", 0.25)
        transformed = mdr.apply_temporal_program(source, program)
        self.assertTrue(torch.equal(transformed[:, :, :21], source[:, :, :1].expand(-1, -1, 21, -1, -1)))
        self.assertTrue(torch.equal(transformed[:, :, -1], source[:, :, -1]))

    def test_invalid_programs_fail_closed(self) -> None:
        for program in (
            ("unknown", 0.0),
            ("identity", 1.0),
            ("speed_up", 1.0),
            ("slow_down", 1.0),
            ("pause_then_catch_up", 0.9),
            ("cyclic_phase", 1.5),
        ):
            with self.subTest(program=program):
                with self.assertRaises(mdr.MDRMotionAnalogyError):
                    mdr.TemporalProgram(*program)


@unittest.skipIf(torch is None, "torch is unavailable")
class CrossIdentityAnalogyTests(unittest.TestCase):
    def test_builds_relative_packet_and_internal_target(self) -> None:
        source = _video()
        donor = _video(0.1)
        program = mdr.TemporalProgram("reverse")
        example = mdr.build_motion_analogy_example(
            source,
            donor,
            program,
            source_identity_sha256=SOURCE_ID,
            donor_identity_sha256=DONOR_ID,
        )
        self.assertTrue(torch.equal(example.source_identity_video, source))
        self.assertTrue(torch.equal(example.motion_donor_before_video, donor))
        self.assertTrue(torch.equal(example.motion_donor_after_video, donor.flip(2)))
        self.assertTrue(torch.equal(example.regression_target_video, source.flip(2)))
        self.assertEqual(
            example.receipt["construction"],
            "source=A,donor_packet=(B,T(B)),target=T(A)",
        )
        self.assertTrue(example.receipt["relative_donor_program_observable"])
        self.assertFalse(example.receipt["external_target_accepted"])
        self.assertFalse(example.receipt["single_after_only_donor_is_main_training_input"])

    def test_target_is_independent_of_donor_pixels(self) -> None:
        source = _video()
        program = mdr.TemporalProgram("speed_up", 0.5)
        first = mdr.build_motion_analogy_example(
            source,
            _video(0.05),
            program,
            source_identity_sha256=SOURCE_ID,
            donor_identity_sha256=DONOR_ID,
        )
        second = mdr.build_motion_analogy_example(
            source,
            _video(-0.15),
            program,
            source_identity_sha256=SOURCE_ID,
            donor_identity_sha256="3" * 64,
        )
        self.assertTrue(torch.equal(first.regression_target_video, second.regression_target_video))
        self.assertFalse(torch.equal(first.motion_donor_after_video, second.motion_donor_after_video))

    def test_same_generic_text_but_different_program_has_exact_own_target(self) -> None:
        source = _video()
        donor = _video(0.1)
        reverse = mdr.build_motion_analogy_example(
            source,
            donor,
            mdr.TemporalProgram("reverse"),
            source_identity_sha256=SOURCE_ID,
            donor_identity_sha256=DONOR_ID,
        )
        phase = mdr.build_motion_analogy_example(
            source,
            donor,
            mdr.TemporalProgram("cyclic_phase", 20.0),
            source_identity_sha256=SOURCE_ID,
            donor_identity_sha256=DONOR_ID,
        )
        self.assertEqual(reverse.instruction, phase.instruction)
        self.assertFalse(torch.equal(reverse.regression_target_video, phase.regression_target_video))
        self.assertNotEqual(reverse.program.digest, phase.program.digest)

    def test_same_identity_and_external_target_slots_are_forbidden(self) -> None:
        with self.assertRaisesRegex(mdr.MDRMotionAnalogyError, "A != donor identity B"):
            mdr.build_motion_analogy_example(
                _video(),
                _video(0.1),
                mdr.TemporalProgram("identity"),
                source_identity_sha256=SOURCE_ID,
                donor_identity_sha256=SOURCE_ID,
            )
        parameters = set(inspect.signature(mdr.build_motion_analogy_example).parameters)
        self.assertFalse(parameters & mdr.FORBIDDEN_EXTERNAL_INPUT_NAMES)

    def test_contract_limits_claim_to_temporal_program_rebinding(self) -> None:
        contract = mdr.motion_analogy_contract()
        self.assertEqual(
            contract["equation"],
            "source=A; donor_packet=(B,T(B)); target=T(A); A!=B",
        )
        self.assertFalse(contract["natural_semantic_action_learned_by_this_pretext"])
        self.assertFalse(contract["single_after_only_donor_authorized_as_main"])
        self.assertIn("relative_packet_vs_after_only_ablation", contract["identification_controls"])


if __name__ == "__main__":
    unittest.main()
