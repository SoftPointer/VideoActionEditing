from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_exact81_video_controls_v1 as controls  # noqa: E402


class Exact81VideoControlIndexTest(unittest.TestCase):
    def test_noop_reverse_and_incomplete_maps_are_exact(self):
        noop = controls.control_frame_indices("zero_or_noop", seed=7)
        reverse = controls.control_frame_indices("reverse", seed=7)
        incomplete = controls.control_frame_indices("incomplete", seed=7)
        self.assertEqual(noop, (0,) * 81)
        self.assertEqual(reverse, tuple(range(80, -1, -1)))
        self.assertEqual(incomplete[:41], tuple(range(41)))
        self.assertEqual(incomplete[41:], (40,) * 40)

    def test_shuffle_keeps_frame_zero_and_exact_frame_multiset(self):
        shuffled = controls.control_frame_indices("temporal_shuffle", seed=20260824)
        self.assertEqual(len(shuffled), 81)
        self.assertEqual(shuffled[0], 0)
        self.assertEqual(sorted(shuffled), list(range(81)))
        self.assertNotEqual(shuffled, tuple(range(81)))
        self.assertNotEqual(shuffled, tuple(range(80, -1, -1)))

    def test_shuffle_is_deterministic_and_seeded(self):
        first = controls.control_frame_indices("temporal_shuffle", seed=1)
        self.assertEqual(first, controls.control_frame_indices("temporal_shuffle", seed=1))
        self.assertNotEqual(first, controls.control_frame_indices("temporal_shuffle", seed=2))

    def test_unknown_role_fails_closed(self):
        with self.assertRaises(controls.Exact81VideoControlError):
            controls.control_frame_indices("forward", seed=1)


if __name__ == "__main__":
    unittest.main()

