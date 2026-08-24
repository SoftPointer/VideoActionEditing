from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_v15a_r3_dynamic_static_routeprobe_review_v1 as builder  # noqa: E402


def _card(index: int) -> dict:
    return {
        "label": f"card {index}",
        "detail": f"detail {index}",
        "artifact": {"path": f"media/card-{index}.mp4"},
    }


def _receipt() -> dict:
    return {
        "title": "strict layout",
        "benchmark_warning": "hard role switch",
        "strict_result_warning": "Q/K-only is not appearance-invariant.",
        "object_legend": [
            {"id": 1, "description": "upper-left white ceramic pouring vessel"},
            {"id": 2, "description": "transparent handled glass vessel"},
            {"id": 3, "description": "lower-left small white teacup"},
        ],
        "rows": [
            {"title": "row one", "note": "authority", "cards": [_card(i) for i in range(3)]},
            {"title": "row two", "note": "triplet", "cards": [_card(i) for i in range(3, 6)]},
        ],
    }


class V15AR3ReviewTests(unittest.TestCase):
    def test_render_is_exact_two_by_three_equal_height_without_forms(self) -> None:
        page = builder.render(_receipt())
        self.assertEqual(page.count('<section class="comparison-row"'), 2)
        self.assertEqual(page.count('<div class="grid">'), 2)
        self.assertEqual(page.count("<video "), 6)
        self.assertIn("grid-template-columns:repeat(3,minmax(0,1fr))", page)
        self.assertIn(".video-shell{width:100%;aspect-ratio:2/3", page)
        self.assertIn("同步播放本事件（6条）", page)
        self.assertEqual(page.count("同步播放本行"), 2)
        self.assertIn("严格结果：route-on 0/2", page)
        self.assertIn("Q/K-only is not appearance-invariant", page)
        for forbidden in ("<form", "<input", "<select", "<textarea"):
            self.assertNotIn(forbidden, page)

    def test_object_legend_is_present_and_unambiguous(self) -> None:
        page = builder.render(_receipt())
        for phrase in (
            "upper-left white ceramic pouring vessel",
            "transparent handled glass vessel",
            "lower-left small white teacup",
        ):
            self.assertIn(phrase, page)

    def test_zero_update_guard_rejects_training(self) -> None:
        valid = {
            "adapter_present": False,
            "base_frozen_before_and_after": True,
            "optimization_steps": 0,
            "trained_checkpoint_loaded": False,
            "training_performed": False,
        }
        builder.validate_zero_update(valid, "valid")
        tampered = dict(valid, optimization_steps=1, training_performed=True)
        with self.assertRaisesRegex(builder.ReviewError, "zero-update"):
            builder.validate_zero_update(tampered, "zero-update")

    def test_remote_root_and_label_binding_rejects_tamper(self) -> None:
        good = f"{builder.FIXED_OUTPUT_ROOT}/{builder.ARMS[0].mp4_name}"
        builder.path_is_bound(good, builder.FIXED_OUTPUT_ROOT,
                              builder.ARMS[0].mp4_name, "good")
        with self.assertRaisesRegex(builder.ReviewError, "root"):
            builder.path_is_bound(
                f"/wrong/root/{builder.ARMS[0].mp4_name}",
                builder.FIXED_OUTPUT_ROOT,
                builder.ARMS[0].mp4_name,
                "tampered",
            )
        with self.assertRaisesRegex(builder.ReviewError, "basename"):
            builder.path_is_bound(
                f"{builder.FIXED_OUTPUT_ROOT}/wrong.mp4",
                builder.FIXED_OUTPUT_ROOT,
                builder.ARMS[0].mp4_name,
                "tampered",
            )

    def test_expected_r3_qk_contract_is_exact(self) -> None:
        self.assertEqual([arm.transport_steps for arm in builder.ARMS], [0, 40, 40])
        self.assertEqual([arm.qk_capture_count for arm in builder.ARMS], [0, 2288, 2288])
        self.assertEqual([arm.qk_replay_count for arm in builder.ARMS], [0, 4576, 4576])
        self.assertTrue(all("V15AR3" in arm.label for arm in builder.ARMS))
        self.assertEqual(len({arm.label for arm in builder.ARMS}), 3)
        self.assertNotIn("plain Frozen", builder.ARMS[0].card_label)
        self.assertIn("not a fully anchor-free baseline", builder.ARMS[0].detail)

    def test_missing_bundle_fails_before_any_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decoded = root / "decoded"
            baseline = root / "baseline"
            output = root / "review"
            decoded.mkdir()
            baseline.mkdir()
            with self.assertRaisesRegex(builder.ReviewError, "completion marker"):
                builder.build(decoded, baseline, output)
            self.assertFalse(output.exists())
            self.assertFalse((root / ".review.building").exists())


if __name__ == "__main__":
    unittest.main()
