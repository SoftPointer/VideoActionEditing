#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "auh_build_saic_t2v_event_bank_detached_review_v1_cpu.sbatch"
)


class DetachedReviewLauncherTests(unittest.TestCase):
    def test_cpu_nonhold_launcher_is_fresh_and_zero_authority(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=32", text)
        self.assertIn("#SBATCH --mem=192G", text)
        self.assertNotIn("#SBATCH --gres", text)
        self.assertNotIn("#SBATCH --hold", text)
        self.assertIn('[[ "${output_root}" != / && ! -e "${output_root}"', text)
        self.assertIn("--workers", text)
        self.assertIn("machine_authority=zero", text)
        self.assertIn("semantic=UNASSESSED", text)
        self.assertIn("event_verified=false", text)
        self.assertIn("optimizer=false", text)
        self.assertIn("observer-protocol.json", text)
        self.assertIn("blind-review.html", text)
        self.assertIn("-eq 2", text)
        self.assertIn("validate \\", text)
        self.assertIn('--output-root "${output_root}"', text)
        self.assertIn("SAIC_T2V_REVIEW_FFMPEG_BIN", text)
        self.assertIn("SAIC_T2V_REVIEW_FFMPEG_SHA256", text)
        self.assertIn("SAIC_T2V_REVIEW_FFPROBE_WRAPPER_SHA256", text)
        self.assertIn('ln -s -- "${ffmpeg_bin}" "${tool_bin}/ffmpeg"', text)
        self.assertIn('export PATH="${tool_bin}:/usr/bin:/bin"', text)
        self.assertIn("ffprobe_pyav_exact81_diagnostic_v1.py", text)
        self.assertIn("test_saic_detached_review_clean_tool_visibility_v1.py", text)

    def test_launcher_runs_tests_on_auh_before_builder(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        builder_test = text.index(
            '"${python_bin}" -B "${method_root}/tests/test_build_saic_t2v_event_bank_detached_review_v1.py"'
        )
        diagnostic_test = text.index(
            '"${python_bin}" -B "${method_root}/tests/test_saic_exact81_media_diagnostics_v1.py"'
        )
        clean_tool_test = text.index(
            '"${python_bin}" -B "${method_root}/tests/test_saic_detached_review_clean_tool_visibility_v1.py"'
        )
        build = text.index(
            '"${python_bin}" -B "${method_root}/tools/build_saic_t2v_event_bank_detached_review_v1.py"'
        )
        self.assertLess(builder_test, build)
        self.assertLess(diagnostic_test, build)
        self.assertLess(clean_tool_test, build)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)


if __name__ == "__main__":
    unittest.main()
