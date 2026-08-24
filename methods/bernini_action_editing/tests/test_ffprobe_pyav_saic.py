from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ffprobe_pyav_saic.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ffprobe_pyav_saic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FFprobePyAVSAICTests(unittest.TestCase):
    def test_launcher_uses_archive_bound_pyav_probe(self) -> None:
        launcher = (
            ROOT / "scripts" / "auh_generate_saic_pure_t2v_event_bank_all8_v1.sbatch"
        ).read_text("utf-8")
        self.assertIn("tools/ffprobe_pyav_saic.py", launcher)
        self.assertIn('ffprobe_bin="${method_root}/tools/ffprobe_pyav_saic.py"', launcher)
        self.assertNotIn("SAIC_T2V_FFPROBE:?", launcher)

    def test_rejects_every_unregistered_invocation(self) -> None:
        module = _load_module()
        with self.assertRaises(SystemExit):
            module._input_path(["-version"])

    def test_requires_absolute_plain_input(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.mp4"
            path.write_bytes(b"not-media")
            with self.assertRaises(SystemExit):
                module._input_path([str(path.relative_to(Path(temp)))])

    def test_end_to_end_matches_exact81_contract_when_pyav_is_available(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("PyAV is not installed in this local environment")
        ffmpeg = os.environ.get("SAIC_TEST_FFMPEG")
        if not ffmpeg:
            self.skipTest("set SAIC_TEST_FFMPEG to exercise media integration")
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp).resolve() / "black.mp4"
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x48:r=25",
                    "-frames:v",
                    "81",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
            command = [
                str(SCRIPT),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames",
                "-of",
                "json",
                str(video),
            ]
            environment = dict(os.environ)
            environment["SAIC_T2V_PYTHON_BIN"] = sys.executable
            value = json.loads(subprocess.check_output(command, text=True, env=environment))
            self.assertEqual(value["streams"][0]["width"], 64)
            self.assertEqual(value["streams"][0]["height"], 48)
            self.assertEqual(value["streams"][0]["nb_frames"], "81")
            self.assertEqual(value["streams"][0]["nb_read_frames"], "81")


if __name__ == "__main__":
    unittest.main()
