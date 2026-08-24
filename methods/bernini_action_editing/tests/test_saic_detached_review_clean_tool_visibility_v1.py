#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"missing remote regression environment: {name}")
    return value


class CleanToolVisibilityTests(unittest.TestCase):
    def test_sixty_workers_see_exact_hash_bound_tools_under_clean_environment(self) -> None:
        python_bin = _required("SAIC_T2V_REVIEW_PYTHON_BIN")
        expected_ffmpeg = _required("SAIC_T2V_REVIEW_EXPECTED_FFMPEG")
        expected_ffprobe = _required("SAIC_T2V_REVIEW_EXPECTED_FFPROBE")
        ffmpeg_sha = _required("SAIC_T2V_REVIEW_FFMPEG_SHA256")
        ffprobe_sha = _required("SAIC_T2V_REVIEW_FFPROBE_SHA256")
        clean_path = _required("SAIC_T2V_REVIEW_CLEAN_PATH")
        program = textwrap.dedent(
            """
            from concurrent.futures import ProcessPoolExecutor
            import hashlib, os, pathlib, shutil, subprocess

            def file_sha(path):
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def check(index):
                observed = []
                for name, expected_path, expected_sha in (
                    ("ffmpeg", os.environ["EXPECTED_FFMPEG"], os.environ["FFMPEG_SHA"]),
                    ("ffprobe", os.environ["EXPECTED_FFPROBE"], os.environ["FFPROBE_SHA"]),
                ):
                    found = shutil.which(name)
                    if found is None:
                        raise RuntimeError(f"worker {index}: {name} absent")
                    resolved = str(pathlib.Path(found).resolve(strict=True))
                    if resolved != expected_path or file_sha(resolved) != expected_sha:
                        raise RuntimeError(f"worker {index}: {name} identity differs")
                    completed = subprocess.run(
                        [resolved, "-version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    if completed.returncode != 0 or not completed.stdout.strip():
                        raise RuntimeError(f"worker {index}: {name} cannot execute")
                    observed.append((name, resolved, expected_sha))
                return index, observed

            with ProcessPoolExecutor(max_workers=16) as executor:
                rows = list(executor.map(check, range(60)))
            if [row[0] for row in rows] != list(range(60)):
                raise SystemExit("60-worker order/closure differs")
            print("clean-tool-workers=60 ffmpeg=true ffprobe=true hashes=true")
            """
        )
        environment = {
            "PATH": clean_path,
            "SAIC_T2V_REVIEW_PYTHON_BIN": python_bin,
            "EXPECTED_FFMPEG": str(Path(expected_ffmpeg).resolve(strict=True)),
            "EXPECTED_FFPROBE": str(Path(expected_ffprobe).resolve(strict=True)),
            "FFMPEG_SHA": ffmpeg_sha,
            "FFPROBE_SHA": ffprobe_sha,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        completed = subprocess.run(
            [python_bin, "-I", "-B", "-c", program],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("clean-tool-workers=60", completed.stdout)

    def test_expected_hashes_match_before_clean_subprocess(self) -> None:
        for path_name, sha_name in (
            ("SAIC_T2V_REVIEW_EXPECTED_FFMPEG", "SAIC_T2V_REVIEW_FFMPEG_SHA256"),
            ("SAIC_T2V_REVIEW_EXPECTED_FFPROBE", "SAIC_T2V_REVIEW_FFPROBE_SHA256"),
        ):
            path = Path(_required(path_name)).resolve(strict=True)
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_symlink())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), _required(sha_name))


if __name__ == "__main__":
    unittest.main()
