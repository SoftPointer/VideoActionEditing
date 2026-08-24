from __future__ import annotations

from contextlib import redirect_stderr
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import ffprobe_pyav_compat as probe  # noqa: E402


def _exact_argv(path: Path) -> list[str]:
    return [*probe._EXPECTED_ARGUMENTS, str(path)]


class _FakeContainer:
    def __init__(self, stream, frame_count: int = 81, decode_error=None):
        self.streams = types.SimpleNamespace(video=[stream])
        self.frame_count = frame_count
        self.decode_error = decode_error
        self.decoded_stream = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def decode(self, stream):
        self.decoded_stream = stream
        if self.decode_error is not None:
            raise self.decode_error
        return iter(range(self.frame_count))


class InvocationContractTests(unittest.TestCase):
    def test_accepts_only_the_episode_loader_argument_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory).resolve() / "clip.mp4"
            media.touch()
            self.assertEqual(probe._input_path(_exact_argv(media)), media)

            invalid = (
                _exact_argv(media)[:-1],
                ["-v", "quiet", *_exact_argv(media)[2:]],
                [
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,nb_frames,avg_frame_rate",
                    "-of",
                    "json",
                    str(media),
                ],
                [*_exact_argv(media), "extra"],
            )
            for argv in invalid:
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        probe._input_path(argv)

    def test_requires_absolute_non_symlink_regular_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            media = root / "clip.mp4"
            media.touch()
            symlink = root / "link.mp4"
            symlink.symlink_to(media)
            candidates = (Path("relative.mp4"), symlink, root, root / "missing.mp4")
            for path in candidates:
                with self.subTest(path=path), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        probe._input_path(_exact_argv(path))


class ReexecContractTests(unittest.TestCase):
    def test_reexecs_with_the_configured_runtime_and_preserves_exact_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory).resolve() / "python"
            runtime.touch(mode=0o700)
            media = Path(directory).resolve() / "clip.mp4"
            media.touch()
            argv = [str(probe.__file__), *_exact_argv(media)]
            with mock.patch.dict(
                os.environ,
                {probe._PYTHON_ENV: str(runtime)},
                clear=True,
            ):
                with mock.patch.object(sys, "argv", argv):
                    with mock.patch.object(
                        os, "execve", side_effect=OSError("sentinel")
                    ) as execute:
                        with redirect_stderr(io.StringIO()):
                            with self.assertRaisesRegex(SystemExit, "2"):
                                probe._reexec_runtime_python()
            executable, child_argv, environment = execute.call_args.args
            self.assertEqual(executable, str(runtime))
            self.assertEqual(child_argv[0], str(runtime))
            self.assertEqual(child_argv[1], str(Path(probe.__file__).resolve()))
            self.assertEqual(child_argv[2:], _exact_argv(media))
            self.assertEqual(environment[probe._REEXEC_MARKER], "1")

    def test_marker_allows_only_the_configured_active_runtime(self):
        executable = str(Path(sys.executable).resolve())
        environment = {
            probe._PYTHON_ENV: executable,
            probe._REEXEC_MARKER: "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            probe._reexec_runtime_python()

        wrong_runtime = Path(executable).parent / "definitely-not-current-python"
        with tempfile.TemporaryDirectory() as directory:
            wrong_runtime = Path(directory).resolve() / "python"
            wrong_runtime.touch(mode=0o700)
            environment[probe._PYTHON_ENV] = str(wrong_runtime)
            with mock.patch.dict(os.environ, environment, clear=True):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        probe._reexec_runtime_python()

    def test_rejects_missing_relative_non_executable_and_bad_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory).resolve() / "python"
            runtime.touch(mode=0o600)
            environments = (
                {},
                {probe._PYTHON_ENV: "relative/python"},
                {probe._PYTHON_ENV: str(runtime)},
                {
                    probe._PYTHON_ENV: str(Path(sys.executable).resolve()),
                    probe._REEXEC_MARKER: "unexpected",
                },
            )
            for environment in environments:
                with self.subTest(environment=environment):
                    with mock.patch.dict(os.environ, environment, clear=True):
                        with redirect_stderr(io.StringIO()):
                            with self.assertRaisesRegex(SystemExit, "2"):
                                probe._reexec_runtime_python()


class PyAVProbeTests(unittest.TestCase):
    def _run_probe(self, stream, frame_count=81, decode_error=None):
        container = _FakeContainer(stream, frame_count, decode_error)
        fake_av = types.SimpleNamespace(open=mock.Mock(return_value=container))
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"av": fake_av}):
                media = Path(directory).resolve() / "clip.mp4"
                media.touch()
                result = probe._probe(media)
        fake_av.open.assert_called_once_with(str(media), mode="r")
        return result, container

    def test_exact_decode_emits_consumer_compatible_fields(self):
        stream = types.SimpleNamespace(average_rate=Fraction(25, 1), frames=81)
        result, container = self._run_probe(stream)
        self.assertEqual(
            result,
            {
                "streams": [
                    {
                        "nb_read_frames": "81",
                        "nb_frames": "81",
                        "avg_frame_rate": "25/1",
                    }
                ]
            },
        )
        self.assertIs(container.decoded_stream, stream)
        self.assertTrue(container.closed)

    def test_unknown_declared_count_is_reported_as_na(self):
        stream = types.SimpleNamespace(average_rate=Fraction(50, 2), frames=0)
        result, _ = self._run_probe(stream)
        self.assertEqual(result["streams"][0]["nb_frames"], "N/A")
        self.assertEqual(result["streams"][0]["nb_read_frames"], "81")
        self.assertEqual(result["streams"][0]["avg_frame_rate"], "25/1")

    def test_fail_closed_on_bad_metadata_empty_or_incomplete_decode(self):
        cases = (
            (types.SimpleNamespace(average_rate=None, frames=81), 81, None),
            (types.SimpleNamespace(average_rate=Fraction(0, 1), frames=81), 81, None),
            (types.SimpleNamespace(average_rate=Fraction(25, 1), frames=80), 81, None),
            (types.SimpleNamespace(average_rate=Fraction(25, 1), frames=0), 0, None),
            (
                types.SimpleNamespace(average_rate=Fraction(25, 1), frames=81),
                81,
                RuntimeError("truncated"),
            ),
        )
        for stream, frame_count, decode_error in cases:
            with self.subTest(stream=stream), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "2"):
                    self._run_probe(stream, frame_count, decode_error)

    def test_fail_closed_without_a_video_stream(self):
        container = _FakeContainer(None)
        container.streams.video = []
        fake_av = types.SimpleNamespace(open=mock.Mock(return_value=container))
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"av": fake_av}):
                with redirect_stderr(io.StringIO()):
                    media = Path(directory).resolve() / "clip.mp4"
                    media.touch()
                    with self.assertRaisesRegex(SystemExit, "2"):
                        probe._probe(media)
        self.assertTrue(container.closed)


class SubprocessContractTests(unittest.TestCase):
    def test_cli_json_and_rejection_without_requiring_local_pyav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            media = root / "clip.mp4"
            media.touch()
            (root / "av.py").write_text(
                """
from fractions import Fraction

class Stream:
    average_rate = Fraction(25, 1)
    frames = 81

class Container:
    def __init__(self):
        self.stream = Stream()
        self.streams = type('Streams', (), {'video': [self.stream]})()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, traceback):
        return False
    def decode(self, stream):
        assert stream is self.stream
        return iter(range(81))

def open(path, mode='r'):
    assert mode == 'r'
    return Container()
""".lstrip(),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment[probe._PYTHON_ENV] = str(Path(sys.executable).resolve())
            environment[probe._REEXEC_MARKER] = "1"
            environment["PYTHONPATH"] = str(root)
            command = [sys.executable, str(Path(probe.__file__).resolve()), *_exact_argv(media)]
            completed = subprocess.run(
                command,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(completed.stdout),
                {
                    "streams": [
                        {
                            "nb_read_frames": "81",
                            "nb_frames": "81",
                            "avg_frame_rate": "25/1",
                        }
                    ]
                },
            )

            rejected = subprocess.run(
                [sys.executable, str(Path(probe.__file__).resolve()), "--version"],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertEqual(rejected.stdout, "")


if __name__ == "__main__":
    unittest.main()
