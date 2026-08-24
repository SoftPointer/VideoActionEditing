#!/usr/bin/env python3
"""Fail-closed ffprobe shim for the audited Bernini EPMC episode loader.

The AUH compute image does not provide ``ffprobe``.  This executable supports
only the one invocation emitted by ``fewshot_episode_io.py`` and obtains the
authoritative frame count by decoding the selected video stream with PyAV.
It re-executes itself with ``BERNINI_EPMC_PYTHON_BIN`` before importing PyAV.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn, Sequence


_PYTHON_ENV = "BERNINI_EPMC_PYTHON_BIN"
_REEXEC_MARKER = "BERNINI_EPMC_FFPROBE_PYAV_REEXEC"
_EXPECTED_ARGUMENTS = (
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=nb_read_frames,nb_frames,avg_frame_rate",
    "-of",
    "json",
)


def _fail(message: str) -> NoReturn:
    print(f"bernini-epmc-ffprobe: {message}", file=sys.stderr)
    raise SystemExit(2)


def _resolved_executable(raw_path: str) -> Path:
    path = Path(raw_path)
    if not raw_path or not path.is_absolute():
        _fail(f"{_PYTHON_ENV} must name an absolute executable")
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError) as error:
        _fail(f"invalid {_PYTHON_ENV}: {type(error).__name__}: {error}")
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        _fail(f"{_PYTHON_ENV} does not resolve to an executable regular file")
    return resolved


def _reexec_runtime_python() -> None:
    """Re-enter this script once using the exact audited runtime Python."""

    runtime = _resolved_executable(os.environ.get(_PYTHON_ENV, ""))
    marker = os.environ.get(_REEXEC_MARKER)
    if marker == "1":
        try:
            current = Path(sys.executable).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            _fail(f"cannot resolve the active Python: {type(error).__name__}: {error}")
        if current != runtime:
            _fail("re-exec marker is set under the wrong Python interpreter")
        return
    if marker is not None:
        _fail("invalid re-exec marker")

    try:
        script = Path(__file__).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(f"cannot resolve probe script: {type(error).__name__}: {error}")
    environment = dict(os.environ)
    environment[_REEXEC_MARKER] = "1"
    try:
        os.execve(
            str(runtime),
            [str(runtime), str(script), *sys.argv[1:]],
            environment,
        )
    except OSError as error:
        _fail(f"runtime Python re-exec failed: {type(error).__name__}: {error}")
    _fail("runtime Python re-exec unexpectedly returned")


def _input_path(argv: Sequence[str]) -> Path:
    """Accept only the literal ffprobe argument vector used by the loader."""

    if len(argv) != len(_EXPECTED_ARGUMENTS) + 1:
        _fail("unsupported invocation")
    if tuple(argv[:-1]) != _EXPECTED_ARGUMENTS:
        _fail("unsupported invocation")

    path = Path(argv[-1])
    if not path.is_absolute():
        _fail("input path must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        _fail(f"cannot stat input: {type(error).__name__}: {error}")
    if not stat.S_ISREG(mode):
        _fail("input must be a non-symlink regular file")
    return path


def _positive_rate_text(value: Any) -> str:
    if value is None:
        _fail("video average frame rate is unavailable")
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        _fail("video average frame rate is invalid")
    try:
        rate = Fraction(int(numerator), int(denominator))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        _fail("video average frame rate is invalid")
    if rate <= 0:
        _fail("video average frame rate is not positive")
    return f"{rate.numerator}/{rate.denominator}"


def _declared_frame_text(value: Any, decoded_frames: int) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        _fail("declared frame count is invalid")
    try:
        declared = int(value)
    except (TypeError, ValueError, OverflowError):
        _fail("declared frame count is invalid")
    if declared <= 0:
        return "N/A"
    if declared != decoded_frames:
        _fail(
            "declared frame count disagrees with the exact decoded frame count "
            f"({declared} != {decoded_frames})"
        )
    return str(declared)


def _probe(path: Path) -> dict[str, Any]:
    try:
        import av
    except Exception as error:
        _fail(f"PyAV import failed: {type(error).__name__}: {error}")

    try:
        with av.open(str(path), mode="r") as container:
            video_streams = list(container.streams.video)
            if not video_streams:
                _fail("input has no video stream")
            stream = video_streams[0]  # exact meaning of ffprobe v:0
            decoded_frames = sum(1 for _ in container.decode(stream))
            if decoded_frames <= 0:
                _fail("selected video stream has no decodable frames")
            rate = _positive_rate_text(getattr(stream, "average_rate", None))
            declared = _declared_frame_text(
                getattr(stream, "frames", None), decoded_frames
            )
    except SystemExit:
        raise
    except Exception as error:
        _fail(f"PyAV decode failed: {type(error).__name__}: {error}")

    return {
        "streams": [
            {
                "nb_read_frames": str(decoded_frames),
                "nb_frames": declared,
                "avg_frame_rate": rate,
            }
        ]
    }


def main() -> int:
    path = _input_path(sys.argv[1:])
    _reexec_runtime_python()
    json.dump(_probe(path), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
