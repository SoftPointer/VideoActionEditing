#!/usr/bin/env python3
"""Minimal ffprobe-compatible exact81 probe backed by runtime PyAV.

AUH login nodes expose ``/usr/bin/ffprobe`` but some MI210 compute images do
not.  This executable implements only the fixed JSON invocation used by the
SAIC geometry-proxy builder.  It re-executes the immutable runtime Python from
``SAIC_T2V_PYTHON_BIN`` so the shebang never selects a different environment.
Every other command line fails closed.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn


_EXPECTED_SHOW_ENTRIES = (
    "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames"
)
_REEXEC_MARKER = "SAIC_FFPROBE_PYAV_REEXEC"


def _fail(message: str) -> NoReturn:
    print(f"saic-ffprobe-pyav: {message}", file=sys.stderr)
    raise SystemExit(2)


def _plain_executable(path_text: str, *, label: str) -> Path:
    path = Path(path_text)
    try:
        value = path.lstat()
    except OSError as error:
        _fail(f"cannot stat {label}: {type(error).__name__}")
    if (
        not path.is_absolute()
        or stat.S_ISLNK(value.st_mode)
        or not stat.S_ISREG(value.st_mode)
        or not os.access(path, os.X_OK)
    ):
        _fail(f"{label} must be an absolute executable regular file")
    return path


def _reexec_runtime_python() -> None:
    python_bin = _plain_executable(
        os.environ.get("SAIC_T2V_PYTHON_BIN", ""),
        label="SAIC_T2V_PYTHON_BIN",
    )
    if os.environ.get(_REEXEC_MARKER) == "1":
        if Path(sys.executable).resolve() != python_bin.resolve():
            _fail("runtime Python identity differs after re-exec")
        return
    environment = dict(os.environ)
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        str(python_bin),
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def _input_path(argv: list[str]) -> Path:
    expected = [
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        _EXPECTED_SHOW_ENTRIES,
        "-of",
        "json",
    ]
    if len(argv) != len(expected) + 1 or argv[:-1] != expected:
        _fail("unsupported invocation")
    path = Path(argv[-1])
    try:
        value = path.lstat()
    except OSError as error:
        _fail(f"cannot stat input: {type(error).__name__}")
    if not path.is_absolute() or stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        _fail("input must be an absolute regular non-symlink file")
    return path


def _fraction_text(value: Any) -> str | None:
    if value is None:
        return None
    fraction = Fraction(value)
    if fraction <= 0:
        return None
    return f"{fraction.numerator}/{fraction.denominator}"


def _probe(path: Path) -> dict[str, Any]:
    try:
        import av
    except Exception as error:
        _fail(f"PyAV import failed: {type(error).__name__}: {error}")
    try:
        container = av.open(str(path), mode="r")
    except Exception as error:
        _fail(f"could not open input: {type(error).__name__}: {error}")
    try:
        streams = list(container.streams.video)
        if len(streams) != 1:
            _fail("input must contain exactly one video stream")
        stream = streams[0]
        frames = sum(1 for _ in container.decode(video=stream.index))
        average_rate = _fraction_text(getattr(stream, "average_rate", None))
        real_rate = _fraction_text(getattr(stream, "base_rate", None))
        if average_rate is None:
            average_rate = real_rate
        if real_rate is None:
            real_rate = average_rate
        if average_rate is None or real_rate is None:
            _fail("video frame rate is unavailable")
        declared_frames = int(getattr(stream, "frames", 0) or 0)
        if declared_frames <= 0:
            declared_frames = frames
        context = stream.codec_context
        return {
            "streams": [
                {
                    "width": int(context.width),
                    "height": int(context.height),
                    "r_frame_rate": real_rate,
                    "avg_frame_rate": average_rate,
                    "nb_frames": str(declared_frames),
                    "nb_read_frames": str(frames),
                }
            ]
        }
    finally:
        container.close()


def main() -> int:
    _reexec_runtime_python()
    path = _input_path(sys.argv[1:])
    json.dump(_probe(path), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
