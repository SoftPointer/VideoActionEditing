#!/usr/bin/env python3
"""Minimal ffprobe-compatible JSON probe backed by PyAV.

AUH compute images do not contain the login node's ``ffprobe`` binary.  The
Wan runtime already carries PyAV, so this executable implements only the
fixed ffprobe invocation used by ``motive.wan22_i2v_batch``.  It deliberately
fails closed for any other command line.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from typing import Any


_EXPECTED_SHOW_ENTRIES = (
    "stream=codec_name,pix_fmt,width,height,r_frame_rate,"
    "avg_frame_rate,nb_frames,nb_read_frames:"
    "format=duration,size,format_name"
)
_REEXEC_MARKER = "MOTIVE_FFPROBE_PYAV_REEXEC"


def _fail(message: str) -> "NoReturn":
    print(f"ffprobe-pyav-compat: {message}", file=sys.stderr)
    raise SystemExit(2)


def _reexec_runtime_python() -> None:
    python_bin = os.environ.get("MOTIVE_WAN22_PYTHON_BIN", "")
    if not python_bin:
        _fail("MOTIVE_WAN22_PYTHON_BIN is required")
    path = Path(python_bin)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        _fail(f"runtime Python is not an executable regular file: {python_bin}")
    if os.environ.get(_REEXEC_MARKER) == "1":
        return
    environment = dict(os.environ)
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        str(path),
        [str(path), str(Path(__file__).resolve()), *sys.argv[1:]],
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
    if not path.is_file():
        _fail(f"input is not a regular file: {path}")
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
        _fail(f"could not open {path}: {type(error).__name__}: {error}")

    try:
        video_streams = list(container.streams.video)
        if not video_streams:
            _fail(f"no video stream: {path}")
        stream = video_streams[0]
        frames = sum(1 for _ in container.decode(video=stream.index))
        codec_context = stream.codec_context
        codec = (
            getattr(codec_context, "name", None)
            or getattr(getattr(codec_context, "codec", None), "name", None)
            or "unknown"
        )
        pixel_format_value = getattr(codec_context, "format", None)
        pixel_format = (
            getattr(pixel_format_value, "name", None)
            or getattr(codec_context, "pix_fmt", None)
            or "unknown"
        )
        avg_rate = _fraction_text(getattr(stream, "average_rate", None))
        real_rate = _fraction_text(getattr(stream, "base_rate", None))
        if avg_rate is None:
            avg_rate = real_rate
        if real_rate is None:
            real_rate = avg_rate
        if avg_rate is None or real_rate is None:
            _fail(f"video frame rate is unavailable: {path}")

        duration: float | None = None
        container_duration = getattr(container, "duration", None)
        if container_duration is not None:
            duration = float(container_duration) / float(av.time_base)
        elif stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        if duration is None or duration <= 0:
            duration = frames / float(Fraction(avg_rate))
        if frames <= 0 or duration <= 0:
            _fail(f"video has no decodable frames: {path}")

        return {
            "streams": [
                {
                    "codec_name": str(codec),
                    "pix_fmt": str(pixel_format),
                    "width": int(codec_context.width),
                    "height": int(codec_context.height),
                    "r_frame_rate": real_rate,
                    "avg_frame_rate": avg_rate,
                    "nb_frames": str(int(getattr(stream, "frames", 0) or 0)),
                    "nb_read_frames": str(frames),
                }
            ],
            "format": {
                "duration": f"{duration:.6f}",
                "size": str(path.stat().st_size),
                "format_name": str(getattr(container.format, "name", "unknown")),
            },
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
