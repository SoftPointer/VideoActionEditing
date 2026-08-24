#!/usr/bin/env python3
"""Strict ffprobe-compatible video probe for AUH compute nodes.

The AUH compute image does not provide the ``ffprobe`` executable.  This
helper implements only the single invocation emitted by
``shared8_contract.probe_video`` and rejects every other command line.  It
re-executes itself with the explicitly pinned experiment Python so the PyAV
decoder comes from the same frozen runtime as model inference.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, NoReturn


_EXPECTED_ARGUMENTS = [
    "-v",
    "error",
    "-count_frames",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
    "-show_entries",
    "format=duration",
    "-of",
    "json",
]
_REEXEC_MARKER = "ACTION_SHARED8_PYAV_PROBE_REEXEC"


def _fail(message: str) -> NoReturn:
    print(f"shared8-ffprobe-pyav: {message}", file=sys.stderr)
    raise SystemExit(2)


def _runtime_python() -> Path:
    raw = os.environ.get("ACTION_BASELINE_PYTHON_BIN", "")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        _fail("ACTION_BASELINE_PYTHON_BIN must be an executable regular file")
    return path


def _reexec_with_runtime_python() -> None:
    python_bin = _runtime_python()
    if os.environ.get(_REEXEC_MARKER) == "1":
        if Path(sys.executable).resolve() != python_bin.resolve():
            _fail("probe did not re-execute with ACTION_BASELINE_PYTHON_BIN")
        return
    environment = dict(os.environ)
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        str(python_bin),
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def _input_path(arguments: list[str]) -> Path:
    if len(arguments) != len(_EXPECTED_ARGUMENTS) + 1:
        _fail("unsupported invocation")
    if arguments[:-1] != _EXPECTED_ARGUMENTS:
        _fail("unsupported invocation")
    path = Path(arguments[-1])
    if not path.is_file() or path.is_symlink():
        _fail(f"input is not a plain file: {path}")
    return path


def _positive_fraction(value: Any) -> Fraction | None:
    if value is None:
        return None
    try:
        fraction = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return fraction if fraction > 0 else None


def _probe(path: Path) -> dict[str, object]:
    try:
        import av
    except Exception as error:  # pragma: no cover - depends on runtime install
        _fail(f"PyAV import failed: {type(error).__name__}: {error}")

    try:
        container = av.open(str(path), mode="r")
    except Exception as error:
        _fail(f"could not open {path}: {type(error).__name__}: {error}")

    try:
        streams = list(container.streams.video)
        if not streams:
            _fail(f"no video stream: {path}")
        stream = streams[0]
        # Pass the selected stream object itself.  ``decode(video=N)`` indexes
        # the list of video streams, whereas ``stream.index`` is the global
        # container-stream index and is therefore unsafe for audio-first files.
        frame_count = sum(1 for _ in container.decode(stream))
        if frame_count <= 0:
            _fail(f"video has no decodable frames: {path}")

        codec_context = stream.codec_context
        codec = (
            getattr(codec_context, "name", None)
            or getattr(getattr(codec_context, "codec", None), "name", None)
            or "unknown"
        )
        width = int(getattr(codec_context, "width", 0) or 0)
        height = int(getattr(codec_context, "height", 0) or 0)
        if width <= 0 or height <= 0:
            _fail(f"invalid video geometry: {path}")

        rate = _positive_fraction(getattr(stream, "average_rate", None))
        if rate is None:
            _fail(f"average video frame rate is unavailable: {path}")

        container_duration = getattr(container, "duration", None)
        if container_duration is None:
            _fail(f"container duration is unavailable: {path}")
        duration = float(container_duration) / float(av.time_base)
        if not math.isfinite(duration) or duration <= 0:
            _fail(f"invalid container duration: {path}")

        return {
            "streams": [
                {
                    "codec_name": str(codec),
                    "width": width,
                    "height": height,
                    "avg_frame_rate": f"{rate.numerator}/{rate.denominator}",
                    "nb_read_frames": str(frame_count),
                }
            ],
            "format": {"duration": f"{duration:.9f}"},
        }
    finally:
        container.close()


def main() -> int:
    _reexec_with_runtime_python()
    path = _input_path(sys.argv[1:])
    json.dump(_probe(path), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
