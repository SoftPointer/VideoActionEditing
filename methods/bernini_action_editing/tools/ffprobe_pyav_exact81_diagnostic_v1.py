#!/usr/bin/env python3
"""Closed ffprobe adapter for SAIC exact81 diagnostics on AUH compute nodes.

The MI210 compute image has no system ``ffprobe``.  This archive-bound tool
implements only ``-version`` and the exact JSON probe invocation used by
``decoded_temporal_event_evaluator_v1.probe_exact81_video``.  It re-executes
the explicitly pinned vace Python and uses that runtime's PyAV decoder.  Every
other argument vector fails closed.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn


SCHEMA_VERSION = "bernini-saic-pyav-exact81-ffprobe-adapter-v1"
_REEXEC_MARKER = "SAIC_T2V_REVIEW_FFPROBE_REEXEC"
_EXPECTED_PREFIX = [
    "-v",
    "error",
    "-count_frames",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height,avg_frame_rate,nb_read_frames",
    "-of",
    "json",
]


def _fail(message: str) -> NoReturn:
    print(f"saic-review-ffprobe: {message}", file=sys.stderr)
    raise SystemExit(2)


def _plain_executable(path_text: str, *, label: str) -> Path:
    path = Path(path_text)
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(f"cannot stat {label}: {type(error).__name__}")
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(path, os.X_OK)
    ):
        _fail(f"{label} must be an absolute executable plain file")
    return path.resolve(strict=True)


def _reexec_runtime_python() -> None:
    python_bin = _plain_executable(
        os.environ.get("SAIC_T2V_REVIEW_PYTHON_BIN", ""),
        label="SAIC_T2V_REVIEW_PYTHON_BIN",
    )
    if os.environ.get(_REEXEC_MARKER) == "1":
        if Path(sys.executable).resolve() != python_bin:
            _fail("runtime Python identity differs after re-exec")
        return
    environment = dict(os.environ)
    environment[_REEXEC_MARKER] = "1"
    os.execve(
        str(python_bin),
        [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def _fraction_text(value: Any) -> str | None:
    if value is None:
        return None
    fraction = Fraction(value)
    if fraction <= 0:
        return None
    return f"{fraction.numerator}/{fraction.denominator}"


def _input_path(argv: list[str]) -> Path:
    if len(argv) != len(_EXPECTED_PREFIX) + 1 or argv[:-1] != _EXPECTED_PREFIX:
        _fail("unsupported invocation")
    path = Path(argv[-1])
    try:
        metadata = path.lstat()
    except OSError as error:
        _fail(f"cannot stat input: {type(error).__name__}")
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        _fail("input must be an absolute plain file")
    return path.resolve(strict=True)


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
        if average_rate is None:
            average_rate = _fraction_text(getattr(stream, "base_rate", None))
        if average_rate is None:
            _fail("video frame rate is unavailable")
        context = stream.codec_context
        return {
            "streams": [
                {
                    "width": int(context.width),
                    "height": int(context.height),
                    "avg_frame_rate": average_rate,
                    "nb_read_frames": str(frames),
                }
            ]
        }
    finally:
        container.close()


def main() -> int:
    _reexec_runtime_python()
    if sys.argv[1:] == ["-version"]:
        print(f"ffprobe adapter {SCHEMA_VERSION} (PyAV runtime)")
        return 0
    path = _input_path(sys.argv[1:])
    json.dump(_probe(path), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
