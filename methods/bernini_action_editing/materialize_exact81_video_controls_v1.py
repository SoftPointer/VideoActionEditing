#!/usr/bin/env python3
"""Create exact81 temporal video controls for a target or selfgen anchor.

The controls are extractor inputs only.  They are never generator conditions,
pixel targets, VAE targets, or flow-matching endpoints.  Shuffle retains frame
zero and permutes ten contiguous eight-frame blocks, preserving local motion
while destroying the global event order.  Reverse matches the existing MEV
full-video reverse convention; incomplete holds frame 40 through the end.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-exact81-temporal-video-controls-v1"
FRAME_COUNT = 81
FPS = 25
CONTROL_ROLES = ("zero_or_noop", "temporal_shuffle", "reverse", "incomplete")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}")


class Exact81VideoControlError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise Exact81VideoControlError(message)


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n"
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        return text.encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Exact81VideoControlError("receipt is not finite JSON") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    named = path.stat()
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or identity != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    ):
        fail(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _shuffle_block_order(seed: int) -> tuple[int, ...]:
    if isinstance(seed, bool) or type(seed) is not int or seed < 0:
        fail("shuffle seed must be a nonnegative integer")
    order = tuple(
        sorted(
            range(10),
            key=lambda block: hashlib.sha256(
                f"exact81-video-shuffle-v1:{seed}:{block}".encode("ascii")
            ).digest(),
        )
    )
    identity = tuple(range(10))
    reverse = tuple(reversed(identity))
    if order in (identity, reverse):
        order = order[3:] + order[:3]
    if sorted(order) != list(range(10)) or order in (identity, reverse):
        fail("shuffle block order is degenerate")
    return order


def control_frame_indices(role: str, *, seed: int) -> tuple[int, ...]:
    if role not in CONTROL_ROLES:
        fail(f"unsupported temporal video control: {role}")
    if role == "zero_or_noop":
        indices = (0,) * FRAME_COUNT
    elif role == "reverse":
        indices = tuple(range(FRAME_COUNT - 1, -1, -1))
    elif role == "incomplete":
        indices = tuple(range(41)) + (40,) * 40
    else:
        blocks = tuple(
            tuple(range(1 + 8 * block, 1 + 8 * (block + 1)))
            for block in range(10)
        )
        order = _shuffle_block_order(seed)
        indices = (0,) + tuple(index for block in order for index in blocks[block])
    if len(indices) != FRAME_COUNT or any(not 0 <= index < FRAME_COUNT for index in indices):
        fail("temporal control frame map differs")
    if role == "temporal_shuffle" and (
        indices == tuple(range(FRAME_COUNT))
        or indices == tuple(range(FRAME_COUNT - 1, -1, -1))
        or indices[0] != 0
    ):
        fail("shuffle must be nontrivial, non-reverse, and retain frame zero")
    return indices


def _require_av() -> tuple[Any, Any]:
    try:
        import av
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise Exact81VideoControlError(
            "exact81 video controls require PyAV and NumPy"
        ) from error
    return av, np


def decode_exact81(path: Path) -> tuple[list[Any], Mapping[str, Any]]:
    av, np = _require_av()
    try:
        with av.open(str(path), "r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                fail("input must contain exactly one video stream")
            stream = streams[0]
            rate = Fraction(stream.average_rate)
            frames = [
                np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                for frame in container.decode(stream)
            ]
            codec = str(stream.codec_context.name)
    except Exact81VideoControlError:
        raise
    except Exception as error:
        raise Exact81VideoControlError(f"cannot decode input video: {path}") from error
    if rate != Fraction(FPS, 1) or len(frames) != FRAME_COUNT:
        fail("input must decode to exact81 at 25 fps")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        fail("input frame geometry changes over time")
    height, width, channels = next(iter(shapes))
    if channels != 3 or height <= 0 or width <= 0 or height % 2 or width % 2:
        fail("input must have positive even RGB geometry")
    return frames, {
        "codec_name": codec,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "height": height,
        "width": width,
    }


def encode_exact81(path: Path, frames: Sequence[Any]) -> None:
    av, _ = _require_av()
    if len(frames) != FRAME_COUNT or path.exists() or path.is_symlink():
        fail("exact81 output path/frame closure differs")
    height, width = map(int, frames[0].shape[:2])
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        fail("partial video output already exists")
    try:
        with av.open(str(temporary), "w", format="mp4") as container:
            stream = container.add_stream(
                "libx264", rate=FPS, options={"crf": "18", "preset": "medium"}
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, FPS)
            for ordinal, array in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = ordinal
                frame.time_base = Fraction(1, FPS)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary, path)
    except Exception:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def probe_exact81(path: Path) -> Mapping[str, Any]:
    av, _ = _require_av()
    try:
        with av.open(str(path), "r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                fail("control output stream closure differs")
            stream = streams[0]
            count = sum(1 for _ in container.decode(stream))
            result = {
                "codec_name": str(stream.codec_context.name),
                "fps": float(Fraction(stream.average_rate)),
                "frame_count": count,
                "height": int(stream.codec_context.height),
                "width": int(stream.codec_context.width),
            }
    except Exact81VideoControlError:
        raise
    except Exception as error:
        raise Exact81VideoControlError(f"cannot probe output video: {path}") from error
    if (
        result["codec_name"] != "h264"
        or result["fps"] != float(FPS)
        or result["frame_count"] != FRAME_COUNT
    ):
        fail("control output is not H.264 exact81/25fps")
    return result


def materialize_controls(
    *,
    input_video: Path | str,
    expected_sha256: str,
    output: Path | str,
    case_id: str,
    anchor_kind: str,
    seed: int,
) -> dict[str, Any]:
    source = Path(input_video).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        fail("input video must be one regular non-symlink file")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        fail("expected input SHA-256 differs")
    if file_sha256(source) != expected_sha256:
        fail("input video SHA-256 differs")
    if _SAFE_ID.fullmatch(case_id) is None or anchor_kind not in ("target", "selfgen"):
        fail("case/anchor identity differs")
    if output_path.exists() or output_path.is_symlink() or output_path == Path("/"):
        fail("output must be a fresh directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames, input_probe = decode_exact81(source)
    scratch = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    published = False
    try:
        controls: dict[str, Any] = {}
        for role in CONTROL_ROLES:
            indices = control_frame_indices(role, seed=seed)
            filename = f"{role}.mp4"
            path = scratch / filename
            encode_exact81(path, [frames[index] for index in indices])
            controls[role] = {
                "filename": filename,
                "sha256": file_sha256(path),
                "frame_indices": list(indices),
                "probe": dict(probe_exact81(path)),
            }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "case_id": case_id,
            "anchor_kind": anchor_kind,
            "input": {
                "sha256": expected_sha256,
                "probe": dict(input_probe),
            },
            "shuffle": {
                "seed": seed,
                "block_size": 8,
                "block_count": 10,
                "frame_zero_retained": True,
                "block_order": list(_shuffle_block_order(seed)),
            },
            "incomplete": {
                "retained_frames": [0, 40],
                "terminal_hold_frame": 40,
            },
            "controls": controls,
            "authority": {
                "extractor_inputs_only": True,
                "generator_condition_or_training_target": False,
                "target_rgb_or_vae_to_trainer": False,
                "optimizer_created": False,
                "parameter_updates": 0,
            },
        }
        receipt["receipt_digest"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        receipt_path = scratch / "receipt.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt, pretty=True))
        if file_sha256(source) != expected_sha256:
            fail("input changed before control publication")
        os.replace(scratch, output_path)
        published = True
        return receipt
    finally:
        if not published and scratch.is_dir() and not scratch.is_symlink():
            shutil.rmtree(scratch)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--anchor-kind", choices=("target", "selfgen"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = materialize_controls(
        input_video=args.input_video,
        expected_sha256=args.input_sha256,
        output=args.output,
        case_id=args.case_id,
        anchor_kind=args.anchor_kind,
        seed=args.seed,
    )
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_ROLES",
    "Exact81VideoControlError",
    "SCHEMA_VERSION",
    "control_frame_indices",
    "materialize_controls",
]

