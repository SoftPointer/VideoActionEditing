#!/usr/bin/env python3
"""Create the deterministic ELAL-3 simulator-GT diagnostic packet.

This builder deliberately produces *toy simulator evidence*, not model outputs
or training data.  It creates exactly one two-entity C1 row and two three-
entity C2 rows.  Every row contains source, clean simulator target, an
appearance-disjoint action anchor, and five registered counterfactuals.

The packet is create-only and self-contained.  RGB videos are encoded as
81-frame/25-fps MP4 files.  Simulator annotations are stored as deterministic
gzip JSON with row-major RLE masks and a lossless sparse representation of the
dense signed-track and visibility/confidence tensors.  No numpy, Pillow, model,
tracker, network access, or pre-existing preview data is used.

The resulting authority is intentionally narrow: ``ELAL3_SIM_DIAGNOSTIC``.
It cannot authorize training, exact160, scientific promotion, or a real-video
claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "elal3-simulator-gt-canary-v1"
ANNOTATION_SCHEMA = "elal3-simulator-media-annotation-v1"
ANNOTATION_RECEIPT_SCHEMA = "elal3-simulator-annotation-receipt-v1"
BUILD_RECEIPT_SCHEMA = "elal3-simulator-gt-build-receipt-v1"
STATUS = "ELAL3_SIM_DIAGNOSTIC"
FRAME_COUNT = 81
FPS = 25
LATENT_FRAME_COUNT = 21
WIDTH = 128
HEIGHT = 96
NEGATIVE_ORDER = (
    "wrong_agent",
    "wrong_object",
    "role_swap",
    "reverse",
    "phase_shuffle",
)
MEDIA_ORDER = ("source", "target", "anchor") + NEGATIVE_ORDER
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ELAL3SimulatorBuilderError(RuntimeError):
    """Raised before an ambiguous diagnostic packet can be published."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed(value: Mapping[str, Any], digest_key: str) -> Dict[str, Any]:
    result = dict(value)
    result.pop(digest_key, None)
    result[digest_key] = object_sha256(result)
    return result


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ELAL3SimulatorBuilderError("refusing to overwrite output: %s" % path) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(path), 0o444)
    except Exception:
        try:
            os.chmod(str(path), 0o600)
            path.unlink()
        except OSError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_create_only(path, canonical_json_bytes(value) + b"\n")


def _gzip_json(value: Any) -> bytes:
    # gzip.compress(..., mtime=0) has a stable empty filename and timestamp.
    return gzip.compress(canonical_json_bytes(value) + b"\n", compresslevel=9, mtime=0)


def _fresh_output_root(value: Path) -> Path:
    requested = value.expanduser()
    if not requested.is_absolute() or requested == Path("/"):
        raise ELAL3SimulatorBuilderError("output root must be an absolute non-root path")
    if _SAFE_NAME.fullmatch(requested.name) is None:
        raise ELAL3SimulatorBuilderError("output root basename is not safe")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise ELAL3SimulatorBuilderError("output parent is unavailable") from error
    if not parent.is_dir() or parent.is_symlink():
        raise ELAL3SimulatorBuilderError("output parent must be a canonical directory")
    requested = parent / requested.name
    if requested.exists() or requested.is_symlink():
        raise ELAL3SimulatorBuilderError("refusing to overwrite output root")
    return requested


def _resolve_executable(value: str, label: str) -> Path:
    located = shutil.which(value)
    if located is None:
        raise ELAL3SimulatorBuilderError("%s executable is unavailable: %s" % (label, value))
    path = Path(located).resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode) or not os.access(str(path), os.X_OK):
        raise ELAL3SimulatorBuilderError("%s is not an executable plain file" % label)
    return path


def _version_line(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    return completed.stdout.splitlines()[0].strip()


def _lerp_keyframes(points: Sequence[Sequence[int]], frame: int) -> Tuple[int, int]:
    if frame <= int(points[0][0]):
        return int(points[0][1]), int(points[0][2])
    if frame >= int(points[-1][0]):
        return int(points[-1][1]), int(points[-1][2])
    for left, right in zip(points, points[1:]):
        f0, x0, y0 = (int(item) for item in left)
        f1, x1, y1 = (int(item) for item in right)
        if f0 <= frame <= f1:
            ratio = float(frame - f0) / float(f1 - f0)
            return int(round(x0 + ratio * (x1 - x0))), int(round(y0 + ratio * (y1 - y0)))
    raise AssertionError("unreachable keyframe interval")


def _shape_pixels(
    shape: str,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    width: int,
    height: int,
) -> set:
    pixels = set()
    x0, x1 = max(0, cx - rx), min(width - 1, cx + rx)
    y0, y1 = max(0, cy - ry), min(height - 1, cy + ry)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx, dy = x - cx, y - cy
            inside = False
            if shape in ("rect", "barrier"):
                inside = True
            elif shape == "circle":
                inside = (dx * dx * ry * ry + dy * dy * rx * rx) <= (rx * rx * ry * ry)
            elif shape == "diamond":
                inside = abs(dx) * ry + abs(dy) * rx <= rx * ry
            elif shape == "triangle":
                relative_y = y - (cy - ry)
                half = int(round(rx * relative_y / float(max(1, 2 * ry))))
                inside = abs(dx) <= half
            else:
                raise ELAL3SimulatorBuilderError("unknown simulator shape: %s" % shape)
            if inside:
                pixels.add(y * width + x)
    return pixels


def _runs(pixels: Iterable[int]) -> List[List[int]]:
    ordered = sorted(pixels)
    if not ordered:
        return []
    result = []
    start = ordered[0]
    previous = start
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append([start, previous - start + 1])
        start = previous = value
    result.append([start, previous - start + 1])
    return result


def _paint(buffer: bytearray, pixels: Iterable[int], color: Sequence[int]) -> None:
    red, green, blue = (int(item) for item in color)
    for index in pixels:
        offset = index * 3
        buffer[offset : offset + 3] = bytes((red, green, blue))


def _draw_rect_outline(
    buffer: bytearray,
    center: Sequence[int],
    size: Sequence[int],
    color: Sequence[int],
    width: int,
    height: int,
) -> None:
    cx, cy = int(center[0]), int(center[1])
    rx, ry = int(size[0]), int(size[1])
    pixels = set()
    for x in range(max(0, cx - rx), min(width - 1, cx + rx) + 1):
        for y in (max(0, cy - ry), min(height - 1, cy + ry)):
            pixels.add(y * width + x)
    for y in range(max(0, cy - ry), min(height - 1, cy + ry) + 1):
        for x in (max(0, cx - rx), min(width - 1, cx + rx)):
            pixels.add(y * width + x)
    _paint(buffer, pixels, color)


def _background(plan: Mapping[str, Any]) -> bytearray:
    base = tuple(int(item) for item in plan["background_rgb"])
    accent = tuple(int(item) for item in plan["grid_rgb"])
    buffer = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            shade = 5 if ((x // 16 + y // 16) % 2) else 0
            color = tuple(max(0, min(255, channel + shade)) for channel in base)
            offset = (y * WIDTH + x) * 3
            buffer[offset : offset + 3] = bytes(color)
    for y in range(0, HEIGHT, 16):
        _paint(buffer, (y * WIDTH + x for x in range(WIDTH)), accent)
    for marker in plan.get("markers", []):
        _draw_rect_outline(
            buffer,
            marker["center"],
            marker["size"],
            marker["color"],
            WIDTH,
            HEIGHT,
        )
    return buffer


def _phase_labels(mode: str) -> Dict[str, Any]:
    orders = {
        "none": [],
        "forward": ["onset", "transition", "terminal", "hold"],
        "reverse": ["transition", "onset", "terminal", "hold"],
        "shuffled": ["terminal", "onset", "transition", "hold"],
    }
    order = orders[mode]
    labels = []
    for phase in range(LATENT_FRAME_COUNT):
        if not order:
            labels.append([0, 0, 0, 0])
        elif phase <= 4:
            labels.append([1 if order[0] == name else 0 for name in ("onset", "transition", "terminal", "hold")])
        elif phase <= 11:
            labels.append([1 if order[1] == name else 0 for name in ("onset", "transition", "terminal", "hold")])
        elif phase <= 15:
            labels.append([1 if order[2] == name else 0 for name in ("onset", "transition", "terminal", "hold")])
        else:
            labels.append([1 if order[3] == name else 0 for name in ("onset", "transition", "terminal", "hold")])
    return {
        "shape": [LATENT_FRAME_COUNT, 4],
        "channels": ["onset", "transition", "terminal", "hold"],
        "pre_event_is_four_channel_zero_complement": True,
        "latent_phase_to_rgb_frame": [0] + [4 * value for value in range(1, LATENT_FRAME_COUNT)],
        "labels": labels,
        "semantic_order": order,
    }


def _authority() -> Dict[str, Any]:
    return {
        "status": STATUS,
        "simulator_only": True,
        "real_video_data": False,
        "model_output": False,
        "training_authorized": False,
        "training_use_forbidden": True,
        "exact160_eligible": False,
        "exact160_claim_authorized": False,
        "formal_c0_c1_c2_go_authorized": False,
        "action_encoder_qualification_authorized": False,
        "scientific_claim_authorized": False,
    }


def _common_appearance(count: int, anchor: bool = False) -> Dict[str, Any]:
    normal = [
        ("circle", [225, 65, 65], 7, 7),
        ("diamond", [65, 130, 235], 7, 7),
        ("barrier", [238, 194, 65], 6, 10),
    ]
    alternate = [
        ("triangle", [85, 215, 150], 8, 8),
        ("rect", [194, 100, 225], 7, 7),
        ("barrier", [245, 125, 55], 8, 8),
    ]
    chosen = alternate if anchor else normal
    names = ("agent", "patient", "object")[:count]
    return {
        name: {
            "shape": chosen[index][0],
            "rgb": chosen[index][1],
            "radius_xy": [chosen[index][2], chosen[index][3]],
            "simulator_instance_signature": ("anchor-" if anchor else "source-") + name,
        }
        for index, name in enumerate(names)
    }


def _plan(
    trajectories: Mapping[str, Sequence[Sequence[int]]],
    roles: Mapping[str, str],
    effect: str,
    *,
    phase_mode: str = "forward",
    anchor: bool = False,
    z_order: Sequence[str] = (),
    markers: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    count = len(trajectories)
    return {
        "trajectories": {key: [list(point) for point in value] for key, value in trajectories.items()},
        "appearance": _common_appearance(count, anchor=anchor),
        "roles": dict(roles),
        "required_effect": effect,
        "phase_mode": phase_mode,
        "z_order": list(z_order or tuple(trajectories.keys())),
        "background_rgb": [24, 42, 55] if not anchor else [55, 32, 58],
        "grid_rgb": [34, 58, 70] if not anchor else [73, 45, 75],
        "markers": [dict(item) for item in markers],
        "appearance_disjoint_from_source": bool(anchor),
    }


def _row_specs() -> List[Dict[str, Any]]:
    goal = {"center": [102, 50], "size": [9, 12], "color": [70, 205, 110]}
    decoy = {"center": [101, 22], "size": [9, 9], "color": [220, 75, 205]}
    c1_roles = {"agent": "agent", "patient": "patient_object"}
    c1_source = {"agent": [[0, 24, 50], [80, 24, 50]], "patient": [[0, 56, 50], [80, 56, 50]]}
    c1_target = {"agent": [[0, 24, 50], [16, 42, 50], [48, 78, 50], [64, 84, 50], [80, 84, 50]], "patient": [[0, 56, 50], [16, 56, 50], [48, 92, 50], [80, 92, 50]]}
    c1 = {
        "row_id": "c1-two-entity-push-to-goal",
        "gate": "C1_TWO_ENTITY_ONE_ROW_OVERFIT",
        "entity_count": 2,
        "instruction": "The designated red agent pushes the blue patient object into the green goal, then both hold the completed state.",
        "participants": [
            {"entity_id": "agent", "semantic_role": "agent", "required_effect": "approach and push patient"},
            {"entity_id": "patient", "semantic_role": "patient_object", "required_effect": "move into goal and remain"},
        ],
        "plans": {
            "source": _plan(c1_source, c1_roles, "no requested edit occurs", phase_mode="none", markers=[goal, decoy]),
            "target": _plan(c1_target, c1_roles, "patient reaches green goal after agent contact and holds", markers=[goal, decoy]),
            "anchor": _plan({"agent": [[0, 18, 32], [16, 40, 32], [48, 82, 32], [80, 82, 32]], "patient": [[0, 52, 32], [16, 52, 32], [48, 96, 32], [80, 96, 32]]}, c1_roles, "same canonical push-to-goal event in a different appearance/scene", anchor=True, markers=[{"center": [105, 32], "size": [9, 12], "color": [80, 210, 125]}]),
            "wrong_agent": _plan({"agent": [[0, 24, 50], [40, 64, 50], [80, 94, 50]], "patient": [[0, 56, 50], [24, 38, 50], [80, 38, 50]]}, {"agent": "wrong_patient", "patient": "wrong_agent"}, "patient acts on agent instead of designated agent", markers=[goal, decoy]),
            "wrong_object": _plan({"agent": [[0, 24, 50], [48, 90, 22], [80, 90, 22]], "patient": [[0, 56, 50], [80, 56, 50]]}, c1_roles, "agent goes to the magenta distractor and leaves patient unchanged", markers=[goal, decoy]),
            "role_swap": _plan({"agent": [[0, 24, 50], [80, 24, 50]], "patient": [[0, 56, 50], [48, 92, 50], [80, 92, 50]]}, {"agent": "inactive_observer", "patient": "self_propelled_agent"}, "patient self-propels; required causal binding is absent", markers=[goal, decoy]),
            "reverse": _plan({"agent": [[0, 24, 50], [32, 42, 50], [64, 18, 50], [80, 18, 50]], "patient": [[0, 56, 50], [32, 56, 50], [64, 30, 50], [80, 30, 50]]}, c1_roles, "agent moves patient away from the goal", phase_mode="reverse", markers=[goal, decoy]),
            "phase_shuffle": _plan({"agent": [[0, 24, 50], [48, 24, 50], [72, 80, 50], [80, 80, 50]], "patient": [[0, 56, 50], [20, 92, 50], [80, 92, 50]]}, c1_roles, "patient reaches terminal location before agent onset", phase_mode="shuffled", markers=[goal, decoy]),
        },
    }

    block_roles = {"agent": "agent", "patient": "patient", "object": "instrument"}
    block_source = {"agent": [[0, 24, 72], [80, 24, 72]], "patient": [[0, 14, 40], [80, 112, 40]], "object": [[0, 56, 72], [80, 56, 72]]}
    block_target = {"agent": [[0, 24, 72], [20, 44, 72], [44, 50, 48], [64, 48, 50], [80, 48, 50]], "patient": [[0, 14, 40], [32, 40, 40], [48, 47, 40], [80, 47, 40]], "object": [[0, 56, 72], [20, 56, 72], [44, 56, 40], [80, 56, 40]]}
    block = {
        "row_id": "c2-three-entity-blocking-response",
        "gate": "C2_THREE_ENTITY_ROLE_OCCLUSION",
        "entity_count": 3,
        "instruction": "The red agent moves the yellow barrier into the blue patient's path; the patient decelerates, stops behind it, and all entities hold.",
        "participants": [
            {"entity_id": "agent", "semantic_role": "agent", "required_effect": "move barrier into path"},
            {"entity_id": "patient", "semantic_role": "patient", "required_effect": "decelerate and stop before barrier"},
            {"entity_id": "object", "semantic_role": "instrument", "required_effect": "occupy path and briefly occlude patient"},
        ],
        "plans": {
            "source": _plan(block_source, block_roles, "patient passes while agent and barrier remain", phase_mode="none", z_order=["agent", "patient", "object"]),
            "target": _plan(block_target, block_roles, "barrier placement causes patient secondary stop response", z_order=["agent", "patient", "object"]),
            "anchor": _plan({"agent": [[0, 20, 78], [20, 42, 78], [44, 52, 52], [80, 52, 52]], "patient": [[0, 12, 30], [32, 42, 30], [48, 48, 30], [80, 48, 30]], "object": [[0, 60, 78], [20, 60, 78], [44, 60, 30], [80, 60, 30]]}, block_roles, "same blocking and secondary response with different appearances", anchor=True, z_order=["agent", "patient", "object"]),
            "wrong_agent": _plan({"agent": [[0, 24, 72], [80, 24, 72]], "patient": [[0, 14, 40], [32, 46, 60], [48, 54, 64], [80, 54, 64]], "object": [[0, 56, 72], [48, 56, 40], [80, 56, 40]]}, {"agent": "inactive", "patient": "wrong_agent", "object": "instrument"}, "patient rather than designated agent moves barrier", z_order=["agent", "patient", "object"]),
            "wrong_object": _plan({"agent": [[0, 24, 72], [48, 100, 76], [80, 100, 76]], "patient": [[0, 14, 40], [80, 112, 40]], "object": [[0, 56, 72], [80, 56, 72]]}, block_roles, "agent approaches a scene decoy; barrier remains out of path", markers=[decoy], z_order=["agent", "patient", "object"]),
            "role_swap": _plan({"agent": [[0, 24, 72], [48, 46, 40], [80, 46, 40]], "patient": [[0, 14, 40], [32, 48, 72], [80, 48, 72]], "object": [[0, 56, 72], [48, 56, 40], [80, 56, 40]]}, {"agent": "patient", "patient": "agent", "object": "instrument"}, "patient moves barrier and designated agent is stopped", z_order=["agent", "patient", "object"]),
            "reverse": _plan({"agent": [[0, 24, 72], [40, 48, 72], [64, 48, 86], [80, 48, 86]], "patient": [[0, 14, 40], [80, 112, 40]], "object": [[0, 56, 72], [40, 56, 72], [64, 56, 88], [80, 56, 88]]}, block_roles, "agent removes barrier away from path and patient continues", phase_mode="reverse", z_order=["agent", "patient", "object"]),
            "phase_shuffle": _plan({"agent": [[0, 24, 72], [48, 24, 72], [72, 50, 48], [80, 50, 48]], "patient": [[0, 14, 40], [24, 47, 40], [56, 92, 40], [80, 92, 40]], "object": [[0, 56, 72], [48, 56, 72], [72, 56, 40], [80, 56, 40]]}, block_roles, "patient stops and passes before barrier transition", phase_mode="shuffled", z_order=["agent", "patient", "object"]),
        },
    }

    hand_roles = {"agent": "agent", "patient": "co_agent", "object": "patient_object"}
    hand_source = {"agent": [[0, 22, 50], [80, 22, 50]], "patient": [[0, 104, 50], [80, 104, 50]], "object": [[0, 31, 50], [80, 31, 50]]}
    hand_target = {"agent": [[0, 22, 50], [40, 66, 50], [56, 76, 50], [80, 76, 50]], "patient": [[0, 104, 50], [40, 100, 50], [56, 90, 50], [80, 90, 50]], "object": [[0, 31, 50], [40, 70, 50], [56, 86, 50], [80, 86, 50]]}
    handover = {
        "row_id": "c2-three-entity-handover-occlusion",
        "gate": "C2_THREE_ENTITY_ROLE_OCCLUSION",
        "entity_count": 3,
        "instruction": "The red agent carries the yellow object to the blue co-agent, transfers it during partial occlusion, and the co-agent holds it at the terminal state.",
        "participants": [
            {"entity_id": "agent", "semantic_role": "agent", "required_effect": "carry and release object"},
            {"entity_id": "patient", "semantic_role": "co_agent", "required_effect": "receive and hold object"},
            {"entity_id": "object", "semantic_role": "patient_object", "required_effect": "change ownership agent-to-co-agent"},
        ],
        "plans": {
            "source": _plan(hand_source, hand_roles, "no transfer; all identities persist", phase_mode="none", z_order=["object", "agent", "patient"]),
            "target": _plan(hand_target, hand_roles, "object ownership transfers under occlusion and terminal hold", z_order=["object", "agent", "patient"]),
            "anchor": _plan({"agent": [[0, 18, 30], [40, 62, 30], [56, 74, 30], [80, 74, 30]], "patient": [[0, 108, 30], [40, 102, 30], [56, 90, 30], [80, 90, 30]], "object": [[0, 28, 30], [40, 68, 30], [56, 86, 30], [80, 86, 30]]}, hand_roles, "same transfer/receive/hold event with different appearances", anchor=True, z_order=["object", "agent", "patient"]),
            "wrong_agent": _plan({"agent": [[0, 22, 50], [80, 22, 50]], "patient": [[0, 104, 50], [48, 38, 50], [80, 38, 50]], "object": [[0, 31, 50], [48, 42, 50], [80, 42, 50]]}, {"agent": "inactive", "patient": "wrong_agent", "object": "patient_object"}, "co-agent takes object without designated agent transfer", z_order=["object", "agent", "patient"]),
            "wrong_object": _plan({"agent": [[0, 22, 50], [48, 96, 22], [80, 96, 22]], "patient": [[0, 104, 50], [80, 104, 50]], "object": [[0, 31, 50], [80, 31, 50]]}, hand_roles, "agent carries scene decoy while registered object remains", markers=[decoy], z_order=["object", "agent", "patient"]),
            "role_swap": _plan({"agent": [[0, 22, 50], [48, 34, 50], [80, 34, 50]], "patient": [[0, 104, 50], [48, 42, 50], [80, 42, 50]], "object": [[0, 31, 50], [48, 38, 50], [80, 38, 50]]}, {"agent": "receiver", "patient": "agent", "object": "patient_object"}, "co-agent becomes giver and required direction is swapped", z_order=["object", "agent", "patient"]),
            "reverse": _plan({"agent": [[0, 22, 50], [48, 8, 50], [80, 8, 50]], "patient": [[0, 104, 50], [80, 104, 50]], "object": [[0, 31, 50], [48, 12, 50], [80, 12, 50]]}, hand_roles, "agent carries object away from receiver", phase_mode="reverse", z_order=["object", "agent", "patient"]),
            "phase_shuffle": _plan({"agent": [[0, 22, 50], [56, 22, 50], [80, 78, 50]], "patient": [[0, 104, 50], [24, 90, 50], [80, 90, 50]], "object": [[0, 31, 50], [20, 86, 50], [80, 86, 50]]}, hand_roles, "object reaches receiver before agent onset", phase_mode="shuffled", z_order=["object", "agent", "patient"]),
        },
    }
    return [c1, block, handover]


def _encode_and_annotate(
    output_path: Path,
    ffmpeg: Path,
    row: Mapping[str, Any],
    variant: str,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "%dx%d" % (WIDTH, HEIGHT),
        "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "1",
        "-g", str(FRAME_COUNT), "-keyint_min", str(FRAME_COUNT), "-sc_threshold", "0",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1",
        "-metadata", "creation_time=1970-01-01T00:00:00Z", "-movflags", "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_records = []
    previous_centers = {}
    base = _background(plan)
    try:
        assert process.stdin is not None
        for frame_index in range(FRAME_COUNT):
            centers = {
                entity_id: _lerp_keyframes(points, frame_index)
                for entity_id, points in plan["trajectories"].items()
            }
            amodal = {}
            for entity_id, center in centers.items():
                appearance = plan["appearance"][entity_id]
                amodal[entity_id] = _shape_pixels(
                    appearance["shape"], center[0], center[1],
                    int(appearance["radius_xy"][0]), int(appearance["radius_xy"][1]),
                    WIDTH, HEIGHT,
                )
            visible = {}
            covered = set()
            for entity_id in reversed(plan["z_order"]):
                visible[entity_id] = amodal[entity_id] - covered
                covered.update(amodal[entity_id])
            rgb = bytearray(base)
            for entity_id in plan["z_order"]:
                _paint(rgb, amodal[entity_id], plan["appearance"][entity_id]["rgb"])
            process.stdin.write(bytes(rgb))
            entities = []
            for entity_id in plan["trajectories"].keys():
                center = centers[entity_id]
                previous = previous_centers.get(entity_id, center)
                visibility_fraction = float(len(visible[entity_id])) / float(max(1, len(amodal[entity_id])))
                entities.append({
                    "entity_id": entity_id,
                    "center_xy": [center[0], center[1]],
                    "amodal_mask_runs": _runs(amodal[entity_id]),
                    "visible_mask_runs": _runs(visible[entity_id]),
                    "signed_track_dxdy_from_previous_frame": [center[0] - previous[0], center[1] - previous[1]],
                    "visibility_fraction": round(visibility_fraction, 8),
                    "track_confidence": 1.0,
                })
            frame_records.append({"frame_index": frame_index, "entities": entities})
            previous_centers = centers
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        raise ELAL3SimulatorBuilderError(
            "ffmpeg failed for %s: %s" % (output_path, stderr.decode("utf-8", "replace")[-2000:])
        )
    if stdout:
        raise ELAL3SimulatorBuilderError("ffmpeg unexpectedly wrote stdout")
    os.chmod(str(output_path), 0o444)
    annotation = {
        "schema_version": ANNOTATION_SCHEMA,
        "status": STATUS,
        "row_id": row["row_id"],
        "media_variant": variant,
        "coordinate_space": "this_media_native_%dx%d_rgb_grid" % (WIDTH, HEIGHT),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "entity_count": row["entity_count"],
        "entity_order": list(plan["trajectories"].keys()),
        "instance_masks": {
            "shape": [row["entity_count"], FRAME_COUNT, HEIGHT, WIDTH],
            "encoding": "row_major_binary_rle_per_entity_frame",
            "amodal_and_visible_masks_present": True,
        },
        "signed_tracks": {
            "dense_shape": [row["entity_count"], FRAME_COUNT, HEIGHT, WIDTH, 2],
            "encoding": "amodal_mask_rle_gated_constant_signed_dxdy_per_entity_frame",
            "coordinate_order": ["dx", "dy"],
            "frame_zero_vector_is_zero": True,
        },
        "visibility_confidence": {
            "dense_shape": [row["entity_count"], FRAME_COUNT, HEIGHT, WIDTH, 2],
            "channels": ["visibility", "confidence"],
            "encoding": "visible_mask_rle_plus_amodal_support_confidence_constant",
            "confidence_inside_amodal_support": 1.0,
        },
        "frames": frame_records,
        "camera_transform": {
            "shape": [FRAME_COUNT, 3, 3],
            "coordinate": "pixel_homography_previous_to_current",
            "encoding": "constant_identity_all_frames",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "phase_labels": _phase_labels(plan["phase_mode"]),
        "roles": plan["roles"],
        "required_effect": plan["required_effect"],
        "terminal_window_rgb_frames_inclusive": [65, 80],
        "appearance": plan["appearance"],
        "appearance_disjoint_from_source": plan["appearance_disjoint_from_source"],
        "simulator_gt": True,
        "tracker_or_estimator_used": False,
        "authority": _authority(),
    }
    return annotation


def _probe_video(ffprobe: Path, path: Path) -> Dict[str, Any]:
    command = [
        str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    value = json.loads(completed.stdout)
    streams = value.get("streams", [])
    if len(streams) != 1:
        raise ELAL3SimulatorBuilderError("expected exactly one video stream: %s" % path)
    stream = streams[0]
    if (
        int(stream.get("nb_read_frames", -1)) != FRAME_COUNT
        or stream.get("r_frame_rate") != "%d/1" % FPS
        or int(stream.get("width", -1)) != WIDTH
        or int(stream.get("height", -1)) != HEIGHT
    ):
        raise ELAL3SimulatorBuilderError("video fails exact81/25fps geometry gate: %s" % path)
    return {
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": WIDTH,
        "height": HEIGHT,
        "frame_count": FRAME_COUNT,
        "fps_num": FPS,
        "fps_den": 1,
        "all_frames_decoded_by_ffprobe": True,
    }


def _media_bundle(
    root: Path,
    ffmpeg: Path,
    ffprobe: Path,
    row: Mapping[str, Any],
    variant: str,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    relative_video = Path("media") / row["row_id"] / (variant + ".mp4")
    video = root / relative_video
    annotation = _encode_and_annotate(video, ffmpeg, row, variant, plan)
    probe = _probe_video(ffprobe, video)
    video_sha = file_sha256(video)
    relative_annotation = Path("annotations") / row["row_id"] / (variant + ".annotations.json.gz")
    annotation_path = root / relative_annotation
    annotation_bytes = _gzip_json(annotation)
    _write_create_only(annotation_path, annotation_bytes)
    annotation_sha = file_sha256(annotation_path)
    relative_receipt = Path("annotations") / row["row_id"] / (variant + ".annotation-receipt.json")
    receipt_path = root / relative_receipt
    receipt = _sealed({
        "schema_version": ANNOTATION_RECEIPT_SCHEMA,
        "status": STATUS,
        "row_id": row["row_id"],
        "media_variant": variant,
        "media": {"path": relative_video.as_posix(), "sha256": video_sha, "probe": probe},
        "annotation": {
            "path": relative_annotation.as_posix(),
            "sha256": annotation_sha,
            "uncompressed_canonical_json_sha256": hashlib.sha256(canonical_json_bytes(annotation) + b"\n").hexdigest(),
            "schema_version": ANNOTATION_SCHEMA,
        },
        "extractor": "deterministic_analytic_simulator_gt_no_tracker",
        "extractor_version": SCHEMA_VERSION,
        "media_and_annotation_coordinate_spaces_are_local": True,
        "authority": _authority(),
    }, "annotation_receipt_digest")
    _write_json(receipt_path, receipt)
    return {
        "variant": variant,
        "role": "action-reference-only" if variant == "anchor" else ("clean-simulator-edited-target" if variant == "target" else variant),
        "path": relative_video.as_posix(),
        "sha256": video_sha,
        "probe": probe,
        "annotation_path": relative_annotation.as_posix(),
        "annotation_sha256": annotation_sha,
        "annotation_receipt_path": relative_receipt.as_posix(),
        "annotation_receipt_sha256": file_sha256(receipt_path),
        "annotation_receipt_digest": receipt["annotation_receipt_digest"],
        "simulator_gt": True,
    }


def _html_bytes(manifest: Mapping[str, Any]) -> bytes:
    cards = []
    labels = {
        "source": "Source",
        "target": "Clean simulator target / edited",
        "anchor": "Appearance-disjoint action anchor",
        "wrong_agent": "Negative: wrong agent",
        "wrong_object": "Negative: wrong object",
        "role_swap": "Negative: role swap",
        "reverse": "Negative: reverse",
        "phase_shuffle": "Negative: phase shuffle",
    }
    for row in manifest["rows"]:
        videos = []
        for variant in MEDIA_ORDER:
            media = row["media"][variant]
            videos.append(
                '<div class="clip"><h3>%s</h3><video controls muted loop preload="metadata" src="%s"></video>'
                '<code>%s</code></div>' % (
                    html.escape(labels[variant]), html.escape(media["path"]), html.escape(media["sha256"][:16]),
                )
            )
        cards.append(
            '<section><h2>%s</h2><p><b>%s</b></p><p>%s</p><div class="grid">%s</div></section>' % (
                html.escape(row["row_id"]), html.escape(row["gate"]), html.escape(row["instruction"]), "".join(videos),
            )
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ELAL-3 simulator diagnostic</title><style>
body{margin:0;background:#0b1117;color:#e8eef5;font:14px system-ui,sans-serif}header{padding:24px;background:#521923;border-bottom:4px solid #ff6070}header h1{margin:0 0 8px}.warning{font-weight:800;font-size:18px;color:#fff0a8}section{margin:22px;padding:18px;background:#14202b;border:1px solid #314454;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:14px;overflow-x:auto}.clip{background:#0c151d;padding:10px;border-radius:8px}.clip h3{min-height:36px;margin:0 0 8px}video{width:100%%;background:#000}code{display:block;margin-top:7px;color:#8fb9d8}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(260px,1fr))}}
</style></head><body><header><h1>ELAL-3 simulator-GT canary packet</h1><div class="warning">SIMULATOR DIAGNOSTIC ONLY · NOT REAL TRAINING RESULTS · TRAINING AND exact160 CLAIMS FORBIDDEN</div><p>These 24 videos are deterministic analytic fixtures for ABI, gradient, overfit and intervention testing. They are not generated by Bernini or any trained model.</p></header>%s</body></html>""" % "".join(cards)
    return document.encode("utf-8")


def _make_read_only_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() and not path.is_symlink():
            os.chmod(str(path), 0o444)
        elif path.is_dir() and not path.is_symlink():
            os.chmod(str(path), 0o555)
    os.chmod(str(root), 0o555)


def build_bundle(output_root: Path, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe") -> Dict[str, Any]:
    root = _fresh_output_root(output_root)
    ffmpeg = _resolve_executable(ffmpeg_bin, "ffmpeg")
    ffprobe = _resolve_executable(ffprobe_bin, "ffprobe")
    root.mkdir(mode=0o700)
    completed = False
    try:
        rows = []
        for spec in _row_specs():
            media = {}
            for variant in MEDIA_ORDER:
                media[variant] = _media_bundle(root, ffmpeg, ffprobe, spec, variant, spec["plans"][variant])
            if len({media[item]["sha256"] for item in MEDIA_ORDER}) != len(MEDIA_ORDER):
                raise ELAL3SimulatorBuilderError("a row contains byte-identical semantic variants")
            rows.append({
                "row_id": spec["row_id"],
                "gate": spec["gate"],
                "entity_count": spec["entity_count"],
                "instruction": spec["instruction"],
                "participants": spec["participants"],
                "media": media,
                "negative_order": list(NEGATIVE_ORDER),
                "terminal_hold_rgb_frames_inclusive": [65, 80],
                "formal_manifest_eligibility": "diagnostic-only-not-exact160",
            })
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "purpose": "deterministic simulator-only ELAL-3 C1/C2 implementation diagnostic",
            "create_only": True,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "latent_frame_count": LATENT_FRAME_COUNT,
            "width": WIDTH,
            "height": HEIGHT,
            "row_count": len(rows),
            "media_count": len(rows) * len(MEDIA_ORDER),
            "c1_row_count": sum(1 for row in rows if row["gate"].startswith("C1_")),
            "c2_row_count": sum(1 for row in rows if row["gate"].startswith("C2_")),
            "media_order": list(MEDIA_ORDER),
            "rows": rows,
            "review_html_path": "index.html",
            "authority": _authority(),
        }
        manifest = _sealed(manifest, "manifest_digest")
        manifest_path = root / "manifest.json"
        _write_json(manifest_path, manifest)
        html_path = root / "index.html"
        _write_create_only(html_path, _html_bytes(manifest))
        inventory = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "build-receipt.json":
                inventory.append({
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                })
        receipt = _sealed({
            "schema_version": BUILD_RECEIPT_SCHEMA,
            "status": STATUS,
            "complete": True,
            "create_only": True,
            "manifest": {"path": "manifest.json", "sha256": file_sha256(manifest_path), "digest": manifest["manifest_digest"]},
            "review_html": {"path": "index.html", "sha256": file_sha256(html_path)},
            "inventory": inventory,
            "inventory_digest": object_sha256(inventory),
            "ffmpeg": {"path": str(ffmpeg), "version_line": _version_line(ffmpeg)},
            "ffprobe": {"path": str(ffprobe), "version_line": _version_line(ffprobe)},
            "exact_row_count": 3,
            "exact_media_count": 24,
            "authority": _authority(),
        }, "build_receipt_digest")
        receipt_path = root / "build-receipt.json"
        _write_json(receipt_path, receipt)
        _make_read_only_tree(root)
        completed = True
        return {
            "output_root": str(root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "index_html_path": str(html_path),
            "index_html_sha256": file_sha256(html_path),
            "build_receipt_path": str(receipt_path),
            "build_receipt_sha256": file_sha256(receipt_path),
            "row_count": 3,
            "media_count": 24,
            "status": STATUS,
        }
    finally:
        if not completed and root.exists() and not root.is_symlink():
            shutil.rmtree(str(root))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    args = _parser().parse_args(list(argv) if argv else None)
    try:
        result = build_bundle(args.output_root, args.ffmpeg_bin, args.ffprobe_bin)
    except (ELAL3SimulatorBuilderError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print("[elal3-sim-builder] FAIL: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
