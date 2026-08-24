#!/usr/bin/env python3
"""Frozen source-only exact7 re-encode plan for BOX-EXP-014.

The plan contains exactly seven hash-bound exact-81 source MP4s.  It contains
no paired-dataset locator, list-valued posterior container, edited video, or
synthetic index-1 locator.  The eighth source (``2d2...``) is bound only as an
already-materialized external index-0 input and is never scheduled for encode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence


SCHEMA_VERSION = "bernini-full30-action-source7-reencode-plan-v1"
EXPERIMENT_ID = "BOX-EXP-014"
FRAME_COUNT = 81
FPS = 25.0
POSTERIOR_PREFIX = (1, 32, 21)
SOURCE_BASE = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
    "VideoEditing/VideoEdit_experiments/"
    "goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z/"
    "wan_next1000_v17/samples"
)
EVENT_ID = "arms-raised-to-both-hands-firmly-on-hips"
Q0_ID = "both-arms-raised-or-held-high-away-from-hips"
ANALYSIS_SPLIT = "fit"
ACTOR_KIND = "adult-human"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[0-9a-f]{16}")


class Source7ReencodePlanError(RuntimeError):
    """Raised before a widened or ambiguous source-only plan can pass."""


def fail(message: str) -> NoReturn:
    raise Source7ReencodePlanError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Source7ReencodePlanError("plan must be finite canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


_ROW_DATA: Sequence[tuple[str, str, tuple[int, ...], str, str, str]] = (
    (
        "57cda7597d924dbb",
        "6409f59896c50f0d19dff7ac1e67f37362aa57968bc64a21a9a8271f5a85fec8",
        (1, 32, 21, 68, 54),
        "fb3bfcf6924f5cf4a9de6f2ad6c48c8b6ac53a27139d0b81d4873fcf1dfd9b11",
        "adult-man-paddleboard-cap",
        "outdoor-lake-paddleboard",
    ),
    (
        "6d4a7f95a52e47e9",
        "d8ea67b2f1ada75894cd3d2b55f877fd7ffd3b0f127119288fbdec37c5925b1a",
        (1, 32, 21, 68, 54),
        "8ade2c60115591a3061017a1eaf5484c3f12be587f9bb6ba71f10ab368909ff3",
        "adult-woman-black-lace-dancer",
        "outdoor-grass-field",
    ),
    (
        "a0b66487ab68498a",
        "62dbdf50c385233087919b686a0c5d064ce1ae47170ac477f6eeb320a885afa7",
        (1, 32, 21, 82, 46),
        "e43a520e42e7321953372d28465d5e12aece253746037c9bc1e4c081474dc2ce",
        "adult-woman-blonde-hair",
        "indoor-modern-room",
    ),
    (
        "38b113317af14f01",
        "eece8b2a298a5488fb689c736ceb074842ceb61e07090aacd7ea9a34d48e2fd6",
        (1, 32, 21, 74, 50),
        "5defa9bd595abe28bad4e3941b5ad9e587c7fa2b49c9288840f8d0a1bb9cd977",
        "adult-woman-sleep-mask",
        "indoor-beige-sofa",
    ),
    (
        "5ae60e8417244e6e",
        "88d7bd4601f6f8c16e8a9d0bbdb5cb75f0c77538061e17e30095ecf1a620ea99",
        (1, 32, 21, 68, 54),
        "2861dbcfbf32e5153f375f30b9cf7e1d02b3d1dcd222768578e50c6ae53b3e02",
        "adult-woman-pigtails-glasses",
        "indoor-floor-by-sofa",
    ),
    (
        "1149c58e43e54add",
        "d24e32daf499850b33b40f13cd537c11fd8a230eda6fa121a24c33e0a79dfe7a",
        (1, 32, 21, 68, 54),
        "93d16dbf9aa20a9690c654fe9417f26156b178563ba0d7122f41060786ef0525",
        "adult-woman-white-dress-blue-stage",
        "blue-lit-stage-floor",
    ),
    (
        "a535e13301e448d7",
        "70ccabb237fd8a2a159d0cbf40fb53d21fb070c3d280d974aa28ec58d0d1130c",
        (1, 32, 21, 70, 52),
        "48b23799bc218750dcd7a9efc3fa920e9a83058bac76b15b7b48457eaadec2cd",
        "adult-woman-white-dress-ponytail",
        "outdoor-stone-fountain",
    ),
)


EXTERNAL_EXISTING_INDEX0: Mapping[str, Any] = {
    "iid": "2d2e28871a5a4856",
    "source_video_sha256": (
        "f12797b095b2108140c32e9ff0cf8ec6ff2af9c5e00dadc086d3f3abe02588d9"
    ),
    "group_id": (
        "a5b4f1766ed70b7349c91950143a33366cb2b5e20969163392cf8d1a0920d9cb"
    ),
    "source_posterior_index0_path": (
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
        "VideoEditing/VideoEdit_experiments/bernini_preservation_recovery_20260814/"
        "runs/source-only-v3-64-16-8-e2b65a33690a-r1/"
        "physical_source_posterior_index0/"
        "2d2e28871a5a4856.source-posterior-index0.pt"
    ),
    "source_posterior_index0_file_sha256": (
        "b6e54e36ca1be4c58c7925dc1a2c11b1f5a3e65443508affbc0c5d9dd6fa9dee"
    ),
    "source_posterior_tensor_raw_sha256": (
        "a65a8a7f73f071d555137e96ddcd53a61aee80704d7acc177daa42a44426f8e6"
    ),
    "expected_posterior_shape": [1, 32, 21, 82, 46],
    "external_bound_input_only": True,
    "opened_by_exact7_reencode": False,
    "reencoded": False,
}


def _source_path(iid: str) -> str:
    return str(SOURCE_BASE / iid / "samples" / iid / "source_video.mp4")


def canonical_plan() -> dict[str, Any]:
    rows = [
        {
            "iid": iid,
            "analysis_split": ANALYSIS_SPLIT,
            "event_id": EVENT_ID,
            "actor_kind": ACTOR_KIND,
            "q0_id": Q0_ID,
            "group_id": group_id,
            "actor_id": actor_id,
            "scene_id": scene_id,
            "source_video_path": _source_path(iid),
            "source_video_sha256": source_sha256,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "expected_posterior_shape": list(shape),
            "output_filename": f"{iid}.source-posterior-index0.pt",
            "vae_encode_calls": 1,
        }
        for iid, source_sha256, shape, group_id, actor_id, scene_id in _ROW_DATA
    ]
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "purpose": "materialize the seven missing source posteriors for the frozen exact8 inventory",
        "scientific_target": "close physical real-source/index0 authority without using any synthetic target",
        "learning_target": "N/A; optimizer-free source-only VAE materialization",
        "numeric_target": {
            "scheduled_source_mp4": 7,
            "vae_encode_calls": 7,
            "physical_posterior_outputs": 7,
            "external_existing_index0": 1,
            "optimizer_updates": 0,
        },
        "dataset": "preopt_r4_canary_real_source_exact8_inventory_snapshot_v1",
        "steps": "decode all 81 RGB frames, resize to the source aspect bucket, and encode once with the pinned Bernini/Wan VAE",
        "baseline": "one frozen exact8 source already has a physical index0 posterior; seven are missing",
        "core_validation": "exact shapes plus physical output reopen, file hash, tensor hash, and source MP4 pre/post hash stability",
        "action_event": {
            "analysis_split": ANALYSIS_SPLIT,
            "event_id": EVENT_ID,
            "actor_kind": ACTOR_KIND,
            "q0_id": Q0_ID,
        },
        "rows": rows,
        "external_existing_index0": dict(EXTERNAL_EXISTING_INDEX0),
        "inventory_snapshot_only": True,
        "exact8_authority_go_claimed": False,
        "teacher_cross_disjointness_pending": True,
        "source_only_reencode_from_source_video": True,
        "vae_encode_calls_per_source": 1,
        "paired_dataset_accessed": False,
        "legacy_source_target_container_opened": False,
        "synthetic_target_index1_path_read": False,
        "synthetic_target_index1_bytes_read": False,
        "synthetic_target_index1_decoded": False,
        "synthetic_target_index1_filtered_on": False,
        "synthetic_target_index1_hashed": False,
        "target_video_path_present": False,
        "target_video_accessed": False,
        "optimizer_created": False,
        "training_authorized": False,
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def validate_plan(value: Any) -> Mapping[str, Any]:
    require(type(value) is dict, "plan must be one object")
    expected = canonical_plan()
    require(value == expected, "plan differs from the frozen exact7 canonical value")
    require(value["plan_digest"] == object_sha256({k: v for k, v in value.items() if k != "plan_digest"}), "plan digest differs")
    rows = value["rows"]
    require(len(rows) == 7, "plan must contain exactly seven re-encode rows")
    require(len({row["iid"] for row in rows}) == 7, "source IIDs must be unique")
    require(len({row["group_id"] for row in rows}) == 7, "source groups must be unique")
    require(all(_IID.fullmatch(row["iid"]) for row in rows), "source IID differs")
    require(all(_SHA256.fullmatch(row["source_video_sha256"]) for row in rows), "source SHA differs")
    require(all(tuple(row["expected_posterior_shape"][:3]) == POSTERIOR_PREFIX for row in rows), "posterior prefix differs")
    require(all(row["expected_posterior_shape"][3] % 2 == 0 and row["expected_posterior_shape"][4] % 2 == 0 for row in rows), "posterior spatial shape differs")
    require(EXTERNAL_EXISTING_INDEX0["iid"] not in {row["iid"] for row in rows}, "external index0 was scheduled for re-encode")
    return value


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["emit"]:
        fail("usage: full30_action_source7_reencode_plan_v1.py emit")
    sys.stdout.buffer.write(canonical_json_bytes(validate_plan(canonical_plan())) + b"\n")
