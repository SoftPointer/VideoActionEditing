#!/usr/bin/env python3
"""Freeze four real source-only-v3 held-out sentinels and six typed prompts.

This is the only raw/full644 boundary used by checkpoint review.  It reads an
explicit safe column projection, copies only the raw source MP4 bytes, and does
not request, resolve, hash, decode or copy any synthetic-target field.  The
four IIDs and their semantic labels are preregistered in the review contract;
they are not chosen from checkpoint quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402
import clean_source_visual_context_training_v1 as source_data  # noqa: E402


RAW_SAFE_COLUMNS = (
    "iid",
    "group_id",
    "family",
    "inputs",
    "source_video_path",
    "source_video_declared_path",
    "source_video_sha256",
    "edit_instruction_sha256",
    "selection_gates_json",
    "strict_selection_gates_all_true",
)


class ReviewAuthoringMaterializationError(RuntimeError):
    """Raised before an unsealed or target-contaminated authoring can exist."""


def fail(message: str) -> NoReturn:
    raise ReviewAuthoringMaterializationError(message)


# The captions describe the raw source, not a generated target.  Non-forward
# controls were manually preregistered without consulting target video bytes.
_MANUAL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "animal-dog-pick": {
        "source_caption": (
            "A black dog lies on a green towel and chews a red-white-blue rope "
            "toy; an antler piece rests at the bottom-left; the camera is locked."
        ),
        "controls": {
            "noop": (
                "Keep the black dog lying on the green towel, continuing the source "
                "action of chewing and lightly tugging the red-white-blue rope toy. "
                "Do not have the dog release the rope toy, reach for the antler, pick "
                "up the antler, or reposition its body. Keep the camera locked off "
                "with no panning, tilting, or zooming. Keep the dog's identity and "
                "appearance, every object, and the background unchanged."
            ),
            "reverse": (
                "Make the requested antler-transfer event unfold in reverse temporal "
                "order: the black dog first holds and chews the antler while lying on "
                "the green towel, then lowers and releases the antler at the bottom-left, "
                "turns back toward the red-white-blue rope toy, grasps that rope toy, "
                "and ends lying still with the rope toy in its mouth. Keep the camera "
                "locked off. Keep the dog's identity and appearance and the rest of "
                "the scene unchanged except for those object-contact consequences."
            ),
            "incomplete": (
                "Have the black dog slowly release the red-white-blue rope toy, lift "
                "its head, turn slightly left, and extend its front paws toward the "
                "antler at the bottom-left, but stop before its mouth grasps or lifts "
                "the antler; end with the antler still on the floor. Keep the camera "
                "locked off. Keep the dog's identity and appearance and the rest of "
                "the scene unchanged except for those partial physical consequences."
            ),
            "camera-only": (
                "Keep the black dog performing its source action of lying on the green "
                "towel and chewing the red-white-blue rope toy, without transferring "
                "to the antler. Introduce only a slow camera push-in combined with a "
                "slight pan to the left over the full clip. Keep the dog's identity, "
                "appearance and motion and every scene object otherwise unchanged."
            ),
            "appearance-only": (
                "Keep the black dog performing its source action of lying on the green "
                "towel and chewing the red-white-blue rope toy, with the camera locked "
                "off and no transfer to the antler. Change only the dog's coat from "
                "black to light cream while preserving its body shape, identity cues, "
                "motion, all objects, lighting, composition and background."
            ),
        },
    },
    "human-runner-jump": {
        "source_caption": (
            "A woman in a blue cap and blue T-shirt runs left-to-right in front "
            "of a yellow wall with a blue curve; the camera is locked."
        ),
        "controls": {
            "noop": (
                "Keep the woman in the blue cap and blue T-shirt performing the source "
                "action: run steadily from left to right across the yellow-and-blue wall. "
                "Do not have her jump, rotate, land, or stop. Keep the camera locked off. "
                "Keep her identity, clothing, body proportions, the wall graphic and the "
                "rest of the scene unchanged."
            ),
            "reverse": (
                "Make the requested run-jump-land event unfold in reverse temporal order: "
                "the woman begins in the stopped landing posture facing slightly toward "
                "the camera, steps backward, rises from the landing into the airborne "
                "pose while undoing the torso rotation, descends back into a running "
                "stride, and finishes running from right to left. Keep the camera locked "
                "off and preserve her identity, blue clothing and the wall scene."
            ),
            "incomplete": (
                "Have the woman in the blue cap and blue T-shirt run left to right, then "
                "begin the requested leap with arms raised and legs bent and reach the "
                "airborne peak, but end before she descends, lands, steps forward, or "
                "comes to a stop. Keep the camera locked off. Keep her identity and "
                "appearance and the rest of the scene unchanged."
            ),
            "camera-only": (
                "Keep the woman performing the source action of running steadily from "
                "left to right, with no jump, rotation, landing or stop. Introduce only "
                "a smooth tracking pan that follows her and a slight push-in over the "
                "clip. Keep her identity, clothing, body proportions, motion and the "
                "yellow-and-blue wall otherwise unchanged."
            ),
            "appearance-only": (
                "Keep the woman performing the source action of running steadily from "
                "left to right with the camera locked off and no jump. Change only her "
                "blue cap and blue T-shirt to bright red while preserving her identity, "
                "body, gait, shorts, shoes, lighting, wall graphic and background."
            ),
        },
    },
    "hand-object-blueprint-roll": {
        "source_caption": (
            "A black-sleeved hand enters a locked overhead desk view and draws on "
            "a flat architectural blueprint with a green marker."
        ),
        "controls": {
            "noop": (
                "Keep the architectural blueprint flat on the desk while the black-sleeved "
                "hand enters from the bottom-left and continues the source action of drawing "
                "on it with the green marker. Do not roll, fold, lift, translate or remove "
                "the blueprint. Keep the camera locked off and preserve the hand, marker, "
                "paper, desk and all background details."
            ),
            "reverse": (
                "Make the requested blueprint-roll event unfold in reverse temporal order: "
                "begin with the blueprint rolled near the center-left of the desk, have the "
                "black-sleeved hand unroll it downward toward the bottom edge until the full "
                "architectural drawing lies flat in its original position, then let the hand "
                "exit at the bottom-left. Keep the camera locked off and preserve every "
                "appearance and the rest of the desk scene."
            ),
            "incomplete": (
                "Have the black-sleeved hand enter from the bottom-left, lift the bottom edge "
                "of the architectural blueprint and roll it upward only halfway toward the "
                "center, then stop and hold the blueprint in that partially rolled state; do "
                "not complete or center the roll. Keep the camera locked off and preserve "
                "every appearance and the rest of the desk scene."
            ),
            "camera-only": (
                "Keep the blueprint flat and keep the black-sleeved hand performing the source "
                "action of drawing with the green marker; do not roll or fold the paper. "
                "Introduce only a slow diagonal camera pan from upper-right to lower-left with "
                "a slight zoom-in. Preserve the hand, marker, blueprint, desk and their motions "
                "and appearances otherwise."
            ),
            "appearance-only": (
                "Keep the blueprint flat and keep the black-sleeved hand performing the source "
                "action of drawing with the camera locked off; do not roll or fold the paper. "
                "Change only the marker body and its drawn ink from green to red while preserving "
                "the hand, drawing motion, blueprint layout, desk, lighting and background."
            ),
        },
    },
    "emitter-fireworks-explode": {
        "source_caption": (
            "A central orange-gold firework burst expands and fades above fixed tents, "
            "string lights and crowd silhouettes; the camera is locked."
        ),
        "controls": {
            "noop": (
                "Keep the central firework performing the source action at its original pace: "
                "radial trails expand, ordinary sparks drift down, and the burst naturally fades. "
                "Do not add the requested accelerated expansion, inward contraction, dense spark "
                "ring or final curtain. Keep the camera locked off and preserve the tents, string "
                "lights, crowd silhouettes, sky, color and composition."
            ),
            "reverse": (
                "Make the requested firework event unfold in reverse temporal order: begin with a "
                "faint vertical curtain of golden sparks, have the falling sparks rise and brighten "
                "into a dense ring, contract that ring into the central core, then reverse the radial "
                "burst until the light converges and dims at its origin. Keep the camera locked off "
                "and preserve the tents, string lights, crowd silhouettes and night sky."
            ),
            "incomplete": (
                "Have the central fireworks burst expand radially outward at increased velocity and "
                "intensify, then begin emitting a dense ring of golden sparks, but stop before the "
                "burst contracts inward, before the sparks form a vertical curtain and before the "
                "core fades completely. Keep the camera locked off and preserve the rest of the scene."
            ),
            "camera-only": (
                "Keep the central firework performing the source action of radial expansion and natural "
                "fading at its original pace, without the requested contraction, dense ring or curtain. "
                "Introduce only a slow upward camera tilt and gentle zoom-in. Preserve the firework's "
                "color and motion and all tents, string lights and crowd silhouettes otherwise."
            ),
            "appearance-only": (
                "Keep the central firework performing the source action of radial expansion and natural "
                "fading at its original pace with the camera locked off. Change only the firework trails "
                "and sparks from orange-gold to blue-violet while preserving their geometry, timing and "
                "motion and preserving the tents, string lights, crowd silhouettes and night sky."
            ),
        },
    },
}


def _plain_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        fail(f"{label} must be an absolute non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReviewAuthoringMaterializationError(f"{label} is unavailable") from error
    if resolved != path or not path.is_file() or path.is_symlink():
        fail(f"{label} must be one canonical plain file")
    return path


def _read_raw_rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - AUH runtime dependency
        raise ReviewAuthoringMaterializationError(
            "pyarrow is required to read the pinned raw/full644 projection"
        ) from error
    try:
        rows = pq.read_table(path, columns=list(RAW_SAFE_COLUMNS)).to_pylist()
    except Exception as error:
        raise ReviewAuthoringMaterializationError(
            "cannot read the pinned raw/full644 safe column projection"
        ) from error
    return rows


def _parse_forward_instruction(row: Mapping[str, Any], *, sentinel_id: str) -> str:
    raw_inputs = row.get("inputs")
    try:
        inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
    except json.JSONDecodeError as error:
        raise ReviewAuthoringMaterializationError(
            f"{sentinel_id} raw inputs JSON is invalid"
        ) from error
    if (
        not isinstance(inputs, list)
        or len(inputs) != 3
        or inputs[0] != {"has_loss": 0, "type": "video"}
        or not isinstance(inputs[1], Mapping)
        or set(inputs[1]) != {"has_loss", "text", "type"}
        or inputs[1].get("has_loss") != 0
        or inputs[1].get("type") != "text"
        or inputs[2] != {"has_loss": 1, "type": "video_gen"}
    ):
        fail(f"{sentinel_id} raw source/text/video_gen input closure differs")
    text = inputs[1].get("text")
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        fail(f"{sentinel_id} raw forward instruction is invalid")
    return text


def _copy_create_only(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        fail("source snapshot destination must be fresh")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def materialize_authoring(
    *,
    source_only_manifest_path: str | Path,
    expected_source_only_file_sha256: str,
    output_dir: str | Path,
    verify_source_only_files: bool = True,
) -> tuple[Path, Mapping[str, Any]]:
    source_manifest_path = _plain_file(
        source_only_manifest_path, label="source-only-v3 manifest"
    )
    observed_manifest_sha = contract.file_sha256(source_manifest_path)
    if observed_manifest_sha != expected_source_only_file_sha256:
        fail("source-only-v3 manifest file SHA differs from the caller pin")
    try:
        source_manifest = source_data.load_source_only_split_manifest(
            source_manifest_path, verify_files=verify_source_only_files
        )
    except Exception as error:
        raise ReviewAuthoringMaterializationError(str(error)) from error
    raw_path = _plain_file(source_data.PINNED_RAW_PARQUET, label="pinned raw/full644")
    if contract.file_sha256(raw_path) != source_data.PINNED_RAW_PARQUET_SHA256:
        fail("pinned raw/full644 file SHA differs")
    raw_rows = _read_raw_rows(raw_path)
    raw_by_iid = {str(row.get("iid")): row for row in raw_rows}
    if len(raw_by_iid) != source_data.FULL644_ROWS:
        fail("pinned raw/full644 IID closure differs")
    heldout_by_iid = {
        row.iid: row for row in source_manifest.rows_for_split("heldout")
    }
    if len(heldout_by_iid) != 8:
        fail("source-only-v3 heldout closure differs")

    output = Path(output_dir).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("authoring output directory must be a fresh absolute path")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink() or output.parent != parent:
        fail("authoring output parent must be a canonical plain directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    source_dir = staging / "sources"
    source_dir.mkdir(mode=0o755)
    authored_rows: list[Mapping[str, Any]] = []
    try:
        for sentinel_id in contract.SENTINEL_ORDER:
            fixed = contract.SENTINEL_IDENTITIES[sentinel_id]
            manual = _MANUAL_SPECS[sentinel_id]
            iid = str(fixed["iid"])
            split_row = heldout_by_iid.get(iid)
            raw = raw_by_iid.get(iid)
            if split_row is None or not isinstance(raw, Mapping):
                fail(f"{sentinel_id} is absent from actual heldout/raw authority")
            gates_raw = raw.get("selection_gates_json")
            try:
                gates = json.loads(gates_raw) if isinstance(gates_raw, str) else gates_raw
            except json.JSONDecodeError as error:
                raise ReviewAuthoringMaterializationError(
                    f"{sentinel_id} selection gates are invalid"
                ) from error
            if (
                split_row.split != "heldout"
                or split_row.group_id != raw.get("group_id")
                or split_row.action_family != raw.get("family")
                or split_row.action_family != fixed["action_family"]
                or split_row.source_video_sha256 != raw.get("source_video_sha256")
                or split_row.source_video_sha256 != fixed["source_video_sha256"]
                or raw.get("source_video_path") != raw.get("source_video_declared_path")
                or raw.get("strict_selection_gates_all_true") is not True
                or not isinstance(gates, Mapping)
                or gates.get("single_dynamic_actor") is not True
            ):
                fail(f"{sentinel_id} raw/source-only heldout identity differs")
            forward = _parse_forward_instruction(raw, sentinel_id=sentinel_id)
            forward_sha = hashlib.sha256(forward.encode("utf-8")).hexdigest()
            if (
                forward_sha != raw.get("edit_instruction_sha256")
                or forward_sha != fixed["forward_instruction_sha256"]
            ):
                fail(f"{sentinel_id} pinned raw forward instruction SHA differs")
            source_video = _plain_file(
                str(raw.get("source_video_path")), label=f"{sentinel_id} raw source MP4"
            )
            if contract.file_sha256(source_video) != fixed["source_video_sha256"]:
                fail(f"{sentinel_id} raw source MP4 bytes differ")
            contract._ffprobe_exact81(source_video)
            staged_video = source_dir / f"{sentinel_id}.mp4"
            _copy_create_only(source_video, staged_video)
            if contract.file_sha256(staged_video) != fixed["source_video_sha256"]:
                fail(f"{sentinel_id} copied source MP4 bytes differ")
            final_video = output / "sources" / staged_video.name
            controls = dict(manual["controls"])
            controls["forward"] = forward
            instructions = {
                branch: controls[branch] for branch in contract.TEXT_BRANCHES
            }
            authored_rows.append(
                {
                    "sentinel_id": sentinel_id,
                    "diversity_role": fixed["diversity_role"],
                    "source_entity_type": fixed["source_entity_type"],
                    "iid": iid,
                    "action_family": fixed["action_family"],
                    "source_video": str(final_video),
                    "source_video_sha256": fixed["source_video_sha256"],
                    "source_caption": manual["source_caption"],
                    "seed": fixed["seed"],
                    "wrong_owner_iid": fixed["wrong_owner_iid"],
                    "latent_shape": fixed["latent_shape"],
                    "instructions": instructions,
                }
            )
        unsigned = {
            "schema_version": contract.AUTHORING_SCHEMA,
            "authoring_id": (
                "stage-b-v3-heldout-diverse4-"
                f"{source_manifest.manifest_digest[:12]}"
            ),
            "source_only_manifest": {
                "path": str(source_manifest_path),
                "file_sha256": observed_manifest_sha,
                "manifest_digest": source_manifest.manifest_digest,
                "selected_split": "heldout",
            },
            "raw_full644": {
                "path": str(raw_path),
                "file_sha256": source_data.PINNED_RAW_PARQUET_SHA256,
                "safe_columns_read": list(RAW_SAFE_COLUMNS),
                "videos_column_read": False,
                "target_video_path_read": False,
                "target_video_bytes_read": False,
                "target_video_copied": False,
                "synthetic_target_semantics_used": False,
            },
            "sentinels": authored_rows,
            "authority": {
                "fixed_before_checkpoint_decode": True,
                "quality_based_selection": False,
                "optimizer_access": False,
                "sentinel_rule": "fixed-actual-v3-heldout-diversity-four-v1",
                "forward_instruction_authority": "pinned-raw-full644-inputs-text",
                "typed_controls_manually_preregistered": True,
                "target_video_available_to_review": False,
            },
        }
        value = {**unsigned, "authoring_digest": contract.object_sha256(unsigned)}
        contract._validate_authoring(value)
        staged_json = staging / "checkpoint_review_authoring_v2.json"
        contract.write_create_only_json(staged_json, value)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    final_json = output / "checkpoint_review_authoring_v2.json"
    observed = json.loads(final_json.read_text(encoding="utf-8"))
    contract._validate_authoring(observed)
    for row in observed["sentinels"]:
        source = _plain_file(row["source_video"], label=f"{row['sentinel_id']} snapshot")
        if contract.file_sha256(source) != row["source_video_sha256"]:
            fail("post-rename source snapshot bytes differ")
        contract._ffprobe_exact81(source)
    return final_json, observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only-manifest", required=True)
    parser.add_argument("--expected-source-only-file-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output, value = materialize_authoring(
        source_only_manifest_path=args.source_only_manifest,
        expected_source_only_file_sha256=args.expected_source_only_file_sha256,
        output_dir=args.output_dir,
        verify_source_only_files=True,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": contract.file_sha256(output),
                "authoring_digest": value["authoring_digest"],
                "sentinel_order": list(contract.SENTINEL_ORDER),
                "source_mp4_count": len(value["sentinels"]),
                "instructions_per_sentinel": len(contract.TEXT_BRANCHES),
                "target_video_bytes_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
