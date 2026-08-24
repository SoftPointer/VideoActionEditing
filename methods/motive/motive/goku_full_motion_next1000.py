"""Materialize the deterministic next-1000 full-motion candidate manifest.

This is a deliberately narrow, create-only projection of the frozen Goku
source prefilter.  It is intended for the production expansion that follows
the exact-128 v16 run:

* the authoritative prefilter must contain exactly 1,235 selected rows;
* the exclusion manifest must contain exactly the 128 already-used IIDs;
* every exclusion row must be byte-identical to its parent JSONL row;
* the remaining 1,107 rows are ordered by the parent's ``selection_rank``;
* the first 1,000 rows are copied byte-for-byte into ``candidates.jsonl``.

The materializer revalidates the complete parent prefilter closure, hashes the
actual source video and lossless anchor for every emitted row, and enforces the
fixed 81-frame/25-fps/704-short-side/dynamic-object/no-cut contract.  It never
rewrites an existing output directory.  ``done.json`` is the terminal commit
marker and binds both authoritative input SHA-256 digests and both preceding
output artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .goku_full_motion_prepare import (
    PREFILTER_SCHEMA,
    FullMotionPrepareError,
    _object_digest,
    _sha256_bytes,
    _sha256_field,
    _strict_jsonl,
    _validate_prefilter_closure,
)


SUMMARY_SCHEMA = "motive-goku-full-motion-next1000-v1"
DONE_SCHEMA = "motive-goku-full-motion-next1000-done-v1"
CANDIDATES_NAME = "candidates.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (CANDIDATES_NAME, SUMMARY_NAME, DONE_NAME)

EXPECTED_PARENT_ROWS = 1_235
EXPECTED_EXCLUDED_ROWS = 128
EXPECTED_REMAINING_ROWS = 1_107
OUTPUT_ROWS = 1_000
EXPECTED_TAIL_ROWS = 107

# Frozen identities documented for the current scale512/v16 expansion.  They
# remain CLI defaults, while callers/tests may explicitly bind another byte-
# identical-shape fixture or a future deliberately-versioned parent.
CURRENT_PARENT_SELECTED_SHA256 = (
    "ed828b935526803c39ac9d679603b274f7d98ac081203b54f6b2b3ba07ff747a"
)
CURRENT_EXACT128_SHA256 = (
    "834e5a70e7c87683730ac644ce233b9343e4fc98eb3b3a45f55f93c8da94688d"
)


class Next1000MaterializeError(RuntimeError):
    """The authoritative inputs or fixed next-1000 projection are invalid."""


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _plain_absolute_file(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Next1000MaterializeError(f"{context} must be non-empty text")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise Next1000MaterializeError(f"{context} must be absolute: {raw}")
    if raw.is_symlink():
        raise Next1000MaterializeError(f"{context} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise Next1000MaterializeError(
            f"{context} cannot be resolved: {raw}"
        ) from error
    if not resolved.is_file():
        raise Next1000MaterializeError(
            f"{context} must resolve to a regular file: {raw}"
        )
    return resolved


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Next1000MaterializeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Next1000MaterializeError(f"{context} must be finite")
    return result


def _validate_parent_row(row: Mapping[str, Any]) -> tuple[str, str, int]:
    iid = row.get("iid")
    if not isinstance(iid, str):  # strict JSONL already checks safe spelling.
        raise Next1000MaterializeError("parent row IID is invalid")
    if row.get("schema_version") != PREFILTER_SCHEMA:
        raise Next1000MaterializeError(
            f"iid={iid} parent prefilter schema differs"
        )
    if row.get("eligible") is not True or row.get("selected") is not True:
        raise Next1000MaterializeError(
            f"iid={iid} is not a selected eligible parent row"
        )
    group_id = row.get("group_id")
    if (
        not isinstance(group_id, str)
        or not group_id
        or group_id != group_id.strip()
        or "\x00" in group_id
    ):
        raise Next1000MaterializeError(f"iid={iid} group_id is invalid")
    rank = row.get("selection_rank")
    if type(rank) is not int or rank <= 0:
        raise Next1000MaterializeError(
            f"iid={iid} selection_rank must be a positive integer"
        )
    try:
        _sha256_field(
            row.get("source_video_sha256"),
            context=f"iid={iid} source video SHA",
        )
        _sha256_field(
            row.get("anchor_sha256"),
            context=f"iid={iid} anchor SHA",
        )
    except FullMotionPrepareError as error:
        raise Next1000MaterializeError(str(error)) from error
    return iid, group_id, rank


def _validate_candidate_geometry(row: Mapping[str, Any]) -> None:
    iid = str(row["iid"])
    media = row.get("media")
    motion = row.get("motion")
    if not isinstance(media, Mapping) or not isinstance(motion, Mapping):
        raise Next1000MaterializeError(
            f"iid={iid} media and motion must be objects"
        )
    if type(media.get("frame_count")) is not int or media["frame_count"] != 81:
        raise Next1000MaterializeError(f"iid={iid} frame_count must equal 81")
    fps = _finite_number(media.get("fps"), context=f"iid={iid} media.fps")
    if fps != 25.0:
        raise Next1000MaterializeError(f"iid={iid} fps must equal 25")
    width = media.get("width")
    height = media.get("height")
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or min(width, height) != 704
    ):
        raise Next1000MaterializeError(
            f"iid={iid} source width/height must have short side 704"
        )
    if type(media.get("short_side")) is not int or media["short_side"] != 704:
        raise Next1000MaterializeError(
            f"iid={iid} short_side must equal 704"
        )
    if motion.get("label") != "dynamic_object":
        raise Next1000MaterializeError(
            f"iid={iid} motion.label must equal dynamic_object"
        )
    scene_cut = _finite_number(
        motion.get("scene_cut_ratio"),
        context=f"iid={iid} motion.scene_cut_ratio",
    )
    if scene_cut != 0.0:
        raise Next1000MaterializeError(
            f"iid={iid} scene_cut_ratio must equal zero"
        )


def _stable_hash(path: Path, *, context: str) -> tuple[str, os.stat_result]:
    before = path.stat()
    digest = _file_sha256(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise Next1000MaterializeError(f"{context} changed while hashing")
    return digest, after


def _verify_candidate_media(
    item: tuple[Mapping[str, Any], Path],
) -> dict[str, Any]:
    row, prefilter_root = item
    iid = str(row["iid"])
    source = _plain_absolute_file(
        row.get("resolved_src_video"),
        context=f"iid={iid} resolved source video",
    )
    source_digest, source_stat = _stable_hash(
        source,
        context=f"iid={iid} source video",
    )
    expected_source_digest = str(row["source_video_sha256"])
    if source_digest != expected_source_digest:
        raise Next1000MaterializeError(
            f"iid={iid} source video SHA-256 differs"
        )

    media = row["media"]
    expected_size = media.get("file_size_bytes")
    expected_mtime = media.get("mtime_ns_at_analysis")
    if type(expected_size) is not int or expected_size != source_stat.st_size:
        raise Next1000MaterializeError(
            f"iid={iid} source file size differs from analyzed metadata"
        )
    if type(expected_mtime) is not int or expected_mtime != source_stat.st_mtime_ns:
        raise Next1000MaterializeError(
            f"iid={iid} source mtime differs from analyzed metadata"
        )

    relative_anchor = row.get("anchor_image")
    if not isinstance(relative_anchor, str) or not relative_anchor.startswith(
        "anchors/"
    ):
        raise Next1000MaterializeError(
            f"iid={iid} anchor_image is not canonical"
        )
    parent_anchor_raw = prefilter_root / relative_anchor
    if parent_anchor_raw.is_symlink():
        raise Next1000MaterializeError(
            f"iid={iid} parent anchor must not be a symlink"
        )
    try:
        parent_anchor = parent_anchor_raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise Next1000MaterializeError(
            f"iid={iid} parent anchor cannot be resolved"
        ) from error
    resolved_anchor = _plain_absolute_file(
        row.get("resolved_anchor_image"),
        context=f"iid={iid} resolved anchor image",
    )
    if resolved_anchor != parent_anchor:
        raise Next1000MaterializeError(
            f"iid={iid} resolved anchor does not name its parent anchor"
        )
    anchor_digest, _ = _stable_hash(
        resolved_anchor,
        context=f"iid={iid} anchor",
    )
    if anchor_digest != row["anchor_sha256"]:
        raise Next1000MaterializeError(f"iid={iid} anchor SHA-256 differs")
    return {
        "iid": iid,
        "source_path": str(source),
        "source_sha256": source_digest,
        "anchor_path": str(resolved_anchor),
        "anchor_sha256": anchor_digest,
    }


def _implementation_sha256() -> str:
    return _file_sha256(Path(__file__).resolve(strict=True))


def _publish_create_only(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if set(files) != set(OUTPUT_NAMES):
        raise RuntimeError("internal output artifact set differs")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"create-only output already exists: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise Next1000MaterializeError(
            f"output parent must be a plain directory: {parent}"
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(parent)
        )
    )
    try:
        # done.json is written last even inside the unpublished staging tree.
        for name in OUTPUT_NAMES:
            path = stage / name
            with path.open("xb") as handle:
                handle.write(files[name])
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"create-only output appeared during publication: {output_dir}"
            )
        os.rename(stage, output_dir)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_next1000_candidates(
    *,
    prefilter_dir: str | Path,
    exclude_manifest: str | Path,
    output_dir: str | Path,
    expected_parent_selected_sha256: str = CURRENT_PARENT_SELECTED_SHA256,
    expected_exclude_manifest_sha256: str = CURRENT_EXACT128_SHA256,
    hash_workers: int = 8,
) -> dict[str, Any]:
    """Validate and atomically publish the fixed next-1000 candidate set."""

    try:
        parent_expected = _sha256_field(
            expected_parent_selected_sha256,
            context="expected parent selected SHA",
        )
        exclude_expected = _sha256_field(
            expected_exclude_manifest_sha256,
            context="expected exclusion manifest SHA",
        )
    except FullMotionPrepareError as error:
        raise Next1000MaterializeError(str(error)) from error
    if type(hash_workers) is not int or hash_workers <= 0:
        raise Next1000MaterializeError("hash_workers must be positive")

    raw_target = Path(output_dir).expanduser()
    if raw_target.is_symlink():
        raise FileExistsError(
            f"create-only output must not be a symlink: {raw_target}"
        )
    target = raw_target.resolve(strict=False)
    if os.path.lexists(target):
        raise FileExistsError(f"create-only output already exists: {target}")
    try:
        root = Path(prefilter_dir).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise Next1000MaterializeError("prefilter directory does not exist") from error
    if Path(prefilter_dir).expanduser().is_symlink() or not root.is_dir():
        raise Next1000MaterializeError(
            "prefilter directory must be a non-symlink directory"
        )
    if target == root or root in target.parents:
        raise Next1000MaterializeError(
            "output directory cannot be the prefilter or one of its descendants"
        )
    implementation_sha256 = _implementation_sha256()

    try:
        parent_rows, parent_lines, parent_raw, parent_summary = (
            _validate_prefilter_closure(root)
        )
    except (FullMotionPrepareError, OSError) as error:
        raise Next1000MaterializeError(
            f"parent prefilter closure is invalid: {error}"
        ) from error
    parent_sha256 = _sha256_bytes(parent_raw)
    if parent_sha256 != parent_expected:
        raise Next1000MaterializeError(
            "parent selected.jsonl SHA-256 differs from the required binding"
        )
    if len(parent_rows) != EXPECTED_PARENT_ROWS:
        raise Next1000MaterializeError(
            f"parent selected row count differs: "
            f"{len(parent_rows)}/{EXPECTED_PARENT_ROWS}"
        )
    parent_summary_raw = (root / "summary.json").read_bytes()
    parent_done_raw = (root / "done.json").read_bytes()

    parent_meta = [_validate_parent_row(row) for row in parent_rows]
    parent_iids = [item[0] for item in parent_meta]
    parent_groups = [item[1] for item in parent_meta]
    parent_ranks = [item[2] for item in parent_meta]
    if len(set(parent_groups)) != EXPECTED_PARENT_ROWS:
        raise Next1000MaterializeError(
            "parent selected group_id values are not unique"
        )
    expected_ranks = list(range(1, EXPECTED_PARENT_ROWS + 1))
    if sorted(parent_ranks) != expected_ranks:
        raise Next1000MaterializeError(
            "parent selection_rank values are not exactly 1..1235"
        )
    if parent_ranks != expected_ranks:
        raise Next1000MaterializeError(
            "parent selected.jsonl is not ordered by selection_rank"
        )
    parent_by_iid = {
        iid: row for iid, row in zip(parent_iids, parent_rows, strict=True)
    }

    try:
        exclusion_rows, exclusion_lines, exclusion_raw = _strict_jsonl(
            Path(exclude_manifest),
            context="exact128 exclusion manifest",
        )
    except (FullMotionPrepareError, OSError) as error:
        raise Next1000MaterializeError(
            f"exclusion manifest is invalid: {error}"
        ) from error
    exclusion_sha256 = _sha256_bytes(exclusion_raw)
    if exclusion_sha256 != exclude_expected:
        raise Next1000MaterializeError(
            "exclusion manifest SHA-256 differs from the required binding"
        )
    if len(exclusion_rows) != EXPECTED_EXCLUDED_ROWS:
        raise Next1000MaterializeError(
            f"exclusion row count differs: "
            f"{len(exclusion_rows)}/{EXPECTED_EXCLUDED_ROWS}"
        )
    exclusion_iids = [str(row["iid"]) for row in exclusion_rows]
    exclusion_groups: list[str] = []
    for iid, exclusion_row in zip(exclusion_iids, exclusion_rows, strict=True):
        parent_row = parent_by_iid.get(iid)
        if parent_row is None:
            raise Next1000MaterializeError(
                f"excluded IID is absent from parent selected.jsonl: {iid}"
            )
        if exclusion_lines[iid] != parent_lines[iid]:
            raise Next1000MaterializeError(
                f"excluded row is not byte-identical to parent row: {iid}"
            )
        _, group_id, _ = _validate_parent_row(exclusion_row)
        exclusion_groups.append(group_id)
    if len(set(exclusion_groups)) != EXPECTED_EXCLUDED_ROWS:
        raise Next1000MaterializeError(
            "exclusion manifest group_id values are not unique"
        )

    excluded = set(exclusion_iids)
    remaining = [row for row in parent_rows if str(row["iid"]) not in excluded]
    if len(remaining) != EXPECTED_REMAINING_ROWS:
        raise Next1000MaterializeError(
            f"post-exclusion row count differs: "
            f"{len(remaining)}/{EXPECTED_REMAINING_ROWS}"
        )
    # Parent order is already proven to be exact selection_rank order.
    chosen = remaining[:OUTPUT_ROWS]
    tail = remaining[OUTPUT_ROWS:]
    if len(chosen) != OUTPUT_ROWS or len(tail) != EXPECTED_TAIL_ROWS:
        raise Next1000MaterializeError("next1000/tail conservation differs")

    chosen_iids: list[str] = []
    chosen_groups: list[str] = []
    chosen_ranks: list[int] = []
    for row in chosen:
        iid, group_id, rank = _validate_parent_row(row)
        _validate_candidate_geometry(row)
        chosen_iids.append(iid)
        chosen_groups.append(group_id)
        chosen_ranks.append(rank)
    if len(set(chosen_iids)) != OUTPUT_ROWS:
        raise Next1000MaterializeError("candidate IID values are not unique")
    if len(set(chosen_groups)) != OUTPUT_ROWS:
        raise Next1000MaterializeError(
            "candidate group_id values are not unique"
        )
    if chosen_ranks != sorted(chosen_ranks):
        raise Next1000MaterializeError(
            "candidate rows are not ordered by parent selection_rank"
        )

    work = [(row, root) for row in chosen]
    if hash_workers == 1:
        media_evidence = [_verify_candidate_media(item) for item in work]
    else:
        with ThreadPoolExecutor(max_workers=hash_workers) as executor:
            media_evidence = list(executor.map(_verify_candidate_media, work))
    if [record["iid"] for record in media_evidence] != chosen_iids:
        raise RuntimeError("internal media verification order differs")

    # Revalidate both authoritative input closures after the potentially long
    # source hashing pass.  This prevents publishing a mixture of two parent
    # or exclusion versions if either path was replaced concurrently.
    try:
        _, _, final_parent_raw, _ = _validate_prefilter_closure(root)
        _, _, final_exclusion_raw = _strict_jsonl(
            Path(exclude_manifest),
            context="exact128 exclusion manifest final stability check",
        )
    except (FullMotionPrepareError, OSError) as error:
        raise Next1000MaterializeError(
            f"authoritative input changed or became invalid: {error}"
        ) from error
    if (
        final_parent_raw != parent_raw
        or (root / "summary.json").read_bytes() != parent_summary_raw
        or (root / "done.json").read_bytes() != parent_done_raw
    ):
        raise Next1000MaterializeError(
            "parent prefilter changed during materialization"
        )
    if final_exclusion_raw != exclusion_raw:
        raise Next1000MaterializeError(
            "exclusion manifest changed during materialization"
        )
    if _implementation_sha256() != implementation_sha256:
        raise Next1000MaterializeError(
            "materializer implementation changed during execution"
        )

    candidate_raw = b"".join(parent_lines[iid] for iid in chosen_iids)
    candidate_sha256 = _sha256_bytes(candidate_raw)
    exclusion_iid_digest = _object_digest(exclusion_iids)
    candidate_iid_digest = _object_digest(chosen_iids)
    tail_iids = [str(row["iid"]) for row in tail]
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "semantics": {
            "source_media_only": True,
            "legacy_qwen_decisions_reused": False,
            "legacy_instructions_authoritative": False,
            "training_eligible": False,
            "next_stage": "qwen3_vl_32b_full_motion_audit",
        },
        "inputs": {
            "parent_prefilter_dir": str(root),
            "parent_selected_path": str(root / "selected.jsonl"),
            "parent_selected_rows": len(parent_rows),
            "parent_selected_sha256": parent_sha256,
            "parent_summary_sha256": _sha256_bytes(parent_summary_raw),
            "parent_done_sha256": _sha256_bytes(parent_done_raw),
            "parent_config": parent_summary.get("config"),
            "exclude_manifest_path": str(
                Path(exclude_manifest).expanduser().resolve(strict=True)
            ),
            "exclude_manifest_rows": len(exclusion_rows),
            "exclude_manifest_sha256": exclusion_sha256,
            "exclude_iid_digest": exclusion_iid_digest,
        },
        "selection": {
            "policy": "exclude_exact128_then_parent_selection_rank_ascending",
            "parent_rows": EXPECTED_PARENT_ROWS,
            "excluded_rows": EXPECTED_EXCLUDED_ROWS,
            "remaining_rows": EXPECTED_REMAINING_ROWS,
            "output_rows": OUTPUT_ROWS,
            "unselected_tail_rows": EXPECTED_TAIL_ROWS,
            "candidate_first_parent_rank": chosen_ranks[0],
            "candidate_last_parent_rank": chosen_ranks[-1],
            "candidate_iids": chosen_iids,
            "candidate_iid_digest": candidate_iid_digest,
            "unselected_tail_iid_digest": _object_digest(tail_iids),
            "parent_row_bytes_preserved": True,
            "iid_unique": True,
            "group_id_unique": True,
        },
        "fixed_contract": {
            "frame_count": 81,
            "fps": 25.0,
            "short_side": 704,
            "motion_label": "dynamic_object",
            "scene_cut_ratio": 0.0,
            "source_sha256_verified": OUTPUT_ROWS,
            "anchor_sha256_verified": OUTPUT_ROWS,
            "source_stat_matches_analysis": OUTPUT_ROWS,
            "resolved_anchor_matches_parent": OUTPUT_ROWS,
        },
        "implementation": {
            "module": "motive.goku_full_motion_next1000",
            "sha256": implementation_sha256,
        },
        "output": {
            "name": CANDIDATES_NAME,
            "rows": OUTPUT_ROWS,
            "bytes": len(candidate_raw),
            "sha256": candidate_sha256,
            "order": "parent_selection_rank_ascending_after_exclusion",
            "row_encoding": "verbatim_parent_selected_jsonl_lines",
        },
    }
    summary_raw = _pretty_bytes(summary)
    summary_sha256 = _sha256_bytes(summary_raw)
    output_sha256 = {
        CANDIDATES_NAME: candidate_sha256,
        SUMMARY_NAME: summary_sha256,
    }
    input_bindings = {
        "parent_selected_sha256": parent_sha256,
        "exclude_manifest_sha256": exclusion_sha256,
    }
    done: dict[str, Any] = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "input_bindings": input_bindings,
        "input_binding_digest": _object_digest(input_bindings),
        "counts": {
            "parent": EXPECTED_PARENT_ROWS,
            "excluded": EXPECTED_EXCLUDED_ROWS,
            "remaining": EXPECTED_REMAINING_ROWS,
            "output": OUTPUT_ROWS,
            "tail": EXPECTED_TAIL_ROWS,
        },
        "candidate_iid_digest": candidate_iid_digest,
        "implementation_sha256": implementation_sha256,
        "output_sha256": output_sha256,
        "artifact_digest": _object_digest(output_sha256),
        "training_eligible": False,
    }
    files = {
        CANDIDATES_NAME: candidate_raw,
        SUMMARY_NAME: summary_raw,
        DONE_NAME: _pretty_bytes(done),
    }
    _publish_create_only(target, files)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed next-1000 Goku source-only candidate manifest "
            "after excluding the exact-128 v16 manifest."
        )
    )
    parser.add_argument("--prefilter-dir", required=True, type=Path)
    parser.add_argument("--exclude-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-parent-selected-sha256",
        default=CURRENT_PARENT_SELECTED_SHA256,
    )
    parser.add_argument(
        "--expected-exclude-manifest-sha256",
        default=CURRENT_EXACT128_SHA256,
    )
    parser.add_argument("--hash-workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = materialize_next1000_candidates(
        prefilter_dir=args.prefilter_dir,
        exclude_manifest=args.exclude_manifest,
        output_dir=args.output_dir,
        expected_parent_selected_sha256=(
            args.expected_parent_selected_sha256
        ),
        expected_exclude_manifest_sha256=(
            args.expected_exclude_manifest_sha256
        ),
        hash_workers=args.hash_workers,
    )
    print(
        "[goku-full-motion-next1000] "
        f"rows={summary['output']['rows']} "
        f"sha256={summary['output']['sha256']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
