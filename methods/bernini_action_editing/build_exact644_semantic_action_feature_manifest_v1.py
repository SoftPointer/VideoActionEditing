#!/usr/bin/env python3
"""Build the frozen exact644 source/anchor feature-extraction manifest.

This is a data-preparation controller for a representation diagnostic.  It
does not train Bernini, does not interpret the self-generated anchor as a
paired edited target, and does not authorize a scientific or production
claim.  The input release is pinned byte-for-byte because accepting a caller
supplied manifest digest would make the population self-authorizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "bernini-full644-self-generated-action-anchor-manifest-v1"
INPUT_FILE_SHA256 = "61da995eb680b9fba7ab3b7d3b6041c7b51c7e95253c74e607ddab6fdd6a61aa"
INPUT_MANIFEST_DIGEST = "96fe6188ad0f5ee72dcd89fbc018835f3f2995e45ff116f07449e863fa9b51d5"
OUTPUT_SCHEMA = "semantic-moments-action-reward-audit-v1"
ROW_COUNT = 644


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"{key} must be a non-empty built-in string")
    return value


def _sha(mapping: Mapping[str, Any], key: str) -> str:
    value = _exact_str(mapping, key)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return value


def build_manifest(source_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve(strict=True)
    actual_sha = file_sha256(source_path)
    if actual_sha != INPUT_FILE_SHA256:
        raise ValueError(f"source manifest SHA differs: {actual_sha}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("source manifest schema differs")
    if source.get("manifest_digest") != INPUT_MANIFEST_DIGEST:
        raise ValueError("source manifest embedded digest differs")
    if source.get("row_count") != ROW_COUNT:
        raise ValueError("source manifest row count differs")
    if source.get("paired_ground_truth_claimed") is not False:
        raise ValueError("source release must not claim paired ground truth")
    rows = source.get("rows")
    if type(rows) is not list or len(rows) != ROW_COUNT:
        raise ValueError("source manifest must contain exact644 rows")

    seen_iids: set[str] = set()
    seen_groups: set[str] = set()
    seen_paths: set[str] = set()
    items: list[dict[str, Any]] = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("every source row must be an object")
        iid = _exact_str(row, "iid")
        group_id = _sha(row, "group_id")
        family = _exact_str(row, "family")
        instruction_sha = _sha(row, "instruction_sha256")
        if iid in seen_iids or group_id in seen_groups:
            raise ValueError("iid and group_id must both be unique")
        seen_iids.add(iid)
        seen_groups.add(group_id)
        media = row.get("media_binding")
        if type(media) is not dict:
            raise ValueError("media_binding must be an object")
        if media.get("frame_count") != 81 or media.get("fps") != 25.0:
            raise ValueError("every media binding must be exact81 at 25 fps")
        common = {
            "iid": iid,
            "base_video_id": iid,
            "group_id": group_id,
            "family": family,
            "instruction_sha256": instruction_sha,
            "strict_selection_gates_all_true": (
                row.get("strict_selection_gates_all_true") is True
            ),
            "source_manifest_digest": INPUT_MANIFEST_DIGEST,
            "paired_ground_truth_claimed": False,
        }
        roles = (
            (
                "source",
                "exact644_source",
                "source_video_path",
                "source_video_sha256",
            ),
            (
                "action_anchor",
                "exact644_action_anchor",
                "action_anchor_video_path",
                "action_anchor_video_sha256",
            ),
        )
        for role, group, path_key, sha_key in roles:
            media_path = _exact_str(media, path_key)
            media_sha = _sha(media, sha_key)
            if not media_path.startswith("/") or media_path in seen_paths:
                raise ValueError("media paths must be unique absolute paths")
            seen_paths.add(media_path)
            items.append(
                {
                    "item_id": f"exact644:{iid}:{role}",
                    "group": group,
                    "path": media_path,
                    "sha256": media_sha,
                    "metadata": {**common, "role": role},
                }
            )

    if len(items) != 2 * ROW_COUNT or len(seen_paths) != 2 * ROW_COUNT:
        raise ValueError("output population must be exact1288 unique media")
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "created_at": "2026-08-19T00:00:00+00:00",
        "authority": "feature_mechanics_diagnostic_only",
        "formal_training_authorized": False,
        "paired_ground_truth_claimed": False,
        "source_release": {
            "path": str(source_path),
            "sha256": INPUT_FILE_SHA256,
            "manifest_digest": INPUT_MANIFEST_DIGEST,
            "row_count": ROW_COUNT,
        },
        "counts": {
            "total": 2 * ROW_COUNT,
            "unique_base_clips": ROW_COUNT,
            "by_group": {
                "exact644_action_anchor": ROW_COUNT,
                "exact644_source": ROW_COUNT,
            },
        },
        "items": items,
    }
    output["manifest_digest"] = object_sha256(output)
    return output


def write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.source_manifest)
    payload = json.dumps(
        manifest,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    write_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "manifest_digest": manifest["manifest_digest"],
                "item_count": len(manifest["items"]),
                "formal_training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
