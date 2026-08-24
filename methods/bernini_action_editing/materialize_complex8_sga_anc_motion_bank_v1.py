#!/usr/bin/env python3
"""Seal a motion-only SGA/ANC anchor bank for Complex8 training.

The bank carries only dense RAFT feature bundles.  A donor video path is used
to authenticate event/variant provenance while authoring, but neither donor
RGB nor its VAE latent is written into the optimizer manifest.  Every target
variant receives the other three appearances of the same action event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-complex8-sga-anc-motion-bank-v1"
PAIR_SCHEMA = "bernini-same-video-motion-pairs-v1"


class MotionBankError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise MotionBankError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pair-manifest", required=True)
    flow = value.add_mutually_exclusive_group(required=True)
    flow.add_argument(
        "--flow-root",
        help="Root containing IID__anchor-vN.safetensors explicit bank files.",
    )
    flow.add_argument(
        "--rotating-cross-flow-root",
        help=(
            "Reuse an exact flow release where eXX-vN stores donor v(N+1 mod 4). "
            "This is valid because explicit latent geometry makes extraction "
            "depend only on the anchor frames, not source pixels."
        ),
    )
    value.add_argument("--output", required=True)
    return value


def load_pair_manifest(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    stored = value.pop("manifest_digest", None)
    rows = value.get("rows")
    if (
        value.get("schema_version") != PAIR_SCHEMA
        or object_sha(value) != stored
        or not isinstance(rows, list)
        or len(rows) != 32
    ):
        fail("pair manifest closure differs")
    return {**value, "manifest_digest": stored}, rows


def materialize(
    *,
    pair_manifest_path: Path,
    flow_root: Optional[Path],
    rotating_cross_flow_root: Optional[Path],
    output: Path,
) -> Mapping[str, Any]:
    pair_manifest_path = pair_manifest_path.expanduser().resolve(strict=True)
    if bool(flow_root) == bool(rotating_cross_flow_root):
        fail("exactly one flow-root mode is required")
    explicit_root = flow_root.expanduser().resolve(strict=True) if flow_root else None
    rotating_root = (
        rotating_cross_flow_root.expanduser().resolve(strict=True)
        if rotating_cross_flow_root else None
    )
    output = output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        fail("output must be fresh")
    pair_manifest, pair_rows = load_pair_manifest(pair_manifest_path)
    rows = []
    seen: set[str] = set()
    for pair in pair_rows:
        iid = pair.get("iid")
        event_id = pair.get("event_id")
        event_ordinal = pair.get("event_ordinal")
        variant_id = pair.get("variant_id")
        variant_index = pair.get("variant_index")
        if (
            not isinstance(iid, str)
            or iid in seen
            or not isinstance(event_id, str)
            or type(event_ordinal) is not int
            or event_ordinal not in range(8)
            or variant_id != f"v{variant_index}"
            or type(variant_index) is not int
            or variant_index not in range(4)
            or pair.get("target_is_self_generated_action_video") is not True
            or pair.get("same_actor_world_target") is not True
        ):
            fail("pair row identity differs")
        candidates = []
        for offset in (1, 2, 3):
            donor_index = (variant_index + offset) % 4
            donor_id = f"v{donor_index}"
            if explicit_root is not None:
                flow = (
                    explicit_root / f"{iid}__anchor-{donor_id}.safetensors"
                ).resolve(strict=True)
            else:
                # The legacy exact-cross extraction chose donor N+1 for row N.
                predecessor = (donor_index - 1) % 4
                flow = (
                    rotating_root
                    / f"e{event_ordinal:02d}-v{predecessor}.safetensors"
                ).resolve(strict=True)
                sidecar = flow.with_suffix(".json")
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                anchor_path = Path(str(metadata.get("anchor", "")))
                if (
                    metadata.get("schema_version")
                    != "bernini-anchor-raft-flow-bundle-v1"
                    or metadata.get("latent_geometry_authority")
                    != "explicit_target_clean_latent"
                    or anchor_path.parent.name != donor_id
                    or metadata.get("anchor_sha256") is None
                ):
                    fail("rotating exact-cross anchor provenance differs")
            candidates.append(
                {
                    "event_id": event_id,
                    "variant_id": donor_id,
                    "appearance_matches_target": False,
                    "rgb_or_vae_latent_in_payload": False,
                    "flow_bundle": str(flow),
                    "flow_bundle_sha256": file_sha256(flow),
                }
            )
        rows.append(
            {
                "iid": iid,
                "event_id": event_id,
                "target_variant_id": variant_id,
                "candidates": candidates,
            }
        )
        seen.add(iid)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "pair_manifest": str(pair_manifest_path),
        "pair_manifest_sha256": file_sha256(pair_manifest_path),
        "pair_manifest_digest": pair_manifest["manifest_digest"],
        "candidate_payload": "dense_flow_12d_only",
        "candidate_count_per_target": 3,
        "candidate_selection": "all_other_variants_same_event",
        "flow_binding_mode": (
            "explicit_iid_anchor_files"
            if explicit_root is not None
            else "reuse_rotating_exact_cross_anchor_only_flow"
        ),
        "anchor_rgb_or_vae_latent_used_by_model": False,
        "anchor_endpoint_used_as_target": False,
        "all_candidates_share_source_owned_target": True,
        "candidate_events_match_target_event": True,
        "candidate_appearances_cross_target": True,
        "qwen_or_vlm_used": False,
        "rows": rows,
    }
    value = {**unsigned, "manifest_digest": object_sha(unsigned)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    materialize(
        pair_manifest_path=Path(args.pair_manifest),
        flow_root=Path(args.flow_root) if args.flow_root else None,
        rotating_cross_flow_root=(
            Path(args.rotating_cross_flow_root)
            if args.rotating_cross_flow_root else None
        ),
        output=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
