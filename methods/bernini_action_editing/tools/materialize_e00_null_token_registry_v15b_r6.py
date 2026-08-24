#!/usr/bin/env python3
"""Print the deterministic E00 r6 64-null registry to stdout.

This is a read-only materializer: it loads the pinned tokenizer and v15b role
asset, computes the exact registry, and writes no file.  Capture stdout and
review it before installing it as the create-only static sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import source_owned_role_locator_v15 as locator  # noqa: E402
import source_owned_role_locator_v15b_e00_asset as role_asset  # noqa: E402


SCHEMA_VERSION = "bernini-e00-null-token-span-registry-v15b-r6"
SPAN_COUNT = 64
REGISTRATION_LENGTHS = (1, 2, 3, 4)


def _fields(value: Mapping[str, Any]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    ids = value["input_ids"]
    mask = value["attention_mask"]
    offsets = value["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        mask = mask[0]
        offsets = offsets[0]
    return list(ids), list(mask), [tuple(item) for item in offsets]


def build_registry(tokenizer: Any) -> Mapping[str, Any]:
    spec, _raw = role_asset.load_e00_v15b_asset()
    encoded = tokenizer(
        spec.model_text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    ids, mask, offsets = _fields(encoded)
    active = sum(mask)
    if (
        locator.object_sha256(ids)
        != "29c64e1005bc625c64a194d7056c6c1d9b15b78bb994c14793d46fa71d00983e"
        or locator.object_sha256(mask)
        != "86c7129afa1c6cc35f5104ce2bf534dff382c6a6e3c2c063b284ed328bcd14a3"
        or active != 56
        or len(ids) != len(mask)
        or len(ids) != len(offsets)
    ):
        raise RuntimeError("pinned E00 tokenization differs")
    occupied = set()
    for role in spec.roles:
        occupied.update(range(role.token_start, role.token_end))
    eligible = {
        index
        for index in range(active)
        if mask[index] == 1
        and offsets[index][1] > offsets[index][0]
        and index not in occupied
    }
    candidates: list[dict[str, Any]] = []
    for length in REGISTRATION_LENGTHS:
        for start in range(active - length + 1):
            stop = start + length
            if any(index not in eligible for index in range(start, stop)):
                continue
            char_start = offsets[start][0]
            char_end = offsets[stop - 1][1]
            if char_start < 0 or char_end <= char_start or char_end > len(spec.model_text):
                raise RuntimeError("eligible token offset differs")
            token_ids = ids[start:stop]
            body = {
                "index": len(candidates),
                "token_start": start,
                "token_end": stop,
                "token_ids": token_ids,
                "token_ids_sha256": locator.object_sha256(token_ids),
                "char_start": char_start,
                "char_end": char_end,
                "text": spec.model_text[char_start:char_end],
                "text_sha256": hashlib.sha256(
                    spec.model_text[char_start:char_end].encode("utf-8")
                ).hexdigest(),
            }
            candidates.append({**body, "span_sha256": locator.object_sha256(body)})
            if len(candidates) == SPAN_COUNT:
                break
        if len(candidates) == SPAN_COUNT:
            break
    if len(candidates) != SPAN_COUNT:
        raise RuntimeError("fewer than 64 eligible deterministic null spans")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "observer_only_null_distribution_not_route",
        "event_id": spec.event_id,
        "role_asset_sha256": role_asset.ASSET_SHA256,
        "role_event_sha256": role_asset.EVENT_SHA256,
        "source_video_sha256": role_asset.SOURCE_VIDEO_SHA256,
        "model_text_sha256": spec.model_text_sha256,
        "tokenizer_tree_sha256": spec.tokenizer_tree_sha256,
        "token_input_ids_sha256": locator.object_sha256(ids),
        "token_attention_mask_sha256": locator.object_sha256(mask),
        "active_token_count": active,
        "active_token_ids": ids[:active],
        "active_token_offsets": [list(item) for item in offsets[:active]],
        "role_occupied_token_indices": sorted(occupied),
        "registration_policy": {
            "candidate_order": "span_length_then_token_start",
            "span_lengths": list(REGISTRATION_LENGTHS),
            "requires_active_attention_mask": True,
            "requires_nonzero_character_offset": True,
            "excludes_all_locked_role_tokens": True,
            "first_exact_count": SPAN_COUNT,
        },
        "span_count": SPAN_COUNT,
        "spans": candidates,
        "anchor_consumed": False,
        "route_authorized": False,
        "training_authorized": False,
        "decode_authorized": False,
    }
    return {**payload, "registry_sha256": locator.object_sha256(payload)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        subfolder="tokenizer",
        fix_mistral_regex=True,
        local_files_only=True,
    )
    print(json.dumps(build_registry(tokenizer), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
