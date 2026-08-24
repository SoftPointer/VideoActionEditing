#!/usr/bin/env python3
"""Build four sealed native-RV2V candidate specs for Job 140846.

Each of the eight complex events receives four current-policy candidates.  A
candidate sees only the real source video and the edit instruction.  Pure-T2V
anchor appearance is deliberately absent from this schema and from every
caption.  Four guidance/seed variants expose an action/preservation trade-off
that can later be ranked by a candidate-side action critic under hard source
preservation gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pair-v5-native-rv2v4-rollout-spec-v1"
CAPTION_CONTRACT = "complete_source_content_caption_with_requested_new_action"
SAMPLING_CONTRACT = {
    "condition_mode": "rv2v4",
    "num_frames": 81,
    "latent_frames": 21,
    "fps": 25,
    "num_inference_steps": 40,
    "source_reference_indices": [0, 27, 53, 80],
    "target_initialization": "official_gen_wanx22_fresh_gaussian",
}
SEMANTIC_INPUT_CLOSURE = {
    "accepted": ["source_video", "complete_caption"],
    "target_video": False,
    "t2v_proposal_media": False,
    "donor_video": False,
    "external_reference": False,
    "mask": False,
    "flow": False,
    "pose": False,
    "track": False,
    "trajectory": False,
}

SOURCE_SHA256_BY_IID = {
    "2f183dbf9e7a4d2e": "888789206a3120c0780be8961dee7fdda520502cb95f573fe211b269aaea53de",
    "186d36f998a14e14": "60618e5a988f3d8b4f48d4ae46bc7739032663a7f8805ee26d47b7d3c193af48",
    "10ed90644f81461d": "63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c",
    "7a33b36459c84289": "c1455b9b89d1f352da69e7bb07e955ee4495df94f5ef6f3f09fe7fd9eac035bb",
    "12eba2f9c15f4d3f": "d699a8d5e35a57f09ae4ba5fc5124e733be9ed18a2bddb2ee90a1ba0232c53f5",
    "7a9e7172ed054f57": "082dac2db2650cac50564182f3ee7db8db452f13dddf53f2f7536f79aeef5df2",
    "4ebd076ff4af4c81": "13ecaf94616c5a9548b80e9c25039def780a9c962dfde6de10a5d9566e3bf66a",
    "a023388fb2374e44": "1164531fd34d3d1273d56930aed139eb1a5d8db708ac3cdc4f7434abc0080799",
}

GUIDANCE_BY_VARIANT = (
    {"omega_txt": 4.0, "omega_vid": 1.25, "omega_img": 4.5},
    {"omega_txt": 5.0, "omega_vid": 1.25, "omega_img": 4.5},
    {"omega_txt": 4.0, "omega_vid": 1.75, "omega_img": 5.5},
    {"omega_txt": 5.5, "omega_vid": 1.50, "omega_img": 5.0},
)

NODE_PLAN = {
    "auh7-1b-gpu-246": ((0, 1, 2, 3), (0, 1)),
    "auh7-1b-gpu-247": ((4, 5, 6, 7), (0, 1)),
    "auh7-1b-gpu-248": ((0, 1, 2, 3), (2, 3)),
    "auh7-1b-gpu-279": ((4, 5, 6, 7), (2, 3)),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def complete_caption(event: Mapping[str, Any]) -> str:
    return (
        "Use the source video as the sole authority for identity, appearance, "
        "clothing or fur color, object instances, background, lighting, framing, "
        "camera and initial state. Frame 0 must retain the original source state; "
        "do not pre-apply the requested endpoint. Perform only this temporal edit: "
        f"{event['action']} {event['constraints']} The edit must be one continuous "
        "81-frame video at 25 fps and must not introduce appearance changes as a "
        "substitute for the requested action."
    )


def candidate(event: Mapping[str, Any], variant: int) -> dict[str, Any]:
    caption = complete_caption(event)
    iid = str(event["source_iid"])
    ordinal = int(event["ordinal"])
    return {
        "candidate_id": f"complex8-e{ordinal:02d}-rv2v-s{variant}",
        "source_video": str(event["geometry_source_video"]),
        "source_video_sha256": SOURCE_SHA256_BY_IID[iid],
        "complete_caption": caption,
        "complete_caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "caption_contract": CAPTION_CONTRACT,
        "seed": 202608172000 + ordinal * 10 + variant,
        "guidance": dict(GUIDANCE_BY_VARIANT[variant]),
    }


def build_specs(authoring: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if authoring.get("schema_version") != "bernini-interaction-complex8-multianchor-authoring-v2":
        raise ValueError("complex8 authoring schema differs")
    events = list(authoring.get("events", ()))
    if len(events) != 8 or [row.get("ordinal") for row in events] != list(range(8)):
        raise ValueError("complex8 event order differs")
    if {str(row.get("source_iid")) for row in events} != set(SOURCE_SHA256_BY_IID):
        raise ValueError("complex8 source identity closure differs")
    specs: dict[str, dict[str, Any]] = {}
    for node, (event_indices, variants) in NODE_PLAN.items():
        rows = [candidate(events[index], variant) for index in event_indices for variant in variants]
        specs[node] = {
            "schema_version": SCHEMA_VERSION,
            "sampling_contract": dict(SAMPLING_CONTRACT),
            "semantic_input_closure": dict(SEMANTIC_INPUT_CLOSURE),
            "groups": [
                {"group_id": "sp4-a", "visible_gpus": [0, 1, 2, 3], "candidates": rows[:4]},
                {"group_id": "sp4-b", "visible_gpus": [4, 5, 6, 7], "candidates": rows[4:]},
            ],
        }
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists():
        raise ValueError("output-dir must be a fresh absolute non-root directory")
    authoring = json.loads(Path(args.authoring_manifest).read_text(encoding="utf-8"))
    specs = build_specs(authoring)
    output.mkdir(parents=True)
    receipt_rows = []
    for node, value in specs.items():
        path = output / f"{node}.json"
        raw = canonical_bytes(value)
        path.write_bytes(raw)
        receipt_rows.append(
            {
                "node": node,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "candidate_ids": [
                    row["candidate_id"] for group in value["groups"] for row in group["candidates"]
                ],
            }
        )
    receipt = {
        "schema_version": "bernini-interaction-complex8-rv2v-spec-build-receipt-v1",
        "source_free_t2v_pixels_or_latents_in_candidate_specs": False,
        "candidate_count": 32,
        "specs": receipt_rows,
    }
    receipt["receipt_digest"] = hashlib.sha256(canonical_bytes(receipt)[:-1]).hexdigest()
    (output / "build-receipt.json").write_bytes(canonical_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
