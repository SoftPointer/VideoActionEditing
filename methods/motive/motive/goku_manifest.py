"""Normalize or sample Goku subject_movement combined JSON files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from .semantics import classify_instruction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Lucy-compatible paired JSONL from Goku combined JSON."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=260108828)
    parser.add_argument(
        "--semantic-classes",
        nargs="+",
        help="Optional prefilter, e.g. continuous_action motion_suppression.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.expanduser()
    combined_dir = dataset_root / "jsons" / "combine_json"
    if not combined_dir.is_dir():
        raise FileNotFoundError(combined_dir)
    files = sorted(combined_dir.glob("*_all.json"))
    if not files:
        raise RuntimeError(f"no *_all.json files found in {combined_dir}")
    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        sample_count = min(args.sample_size, len(files))
        files = random.Random(args.seed).sample(files, sample_count)

    allowed = None if not args.semantic_classes else set(args.semantic_classes)
    rows = []
    label_counts: dict[str, int] = {}
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        instruction = str(raw.get("instruction_en") or raw.get("instruction") or "")
        semantics = classify_instruction(instruction)
        label_counts[semantics.label] = label_counts.get(semantics.label, 0) + 1
        if allowed is not None and semantics.label not in allowed:
            continue
        case_id = str(raw.get("case_id") or path.name.removesuffix("_all.json"))
        rows.append(
            {
                "iid": case_id,
                "prompt": instruction,
                "src_video": str(raw["source_video"]),
                "tgt_video": str(raw["edited_video"]),
                "instruction_semantics": semantics.to_dict(),
                "source_caption": raw.get("source_caption", ""),
                "edited_caption": raw.get("edited_caption", ""),
                "source_json": str(path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[motive-goku-manifest] scanned={len(files)} kept={len(rows)} "
        f"labels={json.dumps(label_counts, sort_keys=True)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
