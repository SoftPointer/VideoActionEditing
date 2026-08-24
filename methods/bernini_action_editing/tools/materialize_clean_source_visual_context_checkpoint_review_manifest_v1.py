#!/usr/bin/env python3
"""Materialize the fixed four-sentinel Stage-B checkpoint-review manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_checkpoint_review_contract_v1 as contract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only-manifest", required=True)
    parser.add_argument("--authoring", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    value = contract.materialize_manifest_value(
        source_only_manifest_path=args.source_only_manifest,
        authoring_path=args.authoring,
        verify_files=True,
        verify_source_media=True,
    )
    output = contract.write_create_only_json(args.output, value)
    contract.load_manifest(
        output,
        expected_file_sha256=contract.file_sha256(output),
        verify_files=True,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": contract.file_sha256(output),
                "manifest_digest": value["manifest_digest"],
                "sentinels": list(contract.SENTINEL_ORDER),
                "checkpoint_steps": list(contract.CHECKPOINT_STEPS),
                "logical_records": (
                    len(contract.SENTINEL_ORDER)
                    * len(contract.CHECKPOINT_STEPS)
                    * len(contract.LOGICAL_ARM_ORDER)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
