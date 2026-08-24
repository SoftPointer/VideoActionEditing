#!/usr/bin/env python3
"""Validate one Stage-A winner confirmation shard and all exact81 media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import stage_a_source_edge_confirmation_contract_v1 as contract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--sentinel-id", choices=contract.SENTINEL_ORDER, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = contract._plain_file(args.manifest, label="confirmation manifest")
    manifest = contract.load_manifest(
        manifest_path,
        expected_file_sha256=args.expected_manifest_sha256,
        verify_files=True,
    )
    receipt, receipt_path, receipt_sha = contract.load_receipt(
        args.output_dir,
        manifest_value=manifest,
        manifest_path=manifest_path,
        manifest_file_sha256=args.expected_manifest_sha256,
        sentinel_id=args.sentinel_id,
        verify_media=True,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "sentinel_id": args.sentinel_id,
                "record_count": len(receipt["records"]),
                "receipt": str(receipt_path),
                "receipt_file_sha256": receipt_sha,
                "receipt_digest": receipt["receipt_digest"],
                "stage_b_admission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
