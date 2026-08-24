#!/usr/bin/env python3
"""Materialize the four-source confirmation execution manifest.

The winning schedule x block-band cell is read only from the SHA-pinned human
authorization.  This tool has intentionally no schedule or band CLI option.
"""

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
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--expected-review-manifest-sha256", required=True)
    parser.add_argument("--dog-formal-output", required=True)
    parser.add_argument("--human-formal-output", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    value = contract.materialize_manifest_value(
        review_manifest_path=args.review_manifest,
        expected_review_manifest_sha256=args.expected_review_manifest_sha256,
        dog_formal_output=args.dog_formal_output,
        human_formal_output=args.human_formal_output,
        authorization_path=args.authorization,
        expected_authorization_sha256=args.expected_authorization_sha256,
        verify_files=True,
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
                "evidence_role": value["evidence_role"],
                "admitted_cell": value["admitted_cell"],
                "sentinel_order": value["sentinel_order"],
                "outputs_per_sentinel": contract.EXPECTED_OUTPUTS,
                "stage_b_admission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
