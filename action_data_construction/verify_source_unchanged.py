#!/usr/bin/env python3
"""Compare the current read-only MEV media inventory to the build receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import iter_jsonl, source_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.build_summary.read_text(encoding="utf-8"))["source_inventory"]
    names = []
    import csv

    with args.events.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            names.append(row["original_filename"])
    source_root = args.events.resolve(strict=True).parent.parent
    observed = source_inventory(source_root, names)
    result = {"expected": expected, "observed": observed, "unchanged": observed == expected}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
