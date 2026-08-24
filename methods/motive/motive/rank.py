"""CLI for ranking cached geometry or gradient representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .archive import assert_archives_compatible, load_feature_archive
from .selection import majority_vote_select, rank_by_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Motive-style cached representations by query similarity."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--vote-percentile", type=float, default=90.0)
    parser.add_argument(
        "--single-query",
        action="store_true",
        help="Require one query and rank by its raw cosine score.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features, ids, feature_metadata = load_feature_archive(args.features)
    queries, query_ids, query_metadata = load_feature_archive(args.queries)
    assert_archives_compatible(feature_metadata, query_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.single_query:
        if len(queries) != 1:
            raise ValueError("--single-query requires exactly one query")
        count = args.top_k
        if count is None:
            if not 0.0 < args.top_fraction <= 1.0:
                raise ValueError("--top-fraction must be in (0, 1]")
            valid_count = int(
                np.sum(np.linalg.norm(features, axis=1) > 1e-8)
            )
            count = max(1, int(np.ceil(valid_count * args.top_fraction)))
        indices, scores = rank_by_query(features, queries[0], top_k=count)
        rows = [
            {
                "rank": rank,
                "id": str(ids[index]),
                "query_id": str(query_ids[0]),
                "score": float(score),
            }
            for rank, (index, score) in enumerate(zip(indices, scores), start=1)
        ]
    else:
        result = majority_vote_select(
            features,
            queries,
            vote_percentile=args.vote_percentile,
            top_k=args.top_k,
            top_fraction=None if args.top_k is not None else args.top_fraction,
        )
        rows = [
            {
                "rank": rank,
                "id": str(ids[index]),
                "votes": int(result.votes[index]),
                "mean_score": float(result.mean_scores[index]),
            }
            for rank, index in enumerate(result.indices, start=1)
        ]

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[motive-rank] selected={len(rows)} candidates={len(features)} "
        f"queries={len(queries)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
