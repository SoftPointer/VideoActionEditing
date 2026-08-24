from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from motive.r7_select_expansion import (
    R7_EXPANSION_SCHEMA,
    load_excluded_iids,
    main,
    select_expansion_rows,
)


def _row(
    index: int,
    *,
    family: str,
    score: float = 0.7,
    label: str = "temporal_action",
    actors: tuple[str, ...] = ("person",),
) -> dict[str, object]:
    return {
        "iid": f"iid-{index:04d}",
        "prompt": f"perform {family}",
        "src_video": f"videos/{index}/source.mp4",
        "tgt_video": f"videos/{index}/edited.mp4",
        "auto_rule": {
            "label": label,
            "action_families": [family],
            "actors": list(actors),
            "score": score,
            "tier": "high" if score >= 0.8 else "possible",
        },
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_legacy_split(
    row: dict[str, object],
    *,
    split: object = "train",
    seed: object = 260108828,
    version: object = "caption-or-path-fallback-v1",
) -> dict[str, object]:
    result = dict(row)
    result["split"] = split
    result["split_provenance"] = {
        "seed": seed,
        "version": version,
    }
    return result


class R7SelectExpansionTests(unittest.TestCase):
    def test_excludes_prior_and_is_deterministic(self) -> None:
        rows = [
            _row(index, family=("run" if index % 2 else "jump"))
            for index in range(12)
        ]
        selected_a, audit_a = select_expansion_rows(
            rows,
            excluded_iids={"iid-0001", "iid-0002"},
            sample_size=6,
            default_family_cap=4,
            seed=7,
        )
        selected_b, audit_b = select_expansion_rows(
            reversed(rows),
            excluded_iids={"iid-0001", "iid-0002"},
            sample_size=6,
            default_family_cap=4,
            seed=7,
        )
        self.assertEqual(
            [row["iid"] for row in selected_a],
            [row["iid"] for row in selected_b],
        )
        self.assertEqual(audit_a["selected_iid_digest"], audit_b["selected_iid_digest"])
        self.assertFalse(
            {"iid-0001", "iid-0002"}
            & {str(row["iid"]) for row in selected_a}
        )
        self.assertTrue(
            all(
                row["r7_expansion_selection"]["split_assigned"] is False
                for row in selected_a
            )
        )
        self.assertTrue(
            all(
                row["r7_expansion_selection"]["legacy_split_quarantine"]
                == {"present": False, "canonical_sha256": None}
                for row in selected_a
            )
        )
        self.assertEqual(
            audit_a["legacy_split_quarantine"],
            audit_b["legacy_split_quarantine"],
        )

    def test_legacy_split_is_strictly_quarantined_and_audited(self) -> None:
        pair = {
            "split": "validation",
            "split_provenance": {
                "seed": 17,
                "version": "legacy-content-split-v3",
            },
        }
        rows = [
            _with_legacy_split(
                _row(0, family="run"),
                split=pair["split"],
                seed=17,
                version="legacy-content-split-v3",
            ),
            _row(1, family="jump"),
        ]
        selected, audit = select_expansion_rows(
            rows,
            excluded_iids=set(),
            sample_size=2,
            seed=9,
        )
        selected_by_iid = {str(row["iid"]): row for row in selected}
        legacy = selected_by_iid["iid-0000"]
        clean = selected_by_iid["iid-0001"]
        digest = _canonical_digest(pair)

        self.assertNotIn("split", legacy)
        self.assertNotIn("split_provenance", legacy)
        self.assertEqual(
            legacy["r7_expansion_selection"]["legacy_split_quarantine"],
            {"present": True, "canonical_sha256": digest},
        )
        self.assertEqual(
            set(
                legacy["r7_expansion_selection"][
                    "legacy_split_quarantine"
                ]
            ),
            {"present", "canonical_sha256"},
        )
        self.assertEqual(
            clean["r7_expansion_selection"]["legacy_split_quarantine"],
            {"present": False, "canonical_sha256": None},
        )

        quarantine = audit["legacy_split_quarantine"]
        self.assertEqual(quarantine["source_rows_with_pair"], 1)
        self.assertEqual(quarantine["source_rows_without_pair"], 1)
        self.assertEqual(quarantine["removed_row_count"], 1)
        self.assertEqual(quarantine["removed_top_level_field_count"], 2)
        self.assertEqual(
            quarantine["split_value_counts"],
            {"validation": 1},
        )
        self.assertEqual(
            quarantine["provenance_version_counts"],
            {"legacy-content-split-v3": 1},
        )
        self.assertEqual(quarantine["provenance_seed_counts"], {"17": 1})
        self.assertEqual(
            quarantine["provenance_sha256_counts"],
            {
                _canonical_digest(pair["split_provenance"]): 1,
            },
        )
        self.assertEqual(
            quarantine["canonical_pair_sha256_counts"],
            {digest: 1},
        )
        self.assertFalse(
            quarantine["output_rows_have_top_level_split_fields"]
        )

    def test_legacy_split_audit_is_input_order_independent(self) -> None:
        rows = [
            _with_legacy_split(
                _row(0, family="run"),
                split="train",
                seed=4,
                version="legacy-a",
            ),
            _with_legacy_split(
                _row(1, family="jump"),
                split="test",
                seed=5,
                version="legacy-b",
            ),
            _row(2, family="turn"),
        ]
        selected_a, audit_a = select_expansion_rows(
            rows,
            excluded_iids=set(),
            sample_size=3,
            seed=13,
        )
        selected_b, audit_b = select_expansion_rows(
            reversed(rows),
            excluded_iids=set(),
            sample_size=3,
            seed=13,
        )
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(
            audit_a["legacy_split_quarantine"],
            audit_b["legacy_split_quarantine"],
        )

    def test_partial_legacy_split_fails_closed_before_filtering(self) -> None:
        only_split = _row(0, family="run", label="endpoint_pose")
        only_split["split"] = "train"
        only_provenance = _row(1, family="jump")
        only_provenance["split_provenance"] = {
            "seed": 1,
            "version": "legacy-v1",
        }
        for row in (only_split, only_provenance):
            with self.subTest(iid=row["iid"]):
                with self.assertRaisesRegex(
                    ValueError,
                    "partial legacy split metadata",
                ):
                    select_expansion_rows(
                        [row],
                        excluded_iids={str(row["iid"])},
                        sample_size=1,
                    )

    def test_invalid_legacy_split_or_provenance_fails_closed(self) -> None:
        invalid_rows = [
            _with_legacy_split(_row(0, family="run"), split="dev"),
            _with_legacy_split(_row(1, family="run"), split=" train"),
            _with_legacy_split(_row(2, family="run"), split=True),
            _with_legacy_split(_row(3, family="run"), seed=True),
            _with_legacy_split(_row(4, family="run"), seed=-1),
            _with_legacy_split(_row(5, family="run"), seed=1.0),
            _with_legacy_split(_row(6, family="run"), version=""),
            _with_legacy_split(_row(7, family="run"), version=" legacy"),
            _with_legacy_split(_row(8, family="run"), version="legacy\x00v1"),
        ]
        extra_key = _with_legacy_split(_row(9, family="run"))
        extra_key["split_provenance"]["extra"] = "forbidden"
        invalid_rows.append(extra_key)
        non_object = _with_legacy_split(_row(10, family="run"))
        non_object["split_provenance"] = ["seed", "version"]
        invalid_rows.append(non_object)

        for row in invalid_rows:
            with self.subTest(iid=row["iid"]):
                with self.assertRaises(ValueError):
                    select_expansion_rows(
                        [row],
                        excluded_iids=set(),
                        sample_size=1,
                    )

    def test_filters_rule_score_label_and_actor(self) -> None:
        rows = [
            _row(0, family="run"),
            _row(1, family="run", score=0.4),
            _row(2, family="run", label="endpoint_pose"),
            _row(3, family="run", actors=()),
        ]
        selected, audit = select_expansion_rows(
            rows,
            excluded_iids=set(),
            sample_size=1,
        )
        self.assertEqual(selected[0]["iid"], "iid-0000")
        self.assertEqual(
            audit["rejection_counts"],
            {"actor_missing": 1, "rule_label": 1, "rule_score": 1},
        )

    def test_family_caps_fail_closed(self) -> None:
        rows = [_row(index, family="open_close") for index in range(5)]
        with self.assertRaisesRegex(ValueError, "only 1 eligible"):
            select_expansion_rows(
                rows,
                excluded_iids=set(),
                sample_size=2,
                family_caps={"open_close": 1},
            )

    def test_duplicate_iid_is_rejected(self) -> None:
        rows = [_row(0, family="run"), _row(0, family="jump")]
        with self.assertRaisesRegex(ValueError, "duplicate input IID"):
            select_expansion_rows(
                rows,
                excluded_iids=set(),
                sample_size=1,
            )

    def test_cli_writes_bound_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.jsonl"
            exclude = root / "exclude.jsonl"
            output = root / "selected.jsonl"
            _write(
                source,
                [
                    _row(index, family=("run" if index % 2 else "jump"))
                    for index in range(8)
                ],
            )
            _write(exclude, [{"iid": "iid-0000"}])
            self.assertEqual(
                main(
                    [
                        "--input",
                        str(source),
                        "--exclude",
                        str(exclude),
                        "--output",
                        str(output),
                        "--sample-size",
                        "4",
                        "--seed",
                        "9",
                    ]
                ),
                0,
            )
            summary = json.loads(
                output.with_suffix(".jsonl.summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["schema_version"], R7_EXPANSION_SCHEMA)
            self.assertEqual(summary["audit"]["selected_rows"], 4)
            self.assertEqual(summary["excluded_unique_iids"], 1)
            self.assertFalse(summary["split_assigned"])
            self.assertFalse(summary["human_labels_asserted"])
            selected = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all("split" not in row for row in selected)
            )
            self.assertEqual(
                summary["audit"]["legacy_split_quarantine"][
                    "removed_row_count"
                ],
                0,
            )

    def test_load_exclusions_rejects_missing_iid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            _write(path, [{"prompt": "missing"}])
            with self.assertRaisesRegex(ValueError, "no IID"):
                load_excluded_iids([path])


if __name__ == "__main__":
    unittest.main()
