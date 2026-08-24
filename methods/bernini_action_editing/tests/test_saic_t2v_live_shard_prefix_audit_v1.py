from __future__ import annotations

import ast
from pathlib import Path
import unittest

import saic_t2v_live_shard_prefix_audit_v1 as live


class SAICT2VLiveShardPrefixAuditTests(unittest.TestCase):
    def test_authority_is_diagnostic_only(self) -> None:
        self.assertFalse(live.AUTHORITY["detached_decoded_event_review_input"])
        self.assertFalse(live.AUTHORITY["merge_or_partial_reuse"])
        self.assertFalse(live.AUTHORITY["scientific_selection"])
        self.assertFalse(live.AUTHORITY["training"])
        self.assertFalse(live.AUTHORITY["optimizer"])

    def test_minimum_completed_range_fails_before_io(self) -> None:
        for value in (-1, 31, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    live.SAICT2VLiveShardPrefixAuditError,
                    "expected minimum completed",
                ):
                    live.audit_live_prefix(
                        "/does/not/exist",
                        group={"group_id": "sp4-a"},
                        root_spec_sha256="a" * 64,
                        source_manifest={},
                        expected_min_completed=value,
                    )

    def test_production_path_reuses_both_deep_validators(self) -> None:
        source = Path(live.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {
            f"{node.func.value.id}.{node.func.attr}"
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            )
        }
        self.assertIn("generation._load_attempt_receipt", calls)
        self.assertIn("rendezvous._validate_completion", calls)
        self.assertIn("final_audit._plain_file", calls)
        self.assertIn("final_audit._under", calls)
        self.assertIn("same-cell Gaussian differs", source)
        self.assertIn("contiguous prefix", source)

    def test_parser_requires_contract_and_group_inputs(self) -> None:
        parser = live.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            [
                "--output-root",
                "/runs/a",
                "--group-id",
                "sp4-a",
                "--root-spec",
                "/runs/a/root.json",
                "--base-v1-spec",
                "/runs/a/base.json",
                "--source-manifest",
                "/runs/a/source.json",
                "--expected-root-spec-sha256",
                "a" * 64,
                "--expected-min-completed",
                "2",
            ]
        )
        self.assertEqual(args.group_id, "sp4-a")
        self.assertEqual(args.expected_min_completed, 2)


if __name__ == "__main__":
    unittest.main()
