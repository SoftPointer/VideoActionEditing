from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive import wan22_select_exact512 as selector


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical(row) for row in rows))


def _generation_row(
    index: int,
    *,
    prefix: str = "scale",
    profile: dict[str, object] | None = None,
) -> dict[str, object]:
    iid = f"{prefix}-sample-{index:04d}"
    group = f"{prefix}-group-{index:04d}"
    prompt = f"Make subject {prefix} {index} perform action {index}."
    row: dict[str, object] = {
        "schema_version": (
            selector.FINALIZER_GENERATION_SCHEMA
            if profile is not None
            else selector.RETAINED_GENERATION_SCHEMA
        ),
        "iid": iid,
        "group_id": group,
        "action_change_substantive": "yes",
        "edit_instruction": prompt,
        "edit_instruction_sha256": _sha(prompt.encode("utf-8")),
        "instruction_contract": dict(selector._INSTRUCTION_CONTRACT),
        "source_edited_caption_provenance_role": (
            "non_executable_provenance"
        ),
        "source_instruction_provenance": prompt,
        "manifest_role": "review_proposal",
        "production_eligible": False,
        "human_review_status": "pending",
        "generation_authorized": False,
        "approval": None,
        "authorization_interface_available": False,
    }
    if profile is not None:
        row["policy_version"] = selector.FINALIZER_POLICY_VERSION
        row["finalization_profile"] = profile
    return row


class Wan22SelectExact512Tests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        generation_count: int = 512,
        review_count: int = 640,
    ) -> Path:
        final = root / "final"
        final.mkdir()
        profile = self._profile()
        review_rows: list[dict[str, object]] = []
        generation_rows: list[dict[str, object]] = []
        # Input order deliberately opposes rank order.
        for index in range(review_count):
            generation = _generation_row(index, profile=profile)
            rank = review_count - index
            proposed = index < generation_count
            finalization = {
                "schema_version": selector.FINALIZER_REVIEW_SCHEMA,
                "policy_version": selector.FINALIZER_POLICY_VERSION,
                "hard_gate_passed": True,
                "hard_gate_failures": [],
                "review_rank": rank,
                "selection_bucket": "proposed" if proposed else "reserve",
                "human_review_status": "pending",
                "human_label": False,
                "generation_authorized": False,
                "manifest_role": "review_proposal",
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
                "profile": profile,
            }
            review_rows.append(
                {
                    "iid": generation["iid"],
                    "group_id": generation["group_id"],
                    "prompt": generation["edit_instruction"],
                    "action_anchor_finalization": finalization,
                }
            )
            if proposed:
                generation_rows.append(generation)

        _write_jsonl(final / selector.REVIEW_NAME, review_rows)
        _write_jsonl(
            final / selector.PROPOSED_NAME,
            review_rows[:generation_count],
        )
        _write_jsonl(
            final / selector.RESERVE_NAME,
            review_rows[generation_count : generation_count + 128],
        )
        _write_jsonl(
            final / selector.PARENT_GENERATION_NAME,
            generation_rows,
        )
        self._write_summary_and_done(final)
        return final

    def _profile(self) -> dict[str, object]:
        config: dict[str, object] = {
            "required_qwen_shard_count": 8,
            "review_limit": 768,
            "proposed_size": 512,
            "reserve_size": 128,
            "max_per_target_verb": 48,
            "category_quotas": {
                "locomotion": 128,
                "posture": 128,
                "interaction": 192,
                "articulated": 64,
            },
            "artifacts": {
                "review": selector.REVIEW_NAME,
                "proposed": selector.PROPOSED_NAME,
                "reserve": selector.RESERVE_NAME,
                "generation": selector.PARENT_GENERATION_NAME,
                "summary": selector.SUMMARY_NAME,
                "done": selector.DONE_NAME,
            },
            "schemas": {
                "row": selector.FINALIZER_REVIEW_SCHEMA,
                "generation": selector.FINALIZER_GENERATION_SCHEMA,
                "summary": selector.FINALIZER_SUMMARY_SCHEMA,
                "done": selector.FINALIZER_DONE_SCHEMA,
            },
            "policy_version": selector.FINALIZER_POLICY_VERSION,
        }
        config_sha = _sha(
            json.dumps(
                config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return {
            "schema_version": selector.FINALIZER_PROFILE_SCHEMA,
            "name": selector.FINALIZER_PROFILE_NAME,
            "config": config,
            "config_sha256": config_sha,
        }

    def _write_summary_and_done(self, final: Path) -> None:
        review_count = len(_read_jsonl(final / selector.REVIEW_NAME))
        generation_count = len(
            _read_jsonl(final / selector.PARENT_GENERATION_NAME)
        )
        reserve_count = len(_read_jsonl(final / selector.RESERVE_NAME))
        output_hashes = {
            name: _sha((final / name).read_bytes())
            for name in selector._SUMMARY_HASHED_OUTPUTS
        }
        implementation_sha = _sha(
            Path(selector.__file__)
            .with_name("goku_action_anchor_finalize.py")
            .read_bytes()
        )
        profile = self._profile()
        summary = {
            "schema_version": selector.FINALIZER_SUMMARY_SCHEMA,
            "policy_version": selector.FINALIZER_POLICY_VERSION,
            "profile": profile,
            "seed": 260730,
            "input": {},
            "hard_gate": {},
            "diversity": {},
            "selection": {
                "mode": "strict_512_plus_128",
                "allow_partial": False,
                "requested_proposed_rows": 512,
                "requested_reserve_rows": 128,
                "effective_proposed_target": 512,
                "effective_reserve_target": 128,
                "review_rows": review_count,
                "proposed_rows": generation_count,
                "reserve_rows": reserve_count,
                "generation_rows": generation_count,
                "requested_category_quotas": {
                    "locomotion": 128,
                    "posture": 128,
                    "interaction": 192,
                    "articulated": 64,
                },
                "effective_category_quotas": {
                    "locomotion": 128,
                    "posture": 128,
                    "interaction": 192,
                    "articulated": 64,
                },
                "proposed_category_counts": {},
                "quota_shortfall_before_backfill": {},
                "review_category_counts": {},
                "reserve_category_counts": {},
                "proposed_target_verb_counts": {},
                "proposal_reserve_disjoint": True,
            },
            "semantics": {
                "manifest_role": "review_proposal",
                "human_review_status": "pending",
                "human_labels_asserted": False,
                "generation_authorized": False,
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            },
            "implementation_sha256": implementation_sha,
            "output_sha256": output_hashes,
        }
        summary_raw = _pretty(summary)
        (final / selector.SUMMARY_NAME).write_bytes(summary_raw)
        done_outputs = {
            name: _sha((final / name).read_bytes())
            for name in selector._FINALIZER_HASHED_OUTPUTS
        }
        profile_raw = json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        done = {
            "schema_version": selector.FINALIZER_DONE_SCHEMA,
            "status": "complete",
            "profile": profile,
            "profile_sha256": _sha(profile_raw),
            "summary_sha256": _sha(summary_raw),
            "implementation_sha256": implementation_sha,
            "output_sha256": done_outputs,
        }
        (final / selector.DONE_NAME).write_bytes(_pretty(done))

    def _retained(
        self,
        root: Path,
        final: Path,
        *,
        overlap: int = 4,
    ) -> Path:
        final_rows = _read_jsonl(final / selector.PARENT_GENERATION_NAME)
        rows = []
        for final_row in final_rows[:overlap]:
            retained_row = dict(final_row)
            retained_row["schema_version"] = selector.RETAINED_GENERATION_SCHEMA
            retained_row.pop("policy_version")
            retained_row.pop("finalization_profile")
            rows.append(retained_row)
        rows.extend(
            _generation_row(index, prefix="retained")
            for index in range(8 - overlap)
        )
        path = root / "prior_exact8.jsonl"
        _write_jsonl(path, rows)
        return path

    def test_retains_exact8_then_deterministically_fills_by_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            retained = self._retained(root, final)
            output_a = root / "exact512-a"
            output_b = root / "exact512-b"
            receipt_a = selector.select_exact512(
                finalizer_dir=final,
                output_dir=output_a,
                retain_exact8_manifest=retained,
            )
            receipt_b = selector.select_exact512(
                finalizer_dir=final,
                output_dir=output_b,
                retain_exact8_manifest=retained,
            )
            self.assertEqual(receipt_a, receipt_b)
            self.assertEqual(receipt_a["policy"], selector.RETAIN_EXACT8_POLICY)
            selection = receipt_a["selection"]
            self.assertEqual(selection["row_count"], 512)
            self.assertEqual(selection["retained_row_count"], 8)
            self.assertEqual(selection["ranked_fill_row_count"], 504)
            output_raw = (
                output_a / selector.OUTPUT_MANIFEST_NAME
            ).read_bytes()
            self.assertTrue(output_raw.startswith(retained.read_bytes()))
            self.assertEqual(output_raw.count(b"\n"), 512)
            self.assertEqual(selection["output_sha256"], _sha(output_raw))
            self.assertEqual(
                output_raw,
                (output_b / selector.OUTPUT_MANIFEST_NAME).read_bytes(),
            )
            receipt_raw = (
                output_a / selector.OUTPUT_RECEIPT_NAME
            ).read_bytes()
            self.assertEqual(receipt_raw, _canonical(json.loads(receipt_raw)))
            source = receipt_a["retention"]["source"]
            self.assertEqual(source["path"], str(retained.resolve()))
            self.assertEqual(source["sha256"], _sha(retained.read_bytes()))
            self.assertEqual(source["row_count"], 8)

            fill_ranks = selection["ordered_fill_review_ranks"]
            self.assertEqual(fill_ranks, sorted(fill_ranks))
            self.assertEqual(len(fill_ranks), len(set(fill_ranks)))
            # Scale rows 0..3 overlap retained rows and must not be repeated.
            output_iids = [
                row["iid"]
                for row in _read_jsonl(
                    output_a / selector.OUTPUT_MANIFEST_NAME
                )
            ]
            self.assertEqual(len(output_iids), len(set(output_iids)))

    def test_rank_only_mode_selects_all_512_in_review_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            output = root / "exact512"
            receipt = selector.select_exact512(
                finalizer_dir=final,
                output_dir=output,
            )
            self.assertEqual(receipt["policy"], selector.RANK_ONLY_POLICY)
            self.assertEqual(receipt["retention"]["source"], None)
            self.assertEqual(receipt["selection"]["retained_row_count"], 0)
            self.assertEqual(
                receipt["selection"]["ordered_fill_review_ranks"],
                list(range(129, 641)),
            )

    def test_parent_artifact_tamper_and_wrong_implementation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            (final / selector.RESERVE_NAME).write_bytes(b"tampered\n")
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError, "hash differs"
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "rejected"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            done = _read_json(final / selector.DONE_NAME)
            done["implementation_sha256"] = "f" * 64
            (final / selector.DONE_NAME).write_bytes(_pretty(done))
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError, "implementation"
            ):
                selector.select_exact512(
                    finalizer_dir=final,
                    output_dir=root / "wrong-implementation",
                )

    def test_rehashed_generation_prompt_and_pending_forgery_fail(self) -> None:
        for field, value, message in (
            ("generation_authorized", True, "exact pending"),
            ("edit_instruction", "A forged instruction.", "SHA differs"),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    final = self._fixture(root)
                    rows = _read_jsonl(
                        final / selector.PARENT_GENERATION_NAME
                    )
                    rows[0][field] = value
                    _write_jsonl(
                        final / selector.PARENT_GENERATION_NAME, rows
                    )
                    self._write_summary_and_done(final)
                    with self.assertRaisesRegex(
                        selector.Wan22Exact512SelectionError, message
                    ):
                        selector.select_exact512(
                            finalizer_dir=final,
                            output_dir=root / "forgery",
                        )

    def test_duplicate_rank_group_and_insufficient_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            review = _read_jsonl(final / selector.REVIEW_NAME)
            first = review[0]["action_anchor_finalization"]
            second = review[1]["action_anchor_finalization"]
            assert isinstance(first, dict) and isinstance(second, dict)
            second["review_rank"] = first["review_rank"]
            _write_jsonl(final / selector.REVIEW_NAME, review)
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError,
                "duplicate review_rank",
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "duplicate-rank"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            generation = _read_jsonl(
                final / selector.PARENT_GENERATION_NAME
            )
            generation[1]["group_id"] = generation[0]["group_id"]
            _write_jsonl(
                final / selector.PARENT_GENERATION_NAME, generation
            )
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError,
                "duplicate generation group_id",
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "duplicate-group"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root, generation_count=511)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError,
                "exactly 512",
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "too-few"
                )

    def test_retained_manifest_strictness_and_identity_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            retained = self._retained(root, final)
            rows = _read_jsonl(retained)
            rows.pop()
            _write_jsonl(retained, rows)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError, "exactly eight"
            ):
                selector.select_exact512(
                    finalizer_dir=final,
                    output_dir=root / "seven",
                    retain_exact8_manifest=retained,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            retained = self._retained(root, final, overlap=0)
            rows = _read_jsonl(retained)
            rows[1]["iid"] = rows[0]["iid"]
            _write_jsonl(retained, rows)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError,
                "duplicate retained exact8 iid",
            ):
                selector.select_exact512(
                    finalizer_dir=final,
                    output_dir=root / "duplicate",
                    retain_exact8_manifest=retained,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            retained = self._retained(root, final)
            rows = _read_jsonl(retained)
            # IID matches a finalizer row, but group does not: reject instead
            # of silently dropping either identity.
            rows[0]["group_id"] = "conflicting-retained-group"
            _write_jsonl(retained, rows)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError, "identity conflict"
            ):
                selector.select_exact512(
                    finalizer_dir=final,
                    output_dir=root / "identity-conflict",
                    retain_exact8_manifest=retained,
                )

    def test_noncanonical_input_profile_forgery_and_no_overwrite_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            rows = _read_jsonl(final / selector.REVIEW_NAME)
            (final / selector.REVIEW_NAME).write_bytes(
                b"".join(
                    (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
                    for row in rows
                )
            )
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError,
                "not canonical JSON",
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "noncanonical"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            summary = _read_json(final / selector.SUMMARY_NAME)
            profile = summary["profile"]
            assert isinstance(profile, dict)
            profile["name"] = "default"
            (final / selector.SUMMARY_NAME).write_bytes(_pretty(summary))
            done = _read_json(final / selector.DONE_NAME)
            done["summary_sha256"] = _sha(
                (final / selector.SUMMARY_NAME).read_bytes()
            )
            outputs = done["output_sha256"]
            assert isinstance(outputs, dict)
            outputs[selector.SUMMARY_NAME] = done["summary_sha256"]
            (final / selector.DONE_NAME).write_bytes(_pretty(done))
            with self.assertRaisesRegex(
                selector.Wan22Exact512SelectionError, "profile differs"
            ):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=root / "wrong-profile"
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            output = root / "exact512"
            selector.select_exact512(
                finalizer_dir=final, output_dir=output
            )
            original = (
                output / selector.OUTPUT_MANIFEST_NAME
            ).read_bytes()
            with self.assertRaises(FileExistsError):
                selector.select_exact512(
                    finalizer_dir=final, output_dir=output
                )
            self.assertEqual(
                (output / selector.OUTPUT_MANIFEST_NAME).read_bytes(),
                original,
            )


if __name__ == "__main__":
    unittest.main()
