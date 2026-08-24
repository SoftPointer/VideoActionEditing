from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive import wan22_select_exact8 as selector


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


class Wan22SelectExact8Tests(unittest.TestCase):
    def _fixture(self, root: Path, *, count: int = 10) -> Path:
        final = root / "final"
        final.mkdir()
        review_rows: list[dict[str, object]] = []
        generation_rows: list[dict[str, object]] = []
        # Input order is deliberately unrelated to review rank.
        ranks = list(range(count, 0, -1))
        for index, rank in enumerate(ranks):
            iid = f"sample-{index:02d}"
            group = f"group-{index:02d}"
            prompt = f"Make subject {index} perform action {index}."
            finalization = {
                "schema_version": selector.FINALIZER_REVIEW_SCHEMA,
                "policy_version": selector.FINALIZER_POLICY_VERSION,
                "hard_gate_passed": True,
                "hard_gate_failures": [],
                "review_rank": rank,
                "selection_bucket": "proposed",
                "human_review_status": "pending",
                "human_label": False,
                "generation_authorized": False,
                "manifest_role": "review_proposal",
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            }
            review_rows.append(
                {
                    "iid": iid,
                    "group_id": group,
                    "prompt": prompt,
                    "action_anchor_finalization": finalization,
                }
            )
            generation_rows.append(
                {
                    "schema_version": selector.FINALIZER_GENERATION_SCHEMA,
                    "iid": iid,
                    "group_id": group,
                    "action_change_substantive": "yes",
                    "edit_instruction": prompt,
                    "edit_instruction_sha256": _sha(
                        prompt.encode("utf-8")
                    ),
                    "instruction_contract": dict(
                        selector._INSTRUCTION_CONTRACT
                    ),
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
            )
        _write_jsonl(final / selector.REVIEW_NAME, review_rows)
        _write_jsonl(final / selector.PROPOSED_NAME, review_rows)
        (final / selector.RESERVE_NAME).write_bytes(b"")
        _write_jsonl(
            final / selector.PARENT_GENERATION_NAME,
            generation_rows,
        )
        self._write_summary_and_done(final)
        return final

    def _write_summary_and_done(self, final: Path) -> None:
        review_count = len(_read_jsonl(final / selector.REVIEW_NAME))
        generation_count = len(
            _read_jsonl(final / selector.PARENT_GENERATION_NAME)
        )
        output_hashes = {
            name: _sha((final / name).read_bytes())
            for name in selector._SUMMARY_HASHED_OUTPUTS
        }
        implementation_sha = _sha(
            Path(selector.__file__)
            .with_name("goku_action_anchor_finalize.py")
            .read_bytes()
        )
        summary = {
            "schema_version": selector.FINALIZER_SUMMARY_SCHEMA,
            "policy_version": selector.FINALIZER_POLICY_VERSION,
            "seed": 260730,
            "input": {},
            "hard_gate": {},
            "diversity": {},
            "selection": {
                "review_rows": review_count,
                "generation_rows": generation_count,
                "proposed_rows": generation_count,
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
        done = {
            "schema_version": selector.FINALIZER_DONE_SCHEMA,
            "status": "complete",
            "summary_sha256": _sha(summary_raw),
            "implementation_sha256": implementation_sha,
            "output_sha256": done_outputs,
        }
        (final / selector.DONE_NAME).write_bytes(_pretty(done))

    def test_deterministic_lowest_eight_and_byte_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            output_a = root / "exact8-a"
            output_b = root / "exact8-b"
            receipt_a = selector.select_exact8(
                finalizer_dir=final,
                output_dir=output_a,
            )
            receipt_b = selector.select_exact8(
                finalizer_dir=final,
                output_dir=output_b,
            )
            self.assertEqual(receipt_a, receipt_b)
            self.assertEqual(
                (output_a / selector.OUTPUT_RECEIPT_NAME).read_bytes(),
                (output_b / selector.OUTPUT_RECEIPT_NAME).read_bytes(),
            )
            self.assertEqual(
                (output_a / selector.OUTPUT_MANIFEST_NAME).read_bytes(),
                (output_b / selector.OUTPUT_MANIFEST_NAME).read_bytes(),
            )
            self.assertEqual(
                receipt_a["selection"]["ordered_review_ranks"],
                list(range(1, 9)),
            )
            expected_iids = [
                f"sample-{index:02d}"
                for index in range(9, 1, -1)
            ]
            self.assertEqual(
                receipt_a["selection"]["ordered_iids"],
                expected_iids,
            )

            parent_lines = {
                json.loads(line)["iid"]: line + b"\n"
                for line in (
                    final / selector.PARENT_GENERATION_NAME
                ).read_bytes().splitlines()
            }
            expected_raw = b"".join(
                parent_lines[iid] for iid in expected_iids
            )
            output_raw = (
                output_a / selector.OUTPUT_MANIFEST_NAME
            ).read_bytes()
            self.assertEqual(output_raw, expected_raw)
            receipt_raw = (
                output_a / selector.OUTPUT_RECEIPT_NAME
            ).read_bytes()
            self.assertEqual(
                receipt_raw,
                _canonical(json.loads(receipt_raw)),
            )
            self.assertEqual(
                receipt_a["selection"]["output_sha256"],
                _sha(output_raw),
            )

    def test_forged_done_hash_and_any_bound_output_tamper_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            done = _read_json(final / selector.DONE_NAME)
            outputs = done["output_sha256"]
            assert isinstance(outputs, dict)
            outputs[selector.PARENT_GENERATION_NAME] = "0" * 64
            (final / selector.DONE_NAME).write_bytes(_pretty(done))
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "hash differs",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "rejected-forgery",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            (final / selector.RESERVE_NAME).write_bytes(b"tampered\n")
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "hash differs",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "rejected-bound-output",
                )

    def test_noncanonical_review_or_generation_input_fails(self) -> None:
        for name in (
            selector.REVIEW_NAME,
            selector.PARENT_GENERATION_NAME,
        ):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    final = self._fixture(root)
                    rows = _read_jsonl(final / name)
                    (final / name).write_bytes(
                        b"".join(
                            (
                                json.dumps(
                                    row,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            ).encode("utf-8")
                            for row in rows
                        )
                    )
                    self._write_summary_and_done(final)
                    with self.assertRaisesRegex(
                        selector.Wan22Exact8SelectionError,
                        "not canonical JSON",
                    ):
                        selector.select_exact8(
                            finalizer_dir=final,
                            output_dir=root / "rejected",
                        )

    def test_duplicate_rank_or_group_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            review = _read_jsonl(final / selector.REVIEW_NAME)
            first = review[0]["action_anchor_finalization"]
            second = review[1]["action_anchor_finalization"]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            second["review_rank"] = first["review_rank"]
            _write_jsonl(final / selector.REVIEW_NAME, review)
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "duplicate review_rank",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "duplicate-rank",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            review = _read_jsonl(final / selector.REVIEW_NAME)
            generation = _read_jsonl(
                final / selector.PARENT_GENERATION_NAME
            )
            duplicate_group = review[0]["group_id"]
            review[1]["group_id"] = duplicate_group
            generation[1]["group_id"] = duplicate_group
            _write_jsonl(final / selector.REVIEW_NAME, review)
            _write_jsonl(
                final / selector.PARENT_GENERATION_NAME,
                generation,
            )
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "duplicate review group_id",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "duplicate-group",
                )

    def test_fewer_than_eight_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root, count=7)
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "fewer than eight",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "too-few",
                )

    def test_pending_flag_or_fully_rehashed_prompt_alteration_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            generation = _read_jsonl(
                final / selector.PARENT_GENERATION_NAME
            )
            generation[0]["generation_authorized"] = True
            _write_jsonl(
                final / selector.PARENT_GENERATION_NAME,
                generation,
            )
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "exact pending",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "authorized-forgery",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            generation = _read_jsonl(
                final / selector.PARENT_GENERATION_NAME
            )
            altered = "Replace the frozen action with a different action."
            generation[0]["edit_instruction"] = altered
            generation[0]["edit_instruction_sha256"] = _sha(
                altered.encode("utf-8")
            )
            generation[0]["source_instruction_provenance"] = altered
            _write_jsonl(
                final / selector.PARENT_GENERATION_NAME,
                generation,
            )
            # Recompute both summary and done bindings.  The cross-file frozen
            # prompt check must still reject this internally consistent forge.
            self._write_summary_and_done(final)
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "differs from frozen prompt",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "prompt-forgery",
                )

    def test_finalizer_implementation_binding_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            done = _read_json(final / selector.DONE_NAME)
            done["implementation_sha256"] = "f" * 64
            (final / selector.DONE_NAME).write_bytes(_pretty(done))
            with self.assertRaisesRegex(
                selector.Wan22Exact8SelectionError,
                "implementation",
            ):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=root / "wrong-implementation",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self._fixture(root)
            output = root / "exact8"
            selector.select_exact8(
                finalizer_dir=final,
                output_dir=output,
            )
            original = (
                output / selector.OUTPUT_MANIFEST_NAME
            ).read_bytes()
            with self.assertRaises(FileExistsError):
                selector.select_exact8(
                    finalizer_dir=final,
                    output_dir=output,
                )
            self.assertEqual(
                (output / selector.OUTPUT_MANIFEST_NAME).read_bytes(),
                original,
            )


if __name__ == "__main__":
    unittest.main()
