from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from motive import goku_atomic_topup as topup


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_candidates(path: Path, count: int) -> None:
    path.write_bytes(
        b"".join(
            json.dumps({"iid": f"c{index:02d}", "rank": index}).encode() + b"\n"
            for index in range(count)
        )
    )


def planner_receipt(path: Path, selection: dict, statuses: list[str]) -> None:
    if len(selection["batch_iids"]) != len(statuses):
        raise AssertionError("test fixture status count differs")
    value = {
        "schema_version": "motive-goku-atomic1000-planner-phase-v1",
        "input": str(Path(selection["batch_manifest"]).resolve()),
        "input_sha256": selection["batch_manifest_sha256"],
        "expected_rows": selection["batch_rows"],
        "ok_rows": statuses.count("ok"),
        "error_rows": statuses.count("error"),
        "minimum_ok": 0,
        "records": [
            {"iid": iid, "status": status, "receipt_digest": sha(iid.encode())}
            for iid, status in zip(selection["batch_iids"], statuses)
        ],
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def atomic_artifacts(
    manifest: Path,
    summary: Path,
    output_root: Path,
    input_path: Path,
    passed_iids: list[str],
    expected_rows: int,
) -> None:
    payload = b"".join(
        json.dumps(
            {"iid": iid, "original_candidate_index": int(iid[1:])},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for iid in passed_iids
    )
    manifest.write_bytes(payload)
    value = {
        "schema_version": topup.ATOMIC_SUMMARY_SCHEMA,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha(input_path.read_bytes()),
        "output_root": str(output_root.resolve()),
        "expected_rows": expected_rows,
        "terminal_rows": expected_rows,
        "ok_rows": len(passed_iids),
        "error_rows": expected_rows - len(passed_iids),
        "dataset_manifest_path": str(manifest.resolve()),
        "dataset_manifest_sha256": sha(payload),
        "summary_digest": None,
    }
    value["summary_digest"] = sha(
        json.dumps(
            {key: item for key, item in value.items() if key != "summary_digest"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    summary.write_text(json.dumps(value) + "\n", encoding="utf-8")


class AtomicTopupTests(unittest.TestCase):
    def _select(
        self,
        candidates: Path,
        output_dir: Path,
        *,
        count: int,
        index: int,
        start: int,
        size: int,
        minimum_workers: int = 1,
        stage: str = "smoke",
        resume: bool = False,
    ) -> dict:
        argv = [
            "select-batch",
            "--candidates",
            str(candidates),
            "--expected-candidates",
            str(count),
            "--expected-candidates-sha256",
            sha(candidates.read_bytes()),
            "--output-dir",
            str(output_dir),
            "--batch-index",
            str(index),
            "--start-index",
            str(start),
            "--batch-size",
            str(size),
            "--minimum-workers",
            str(minimum_workers),
            "--stage",
            stage,
        ]
        if resume:
            argv.append("--resume")
        self.assertEqual(topup.main(argv), 0)
        return json.loads((output_dir / "selection_receipt.json").read_text())

    def test_batch_selection_is_parent_ordered_tail_merged_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            write_candidates(candidates, 10)
            first = self._select(
                candidates, root / "batch0", count=10, index=0, start=0, size=4,
                minimum_workers=3,
            )
            self.assertEqual(first["batch_iids"], ["c00", "c01", "c02", "c03"])
            second = self._select(
                candidates, root / "batch1", count=10, index=1, start=4, size=4,
                minimum_workers=3,
            )
            self.assertEqual(second["batch_iids"], [f"c{i:02d}" for i in range(4, 10)])
            self.assertTrue(second["tail_merged_to_preserve_worker_floor"])
            resumed = self._select(
                candidates, root / "batch1", count=10, index=1, start=4, size=4,
                minimum_workers=3, resume=True,
            )
            self.assertEqual(resumed, second)
            with self.assertRaises(topup.AtomicTopupError):
                self._select(
                    candidates, root / "batch1", count=10, index=1, start=4,
                    size=4, minimum_workers=3,
                )

    def test_dynamic_progress_reaches_target_after_failed_prefix_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            write_candidates(candidates, 10)
            candidate_sha = sha(candidates.read_bytes())
            atomic_root = root / "atomic-root"
            atomic_root.mkdir()

            selection0 = self._select(
                candidates, root / "batch0", count=10, index=0, start=0, size=4
            )
            planner0 = root / "planner0.json"
            planner_receipt(planner0, selection0, ["error", "ok", "error", "ok"])
            input0 = root / "atomic0.jsonl"
            input0.write_text('{"iid":"c01"}\n{"iid":"c03"}\n', encoding="utf-8")
            manifest0, summary0 = root / "manifest0.jsonl", root / "summary0.json"
            atomic_artifacts(manifest0, summary0, atomic_root, input0, ["c01", "c03"], 2)
            progress0 = root / "progress0.json"
            self.assertEqual(
                topup.main(
                    [
                        "publish-progress", "--candidates", str(candidates),
                        "--expected-candidates", "10",
                        "--expected-candidates-sha256", candidate_sha,
                        "--selection", str(root / "batch0" / "selection_receipt.json"),
                        "--planner-receipt", str(planner0),
                        "--atomic-manifest", str(manifest0),
                        "--atomic-summary", str(summary0),
                        "--target-atomic-ok", "3", "--output", str(progress0),
                    ]
                ),
                0,
            )
            self.assertEqual(json.loads(progress0.read_text())["status"], "continue")

            selection1 = self._select(
                candidates, root / "batch1", count=10, index=1, start=4, size=4,
                minimum_workers=3, stage="full",
            )
            planner1 = root / "planner1.json"
            planner_receipt(planner1, selection1, ["error", "ok", "error", "error", "error", "error"])
            input1 = root / "atomic1.jsonl"
            input1.write_text(
                '{"iid":"c01"}\n{"iid":"c03"}\n{"iid":"c05"}\n', encoding="utf-8"
            )
            manifest1, summary1 = root / "manifest1.jsonl", root / "summary1.json"
            atomic_artifacts(
                manifest1, summary1, atomic_root, input1, ["c01", "c03", "c05"], 3
            )
            progress1 = root / "progress1.json"
            self.assertEqual(
                topup.main(
                    [
                        "publish-progress", "--candidates", str(candidates),
                        "--expected-candidates", "10",
                        "--expected-candidates-sha256", candidate_sha,
                        "--selection", str(root / "batch1" / "selection_receipt.json"),
                        "--planner-receipt", str(planner1),
                        "--atomic-manifest", str(manifest1),
                        "--atomic-summary", str(summary1),
                        "--target-atomic-ok", "3", "--previous-progress", str(progress0),
                        "--output", str(progress1),
                    ]
                ),
                0,
            )
            final_progress = json.loads(progress1.read_text())
            self.assertEqual(final_progress["status"], "target_reached")
            self.assertEqual(final_progress["atomic_ok_rows"], 3)
            self.assertEqual(final_progress["consumed_rows"], 10)

            gate_manifest = root / "smoke8.jsonl"
            gate_receipt = root / "smoke8.json"
            self.assertEqual(
                topup.main(
                    [
                        "publish-gate", "--candidates", str(candidates),
                        "--expected-candidates", "10",
                        "--expected-candidates-sha256", candidate_sha,
                        "--atomic-manifest", str(manifest1), "--progress", str(progress1),
                        "--target-ok", "3", "--output-manifest", str(gate_manifest),
                        "--output-receipt", str(gate_receipt),
                    ]
                ),
                0,
            )
            gate = json.loads(gate_receipt.read_text())
            self.assertEqual(gate["selected_iids"], ["c01", "c03", "c05"])
            self.assertEqual(gate["selected_parent_indices"], [1, 3, 5])
            self.assertEqual(
                gate["selection_policy"],
                "first_final_atomic_passes_in_parent_candidate_order",
            )

    def test_pool_exhaustion_is_a_durable_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            write_candidates(candidates, 4)
            selection = self._select(
                candidates, root / "batch0", count=4, index=0, start=0, size=4
            )
            planner = root / "planner.json"
            planner_receipt(planner, selection, ["ok", "ok", "error", "error"])
            atomic_root = root / "atomic-root"
            atomic_root.mkdir()
            atomic_input = root / "atomic.jsonl"
            atomic_input.write_text('{"iid":"c00"}\n{"iid":"c01"}\n', encoding="utf-8")
            manifest, summary = root / "manifest.jsonl", root / "summary.json"
            atomic_artifacts(manifest, summary, atomic_root, atomic_input, ["c00", "c01"], 2)
            progress = root / "progress.json"
            topup.main(
                [
                    "publish-progress", "--candidates", str(candidates),
                    "--expected-candidates", "4",
                    "--expected-candidates-sha256", sha(candidates.read_bytes()),
                    "--selection", str(root / "batch0" / "selection_receipt.json"),
                    "--planner-receipt", str(planner), "--atomic-manifest", str(manifest),
                    "--atomic-summary", str(summary), "--target-atomic-ok", "3",
                    "--output", str(progress),
                ]
            )
            self.assertEqual(json.loads(progress.read_text())["status"], "pool_exhausted")
            with self.assertRaises(topup.AtomicTopupError):
                topup.main(
                    [
                        "publish-gate", "--candidates", str(candidates),
                        "--expected-candidates", "4",
                        "--expected-candidates-sha256", sha(candidates.read_bytes()),
                        "--atomic-manifest", str(manifest), "--progress", str(progress),
                        "--target-ok", "3", "--output-manifest", str(root / "gate.jsonl"),
                        "--output-receipt", str(root / "gate.json"),
                    ]
                )

    def test_smoke_gate_passes_when_all_first_eight_candidates_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            write_candidates(candidates, 16)
            candidate_sha = sha(candidates.read_bytes())
            selection = self._select(
                candidates, root / "batch0", count=16, index=0, start=0, size=16,
                minimum_workers=2,
            )
            planner = root / "planner.json"
            planner_receipt(planner, selection, ["error"] * 8 + ["ok"] * 8)
            atomic_root = root / "atomic-root"
            atomic_root.mkdir()
            passed = [f"c{index:02d}" for index in range(8, 16)]
            atomic_input = root / "atomic.jsonl"
            atomic_input.write_bytes(
                b"".join(json.dumps({"iid": iid}).encode() + b"\n" for iid in passed)
            )
            manifest, summary = root / "cumulative.jsonl", root / "summary.json"
            atomic_artifacts(
                manifest, summary, atomic_root, atomic_input, passed, expected_rows=8
            )
            progress = root / "progress.json"
            topup.main(
                [
                    "publish-progress", "--candidates", str(candidates),
                    "--expected-candidates", "16",
                    "--expected-candidates-sha256", candidate_sha,
                    "--selection", str(root / "batch0" / "selection_receipt.json"),
                    "--planner-receipt", str(planner), "--atomic-manifest", str(manifest),
                    "--atomic-summary", str(summary), "--target-atomic-ok", "8",
                    "--output", str(progress),
                ]
            )
            gate_manifest, gate = root / "smoke8.jsonl", root / "smoke8.json"
            topup.main(
                [
                    "publish-gate", "--candidates", str(candidates),
                    "--expected-candidates", "16",
                    "--expected-candidates-sha256", candidate_sha,
                    "--atomic-manifest", str(manifest), "--progress", str(progress),
                    "--target-ok", "8", "--output-manifest", str(gate_manifest),
                    "--output-receipt", str(gate),
                ]
            )
            receipt = json.loads(gate.read_text())
            self.assertEqual(receipt["selected_iids"], passed)
            self.assertEqual(receipt["selected_parent_indices"], list(range(8, 16)))
            self.assertEqual(len(gate_manifest.read_bytes().splitlines()), 8)

    def test_out_of_parent_order_atomic_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            write_candidates(candidates, 4)
            selection = self._select(
                candidates, root / "batch0", count=4, index=0, start=0, size=4
            )
            planner = root / "planner.json"
            planner_receipt(planner, selection, ["ok", "ok", "ok", "error"])
            atomic_root = root / "atomic-root"
            atomic_root.mkdir()
            atomic_input = root / "atomic.jsonl"
            atomic_input.write_text("{}\n{}\n{}\n", encoding="utf-8")
            manifest, summary = root / "manifest.jsonl", root / "summary.json"
            atomic_artifacts(manifest, summary, atomic_root, atomic_input, ["c02", "c00"], 3)
            with self.assertRaises(topup.AtomicTopupError):
                topup.main(
                    [
                        "publish-progress", "--candidates", str(candidates),
                        "--expected-candidates", "4",
                        "--expected-candidates-sha256", sha(candidates.read_bytes()),
                        "--selection", str(root / "batch0" / "selection_receipt.json"),
                        "--planner-receipt", str(planner), "--atomic-manifest", str(manifest),
                        "--atomic-summary", str(summary), "--target-atomic-ok", "3",
                        "--output", str(root / "progress.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
