from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "methods"
    / "motive"
    / "scripts"
    / "auh_full_motion_select128_controller.sh"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


MOCK_SELECTOR = r'''from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def sha(raw): return hashlib.sha256(raw).hexdigest()

parser=argparse.ArgumentParser()
parser.add_argument("--generation-manifest",required=True)
parser.add_argument("--finalizer-done",required=True)
parser.add_argument("--generation-shard-manifest-dir",required=True)
parser.add_argument("--generation-shard-index-dir",required=True)
parser.add_argument("--postcheck-output",action="append",required=True)
parser.add_argument("--output-dir",required=True)
parser.add_argument("--exact-size",type=int,required=True)
parser.add_argument("--min-multi-unit",type=int,required=True)
parser.add_argument("--ffprobe",required=True)
parser.add_argument("--ffmpeg",required=True)
args=parser.parse_args()
if len(args.postcheck_output)!=32: raise SystemExit("not 32 postcheck outputs")
if args.exact_size!=128 or args.min_multi_unit!=32: raise SystemExit("policy differs")
log=Path(os.environ["MOCK_SELECTOR_ARGV_LOG"])
log.write_bytes(canonical(sys.argv[1:])+b"\n")
root=Path(args.output_dir)
root.mkdir()
(root/"samples").mkdir()
manifest_raw=b"".join(canonical({"iid":f"iid-{index:03d}"})+b"\n" for index in range(128))
(root/"dataset_manifest.jsonl").write_bytes(manifest_raw)
summary_raw=b"{}\n"
(root/"summary.json").write_bytes(summary_raw)
artifacts={
 "dataset_manifest.jsonl":{"sha256":sha(manifest_raw),"bytes":len(manifest_raw),"rows":128},
 "summary.json":{"sha256":sha(summary_raw),"bytes":len(summary_raw),"rows":1},
}
payload={
 "schema_version":"motive-goku-full-motion-dataset-done-v1",
 "status":"complete",
 "config":{"exact_size":128,"min_multi_unit":32,"selection_order":"mock","postcheck_requirement":"mock"},
 "counts":{"selected":128,"multi_unit":40,"single_unit":88,"by_dynamic_unit_count":{"1":88,"2":40,"3":0}},
 "selection_iids":[f"iid-{index:03d}" for index in range(128)],
 "artifacts":artifacts,
}
done=dict(payload); done["done_digest"]=sha(canonical(payload))
(root/"done.json").write_bytes(canonical(done)+b"\n")
'''


class FullMotionSelect128ControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.snapshot = self.root / "snapshot"
        package = self.snapshot / "methods" / "motive" / "motive"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "goku_full_motion_select128.py").write_text(
            MOCK_SELECTOR, encoding="utf-8"
        )
        (self.snapshot / "SOURCE_FILES.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )

        self.final = self.root / "final"
        self.final.mkdir()
        self.primary = self.final / "primary_256.jsonl"
        self.generation_done = self.final / "done.json"
        self.primary.write_text("{}\n", encoding="utf-8")
        self.generation_done.write_text("{}\n", encoding="utf-8")

        self.shard_root = self.root / "generation_shards"
        shard_leaf = self.shard_root / "shards"
        shard_leaf.mkdir(parents=True)
        (self.shard_root / "summary.json").write_text("{}\n", encoding="utf-8")
        (self.shard_root / "done.json").write_text("{}\n", encoding="utf-8")
        for index in range(32):
            (shard_leaf / f"shard_{index:03d}.jsonl").write_text(
                "{}\n" * 8, encoding="utf-8"
            )

        self.wan_output = self.root / "wan_output"
        wan_shards = self.wan_output / "wan_shards"
        wan_shards.mkdir(parents=True)
        for index in range(32):
            (wan_shards / f"shard_{index:03d}").mkdir()

        self.postcheck = self.root / "postcheck"
        self.postcheck.mkdir()
        self.status_rows: list[dict[str, str]] = []
        for index in range(32):
            shard = f"shard_{index:03d}"
            output = self.postcheck / f"postcheck_{shard}.jsonl"
            receipt = self.postcheck / f"postcheck_{shard}.receipt.json"
            output.write_text("{}\n", encoding="utf-8")
            receipt.write_text("{}\n", encoding="utf-8")
            self.status_rows.append(
                {
                    "shard": shard,
                    "wave": str(index // 8),
                    "slot": str(index % 8),
                    "node": f"auh{index % 4 + 1}",
                    "status": "complete",
                    "exit_code": "0",
                    "output": str(output),
                    "receipt": str(receipt),
                }
            )
        header = "shard\twave\tslot\tnode\tstatus\texit_code\toutput\treceipt\n"
        body = "".join(
            "\t".join(row[key] for key in row) + "\n"
            for row in self.status_rows
        )
        self.status_path = self.postcheck / "dispatcher_status.tsv"
        self.status_path.write_text(header + body, encoding="utf-8")

        self.dispatcher_receipt = self.postcheck / "dispatcher_receipt.json"
        self.exact128 = self.root / "exact128"
        self.controller_receipt = self.root / "exact128_controller_receipt.json"
        self.argv_log = self.root / "selector_argv.json"
        self.python = str(Path(sys.executable).resolve(strict=True))

    def _publish_dispatcher_receipt(
        self, *, complete: bool = True, version: int = 2
    ) -> None:
        rows = [dict(row) for row in self.status_rows]
        if not complete:
            rows[-1]["status"] = "postcheck_failed"
            rows[-1]["exit_code"] = "7"
            header = "shard\twave\tslot\tnode\tstatus\texit_code\toutput\treceipt\n"
            body = "".join(
                "\t".join(row[key] for key in row) + "\n" for row in rows
            )
            self.status_path.write_text(header + body, encoding="utf-8")
        status_raw = self.status_path.read_bytes()
        payload = {
            "schema_version": (
                "motive-goku-full-motion-postcheck-dispatch-receipt-"
                f"v{version}"
            ),
            "status": "complete" if complete else "partial_failure",
            "slurm_job_id": 123,
            "nodes": ["auh1", "auh2", "auh3", "auh4"],
            "source_snapshot": str(self.snapshot),
            "generation_shard_dir": str(self.shard_root / "shards"),
            "wan_shards_root": str(self.wan_output / "wan_shards"),
            "model": "/models/Qwen3-VL-32B-Instruct",
            "status_tsv": str(self.status_path),
            "status_tsv_sha256": _sha(status_raw),
            "completed_shards": 32 if complete else 31,
            "failed_shards": [] if complete else ["shard_031"],
            "shards": rows,
            "completed_at_utc": "2026-08-01T00:00:00+00:00",
        }
        if version == 2:
            executable = Path(self.python)
            record = {
                "path": str(executable),
                "sha256": _sha(executable.read_bytes()),
            }
            payload["media_tools"] = {
                "ffprobe": dict(record),
                "ffmpeg": dict(record),
            }
        elif version != 1:
            raise ValueError(f"unsupported dispatcher fixture version: {version}")
        receipt = dict(payload)
        receipt["receipt_digest"] = _sha(_canonical(payload))
        self.dispatcher_receipt.write_bytes(_canonical(receipt) + b"\n")

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "MOTIVE_FULL_MOTION_SOURCE_SNAPSHOT": str(self.snapshot),
                "MOTIVE_FULL_MOTION_SELECT_PYTHON": self.python,
                "MOTIVE_FULL_MOTION_GENERATION_PRIMARY": str(self.primary),
                "MOTIVE_FULL_MOTION_GENERATION_DONE": str(
                    self.generation_done
                ),
                "MOTIVE_FULL_MOTION_SHARD_MANIFEST_DIR": str(self.shard_root),
                "MOTIVE_FULL_MOTION_WAN_OUTPUT_ROOT": str(self.wan_output),
                "MOTIVE_FULL_MOTION_POSTCHECK_OUTPUT_ROOT": str(self.postcheck),
                "MOTIVE_FULL_MOTION_EXACT128_OUTPUT": str(self.exact128),
                "MOTIVE_FULL_MOTION_EXACT128_RECEIPT": str(
                    self.controller_receipt
                ),
                "MOTIVE_FULL_MOTION_EXACT128_WAIT_SECONDS": "5",
                "MOTIVE_FULL_MOTION_EXACT128_POLL_SECONDS": "1",
                "MOTIVE_FULL_MOTION_FFPROBE": self.python,
                "MOTIVE_FULL_MOTION_FFMPEG": self.python,
                "MOCK_SELECTOR_ARGV_LOG": str(self.argv_log),
            }
        )
        return environment

    def test_script_has_valid_bash_syntax_and_frozen_policy(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            "motive-goku-full-motion-postcheck-dispatch-receipt-v2",
            'f"postcheck_{shard}.jsonl"',
            'f"postcheck_{shard}.receipt.json"',
            "-m motive.goku_full_motion_select128",
            '--generation-shard-index-dir "${wan_shards_root}"',
            '--exact-size "${exact_size}"',
            '--min-multi-unit "${min_multi_unit}"',
            "motive-goku-full-motion-select128-controller-receipt-v1",
            "create-only exact128 output already exists",
            'cd "${code_root}"',
        ):
            self.assertIn(marker, text)
        self.assertNotIn(
            "motive-goku-full-motion-postcheck-dispatch-receipt-v1", text
        )
        self.assertNotIn("ssh ", text)
        self.assertNotIn("srun ", text)

    def test_waits_for_terminal_success_then_passes_actual_32_outputs(self) -> None:
        process = subprocess.Popen(
            ["bash", str(SCRIPT)],
            env=self._environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.2)
        self.assertIsNone(process.poll())
        self._publish_dispatcher_receipt(complete=True, version=2)
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("complete exact=128 min_multi=32", stdout)

        argv = json.loads(self.argv_log.read_text(encoding="utf-8"))
        self.assertEqual(argv.count("--postcheck-output"), 32)
        observed = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--postcheck-output"
        ]
        expected = [
            str(self.postcheck / f"postcheck_shard_{index:03d}.jsonl")
            for index in range(32)
        ]
        self.assertEqual(observed, expected)
        for option, value in (
            ("--generation-manifest", str(self.primary)),
            ("--finalizer-done", str(self.generation_done)),
            ("--generation-shard-manifest-dir", str(self.shard_root)),
            (
                "--generation-shard-index-dir",
                str(self.wan_output / "wan_shards"),
            ),
            ("--exact-size", "128"),
            ("--min-multi-unit", "32"),
        ):
            self.assertEqual(argv[argv.index(option) + 1], value)

        receipt = json.loads(
            self.controller_receipt.read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(
            receipt["config"], {"exact_size": 128, "min_multi_unit": 32}
        )
        self.assertEqual(receipt["output"]["counts"]["selected"], 128)
        self.assertGreaterEqual(receipt["output"]["counts"]["multi_unit"], 32)
        self.assertEqual(len(receipt["postcheck_dispatch"]["outputs"]), 32)
        payload = dict(receipt)
        digest = payload.pop("receipt_digest")
        self.assertEqual(digest, _sha(_canonical(payload)))

    def test_partial_dispatch_receipt_fails_without_selector_publication(self) -> None:
        self._publish_dispatcher_receipt(complete=False)
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=self._environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("terminal receipt/32-output closure is invalid", completed.stderr)
        self.assertFalse(self.exact128.exists())
        self.assertFalse(self.controller_receipt.exists())
        self.assertFalse(self.argv_log.exists())

    def test_legacy_v1_dispatch_receipt_is_rejected(self) -> None:
        self._publish_dispatcher_receipt(complete=True, version=1)
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=self._environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "terminal receipt/32-output closure is invalid", completed.stderr
        )
        self.assertFalse(self.exact128.exists())
        self.assertFalse(self.controller_receipt.exists())
        self.assertFalse(self.argv_log.exists())

    def test_existing_dataset_is_rejected_before_wait_or_overwrite(self) -> None:
        self.exact128.mkdir()
        completed = subprocess.run(
            ["bash", str(SCRIPT)],
            env=self._environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("create-only exact128 output already exists", completed.stderr)
        self.assertFalse(self.controller_receipt.exists())
        self.assertFalse(self.argv_log.exists())


if __name__ == "__main__":
    unittest.main()
